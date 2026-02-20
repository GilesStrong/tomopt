r"""
Usage:
1. Instantiate MuonDataManager with ROOT file paths.
2. Call `load_all_data()` to load and cache datasets.
3. Generate train/val/test splits using `create_train_val_test_split()`.
4. Create PyTorch datasets via `create_datasets()` for model training using generated split indices.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np

import torch
from torch import Tensor
import pickle
import uproot

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import random
import logging
from tqdm import tqdm

__all__=["MuonTomographyDataset", "MuonDataManager"]


class MuonTomographyDataset(Dataset):
    """
    Dataset class used by MuonDataManager.
    """
    
    def __init__(self, 
                 data_indices: List[Tuple[int, int]], 
                 cached_data: Dict[Tuple[int, int], np.ndarray],
                 labels: Optional[Dict[Tuple[int, int], int]] = None,
                 transform=None):
        """
        Args:
            data_indices: List of (file_idx, scan_idx) tuples
            cached_data: Dictionary mapping (file_idx, scan_idx) to data arrays
            labels: Optional dictionary mapping indices to labels
            transform: Optional transform to apply to data
        """
        self.data_indices = data_indices
        self.cached_data = cached_data
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.data_indices)
        
    def __getitem__(self, idx):
        file_idx, scan_idx = self.data_indices[idx]
        data = self.cached_data[(file_idx, scan_idx)]
        
        # Handle tuple data (features, gt, preds)
        if isinstance(data, tuple) and len(data) == 3:
            features, gt, preds = data
        elif isinstance(data, tuple) and len(data) == 2:
            features, gt = data
        else:
            # Fallback for old format
            features = data
            gt = None
            preds = None
        
        # Handle jagged/irregular data structures for features
        try:
            # First try to convert directly if it's already a regular array
            if isinstance(features, np.ndarray) and features.dtype != object:
                features_array = np.array(features, copy=True, dtype=np.float32)
            else:
                # Handle jagged/irregular data
                if isinstance(features, (list, tuple)) or (isinstance(features, np.ndarray) and features.dtype == object):
                    # For jagged data, keep as-is and let collate function handle padding
                    features_array = features
                else:
                    # Try to convert to array, but catch shape errors
                    features_array = np.asarray(features, dtype=np.float32)
                    
        except ValueError as e:
            if "inhomogeneous shape" in str(e) or "setting an array element with a sequence" in str(e):
                # This is jagged data - keep as original structure
                features_array = features
            else:
                raise e
        
        # Convert features to tensor only if we have a proper array
        if isinstance(features_array, np.ndarray):
            features_tensor = torch.from_numpy(features_array).float().clone()
        else:
            # For jagged data, convert each sub-array to tensor
            if isinstance(features_array, (list, tuple)):
                features_tensor = [torch.from_numpy(np.asarray(x, dtype=np.float32)).float() 
                                 for x in features_array if x is not None]
            else:
                features_tensor = torch.from_numpy(np.asarray(features_array, dtype=np.float32)).float()
        
        # Handle ground truth
        gt_tensor = None
        if gt is not None:
            try:
                if isinstance(gt, np.ndarray):
                    gt_tensor = torch.from_numpy(gt.astype(np.float32)).float().clone()
                else:
                    gt_tensor = torch.from_numpy(np.asarray(gt, dtype=np.float32)).float()
            except:
                gt_tensor = gt  # Keep as-is if conversion fails

                gt_tensor = None
                
        # Handle TomOpt preds
        if preds is not None:
            try:
                if isinstance(preds, np.ndarray):
                    preds = torch.from_numpy(preds.astype(np.float32)).float().clone()
                else:
                    preds = torch.from_numpy(np.asarray(preds, dtype=np.float32)).float()
            except:
                preds_tensor = preds  # Keep as-is if conversion fails
        
        if self.transform:
            features_tensor = self.transform(features_tensor)
            
        # Return (features, gt) tuple along with labels if present
        if preds is not None and gt_tensor is not None: data_tuple = (features_tensor, gt_tensor, preds_tensor)
        else: data_tuple = (features_tensor, gt_tensor) if gt_tensor is not None else features_tensor
        
        if self.labels is not None:
            label = self.labels[(file_idx, scan_idx)]
            return data_tuple, torch.tensor(label, dtype=torch.long)
        
        return data_tuple

class MuonDataManager:
    """
    Manager class for efficiently loading and splitting muon tomography data from ROOT files.
    Pre-loads and caches datasets.
    """
    
    def __init__(self, 
                 root_files: List[str], 
                 tree_name: str = "tree",
                 branch_names: List[str] = None,
                 cache_dir: Optional[str] = None):
        """
        Args:
            root_files: List of paths to ROOT files
            tree_name: Name of the tree in ROOT files
            branch_names: List of branch names to load (if None, loads all)
            cache_dir: Directory to cache preprocessed data
        """
        self.root_files = root_files
        self.tree_name = tree_name
        self.branch_names = branch_names
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cached_data = {}
        self.metadata = {}
        
        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True)
            
    def _get_cache_path(self, file_idx: int) -> Path:
        """Get cache file path for a given file index."""
        return self.cache_dir / f"cached_file_{file_idx}.pkl"
        
    def _load_single_file(self, file_idx: int, file_path: str) -> Dict:
        """Load and preprocess a single ROOT file."""
        cache_path = self._get_cache_path(file_idx) if self.cache_dir else None
        
        # Check if cached version exists
        if cache_path and cache_path.exists():
            logging.info(f"Loading cached data for file {file_idx}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        logging.info(f"Processing ROOT file {file_idx}: {file_path}")
        file_data = {}
        
        with uproot.open(file_path) as f:
            tree = f[self.tree_name]
            
            # Get branch names if not specified
            if self.branch_names is None:
                branches = tree.keys()
            else:
                branches = self.branch_names
                
            # Load all branches for this file
            arrays = tree.arrays(branches, library="np")
            
            # Process each scan (assuming 50 scans per file)
            n_scans = 50
            for scan_idx in range(n_scans):
                scan_data = {}
                for branch in branches:
                    # Extract data for this specific scan
                    # Adjust this based on your actual data structure
                    if hasattr(arrays[branch], '__len__') and len(arrays[branch]) > scan_idx:
                        scan_data[branch] = arrays[branch][scan_idx]
                    
                # Combine branches into single array (modify as needed)
                combined_data = self._combine_branches(scan_data)
                file_data[scan_idx] = combined_data
                
        # Cache the processed data
        if cache_path:
            with open(cache_path, 'wb') as f:
                pickle.dump(file_data, f)
                
        return file_data
        
    def _combine_branches(self, scan_data: Dict) -> tuple:
        """
        Combine multiple branches into a single array and extract ground truth and preds obtained from TomOpt.
        Returns both features and ground truth.
        """
        arrays = []
        gt = None
        preds = None
        
        for branch_name, data in scan_data.items():
            if isinstance(data, np.ndarray) and branch_name != 'gt' and branch_name != 'preds':
                # Ensure consistent dtype and shape
                data = np.asarray(data, dtype=np.float32)
                arrays.append(data)
            elif branch_name == 'gt':
                gt = np.asarray(data, dtype=np.float32) if data is not None else None
            elif branch_name == 'preds':
                preds = np.asarray(data, dtype=np.float32) if data is not None else None
            
        if arrays:
            # For jagged arrays, you might want to handle them differently
            # This assumes all branches have compatible shapes
            try:
                result = np.stack(arrays, axis=-1)
            except ValueError:
                # If stacking fails due to shape mismatch, concatenate instead
                result = np.concatenate([arr.flatten() for arr in arrays])
            return result.astype(np.float32), gt, preds
        else:
            return np.array([], dtype=np.float32), gt, preds  # Handle empty case
    
    def load_all_data(self) -> None:
        """Load and cache all ROOT files."""
        logging.info("Loading all ROOT files...")
        
        for file_idx, file_path in enumerate(tqdm(self.root_files, desc="Loading files")):
            file_data = self._load_single_file(file_idx, file_path)
            
            # Store in main cache
            for scan_idx, data in file_data.items():
                self.cached_data[(file_idx, scan_idx)] = data
                
        logging.info(f"Loaded {len(self.cached_data)} scans from {len(self.root_files)} files")
    
    def create_train_val_test_split(self, 
                                   n_positions: int = 49,
                                   n_materials: int = 5, 
                                   n_repetitions: int = 20,
                                   val_combos: int = 200,
                                   test_combos: int = 200,
                                   random_seed: int = 32) -> Tuple[List, List, List]: # seed was 42
        """
        Create train/validation/test splits based on (material, position) combinations.
        
        Args:
            n_positions: Number of positions (50)
            n_materials: Number of materials (5)
            n_repetitions: Number of repetitions per combo (10)
            val_combos: Number of (material, position) combos for validation
            test_combos: Number of (material, position) combos for test
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Create all possible (material, position) combinations
        all_combos = [(material, position) 
                      for material in range(n_materials) 
                      for position in range(n_positions)]
        
        # Randomly shuffle and split
        random.shuffle(all_combos)
        
        test_combos_list = all_combos[:test_combos]
        val_combos_list = all_combos[test_combos:test_combos + val_combos]
        train_combos_list = all_combos[test_combos + val_combos:]
        
        def get_indices_for_combos(combo_list):
            indices = []
            for material, position in combo_list:
                # Calculate file and scan indices based on your storage pattern
                # 50 positions × 5 materials × 10 repetitions = 2500 total scans
                # Stored as: 50 files × 50 scans per file
                
                for rep in range(n_repetitions):
                    # Calculate the global scan index
                    # global_scan_idx = (position * n_materials * n_repetitions + 
                    #                  material * n_repetitions + rep)
                    global_scan_idx = (rep * n_positions * n_materials +
                       material * n_positions +
                       position)
                        
                    # Convert to (file_idx, scan_idx)
                    file_idx = global_scan_idx // 50  # 50 scans per file
                    scan_idx = global_scan_idx % 50
                    
                    indices.append((file_idx, scan_idx))
                    
            return indices
        
        train_indices = get_indices_for_combos(train_combos_list)
        val_indices = get_indices_for_combos(val_combos_list)
        test_indices = get_indices_for_combos(test_combos_list)
        
        logging.info(f"Split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
        
        return train_indices, val_indices, test_indices
    
    def create_datasets(self, 
                       train_indices: List,
                       val_indices: List, 
                       test_indices: List,
                       labels: Optional[Dict] = None,
                       train_transform=None,
                       val_transform=None) -> Tuple[Dataset, Dataset, Dataset]:
        """Create PyTorch datasets for train/val/test."""
        
        train_dataset = MuonTomographyDataset(
            train_indices, self.cached_data, labels, train_transform
        )
        
        val_dataset = MuonTomographyDataset(
            val_indices, self.cached_data, labels, val_transform
        )
        
        test_dataset = MuonTomographyDataset(
            test_indices, self.cached_data, labels, val_transform
        )
        
        return train_dataset, val_dataset, test_dataset
