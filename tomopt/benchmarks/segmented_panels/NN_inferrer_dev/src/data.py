
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
    Can operate in entirely cached (in_memory=True) or lazy mode (in_memory=False).
    """
    def __init__(self, 
                 data_indices: List[str], 
                 cached_data: Dict[str, np.ndarray],
                 hdf5_path: Optional[str] = None,
                 in_memory: bool = True,
                 labels: Optional[Dict[str, int]] = None,
                 transform=None):
        self.data_indices = data_indices
        self.cached_data = cached_data
        self.hdf5_path = hdf5_path
        self.in_memory = in_memory
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.data_indices)
        
    def __getitem__(self, idx):
        scan_name = self.data_indices[idx]
        
        if self.in_memory:
            data = self.cached_data[scan_name]
        else:
            import h5py
            with h5py.File(self.hdf5_path, 'r') as h5f:
                grp = h5f['data'][scan_name]
                features = grp['features'][:]
                gt = grp['gt'][:] if 'gt' in grp else None
                preds = grp['preds'][:] if 'preds' in grp else None
                
                if preds is not None and gt is not None: data = (features, gt, preds)
                elif gt is not None: data = (features, gt)
                else: data = features
        
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
                    preds_tensor = torch.from_numpy(preds.astype(np.float32)).float().clone()
                else:
                    preds_tensor = torch.from_numpy(np.asarray(preds, dtype=np.float32)).float()
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
    Manager class for efficiently loading and splitting muon tomography data from HDF5 files.
    """
    
    def __init__(self, hdf5_path: str):
        """
        Args:
            hdf5_path: Path to the generated HDF5 dataset file.
        """
        self.hdf5_path = hdf5_path
        self.cached_data = {}
        
    def load_all_data(self) -> None:
        """
        Loads the data from the HDF5 file into the cache. 
        For very large files, consider memory-mapping (lazy loading) instead of full cache.
        """
        logging.info(f"Loading HDF5 dataset from {self.hdf5_path}...")
        
        import h5py
        with h5py.File(self.hdf5_path, 'r') as h5f:
            if 'data' not in h5f:
                logging.error("No 'data' group found in HDF5 file.")
                return
                
            data_group = h5f['data']
            
            for scan_name, scan_grp in tqdm(data_group.items(), desc="Caching scans"):
                # Load features and labels into cache
                features = scan_grp["features"][:]
                gt = scan_grp["gt"][:] if "gt" in scan_grp else None
                preds = scan_grp["preds"][:] if "preds" in scan_grp else None
                
                # Format exactly as MuonTomographyDataset expects it
                # The old data.py expected (features, gt, preds) mapped by a key.
                # We will use the scan_name (e.g., 'scan_0') as the key.
                if preds is not None and gt is not None:
                    data_tuple = (features, gt, preds)
                elif gt is not None:
                    data_tuple = (features, gt)
                else:
                    data_tuple = features
                    
                self.cached_data[scan_name] = data_tuple
                
        logging.info(f"Loaded {len(self.cached_data)} scans from HDF5.")
    
    def create_train_val_test_split(self, 
                                   n_positions: int = 49,
                                   n_materials: int = 5, 
                                   n_repetitions: int = 20,
                                   val_combos: int = 200,
                                   test_combos: int = 200,
                                   random_seed: int = 32) -> Tuple[List, List, List]: 
        """
        Create train/validation/test splits based on (material, position) combinations by reading HDF5 attributes.
        """
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Create all possible (material, position) combinations
        all_combos = [(material, position) 
                      for material in range(n_materials) 
                      for position in range(n_positions)]
        
        # Randomly shuffle and split
        random.shuffle(all_combos)
        
        test_combos_list = set(all_combos[:test_combos])
        val_combos_list = set(all_combos[test_combos:test_combos + val_combos])
        train_combos_list = set(all_combos[test_combos + val_combos:])
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        import h5py
        with h5py.File(self.hdf5_path, 'r') as h5f:
            if 'data' not in h5f:
                return [], [], []
                
            data_group = h5f['data']
            
            for scan_name, scan_grp in data_group.items():
                # Read attributes directly
                mat = scan_grp.attrs.get("material_id")
                pos = scan_grp.attrs.get("position")
                
                if mat is None or pos is None:
                    continue
                    
                combo = (int(mat), int(pos))
                
                if combo in test_combos_list:
                    test_indices.append(scan_name)
                elif combo in val_combos_list:
                    val_indices.append(scan_name)
                elif combo in train_combos_list:
                    train_indices.append(scan_name)
        
        logging.info(f"Split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
        
        return train_indices, val_indices, test_indices
    
    def create_datasets(self, 
                       train_indices: List[str],
                       val_indices: List[str], 
                       test_indices: List[str],
                       labels: Optional[Dict] = None,
                       in_memory: bool = True,
                       train_transform=None,
                       val_transform=None) -> Tuple[Dataset, Dataset, Dataset]:
        """Create PyTorch datasets for train/val/test."""
        
        train_dataset = MuonTomographyDataset(
            train_indices, self.cached_data, self.hdf5_path, in_memory, labels, train_transform
        )
        
        val_dataset = MuonTomographyDataset(
            val_indices, self.cached_data, self.hdf5_path, in_memory, labels, val_transform
        )
        
        test_dataset = MuonTomographyDataset(
            test_indices, self.cached_data, self.hdf5_path, in_memory, labels, val_transform
        )
        
        return train_dataset, val_dataset, test_dataset

