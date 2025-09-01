from typing import Dict, Tuple
import numpy as np

from torch.utils.data import Dataset
import torch
from torch import Tensor
import pickle

class DatasetFromTomOptRecords(Dataset):
    def __init__(self, records_dir:str):
        super().__init__()
        self.records = self._get_records_from_pickle(records_dir)
        self.x0_maps = self._get_ground_truth(self.records)
        self.poca_xyz = self.records["poca_rec"].poca_xyz_batch
        self.poca_xyz_unc = self.records["poca_rec"].poca_xyz_unc_batch
        self.theta = self.records["poca_rec"].poca_theta_mcs_batch
        self.theta_unc = self.records["poca_rec"].poca_theta_mcs_unc_batch
        self.hits_above = [self.records["hr"].split_above_below(rec_batch)[0] for rec_batch in self.records["hr"].reco_hits_batch]
        self.hits_below = [self.records["hr"].split_above_below(rec_batch)[1] for rec_batch in self.records["hr"].reco_hits_batch]
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        assert len(self.poca_xyz) == len(self.x0_maps), "Mismatch in data length"

        # Precompute normalization stats
        all_feats = []
        for i in range(len(self)):
            xyz = self.poca_xyz[i]         # (N, 3)
            xyz_unc = self.poca_xyz_unc[i] # (N, 3)
            theta = self.theta[i]          # (N, 1)
            theta_unc = self.theta_unc[i]  # (N, 1)
            hits_above = self.hits_above[i] # (N, 3)
            hits_below = self.hits_below[i] # (N, 3)
            N = xyz.shape[0]
            pad = torch.zeros(N, 2, dtype=xyz.dtype, device=xyz.device)
            
            # per‐panel mean position
            ha_mean = hits_above.mean(dim=1)    # (N, 3)
            hb_mean = hits_below.mean(dim=1)

            # per‐panel spread (std)
            ha_std  = hits_above.std(dim=1)     # (N, 3)
            hb_std  = hits_below.std(dim=1)

            feats = torch.cat([xyz, theta, xyz_unc, theta_unc, ha_mean, ha_std, hb_mean, hb_std, pad], dim=1)  # (N, 22
            all_feats.append(feats)

        stacked = torch.cat(all_feats, dim=0)  # (total_N, 10)
        self.mean = stacked.mean(dim=0)        # (10,)
        self.std = stacked.std(dim=0) + 1e-8   # (10,)

    def __len__(self):
        return len(self.x0_maps)

    def __getitem__(self, idx):
        xyz = self.poca_xyz[idx]         # (N, 3)
        xyz_unc = self.poca_xyz_unc[idx] # (N, 3)
        theta = self.theta[idx]          # (N, 1)
        theta_unc = self.theta_unc[idx]  # (N, 1)
        hits_above = self.hits_above[idx] # (N, 3)
        hits_below = self.hits_below[idx] # (N, 3)
        N = xyz.shape[0]
        pad = torch.zeros(N, 2, dtype=xyz.dtype, device=xyz.device)
        
        # per‐panel mean position
        ha_mean = hits_above.mean(dim=1)    # (N, 3)
        hb_mean = hits_below.mean(dim=1)

        # per‐panel spread (std)
        ha_std  = hits_above.std(dim=1)     # (N, 3)
        hb_std  = hits_below.std(dim=1)

        # Stack features
        features = torch.cat([xyz, theta, xyz_unc, theta_unc, ha_mean, ha_std, hb_mean, hb_std, pad], dim=1)  # (N, 22)

        # # Normalize
        # features = (features.to(xyz.device) - self.mean.to(xyz.device)) / self.std.to(xyz.device)

        x0 = self.x0_maps[idx]  # (n_z, n_x, n_y)
        
        return features, x0

    def _get_records_from_pickle(self, record_dir: str) -> Dict:
        """
        Returns the dictionary of PoCA and hit data stored in the records pickle file.
        """
        with open(record_dir, "rb") as f:
            records_dict = pickle.load(f)
        return records_dict

    def _get_ground_truth(self, records: Dict) -> Tensor:
        """
        Returns the ground truth tensor of voxelized X0 of volumes.
        """
        gt_list   = [torch.tensor(p[1]) for p in records['preds']] # list of ground truth tensors
        gt_tensor   = torch.stack(gt_list)    # (B, Nx, Ny, Nz)
        return gt_tensor

    def dispersed_split(self, num_positions=49, num_repeats=20, seed=42):
        rng = np.random.default_rng(seed)
        
        # All positions
        all_positions = np.arange(num_positions)
        
        # Shuffle positions for dispersion
        rng.shuffle(all_positions)
        
        # Number for training
        num_train_positions = num_positions - 20  # 0 positions for val+test
        train_positions = sorted(all_positions[:num_train_positions])
        
        # Remaining positions for val/test
        holdout_positions = sorted(all_positions[num_train_positions:])
        
        # Split holdouts evenly into val/test
        rng.shuffle(holdout_positions)
        mid = len(holdout_positions) // 2
        val_positions = holdout_positions[:mid]
        test_positions = holdout_positions[mid:]
        
        # Map to actual indices over repeats
        train_idx, val_idx, test_idx = [], [], []
        for rep in range(num_repeats):
            base = rep * num_positions
            train_idx.extend(base + np.array(train_positions))
            val_idx.extend(base + np.array(val_positions))
            test_idx.extend(base + np.array(test_positions))
        
        return np.array(train_idx), np.array(val_idx), np.array(test_idx), train_positions, val_positions, test_positions

    def get_grid_shape(self) -> Tuple:
        """
        Returns the grid shape of the volume (n_x, n_y, n_z), which is the 
        shape of the ground truth tensor after permuting from zxy to xyz.
        """
        grid_shape = self.x0_maps[0].permute(1, 2, 0).shape # (ZXY) -> (XYZ)
        return grid_shape

    def get_voxel_centers(self, voxel_size: float = 0.1, origin: Tuple[float, float, float]=(0.0, 0.0, 0.3)) -> Tensor:
        """
        Returns: (N_voxels, 3) tensor of voxel center coordinates
        """
        nx, ny, nz = self.get_grid_shape()
        ox, oy, oz = origin
    
        # Create grid indices
        x = torch.arange(nx)
        y = torch.arange(ny)
        z = torch.arange(nz)
    
        # Convert to physical coordinates (center of each voxel)
        xv, yv, zv = torch.meshgrid(x, y, z, indexing='ij')
        centers = torch.stack([xv, yv, zv], dim=-1).float()  # (nx, ny, nz, 3)
        centers = centers * voxel_size + voxel_size / 2.0
    
        # Shift by origin if provided
        centers[..., 0] += ox
        centers[..., 1] += oy
        centers[..., 2] += oz
    
        return centers.view(-1, 3)  # (N_voxels, 3)
        
        
        
        