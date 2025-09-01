import torch
import torch.nn as nn
from torch import Tensor

import sys
sys.path.append('/home/ucl/cp3/zdaher/TomOpt_SegmentedPanels/NN_inferrer_pointnet_dev/')

from src.model import AbsVoxelNNInferrer

class LocalAggVoxelX0InferNet(AbsVoxelNNInferrer):
    """
    Neural network for local feature aggregation of 3D muon PoCA points
    into voxel-wise predictions, using attention-based pooling.

    Each voxel aggregates features from nearby PoCA points within a radius,
    using a learned attention score. The final output is a scalar prediction
    per voxel, reshaped into a 3D grid.

    Args:
        voxel_centers (Tensor): (V, 3) coordinates of voxel centers.
        voxel_shape (tuple): (X Y Z) shape of the output 3D grid.
        radius (float): Radius (in meters) around each voxel cemter to aggregate points from.
    """
    
    def __init__(self, voxel_centers:Tensor, voxel_shape:tuple, radius:float=0.1, device:str="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.register_buffer('voxel_centers', voxel_centers)
        self.voxel_shape = voxel_shape
        self.n_voxels = voxel_centers.shape[0]
        self.radius = radius
        self.device = device
        self.to(self.device)

        # Processes each POCA point's features
        self.point_mlp = nn.Sequential(
            nn.Linear(8, 16),
            #nn.GroupNorm(4, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(p=0.05),  # Reduced dropout
            nn.Linear(16, 16),   # Smaller hidden size
            # nn.GroupNorm(4, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
        )

        # Processes aggregated voxel features 
        self.voxel_mlp = nn.Sequential(
            nn.Linear(16 + 3, 256),
            #nn.GroupNorm(4, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 128),
            #nn.GroupNorm(4, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            #nn.Dropout(p=0.1),
            nn.Linear(128, 1),
            #nn.Softplus()  # predict log(X0)
        )

        self.refine_conv = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 1, kernel_size=3, padding=1),
            nn.Softplus()
        )
        
        self.refine_unet = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1), nn.ReLU(),
            nn.MaxPool3d(2), 
            nn.Conv3d(8,16,3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(16,8,3,padding=1), nn.ReLU(),
            nn.Conv3d(8,1,3,padding=1), nn.Softplus()
        )
        # Attention mechanism to score each point
        self.attn_score = nn.Linear(16, 1)
        
        self._initialize_weights() 
        
    def _initialize_weights(self) -> None: 
        """
        Xavier initialization for linear layers.
        """
        for m in self.modules(): 
            if isinstance(m, nn.Linear): 
                torch.nn.init.xavier_uniform_(m.weight) 
                if m.bias is not None: torch.nn.init.constant_(m.bias, 0) 
    
    def forward(self, poca_tensor, point_mask=None) -> Tensor:
        """
        Args:
            poca_tensor (Tensor): (B, N, 22) tensor of input POCA features.
            point_mask (BoolTensor, optional): (B, N) mask for unpadded points.

        Returns:
            Tensor: (B, X, Y, Z) 3D output per batch, one scalar per voxel.
        """
        B, N, _ = poca_tensor.shape
        V = self.n_voxels
        
        all_outputs = []

        for b in range(B):
            if point_mask is not None: 
                # Only process real points, not padded ones 
                real_points = point_mask[b]
                poca_tensor_masked = poca_tensor[b][real_points]
                poca_xyz = poca_tensor[b, real_points, 0:3]  # (N, 3)
                # poca_xyz_unc = poca_tensor[b, real_points, 4:7]  # (N, 3)       
                poca_msc = poca_tensor[b, real_points, 3]  # (N, 1)
                # poca_msc_unc = poca_tensor[b, real_points, 7]  # (N, 1)
                # poca_hits = poca_tensor[b, real_points, 8:]  # (N, 12) xyz of mean/std of hits above/below
                # poca_useful = torch.cat([poca_xyz, poca_xyz_unc, poca_msc, poca_msc_unc], dim =1)
                # muon_feats = self.point_mlp(poca_useful) 
                muon_feats = self.point_mlp(poca_tensor_masked)  # (N, C)
            else: 
                poca_xyz = poca_tensor[b, :, 0:3]             
                #poca_xyz_unc = poca_tensor[b, :, 4:7]        
                # poca_msc = poca_tensor[b, :, 3:4]
                # poca_msc_unc = poca_tensor[b, :, 7:8]
                # poca_hits = poca_tensor[b, :, 8:]
                # poca_useful = torch.cat([poca_xyz, poca_xyz_unc, poca_msc, poca_msc_unc], dim =1)
                # muon_feats = self.point_mlp(poca_useful) 
                muon_feats = self.point_mlp(poca_tensor[b]) 

            # Compute distance from each voxel to each point
            diff    = self.voxel_centers[:, None, :] - poca_xyz[None, :, :]  # (V,N,3)
            dists   = diff.norm(dim=2)  # (V,N)

            # radius‐mask
            mask    = dists < self.radius  # (V,N)

            # Compute raw attention scores per point: shape (N,)
            raw_scores = self.attn_score(muon_feats).squeeze(-1)  # (N,)

            # Expand to (V, N)
            raw_scores_V = raw_scores.unsqueeze(0).expand(V, -1)  # (V, N)

            # Mask out-of-radius points with large negative value for softmax
            neg_inf = -1e9
            raw_scores_V = raw_scores_V.masked_fill(~mask, neg_inf)  # (V, N)

            # Normalize scores into attention weights (softmax over points)
            alpha = torch.softmax(raw_scores_V, dim=1)               # (V, N)
            
            # ALternative: Gaussian weighting instead of learned attention
            # weights = torch.exp(-0.5 * (dists / self.radius) ** 2)*mask
            # alpha = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

            # Weighted sum of features: expand muon_feats to (V, N, C)
            muon_feats_V = muon_feats.unsqueeze(0).expand(V, -1, -1) # (V, N, C)
            agg_feat = (alpha.unsqueeze(-1) * muon_feats_V).sum(dim=1)  # (V, C)
            
            # max pooling
            #agg_feat, _ = muon_feats_V.max(dim=1)
            #agg_feat, _ = (alpha.unsqueeze(-1) * muon_feats_V).max(dim=1)  # (V, C)

            # Concatenate voxel coordinates
            voxel_input = torch.cat([agg_feat, self.voxel_centers], dim=1)  # (V, C+3)

            # Pass through voxel MLP to get output scalar per voxel
            voxel_out = self.voxel_mlp(voxel_input) # (V, 1)

            # Reshape to voxel grid (X, Y, Z)
            voxel_3d = voxel_out.view(*self.voxel_shape) # (X, Y, Z)

            all_outputs.append(voxel_3d)
            
        return torch.stack(all_outputs, dim=0) # (B, X, Y, Z)