import torch
import torch.nn as nn
from torch import Tensor
import sys
sys.path.append('/home/ucl/cp3/zdaher/tomopt_segmentedPanels_NNInference_branch/tomopt/tomopt/benchmarks/segmented_panels/NN_inferrer_dev/')

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
    

    def forward(self, poca_tensor, point_mask=None):
        B, N, _ = poca_tensor.shape
        V = self.n_voxels
        device = poca_tensor.device
    
        if point_mask is None:
            point_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    
        # Flatten points across batch
        poca_flat = poca_tensor.reshape(B * N, -1)
        mask_flat = point_mask.reshape(B * N)
        poca_flat = poca_flat[mask_flat]             # (total_valid_points, feat_dim)
        xyz_flat = poca_flat[:, :3]                  # (total_valid_points, 3)
    
        # batch index per muon
        batch_ids = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
        batch_ids = batch_ids.reshape(-1)[mask_flat] # (total_valid_points,)
    
        # point MLP
        muon_feats = self.point_mlp(poca_flat)      # (total_valid_points, C)
        raw_scores = self.attn_score(muon_feats).squeeze(-1)  # (total_valid_points,)
    
        outputs = []
    
        for b in range(B):
            # select points belonging to this batch
            idx = (batch_ids == b)
            xyz_b = xyz_flat[idx]         # (N_b,3)
            feats_b = muon_feats[idx]     # (N_b,C)
            scores_b = raw_scores[idx]    # (N_b,)
    
            # distances to voxels
            diff = self.voxel_centers[:, None, :] - xyz_b[None, :, :]  # (V,N_b,3)
            dists = diff.norm(dim=2)                                    # (V,N_b)
            mask = dists < self.radius
    
            # attention weights
            raw_scores_V = scores_b.unsqueeze(0).expand(V, -1)
            raw_scores_V = raw_scores_V.masked_fill(~mask, -1e9)
            alpha = torch.softmax(raw_scores_V, dim=1)                  # (V,N_b)
    
            # weighted sum
            agg_feat = alpha @ feats_b                                   # (V,C)
    
            voxel_input = torch.cat([agg_feat, self.voxel_centers], dim=1)  # (V,C+3)
            voxel_out = self.voxel_mlp(voxel_input)                         # (V,1)
            voxel_out = voxel_out.view(*self.voxel_shape)                   # (X,Y,Z)
            outputs.append(voxel_out)
    
        return torch.stack(outputs, dim=0)  # (B,X,Y,Z)



    
