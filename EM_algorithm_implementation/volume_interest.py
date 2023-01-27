from math import tan
import torch
from torch import tensor
import numpy as np
from typing import Tuple


class VolumeOfInterest:


    def __init__(self,low_high_edges:Tuple[list[float]] = ([0.,0.,0.2],[1, 1, 0.8]),
                n_vox_xyz:Tuple[int] = (10,10,6),
                voxel_width:float = .10):

        '''
        Class representing a certain Volume Of Interest (VOI)

        INPUT:
                low_high_edges = ([x_low,y_low,z_low], [x_high,y_high,z_high]) in m   
                # voxels along x,y,z = [Nx,Ny,Nz] 
                voxel_width    = .10 m default   

        Features:
                voxels_edges   = tensor: [dx, dy, dz, 2, 3] lower and upper edges of every voxels
                voxels_centers = tensor: [dx, dy, dz, 3] position of each voxels center

            
        '''

        # Voxel width
        self.vox_width = voxel_width
        
        # Lowest and highest edges of volume in x, y and z
        self.xyz_min = low_high_edges[0]
        self.xyz_max = low_high_edges[1]
        
        # Number of voxels
        self.n_vox_xyz = torch.tensor(n_vox_xyz)  #self.compute_N_voxel()
        
        # Voxelization
        self.voxel_centers,self.voxel_edges = self.generate_voxels()

#         # True X0
#         self.x0_true = None
#         self.density_map_true = None

#         # Pred X0
#         self.x0_pred = None
#         self.density_map_pred = None

    
    def compute_voxel_centers(self, x_min_: float, 
                                    x_max_: float,
                                    Nvoxel_: int) -> torch.tensor:
                                    
        '''
        x_min,max border of the volume of interset for a given coordinate
                
        return voxels centers position along given coordinate
        '''
        xs_ = torch.linspace(x_min_,x_max_,Nvoxel_+1)
        xs_ += self.vox_width/2
        return xs_[:-1]

    def generate_voxels(self):
        
        '''
        returns voxels edges and voxels centers for the volumne of inte
        '''
        
        voxels_centers = torch.zeros((self.n_vox_xyz[0],self.n_vox_xyz[1],self.n_vox_xyz[2],3),dtype=torch.double)

        xs_ = self.compute_voxel_centers(x_min_=self.xyz_min[0], x_max_=self.xyz_max[0],
                                    Nvoxel_= self.n_vox_xyz[0])
        ys_ = self.compute_voxel_centers(x_min_=self.xyz_min[1], x_max_=self.xyz_max[1],
                                    Nvoxel_= self.n_vox_xyz[1])
        zs_ = self.compute_voxel_centers(x_min_=self.xyz_min[2], x_max_=self.xyz_max[2],
                                    Nvoxel_= self.n_vox_xyz[2]).flip(-1)

        for i in range(len(ys_)):
            for j in range(len(zs_)):
                voxels_centers[:,i,j,0]=torch.round(xs_, decimals=3)
        for i in range(len(xs_)):
            for j in range(len(zs_)):
                voxels_centers[i,:,j,1]=torch.round(ys_, decimals=3)
        for i in range(len(xs_)):
            for j in range(len(ys_)):
                voxels_centers[i,j,:,2]=torch.round(zs_, decimals=3)


        voxels_edges = torch.zeros((self.n_vox_xyz[0],self.n_vox_xyz[1],self.n_vox_xyz[2],2,3))

        voxels_edges[:,:,:,0,:] = torch.round(voxels_centers-self.vox_width/2, decimals=3)
        voxels_edges[:,:,:,1,:] = torch.round(voxels_centers+self.vox_width/2, decimals=3)

        return voxels_centers, voxels_edges
    
