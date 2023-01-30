import math
import torch
from torch import tensor
import numpy as np
import pandas 
from volume_interest import VolumeOfInterest

class Tracking:

    def __init__(self, voi: VolumeOfInterest, data:pandas.DataFrame):
        
        self.voi=voi
        self.voxel_edges = voi.voxel_edges
        self.Nvox_Z=voi.n_vox_xyz[2]

        self.data=data
        self.events=len(self.data)

        self.tracks = self.compute_discrete_tracks()
        self.triggered_voxels = self.find_triggered_voxels()

    def compute_discrete_tracks(self):
        
        '''
        Compute x,y,z position at Zmax and Zmin of each voxel layer (for incoming and outgoing tracks)
        
        INPUT: - Zmin Zmax of each voxel layer
               - theta_xy_in_x,y, theta_xy_out_x,y: zentih angle in x,y for incoming and outgoing track
               - xyz_in_x,y,z, xyz_out_x,y,z: muon position entering and exiting the Volume of Interest
               
        OUTPUT: - track_in_discrete, track_out_discrete: x,y,z position at Zmax and Zmin of each voxel layer (for incoming and outgoing tracks), size = [coordinate,Nlayer_along_Z + 1, Nevents] ([3,7,9999])
        '''

        #Nvox_Z = N_voxels[2]
        voxels_edges=self.voxel_edges
        #entry points
        xyz_in_x = self.data['xyz_in_x']
        xyz_in_y = self.data['xyz_in_y']
        xyz_in_z = self.data['xyz_in_z']
        
        #exit points 
        xyz_out_x = self.data['xyz_out_x']
        xyz_out_y = self.data['xyz_out_y']
        xyz_out_z = self.data['xyz_out_z']
        
        
        #directions 
        dx= xyz_out_x - xyz_in_x
        dy= xyz_out_y - xyz_in_y
        dz= xyz_out_z - xyz_in_z
        
        
        #angles
        theta_x=torch.arctan(tensor(dx/dz))
        theta_y=torch.arctan(tensor(dy/dz))
        
        Z_discrete = torch.linspace(torch.min(voxels_edges[:,:,:,:,2]).item(),
                                    torch.max(voxels_edges[:,:,:,:,2]).item(),
                                    self.Nvox_Z+1) 

        Z_discrete.unsqueeze_(1)
        
        Z_discrete = torch.round(Z_discrete.expand(len(Z_discrete),len(self.data['mom'])),decimals=3)
        
        x_track = tensor(xyz_in_x) + (xyz_out_z[0]-Z_discrete)*torch.tan(theta_x)

        y_track = tensor(xyz_in_y) + (xyz_out_z[0]-Z_discrete)*torch.tan(theta_y)

        return torch.stack([x_track,y_track,Z_discrete.flip(dims=(0,))])


    def find_triggered_voxels(self)->None:
        
        def compute_xy_min_max(X_discrete,Y_discrete):
        
            mask_x = X_discrete[0,:]>X_discrete[1,:]
            mask_y = Y_discrete[0,:]>Y_discrete[1,:]

            X_max = torch.where(mask_x,X_discrete[:-1,:],X_discrete[1:,:])
            X_min = torch.where(mask_x,X_discrete[1:,:],X_discrete[:-1,:])

            Y_max = torch.where(mask_y,Y_discrete[:-1,:],Y_discrete[1:,:])
            Y_min = torch.where(mask_y,Y_discrete[1:,:],Y_discrete[:-1,:])

            return X_max,X_min,Y_max,Y_min
            
        X_max,X_min,Y_max,Y_min = compute_xy_min_max(self.tracks[0],
                                                    self.tracks[1])

        hit_voxels_indices = []

        for ev in range(self.events):

            mask_x = (self.voi.voxel_edges[:,:,:,1,0]>=X_min[:,ev]) & ((self.voi.voxel_edges[:,:,:,0,0]<=X_max[:,ev]))
            mask_y = (self.voi.voxel_edges[:,:,:,1,1]>=Y_min[:,ev]) & ((self.voi.voxel_edges[:,:,:,0,1]<=Y_max[:,ev]))

            mask = mask_x & mask_y

            hit_voxels_indices.append(((mask==True).nonzero()))


        return  hit_voxels_indices