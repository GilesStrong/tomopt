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
        self.triggered_voxels = self.identify_triggered_voxels()

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

    
    def identify_triggered_voxels(self):

        """
        Identifying indices of voxels triggered given a muon trajectory. 

        INPUT: Muons tracks - tensor of size = [coordinate,Nlayer_along_Z + 1, Nevents] ([3,7,9999])
        OUPUT: List of voxels' indices triggered by every muon - tensor of size = [Nevents, N_var_Voxel_indices]

        Logic: 
                check along every Z level whether track of the muon is within >Xin & <Xout
                if satisfied
                save the indices of the voxel .... indices would be 3D coordinates within N,N,N1
        """
        from fastprogress import progress_bar

        print('\nVoxel triggering: in pogress')
        event_triggered_voxels=[]
        level_triggered_voxels=tensor(())
        for event in progress_bar(range(self.events)):
            for z_i in range(0,self.Nvox_Z):

                muon_x_position_in  = self.tracks[0,z_i,event]
                muon_y_position_in  = self.tracks[1,z_i,event]
                muon_z_position_in  = self.tracks[2,z_i,event]
                muon_x_position_out = self.tracks[0,z_i+1,event]
                muon_y_position_out = self.tracks[1,z_i+1,event]
                muon_z_position_out = self.tracks[2,z_i+1,event]

                if muon_x_position_in>muon_x_position_out:
                    x_max=muon_x_position_in
                    x_min=muon_x_position_out
                else:
                    x_max=muon_x_position_out
                    x_min=muon_x_position_in

                if muon_y_position_in>muon_y_position_out:
                    y_max=muon_y_position_in
                    y_min=muon_y_position_out
                else:
                    y_max=muon_y_position_out
                    y_min=muon_y_position_in

                if muon_z_position_in>muon_z_position_out:
                    z_max=muon_z_position_in
                    z_min=muon_z_position_out
                else:
                    z_max=muon_z_position_out
                    z_min=muon_z_position_in

                mask_x = (self.voxel_edges[:,:,:,0,0]<x_max) & (self.voxel_edges[:,:,:,1,0]>x_min)
                mask_y = (self.voxel_edges[:,:,:,0,1]<y_max) & (self.voxel_edges[:,:,:,1,1]>y_min)
                mask_z = (self.voxel_edges[:,:,:,0,2]<=z_max) & (self.voxel_edges[:,:,:,1,2]>=z_min)


                mask = mask_x & mask_y & mask_z
                triggered_voxels = (mask==True).nonzero()
                level_triggered_voxels = torch.cat((level_triggered_voxels, triggered_voxels), 0)
            event_triggered_voxels.append(torch.unique(level_triggered_voxels, dim=0))
            level_triggered_voxels = tensor(())
        print('\nVoxel triggering: DONE')

        return event_triggered_voxels


        
    
    