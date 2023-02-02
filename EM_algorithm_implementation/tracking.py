import math
import torch
from torch import tensor
import numpy as np
import pandas 
from volume_interest import VolumeOfInterest
import collections
import math
from operator import itemgetter

class Tracking:

    def __init__(self, voi: VolumeOfInterest, data:pandas.DataFrame):
        
        self.voi=voi
        self.voxel_edges = voi.voxel_edges
        self.Nvox_Z=voi.n_vox_xyz[2]

        self.data=data
        self.events=len(self.data)

        self.tracks = self.compute_discrete_tracks()
        self.triggered_voxels = self.find_triggered_voxels()
            
        # determining the overall length of straight line from incoming point to outgoing point
        self.path_length= np.sqrt((data['xyz_in_x'] - data['xyz_out_x'])**2 + (data['xyz_in_y'] - data['xyz_out_y'])**2 + (data['xyz_in_z'] - data['xyz_out_z'])**2)
        
        # determining the cosine directors of line

        self.cos_theta_x = (self.data['xyz_in_x'] - self.data['xyz_out_x']) / self.path_length 
        self.cos_theta_y = (self.data['xyz_in_y'] - self.data['xyz_out_y']) / self.path_length
        self.cos_theta_z = (self.data['xyz_in_z'] - self.data['xyz_out_z']) / self.path_length
        
        self.Ix, self.Iy, self.Iz = self.axis_Indices()
        
        self.alpha_x_l, self.alpha_x_r, self.alpha_y_l, self.alpha_y_r, self.alpha_z_l, self.alpha_z_r=self.compute_alpha_vals()
        self.intersection_coordinates=self.compute_intersection_coord()
    
    
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
        
        Z_discrete = torch.round(Z_discrete.expand(len(Z_discrete),len(self.data)),decimals=3)
        
        x_track = tensor(xyz_in_x) + (xyz_out_z[0]-Z_discrete)*torch.tan(theta_x)

        y_track = tensor(xyz_in_y) + (xyz_out_z[0]-Z_discrete)*torch.tan(theta_y)

        return torch.stack([x_track,y_track,Z_discrete.flip(dims=(0,))])

    
    def parametric_equation(t, index):
        return (
            data['xyz_in_x'][index] + (data['xyz_out_x'][index] - data['xyz_in_x'][index]) * t,
            data['xyz_in_y'][index] + (data['xyz_out_y'][index] - data['xyz_in_y'][index]) * t,
            data['xyz_in_z'][index] + (data['xyz_out_z'][index] - data['xyz_in_z'][index]) * t,
            
        )
    
        
    
    def find_triggered_voxels_old(self):
        
        def max_val(l, i):
            return max(enumerate(sub[i] for sub in l), key=itemgetter(1))
        
        def compute_xy_min_max(X_discrete,Y_discrete,Z_discrete):
        
            mask_x = X_discrete[0,:]>X_discrete[1,:]
            mask_y = Y_discrete[0,:]>Y_discrete[1,:]
            mask_z = Z_discrete[0,:]>Z_discrete[1,:]

            X_max = torch.where(mask_x,X_discrete[:-1,:],X_discrete[1:,:])
            X_min = torch.where(mask_x,X_discrete[1:,:],X_discrete[:-1,:])

            Y_max = torch.where(mask_y,Y_discrete[:-1,:],Y_discrete[1:,:])
            Y_min = torch.where(mask_y,Y_discrete[1:,:],Y_discrete[:-1,:])
            
            Z_max = torch.where(mask_z,Z_discrete[:-1,:],Z_discrete[1:,:])
            Z_min = torch.where(mask_z,Z_discrete[1:,:],Z_discrete[:-1,:])

            return X_max,X_min,Y_max,Y_min,Z_max,Z_min
            
        X_max,X_min,Y_max,Y_min,Z_max,Z_min = compute_xy_min_max(self.tracks[0], self.tracks[1], self.tracks[2])

        hit_voxels_indices = []

        for ev in range(self.events):

            mask_x = (self.voi.voxel_edges[:,:,:,1,0]>=X_min[:,ev]) & ((self.voi.voxel_edges[:,:,:,0,0]<=X_max[:,ev]))
            mask_y = (self.voi.voxel_edges[:,:,:,1,1]>=Y_min[:,ev]) & ((self.voi.voxel_edges[:,:,:,0,1]<=Y_max[:,ev]))
            mask_z = (self.voi.voxel_edges[:,:,:,1,2]>=Z_min[:,ev]) & ((self.voi.voxel_edges[:,:,:,0,2]<=Z_max[:,ev]))

            mask = mask_x & mask_y & mask_z

            hit_voxels_indices.append(((mask==True).nonzero()))
        
        triggered_voxels=self.filter_hit_voxels(hit_voxels_indices)


        return triggered_voxels
    
    def filter_hit_voxels(self,hit_voxels):
        
        def computeY(p1,p2,x,z):
            x1,y1,z1 = p1
            x2,y2,z2 = p2

            if x2 - x1 != 0:
                t = (x - x1) / (x2 - x1)
            elif z2 - z1 != 0:
                t = (z - z1) / (z2 - z1)
            else:
                return 0

            return y1 + (y2 - y1)*t
        
        def computeX(p1,p2,y,z):
            x1,y1,z1 = p1
            x2,y2,z2 = p2

            if y2 - y1 != 0:
                t = (y - y1) / (y2 - y1)
            elif z2 - z1 != 0:
                t = (z - z1) / (z2 - z1)
            else:
                return 0

            return x1 + (x2 - x1)*t
        
        triggered_voxels=[]
        
        for ev in range(self.events):
            
            ev_triggered_vox=[]
            p_entry=(self.data['xyz_in_x'][ev],self.data['xyz_in_y'][ev],self.data['xyz_in_z'][ev])
            p_exit=(self.data['xyz_out_x'][ev],self.data['xyz_out_y'][ev],self.data['xyz_out_z'][ev])
            voxels=hit_voxels[ev]
            
            if len(voxels)>np.sqrt(self.voi.n_vox_xyz[0]**2+self.voi.n_vox_xyz[1]**2):
                for vox in voxels:



                    x,y,z=vox[0],vox[1],vox[2]

                    centr_x=self.voi.voxel_centers[x,y,z,0]
                    centr_y=self.voi.voxel_centers[x,y,z,1]
                    centr_z=self.voi.voxel_centers[x,y,z,2]

                    x_comp=computeX(p_entry,p_exit,centr_y,centr_z)
                    y_comp=computeY(p_entry,p_exit,centr_x,centr_z)

                    if abs(x_comp-centr_x)<self.voi.vox_width and abs(y_comp-centr_y)<self.voi.vox_width:
                        ev_triggered_vox.append(vox.numpy())

                triggered_voxels.append(ev_triggered_vox)
            else:
                triggered_voxels.append(voxels)
        
        return triggered_voxels
    
    def find_triggered_voxels(self):
        
        triggered_voxels=[]
        
        for ev in range(0,self.events):
            center = self.voi.voxel_centers
            edge = self.voi.voxel_edges
            IN = self.tracks[:,0,ev]
            OUT = self.tracks[:,-1,ev]

            z = torch.linspace(0.8,0.2,100)
            theta_x =math.atan((OUT[0]-IN[0])/(OUT[2]-IN[2]))
            theta_y =math.atan((OUT[1]-IN[1])/(OUT[2]-IN[2]))

            x = IN[0]+ (z-IN[2])*math.tan(theta_x)
            y = IN[1]+ (z-IN[2])*math.tan(theta_y)

            xyz = torch.zeros(100,3)
            xyz[:,0]=x
            xyz[:,1]=y
            xyz[:,2]=z

            edge = edge[:,:,:,:,None,:]
            edge = edge.expand(10,10,6,2,len(xyz),3)
            maskx = (edge[:,:,:,0,:,0]<xyz[:,0]) & (edge[:,:,:,1,:,0]>xyz[:,0])
            masky = (edge[:,:,:,0,:,1]<xyz[:,1]) & (edge[:,:,:,1,:,1]>xyz[:,1])
            maskz = (edge[:,:,:,0,:,2]<xyz[:,2]) & (edge[:,:,:,1,:,2]>xyz[:,2])

            mask = maskx&masky&maskz
            indices = (mask==True).nonzero()

            indices = indices[:,:-1]
            indices = indices.unique(dim=0).tolist()
            triggered_voxels.append(indices)
        return triggered_voxels


    def axis_Indices(self):
        '''
        ordered lists of indices {Ix}, {Iy} and {Iz}, containing:
        for each coordinate, the integer k associated with the voxels crossed from the muon in its trajectory
        '''
        Ix=[]
        Iy=[]
        Iz=[]
        for event in self.triggered_voxels:
            sorter = lambda x: (x[2], x[0], x[1])
            sorted_vox = sorted(event, key=sorter, reverse=True)
            
            Ix_ev=[]
            Iy_ev=[]
            Iz_ev=[]

            for vox in sorted_vox:
                Ix_ev.append(vox[0])
                Iy_ev.append(vox[1])
                Iz_ev.append(vox[2])
            
            Ix.append(Ix_ev)
            Iy.append(Iy_ev)
            Iz.append(Iz_ev)

        return Ix, Iy, Iz
    
    def compute_alpha_vals(self):
        '''
        intersection between each of the left/right edges of volume in x/y/z and the trajectory
        '''
        
        #intersection between each of the left and right edges of volume in x and the trajectory
        alpha_x_l = []
        alpha_x_r = []        
        
        #intersection between each of the left and right edges of volume in y and the trajectory
        alpha_y_l = []
        alpha_y_r = []
        
        #intersection between each of the left and right edges of volume in z and the trajectory
        alpha_z_l = []
        alpha_z_r = []
        
        for i in range(0, self.events):
            x_in, y_in, z_in = self.data['xyz_in_x'][i], self.data['xyz_in_y'][i], self.data['xyz_in_z'][i]
            x_out, y_out, z_out = self.data['xyz_out_x'][i], self.data['xyz_out_y'][i], self.data['xyz_out_z'][i]
            
            alpha_x_l_i=[]
            alpha_x_r_i=[]
            
            alpha_y_l_i=[]
            alpha_y_r_i=[]
            
            alpha_z_l_i=[]
            alpha_z_r_i=[]
            
            
            
            for j in range(0, len(self.Ix[i])):
                x, y, z= self.Ix[i][j],self.Iy[i][j],self.Iz[i][j]
                
                
                alpha_l= (self.voi.voxel_edges[x, y, z, 0, 0] - self.data['xyz_in_x'][i])/self.cos_theta_x[i]
                alpha_r= (self.voi.voxel_edges[x, y, z, 1, 0] - self.data['xyz_in_x'][i])/self.cos_theta_x[i]
                
                alpha_x_l_i.append(alpha_l)
                alpha_x_r_i.append(alpha_r)
                
                alpha_l= (self.voi.voxel_edges[x, y, z, 0, 1] - self.data['xyz_in_y'][i])/self.cos_theta_y[i]
                alpha_r= (self.voi.voxel_edges[x, y, z, 1, 1] - self.data['xyz_in_y'][i])/self.cos_theta_y[i]
                
                alpha_y_l_i.append(alpha_l)
                alpha_y_r_i.append(alpha_r)
                
                alpha_l= (self.voi.voxel_edges[x, y, z, 0, 2] - self.data['xyz_in_z'][i])/self.cos_theta_z[i]
                alpha_r= (self.voi.voxel_edges[x, y, z, 1, 2] - self.data['xyz_in_z'][i])/self.cos_theta_z[i]
                
                alpha_z_l_i.append(alpha_l)
                alpha_z_r_i.append(alpha_r)
                
            
            alpha_x_l.append(alpha_x_l_i)
            alpha_x_r.append(alpha_x_r_i)
            
            alpha_y_l.append(alpha_y_l_i)
            alpha_y_r.append(alpha_y_r_i)
                
            alpha_z_l.append(alpha_z_l_i)
            alpha_z_r.append(alpha_z_r_i)

        
        return alpha_x_l, alpha_x_r, alpha_y_l, alpha_y_r, alpha_z_l, alpha_z_r
    
    
    def compute_intersection_coord(self):
        
        def max_val(l, i):
            return max(enumerate(sub[i] for sub in l), key=itemgetter(1))
        
        
        # parametric equation of a line given coordinate of one point in space and associated cosine director
        def trajectory(alpha, index):
            return (
            round(self.data['xyz_in_x'][index] + self.cos_theta_x[index] * alpha.numpy(),4), 
            round(self.data['xyz_in_y'][index] + self.cos_theta_y[index] * alpha.numpy(),4), 
            round(self.data['xyz_in_z'][index] + self.cos_theta_z[index] * alpha.numpy(),4)
            )
        
        coordinates=[]
        t_0=tensor(0)
        
        # determining the coordinates of each intersection of trajectory with edges of each hit voxel
        # (keeping in mind that incoming and outgoing measured points on the track do not necessarily correspond to entry and exit points
        # in the volume)
        
        for ev in range(0, self.events):
            
            x_in=self.data['xyz_in_x'][ev]
            x_out=self.data['xyz_out_x'][ev]
            
            y_in=self.data['xyz_in_y'][ev]
            y_out=self.data['xyz_out_y'][ev]
            
            z_in=self.data['xyz_in_z'][ev]
            z_out=self.data['xyz_out_z'][ev]
            ev_coord=[]
            
            if len(self.triggered_voxels[ev])>1: #event has more than one triggered voxels 
                
                
                ev_coord.append((0,0,0))
                
                for hit_vox in range(0, len(self.triggered_voxels[ev])-1): 
                    vox=(self.Ix[ev][hit_vox],self.Iy[ev][hit_vox],self.Iz[ev][hit_vox])
                    
                    if self.Iz[ev][hit_vox] != self.Iz[ev][hit_vox+1]: #checking if muon changed layer
                        coord=trajectory(self.alpha_z_l[ev][hit_vox+1],ev)

                    elif self.Iy[ev][hit_vox] > self.Iy[ev][hit_vox+1]: #particle is moving from right to left
                        coord=trajectory(self.alpha_y_l[ev][hit_vox],ev)

                    elif self.Iy[ev][hit_vox] < self.Iy[ev][hit_vox+1]: #particle is moving from left to right in y dir
                        coord=trajectory(self.alpha_y_r[ev][hit_vox],ev)
                    
                    #particle has crossed two or more cubes belonging to the same layer along the z axis
                    #and at the same time two or more cubes with the same index for the y dimension
                    
                    elif self.Ix[ev][hit_vox] > self.Ix[ev][hit_vox+1]:
                        coord=trajectory(self.alpha_x_l[ev][hit_vox],ev)

                    else:
                        coord=trajectory(self.alpha_x_r[ev][hit_vox],ev)

                    if coord not in ev_coord:
                        ev_coord.append(coord)

                ev_coord.append((0,0,0))
                
            else:
                ev_coord.append((0,0,0))
            
            
            #if the measured incoming point lies on the upper face of volume, then substitute the first element of list of coordinates determined
            #before by the true coordinates of incoming point from the dataset; else the coordinates remain zeros
            
            if len(self.triggered_voxels[ev])>0:
                
                if x_in >= 0 and x_in <= 1 and y_in >= 0 and y_in <= 1:
                    ev_coord[0] = (x_in, y_in, z_in)

                if x_out >= 0 and x_out <= 1 and y_out >= 0 and y_out <= 1:
                    ev_coord[-1]=(x_out, y_out, z_out)
                
                # checking all other cases in which the outgoing measured point doesn't lie on the lower layer of volume and replacing existing zeros with adequate coordinates
                exit_vox_indx=max_val(self.triggered_voxels[ev],2)[0]
                exit_vox_x=self.voi.voxel_centers[self.triggered_voxels[ev][exit_vox_indx][0],self.triggered_voxels[ev][exit_vox_indx][1],self.triggered_voxels[ev][exit_vox_indx][2],0]
                exit_vox_y=self.voi.voxel_centers[self.triggered_voxels[ev][exit_vox_indx][0],self.triggered_voxels[ev][exit_vox_indx][1],self.triggered_voxels[ev][exit_vox_indx][2],1]

                if ev_coord[-1][0]==0 and x_out > 1 and y_out>= 0 and y_out<= 1 and len(self.alpha_x_r[ev])>0:
                    ev_coord[-1] = trajectory(self.alpha_x_r[ev][-1], ev)

                elif ev_coord[-1][0]==0 and x_out < 0 and y_out>= 0 and y_out<= 1 and len(self.alpha_x_l[ev])>0:
                    ev_coord[-1] = trajectory(self.alpha_x_l[ev][-1], ev)

                elif ev_coord[-1][0]==0 and y_out > 1 and x_out>= 0 and x_out<= 1 and len(self.alpha_y_r[ev])>0:
                    ev_coord[-1] = trajectory(self.alpha_y_r[ev][-1], ev)

                elif ev_coord[-1][0]==0 and y_out < 0 and x_out>= 0 and x_out<= 1 and len(self.alpha_y_l[ev])>0:
                    ev_coord[-1] = trajectory(self.alpha_y_l[ev][-1], ev)

                # checking all other cases in which the outgoing measured point doesn't lie on the lower layer of volume and replacing existing zeros with adequate coordinates 

                elif ev_coord[-1][0]==0 and y_out > 1 and x_out > 1 and len(self.alpha_x_r[ev])>0:
                    
                    if abs(x_out - exit_vox_x.numpy()) > abs(y_out - exit_vox_y.numpy()):
                        ev_coord[-1] = trajectory(self.alpha_x_r[ev][-1], ev)
                    else:
                        ev_coord[-1] = trajectory(self.alpha_y_r[ev][-1], ev)

                elif ev_coord[-1][0]==0 and y_out <0 and x_out > 1 and len(self.alpha_x_r[ev])>0:

                    if abs(x_out - exit_vox_x.numpy()) > abs(y_out - exit_vox_y.numpy()):
                        ev_coord[-1] = trajectory(self.alpha_x_r[ev][-1], ev)
                    else:
                        ev_coord[-1] = trajectory(self.alpha_y_l[ev][-1], ev)


                elif ev_coord[-1][0]==0 and y_out <0 and x_out < 0 and len(self.alpha_x_r[ev])>0:
                    
                    if abs(x_out - exit_vox_x.numpy()) > abs(y_out - exit_vox_y.numpy()):
                        ev_coord[-1] = trajectory(self.alpha_x_l[ev][-1], ev)
                    else:
                        ev_coord[-1] = trajectory(self.alpha_y_l[ev][-1], ev)
            
            
            # checking all other cases in which the incoming measured point doesn't lie on the upper layer of volume and replacing existing zeros with adequate coordinates
            
                centr_vox_x=self.voi.voxel_centers[self.triggered_voxels[ev][0][0],self.triggered_voxels[ev][0][1],self.triggered_voxels[ev][0][2],0]
                centr_vox_y=self.voi.voxel_centers[self.triggered_voxels[ev][0][0],self.triggered_voxels[ev][0][1],self.triggered_voxels[ev][0][2],1]

                #checking initial x coordinate
                if ev_coord[0][0]==0 and x_in>1 and y_in>=0 and y_in<= 1:
                    #update entry point 
                    ev_coord[0] = trajectory(self.alpha_x_r[ev][0], ev)

                elif ev_coord[0][0]==0 and x_in<0 and y_in>=0 and y_in<= 1:
                    ev_coord[0] = trajectory(self.alpha_x_l[ev][0], ev)

                elif ev_coord[0][0]==0 and y_in>1 and x_in>=0 and x_in<=1:
                    ev_coord[0] = trajectory(self.alpha_y_r[ev][0], ev)

                elif ev_coord[0][0]==0 and y_in < 0 and x_in>= 0 and x_in<= 1:
                    ev_coord[0] = trajectory(self.alpha_y_l[ev][0], ev)

                elif ev_coord[0][0]==0 and y_in > 1 and x_in > 1:

                    #if len(self.triggered_voxels[ev])>1:

                    if abs( x_in - centr_vox_x ) > abs( y_in - centr_vox_y ):
                        ev_coord[0] = trajectory(self.alpha_x_r[ev][0], ev)

                    else:
                        ev_coord[0] = trajectory(self.alpha_y_r[ev][0], ev)

                elif ev_coord[0][0]==0 and y_in < 0 and x_in > 1:

                    if abs( x_in - centr_vox_x ) > abs( y_in - centr_vox_y ):
                        ev_coord[0] = trajectory(self.alpha_x_r[ev][0], ev)

                    else:
                        ev_coord[0] = trajectory(self.alpha_y_l[ev][0], ev)

                elif ev_coord[0][0]==0 and y_in < 0 and x_in < 0:

                    if abs( x_in - centr_vox_x ) > abs( y_in - centr_vox_y ):
                        ev_coord[0] = trajectory(self.alpha_x_l[ev][0], ev)

                    else:
                        ev_coord[0] = trajectory(self.alpha_y_l[ev][0], ev)

                elif ev_coord[0][0]==0 and y_in > 1 and x_in < 0:

                    if abs( x_in - centr_vox_x ) > abs( y_in - centr_vox_y ):
                        ev_coord[0] = trajectory(self.alpha_x_l[ev][0], ev)

                    else:
                        ev_coord[0] = trajectory(self.alpha_y_r[ev][0], ev)
            
            coordinates.append(ev_coord)
        
        return coordinates
    