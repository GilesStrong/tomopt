import warnings
warnings.filterwarnings("ignore")
import math
import torch
from torch import tensor
import numpy as np
import pandas 
import math
from numpy import cross
from skspatial.objects import Line
from skspatial.objects import Points
#from skspatial.plotting import plot_3d
from fastprogress import progress_bar
from operator import itemgetter
from volume_interest import *
class Tracking:

    def __init__(self, voi: VolumeOfInterest, data:pandas.DataFrame, em_iter:int):
        
        self.voi=voi
        self.voxel_edges = voi.voxel_edges
        self.Nvox_Z=voi.n_vox_xyz[2]

        self.data=data
        self.em_iter=em_iter
        self.data=data
        self.tracks = self.compute_discrete_tracks()
        self.triggered_voxels,self.indices,self.L1_vox,self.L6_vox= self.find_triggered_voxels()
        self.tracks = self.compute_discrete_tracks()
        
        
        self.path_length= np.sqrt((self.data['xyz_in_x'] - self.data['xyz_out_x'])**2 + (self.data['xyz_in_y'] - self.data['xyz_out_y'])**2 + (self.data['xyz_in_z'] - self.data['xyz_out_z'])**2)
        
        self.cos_theta_x = (self.data['xyz_in_x'] - self.data['xyz_out_x']) / self.path_length 
        self.cos_theta_y = (self.data['xyz_in_y'] - self.data['xyz_out_y']) / self.path_length
        self.cos_theta_z = (self.data['xyz_in_z'] - self.data['xyz_out_z']) / self.path_length
        
        
        self.alpha_x_l, self.alpha_x_r, self.alpha_y_l, self.alpha_y_r, self.alpha_z_l, self.alpha_z_r=self.compute_alpha_vals()
        self.intersection_coordinates=self.compute_intersection_coord()
        self.W, self.M,self.L, self.T, self.Hit = self.calculate_path_length()
        self.Dx,self.Dy=self.get_observed_data()
        self.pr,self._lambda_=self.compute_init_scatter_density()
        self.rad_length,self.scattering_density=self.em_reconstruction()

    
    def compute_discrete_tracks(self):
        
        '''
        Function computes x,y,z position at Zmax and Zmin of each voxel layer (for incoming and outgoing tracks)
        
        Outputs: 
                track_in_discrete, track_out_discrete: x,y,z position at Zmax and Zmin of each voxel 
                layer (for incoming and outgoing tracks), 
                size = [coordinate,Nlayer_along_Z + 1, Nevents] 
        '''

        voxels_edges=self.voxel_edges
        
        xyz_in_x = self.data['xyz_in_x']
        xyz_in_y = self.data['xyz_in_y']
        xyz_in_z = self.data['xyz_in_z']
    
        xyz_out_x = self.data['xyz_out_x']
        xyz_out_y = self.data['xyz_out_y']
        xyz_out_z = self.data['xyz_out_z']
         
        dx= xyz_out_x - xyz_in_x
        dy= xyz_out_y - xyz_in_y
        dz= xyz_out_z - xyz_in_z
        
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


    def find_triggered_voxels(self):
        
        '''
        Function identifies the list of voxels that were triggered by the particle, along with 
        their respective indices in the input data file.
        
        Outputs:

            triggered_voxels: A list of lists of tuples, where each tuple represents a voxel. 
                              The list contains indices of all the voxels that were triggered 
                              by the particle.
                              
            vox_loc: A list of the indices of events where the triggered voxels were detected.
        '''
        def common(a,b): 
            c = [value for value in a if value in b] 
            return c
        
        triggered_voxels = []
        vox_loc = []
        events_at_boundary_layer1=[]
        events_at_boundary_layer6=[]

        boundary_voxels_1=[[0,0,0],[0,1,0],[0,2,0],[0,3,0],[0,4,0],[0,5,0],[0,6,0],[0,7,0],[0,8,0],[0,8,0],
                         [9,0,0],[9,1,0],[9,2,0],[9,3,0],[9,4,0],[9,5,0],[9,6,0],[9,7,0],[9,8,0],
                         [0,9,0],[1,9,0],[2,9,0],[3,9,0],[4,9,0],[5,9,0],[6,9,0],[7,9,0],[8,9,0],[9,9,0],
                         [1,0,0],[2,0,0],[3,0,0],[4,0,0],[5,0,0],[6,0,0],[7,0,0],[8,0,0],[9,0,0]
                        ]
        boundary_voxels_6=[[0,0,5],[0,1,5],[0,2,5],[0,3,5],[0,4,5],[0,5,5],[0,6,5],[0,7,5],[0,8,5],[0,8,5],
                         [9,0,5],[9,1,5],[9,2,5],[9,3,5],[9,4,5],[9,5,5],[9,6,5],[9,7,5],[9,8,5],
                         [0,9,5],[1,9,5],[2,9,5],[3,9,5],[4,9,5],[5,9,5],[6,9,5],[7,9,5],[8,9,5],[9,9,5],
                         [1,0,5],[2,0,5],[3,0,5],[4,0,5],[5,0,5],[6,0,5],[7,0,5],[8,0,5],[9,0,5]]

        for ev in range(len(self.data)):
            center = self.voi.voxel_centers
            edge = self.voi.voxel_edges
            IN = self.tracks[:, 0, ev]
            OUT = self.tracks[:, -1, ev]

            z = torch.linspace(0.8, 0.2, 100)
            theta_x = math.atan((OUT[0] - IN[0]) / (OUT[2] - IN[2]))
            theta_y = math.atan((OUT[1] - IN[1]) / (OUT[2] - IN[2]))

            x = IN[0] + (z - IN[2]) * math.tan(theta_x)
            y = IN[1] + (z - IN[2]) * math.tan(theta_y)

            xyz = torch.zeros(100, 3)
            xyz[:, 0] = x
            xyz[:, 1] = y
            xyz[:, 2] = z

            edge = edge[:, :, :, :, None, :]
            edge = edge.expand(10, 10, 6, 2, len(xyz), 3)
            mask = ((edge[:, :, :, 0, :, 0] < xyz[:, 0]) & (edge[:, :, :, 1, :, 0] > xyz[:, 0]) &
                    (edge[:, :, :, 0, :, 1] < xyz[:, 1]) & (edge[:, :, :, 1, :, 1] > xyz[:, 1]) &
                    (edge[:, :, :, 0, :, 2] < xyz[:, 2]) & (edge[:, :, :, 1, :, 2] > xyz[:, 2]))

            indices = torch.nonzero(mask, as_tuple=False)
            indices = indices[:, :-1].unique(dim=0).tolist()

            if len(indices) > 0:
                
                indices = sorted(indices, key=lambda k: (k[2], k[1], k[0]))

                pair = (indices[0], indices[-1])
                if pair[0][1:] == pair[1][1:] and ((pair[0][0] > pair[1][0] and indices[0][0] < indices[-1][0]) or (pair[0][0] < pair[1][0] and indices[0][0] > indices[-1][0])):
                    indices.reverse()

                triggered_voxels.append(indices)
                vox_loc.append(ev)
                if len(common(indices,boundary_voxels_1)):
                    events_at_boundary_layer1.append(ev)
                if len(common(indices,boundary_voxels_6)):
                    events_at_boundary_layer6.append(ev)
                
                    

        return triggered_voxels, vox_loc, events_at_boundary_layer1, events_at_boundary_layer6

    
    
    def compute_alpha_vals(self):
        
        '''
        Function computes the alpha values for all voxels triggered by muons.
        
        alpha parameter is often used to calculate the attenuation or absorption of the muon as it passes 
        through a material or tissue.
        
        The subscripts _l and _r in the list names indicate whether the alpha value corresponds to the left 
        or right edge of a voxel along a particular axis.
        
        
        Outputs:
            alpha_x_l, alpha_x_r, 
            alpha_y_l, alpha_y_r, 
            alpha_z_l, alpha_z_r: Each list contains sublists, which represent alpha values calculated 
                                  from the input data for different voxels.
        '''
    
        alpha_x_l = []
        alpha_x_r = []        
        
        alpha_y_l = []
        alpha_y_r = []
        
        alpha_z_l = []
        alpha_z_r = []
        
        for i in range(0, len(self.triggered_voxels)):
            x_in, y_in, z_in = self.data.iloc[self.indices[i]]['xyz_in_x'], self.data.iloc[self.indices[i]]['xyz_in_y'], self.data.iloc[self.indices[i]]['xyz_in_z']
            x_out, y_out, z_out = self.data.iloc[self.indices[i]]['xyz_out_x'], self.data.iloc[self.indices[i]]['xyz_out_y'], self.data.iloc[self.indices[i]]['xyz_out_z']
            
            alpha_x_l_i=[]
            alpha_x_r_i=[]
            
            alpha_y_l_i=[]
            alpha_y_r_i=[]
            
            alpha_z_l_i=[]
            alpha_z_r_i=[]
                        
            
            for j in range(0, len(self.triggered_voxels[i])):
                x, y, z= self.triggered_voxels[i][j][0],self.triggered_voxels[i][j][1],self.triggered_voxels[i][j][2]
                
                
                alpha_l= (self.voi.voxel_edges[x, y, z, 0, 0] - x_in)/self.cos_theta_x.iloc[self.indices[i]]
                alpha_r= (self.voi.voxel_edges[x, y, z, 1, 0] - x_in)/self.cos_theta_x.iloc[self.indices[i]]
                
                alpha_x_l_i.append(alpha_l)
                alpha_x_r_i.append(alpha_r)
                
                alpha_l= (self.voi.voxel_edges[x, y, z, 0, 1] - y_in)/self.cos_theta_y.iloc[self.indices[i]]
                alpha_r= (self.voi.voxel_edges[x, y, z, 1, 1] - y_in)/self.cos_theta_y.iloc[self.indices[i]]
                
                alpha_y_l_i.append(alpha_l)
                alpha_y_r_i.append(alpha_r)
                
                alpha_l= (self.voi.voxel_edges[x, y, z, 0, 2] - z_in)/self.cos_theta_z.iloc[self.indices[i]]
                alpha_r= (self.voi.voxel_edges[x, y, z, 1, 2] - z_in)/self.cos_theta_z.iloc[self.indices[i]]
                
                alpha_z_l_i.append(alpha_l)
                alpha_z_r_i.append(alpha_r)
                
            
            alpha_x_l.append(alpha_x_r_i)
            alpha_x_r.append(alpha_x_l_i)
            
            alpha_y_l.append(alpha_y_l_i)
            alpha_y_r.append(alpha_y_r_i)
                
            alpha_z_l.append(alpha_z_r_i)
            alpha_z_r.append(alpha_z_l_i)

        
        return alpha_x_l, alpha_x_r, alpha_y_l, alpha_y_r, alpha_z_l, alpha_z_r  
    
    def compute_intersection_coord(self):
        
        '''
        Function calculates the trajectory of particles passing through a given volume 
        of interest, based on their start and end positions, and the intersection points 
        between voxels and particle trajectories.
        
        Outputs:
                coordinates: list of coordinates that describe the trajectory of each particle 
                             with shape (num_events,).
        '''
        def max_val(l, i):
            return max(enumerate(sub[i] for sub in l), key=itemgetter(1))
        
        
        def trajectory(alpha, index):
            return (
           self.data.iloc[index]['xyz_in_x'] + self.cos_theta_x.iloc[index] * alpha.numpy(), 
           self.data.iloc[index]['xyz_in_y'] + self.cos_theta_y.iloc[index] * alpha.numpy(), 
           self.data.iloc[index]['xyz_in_z'] + self.cos_theta_z.iloc[index] * alpha.numpy()
            )
        
        coordinates=[]
        triggered_voxels=self.triggered_voxels
        
        
        for ev in range(0, len(self.triggered_voxels)):
            ev_indx=self.indices[ev]
            x_in=self.data.iloc[ev_indx]['xyz_in_x']
            x_out=self.data.iloc[ev_indx]['xyz_out_x']
            y_in=self.data.iloc[ev_indx]['xyz_in_y']
            y_out=self.data.iloc[ev_indx]['xyz_out_y']
            z_in=self.data.iloc[ev_indx]['xyz_in_z']
            z_out=self.data.iloc[ev_indx]['xyz_out_z']
            ev_coord=[]
            
            if len(self.triggered_voxels[ev])>0: 
                ev_coord.append((0,0,0))
                
                for hit_vox in range(0, len(self.triggered_voxels[ev])-1): 
                    curr_vox_x, curr_vox_y, curr_vox_z = triggered_voxels[ev][hit_vox]
                    next_vox_x, next_vox_y, next_vox_z = triggered_voxels[ev][hit_vox + 1]
                
                    vox = (curr_vox_x, curr_vox_y, curr_vox_z)
                    if curr_vox_z != next_vox_z: 
                        coord=trajectory(self.alpha_z_l[ev][hit_vox+1],ev_indx)

                    elif curr_vox_y > next_vox_y: 
                        coord=trajectory(self.alpha_y_l[ev][hit_vox],ev_indx)

                    elif curr_vox_y < next_vox_y: 
                        coord=trajectory(self.alpha_y_r[ev][hit_vox],ev_indx)
                    
                    elif curr_vox_x > next_vox_x:
                        coord=trajectory(self.alpha_x_l[ev][hit_vox],ev_indx)

                    else:
                        coord=trajectory(self.alpha_x_r[ev][hit_vox],ev_indx)

                    if coord not in ev_coord and (coord[2]<=0.8 and coord[2]>=0.2):
                        ev_coord.append(coord)

                ev_coord.append((0,0,0))
            
            
            if len(self.triggered_voxels[ev])>0:
                
                if x_in >= 0 and x_in <= 1 and y_in >= 0 and y_in <= 1:
                    ev_coord[0] = (x_in, y_in, z_in)
                

                if x_out >= 0 and x_out <= 1 and y_out >= 0 and y_out <= 1:
                    ev_coord[-1]=(x_out, y_out, z_out)
                
                exit_vox_indx=max_val(self.triggered_voxels[ev],2)[0]
                exit_vox_x=self.voi.voxel_centers[self.triggered_voxels[ev][exit_vox_indx][0],self.triggered_voxels[ev][exit_vox_indx][1],self.triggered_voxels[ev][exit_vox_indx][2],0]
                exit_vox_y=self.voi.voxel_centers[self.triggered_voxels[ev][exit_vox_indx][0],self.triggered_voxels[ev][exit_vox_indx][1],self.triggered_voxels[ev][exit_vox_indx][2],1]
                
                if ev_coord[-1][0]==0 and x_out > 1 and y_out>= 0 and y_out<= 1 and len(self.alpha_x_r[ev])>0:
                    coord = trajectory(self.alpha_x_r[ev][-1], ev_indx)
                    if coord[2]>=0.8 and coord[2]<=0.2:
                        ev_coord[-1]=coord

                elif ev_coord[-1][0]==0 and x_out < 0 and y_out>= 0 and y_out<= 1 and len(self.alpha_x_l[ev])>0:
                    coord = trajectory(self.alpha_x_l[ev][-1], ev_indx)
                    if coord[2]>=0.8 and coord[2]<=0.2:
                        ev_coord[-1]=coord

                elif ev_coord[-1][0]==0 and y_out > 1 and x_out>= 0 and x_out<= 1 and len(self.alpha_y_r[ev])>0:
                    coord = trajectory(self.alpha_y_r[ev][-1], ev_indx)
                    if coord[2]>=0.8 and coord[2]<=0.2:
                        ev_coord[-1]=coord

                elif ev_coord[-1][0]==0 and y_out < 0 and x_out>= 0 and x_out<= 1 and len(self.alpha_y_l[ev])>0:
                    coord = trajectory(self.alpha_y_l[ev][-1], ev_indx)
                    if coord[2]>=0.8 and coord[2]<=0.2:
                        ev_coord[-1]=coord


                elif ev_coord[-1][0]==0 and y_out > 1 and x_out > 1 and len(self.alpha_x_r[ev])>0:
                    
                    if abs(x_out - exit_vox_x.numpy()) > abs(y_out - exit_vox_y.numpy()):
                        coord = trajectory(self.alpha_x_r[ev][-1], ev_indx)
                        if coord[2]>=0.8 and coord[2]<=0.2:
                            ev_coord[-1]=coord
                    else:
                        coord = trajectory(self.alpha_y_r[ev][-1], ev_indx)

                elif ev_coord[-1][0]==0 and y_out <0 and x_out > 1 and len(self.alpha_x_r[ev])>0:

                    if abs(x_out - exit_vox_x.numpy()) > abs(y_out - exit_vox_y.numpy()):
                        coord = trajectory(self.alpha_x_r[ev][-1], ev_indx)
                        if coord[2]>=0.8 and coord[2]<=0.2:
                            ev_coord[-1]=coord
                    else:
                        coord = trajectory(self.alpha_y_l[ev][-1], ev_indx)
                        if coord[2]>=0.8 and coord[2]<=0.2:
                            ev_coord[-1]=coord


                elif ev_coord[-1][0]==0 and y_out <0 and x_out < 0 and len(self.alpha_x_r[ev])>0:
                    
                    if abs(x_out - exit_vox_x.numpy()) > abs(y_out - exit_vox_y.numpy()):
                        coord = trajectory(self.alpha_x_l[ev][-1], ev_indx)
                        if coord[2]>=0.8 and coord[2]<=0.2:
                            ev_coord[-1]=coord
                    else:
                        coord = trajectory(self.alpha_y_l[ev][-1], ev_indx)
                        if coord[2]>=0.8 and coord[2]<=0.2:
                            ev_coord[-1]=coord
            
                centr_vox_x=self.voi.voxel_centers[self.triggered_voxels[ev][0][0],self.triggered_voxels[ev][0][1],self.triggered_voxels[ev][0][2],0]
                centr_vox_y=self.voi.voxel_centers[self.triggered_voxels[ev][0][0],self.triggered_voxels[ev][0][1],self.triggered_voxels[ev][0][2],1]

                if ev_coord[0][0]==0 and x_in>1 and y_in>=0 and y_in<= 1:
                    coord = trajectory(self.alpha_x_r[ev][0], ev_indx)
                    if coord[2]>=0.8 and coord[2]<=0.2:
                            ev_coord[0]=coord

                elif ev_coord[0][0]==0 and x_in<0 and y_in>=0 and y_in<= 1:
                    coord = trajectory(self.alpha_x_l[ev][0], ev_indx)
                    if coord[2]>=0.8 and coord[2]<=0.2:
                            ev_coord[0]=coord

                elif ev_coord[0][0]==0 and y_in>1 and x_in>=0 and x_in<=1:
                    coord = trajectory(self.alpha_y_r[ev][0], ev_indx)
                    if coord[2]>=0.8 and coord[2]<=0.2:
                            ev_coord[0]=coord

                elif ev_coord[0][0]==0 and y_in < 0 and x_in>= 0 and x_in<= 1:
                    coord = trajectory(self.alpha_y_l[ev][0], ev_indx)
                    if coord[2]>=0.8 and coord[2]<=0.2:
                            ev_coord[0]=coord

                elif ev_coord[0][0]==0 and y_in > 1 and x_in > 1:
                    if abs( x_in - centr_vox_x ) > abs( y_in - centr_vox_y ):
                        coord = trajectory(self.alpha_x_r[ev][0], ev_indx)
                    else:
                        coord = trajectory(self.alpha_y_r[ev][0], ev_indx)
                    if coord[2]>=0.8 and coord[2]<=0.2:
                        ev_coord[0]=coord

                elif ev_coord[0][0]==0 and y_in < 0 and x_in > 1:
                    if abs( x_in - centr_vox_x ) > abs( y_in - centr_vox_y ):
                        coord = trajectory(self.alpha_x_r[ev][0], ev_indx)
                    else:
                        coord = trajectory(self.alpha_y_l[ev][0], ev_indx)
                    if coord[2]>=0.8 and coord[2]<=0.2:
                        ev_coord[0]=coord

                elif ev_coord[0][0]==0 and y_in < 0 and x_in < 0:
                    if abs( x_in - centr_vox_x ) > abs( y_in - centr_vox_y ):
                        coord = trajectory(self.alpha_x_l[ev][0], ev_indx)
                    else:
                        coord = trajectory(self.alpha_y_l[ev][0], ev_indx)
                    if coord[2]>=0.8 and coord[2]<=0.2:
                        ev_coord[0]=coord

                elif ev_coord[0][0]==0 and y_in > 1 and x_in < 0:
                    if abs( x_in - centr_vox_x ) > abs( y_in - centr_vox_y ):
                        coord = trajectory(self.alpha_x_l[ev][0], ev_indx)
                    else:
                        coord = trajectory(self.alpha_y_r[ev][0], ev_indx)
                    if coord[2]>=0.8 and coord[2]<=0.2:
                        ev_coord[0]=coord
                        
            if ev_coord[0]==(0,0,0):
                del ev_coord[0]
            if ev_coord[-1]==(0,0,0):
                del ev_coord[-1]
            coordinates.append(ev_coord)
        
        return coordinates
    
    def calculate_path_length(self):
        
        '''
        Function calculates the path lengths between intersections along the path taken by muon through a medium.
        
        Outputs:

            W: a torch tensor of shape (len(self.triggered_voxels), N, N, N1, 2, 2), 
                where N, N1 are the dimensions of the voxel grid. It contains the path integrals 
                for each voxel and each interval in the 2x2 matrix format.
            
            M: a torch tensor of shape (N, N, N1), containing the number of path integrals that 
                pass through each voxel.
            
            L: a torch tensor of shape (len(self.triggered_voxels), N, N, N1) 
                         containing the length of the muon path in each voxel.
            
            T: a torch tensor of shape (len(self.triggered_voxels), N, N, N1) containing the 
                accumulated length of the muon path from the end of each voxel to the end of the path.
            
            Hit: a torch tensor of shape (len(self.triggered_voxels), N, N, N1), containing boolean values 
                  indicating which voxels are crossed by the muon path.
        
        '''


        def compute_path_length(coordinates):
            path_len=[]
            for coords_arr in coordinates:
                coords_arr = np.array(coords_arr)
                diff = np.diff(coords_arr, axis=0)
                if len(diff)>0:
                    path_len.append(np.sqrt(np.sum(diff**2, axis=1)))
            return path_len

        def sum1(l):
                    from itertools import accumulate
                    return list(accumulate(l))
        path_len=compute_path_length(self.intersection_coordinates)

        N,N1=self.voi.n_vox_xyz[0],self.voi.n_vox_xyz[2]
        M = torch.zeros(self.voi.n_vox_xyz[0],self.voi.n_vox_xyz[1],self.voi.n_vox_xyz[2])
        Hit = torch.zeros(len(self.triggered_voxels),self.voi.n_vox_xyz[0],self.voi.n_vox_xyz[1],self.voi.n_vox_xyz[2])
        L = torch.zeros(len(self.triggered_voxels), self.voi.n_vox_xyz[0], self.voi.n_vox_xyz[1], self.voi.n_vox_xyz[2])
        T = torch.zeros(len(self.triggered_voxels), self.voi.n_vox_xyz[0], self.voi.n_vox_xyz[1], self.voi.n_vox_xyz[2])
        W = torch.zeros(len(self.triggered_voxels), N, N, N1, 2, 2)

        for i in range(len(self.triggered_voxels)):
            idices_len_cumsum_ev = []
            voxels = self.triggered_voxels[i]
            if i<len(path_len):
                lrev = list(reversed(path_len[i]))
                lrev_cum_sum = sum1(lrev)
                #lrev_cum_sum.pop()
                lrev_cum_sum_rev = list(reversed(lrev_cum_sum))
                lrev_cum_sum_rev.append(0)

                for j in range(len(voxels)):
                    vox = voxels[j]
                    if j < len(path_len[i]):
                        L[i, vox[0], vox[1], vox[2]] = path_len[i][j]
                    if j < len(lrev_cum_sum_rev):
                        T[i, vox[0], vox[1], vox[2]] = lrev_cum_sum_rev[j]

            W[i, :, :, :, 0, 0] = L[i, :, :, :]
            W[i, :, :, :, 0, 1] = (L[i, :, :, :] ** 2) / 2 + L[i, :, :, :] * T[i, :, :, :]
            W[i, :, :, :, 1, 0] = (L[i, :, :, :] ** 2) / 2 + L[i, :, :, :] * T[i, :, :, :]
            W[i, :, :, :, 1, 1] = (L[i, :, :, :] ** 3) / 2 + (L[i, :, :, :] ** 2) * T[i, :, :, :] + L[i, :, :, :] * (T[i, :, :, :] ** 2)

            Hit[i,:,:,:] = (L[i,:,:,:] != 0)
            M[:,:,:] += Hit[i,:,:,:]

        return W, M, L, T, Hit


    def compute_init_scatter_density(self):
        
        '''
        Function calculates the initial scatter density based on the input voxelized object.
        
        Outputs:

            pr: a torch tensor of shape (N, N, N1), 
                containing the density of the medium that the muon path crosses
            lambda: a torch tensor of shape (N, N, N1), containing the attenuation 
                    coefficient of the medium that the muon path crosses
        '''
        
        #0.977
        L_rad=torch.full((self.voi.n_vox_xyz[0],self.voi.n_vox_xyz[1],self.voi.n_vox_xyz[2]), 0.3166)
        p0 = 5e9
        _lambda_ = ((15e6/p0)**2) * (1/L_rad)
        pr=p0/(self.data['mom']*1e9)
        return  pr,_lambda_   
    
    def get_observed_data(self):
        
        '''
        Function computes the differences in thetas and path lengths in x and y directions, respectively, 
        for voxels triggered by muons.
        
        Outputs: 
            Dx and Dy: tensors with shape (len(self.triggered_voxels), 2), representing the calculated 
                       differences in thetas and path lengths in x and y directions, respectively.
        '''
        Dy = torch.zeros(len(self.triggered_voxels),2)
        Dx = torch.zeros(len(self.triggered_voxels),2)
        indx_count=0

        for i in self.indices:
            deltathetax = self.data['theta_out_x'][i] - self.data['theta_in_x'][i]
            deltax = self.data['xyz_out_x'][i] - self.data['xyz_in_x'][i]

            deltathetay = self.data['theta_out_y'][i] - self.data['theta_in_y'][i]
            deltay = self.data['xyz_out_y'][i] - self.data['xyz_in_y'][i]
        

            x_projected = self.data['xyz_in_x'][i] + np.tan(self.data['theta_in_x'][i])*(self.data['xyz_in_z'][i] - self.data['xyz_out_z'][i])
            y_projected = self.data['xyz_in_y'][i] + np.tan(self.data['theta_in_y'][i])*(self.data['xyz_in_z'][i] - self.data['xyz_out_z'][i])

            deltatheta_x_comp=((self.data['xyz_out_x'][i] - x_projected)*(np.sqrt(1 + np.tan(self.data['theta_in_x'][i])**2 + np.tan(self.data['theta_in_y'][i])**2) / np.sqrt(1 + np.tan(self.data['theta_in_x'][i])**2))*np.cos((self.data['theta_out_x'][i] + self.data['theta_in_x'][i])/2))            
            deltatheta_y_comp=((self.data['xyz_out_y'][i] - y_projected)*(np.sqrt(1 + np.tan(self.data['theta_in_x'][i])**2 + np.tan(self.data['theta_in_y'][i])**2) / np.sqrt(1 + np.tan(self.data['theta_in_y'][i])**2))*np.cos((self.data['theta_out_y'][i] + self.data['theta_in_y'][i])/2))
                                 
            Dx[indx_count,0]=deltathetax
            Dx[indx_count,1]=deltatheta_x_comp
            
            Dy[indx_count,0]=deltathetay
            Dy[indx_count,1]=deltatheta_y_comp
            indx_count=indx_count+1
            
        
        return Dx, Dy
    

    def em_reconstruction(self):
        
        '''
        Function performs the expectation-maximization (EM) algorithm for reconstruction of images 
        in a muon scattering tomography setting.

        Outputs:
                rad_len: A tensor with shape (n_iter, N, N, N1). 
                          It represents the final reconstructed image for every iteration.
                scatter_density: A tensor with shape (n_iter, N, N, N1). 
                          It represents the scatter density of the object being reconstructed.
        '''
        n_events=len(self.triggered_voxels)
        N,N1=self.voi.n_vox_xyz[0],self.voi.n_vox_xyz[2]
        scatter_density = torch.zeros(self.em_iter,N,N,N1)
        scatter_density[0,:,:,:]=self._lambda_
        p_0 = 5e9

        for itr in range(self.em_iter-1):


            sigma_D=torch.zeros(n_events,2,2)
            w_h = torch.zeros(n_events,N,N,N1,2,2)
            w_h_sum=torch.zeros(n_events,2,2)
            Sx = torch.zeros(n_events,N,N,N1)
            Sy = torch.zeros(n_events,N,N,N1)
            S = torch.zeros(n_events,N,N,N1)
            result= torch.zeros(N,N,N1)
            lambda_itr = scatter_density[itr]


            for i in range(n_events):
                
                
                pr_i=self.pr[self.indices[i]]
                tDx_i = torch.transpose(self.Dx[i], 0, -1)                
                tDy_i = torch.transpose(self.Dy[i], 0, -1)
                w_h[i,:,:,:,:,:] = self.W[i,:,:,:,:,:] * lambda_itr[:,:,:,None,None]
                w_h_sum[i,:,:] = torch.sum(w_h[i,:,:,:,:,:], (0,1,2))
                epsilon = 1e-6
                sigma_D[i,:,:] = (pr_i**2)*w_h_sum[i,:,:] + epsilon*torch.eye(2)
                sigma_D_inv=torch.linalg.inv(sigma_D[i,:,:])
                mtr_y_1=torch.matmul(tDy_i,sigma_D_inv)
                mtr_x_1=torch.matmul(tDx_i,sigma_D_inv)

                xx, yy, zz = torch.meshgrid(torch.arange(N), torch.arange(N), torch.arange(N1))
                mask = self.Hit[i].bool()
                xx = xx[mask]
                yy = yy[mask]
                zz = zz[mask]

                lambda_j = lambda_itr[xx,yy,zz]
                mtr_x_2=torch.matmul(mtr_x_1,self.W[i,xx,yy,zz,:,:]) 
                mtr_x_3=torch.matmul(mtr_x_2,sigma_D_inv)
                mtr_x_4=torch.matmul(mtr_x_3,self.Dx[i]) 
                mtr_y_2=torch.matmul(mtr_y_1,self.W[i,xx,yy,zz,:,:])
                mtr_y_3=torch.matmul(mtr_y_2,sigma_D_inv)
                mtr_y_4=torch.matmul(mtr_y_3,self.Dy[i]) 
                mtr_5 = torch.einsum('ijk,ikl->ijl', sigma_D_inv.unsqueeze(0).expand(len(xx), 2, 2), self.W[i, xx, yy, zz, :, :]).diagonal(dim1=1, dim2=2).sum(dim=1)

                mask = self.Hit[i].bool()
                Sx[i,mask] = (2*lambda_j) + ((mtr_x_4 - mtr_5))*((pr_i**2)*(lambda_j**2))
                Sy[i,mask] = (2*lambda_j) + ((mtr_y_4 - mtr_5))*((pr_i**2)*(lambda_j**2))
                S[i] = (Sx[i]+Sy[i])/2



            scatter_density[itr+1,:,:,:] = torch.sum(S, axis=0, out=scatter_density[itr+1,:,:,:])/(2*self.M)

        rad_len = ((15e6/p_0)**2) /scatter_density  

        return rad_len,scatter_density
    
    


