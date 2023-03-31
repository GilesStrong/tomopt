import warnings
warnings.filterwarnings("ignore")
import math
import torch
from torch import tensor
import numpy as np
import pandas
import math
from fastprogress import progress_bar
from operator import itemgetter
from volume_interest import *
class Tracking:
    
    def __init__(self, voi: VolumeOfInterest, data:pandas.DataFrame, poca: bool=False):
        
        self.voi=voi
        self.voxel_edges = voi.voxel_edges
        self.Nvox_Z=voi.n_vox_xyz[2]
        self.data=data
        
        if poca:
            self.points_in_POCA, self.points_out_POCA = self.compute_discrete_tracks(poca=poca)
            
            self.triggered_voxels_in, self.indices_in, self.ev_hit_vox_count_in = self.find_triggered_voxels(poca=poca)
            self.triggered_voxels_out, self.indices_out, self.ev_hit_vox_count_out = self.find_triggered_voxels(_in_=False,poca=poca)
            
            self.path_length_in = np.sqrt((self.data['xyz_in_x'] - self.data['location_x'])**2 + (self.data['xyz_in_y'] - self.data['location_y'])**2 + (self.data['xyz_in_z'] - self.data['location_z'])**2)
            self.path_length_out = np.sqrt((self.data['location_x'] - self.data['xyz_out_x'])**2 + (self.data['location_y'] - self.data['xyz_out_y'])**2 + (self.data['location_z'] - self.data['xyz_out_z'])**2)
            
            self.cos_theta_x_in = (self.data['xyz_in_x'] - self.data['location_x']) / self.path_length_in
            self.cos_theta_y_in = (self.data['xyz_in_y'] - self.data['location_y']) / self.path_length_in
            self.cos_theta_z_in = (self.data['xyz_in_z'] - self.data['location_z']) / self.path_length_in
            
            self.cos_theta_x_out = (self.data['location_x'] - self.data['xyz_out_x']) / self.path_length_out
            self.cos_theta_y_out = (self.data['location_y'] - self.data['xyz_out_y']) / self.path_length_out
            self.cos_theta_z_out = (self.data['location_z'] - self.data['xyz_out_z']) / self.path_length_out
            
            self.alpha_x_l_in, self.alpha_x_r_in, self.alpha_y_l_in, self.alpha_y_r_in, self.alpha_z_l_in, self.alpha_z_r_in = self.compute_alpha_vals(_in_=True,poca=poca)
            self.alpha_x_l_out, self.alpha_x_r_out, self.alpha_y_l_out, self.alpha_y_r_out, self.alpha_z_l_out, self.alpha_z_r_out = self.compute_alpha_vals(poca=poca)
            
            self.intersection_coordinates_in = self.compute_intersection_coords(poca=poca)
            self.intersection_coordinates_out = self.compute_intersection_coords(out=True,poca=poca)
            
            self.triggered_voxels, self.intersection_coordinates, self.indices= self._merge_()
            
        else:
            self.tracks = self.compute_discrete_tracks()
            self.triggered_voxels, self.indices, self.ev_hit_vox_count = self.find_triggered_voxels(poca=poca)
            self.path_length = np.sqrt((self.data['xyz_in_x'] - self.data['xyz_out_x'])**2 + (self.data['xyz_in_y'] - self.data['xyz_out_y'])**2 + (self.data['xyz_in_z'] - self.data['xyz_out_z'])**2)
            self.cos_theta_x = (self.data['xyz_in_x'] - self.data['xyz_out_x']) / self.path_length 
            self.cos_theta_y = (self.data['xyz_in_y'] - self.data['xyz_out_y']) / self.path_length
            self.cos_theta_z = (self.data['xyz_in_z'] - self.data['xyz_out_z']) / self.path_length
            self.alpha_x_l, self.alpha_x_r, self.alpha_y_l, self.alpha_y_r, self.alpha_z_l, self.alpha_z_r = self.compute_alpha_vals()
            self.intersection_coordinates = self.compute_intersection_coords()
            
        self.W, self.M, self.Path_Length, self.T2, self.Hit = self.calculate_path_length()
        self.Dx, self.Dy = self.compute_observed_data()
    
    def compute_discrete_tracks(self, poca=False):
        
        '''
        Function computes x,y,z position at Zmax and Zmin of each voxel layer (for incoming and outgoing tracks)
        
        Outputs: 
                two tensors/lists (in case of PoCA) having x,y,z position at Zmax and Zmin of each voxel 
                layer (for incoming and outgoing tracks), size = [coordinate,Nlayer_along_Z + 1, Nevents] 
                
        '''


        if poca:
            k = np.linspace(0, 1, num=100)
            points_in_POCA = np.stack([    self.data['xyz_in_x'][:, np.newaxis] + (self.data['location_x'] - self.data['xyz_in_x'])[:, np.newaxis] * k,
                self.data['xyz_in_y'][:, np.newaxis] + (self.data['location_y'] - self.data['xyz_in_y'])[:, np.newaxis] * k,
                self.data['xyz_in_z'][:, np.newaxis] + (self.data['location_z'] - self.data['xyz_in_z'])[:, np.newaxis] * k
            ], axis=-1)

            points_out_POCA = np.stack([    self.data['location_x'][:, np.newaxis] + (self.data['xyz_out_x'] - self.data['location_x'])[:, np.newaxis] * k,
                self.data['location_y'][:, np.newaxis] + (self.data['xyz_out_y'] - self.data['location_y'])[:, np.newaxis] * k,
                self.data['location_z'][:, np.newaxis] + (self.data['xyz_out_z'] - self.data['location_z'])[:, np.newaxis] * k
            ], axis=-1)

            return points_in_POCA, points_out_POCA
        
        else:
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
    
    def find_triggered_voxels(self,_in_=True,poca=False):
        
        '''
        Function identifies the list of voxels that were triggered by the particle, along with 
        their respective indices in the input data file.
        
        Returns:

            triggered_voxels: A list of lists of tuples, where each tuple represents a voxel. 
                              The list contains indices of all the voxels that were triggered 
                              by the particle.
                              
            vox_loc: A list of the indices of events where the triggered voxels were detected.
        '''
        
        triggered_voxels = []
        vox_loc = []
        ev_hit_vox_count=torch.zeros(len(self.data))
        
        if not poca:
            zmax,zmin=self.voi.xyz_max[2], self.voi.xyz_min[2]
        
        print('Finding triggered voxels.')
        
        for ev in progress_bar(range(len(self.data))):
            edge = self.voi.voxel_edges
            
            if _in_ and poca:
                
                x1,y1=self.data['xyz_in_x'][ev],self.data['xyz_in_y'][ev]
                x2,y2=self.data['location_x'][ev],self.data['location_y'][ev]
                IN = self.points_in_POCA[ev][0]
                OUT = self.points_in_POCA[ev][-1]
                zmax,zmin=self.data['xyz_in_z'][ev], self.data['location_z'][ev]
            
            elif poca:
                
                x1,y1=self.data['location_x'][ev],self.data['location_y'][ev]
                x2,y2=self.data['xyz_out_x'][ev],self.data['xyz_out_y'][ev]
                IN = self.points_out_POCA[ev][0]
                OUT = self.points_out_POCA[ev][-1]
                zmax,zmin=self.data['location_z'][ev], self.data['xyz_out_z'][ev]
            
            else:
                
                x1,y1,z1=self.data['xyz_in_x'][ev],self.data['xyz_in_y'][ev],self.data['xyz_in_z'][ev]
                x2,y2,z2=self.data['xyz_out_x'][ev],self.data['xyz_out_y'][ev],self.data['xyz_out_z'][ev]
                IN = self.tracks[:, 0, ev]
                OUT = self.tracks[:, -1, ev]
                
                
            theta_x = math.atan((OUT[0] - IN[0]) / (OUT[2] - IN[2]))
            theta_y = math.atan((OUT[1] - IN[1]) / (OUT[2] - IN[2]))

            z = torch.linspace(zmax, zmin, 100)
            x = IN[0] + (z - IN[2]) * math.tan(theta_x)
            y = IN[1] + (z - IN[2]) * math.tan(theta_y)
            
            xyz = torch.zeros(100, 3)
            xyz[:, 0] = x
            xyz[:, 1] = y
            xyz[:, 2] = z

            edge = edge[:, :, :, :, None, :]
            edge = edge.expand(self.voi.n_vox_xyz[0], self.voi.n_vox_xyz[1], self.voi.n_vox_xyz[2], 2, len(xyz), 3)
            mask = ((edge[:, :, :, 0, :, 0] < xyz[:, 0]) & (edge[:, :, :, 1, :, 0] > xyz[:, 0]) &
                    (edge[:, :, :, 0, :, 1] < xyz[:, 1]) & (edge[:, :, :, 1, :, 1] > xyz[:, 1]) &
                    (edge[:, :, :, 0, :, 2] < xyz[:, 2]) & (edge[:, :, :, 1, :, 2] > xyz[:, 2]))

            indices = torch.nonzero(mask, as_tuple=False)
            indices = indices[:, :-1].unique(dim=0).tolist()

            sign_x, sign_y = (-1, -1) if x1 > x2 and y1 > y2 else \
                             (-1,  1) if x1 > x2 and y1 < y2 else \
                             ( 1, -1) if x1 < x2 and y1 > y2 else \
                             ( 1,  1)
            key = lambda k: (k[2], sign_x * k[0], sign_y * k[1])

            indices = sorted(indices, key=key)
            
            if len(indices)>0:
                pair = (indices[0], indices[-1])
                if pair[0][1:] == pair[1][1:] and ((pair[0][0] > pair[1][0] and indices[0][0] < indices[-1][0]) or (pair[0][0] < pair[1][0] and indices[0][0] > indices[-1][0])):
                    indices.reverse()

            triggered_voxels.append(indices)
            ev_hit_vox_count[ev]=len(indices)
            vox_loc.append(ev)
                
        return triggered_voxels, vox_loc, ev_hit_vox_count
    
    def compute_alpha_vals(self, _in_=False, poca=False):
        
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
    
        alpha_x_l, alpha_x_r, alpha_y_l, alpha_y_r, alpha_z_l, alpha_z_r = [], [], [], [], [], []
        
        if _in_ and poca:
            triggered_voxels=self.triggered_voxels_in
            indices=self.indices_in
            cos_theta_x, cos_theta_y, cos_theta_z = self.cos_theta_x_in,self.cos_theta_y_in,self.cos_theta_z_in
        
        elif poca:
            triggered_voxels=self.triggered_voxels_out
            indices=self.indices_out
            cos_theta_x, cos_theta_y, cos_theta_z = self.cos_theta_x_out,self.cos_theta_y_out,self.cos_theta_z_out
        
        else:
            triggered_voxels=self.triggered_voxels
            indices=self.indices
            cos_theta_x, cos_theta_y, cos_theta_z = self.cos_theta_x,self.cos_theta_y,self.cos_theta_z
    
        for i in range(0, len(triggered_voxels)):
            alpha_x_l_i, alpha_x_r_i, alpha_y_l_i, alpha_y_r_i, alpha_z_l_i, alpha_z_r_i=[], [], [], [], [], []
            
            if len(triggered_voxels[i])>0:
                if _in_ and poca:
                    x_in, y_in, z_in = self.data.iloc[indices[i]]['xyz_in_x'], self.data.iloc[indices[i]]['xyz_in_y'], self.data.iloc[indices[i]]['xyz_in_z']
                
                elif poca:
                    x_in, y_in, z_in = self.data.iloc[indices[i]]['location_x'], self.data.iloc[indices[i]]['location_y'], self.data.iloc[indices[i]]['location_z']
                
                else:
                    x_in, y_in, z_in = self.data.iloc[self.indices[i]]['xyz_in_x'], self.data.iloc[self.indices[i]]['xyz_in_y'], self.data.iloc[self.indices[i]]['xyz_in_z']
            
                for j in range(0, len(triggered_voxels[i])):
                    x, y, z= triggered_voxels[i][j][0],triggered_voxels[i][j][1],triggered_voxels[i][j][2]

                    alpha_l= (self.voi.voxel_edges[x, y, z, 0, 0] - x_in)/cos_theta_x.iloc[indices[i]]
                    alpha_r= (self.voi.voxel_edges[x, y, z, 1, 0] - x_in)/cos_theta_x.iloc[indices[i]]

                    alpha_x_l_i.append(alpha_l)
                    alpha_x_r_i.append(alpha_r)

                    alpha_l= (self.voi.voxel_edges[x, y, z, 0, 1] - y_in)/cos_theta_y.iloc[indices[i]]
                    alpha_r= (self.voi.voxel_edges[x, y, z, 1, 1] - y_in)/cos_theta_y.iloc[indices[i]]

                    alpha_y_l_i.append(alpha_l)
                    alpha_y_r_i.append(alpha_r)

                    alpha_l= (self.voi.voxel_edges[x, y, z, 0, 2] - z_in)/cos_theta_z.iloc[indices[i]]
                    alpha_r= (self.voi.voxel_edges[x, y, z, 1, 2] - z_in)/cos_theta_z.iloc[indices[i]]

                    alpha_z_l_i.append(alpha_l)
                    alpha_z_r_i.append(alpha_r)
                
            
            alpha_x_l.append(alpha_x_l_i)
            alpha_x_r.append(alpha_x_r_i)
            
            alpha_y_l.append(alpha_y_l_i)
            alpha_y_r.append(alpha_y_r_i)
                
            alpha_z_l.append(alpha_z_r_i)
            alpha_z_r.append(alpha_z_l_i)

        return alpha_x_l, alpha_x_r, alpha_y_l, alpha_y_r, alpha_z_l, alpha_z_r  
    
    def compute_intersection_coords(self,out=False,poca=False):
        '''
        Function calculates the intersection points between voxels and muon trajectories.
        
        Outputs:
                coordinates: list of coordinates that describe the trajectory of each particle 
                             with shape (num_events,).
        '''
        
        def max_val(lst, index):
            return max(enumerate(sub[index] for sub in lst), key=itemgetter(1))
        
        def trajectory(alpha, index, out=False):
            location = ('xyz_in_x', 'xyz_in_y', 'xyz_in_z') if not out else ('location_x', 'location_y', 'location_z') 
            data = self.data.iloc[index]
            
            if poca:
                
                x = data[location[0]] + self.cos_theta_x_in.iloc[index] * alpha.numpy() if not out else data[location[0]] + self.cos_theta_x_out.iloc[index] * alpha.numpy()
                y = data[location[1]] + self.cos_theta_y_in.iloc[index] * alpha.numpy() if not out else data[location[1]] + self.cos_theta_y_out.iloc[index] * alpha.numpy()
                z = data[location[2]] + self.cos_theta_z_in.iloc[index] * alpha.numpy() if not out else data[location[2]] + self.cos_theta_z_out.iloc[index] * alpha.numpy()
            
            else:
                
                x = data[location[0]] + self.cos_theta_x.iloc[index] * alpha.numpy()
                y = data[location[1]] + self.cos_theta_y.iloc[index] * alpha.numpy()
                z = data[location[2]] + self.cos_theta_z.iloc[index] * alpha.numpy()
                
            x = 0 if x < 0 else x
            
            y = 0 if y < 0 else y
            
            return x, y, z
        
        coordinates = []
        
        if poca:
            triggered_voxels = self.triggered_voxels_out if out else self.triggered_voxels_in
            alpha_x_l, alpha_x_r, alpha_y_l, alpha_y_r, alpha_z_l = (
                self.alpha_x_l_out if out else self.alpha_x_l_in,
                self.alpha_x_r_out if out else self.alpha_x_r_in,
                self.alpha_y_l_out if out else self.alpha_y_l_in,
                self.alpha_y_r_out if out else self.alpha_y_r_in,
                self.alpha_z_l_out if out else self.alpha_z_l_in,
            )
        
        else:
            triggered_voxels = self.triggered_voxels
            alpha_x_l, alpha_x_r, alpha_y_l, alpha_y_r, alpha_z_l = (
                self.alpha_x_l,
                self.alpha_x_r,
                self.alpha_y_l,
                self.alpha_y_r,
                self.alpha_z_l,
            )

        xmax,ymax,zmax=self.voi.xyz_max[0],self.voi.xyz_max[1],self.voi.xyz_max[2]
        xmin,ymin,zmin=self.voi.xyz_min[0],self.voi.xyz_min[1],self.voi.xyz_min[2]
        print('Computing voxels and muon trajectroies intersection coordinates.')
        for ev in progress_bar(range(0, len(triggered_voxels))):
            if poca:
                ev_indx = self.indices_out[ev] if out else self.indices_in[ev]
                x_in, x_out = (self.data.iloc[ev_indx][['location_x', 'xyz_out_x']] if out 
                            else self.data.iloc[ev_indx][['xyz_in_x', 'location_x']])
                y_in, y_out = (self.data.iloc[ev_indx][['location_y', 'xyz_out_y']] if out 
                            else self.data.iloc[ev_indx][['xyz_in_y', 'location_y']])
                z_in, z_out = (self.data.iloc[ev_indx][['location_z', 'xyz_out_z']] if out 
                            else self.data.iloc[ev_indx][['xyz_in_z', 'location_z']])
            else:
                ev_indx = self.indices[ev]
                x_in, x_out = (self.data.iloc[ev_indx][['xyz_in_x', 'xyz_out_x']]) 
                y_in, y_out = (self.data.iloc[ev_indx][['xyz_in_y', 'xyz_out_y']])
                z_in, z_out = (self.data.iloc[ev_indx][['xyz_in_z', 'xyz_out_z']])

            ev_coord = [(0, 0, 0)] if len(triggered_voxels[ev]) > 0 else []

            for hit_vox in range(len(triggered_voxels[ev]) - 1):
                curr_vox_x, curr_vox_y, curr_vox_z = triggered_voxels[ev][hit_vox]
                next_vox_x, next_vox_y, next_vox_z = triggered_voxels[ev][hit_vox + 1]

                if curr_vox_z != next_vox_z:
                    coord = trajectory(alpha_z_l[ev][hit_vox + 1], ev_indx, out)
                elif curr_vox_y > next_vox_y:
                    coord = trajectory(alpha_y_l[ev][hit_vox], ev_indx, out)
                elif curr_vox_y < next_vox_y:
                    coord = trajectory(alpha_y_r[ev][hit_vox], ev_indx, out)
                elif curr_vox_x > next_vox_x:
                    coord = trajectory(alpha_x_l[ev][hit_vox], ev_indx, out)
                else:
                    coord = trajectory(alpha_x_r[ev][hit_vox], ev_indx, out)
                    
                if coord not in ev_coord and (zmin <= coord[2] <= zmax):
                    ev_coord.append(coord)

            ev_coord.append((0, 0, 0))

            if len(triggered_voxels[ev]) > 0:
                if xmin <= x_in <= xmax and ymin <= y_in <= ymax:
                    ev_coord[0] = (x_in, y_in, z_in)

                if xmin <= x_out <= xmax and ymin <= y_out <= ymax:
                    ev_coord[-1] = (x_out, y_out, z_out)

                exit_vox_indx = max_val(triggered_voxels[ev], 2)[0]
                exit_vox_x, exit_vox_y, _ = self.voi.voxel_centers[        triggered_voxels[ev][exit_vox_indx][0],
                    triggered_voxels[ev][exit_vox_indx][1],
                    triggered_voxels[ev][exit_vox_indx][2]
                ]

                if ev_coord[-1][0] == 0:
                    if x_out > xmax and ymin <= y_out <= ymax and len(alpha_x_r[ev]) > 0:
                        coord = trajectory(alpha_x_r[ev][-1], ev_indx, out)
                        if zmin <= coord[2] <= zmax:
                            ev_coord[-1] = coord

                    elif x_out < xmin and ymin <= y_out <= ymax and len(alpha_x_l[ev]) > 0:
                        coord = trajectory(alpha_x_l[ev][-1], ev_indx, out)
                        if zmin <= coord[2] <= zmax:
                            ev_coord[-1] = coord

                    elif y_out > ymax and xmin <= x_out <= xmax and len(alpha_y_r[ev]) > 0:
                        coord = trajectory(alpha_y_r[ev][-1], ev_indx, out)
                        if zmin <= coord[2] <= zmax:
                            ev_coord[-1] = coord

                    elif y_out < ymin and xmin <= x_out <= xmax and len(alpha_y_l[ev]) > 0:
                        coord = trajectory(alpha_y_l[ev][-1], ev_indx, out)
                        if zmin <= coord[2] <= zmax:
                            ev_coord[-1] = coord

                    elif y_out > ymax and x_out > xmax and len(alpha_x_r[ev]) > 0:
                        if abs(x_out - exit_vox_x) > abs(y_out - exit_vox_y):
                            coord = trajectory(alpha_x_r[ev][-1], ev_indx, out)
                        else:
                            coord = trajectory(alpha_y_r[ev][-1], ev_indx, out)
                        if zmin <= coord[2] <= zmax:
                            ev_coord[-1] = coord

                    elif y_out < ymin and x_out > xmax and len(alpha_x_r[ev]) > 0:
                        if abs(x_out - exit_vox_x) > abs(y_out - exit_vox_y):
                            coord = trajectory(alpha_x_r[ev][-1], ev_indx, out)
                        else:
                            coord = trajectory(alpha_y_l[ev][-1], ev_indx, out)
                        if zmin <= coord[2] <= zmax:
                            ev_coord[-1] = coord

                centr_vox_x, centr_vox_y = self.voi.voxel_centers[int(triggered_voxels[ev][0][0]), int(triggered_voxels[ev][0][1]), int(triggered_voxels[ev][0][2]), 0:2]

                if ev_coord[0][0] == 0:
                    if xmax < x_in and y_in <= ymax:
                        coord = trajectory(alpha_x_r[ev][0], ev_indx)
                    elif x_in < xmin and y_in <= ymax:
                        coord = trajectory(alpha_x_l[ev][0], ev_indx, out)
                    elif y_in > ymax and xmin <= x_in <= xmax:
                        coord = trajectory(alpha_y_r[ev][0], ev_indx, out)
                    elif y_in < ymin and xmin <= x_in <= xmax:
                        coord = trajectory(alpha_y_l[ev][0], ev_indx, out)
                    elif y_in > ymax and x_in > xmax:
                        coord = trajectory(alpha_x_r[ev][0], ev_indx, out) if abs(x_in - centr_vox_x) > abs(y_in - centr_vox_y) else trajectory(alpha_y_r[ev][0], ev_indx, out)
                    elif y_in < ymin and x_in > xmax:
                        coord = trajectory(alpha_x_r[ev][0], ev_indx, out) if abs(x_in - centr_vox_x) > abs(y_in - centr_vox_y) else trajectory(alpha_y_l[ev][0], ev_indx, out)
                    elif y_in < ymin and x_in < xmin:
                        coord = trajectory(alpha_x_l[ev][0], ev_indx, out) if abs(x_in - centr_vox_x) > abs(y_in - centr_vox_y) else trajectory(alpha_y_l[ev][0], ev_indx, out)
                    elif y_in > ymax and x_in < xmin:
                        coord = trajectory(alpha_x_l[ev][0], ev_indx, out) if abs(x_in - centr_vox_x) > abs(y_in - centr_vox_y) else trajectory(alpha_y_r[ev][0], ev_indx, out)
                        
                    if zmin <= coord[2] <= zmax:
                        ev_coord[0] = coord     

            if len(ev_coord)>0 and (ev_coord[-1]==(0,0,0) or ev_coord[-1][2]<zmin or ev_coord[-1][2]>zmax):
                del ev_coord[-1]
            if len(ev_coord)>0 and (ev_coord[0]==(0,0,0) or ev_coord[0][2]<zmin or ev_coord[0][2]>zmax):
                del ev_coord[0]
            
            coordinates.append(ev_coord)

        return coordinates
    
    def _merge_(self):
        
        vox_in, vox_out = self.triggered_voxels_in, self.triggered_voxels_out
        coord_in, coord_out = self.intersection_coordinates_in, self.intersection_coordinates_out
        triggered_voxels, indices, coordinates = [], [], [] 
        
        for i in range(len(self.data)):
            x1,y1=self.data['xyz_in_x'][i],self.data['xyz_in_y'][i]
            x2,y2=self.data['xyz_out_x'][i],self.data['xyz_out_y'][i]
            
            if len(coord_in[i])>1:
                voxels=vox_in[i]+vox_out[i]
                coords=coord_in[i]+coord_out[i]
            
                indices.append(i)
                voxels = [tuple(v) for v in voxels] 
                voxels=list(set(voxels))
                
                coords = [tuple(coord) for coord in coords] 
                coords=list(set(coords))
                
                
                sign_x, sign_y = (-1, -1) if x1 > x2 and y1 > y2 else \
                                 (-1,  1) if x1 > x2 and y1 < y2 else \
                                 ( 1, -1) if x1 < x2 and y1 > y2 else \
                                 ( 1,  1)
                
                key = lambda k: (k[2], sign_x * k[0], sign_y * k[1])
                voxels=sorted(voxels,key=key)
                coords=sorted(coords,key=lambda k: (-k[2], sign_x * k[0], sign_y * k[1]))
                triggered_voxels.append(voxels)
                coordinates.append(coords)  

        return triggered_voxels, coordinates, indices
    
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
        
        path_len=compute_path_length(self.intersection_coordinates)
        N,N1=self.voi.n_vox_xyz[0],self.voi.n_vox_xyz[2]
        M = torch.zeros(self.voi.n_vox_xyz[0],self.voi.n_vox_xyz[1],self.voi.n_vox_xyz[2])
        Hit = torch.zeros(len(self.triggered_voxels),self.voi.n_vox_xyz[0],self.voi.n_vox_xyz[1],self.voi.n_vox_xyz[2])
        L = torch.zeros(len(self.triggered_voxels), self.voi.n_vox_xyz[0], self.voi.n_vox_xyz[1], self.voi.n_vox_xyz[2])
        T = torch.zeros(len(self.triggered_voxels), self.voi.n_vox_xyz[0], self.voi.n_vox_xyz[1], self.voi.n_vox_xyz[2])
        W = torch.zeros(len(self.triggered_voxels), N, N, N1, 2, 2)
        print('Computing inputs for the EM steps: (W, L, T and M)')
        for i in progress_bar(range(len(self.triggered_voxels))):
                voxels = self.triggered_voxels[i]
                if i<len(path_len):
                    lrev = list(reversed(path_len[i]))
                    lrev_cum_sum = torch.cumsum(torch.tensor(lrev), dim=0)
                    lrev_cum_sum_rev = list(reversed(lrev_cum_sum))
                    lrev_cum_sum_rev=lrev_cum_sum_rev[1:]
                    lrev_cum_sum_rev.append(path_len[i][-1])
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
                W[i, :, :, :, 1, 1] = (L[i, :, :, :] ** 3) / 3 + (L[i, :, :, :] ** 2) * T[i, :, :, :] + L[i, :, :, :] * (T[i, :, :, :] ** 2)

                Hit[i,:,:,:] = (L[i,:,:,:] != 0)
                M[:,:,:] += Hit[i,:,:,:]

        return W, M, L, T, Hit
    
    def compute_observed_data(self):
        
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
        print('Computing inputs for the EM steps: (Dx and Dx)')
        for i in self.indices:
            x0,x1,y0,y1,z0,z1=self.data['xyz_in_x'][i],self.data['xyz_out_x'][i],self.data['xyz_in_y'][i],self.data['xyz_out_y'][i],self.data['xyz_in_z'][i],self.data['xyz_out_z'][i]
            theta_x0,theta_x1,theta_y0,theta_y1=self.data['theta_in_x'][i],self.data['theta_out_x'][i],self.data['theta_in_y'][i],self.data['theta_out_y'][i]
            
            deltathetax = theta_x1 - theta_x0
            deltathetay = theta_y1 - theta_y0
            
            Lxy=np.sqrt(1 + np.tan(theta_x0)**2 + np.tan(theta_y0)**2)
            
            xp = x0 + np.tan(theta_x0)*abs(z0 - z1)
            yp = y0 + np.tan(theta_y0)*abs(z0 - z1)
            
            deltatheta_x_comp=((x1 - xp)*( Lxy / np.sqrt(1 + np.tan(theta_x0)**2))*np.cos((theta_x1 + theta_x0)/2))
            deltatheta_y_comp=((y1 - yp)*( Lxy / np.sqrt(1 + np.tan(theta_y0)**2))*np.cos((theta_y1 + theta_y0)/2))
            
            Dx[indx_count,0]=deltathetax
            Dy[indx_count,0]=deltathetay
            
            Dx[indx_count,1]=deltatheta_x_comp
            Dy[indx_count,1]=deltatheta_y_comp

            indx_count=indx_count+1
        
        return Dx, Dy