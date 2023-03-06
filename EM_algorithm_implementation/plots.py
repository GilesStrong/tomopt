import warnings
warnings.filterwarnings("ignore")
#Usual suspects
import matplotlib.pyplot as plt
import pandas
import numpy as np 
import torch
from torch import tensor
import seaborn as sns
from tracking import *
from volume_interest import *
from mpl_toolkits.mplot3d import axes3d
from numpy.random import rand
from IPython.display import HTML
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
from matplotlib.pyplot import figure
from matplotlib import pyplot
import matplotlib.cm as cm
import random
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors

from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import vectors

def plot_rad_len_across_iter(em_iter, rad_len, org_rad_len):
    fig, ax = plt.subplots(2, 6, figsize=(18, 8))
    itr = 1
    fig.subplots_adjust(hspace=0.5) # Add vertical space between subplots
    cbar_ax = fig.add_axes([1.05, 0.15, 0.02, 0.7])
    for i in range(2):
        for j in range(6):
            # if i == 0:
                # hmap = sns.heatmap(org_rad_len[:, :, j], cmap='cividis', norm=LogNorm(vmin=torch.min(org_rad_len[:, :, j]), vmax=torch.max(org_rad_len[:, :, j])), ax=ax[i, j], cbar=i == 0, cbar_ax=None if i != 0 else cbar_ax)
            if i == 0:
                hmap = sns.heatmap(rad_len[1, :, :, j], norm=LogNorm(vmin=torch.min(rad_len[1, :, :, j]), vmax=torch.max(rad_len[1, :, :, j])), cmap='cividis', ax=ax[i, j], cbar=i == 0, cbar_ax=None if i != 0 else cbar_ax)
            elif i == 1:
                hmap = sns.heatmap(rad_len[em_iter-1, :, :, j], norm=LogNorm(vmin=torch.min(rad_len[em_iter-1, :, :, j]), vmax=torch.max(rad_len[em_iter-1, :, :, j])), cmap='cividis', ax=ax[i, j], cbar=i == 0, cbar_ax=None if i != 0 else cbar_ax)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].set_xlabel('x')  # modify the x-axis label
            ax[i, j].set_ylabel('y')
            ax[i, j].set_title(f'Layer {j+1}')
            ax[i, j].set_aspect('equal')
            
        if i == 0 or i == 1:
            ax[i, 0].set_title(f"Iteration {itr} - Layer {1}")
            itr += em_iter-1
        
    plt.show()


def plot_diff_pred_org_Lrad(L_rad: np.ndarray, event: int, original_L_rad: np.ndarray, log_scale=False) -> None:
    
    '''
    Function plots 2D heatmaps of radiation length values for each layer in the detector.
    
    Inputs:
            L_rad: 3D tensor with shape (10, 10, 6), representing the radiation length values 
               for each voxel in the detector.
            event: integer representing the event number.
            original_L_rad: original radiation length
            log_scale: flag to indicate if log scale is to be used for color mapping
    '''
    
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), gridspec_kw={"hspace": 0.4})

    cbar_ax = fig.add_axes([1, 0.15, 0.02, 0.7])
    for i, ax in enumerate(axs.flat):
        z = L_rad[:10, :10, i].ravel() - original_L_rad[:10, :10, i].ravel()
        if log_scale:
            hmap = sns.heatmap(z.reshape((10, 10)), norm=LogNorm(vmin=L_rad.min(), vmax=L_rad.max()), cmap='cividis', ax=ax, cbar=i == 0, cbar_ax=None if i != 0 else cbar_ax)
        else:
            hmap = sns.heatmap(z.reshape((10, 10)), cmap='cividis', ax=ax, cbar=i == 0, cbar_ax=None if i != 0 else cbar_ax, vmin=L_rad.min(), vmax=L_rad.max())
        ax.set_xlabel('x')  # modify the x-axis label
        ax.set_ylabel('y')
        ax.set_title(f'Layer {i+1}')

    if hmap.collections[0].colorbar is not None:
        cbar = hmap.collections[0].colorbar
        cbar.ax.set_ylabel('Intensity')
    
    plt.suptitle(f'Heatmaps of Difference between predicted and original radiation length for Events {event+1}')
    plt.subplots_adjust(top=0.9, bottom=0.1)
    plt.show()
    
    diff = L_rad.sum(axis=(0, 1)) - original_L_rad.sum(axis=(0, 1))
    plt.figure()
    plt.plot(range(1, len(diff)+1), diff)
    plt.xlabel('Layer')
    plt.ylabel('Difference')
    plt.title('Difference between predicted and original radiation lengths')
    
    plt.subplots_adjust(top=0.9, bottom=0.1)
    plt.show()

#write it as pickle file

matr_lrad_dict={
    0.3528:'Be',
    0.0056:'Pb',
    0.0031:'U',
    0.089:'Al',
    0.0176:'Fe',  
    303.9:'Air',
    0.0144:'Cu'
         }



def group_indices_by_value(tensor):
    unique_values = torch.unique(tensor)
    indices_dict = {}

    for value in unique_values:
        
        indices = torch.nonzero(tensor == value, as_tuple=False)
        indices_list = indices.tolist()
        val=round(value.item(),4)
        indices_dict[val] = indices_list

    return indices_dict

def radiation_distribution(L_rad, org_L_rad, em_itr):
    stats = importr("stats")
    org_lrad_indices=group_indices_by_value(org_L_rad)
    pred_l_rad_mat={}
    for matr in org_lrad_indices:
        coordinates=org_lrad_indices.get(matr)
        pred_l_rad_mat[matr]=[np.repeat(0,len(coordinates))]*em_itr
        for itr in range(0,em_itr):
            out=[]
            for coord_indx in range(0,len(coordinates)):
                x,y,z=coordinates[coord_indx]
                out.append(float(L_rad[itr,x,y,z]))
            pred_l_rad_mat[matr][itr]=out
    return pred_l_rad_mat




def compute_density(lrad_distrb,itermax,log_scale=False):
    stats = importr("stats")
    axes_limits={}
    itr_matr_lrad={}
    itr_matr_limits={}
    x_lim=[]
    y_lim=[]
    for matr in lrad_distrb:
        matr_lrad_info=[]
        matr_lrad_lim=[]
        
        for i in range(itermax):
            if log_scale:
                column = vectors.FloatVector(np.log(lrad_distrb[matr][i]))
            else:
                column = vectors.FloatVector(lrad_distrb[matr][i])
            output = stats.density(column, adjust=1)
            x = np.array(output[0])
            y = np.array(output[1])
            x_min=min(x)
            x_max=max(x)
            
            y_min=min(y)
            y_max=max(y)
            
            matr_lrad_info.append((x,y))
            matr_lrad_lim.append([(x_min,x_max),(y_min,y_max)])
            
        itr_matr_lrad[matr]=matr_lrad_info
        itr_matr_limits[matr]=matr_lrad_lim
    
    
    for i in range(itermax):
        y_min=[]
        x_min=[]
        y_max=[]
        x_max=[]
        
        for matr in itr_matr_lrad:
            y_min.append(itr_matr_limits[matr][i][1][0])
            x_min.append(itr_matr_limits[matr][i][0][0])
            x_max.append(itr_matr_limits[matr][i][0][1])
            y_max.append(itr_matr_limits[matr][i][1][1])
        
        x_lim.append((min(x_min),max(x_max)))
        y_lim.append((min(y_min),max(y_max)))
    
        
        
    return itr_matr_lrad, x_lim, y_lim
 

def generate_distinct_hex_colors(N):
    """
    Generate N distinct hex color strings.

    Parameters:
    N (int): the number of hex colors to generate

    Returns:
    list: a list of N distinct hex color strings
    """
    hex_colors = set()
    while len(hex_colors) < N:
        # generate a random hex color string
        hex_color = '#{0:06x}'.format(random.randint(0, 0xFFFFFF))
        # add it to the set if it's not already in there
        hex_colors.add(hex_color)
    return list(hex_colors)


def plot_density_for_LRad(matr_lrad, x_lim, y_lim,itr,rand_colors,log_scale):
    
    x=[]
    y=[]
    
    figure(figsize=(8, 6), dpi=125)
    i=0
    for matr in matr_lrad:
        plt.plot(matr_lrad[matr][itr][0], matr_lrad[matr][itr][1], color=rand_colors[i], label=matr_lrad_dict[matr],alpha=0.35)
        plt.fill_between(matr_lrad[matr][itr][0], matr_lrad[matr][itr][1], 0,color=rand_colors[i],alpha=0.5)
        i=i+1
    plt.legend(loc="upper left")
    plt.title('Iteration {}'.format(itr))
    plt.ylabel('Density')
    if not log_scale:
        plt.xlabel('Radiation Length (X0)')
    else:
        plt.xlabel('Log( Radiation Length (X0) )')
    plt.xlim(x_lim[itr][0], x_lim[itr][1])
    plt.ylim(y_lim[itr][0], y_lim[itr][1]+1)
    plt.show()




def plot_lrad_case_study(material_1,material_2,xlim,ylim,itr,material_1_label,material_2_label):
    
    #plt.subplot(211)
    x_beryl = material_1[0]
    y_beryl = material_1[1]

    x_lead = material_2[0]
    y_lead = material_2[1]

    # plot the graph
    figure(figsize=(8, 6), dpi=125)
    plt.plot(x_beryl, y_beryl, color="pink", label=material_1_label,alpha=0.5)
    plt.fill_between(x_beryl, y_beryl, 0,color="pink")
    plt.plot(x_lead, y_lead, color="purple", label=material_2_label,alpha=0.2)
    plt.fill_between(x_lead, y_lead, 0,color="purple", alpha=0.4)
    plt.legend(loc="upper left")
    plt.title('Iteration {}'.format(itr))
    plt.ylabel('Density')
    plt.xlabel('Radiation Length (X0)')
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1]+1)
    plt.show()
       


def plot_muon_hits_count(hits):
    fig, ax = plt.subplots(3, 2, figsize=(18, 18))
    fig.suptitle('Counts of Muon Hits per Layer', fontsize=20)
    figure(figsize=(8, 6), dpi=80)

    z0=hits[:,:,0]
    z1=hits[:,:,1]
    z2=hits[:,:,2]
    z3=hits[:,:,3]
    z4=hits[:,:,4]
    z5=hits[:,:,5]

    ax[0, 0].imshow(z0, interpolation ='None',cmap='YlGn',  aspect = 'auto')
    ax[0, 1].imshow(z1, interpolation ='None',cmap='YlGn',  aspect = 'auto')
    ax[1, 0].imshow(z2, interpolation ='None',cmap='YlGn',  aspect = 'auto')
    ax[1, 1].imshow(z3, interpolation ='None',cmap='YlGn',  aspect = 'auto')
    ax[2, 0].imshow(z4, interpolation ='None',cmap='YlGn',  aspect = 'auto')
    ax[2, 1].imshow(z5, interpolation ='None',cmap='YlGn',  aspect = 'auto')
    
    

    for (j,i),label in np.ndenumerate(z0):
        ax[0,0].text(i,j,label,ha='center',va='center',fontsize=12)
        ax[0,0].set_title('Layer 1',fontsize=16)
    for (j,i),label in np.ndenumerate(z1):
        ax[0,1].text(i,j,label,ha='center',va='center',fontsize=12)
        ax[0,1].set_title('Layer 2',fontsize=16)
    for (j,i),label in np.ndenumerate(z2):
        ax[1,0].text(i,j,label,ha='center',va='center',fontsize=12)
        ax[1,0].set_title('Layer 3',fontsize=16)
    for (j,i),label in np.ndenumerate(z3):
        ax[1,1].text(i,j,label,ha='center',va='center',fontsize=12)
        ax[1,1].set_title('Layer 4',fontsize=16)
    for (j,i),label in np.ndenumerate(z4):
        ax[2,0].text(i,j,label,ha='center',va='center',fontsize=12)
        ax[2,0].set_title('Layer 5',fontsize=16)
    for (j,i),label in np.ndenumerate(z5):
        ax[2,1].text(i,j,label,ha='center',va='center',fontsize=12)
        ax[2,1].set_title('Layer 6',fontsize=16)

def plot_muon_track(VOI:VolumeOfInterest, event:int=0, tracking:Tracking=None)->None:
    
    
    # define figures
    fig,ax = plt.subplots(ncols=2,figsize=(15,5))
    fig.suptitle('Volume of interest')
    ax = ax.ravel()

    for i in range(VOI.n_vox_xyz[0]):
        ax[0].axvline(x = VOI.voxel_edges[i,0,0,0,0],ymin=VOI.xyz_min[2],ymax=VOI.xyz_max[2],color='lightgrey')
        ax[0].axvline(x = VOI.voxel_edges[i,0,0,1,0],ymin=VOI.xyz_min[2],ymax=VOI.xyz_max[2],color='lightgrey')
    
    for i in range(VOI.n_vox_xyz[1]):
        ax[1].axvline(x = VOI.voxel_edges[i,0,0,0,0],ymin=VOI.xyz_min[2],ymax=VOI.xyz_max[2],color='lightgrey')
        ax[1].axvline(x = VOI.voxel_edges[i,0,0,1,0],ymin=VOI.xyz_min[2],ymax=VOI.xyz_max[2],color='lightgrey')


    for z in range(VOI.n_vox_xyz[2]):
        ax[0].axhline(y = VOI.voxel_edges[0,0,z,0,2],xmin=VOI.xyz_min[0],xmax=VOI.xyz_max[0],color='lightgrey')
        ax[0].axhline(y = VOI.voxel_edges[0,0,z,1,2],xmin=VOI.xyz_min[0],xmax=VOI.xyz_max[0],color='lightgrey')

        ax[1].axhline(y = VOI.voxel_edges[0,0,z,0,2],xmin=VOI.xyz_min[0],xmax=VOI.xyz_max[0],color='lightgrey')
        ax[1].axhline(y = VOI.voxel_edges[0,0,z,1,2],xmin=VOI.xyz_min[0],xmax=VOI.xyz_max[0],color='lightgrey')

    ax[0].set_title('XZ view')
    ax[0].set_aspect('equal')
    ax[0].set_xlim([0,1])
    ax[0].set_ylim([0,1])
    ax[0].set_xlabel('x [m]')
    ax[0].set_ylabel('z [m]')

    ax[1].set_title('XZ view')
    ax[1].set_aspect('equal')
    ax[1].set_xlim([0,1])
    ax[1].set_ylim([0,1])
    ax[1].set_xlabel('y [m]')
    ax[1].set_ylabel('z [m]')

    if(tracking is not None):

        fig.suptitle('Tracking for event = {}'.format(event))
        # plot tracks
        ax[0].plot(tracking.tracks[0,:,event],tracking.tracks[2,:,event],color='maroon',label='Muon track')
        ax[1].plot(tracking.tracks[1,:,event],tracking.tracks[2,:,event],color='maroon',label='Muon track')

        #plot xy_in and xy_out
        ax[0].scatter(tracking.data['xyz_in_x'][event],tracking.data['xyz_in_z'][event],marker='v',color='darkgreen',label='Entry point')
        ax[0].scatter(tracking.data['xyz_out_x'][event],tracking.data['xyz_out_z'][event],marker='v',color='blue',label='Exit point')

        ax[1].scatter(tracking.data['xyz_in_y'][event],tracking.data['xyz_in_z'][event],marker='v',color='darkgreen',label='Entry point')
        ax[1].scatter(tracking.data['xyz_out_y'][event],tracking.data['xyz_out_z'][event],marker='v',color='blue',label='Exit point')

        ax[0].legend()
        ax[1].legend()

    plt.tight_layout()
    plt.show()


def plot_discrete_track_2d(VOI:VolumeOfInterest, event:int=0, tracking:Tracking=None)->None:
    N_voxels = VOI.n_vox_xyz #[Nvox along x, Nvox along y, Nvox along z]
    xyz_max = VOI.xyz_max
    xyz_min = VOI.xyz_min


    # define figures
    fig,ax = plt.subplots(ncols=2,figsize=(15,5))
    ax = ax.ravel()

    fig.suptitle('Tracking for event = {}'.format(event),y=0.95)
    
    # draw blank voxels
    for i in range(N_voxels[0]):
        ax[0].axvline(x = VOI.voxel_edges[i,0,0,0,0],ymin=xyz_min[2],ymax=xyz_max[2],color='lightgrey')
        ax[0].axvline(x = VOI.voxel_edges[i,0,0,1,0],ymin=xyz_min[2],ymax=xyz_max[2],color='lightgrey')

        ax[1].axvline(x = VOI.voxel_edges[i,0,0,0,0],ymin=xyz_min[2],ymax=xyz_max[2],color='lightgrey')
        ax[1].axvline(x = VOI.voxel_edges[i,0,0,1,0],ymin=xyz_min[2],ymax=xyz_max[2],color='lightgrey')


    for z in range(N_voxels[2]):
        ax[0].axhline(y = VOI.voxel_edges[0,0,z,0,2],xmin=xyz_min[0],xmax=xyz_max[0],color='lightgrey')
        ax[0].axhline(y = VOI.voxel_edges[0,0,z,1,2],xmin=xyz_min[0],xmax=xyz_max[0],color='lightgrey')

        ax[1].axhline(y = VOI.voxel_edges[0,0,z,0,2],xmin=xyz_min[0],xmax=xyz_max[0],color='lightgrey')
        ax[1].axhline(y = VOI.voxel_edges[0,0,z,1,2],xmin=xyz_min[0],xmax=xyz_max[0],color='lightgrey')
        
    x_intersec=[]
    y_intersec=[]
    z_intersec=[]
    for voxel in tracking.intersection_coordinates[event]:
        x_intersec.append(voxel[0])
        y_intersec.append(voxel[1])
        z_intersec.append(voxel[2])
        
        

    ax[0].set_aspect('equal')
    ax[0].set_xlim([0,1])
    ax[0].set_ylim([0,1])
    ax[0].set_xlabel('x [m]')
    ax[0].set_ylabel('z [m]')


    ax[1].set_aspect('equal')
    ax[1].set_xlim([0,1])
    ax[1].set_ylim([0,1])
    ax[1].set_xlabel('y [m]')
    ax[1].set_ylabel('z [m]')

    # plot tracks
    ax[0].plot(tracking.tracks[0,:,tracking.indices[event]],tracking.tracks[2,:,tracking.indices[event]],color='maroon',label='Muon track')
    ax[1].plot(tracking.tracks[1,:,tracking.indices[event]],tracking.tracks[2,:,tracking.indices[event]],color='maroon',label='Muon track')

    #plot xy_in and xy_out
    ax[0].scatter(tracking.data['xyz_in_x'][tracking.indices[event]],tracking.data['xyz_in_z'][tracking.indices[event]],marker='v',color='darkgreen',label='Entry point')
    ax[0].scatter(tracking.data['xyz_out_x'][tracking.indices[event]],tracking.data['xyz_out_z'][tracking.indices[event]],marker='v',color='blue',label='Exit point')
    ax[0].scatter(tracking.tracks[0,1:-1,tracking.indices[event]],tracking.tracks[2,1:-1,tracking.indices[event]],marker='x',label='muon entering layer')
    ax[0].scatter(x_intersec,z_intersec,marker='*',color='orange',label='muon layer intersect')
    
    
    ax[1].scatter(tracking.data['xyz_in_y'][tracking.indices[event]],tracking.data['xyz_in_z'][tracking.indices[event]],marker='v',color='darkgreen',label='Entry point')
    ax[1].scatter(tracking.data['xyz_out_y'][tracking.indices[event]],tracking.data['xyz_out_z'][tracking.indices[event]],marker='v',color='blue',label='Exit point')
    ax[1].scatter(tracking.tracks[1,1:-1,tracking.indices[event]],tracking.tracks[2,1:-1,tracking.indices[event]],marker='x',label='muon entering layer')
    ax[1].scatter(y_intersec,z_intersec,marker='*',color='orange',label='muon layer intersect')
    
    
    # Plot trigger voxels
    i=0
    for vox in tracking.triggered_voxels[event]:
        
        ix = int(vox[0])
        iy = int(vox[1])
        iz = int(vox[2])
        
        if(i==0):
            ax[0].scatter(VOI.voxel_centers[ix,iy,iz,0],VOI.voxel_centers[ix,iy,iz,2],label='triggered voxel',color='yellowgreen')
            ax[1].scatter(VOI.voxel_centers[ix,iy,iz,1],VOI.voxel_centers[ix,iy,iz,2],label='triggered voxel',color='yellowgreen')        
            
        # XZ view
        ax[0].scatter(VOI.voxel_centers[ix,iy,iz,0],VOI.voxel_centers[ix,iy,iz,2],color='yellowgreen')
        
        # YZ view
        ax[1].scatter(VOI.voxel_centers[ix,iy,iz,1],VOI.voxel_centers[ix,iy,iz,2],color='yellowgreen')        
        i+=1
    ax[0].legend()
    ax[1].legend()

    plt.show()

def plot_discrete_track_3d(VOI:VolumeOfInterest, event:int=0, tracking:Tracking=None)->None:

    Nvox_X=VOI.n_vox_xyz[0].numpy()
    Nvox_Y=VOI.n_vox_xyz[1].numpy()
    Nvox_Z=VOI.n_vox_xyz[2].numpy()
    import numpy as np
    x, y, z = np.indices((Nvox_X, Nvox_Y, Nvox_Z))
    voxelarray = (x <= Nvox_X) & (y <= Nvox_Y) & (z <= Nvox_Z)
    triggered_voxels_mask = []
    colors = np.zeros(voxelarray.shape, dtype=object)
    
    for i in tracking.triggered_voxels[event]:
        ix,iy,iz = i[0],i[1],i[2]
        triggered_voxels_mask.append((x==ix) & (y==iy)& ((z)==(5-iz)))
        
    for i in triggered_voxels_mask:
        colors[i] = 'yellowgreen'
    colors = np.where((colors==0),'white','yellowgreen')
    
    fig = plt.figure(figsize=(10,10))
    fig.suptitle('Tracking for event = {}'.format(event),y=0.95)
    ax = fig.add_subplot(projection='3d')
    
    ax.plot3D(tracking.tracks[0,:,tracking.indices[event]].numpy()*10,tracking.tracks[1,:,tracking.indices[event]].numpy()*10,(tracking.tracks[2,:,tracking.indices[event]].numpy()-0.2)*10,color='maroon',label='Muon track')
    
    ax.set_xlim3d(0, Nvox_X)
    ax.set_ylim3d(0, Nvox_Y)
    ax.set_zlim3d(0, Nvox_Z)
    
    ax.set_xlabel('Voxel indice along x')
    ax.set_ylabel('Voxel indice along y')
    ax.set_zlabel('Voxel indice along z')
    
    ax.voxels(voxelarray,alpha=0.0001,edgecolors='whitesmoke')
    
    j=0
    for i in triggered_voxels_mask:
        if(j==0):
            ax.voxels(i, edgecolor='k',color='yellowgreen',alpha=.01,label='hit voxels')
        j+1
        ax.voxels(i, edgecolor='k',color='yellowgreen',alpha=.01)
    
    x_intersec=[]
    y_intersec=[]
    z_intersec=[]
    for voxel in tracking.intersection_coordinates[event]:
        x_intersec.append(voxel[0]*10)
        y_intersec.append(voxel[1]*10)
        z_intersec.append((voxel[2]-0.2)*10)
    
    ax.scatter(x_intersec,y_intersec,z_intersec,marker='*',color='orange',label='muon layer intersect')
    ax.scatter(tracking.data['xyz_in_x'][tracking.indices[event]]*10,tracking.data['xyz_in_y'][tracking.indices[event]]*10,(tracking.data['xyz_in_z'][tracking.indices[event]]-0.2)*10,marker='v',color='darkgreen',label='Entry point')
    ax.scatter(tracking.data['xyz_out_x'][tracking.indices[event]]*10,tracking.data['xyz_out_y'][tracking.indices[event]]*10,(tracking.data['xyz_out_z'][tracking.indices[event]]-0.2)*10,marker='v',color='blue',label='Exit point')
    
    plt.show()
    

    
    import numpy as np


def plot_event_Lrad_heatmaps(L_rad: np.ndarray, event: int, log_scale=False, org_Lrad=False) -> None:
    '''
    Function plots 2D heatmaps of radiation length values for each layer in the detector.
    
    Inputs:
            L_rad: 3D tensor with shape (10, 10, 6), representing the radiation length values 
               for each voxel in the detector.
            event: integer representing the event number.
    '''
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), gridspec_kw={"hspace": 0.4})
    cbar_ax = fig.add_axes([1.05, 0.15, 0.02, 0.7])
    z_min, z_max = torch.min(L_rad), torch.max(L_rad)
    if log_scale:
        norm = colors.LogNorm(vmin=z_min, vmax=z_max)
    else:
        norm = colors.Normalize(vmin=z_min, vmax=z_max)
    for i, ax in enumerate(axs.flat):
        z = L_rad[:10, :10, i].ravel()
        hmap = sns.heatmap(z.reshape((10, 10)), norm=norm, cmap='cividis', ax=ax, cbar=i == 0, cbar_ax=None if i != 0 else cbar_ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Layer {i+1}')
    if hmap.collections[0].colorbar is not None:
        cbar = hmap.collections[0].colorbar
        cbar.ax.set_ylabel('Intensity')
    
    if org_Lrad:
        mat=np.unique(L_rad).tolist()
        mat_names=""
        for m in range(0,len(mat)):
            mat_names=mat_names+matr_lrad_dict[round(float(mat[m]),4)]
            if m+1<len(mat):
                mat_names=mat_names+", "
        plt.suptitle(f'Heatmaps of Radiation Length for {mat_names}')
    else:
        plt.suptitle(f'Heatmaps of Radiation Length for Event {event+1}')
    
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()

    

