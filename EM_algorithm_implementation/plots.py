#Usual suspects
import matplotlib.pyplot as plt
import pandas
import numpy as np 
import torch
from torch import tensor

from tracking import *
from volume_interest import *
from mpl_toolkits.mplot3d import axes3d
from numpy.random import rand
from IPython.display import HTML
from matplotlib import animation


def plot_muon_track(VOI:VolumeOfInterest, event:int=0, tracking:Tracking=None)->None:
    
    
    # define figures
    fig,ax = plt.subplots(ncols=2,figsize=(15,5))
    fig.suptitle('Volume of interest')
    ax = ax.ravel()

    for i in range(VOI.n_vox_xyz[0]):
        ax[0].axvline(x = VOI.voxel_edges[i,0,0,0,0],ymin=VOI.xyz_min[2],ymax=VOI.xyz_max[2])
        ax[0].axvline(x = VOI.voxel_edges[i,0,0,1,0],ymin=VOI.xyz_min[2],ymax=VOI.xyz_max[2])
    
    for i in range(VOI.n_vox_xyz[1]):
        ax[1].axvline(x = VOI.voxel_edges[i,0,0,0,0],ymin=VOI.xyz_min[2],ymax=VOI.xyz_max[2])
        ax[1].axvline(x = VOI.voxel_edges[i,0,0,1,0],ymin=VOI.xyz_min[2],ymax=VOI.xyz_max[2])


    for z in range(VOI.n_vox_xyz[2]):
        ax[0].axhline(y = VOI.voxel_edges[0,0,z,0,2],xmin=VOI.xyz_min[0],xmax=VOI.xyz_max[0])
        ax[0].axhline(y = VOI.voxel_edges[0,0,z,1,2],xmin=VOI.xyz_min[0],xmax=VOI.xyz_max[0])

        ax[1].axhline(y = VOI.voxel_edges[0,0,z,0,2],xmin=VOI.xyz_min[0],xmax=VOI.xyz_max[0])
        ax[1].axhline(y = VOI.voxel_edges[0,0,z,1,2],xmin=VOI.xyz_min[0],xmax=VOI.xyz_max[0]) 

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
        ax[0].plot(tracking.tracks[0,:,event],tracking.tracks[2,:,event],color='red',label='Muon track')
        ax[1].plot(tracking.tracks[1,:,event],tracking.tracks[2,:,event],color='red',label='Muon track')

        #plot xy_in and xy_out
        ax[0].scatter(tracking.data['xyz_in_x'][event],tracking.data['xyz_in_z'][event],color='blue',label='Entry point')
        ax[0].scatter(tracking.data['xyz_out_x'][event],tracking.data['xyz_out_z'][event],color='red',label='Exit point')

        ax[1].scatter(tracking.data['xyz_in_y'][event],tracking.data['xyz_in_z'][event],color='blue',label='Entry point')
        ax[1].scatter(tracking.data['xyz_out_y'][event],tracking.data['xyz_out_z'][event],color='red',label='Exit point')

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

    fig.suptitle('Tracking for event = {}'.format(event))
    
    # draw blank voxels
    for i in range(N_voxels[0]):
        ax[0].axvline(x = VOI.voxel_edges[i,0,0,0,0],ymin=xyz_min[2],ymax=xyz_max[2])
        ax[0].axvline(x = VOI.voxel_edges[i,0,0,1,0],ymin=xyz_min[2],ymax=xyz_max[2])

        ax[1].axvline(x = VOI.voxel_edges[i,0,0,0,0],ymin=xyz_min[2],ymax=xyz_max[2])
        ax[1].axvline(x = VOI.voxel_edges[i,0,0,1,0],ymin=xyz_min[2],ymax=xyz_max[2])


    for z in range(N_voxels[2]):
        ax[0].axhline(y = VOI.voxel_edges[0,0,z,0,2],xmin=xyz_min[0],xmax=xyz_max[0])
        ax[0].axhline(y = VOI.voxel_edges[0,0,z,1,2],xmin=xyz_min[0],xmax=xyz_max[0])

        ax[1].axhline(y = VOI.voxel_edges[0,0,z,0,2],xmin=xyz_min[0],xmax=xyz_max[0])
        ax[1].axhline(y = VOI.voxel_edges[0,0,z,1,2],xmin=xyz_min[0],xmax=xyz_max[0]) 

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
    ax[0].plot(tracking.tracks[0,:,event],tracking.tracks[2,:,event],color='red',label='Muon track')
    ax[1].plot(tracking.tracks[1,:,event],tracking.tracks[2,:,event],color='red',label='Muon track')

    #plot xy_in and xy_out
    ax[0].scatter(tracking.data['xyz_in_x'][event],tracking.data['xyz_in_z'][event],color='blue',label='Entry point')
    ax[0].scatter(tracking.data['xyz_out_x'][event],tracking.data['xyz_out_z'][event],color='red',label='Exit point')
    ax[0].scatter(tracking.tracks[0,1:-1,event],tracking.tracks[2,1:-1,event],marker='x',label='muon entering layer')
    
    ax[1].scatter(tracking.data['xyz_in_y'][event],tracking.data['xyz_in_z'][event],color='blue',label='Entry point')
    ax[1].scatter(tracking.data['xyz_out_y'][event],tracking.data['xyz_out_z'][event],color='red',label='Exit point')
    ax[1].scatter(tracking.tracks[1,1:-1,event],tracking.tracks[2,1:-1,event],marker='x',label='muon entering layer')
    
    
    # Plot trigger voxels
    i=0
    for vox in tracking.triggered_voxels[event]:
        
        ix = int(vox[0].numpy())
        iy = int(vox[1].numpy())
        iz = int(vox[2].numpy())
        
        if(i==0):
            ax[0].scatter(VOI.voxel_centers[ix,iy,iz,0],VOI.voxel_centers[ix,iy,iz,2],label='Triggered voxel',color='green')
            ax[1].scatter(VOI.voxel_centers[ix,iy,iz,1],VOI.voxel_centers[ix,iy,iz,2],label='Triggered voxel',color='green')        
            
        # XZ view
        ax[0].scatter(VOI.voxel_centers[ix,iy,iz,0],VOI.voxel_centers[ix,iy,iz,2],color='green')
        
        # YZ view
        ax[1].scatter(VOI.voxel_centers[ix,iy,iz,1],VOI.voxel_centers[ix,iy,iz,2],color='green')        
        i+=1
    ax[0].legend()
    ax[1].legend()

    plt.show()

def plot_discrete_track_3d(VOI:VolumeOfInterest, event:int=0, tracking:Tracking=None)->None:

    Nvox_X=VOI.n_vox_xyz[0].numpy()
    Nvox_Y=VOI.n_vox_xyz[1].numpy()
    Nvox_Z=VOI.n_vox_xyz[2].numpy()

    x, y, z = np.indices((Nvox_X, Nvox_Y, Nvox_Z))
    voxelarray = (x <= Nvox_X) & (y <= Nvox_Y) & (z <= Nvox_Z)
    triggered_voxels_mask = []
    colors = np.zeros(voxelarray.shape, dtype=object)
    
    for i in tracking.triggered_voxels[event]:
        ix,iy,iz = i[0].item(),i[1].item(),i[2].item()
        triggered_voxels_mask.append((x==ix) & (y==iy)& ((z)==(5-iz)))
        
    for i in triggered_voxels_mask:
        colors[i] = 'red'
    colors = np.where((colors==0),'white','red')
    
    fig = plt.figure(figsize=(10,10))
    fig.suptitle('Tracking for event = {}'.format(event))
    ax = fig.add_subplot(projection='3d')
    
    ax.plot3D(tracking.tracks[0,:,event].numpy()*10,tracking.tracks[1,:,event].numpy()*10,(tracking.tracks[2,:,event].numpy()-0.2)*10,color='navy',label='Muon track')
    
    ax.set_xlim3d(0, Nvox_X)
    ax.set_ylim3d(0, Nvox_Y)
    ax.set_zlim3d(0, Nvox_Z)
    
    ax.set_xlabel('Voxel indice along x')
    ax.set_ylabel('Voxel indice along y')
    ax.set_zlabel('Voxel indice along z')
    
    ax.voxels(voxelarray,alpha=0.001,edgecolors='yellow')
    
    j=0
    for i in triggered_voxels_mask:
        if(j==0):
            ax.voxels(i, edgecolor='k',color='lightcoral',alpha=.05,label='hit voxels')
        j+1
        ax.voxels(i, edgecolor='k',color='lightcoral',alpha=.01)

    
    plt.show()