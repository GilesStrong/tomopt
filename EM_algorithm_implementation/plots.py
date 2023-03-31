import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from volume_interest import *
from numpy.random import rand
import pickle
from matplotlib.pyplot import figure
import random
import matplotlib.colors as colors
import imageio
import os
from rpy2.robjects.packages import importr
from rpy2.robjects import vectors


with open('materials_radiation_length.pkl', 'rb') as f:
    matr_lrad_dict = pickle.load(f)


def hist_gif(data, itermax, init_lrad, org_lrad, xmin, xmax, ymin, ymax, duration):
    images = []
    for i in range(1, itermax):
        plt.figure()
        plt.hist(data[i], bins=50, alpha=.2, color='blue', label='Estimated X0')
        plt.axvline(x=init_lrad, c='red', label='Initial X0')
        plt.axvline(x=org_lrad, c='green', label='Original X0')
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        plt.legend(loc='upper right')
        plt.text(0.3, 73, f"Iteration {i}", fontsize=12)
        plt.savefig(f"histogram_{i}.png")
        filename = f"histogram_{i}.png"
        images.append(imageio.imread(filename))

    imageio.mimsave("histogram.gif", images, duration=0.4)


def mean_squared_error(y_pred, y_true):
    n = y_true.numel()
    mse = torch.sum((y_true - y_pred) ** 2) / n
    return mse.item()


def compute_com_mse(lrad_reconstr, lrad_org):
    mse = []
    for i in range(6):
        mse.append(mean_squared_error(lrad_reconstr[:, :, i], lrad_org[:, :, i]))
    return round(sum(mse), 5)


def plot_mse_per_iteration(tracks, lrad_org, layer=0, per_layer=False, itermax=5):
    mse = []
    iterations = range(tracks.em_iter)

    for lrad_reconstr in tracks.rad_length[1:]:
        if per_layer:
            mse.append(mean_squared_error(lrad_reconstr[:, :, layer], lrad_org[:, :, layer]))
        else:
            mse.append(compute_com_mse(lrad_reconstr, lrad_org))

    plt.plot(iterations[1:itermax + 1], mse[0:itermax])

    if per_layer:
        plt.title(f'Layer {layer+1} Estimated Radiation Length MSE per Iteration')  # Title of the plot
    else:
        plt.title('Estimated Radiation Length MSE per Iteration')  # Title of the plot

    plt.xlabel('Iteration')
    plt.ylabel('Mean Square Error')
    plt.show()


def plot_rad_len_across_iter(em_itr, rad_len, org_rad_len, lrad_err, log_scale=False, mse=[], mse_total=1, modelling_appr=''):
    fig, ax = plt.subplots(3, 6, figsize=(18, 10))
    fig.subplots_adjust(hspace=0.75)
    cbar_ax = fig.add_axes([1.05, 0.4, 0.02, 0.45])
    L_rad = rad_len
    z_min, z_max = torch.min(L_rad), torch.max(L_rad)
    csfont = {'fontname': 'Comic Sans MS'}

    if log_scale:
        if z_min < 0:
            z_min = torch.min(torch.where(L_rad >= 0, L_rad, torch.tensor(float('inf'))))
        norm = colors.LogNorm(vmin=z_min, vmax=z_max)
    else:

        norm = colors.Normalize(vmin=z_min, vmax=z_max)
    if len(modelling_appr) > 0:
        plt.suptitle(f'MSE = {round(mse_total,4)}, #Epochs = {em_itr}, Modeling Approach: {modelling_appr}', fontsize=24)
    else:
        plt.suptitle(f'MSE = {round(mse_total,4)}, #Epochs = {em_itr}', fontsize=24, **csfont)

    for i in range(2):
        for j in range(len(rad_len[0][0])):
            if i == 0:
                sns.heatmap(org_rad_len[:, :, j], norm=norm, cmap='cividis', vmin=torch.min(org_rad_len[:, :, j]),
                            vmax=torch.max(org_rad_len[:, :, j]), ax=ax[i, j], cbar=i == 0, cbar_ax=None if i != 0 else cbar_ax)
            elif i == 1:
                sns.heatmap(rad_len[:, :, j], norm=norm, vmin=torch.min(rad_len[:, :, j]), vmax=torch.max(
                    rad_len[:, :, j]), cmap='cividis', ax=ax[i, j], cbar=i == 0, cbar_ax=None if i != 0 else cbar_ax)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].set_xlabel('x', fontsize=16)
            ax[i, j].set_ylabel('y', fontsize=16)
            if i == 0:
                ax[i, j].set_title(f'Layer {j+1}', fontsize=18)
            ax[i, j].set_aspect('equal')

        if i == 0:
            ax[i, 0].set_title(f"Original X0 - Layer {1}", fontsize=18)
        elif i == 1:
            ax[i, 0].set_title(f"Iteration {em_itr}", fontsize=18)

    cbar_ax3 = fig.add_axes([1.05, 0.05, 0.02, 0.25])
    z_min3, z_max3 = torch.min(lrad_err), torch.max(lrad_err)

    if log_scale:
        if z_min < 0:
            z_min3 = torch.min(torch.where(lrad_err >= 0, lrad_err, torch.tensor(float('inf'))))
        norm3 = colors.LogNorm(vmin=z_min3, vmax=z_max3)

    else:
        norm3 = colors.Normalize(vmin=z_min3, vmax=z_max3)

    for j in range(len(rad_len[0][0])):
        sns.heatmap(lrad_err[:, :, j], norm=norm3, cmap='PuBu', vmin=z_min3, vmax=z_max3,
                    ax=ax[2, j], cbar=j == 0, cbar_ax=None if j != 0 else cbar_ax3)
        ax[2, j].set_xticks([])
        ax[2, j].set_yticks([])
        ax[2, j].set_xlabel('x', fontsize=16)
        ax[2, j].set_ylabel('y', fontsize=16)
        ax[2, j].set_aspect('equal')
        mse_layer_ = round((mse[j] / mse_total) * 100)
        ax[2, j].set_title(f'MSE = {mse_layer_} %', fontsize=19)

    plt.show()


def group_indices_by_value(tensor):
    unique_values = torch.unique(tensor)
    indices_dict = {}

    for value in unique_values:
        indices = torch.nonzero(tensor == value, as_tuple=False)
        indices_list = indices.tolist()
        val = round(value.item(), 4)
        indices_dict[val] = indices_list

    return indices_dict


def radiation_distribution(L_rad, org_L_rad, em_itr):
    org_lrad_indices = group_indices_by_value(org_L_rad)
    pred_l_rad_mat = {}
    for matr in org_lrad_indices:
        coordinates = org_lrad_indices.get(matr)
        pred_l_rad_mat[matr] = [np.repeat(0, len(coordinates))] * em_itr
        for itr in range(0, em_itr):
            out = []
            for coord_indx in range(0, len(coordinates)):
                x, y, z = coordinates[coord_indx]
                out.append(float(L_rad[itr, x, y, z]))
            pred_l_rad_mat[matr][itr] = out
    return pred_l_rad_mat


def compute_density(lrad_distrb, itermax, log_scale=False):
    stats = importr("stats")
    itr_matr_lrad = {}
    itr_matr_limits = {}
    x_lim = []
    y_lim = []
    for matr in lrad_distrb:
        matr_lrad_info = []
        matr_lrad_lim = []

        for i in range(itermax):
            if log_scale:
                column = vectors.FloatVector(np.log(lrad_distrb[matr][i]))
            else:
                column = vectors.FloatVector(lrad_distrb[matr][i])

            output = stats.density(column, adjust=1)

            x = np.array(output[0])
            y = np.array(output[1])

            x_min = min(x)
            x_max = max(x)

            y_min = min(y)
            y_max = max(y)

            matr_lrad_info.append((x, y))
            matr_lrad_lim.append([(x_min, x_max), (y_min, y_max)])

        itr_matr_lrad[matr] = matr_lrad_info
        itr_matr_limits[matr] = matr_lrad_lim

    for i in range(itermax):
        y_min = []
        x_min = []
        y_max = []
        x_max = []

        for matr in itr_matr_lrad:
            y_min.append(itr_matr_limits[matr][i][1][0])
            x_min.append(itr_matr_limits[matr][i][0][0])
            x_max.append(itr_matr_limits[matr][i][0][1])
            y_max.append(itr_matr_limits[matr][i][1][1])

        x_lim.append((min(x_min), max(x_max)))
        y_lim.append((min(y_min), max(y_max)))

    return itr_matr_lrad, x_lim, y_lim


def generate_distinct_hex_colors(N):
    hex_colors = set()
    while len(hex_colors) < N:
        hex_color = '#{0:06x}'.format(random.randint(0, 0xFFFFFF))
        hex_colors.add(hex_color)
    return list(hex_colors)


def plot_density_for_LRad(matr_lrad, x_lim, y_lim, itr, rand_colors, log_scale):

    figure(figsize=(8, 6), dpi=125)

    i = 0
    for matr in matr_lrad:
        plt.plot(matr_lrad[matr][itr][0], matr_lrad[matr][itr][1], color=rand_colors[i], label=matr_lrad_dict[matr], alpha=0.35)
        plt.fill_between(matr_lrad[matr][itr][0], matr_lrad[matr][itr][1], 0, color=rand_colors[i], alpha=0.5)
        i = i + 1

    plt.legend(loc="upper left")
    plt.title('Iteration {}'.format(itr))
    plt.ylabel('Density')

    if not log_scale:
        plt.xlabel('Radiation Length (X0)')
    else:
        plt.xlabel('Log( Radiation Length (X0) )')

    plt.xlim(x_lim[itr][0], x_lim[itr][1])
    plt.ylim(y_lim[itr][0], y_lim[itr][1] + 1)
    plt.show()


def plot_lrad_case_study(material_1, material_2, xlim, ylim, itr, material_1_label, material_2_label):
    x_beryl = material_1[0]
    y_beryl = material_1[1]

    x_lead = material_2[0]
    y_lead = material_2[1]

    figure(figsize=(8, 6), dpi=125)
    plt.plot(x_beryl, y_beryl, color="pink", label=material_1_label, alpha=0.5)
    plt.fill_between(x_beryl, y_beryl, 0, color="pink")
    plt.plot(x_lead, y_lead, color="purple", label=material_2_label, alpha=0.2)
    plt.fill_between(x_lead, y_lead, 0, color="purple", alpha=0.4)
    plt.legend(loc="upper left")
    plt.title('Iteration {}'.format(itr))
    plt.ylabel('Density')
    plt.xlabel('Radiation Length (X0)')
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1] + 1)
    plt.show()


def plot_muon_hits_count(hits):
    fig, ax = plt.subplots(3, 2, figsize=(18, 18))
    fig.suptitle('Counts of Muon Hits per Layer', fontsize=20)
    figure(figsize=(8, 6), dpi=80)

    z0 = hits[:, :, 0]
    z1 = hits[:, :, 1]
    z2 = hits[:, :, 2]
    z3 = hits[:, :, 3]
    z4 = hits[:, :, 4]
    z5 = hits[:, :, 5]

    ax[0, 0].imshow(z0, interpolation='None', cmap='YlGn', aspect='auto')
    ax[0, 1].imshow(z1, interpolation='None', cmap='YlGn', aspect='auto')
    ax[1, 0].imshow(z2, interpolation='None', cmap='YlGn', aspect='auto')
    ax[1, 1].imshow(z3, interpolation='None', cmap='YlGn', aspect='auto')
    ax[2, 0].imshow(z4, interpolation='None', cmap='YlGn', aspect='auto')
    ax[2, 1].imshow(z5, interpolation='None', cmap='YlGn', aspect='auto')

    for (j, i), label in np.ndenumerate(z0):
        ax[0, 0].text(i, j, label, ha='center', va='center', fontsize=12)
        ax[0, 0].set_title('Layer 1', fontsize=16)
    for (j, i), label in np.ndenumerate(z1):
        ax[0, 1].text(i, j, label, ha='center', va='center', fontsize=12)
        ax[0, 1].set_title('Layer 2', fontsize=16)
    for (j, i), label in np.ndenumerate(z2):
        ax[1, 0].text(i, j, label, ha='center', va='center', fontsize=12)
        ax[1, 0].set_title('Layer 3', fontsize=16)
    for (j, i), label in np.ndenumerate(z3):
        ax[1, 1].text(i, j, label, ha='center', va='center', fontsize=12)
        ax[1, 1].set_title('Layer 4', fontsize=16)
    for (j, i), label in np.ndenumerate(z4):
        ax[2, 0].text(i, j, label, ha='center', va='center', fontsize=12)
        ax[2, 0].set_title('Layer 5', fontsize=16)
    for (j, i), label in np.ndenumerate(z5):
        ax[2, 1].text(i, j, label, ha='center', va='center', fontsize=12)
        ax[2, 1].set_title('Layer 6', fontsize=16)


def plot_muon_track(VOI: VolumeOfInterest, event: int = 0, tracking: Tracking = None) -> None:
    fig, ax = plt.subplots(ncols=2, figsize=(15, 5))
    fig.suptitle('Volume of interest')
    ax = ax.ravel()
    xyz_max = VOI.xyz_max
    xyz_min = VOI.xyz_min

    for i in range(VOI.n_vox_xyz[0]):
        ax[0].axvline(x=VOI.voxel_edges[i, 0, 0, 0, 0], ymin=VOI.xyz_min[2], ymax=VOI.xyz_max[2], color='lightgrey')
        ax[0].axvline(x=VOI.voxel_edges[i, 0, 0, 1, 0], ymin=VOI.xyz_min[2], ymax=VOI.xyz_max[2], color='lightgrey')

    for i in range(VOI.n_vox_xyz[1]):
        ax[1].axvline(x=VOI.voxel_edges[i, 0, 0, 0, 0], ymin=VOI.xyz_min[2], ymax=VOI.xyz_max[2], color='lightgrey')
        ax[1].axvline(x=VOI.voxel_edges[i, 0, 0, 1, 0], ymin=VOI.xyz_min[2], ymax=VOI.xyz_max[2], color='lightgrey')

    for z in range(VOI.n_vox_xyz[2]):
        ax[0].axhline(y=VOI.voxel_edges[0, 0, z, 0, 2], xmin=VOI.xyz_min[0], xmax=VOI.xyz_max[0], color='lightgrey')
        ax[0].axhline(y=VOI.voxel_edges[0, 0, z, 1, 2], xmin=VOI.xyz_min[0], xmax=VOI.xyz_max[0], color='lightgrey')

        ax[1].axhline(y=VOI.voxel_edges[0, 0, z, 0, 2], xmin=VOI.xyz_min[0], xmax=VOI.xyz_max[0], color='lightgrey')
        ax[1].axhline(y=VOI.voxel_edges[0, 0, z, 1, 2], xmin=VOI.xyz_min[0], xmax=VOI.xyz_max[0], color='lightgrey')

    ax[0].set_title('XZ view')
    ax[0].set_aspect('equal')
    ax[0].set_xlim([xyz_min[0], xyz_max[0]])
    ax[0].set_ylim([xyz_min[1], xyz_max[1]])
    ax[0].set_xlabel('x [m]')
    ax[0].set_ylabel('z [m]')

    ax[1].set_title('XZ view')
    ax[1].set_aspect('equal')
    ax[1].set_xlim([xyz_min[0], xyz_max[0]])
    ax[1].set_ylim([xyz_min[1], xyz_max[1]])
    ax[1].set_xlabel('y [m]')
    ax[1].set_ylabel('z [m]')

    if (tracking is not None):

        fig.suptitle('Tracking for event = {}'.format(event))
        ax[0].plot(tracking.tracks[0, :, event], tracking.tracks[2, :, event], color='maroon', label='Muon track')
        ax[1].plot(tracking.tracks[1, :, event], tracking.tracks[2, :, event], color='maroon', label='Muon track')

        ax[0].scatter(tracking.data['xyz_in_x'][event], tracking.data['xyz_in_z'][event], marker='v', color='darkgreen', label='Entry point')
        ax[0].scatter(tracking.data['xyz_out_x'][event], tracking.data['xyz_out_z'][event], marker='v', color='blue', label='Exit point')

        ax[1].scatter(tracking.data['xyz_in_y'][event], tracking.data['xyz_in_z'][event], marker='v', color='darkgreen', label='Entry point')
        ax[1].scatter(tracking.data['xyz_out_y'][event], tracking.data['xyz_out_z'][event], marker='v', color='blue', label='Exit point')

        ax[0].legend()
        ax[1].legend()

    plt.tight_layout()
    plt.show()


def plot_discrete_track_2d(VOI: VolumeOfInterest, event: int = 0, tracking: Tracking = None) -> None:
    N_voxels = VOI.n_vox_xyz
    xyz_max = VOI.xyz_max
    xyz_min = VOI.xyz_min

    fig, ax = plt.subplots(ncols=2, figsize=(15, 5))
    ax = ax.ravel()

    fig.suptitle('Tracking for event = {}'.format(event), y=0.95)

    for i in range(N_voxels[0]):
        ax[0].axvline(x=VOI.voxel_edges[i, 0, 0, 0, 0], ymin=xyz_min[2], ymax=xyz_max[2], color='lightgrey')
        ax[0].axvline(x=VOI.voxel_edges[i, 0, 0, 1, 0], ymin=xyz_min[2], ymax=xyz_max[2], color='lightgrey')

        ax[1].axvline(x=VOI.voxel_edges[i, 0, 0, 0, 0], ymin=xyz_min[2], ymax=xyz_max[2], color='lightgrey')
        ax[1].axvline(x=VOI.voxel_edges[i, 0, 0, 1, 0], ymin=xyz_min[2], ymax=xyz_max[2], color='lightgrey')

    for z in range(N_voxels[2]):
        ax[0].axhline(y=VOI.voxel_edges[0, 0, z, 0, 2], xmin=xyz_min[0], xmax=xyz_max[0], color='lightgrey')
        ax[0].axhline(y=VOI.voxel_edges[0, 0, z, 1, 2], xmin=xyz_min[0], xmax=xyz_max[0], color='lightgrey')

        ax[1].axhline(y=VOI.voxel_edges[0, 0, z, 0, 2], xmin=xyz_min[0], xmax=xyz_max[0], color='lightgrey')
        ax[1].axhline(y=VOI.voxel_edges[0, 0, z, 1, 2], xmin=xyz_min[0], xmax=xyz_max[0], color='lightgrey')

    x_intersec = []
    y_intersec = []
    z_intersec = []

    for voxel in tracking.intersection_coordinates[event]:
        x_intersec.append(voxel[0])
        y_intersec.append(voxel[1])
        z_intersec.append(voxel[2])

    ax[0].set_aspect('equal')
    ax[0].set_xlim([xyz_min[0], xyz_max[0]])
    ax[0].set_ylim([xyz_min[1], xyz_max[1]])
    ax[0].set_xlabel('x [m]')
    ax[0].set_ylabel('z [m]')

    ax[1].set_aspect('equal')
    ax[1].set_xlim([xyz_min[0], xyz_max[0]])
    ax[1].set_ylim([xyz_min[1], xyz_max[1]])
    ax[1].set_xlabel('y [m]')
    ax[1].set_ylabel('z [m]')

    ax[0].plot(tracking.tracks[0, :, tracking.indices[event]], tracking.tracks[2, :, tracking.indices[event]], color='maroon', label='Muon track')
    ax[1].plot(tracking.tracks[1, :, tracking.indices[event]], tracking.tracks[2, :, tracking.indices[event]], color='maroon', label='Muon track')

    ax[0].scatter(tracking.data['xyz_in_x'][tracking.indices[event]], tracking.data['xyz_in_z']
                  [tracking.indices[event]], marker='v', color='darkgreen', label='Entry point')
    ax[0].scatter(tracking.data['xyz_out_x'][tracking.indices[event]], tracking.data['xyz_out_z']
                  [tracking.indices[event]], marker='v', color='blue', label='Exit point')
    ax[0].scatter(tracking.tracks[0, 1:-1, tracking.indices[event]], tracking.tracks[2, 1:-1, tracking.indices[event]], marker='x', label='muon entering layer')
    ax[0].scatter(x_intersec, z_intersec, marker='*', color='orange', label='muon layer intersect')

    ax[1].scatter(tracking.data['xyz_in_y'][tracking.indices[event]], tracking.data['xyz_in_z']
                  [tracking.indices[event]], marker='v', color='darkgreen', label='Entry point')
    ax[1].scatter(tracking.data['xyz_out_y'][tracking.indices[event]], tracking.data['xyz_out_z']
                  [tracking.indices[event]], marker='v', color='blue', label='Exit point')
    ax[1].scatter(tracking.tracks[1, 1:-1, tracking.indices[event]], tracking.tracks[2, 1:-1, tracking.indices[event]], marker='x', label='muon entering layer')
    ax[1].scatter(y_intersec, z_intersec, marker='*', color='orange', label='muon layer intersect')

    i = 0
    for vox in tracking.triggered_voxels[event]:
        ix = int(vox[0])
        iy = int(vox[1])
        iz = int(vox[2])

        if (i == 0):
            ax[0].scatter(VOI.voxel_centers[ix, iy, iz, 0], VOI.voxel_centers[ix, iy, iz, 2], label='triggered voxel', color='yellowgreen')
            ax[1].scatter(VOI.voxel_centers[ix, iy, iz, 1], VOI.voxel_centers[ix, iy, iz, 2], label='triggered voxel', color='yellowgreen')

        ax[0].scatter(VOI.voxel_centers[ix, iy, iz, 0], VOI.voxel_centers[ix, iy, iz, 2], color='yellowgreen')

        ax[1].scatter(VOI.voxel_centers[ix, iy, iz, 1], VOI.voxel_centers[ix, iy, iz, 2], color='yellowgreen')
        i += 1

    ax[0].legend()
    ax[1].legend()

    plt.show()


def plot_discrete_track_3d(VOI: VolumeOfInterest, event: int = 0, tracking: Tracking = None) -> None:

    Nvox_X = VOI.n_vox_xyz[0].numpy()
    Nvox_Y = VOI.n_vox_xyz[1].numpy()
    Nvox_Z = VOI.n_vox_xyz[2].numpy()

    x, y, z = np.indices((Nvox_X, Nvox_Y, Nvox_Z))
    voxelarray = (x <= Nvox_X) & (y <= Nvox_Y) & (z <= Nvox_Z)
    triggered_voxels_mask = []
    colors = np.zeros(voxelarray.shape, dtype=object)

    for i in tracking.triggered_voxels[event]:
        ix, iy, iz = i[0], i[1], i[2]
        triggered_voxels_mask.append((x == ix) & (y == iy) & ((z) == (5 - iz)))

    for i in triggered_voxels_mask:
        colors[i] = 'yellowgreen'

    colors = np.where((colors == 0), 'white', 'yellowgreen')

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle('Tracking for event = {}'.format(event), y=0.95)
    ax = fig.add_subplot(projection='3d')

    ax.plot3D(tracking.tracks[0, :, tracking.indices[event]].numpy() * 10, tracking.tracks[1, :, tracking.indices[event]].numpy()
              * 10, (tracking.tracks[2, :, tracking.indices[event]].numpy() - 0.2) * 10, color='maroon', label='Muon track')

    ax.set_xlim3d(0, Nvox_X)
    ax.set_ylim3d(0, Nvox_Y)
    ax.set_zlim3d(0, Nvox_Z)

    ax.set_xlabel('Voxel indice along x')
    ax.set_ylabel('Voxel indice along y')
    ax.set_zlabel('Voxel indice along z')

    ax.voxels(voxelarray, alpha=0.0001, edgecolors='whitesmoke')

    j = 0
    for i in triggered_voxels_mask:
        if (j == 0):
            ax.voxels(i, edgecolor='k', color='yellowgreen', alpha=.01, label='hit voxels')
        j + 1
        ax.voxels(i, edgecolor='k', color='yellowgreen', alpha=.01)

    x_intersec = []
    y_intersec = []
    z_intersec = []

    for voxel in tracking.intersection_coordinates[event]:
        x_intersec.append(voxel[0] * 10)
        y_intersec.append(voxel[1] * 10)
        z_intersec.append((voxel[2] - 0.2) * 10)

    ax.scatter(x_intersec, y_intersec, z_intersec, marker='*', color='orange', label='muon layer intersect')
    ax.scatter(tracking.data['xyz_in_x'][tracking.indices[event]] * 10, tracking.data['xyz_in_y'][tracking.indices[event]] * 10,
               (tracking.data['xyz_in_z'][tracking.indices[event]] - 0.2) * 10, marker='v', color='darkgreen', label='Entry point')
    ax.scatter(tracking.data['xyz_out_x'][tracking.indices[event]] * 10, tracking.data['xyz_out_y'][tracking.indices[event]]
               * 10, (tracking.data['xyz_out_z'][tracking.indices[event]] - 0.2) * 10, marker='v', color='blue', label='Exit point')

    plt.show()


def plot_event_Lrad_heatmaps(L_rad: np.ndarray, event: int, log_scale=False, org_Lrad=False) -> None:
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), gridspec_kw={"hspace": 0.4})
    cbar_ax = fig.add_axes([1.05, 0.15, 0.02, 0.7])
    z_min, z_max = torch.min(L_rad), torch.max(L_rad)

    if log_scale:
        if z_min < 0:
            z_min = torch.min(torch.where(L_rad >= 0, L_rad, torch.tensor(float('inf'))))
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
        mat = np.unique(L_rad).tolist()
        mat_names = ""
        for m in range(0, len(mat)):
            mat_names = mat_names + matr_lrad_dict[round(float(mat[m]), 4)]
            if m + 1 < len(mat):
                mat_names = mat_names + ", "
        plt.suptitle(f'Heatmaps of Radiation Length for {mat_names}')
    else:
        plt.suptitle(f'Heatmaps of Radiation Length for Iteration {event+1}')

    fig.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()


def concat(lst, z, x1, x2, y1, y2, itr):
    out = []

    for i in range(x1, x2):
        for j in range(y1, y2):
            out.append(float(lst[itr, i, j, z]))
    return out


def compute_density_case_study(X0_f_beryllium, X0_f_lead, itermax, log_scale=False):
    stats = importr("stats")
    dens_lead = []
    dens_beryllium = []
    xlim = []
    ylim = []
    for i in range(itermax):
        if not log_scale:
            column = vectors.FloatVector(X0_f_beryllium[i])
            output = stats.density(column, adjust=1)
            x_beryl = np.array(output[0])
            y_beryl = np.array(output[1])
            dens_beryllium.append((x_beryl, y_beryl))

            column = vectors.FloatVector(X0_f_lead[i])
            output = stats.density(column, adjust=1)
            x_lead = np.array(output[0])
            y_lead = np.array(output[1])
            dens_lead.append((x_lead, y_lead))

            x_min_lead = min(x_lead)
            x_max_lead = max(x_lead)
            x_min_beryl = min(x_beryl)
            x_max_beryl = max(x_beryl)

            y_min_lead = min(y_lead)
            y_max_lead = max(y_lead)
            y_min_beryl = min(y_beryl)
            y_max_beryl = max(y_beryl)

            x_min = min(x_min_lead, x_min_beryl)
            x_max = max(x_max_lead, x_max_beryl)

            y_min = min(y_min_lead, y_min_beryl)
            y_max = max(y_max_lead, y_max_beryl)

            xlim.append((x_min, x_max))
            ylim.append((y_min, y_max))

        if log_scale:
            column = vectors.FloatVector(np.log(X0_f_beryllium[i]))
            output = stats.density(column, adjust=1)
            x_beryl = np.array(output[0])
            y_beryl = np.array(output[1])
            dens_beryllium.append((x_beryl, y_beryl))

            column = vectors.FloatVector(np.log(X0_f_lead[i]))
            output_log = stats.density(column, adjust=1)
            column = vectors.FloatVector(X0_f_lead[i])
            output = stats.density(column, adjust=1)

            _x_lead = output[0]
            x_lead = output_log[0]
            y_lead = output_log[1]
            dens_lead.append((x_lead, y_lead))

            x_min_lead = min(_x_lead)
            x_max_lead = max(_x_lead)
            x_min_beryl = min(x_beryl)
            x_max_beryl = max(x_beryl)

            y_min_lead = min(y_lead)
            y_max_lead = max(y_lead)
            y_min_beryl = min(y_beryl)
            y_max_beryl = max(y_beryl)

            x_min = min(x_min_lead, x_min_beryl)
            x_max = max(x_max_lead, x_max_beryl)

            y_min = min(y_min_lead, y_min_beryl)
            y_max = max(y_max_lead, y_max_beryl)

            xlim.append((x_min, x_max))
            ylim.append((y_min, y_max))

    return dens_beryllium, dens_lead, xlim, ylim


def plot_density_for_LRad_case_study(material_1, material_2, xlim, ylim, itr, material_1_label, material_2_label, log_scale=False):

    x_beryl = material_1[0]
    y_beryl = material_1[1]

    x_lead = material_2[0]
    y_lead = material_2[1]

    if log_scale:
        x_lead = np.log(material_2[0])
        x_beryl = np.log(material_1[0])

    figure(figsize=(8, 6), dpi=125)
    plt.plot(x_beryl, y_beryl, color="pink", label=material_1_label, alpha=0.5)
    plt.fill_between(x_beryl, y_beryl, 0, color="pink")
    plt.plot(x_lead, y_lead, color="purple", label=material_2_label, alpha=0.2)
    plt.fill_between(x_lead, y_lead, 0, color="purple", alpha=0.4)
    plt.legend(loc="upper left")
    plt.title('Iteration {}'.format(itr))
    plt.ylabel('Density')
    plt.xlabel('Radiation Length (X0)')
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1] + 1)
    plt.show()
