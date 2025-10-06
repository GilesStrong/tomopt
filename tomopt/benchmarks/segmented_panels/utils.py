import pickle
from functools import partial
from typing import Any, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from torch import Tensor

# segmented panel imports
from tomopt.benchmarks.segmented_panels.callbacks import PredHitRecord
from tomopt.benchmarks.segmented_panels.detector_config import get_layers
from tomopt.benchmarks.segmented_panels.layer import SegmentedPanelDetectorLayer
from tomopt.benchmarks.segmented_panels.panel import SegmentedSigmoidDetectorPanel
from tomopt.benchmarks.segmented_panels.wrapper import SegmentedPanelVolumeWrapper

# tomopt imports
from tomopt.inference import ScatterBatch
from tomopt.optimisation.callbacks import Callback
from tomopt.optimisation.data.passives import PassiveYielder
from tomopt.volume import PassiveLayer, Volume

"""_description_
This module contains utility functions for analysis and performance evaluation.
"""

_all_ = [
    "get_angular_resolution",
    "get_angular_resolution_with_noise_hits",
    "zenith_angle_deviation",
    "get_ssim",
    "get_mse_passives",
    "get_preds_from_volume",
    "draw",
    "set_plot_style",
]


def get_angular_resolution(hit_record: PredHitRecord, volume: Volume, show_plot: bool = True) -> Tuple[Tensor, Tensor]:
    """
    Computes the angular resolution of the reconstructed hits
    of the passive volumes.

    Arguments:
        hit_record: The hit record containing reconstructed and generated hits.

    Returns:
        Tensor of the angular resolution in degrees of the passive volumes
    """

    angular_resolutions_in = torch.zeros(len(hit_record.reco_hits_batch))
    angular_resolutions_out = torch.zeros(len(hit_record.reco_hits_batch))

    for i, reco_hits in enumerate(hit_record.reco_hits_batch):
        # Extract reconstructed and generated hits
        gen_above, gen_below = hit_record.split_above_below(hit_record.gen_hits_batch[i])
        reco_above, reco_below = hit_record.split_above_below(reco_hits)
        uncs_above, uncs_below = hit_record.split_above_below(hit_record.hit_uncs_batch[i])

        # For reconstructed tracks
        reco_above_vecs, reco_above_starts = ScatterBatch.get_muon_trajectory(reco_above, uncs_above, volume.lw)
        reco_below_vecs, reco_below_starts = ScatterBatch.get_muon_trajectory(reco_below, uncs_below, volume.lw)

        # For generated tracks
        # Note: gen hits are "perfect"
        # so pass dummy uncertainties (small or constant)
        dummy_uncs = torch.ones_like(gen_above) * 1e-8

        gen_above_vecs, gen_above_starts = ScatterBatch.get_muon_trajectory(gen_above, dummy_uncs, volume.lw)
        gen_below_vecs, gen_below_starts = ScatterBatch.get_muon_trajectory(gen_below, dummy_uncs, volume.lw)

        diff_zenith_incoming = zenith_angle_deviation(reco_above_vecs, gen_above_vecs)
        diff_zenith_outgoing = zenith_angle_deviation(reco_below_vecs, gen_below_vecs)
        incoming_resolution = diff_zenith_incoming.std(unbiased=False).item()
        outgoing_resolution = diff_zenith_outgoing.std(unbiased=False).item()

        # Store the angular resolutions
        angular_resolutions_in[i] = incoming_resolution
        angular_resolutions_out[i] = outgoing_resolution

        # Now plot both with the SAME bins
        axrange = (-0.5, 0.5)
        plt.hist(diff_zenith_incoming.numpy(), bins=100, alpha=0.5, label=f"Incoming (σ={incoming_resolution:.5f} deg)", range=axrange)
        plt.hist(
            diff_zenith_outgoing.numpy(),
            bins=100,
            alpha=0.5,
            label=f"Outgoing (σ={outgoing_resolution:.5f} deg)",
            range=axrange,
        )
        plt.xlabel("Angular error (degrees)", fontsize=16)
        plt.ylabel("Counts", fontsize=16)
        plt.title("Angular error between reco and gen muon tracks", fontsize=16)
        plt.legend(fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        if show_plot:
            plt.show()
        else:
            plt.close()

    return angular_resolutions_in, angular_resolutions_out


def get_angular_resolution_with_noise_hits(hit_record: PredHitRecord, volume: Volume, show_plot: bool = True) -> Tuple[Tensor, Tensor]:
    """
    Computes the angular resolution of the reconstructed hits
    of the passive volumes.

    Arguments:
        hit_record: The hit record containing reconstructed and generated hits.

    Returns:
        Tensor of the angular resolution in degrees of the passive volumes
    """
    from itertools import product

    # Sampling noise hits
    def sample_noise_hits(
        hits_tensor: Tensor,
        uncs_tensor: Tensor,
        n_noise_range: Tuple[int, int] = (1, 7),
        noise_box: Tensor = volume.lw,
        noise_unc: float = 1.0,
        filename: str = "noisy_hits",
    ) -> Tuple[List[list], List[list]]:
        n_muons, n_panels = hits_tensor.shape[:2]
        print(n_muons)
        print(n_panels)
        noisy_hits = []
        noisy_uncs = []

        for i in range(n_muons):
            print(i)
            panel_hits = []
            panel_uncs = []
            for j in range(n_panels):
                true_hit = hits_tensor[i][j]
                true_unc = uncs_tensor[i][j]
                n_noise = torch.randint(*n_noise_range, size=(1,)).cpu().detach().numpy()

                noise = torch.stack(
                    [
                        torch.empty(n_noise).uniform_(0, noise_box[0].cpu().detach().numpy()),
                        torch.empty(n_noise).uniform_(0, noise_box[1].cpu().detach().numpy()),
                        torch.full((n_noise,), hits_tensor[i, j, 2].item()),
                    ],
                    dim=1,
                ).to(true_hit)
                noise_unc_tensor = torch.full((n_noise, 3), float(noise_unc), device=true_hit.device)
                noise_unc_tensor[:, 2] = 0.0

                panel_hits.append(torch.cat([true_hit.unsqueeze(0), noise], dim=0))
                # indices = torch.randperm(panel_hits[-1].shape[0])
                # panel_hits[-1] = panel_hits[-1][indices]
                panel_uncs.append(torch.cat([true_unc.unsqueeze(0), noise_unc_tensor], dim=0))
            noisy_hits.append(panel_hits)
            noisy_uncs.append(panel_uncs)
        # Save noisy_hits to pickle
        with open(f"{filename}.pkl", "wb") as f:
            pickle.dump(noisy_hits, f)

        return noisy_hits, noisy_uncs

    # Brute-force fitting
    def find_best_track_with_uncs(panel_hits: List[Tensor], panel_uncs: List[Tensor], lw: Tensor) -> Tuple[Tensor, Tensor]:
        best_chi2 = float("inf")
        best_comb = None
        for hit_comb, unc_comb in zip(product(*panel_hits), product(*panel_uncs)):
            hits = torch.stack(hit_comb).unsqueeze(0)
            uncs = torch.stack(unc_comb).unsqueeze(0)
            vec, start = ScatterBatch.get_muon_trajectory(hits, uncs, lw)
            dz = hits[0, :, 2:3] - start[:, 2:3]
            pred_xy = start[:, :2] + dz * (vec[:, :2] / vec[:, 2:3])
            residuals = (hits[0, :, :2] - pred_xy) ** 2 / (uncs[0, :, :2] ** 2)
            chi2 = residuals.sum()
            if chi2 < best_chi2:
                best_chi2 = chi2
                best_comb = (vec, start)
        return best_comb

    # Fit all muons
    def fit_all_muons_brute_with_uncs(
        hits_tensor: Tensor, uncs_tensor: Tensor, lw: Tensor, n_noise_range: Tuple[int, int] = (3, 10), noise_unc: float = 0.1, filename: str = "noisy_hits"
    ) -> Tuple[Tensor, Tensor]:
        noisy_muons, noisy_uncs = sample_noise_hits(hits_tensor, uncs_tensor, n_noise_range, lw, noise_unc, filename=filename)
        vecs, starts = [], []
        for panel_hits, panel_uncs in zip(noisy_muons, noisy_uncs):
            result = find_best_track_with_uncs(panel_hits, panel_uncs, lw)
            if result is not None:
                vec, start = result
                vecs.append(vec)
                starts.append(start)
        return torch.cat(vecs, dim=0), torch.cat(starts, dim=0)

    angular_resolutions_in = torch.zeros(len(hit_record.reco_hits_batch))
    angular_resolutions_out = torch.zeros(len(hit_record.reco_hits_batch))

    for i, reco_hits in enumerate(hit_record.reco_hits_batch):
        # Extract reconstructed and generated hits
        gen_above, gen_below = hit_record.split_above_below(hit_record.gen_hits_batch[i])
        reco_above, reco_below = hit_record.split_above_below(reco_hits)
        uncs_above, uncs_below = hit_record.split_above_below(hit_record.hit_uncs_batch[i])

        # # For reconstructed tracks
        # reco_above_vecs, reco_above_starts = ScatterBatch.get_muon_trajectory(
        #     reco_above, uncs_above, volume.lw)
        # reco_below_vecs, reco_below_starts = ScatterBatch.get_muon_trajectory(
        #     reco_below, uncs_below, volume.lw)
        # i commented avove

        # extract and fit hits with noise injected
        reco_above_vecs, reco_above_starts = fit_all_muons_brute_with_uncs(reco_above, uncs_above, volume.lw, filename="incoming_noisy_hits")
        reco_below_vecs, reco_below_starts = fit_all_muons_brute_with_uncs(reco_below, uncs_below, volume.lw, filename="outgoing_noisy_hits")

        # For generated tracks
        # Note: gen hits are "perfect"
        # so pass dummy uncertainties (small or constant)
        dummy_uncs = torch.ones_like(gen_above) * 1e-8

        gen_above_vecs, gen_above_starts = ScatterBatch.get_muon_trajectory(gen_above, dummy_uncs, volume.lw)
        gen_below_vecs, gen_below_starts = ScatterBatch.get_muon_trajectory(gen_below, dummy_uncs, volume.lw)

        diff_zenith_incoming = zenith_angle_deviation(reco_above_vecs, gen_above_vecs)
        diff_zenith_outgoing = zenith_angle_deviation(reco_below_vecs, gen_below_vecs)
        incoming_resolution = diff_zenith_incoming.std(unbiased=False).item()
        outgoing_resolution = diff_zenith_outgoing.std(unbiased=False).item()

        # Store the angular resolutions
        angular_resolutions_in[i] = incoming_resolution
        angular_resolutions_out[i] = outgoing_resolution

        # Now plot both with the SAME bins
        axrange = (-10.5, 10.5)
        plt.hist(diff_zenith_incoming.numpy(), bins=100, alpha=0.5, label=f"Incoming (σ={incoming_resolution:.5f} deg)", range=axrange)
        plt.hist(
            diff_zenith_outgoing.numpy(),
            bins=100,
            alpha=0.5,
            label=f"Outgoing (σ={outgoing_resolution:.5f} deg)",
            range=axrange,
        )
        plt.xlabel("Angular error (degrees)")
        plt.ylabel("Counts")
        plt.title("Angular error between reco and gen muon tracks")
        plt.legend()
        if show_plot:
            plt.show()
        else:
            plt.close()

    return angular_resolutions_in, angular_resolutions_out


def zenith_angle_deviation(v_reco: Tensor, v_gen: Tensor) -> Tensor:
    """
    Computes the difference between the zenith angles of two batches of vectors with respect to the z-axis.

    Arguments:
        v_reco, v_gen: (n_muons, 3) tensors representing the reconstructed and generated vectors.

    Returns:
        (n_muons,) tensor of differences in zenith angles in degrees
    """
    # Normalize vectors
    v_reco = v_reco / v_reco.norm(dim=1, keepdim=True)
    v_gen = v_gen / v_gen.norm(dim=1, keepdim=True)

    # Compute zenith angles (cosine of the zenith angle is the z-component
    # of the normalized vector)
    zenith_reco = torch.acos(v_reco[:, 2])  # acos of z-component
    zenith_gen = torch.acos(v_gen[:, 2])

    # Difference in zenith angles
    delta_zenith = zenith_reco - zenith_gen

    # Convert from radians to degrees
    delta_zenith_degrees = delta_zenith * 180 / torch.pi

    return delta_zenith_degrees


def get_ssim(prediction: List[Tuple[np.ndarray, np.ndarray]], window_size: int = 3, sigma: float = 1.5, bca: bool = False) -> Tensor:

    def normalize(x: np.ndarray) -> Tensor:
        """
        Normalize the input tensor to the range [0, 1].
        """
        x_min = x.min()
        x_max = x.max()
        return (x - x_min) / (x_max - x_min)

    ssim = torch.zeros(len(prediction))
    for i, (pred, targ) in enumerate(prediction):
        # Ensure input has shape [Batch, Channels, Z, X, Y]
        pred_tensor = torch.tensor(pred).unsqueeze(0).unsqueeze(0)
        target_tensor = torch.tensor(targ).unsqueeze(0).unsqueeze(0)

        if bca:
            target_tensor = normalize(np.log(targ))
            pred_tensor = normalize(pred)

        # Compute local means
        mu_x = F.avg_pool3d(pred_tensor, kernel_size=window_size, stride=1, padding=window_size // 2)
        mu_y = F.avg_pool3d(target_tensor, kernel_size=window_size, stride=1, padding=window_size // 2)

        # Compute variances and covariance
        sigma_x = F.avg_pool3d(pred_tensor**2, kernel_size=window_size, stride=1, padding=window_size // 2) - mu_x**2
        sigma_y = F.avg_pool3d(target_tensor**2, kernel_size=window_size, stride=1, padding=window_size // 2) - mu_y**2
        sigma_xy = F.avg_pool3d(pred_tensor * target_tensor, kernel_size=window_size, stride=1, padding=window_size // 2) - mu_x * mu_y

        # Constants for numerical stability
        C1, C2 = 1e-4, 1e-4

        # Compute SSIM map
        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))

        # Take the mean SSIM score and convert it to a loss (since we minimize)
        ssim[i] = ssim_map.mean()

    return ssim


def get_mse_passives(prediction: List[Tuple[np.ndarray, np.ndarray]]) -> Tensor:
    """
    Computes the mean squared error (MSE) between the predicted
    and target values of the passive volumes.

    Arguments:
        prediction: List of tuples corresponding to each passive volume,
        each tuple consisting of the predicted and target values.

    Returns:
        Tesnor of MSE values of the passive volumes.
    """
    mse = torch.zeros(len(prediction))
    for i, (pred, targ) in enumerate(prediction):
        pred_tensor = torch.Tensor(pred)
        targ_tensor = torch.Tensor(targ)
        mse_loss = F.mse_loss(pred_tensor, targ_tensor, reduction="mean")
        mse[i] = mse_loss.item()
    return mse


def get_preds_from_volume(passives: PassiveYielder, det_layer_size: int = 4, n_muons: int = 1000) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Callback]:
    """
    Get the predictions from the volume.

    Arguments:
        det_layer_size: Size of the detector layer.

    Returns:
        Tuple of list of predictions and hit record.
    """
    hr = PredHitRecord()
    layers = get_layers(det_layer_size=det_layer_size)
    volume = Volume(layers)
    wrapper = SegmentedPanelVolumeWrapper(
        volume,
        xy_pos_opt=partial(torch.optim.SGD, lr=1e3),
        xy_span_opt=partial(torch.optim.SGD, lr=1e3),
        z_pos_opt=partial(torch.optim.SGD, lr=1e3),
        gap_opt=partial(torch.optim.SGD, lr=1e3),
    )
    return wrapper.predict(passives, n_mu_per_volume=n_muons, mu_bs=500, cbs=[hr]), hr


def draw(
    volume: Volume,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    zlim: Tuple[float, float],
    rotate: float = 90,
    filename: Optional[str] = None,
    show_plot: bool = True,
) -> None:
    r"""
    Draws the layers/panels pertaining to the volume.
    When using this in a jupyter notebook, use "%matplotlib notebook" to have an interactive plot that you can rotate.

    Arguments:
        xlim: the x axis range for the three-dimensional plot.
        ylim: the y axis range for the three-dimensional plot.
        zlim: the z axis range for the three-dimensional plot.
    """
    ax = plt.figure(figsize=(9, 9)).add_subplot(projection="3d")
    ax.computed_zorder = False
    # TODO: find a way to fix transparency overlap in order to have passive layers in front of bottom active layers.
    passivearrays: List[Any] = []
    activearrays: List[Any] = []

    for layer in volume.layers:
        if isinstance(layer, PassiveLayer):
            lw, thez, size = layer.get_lw_z_size()
            roundedz = np.round(thez.item(), 2)
            # TODO: split these to allow for different alpha values (want: more transparent in front, more opaque in the back)
            rect = [
                [
                    (0, 0, roundedz - size),
                    (0 + lw[0].item(), 0, roundedz - size),
                    (0 + lw[0].item(), 0 + lw[1].item(), roundedz - size),
                    (0, 0 + lw[1].item(), roundedz - size),
                ],
                [(0, 0, roundedz - size), (0 + lw[0].item(), 0, roundedz - size), (0 + lw[0].item(), 0, roundedz), (0, 0, roundedz)],
                [
                    (0, 0 + lw[1].item(), roundedz - size),
                    (0 + lw[0].item(), 0 + lw[1].item(), roundedz - size),
                    (0 + lw[0].item(), 0 + lw[1].item(), roundedz),
                    (0, 0 + lw[1].item(), roundedz),
                ],
                [(0, 0, roundedz - size), (0, 0 + lw[1].item(), roundedz - size), (0, 0 + lw[1].item(), roundedz), (0, 0, roundedz)],
                [
                    (0 + lw[0].item(), 0, roundedz - size),
                    (0 + lw[0].item(), 0 + lw[1].item(), roundedz - size),
                    (0 + lw[0].item(), 0 + lw[1].item(), roundedz),
                    (0 + lw[0].item(), 0, roundedz),
                ],
            ]

            col = "red" if isinstance(layer, SegmentedSigmoidDetectorPanel) else ("blue" if isinstance(layer, PassiveLayer) else "black")

            passivearrays.append([rect, col, roundedz, 1])
            continue

        if isinstance(layer, SegmentedPanelDetectorLayer):
            for i, p in layer.yield_zordered_panels():
                col = "red" if isinstance(p, SegmentedSigmoidDetectorPanel) else ("blue" if isinstance(p, PassiveLayer) else "black")

                if not isinstance(p.xy, Tensor):
                    raise ValueError("Panel xy is not a tensor, for some reason")
                if not isinstance(p.z, Tensor):
                    raise ValueError("Panel z is not a tensor, for some reason")

                n_panels = p.n_panels
                gap_size = p.gap_size.item()
                panel_size = p.get_scaled_xy_span().cpu().detach().numpy() / n_panels

                center_x = p.xy[0].cpu().detach().numpy()
                center_y = p.xy[1].cpu().detach().numpy()

                indices = np.arange(n_panels) - (n_panels - 1) / 2
                centers_x = center_x + indices * (panel_size[0] + gap_size)
                centers_y = center_y + indices * (panel_size[0] + gap_size)

                panel_x_maxs = []
                panel_y_mins = []
                panel_x_mins = []
                panel_y_maxs = []

                for cx in centers_x:
                    start = cx - panel_size[0] / 2
                    end = cx + panel_size[0] / 2
                    panel_x_mins.append(start)
                    panel_x_maxs.append(end)

                for cy in centers_y:
                    start = cy - panel_size[0] / 2
                    end = cy + panel_size[0] / 2
                    panel_y_mins.append(start)
                    panel_y_maxs.append(end)

                for i in range(n_panels):
                    for j in range(n_panels):
                        x_min = panel_x_mins[i]
                        x_max = panel_x_maxs[i]
                        y_min = panel_y_mins[j]
                        y_max = panel_y_maxs[j]

                        rect = [
                            [
                                [x_min, y_min, p.z.data[0].item()],
                                [x_max, y_min, p.z.data[0].item()],
                                [x_max, y_max, p.z.data[0].item()],
                                [x_min, y_max, p.z.data[0].item()],
                                [x_min, y_min, p.z.data[0].item()],
                            ]
                        ]
                        activearrays.append([rect, col, p.z.data[0].item(), 0.2])
        else:
            raise TypeError("Volume.draw does not yet support layers of type", type(layer))

    allarrays = activearrays + passivearrays

    for voxelandcolour in allarrays:
        ax.add_collection3d(
            Poly3DCollection(
                voxelandcolour[0],
                facecolors=voxelandcolour[1],
                linewidths=1,
                edgecolors=voxelandcolour[1],
                alpha=voxelandcolour[3],
                zorder=voxelandcolour[2],
                sort_zpos=voxelandcolour[2],
            )
        )
    plt.ylim(xlim)
    plt.xlim(ylim)
    ax.set_zlim(zlim)

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    ax.set_zlabel("z [m]")

    plt.title("Volume layers")

    red_patch = mpatches.Patch(color="red", label="Active Detector Layers")
    pink_patch = mpatches.Patch(color="blue", label="Passive Layers")

    ax.legend(handles=[red_patch, pink_patch])

    ax.view_init(azim=rotate, elev=rotate)

    if filename is not None:
        plt.savefig(filename)
        print(f"Saved figure to {filename}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def set_plot_style(title_size: int = 18, label_size: int = 18, tick_size: int = 16, legend_size: int = 16) -> None:
    plt.rcParams.update(
        {
            "axes.titlesize": title_size,  # Title font size
            "axes.labelsize": label_size,  # X and Y axis label font size
            "xtick.labelsize": tick_size,  # X tick font size
            "ytick.labelsize": tick_size,  # Y tick font size
            "legend.fontsize": legend_size,  # Legend font size
        }
    )
