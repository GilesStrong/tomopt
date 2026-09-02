import itertools
import os
import sys
from functools import partial
from typing import Any, Callable, List, Tuple, Union

import h5py
import numpy as np
import torch
from mypy_extensions import NamedArg
from torch import Tensor

from tomopt.core import X0
from tomopt.optimisation.data import PassiveYielder
from tomopt.optimisation.loss import VoxelX0Loss
from tomopt.optimisation.wrapper.volume_wrapper import (
    PanelVolumeWrapper,
    SegmentedPanelVolumeWrapper,
)
from tomopt.volume import Volume

from .pred_callbacks import PredHitRecord, PredPocaRecord
from .volume import get_layers

sys.path.append("/home/ucl/cp3/zdaher/")

OUTPUT_DIR = "datasets/"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "dataset_test.h5",
)

# ============================================================
# Configs
# ============================================================

# materials to be assigned to the small block in the VOI
# the rest of the VOI is air, with steel borders all around
MATERIALS = ["uranium", "lead", "iron"]

# number of scan repetitions per material/position combination
N_REPETITIONS = 1

# Generated muons per volume
N_MU_PER_VOLUME = 50000

# number of muons in a muon batch
MU_BATCH_SIZE = 5000

# length and width of the material VOI in meters
LW = torch.tensor([1.0, 1.0])

# voxel size in meters
VOXEL_SIZE = 0.1

# steel border thickness in voxels
BORDER_THICKNESS = 1

# number of voxels in the material block in each of XY dimension
BLOCK_SIZE = 2

# detector type: "segmented" or "panel"
detector_type = "segmented"

# number of panels on either side of the VOI (top/bottom)
N_PANELS = 3

# separation between consecutive panels ina detector layer in meters
PANEL_Z_SPACING = 0.1

# initial gap between segments of a segmented detector panel in meters
# wil not be used if type of detector panel is "panel" (i.e. not segmented)
INIT_GAP = 0.1

# resolution of the detector panels in meters
INIT_RES = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Defining the material VOI
# ============================================================


def mat_block_rad_length(
    mat: str,
    x_idx: int,
    y_idx: int,
    lw: torch.Tensor,
    size: float,
) -> Callable[[NamedArg(Tensor, "z"), NamedArg(Tensor, "lw"), NamedArg(float, "size")], Tensor]:  # noqa F821
    """
    Defines the radiation length function for a VOI with a small block of
    a given material at a given position, surrounded by air and steel borders.

    Parameters
    ----------
    mat : str
        Material name for the small block in the VOI.

    x_idx : int
        X index of the small block in the VOI grid.

    y_idx : int
        Y index of the small block in the VOI grid.

    lw : torch.Tensor
        Length and width of the VOI in meters.

    size : float
        Voxel size in meters.

    Returns
    -------
    callable
        A function that takes a z position and returns the radiation length
        grid for the VOI at that z position.
    """

    def rad_length(
        *,
        z: Tensor,
        lw: Tensor = lw,
        size: float = size,
    ) -> torch.Tensor:
        """
        Returns the radiation length grid of shape (lw/size) for the VOI at a given z position.
        """

        grid_shape = (lw / size).long().tolist()

        # Start with air
        rad_length = torch.full(
            grid_shape,
            X0["air"],
        )

        # Steel borders in XY
        border_thickness = BORDER_THICKNESS
        rad_length[:border_thickness, :] = X0["steel"]
        rad_length[-border_thickness:, :] = X0["steel"]
        rad_length[:, :border_thickness] = X0["steel"]
        rad_length[:, -border_thickness:] = X0["steel"]

        # Full steel layers at top/bottom of the VOI
        if z > 0.3 and z <= 0.4:
            rad_length[:, :] = X0["steel"]

        if z > 0.6 and z <= 0.7:
            rad_length[:, :] = X0["steel"]

        # Target block
        if z > 0.4 and z <= 0.5:
            rad_length[
                x_idx : x_idx + BLOCK_SIZE,
                y_idx : y_idx + BLOCK_SIZE,
            ] = X0[mat]

        return rad_length

    return rad_length


# ============================================================
# Feature builder
# ============================================================


def build_features(
    hr: PredHitRecord,
    poca_rec: PredPocaRecord,
    preds: List[Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]],
) -> Tuple[Any, np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """
    Construct the per-muon feature array.

    Features:

        0-2     PoCA xyz
        3-5     PoCA xyz uncertainty
        6       theta
        7       theta uncertainty
        8-16    3 upper hits, xyz each
        17-25   3 lower hits, xyz each

    Total = 26 features.
    """

    def to_np(lst: List[Tensor]) -> List[np.ndarray]:

        if len(lst) == 0:
            return []

        if isinstance(
            lst[0],
            torch.Tensor,
        ):
            return [x.detach().cpu().numpy() for x in lst]

        return lst

    # ========================================================
    # PoCA
    # ========================================================

    poca_xyz: np.ndarray = np.array(
        to_np(poca_rec.poca_xyz_batch),
        dtype=np.float32,
    )

    poca_xyz_unc: np.ndarray = np.array(
        to_np(poca_rec.poca_xyz_unc_batch),
        dtype=np.float32,
    )

    theta: np.ndarray = np.array(
        to_np(poca_rec.poca_theta_mcs_batch),
        dtype=np.float32,
    )

    theta_unc: np.ndarray = np.array(
        to_np(poca_rec.poca_theta_mcs_unc_batch),
        dtype=np.float32,
    )

    # ========================================================
    # Hits
    # ========================================================

    hits_above: np.ndarray = np.array(
        to_np([hr.split_above_below(h)[0] for h in hr.reco_hits_batch]),
        dtype=np.float32,
    )

    hits_below: np.ndarray = np.array(
        to_np([hr.split_above_below(h)[1] for h in hr.reco_hits_batch]),
        dtype=np.float32,
    )

    # ========================================================
    # Expected shapes
    #
    # PoCA:
    #   (1, N, 3)
    #
    # Hits:
    #   (1, N, 3, 3)
    # ========================================================

    if poca_xyz.ndim != 3:
        raise ValueError(f"Unexpected PoCA shape: " f"{poca_xyz.shape}")

    if hits_above.ndim != 4:
        raise ValueError(f"Unexpected upper-hit shape: " f"{hits_above.shape}")

    if hits_below.ndim != 4:
        raise ValueError(f"Unexpected lower-hit shape: " f"{hits_below.shape}")

    # ========================================================
    # Remove callback batch dimension
    # ========================================================

    poca_xyz = poca_xyz[0]
    poca_xyz_unc = poca_xyz_unc[0]

    theta = theta[0]
    theta_unc = theta_unc[0]

    hits_above = hits_above[0]
    hits_below = hits_below[0]

    # ========================================================
    # Number of reconstructed muons
    # ========================================================

    n_muons = poca_xyz.shape[0]

    # ========================================================
    # Flatten hits per muon
    #
    # (N, 3 panels, 3 xyz) -> N, 9)
    # ========================================================

    hits_above = hits_above.reshape(
        n_muons,
        9,
    )

    hits_below = hits_below.reshape(
        n_muons,
        9,
    )

    # ========================================================
    # PoCA features
    # ========================================================

    poca_xyz = poca_xyz.reshape(
        n_muons,
        3,
    )

    poca_xyz_unc = poca_xyz_unc.reshape(
        n_muons,
        3,
    )

    theta = theta.reshape(
        n_muons,
        1,
    )

    theta_unc = theta_unc.reshape(
        n_muons,
        1,
    )

    # ========================================================
    # Concatenate
    # ========================================================

    features = np.concatenate(
        [
            poca_xyz,
            poca_xyz_unc,
            theta,
            theta_unc,
            hits_above,
            hits_below,
        ],
        axis=1,
    ).astype(np.float32)

    # ========================================================
    # Ground truth / predictions
    # ========================================================

    gt: np.ndarray = np.array(
        [p[1].flatten() for p in preds],
        dtype=np.float32,
    )

    preds_arr: np.ndarray = np.array(
        [p[0].flatten() for p in preds],
        dtype=np.float32,
    )

    # ========================================================
    # Sanity checks
    # ========================================================

    if features.ndim != 2:
        raise ValueError(f"Expected features to be 2D, " f"got {features.shape}")

    if features.shape[1] != 26:
        raise ValueError(f"Expected 26 features per muon, " f"got {features.shape}")

    # preds/gt are per-volume, NOT per-muon.
    if len(preds) != 1:
        raise ValueError(f"Expected one prediction for one volume, " f"got {len(preds)}")

    print(f"     features = {features.shape}")
    print(f"     gt       = {gt.shape}")
    print(f"     preds    = {preds_arr.shape}")

    return features, gt, preds_arr


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    # ========================================================
    # Position grid
    # ========================================================

    grid_size = (LW / VOXEL_SIZE).long()

    valid_range_x = range(
        BORDER_THICKNESS,
        grid_size[0] - BORDER_THICKNESS - BLOCK_SIZE + 1,
    )

    valid_range_y = range(
        BORDER_THICKNESS,
        grid_size[1] - BORDER_THICKNESS - BLOCK_SIZE + 1,
    )

    positions = list(
        itertools.product(
            valid_range_x,
            valid_range_y,
        )
    )

    n_positions = len(positions)

    # ========================================================
    # Dataset size
    # ========================================================

    total_scans = len(MATERIALS) * n_positions * N_REPETITIONS

    print()
    print("=" * 60)
    print("Dataset configuration")
    print("=" * 60)
    print(f"Materials       : {len(MATERIALS)}")
    print(f"Positions       : {n_positions}")
    print(f"Repetitions     : {N_REPETITIONS}")
    print(f"Muons/volume    : {N_MU_PER_VOLUME}")
    print(f"Muon batch      : {MU_BATCH_SIZE}")
    print(f"Total scans     : {total_scans}")
    print(f"Output          : {OUTPUT_FILE}")
    print("=" * 60)
    print()

    # ========================================================
    # EXACT ORIGINAL DETECTOR
    # ========================================================

    print("Creating original segmented-panel detector...")

    volume = Volume(
        get_layers(
            panel_z_spacing=PANEL_Z_SPACING,
            n_panels=N_PANELS,
            init_gap=INIT_GAP,
            detector_type=detector_type,
            init_res=INIT_RES,
        )
    )

    # ========================================================
    # EXACT ORIGINAL WRAPPER
    # ========================================================

    wrapper: Union[PanelVolumeWrapper, SegmentedPanelVolumeWrapper] = None

    if detector_type == "segmented":
        wrapper = SegmentedPanelVolumeWrapper(
            volume,
            xy_pos_opt=partial(
                torch.optim.SGD,
                lr=1e3,
            ),
            xy_span_opt=partial(
                torch.optim.SGD,
                lr=1e3,
            ),
            z_pos_opt=partial(
                torch.optim.SGD,
                lr=1e3,
            ),
            gap_opt=partial(
                torch.optim.SGD,
                lr=1e3,
            ),
            loss_func=VoxelX0Loss(
                target_budget=None,
                cost_coef=None,
            ),
        )

    elif detector_type == "panel":
        wrapper = PanelVolumeWrapper(
            volume,
            xy_pos_opt=partial(
                torch.optim.SGD,
                lr=1e3,
            ),
            xy_span_opt=partial(
                torch.optim.SGD,
                lr=1e3,
            ),
            z_pos_opt=partial(
                torch.optim.SGD,
                lr=1e3,
            ),
            loss_func=VoxelX0Loss(
                target_budget=None,
                cost_coef=None,
            ),
        )

    # ========================================================
    # Open SINGLE HDF5
    # ========================================================

    with h5py.File(
        OUTPUT_FILE,
        "a",
    ) as h5f:

        # ----------------------------------------------------
        # Main group
        # ----------------------------------------------------

        if "data" not in h5f:

            data_group = h5f.create_group("data")

        else:

            data_group = h5f["data"]

        # ----------------------------------------------------
        # File-level metadata
        # ----------------------------------------------------

        h5f.attrs["n_materials"] = len(MATERIALS)

        h5f.attrs["n_positions"] = n_positions

        h5f.attrs["n_repetitions"] = N_REPETITIONS

        h5f.attrs["n_mu_per_volume"] = N_MU_PER_VOLUME

        h5f.attrs["mu_batch_size"] = MU_BATCH_SIZE

        h5f.attrs["n_panels"] = N_PANELS

        h5f.attrs["panel_z_spacing"] = PANEL_Z_SPACING

        h5f.attrs["init_gap"] = INIT_GAP

        # ----------------------------------------------------
        # Existing scans
        # ----------------------------------------------------

        existing_scans = len(data_group)

        print(f"Existing scans: " f"{existing_scans}")

        # ====================================================
        # Generation loop
        # ====================================================

        for mat_id, mat in enumerate(MATERIALS):

            for pos_id, (
                x,
                y,
            ) in enumerate(positions):

                # ------------------------------------------------
                # Material/position geometry
                # ------------------------------------------------

                rad_func = mat_block_rad_length(
                    mat=mat,
                    x_idx=x,
                    y_idx=y,
                    lw=LW,
                    size=VOXEL_SIZE,
                )

                # ------------------------------------------------
                # Repetitions
                # ------------------------------------------------

                for rep in range(N_REPETITIONS):

                    # ------------------------------------------------
                    # Deterministic global scan index
                    #
                    # Same ordering as your original ROOT generation:
                    #
                    # material
                    #     position x
                    #         position y
                    #             repetition
                    # ------------------------------------------------

                    global_scan_idx = (mat_id * n_positions + pos_id) * N_REPETITIONS + rep

                    scan_name = f"scan_" f"{global_scan_idx:06d}"

                    # ------------------------------------------------
                    # Resume
                    # ------------------------------------------------

                    if scan_name in data_group:

                        print(f"[SKIP] " f"{scan_name} " f"already exists")

                        continue

                    print()
                    print(f"[{global_scan_idx + 1}/" f"{total_scans}] " f"{scan_name}")

                    print(f"  material   = " f"{mat} ({mat_id})")

                    print(f"  position   = " f"{pos_id} " f"(x={x}, y={y})")

                    print(f"  repetition = " f"{rep}")

                    # ------------------------------------------------
                    # Passive volume
                    # ------------------------------------------------

                    passives = PassiveYielder([rad_func])

                    # ------------------------------------------------
                    # Callbacks
                    # ------------------------------------------------

                    hr = PredHitRecord()

                    poca_rec = PredPocaRecord()

                    # ------------------------------------------------
                    # Generate data
                    #
                    # EXACTLY matching original:
                    #
                    # n_mu_per_volume=50000
                    # mu_bs=5000
                    # ------------------------------------------------

                    preds = wrapper.predict(
                        passives,
                        n_mu_per_volume=(N_MU_PER_VOLUME),
                        mu_bs=MU_BATCH_SIZE,
                        cbs=[
                            hr,
                            poca_rec,
                        ],
                    )

                    # ------------------------------------------------
                    # Convert to fixed feature representation
                    # ------------------------------------------------

                    (
                        features,
                        gt,
                        preds_arr,
                    ) = build_features(
                        hr,
                        poca_rec,
                        preds,
                    )

                    # ------------------------------------------------
                    # Create scan
                    # ------------------------------------------------

                    scan_grp = data_group.create_group(scan_name)

                    # ------------------------------------------------
                    # Store arrays
                    # ------------------------------------------------

                    scan_grp.create_dataset(
                        "features",
                        data=features,
                        compression="gzip",
                    )

                    scan_grp.create_dataset(
                        "gt",
                        data=gt,
                        compression="gzip",
                    )

                    scan_grp.create_dataset(
                        "preds",
                        data=preds_arr,
                        compression="gzip",
                    )

                    # ------------------------------------------------
                    # Store metadata
                    # ------------------------------------------------

                    scan_grp.attrs["material_id"] = mat_id

                    scan_grp.attrs["material_name"] = mat

                    scan_grp.attrs["position"] = pos_id

                    scan_grp.attrs["position_x"] = x

                    scan_grp.attrs["position_y"] = y

                    scan_grp.attrs["repetition"] = rep

                    # ------------------------------------------------
                    # Flush after every scan
                    #
                    # Safer for a long 50k-muon generation.
                    # ------------------------------------------------

                    h5f.flush()

                    print(f"  saved " f"{scan_name}")

        # ========================================================
        # Final flush
        # ========================================================

        h5f.flush()

    # ============================================================
    # Finished
    # ============================================================

    print()
    print("=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Expected scans: {total_scans}")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
