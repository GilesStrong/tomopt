"""
Volume wrapper for co-design optimization of segmented detector panels.

This module provides a specialized wrapper class for optimizing detector geometries
with segmented panels, including position, span, gap, and software parameters.
Integrates with hypothesis testing for statistical power optimization.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Type

import matplotlib.pyplot as plt
from fastprogress import master_bar, progress_bar

# tomopt imports
from tomopt.core import PartialOpt
from tomopt.inference import AbsVolumeInferrer, PanelX0Inferrer, ScatterBatch
from tomopt.muon import AbsMuonGenerator, MuonBatch
from tomopt.optimisation.data import PassiveYielder
from tomopt.optimisation.loss.loss import AbsDetectorLoss
from tomopt.optimisation.wrapper import AbsVolumeWrapper
from tomopt.volume import Volume

# Segmented panel imports
# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from diagnostics import GradientDebugPlotter  # type: ignore[import] # noqa: E402
from layer import SegmentedPanelDetectorLayer  # type: ignore[import] # noqa: E402

__all__ = ["CoDesignVolumeWrapper"]


class CoDesignVolumeWrapper(AbsVolumeWrapper):
    """
    Volume wrapper for co-design optimization of segmented detector panels.

    Extends the base volume wrapper to support optimization of detector hardware
    parameters (position, span, gaps) alongside software parameters (smoothing).
    Designed to work with hypothesis testing loss functions that optimize
    statistical detection power.

    The wrapper manages:
    - Panel position optimization (xy and z coordinates)
    - Panel span optimization (detector size)
    - Gap size optimization between panels
    - Software smoothing parameter optimization
    - Optional budget allocation optimization
    - Training history tracking for diagnostics

    Args:
        volume: Volume containing detectors to be optimized
        xy_pos_opt: Optimizer for panel xy positions
        z_pos_opt: Optimizer for panel z positions
        xy_span_opt: Optional optimizer for panel xy spans
        gap_opt: Optimizer for gap sizes between panels
        log_sigma_sw_opt: Optional optimizer for log software smoothing parameter
        budget_opt: Optional optimizer for budget allocation weights
        loss_func: Loss function for optimization (required for training)
        partial_scatter_inferrer: Class for inferring muon scatter variables
        partial_volume_inferrer: Class for inferring volume targets
        mu_generator: Optional muon generator (defaults to MuonGenerator2016)

    Attributes:
        background_bs: Batch size for background volumes
        history: Dictionary tracking training metrics per epoch:
            - train_loss: Training loss values
            - val_loss: Validation loss values
            - log_sigma_sw: Log software smoothing parameter
            - sigma_hw: Hardware angular resolution
    """

    # def __init__(
    #     self,
    #     volume: Volume,
    #     *,
    #     xy_pos_opt: PartialOpt,
    #     z_pos_opt: PartialOpt,
    #     xy_span_opt: Optional[PartialOpt] = None,
    #     gap_opt: PartialOpt,
    #     log_sigma_sw_opt: Optional[PartialOpt],
    #     budget_opt: Optional[PartialOpt] = None,
    #     loss_func: Optional[AbsDetectorLoss] = None,
    #     partial_scatter_inferrer: Type[ScatterBatch] = ScatterBatch,
    #     partial_volume_inferrer: Type[AbsVolumeInferrer] = PanelX0Inferrer,
    #     mu_generator: Optional[AbsMuonGenerator] = None,
    # ) -> None:
    #     """Initialize co-design volume wrapper with optimizers and settings."""
    #     super().__init__(
    #         volume=volume,
    #         partial_opts={
    #             "xy_pos_opt": xy_pos_opt,
    #             "z_pos_opt": z_pos_opt,
    #             "xy_span_opt": xy_span_opt,
    #             "gap_opt": gap_opt,
    #             "log_sigma_sw_opt": log_sigma_sw_opt,
    #             "budget_opt": budget_opt,
    #         },
    #         loss_func=loss_func,
    #         mu_generator=mu_generator,
    #         partial_scatter_inferrer=partial_scatter_inferrer,
    #         partial_volume_inferrer=partial_volume_inferrer,
    #     )

    #     self.debug_plotter = GradientDebugPlotter(save_path="grad_debug_plots")
    #     # Training history for diagnostics
    #     self.history: Dict[str, List[Optional[float]]] = {
    #     "train_loss": [],
    #     "val_loss": [],
    #     "log_sigma_sw": [],
    #     "sigma_hw": [],
    # }

    # Modified __init__ signature (remove log_sigma_sw_opt parameter)
    def __init__(
        self,
        volume: Volume,
        *,
        xy_pos_opt: PartialOpt,
        z_pos_opt: PartialOpt,
        xy_span_opt: Optional[PartialOpt] = None,
        gap_opt: PartialOpt,
        budget_opt: Optional[PartialOpt] = None,
        loss_func: Optional[AbsDetectorLoss] = None,
        partial_scatter_inferrer: Type[ScatterBatch] = ScatterBatch,
        partial_volume_inferrer: Type[AbsVolumeInferrer] = PanelX0Inferrer,
        mu_generator: Optional[AbsMuonGenerator] = None,
    ) -> None:
        """
        Initialize co-design volume wrapper.

        NOTE: log_sigma_sw_opt parameter REMOVED.
        Software parameter now optimized internally by loss function.
        """
        super().__init__(
            volume=volume,
            partial_opts={
                "xy_pos_opt": xy_pos_opt,
                "z_pos_opt": z_pos_opt,
                "xy_span_opt": xy_span_opt,
                "gap_opt": gap_opt,
                "budget_opt": budget_opt,
                # log_sigma_sw_opt REMOVED
            },
            loss_func=loss_func,
            mu_generator=mu_generator,
            partial_scatter_inferrer=partial_scatter_inferrer,
            partial_volume_inferrer=partial_volume_inferrer,
        )

        if hasattr(self, "debug_plotter"):
            self.debug_plotter = GradientDebugPlotter(save_path="grad_debug_plots")

        self.debug = True  # Enable debug output

        # Training history for diagnostics
        self.history: Dict[str, List[Optional[float]]] = {
            "train_loss": [],
            "val_loss": [],
            "log_sigma_sw": [],
            "sigma_hw": [],
        }

    @classmethod
    def from_save(
        cls,
        name: str,
        *,
        volume: Volume,
        xy_pos_opt: PartialOpt,
        z_pos_opt: PartialOpt,
        xy_span_opt: Optional[PartialOpt] = None,
        gap_opt: Optional[PartialOpt] = None,
        # log_sigma_sw_opt: Optional[PartialOpt] = None,
        budget_opt: Optional[PartialOpt] = None,
        loss_func: Optional[AbsDetectorLoss] = None,
        partial_scatter_inferrer: Type[ScatterBatch] = ScatterBatch,
        partial_volume_inferrer: Type[AbsVolumeInferrer] = PanelX0Inferrer,
        mu_generator: Optional[AbsMuonGenerator] = None,
    ) -> CoDesignVolumeWrapper:
        """
        Create wrapper instance and load saved detector and optimizer parameters.

        Instantiates a new CoDesignVolumeWrapper and loads previously saved
        detector configurations and optimizer states from file.

        Args:
            name: Filename containing saved parameters
            volume: Volume with detectors to be optimized
            xy_pos_opt: Uninitialized optimizer for panel xy positions
            z_pos_opt: Uninitialized optimizer for panel z positions
            xy_span_opt: Optional uninitialized optimizer for panel xy spans
            gap_opt: Optional uninitialized optimizer for gap sizes
            log_sigma_sw_opt: Optional uninitialized optimizer for log smoothing parameter
            budget_opt: Optional uninitialized optimizer for budget allocation
            loss_func: Optional loss function (required for optimization)
            partial_scatter_inferrer: Uninitialized class for scatter inference
            partial_volume_inferrer: Uninitialized class for volume inference
            mu_generator: Optional muon generator (defaults to MuonGenerator2016)

        Returns:
            Initialized wrapper with loaded parameters
        """
        vw = cls(
            volume=volume,
            xy_pos_opt=xy_pos_opt,
            z_pos_opt=z_pos_opt,
            xy_span_opt=xy_span_opt,
            gap_opt=gap_opt,
            # log_sigma_sw_opt=log_sigma_sw_opt,
            budget_opt=budget_opt,
            loss_func=loss_func,
            partial_scatter_inferrer=partial_scatter_inferrer,
            partial_volume_inferrer=partial_volume_inferrer,
            mu_generator=mu_generator,
        )
        vw.load(name)
        return vw

    def _scan_volumes(self, passives: PassiveYielder) -> None:
        """
        Scan all volumes and perform optimization updates.

        Processes volumes in batches, computing losses and updating detector
        parameters. For hypothesis testing, volumes are processed in two phases:
        background and signal. After each complete batch, the loss is computed
        and parameters are updated if in training mode.

        The method handles three distinct phases:
        1. Background phase: Collects background scatter distributions
        2. Signal phase: Collects signal scatter distributions
        3. Optimization: Computes loss and updates parameters

        Args:
            passives: Yielder providing passive volumes and targets

        Note:
            In training mode, incomplete batches at the end are skipped to
            ensure consistent batch statistics. In test mode, all volumes
            are processed regardless of batch boundaries.
        """
        if self.fit_params.state == "test":
            self.fit_params.passive_bar = master_bar(passives)

        iterator = self.fit_params.passive_bar if self.fit_params.state == "test" else passives

        for i, (passive, target) in enumerate(iterator):
            self.fit_params.volume_id = i

            # Volume batch initialization
            if self.fit_params.state != "test" and (i == 0):
                self.fit_params.loss_val = None
                for c in self.fit_params.cbs:
                    c.on_volume_batch_begin()

            # Load and scan volume
            self.volume.load_rad_length(passive, target)
            for c in self.fit_params.cbs:
                c.on_volume_begin()
            self._scan_volume()
            for c in self.fit_params.cbs:
                c.on_volume_end()

            # Handle phase transition (background -> signal)
            batch_halfway = (i + 1) % (self.fit_params.passive_bs / 2) == 0
            not_batch_end = (i + 1) % self.fit_params.passive_bs != 0

            if self.fit_params.state != "test" and batch_halfway and not_batch_end:
                for c in self.fit_params.cbs:
                    if hasattr(c, "is_signal_phase"):
                        c.is_signal_phase = not c.is_signal_phase
                        c.on_volume_batch_begin()

            # Process complete batch
            if self.fit_params.state != "test" and (i + 1) % self.fit_params.passive_bs == 0:
                self._process_batch_end(i, len(passives))

                # Exit if no more complete batches remain
                if len(passives) - (i + 1) == 0:
                    print("Exiting training epoch")
                    break

    # def _process_batch_end(self, current_idx: int, total_volumes: int) -> None:
    #     """
    #     Process end of volume batch: compute loss and update parameters.

    #     Called when a complete batch has been processed. Triggers hypothesis
    #     test callbacks, computes loss, performs backpropagation (if training),
    #     and updates detector parameters.

    #     Args:
    #         current_idx: Index of current volume in epoch
    #         total_volumes: Total number of volumes in epoch

    #     Note:
    #         Accumulates loss across batches and averages at end of epoch.
    #         Only performs parameter updates in training mode.
    #     """
    #     # Trigger hypothesis test data collection
    #     for c in self.fit_params.cbs:
    #         if hasattr(c, "is_signal_phase"):
    #             print(current_idx)
    #             c.on_volume_batch_end_hypothesis()  # type: ignore[attr-defined]

    #     # Compute loss for current batch
    #     loss = self.loss_func()

    #     # Accumulate loss
    #     if self.fit_params.loss_val is None:
    #         self.fit_params.loss_val = loss
    #         print(f"Epoch {self.fit_params.epoch} Volume: {current_idx}, " f"pre-summed loss, state={self.fit_params.state}")
    #     else:
    #         self.fit_params.loss_val = self.fit_params.loss_val + loss
    #         print(f"Epoch {self.fit_params.epoch} Volume: {current_idx}, " f"summed loss, state={self.fit_params.state}")

    #     # Finalize and update at end of epoch
    #     is_epoch_end = (current_idx + 1) == total_volumes
    #     is_training_mode = self.fit_params.state != "test"
    #     has_loss_func = self.loss_func is not None

    #     if is_training_mode and has_loss_func and is_epoch_end:
    #         self._finalize_epoch(current_idx, total_volumes)

    def _finalize_epoch(self, current_idx: int, total_volumes: int) -> None:
        """
        Finalize epoch: average loss and perform parameter updates.

        Computes mean loss across all batches, performs backpropagation
        (if training), updates parameters, and records metrics to history.

        Args:
            current_idx: Index of current volume
            total_volumes: Total number of volumes in epoch
        """
        # Compute mean loss across batches
        if self.fit_params.loss_val is not None:
            n_batches = total_volumes / self.fit_params.passive_bs
            self.fit_params.mean_loss = self.fit_params.loss_val / n_batches
            print(f"Epoch {self.fit_params.epoch} Volume: {current_idx}: " f"Loss = {self.fit_params.mean_loss.item():.4f}, " f"state={self.fit_params.state}")
        else:
            self.fit_params.mean_loss = None

        self.fit_params.loss_val = None

        if self.fit_params.state == "train":
            self._perform_training_update(current_idx)
        elif self.fit_params.state == "valid":
            self._record_validation_metrics()

    # def _perform_training_update(self, current_idx: int) -> None:
    #     """
    #     Perform training update: backpropagation and parameter optimization.

    #     Executes the complete training update cycle:
    #     1. Zero gradients
    #     2. Trigger pre-backward callbacks
    #     3. Compute gradients via backpropagation
    #     4. Trigger post-backward callbacks (e.g., gradient clipping)
    #     5. Update parameters with optimizer step
    #     6. Conform detector parameters to constraints
    #     7. Record training metrics

    #     Args:
    #         current_idx: Index of current volume (for logging)
    #     """
    #     print(current_idx)

    #     # Zero all optimizer gradients
    #     for o in self.opts.values():
    #         o.zero_grad()

    #     # Pre-backward callbacks
    #     for c in self.fit_params.cbs:
    #         c.on_backwards_begin()

    #     # Compute gradients
    #     if self.fit_params.mean_loss is not None:
    #         self.fit_params.mean_loss.backward()

    #     # Post-backward callbacks (e.g., gradient clipping)
    #     for c in self.fit_params.cbs:
    #         c.on_backwards_end()

    #     self.debug_plotter.update(self.loss_func.debug_tensors, self.fit_params.epoch)
    #     self.debug_plotter.plot(epoch=self.fit_params.epoch)

    #     # Update parameters
    #     if self.fit_params.mean_loss is not None and not self.fit_params.skip_opt_step:
    #         for o in self.opts.values():
    #             o.step()

    #     # Record training metrics
    #     self.history["train_loss"].append(self.fit_params.mean_loss.item() if self.fit_params.mean_loss is not None else None)
    #     self.history["log_sigma_sw"].append(self.loss_func.log_sigma_sw.detach().cpu().item())

    #     # Post-step callbacks
    #     for c in self.fit_params.cbs:
    #         c.on_step_end()

    #     # Ensure detector parameters satisfy constraints
    #     for d in self.volume.get_detectors():
    #         d.conform_detector()

    #     # Record hardware resolution if available
    #     sigma_hw_val = self._extract_sigma_hw()
    #     self.history["sigma_hw"].append(sigma_hw_val)

    def _record_validation_metrics(self) -> None:
        """
        Record validation metrics to history.

        Saves validation loss to history for monitoring convergence
        and detecting overfitting.
        """
        print("=" * 27 + "VALIDATION" + "=" * 27)
        self.history["val_loss"].append(self.fit_params.mean_loss.item() if self.fit_params.mean_loss is not None else None)

    def _extract_sigma_hw(self) -> Optional[float]:
        """
        Extract hardware angular resolution from loss function.

        Safely attempts to extract the hardware resolution parameter
        from the loss function, handling cases where it may not be
        available or accessible.

        Returns:
            Hardware resolution value or None if unavailable
        """
        sigma_hw_val = None
        if getattr(self.loss_func, "sigma_hw", None) is not None:
            try:
                sigma_hw_val = float(self.loss_func.sigma_hw.detach().cpu().item())
            except Exception:
                sigma_hw_val = None
        return sigma_hw_val

    def _scan_volume(self) -> None:
        """
        Pass multiple muon batches through a single volume.

        Generates and propagates muon batches through the current volume,
        computing scatter angles and trajectories. The number of muons
        and batch size are controlled by fit_params.

        For each muon batch:
        1. Generate muons from configured generator
        2. Propagate through volume (detector interactions)
        3. Infer scatter variables and trajectories
        4. Trigger scatter-end callbacks for data collection
        """
        self.fit_params.pred = None

        # Setup progress bar
        n_batches = self.fit_params.n_mu_per_volume // self.fit_params.mu_bs
        if self.fit_params.state != "test":
            muon_bar = progress_bar(range(n_batches), display=False, leave=False)
        else:
            muon_bar = progress_bar(range(n_batches), parent=self.fit_params.passive_bar)

        # Process muon batches
        for _ in muon_bar:
            # Generate muon batch
            self.fit_params.mu = MuonBatch(self.mu_generator(self.fit_params.mu_bs), init_z=self.volume.h, device=self.fit_params.device)

            # Pre-batch callbacks
            for c in self.fit_params.cbs:
                c.on_mu_batch_begin()

            # Propagate muons through volume
            self.volume(self.fit_params.mu)

            # Infer scatter variables
            self.fit_params.sb = self.partial_scatter_inferrer(mu=self.fit_params.mu, volume=self.volume)

            # Post-scatter callbacks (data collection)
            for c in self.fit_params.cbs:
                c.on_scatter_end()

            # Post-batch callbacks
            for c in self.fit_params.cbs:
                c.on_mu_batch_end()

    # def _build_opt(self, **kwargs: PartialOpt) -> None:
    #     """
    #     Initialize optimizers by associating them with detector parameters.

    #     Creates optimizer instances for each optimizable parameter group:
    #     - xy_pos_opt: Panel xy positions
    #     - z_pos_opt: Panel z positions
    #     - xy_span_opt: Panel xy spans (optional)
    #     - gap_opt: Gap sizes between panels
    #     - log_sigma_sw_opt: Log software smoothing parameter
    #     - budget_opt: Budget allocation weights (optional)

    #     Only segmented panel detector layers are included in optimization.

    #     Args:
    #         **kwargs: Uninitialized optimizers passed as keyword arguments

    #     Note:
    #         Optimizers must be provided as PartialOpt objects that accept
    #         parameter iterators and return initialized optimizer instances.
    #     """
    #     # Filter for segmented panel detectors
    #     all_dets = self.volume.get_detectors()
    #     dets: List[SegmentedPanelDetectorLayer] = []
    #     for d in all_dets:
    #         if isinstance(d, SegmentedPanelDetectorLayer):
    #             dets.append(d)

    #     # Initialize optimizers for each parameter group
    #     self.opts = {
    #         "xy_pos_opt": kwargs["xy_pos_opt"]((p.xy for l in dets for p in l.panels)),
    #         "z_pos_opt": kwargs["z_pos_opt"]((p.z for l in dets for p in l.panels)),
    #         "xy_span_opt": kwargs["xy_span_opt"]((p.xy_span for l in dets for p in l.panels)),
    #         "gap_opt": kwargs["gap_opt"]((l.gap_size for l in dets)),
    #         "log_sigma_sw_opt": kwargs["log_sigma_sw_opt"](p for p in [self.loss_func.log_sigma_sw]),
    #     }

    #     # Add budget optimizer if provided
    #     if kwargs["budget_opt"] is not None:
    #        self.opts["budget_opt"] = kwargs["budget_opt"]((p for p in [self.volume.budget_weights]))

    # -------------------------------------------------------------------------
    # Diagnostic Plotting Methods
    # -------------------------------------------------------------------------

    def plot_train_loss(self) -> None:
        """
        Plot training loss evolution over epochs.

        Creates a line plot showing training loss at each epoch.
        Useful for monitoring convergence and detecting training issues.

        Note:
            Prints a message if no training history is available yet.
        """
        if not self.history["train_loss"]:
            print("No training loss history recorded yet.")
            return

        plt.figure(figsize=(6, 4))
        plt.plot(self.history["train_loss"], marker="o", markersize=3)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss per Epoch")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_log_sigma_sw(self) -> None:
        """
        Plot log software smoothing parameter evolution over epochs.

        Creates a line plot showing how the log smoothing parameter
        changes during training. Useful for monitoring software
        parameter optimization.

        Note:
            Prints a message if no history is available yet.
        """
        if not self.history["log_sigma_sw"]:
            print("No log_sigma_sw history recorded yet.")
            return

        plt.figure(figsize=(6, 4))
        plt.plot(self.history["log_sigma_sw"], marker="o", markersize=3)
        plt.xlabel("Epoch")
        plt.ylabel("log(σ_sw)")
        plt.title("Log Software Smoothing per Epoch")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_sigma_hw(self) -> None:
        """
        Plot hardware angular resolution evolution over epochs.

        Creates a line plot showing how the hardware resolution
        (derived from detector geometry) changes during training.
        Useful for monitoring the effect of geometric optimization.

        Note:
            Prints a message if no history is available yet.
        """
        if not self.history["sigma_hw"]:
            print("No sigma_hw history recorded yet.")
            return

        plt.figure(figsize=(6, 4))
        plt.plot(self.history["sigma_hw"], marker="o", markersize=3)
        plt.xlabel("Epoch")
        plt.ylabel("σ_hw")
        plt.title("Hardware Angular Resolution per Epoch")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_all_metrics(self) -> None:
        """
        Plot all tracked metrics in a single figure.

        Creates a multi-panel figure showing:
        - Training and validation loss
        - Log software smoothing parameter
        - Hardware angular resolution

        Provides a comprehensive view of training progress and
        parameter evolution.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Training and validation loss
        if self.history["train_loss"]:
            axes[0, 0].plot(self.history["train_loss"], label="Train", marker="o", markersize=3)
        if self.history["val_loss"]:
            axes[0, 0].plot(self.history["val_loss"], label="Validation", marker="s", markersize=3)
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training & Validation Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # Log software smoothing
        if self.history["log_sigma_sw"]:
            axes[0, 1].plot(self.history["log_sigma_sw"], marker="o", markersize=3, color="C2")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("log(σ_sw)")
        axes[0, 1].set_title("Log Software Smoothing")
        axes[0, 1].grid(alpha=0.3)

        # Hardware resolution
        if self.history["sigma_hw"]:
            axes[1, 0].plot(self.history["sigma_hw"], marker="o", markersize=3, color="C3")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("σ_hw")
        axes[1, 0].set_title("Hardware Angular Resolution")
        axes[1, 0].grid(alpha=0.3)

        # Hide unused subplot
        axes[1, 1].axis("off")

        plt.tight_layout()
        plt.show()

        """
    Modified wrapper to handle separated hardware/software optimization.

    Key changes:
    - Removed log_sigma_sw_opt from wrapper (now handled in loss inner loop)
    - Added cache invalidation before hardware updates
    - Hardware optimization in outer loop, software in inner loop
    """

    def _build_opt(self, **kwargs: PartialOpt) -> None:
        """
        Initialize optimizers - SOFTWARE PARAMETER EXCLUDED.

        log_sigma_sw is now optimized internally by the loss function,
        so we only create optimizers for hardware parameters here.
        """
        # Filter for segmented panel detectors
        all_dets = self.volume.get_detectors()
        dets = []
        for d in all_dets:
            if isinstance(d, SegmentedPanelDetectorLayer):
                dets.append(d)

        # Initialize optimizers for HARDWARE parameters only
        self.opts = {
            "xy_pos_opt": kwargs["xy_pos_opt"]((p.xy for l in dets for p in l.panels)),
            "z_pos_opt": kwargs["z_pos_opt"]((p.z for l in dets for p in l.panels)),
            "xy_span_opt": kwargs["xy_span_opt"]((p.xy_span for l in dets for p in l.panels)),
            "gap_opt": kwargs["gap_opt"]((l.gap_size for l in dets)),
        }

        # NOTE: log_sigma_sw_opt is NO LONGER HERE
        # Software parameter optimized internally by loss function

        # Add budget optimizer if provided
        if kwargs.get("budget_opt") is not None:
            self.opts["budget_opt"] = kwargs["budget_opt"]((p for p in [self.volume.budget_weights]))

    def _perform_training_update(self, current_idx: int) -> None:
        """
        Perform training update with separated hardware/software optimization.

        Modified flow:
        1. Loss function has already optimized software in inner loop
        2. The returned loss has gradients to hardware parameters
        3. We optimize hardware parameters
        4. Invalidate cache for next iteration
        """
        print(f"[Outer Loop] Epoch {self.fit_params.epoch}, Volume {current_idx}")

        # The loss already contains:
        # - Optimized software parameter (from inner loop)
        # - Fresh computation graph with hardware gradients

        # Zero hardware optimizer gradients
        for o in self.opts.values():
            o.zero_grad()

        # Pre-backward callbacks
        for c in self.fit_params.cbs:
            c.on_backwards_begin()

        # Compute gradients for HARDWARE parameters
        if self.fit_params.mean_loss is not None:
            self.fit_params.mean_loss.backward()

            if self.debug:
                print(" [Outer Loop] Computed hardware gradients")

        # Post-backward callbacks (e.g., gradient clipping)
        for c in self.fit_params.cbs:
            c.on_backwards_end()

        # Debug plotting
        if hasattr(self, "debug_plotter"):
            self.debug_plotter.update(self.loss_func.debug_tensors, self.fit_params.epoch)
            self.debug_plotter.plot(epoch=self.fit_params.epoch)

        # Update HARDWARE parameters
        if self.fit_params.mean_loss is not None and not self.fit_params.skip_opt_step:
            for opt_name, o in self.opts.items():
                o.step()
                if self.debug:
                    print(f" [Outer Loop] Updated {opt_name}")

            # CRITICAL: Invalidate cache after hardware update
            self.loss_func.invalidate_cache()
            if self.debug:
                print("  [Outer Loop] Cache invalidated - new samples needed")

        # Record training metrics
        self.history["train_loss"].append(self.fit_params.mean_loss.item() if self.fit_params.mean_loss is not None else None)
        self.history["log_sigma_sw"].append(self.loss_func.log_sigma_sw.detach().cpu().item())

        # Post-step callbacks
        for c in self.fit_params.cbs:
            c.on_step_end()

        # Ensure detector parameters satisfy constraints
        for d in self.volume.get_detectors():
            d.conform_detector()

        # Record hardware resolution
        sigma_hw_val = self._extract_sigma_hw()
        self.history["sigma_hw"].append(sigma_hw_val)

        if self.debug:
            print("[Outer Loop] Hardware update complete\n")

    def _process_batch_end(self, current_idx: int, total_volumes: int) -> None:
        """
        Process end of volume batch.

        Modified to work with inner-outer loop structure:
        - Scatter data is collected and set in loss function
        - Loss function performs inner loop optimization
        - Wrapper performs outer loop optimization
        """
        # Trigger hypothesis test data collection
        for c in self.fit_params.cbs:
            if hasattr(c, "is_signal_phase"):
                c.on_volume_batch_end_hypothesis()  # type: ignore[attr-defined]

        # At this point, scatter data has been set in the loss function
        # The loss function's forward() will:
        # 1. Cache the scatter samples
        # 2. Run inner loop to optimize sigma_sw
        # 3. Return the final converged loss

        if self.debug:
            print(f"\n{'='*70}")
            print(f"Batch {current_idx} complete - computing loss...")
            print(f"{'='*70}")

        # Compute loss (triggers inner loop optimization)
        loss = self.loss_func()

        # Accumulate loss
        if self.fit_params.loss_val is None:
            self.fit_params.loss_val = loss
            if self.debug:
                print(f"[Batch] First batch loss: {loss.item():.6f}")
        else:
            self.fit_params.loss_val = self.fit_params.loss_val + loss
            if self.debug:
                print(f"[Batch] Accumulated loss: {self.fit_params.loss_val.item():.6f}")

        # Finalize and update at end of epoch
        is_epoch_end = (current_idx + 1) == total_volumes
        is_training_mode = self.fit_params.state != "test"
        has_loss_func = self.loss_func is not None

        if is_training_mode and has_loss_func and is_epoch_end:
            self._finalize_epoch(current_idx, total_volumes)
