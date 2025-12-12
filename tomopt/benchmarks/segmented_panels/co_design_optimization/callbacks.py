"""
Callbacks for hypothesis testing and learning rate scheduling in tomographic optimization.

This module provides callback classes for:
- Hypothesis testing on scatter distributions
- Learning rate scheduling (1-cycle, plateau reduction)
- Live visualization of training metrics
- Gradient clipping for specific parameters
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output, display

# tomopt imports
from tomopt.optimisation.callbacks import (  # type: ignore[attr-defined]
    AbsOptSchedule,
    Callback,
)


class HypothesisTestCallback(Callback):
    """
    Perform hypothesis testing on passive scatter distributions after each epoch.

    This callback collects scattering angle distributions during both background
    and signal phases, computes angular resolution, and passes the data to the
    loss function for statistical hypothesis testing.

    The callback operates in two phases:
    1. Background phase: Collects background scatter distributions
    2. Signal phase: Collects signal scatter distributions and computes resolution

    Attributes:
        is_signal_phase: Flag indicating current collection phase
        scatter_dists_bkg: List of background scatter tensors
        scatter_dists_sig: List of signal scatter tensors
        angular_res_means: List of mean angular resolutions per volume
        scatter_batch: Temporary storage for scatters within a volume
        sigma_theta: Temporary storage for angular resolutions within a volume
    """

    def __init__(self) -> None:
        """Initialize the hypothesis test callback."""
        super().__init__()
        self.is_signal_phase: bool = False

    def on_volume_batch_begin(self) -> None:
        """
        Initialize data collection containers for the current batch phase.

        Called at the start of each batch processing phase. Initializes
        background containers on first call, signal containers on second call.
        """
        if not self.is_signal_phase:
            # Background phase - first call
            self.scatter_dists_bkg: Optional[List[torch.Tensor]] = []
            self.angular_res_means: Optional[List[torch.Tensor]] = []
        else:
            # Signal phase - second call
            self.scatter_dists_sig: List[torch.Tensor] = []

    def on_volume_begin(self) -> None:
        """
        Initialize temporary storage for the current volume.

        Called at the start of processing each volume. Resets batch-level
        containers for scatter data and angular resolutions.
        """
        self.scatter_batch: List[torch.Tensor] = []
        self.sigma_theta: List[torch.Tensor] = []

    def on_scatter_end(self) -> None:
        """
        Collect scatter data and angular resolution after each scatter event.

        Stores total scatter angle and computes angular resolution from the
        difference between generated and reconstructed angles for both
        incoming and outgoing particles.
        """
        self.scatter_batch.append(self.wrapper.fit_params.sb.total_scatter)

        # Compute angular deviations
        delta_theta_in = self.wrapper.fit_params.sb.theta_in - self.wrapper.fit_params.sb.theta_in_gen  # type: ignore[attr-defined]
        delta_theta_out = self.wrapper.fit_params.sb.theta_out - self.wrapper.fit_params.sb.theta_out_gen  # type: ignore[attr-defined]

        # Average angular resolution from both angles
        self.sigma_theta.append((delta_theta_in.std() + delta_theta_out.std()) / 2)

    def on_volume_end(self) -> None:
        """
        Aggregate scatter data for the completed volume.

        Concatenates all scatters from the current volume and stores them
        in the appropriate phase container (background or signal).
        For signal phase, also computes and stores mean angular resolution.
        """
        scatters = torch.cat(self.scatter_batch, dim=0)

        if not self.is_signal_phase:
            self.scatter_dists_bkg.append(scatters)
        else:
            self.scatter_dists_sig.append(scatters)
            self.angular_res_means.append(torch.mean(torch.stack(self.sigma_theta)))

    def on_volume_batch_end_hypothesis(self) -> None:
        """
        Finalize hypothesis test data and pass to loss function.

        Called after all volumes in the batch are processed. Computes
        hardware angular resolution from detector geometry and passes
        all collected data to the loss function for hypothesis testing.

        The hardware resolution is computed separately for top and bottom
        detectors and combined assuming independent angle estimates.
        """
        # Compute angular resolution from detector geometry
        sigma_theta_top = self.compute_side_angular_resolution(side_index=0)
        sigma_theta_bottom = self.compute_side_angular_resolution(side_index=1)

        # Combine resolutions (assuming independent incoming/outgoing estimates)
        sigma_hw = torch.sqrt(sigma_theta_top**2 + sigma_theta_bottom**2)

        # Pass data to loss function
        self.wrapper.loss_func.set_scatter_data(
            scatters_bkg=self.scatter_dists_bkg, scatters_sig=self.scatter_dists_sig, sigma_hw=sigma_hw, epoch=self.wrapper.fit_params.epoch
        )

    def compute_side_angular_resolution(self, side_index: int) -> torch.Tensor:
        """
        Compute angular resolution for one detector side using weighted least squares.

        Uses the detector panel positions and resolutions to compute the
        angular resolution via error propagation. The calculation accounts
        for multiple panels at different z-positions with varying spatial
        resolutions.

        Args:
            side_index: Index of detector side (0=top, 1=bottom)

        Returns:
            Angular resolution σ_θ for the specified detector side

        Notes:
            Uses weighted variance formulation for numerical stability.
            Clamps minimum z-variance to prevent degeneracy when panels
            collapse to same position during optimization.
        """
        device = self.wrapper.device
        det = self.wrapper.volume.get_detectors()[side_index]
        panels = det.panels  # typically 3 panels per detector

        # Extract panel positions and resolutions
        z = torch.stack([p.z for p in panels]).to(device).squeeze(-1)  # [n_panels]
        sigma_x = torch.stack([1.0 / p.resolution for p in panels]).to(device)  # [n_panels]

        # Compute weights (inverse variance)
        w = 1.0 / (sigma_x**2 + 1e-20)

        # Weighted mean and variance of panel positions
        A = w.sum()
        z_mean = (w * z).sum() / (A + 1e-12)
        var_z = (w * (z - z_mean) ** 2).sum() / (A + 1e-12)

        # Prevent degeneracy when panels collapse in z
        # This avoids σ_hw blowing up during optimization
        var_z = torch.clamp(var_z, min=1e-4)

        # Angular resolution from error propagation
        sigma_theta_sq = 1.0 / (A * var_z)
        sigma_theta = torch.sqrt(sigma_theta_sq)

        return sigma_theta


class LiveLrPlotCallback(Callback):
    """
    Display live learning rate evolution during training.

    Creates and updates a matplotlib figure showing learning rate trajectories
    for all optimizers and parameter groups. Updates are displayed in-place
    in Jupyter notebooks using IPython display functionality.

    Attributes:
        epoch: Current epoch counter
        traces: Dictionary mapping curve names to (epochs, LR values)
        fig: Matplotlib figure for plotting
        ax: Matplotlib axes for plotting
    """

    def on_train_begin(self) -> None:
        """
        Initialize plotting infrastructure at start of training.

        Creates figure, axes, and trace storage. Displays initial empty plot.
        """
        self.epoch: int = 0
        self.traces: Dict[str, Tuple[List[int], List[float]]] = {}

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Learning Rate")
        self.ax.set_title("Live Learning Rate (All Optimizers)")

        display(self.fig)

    def on_epoch_end(self) -> None:
        """
        Update learning rate plot after each epoch.

        Collects current learning rates from all optimizers and parameter groups,
        updates traces, and refreshes the plot display. Only operates during
        training phase.
        """
        if self.wrapper.fit_params.state != "train":
            return

        updated = False

        # Collect LR from all optimizer param groups
        for opt_name, opt in self.wrapper.opts.items():
            for gi, pg in enumerate(opt.param_groups):
                lr = pg.get("lr", 0.0)
                if lr <= 0:
                    continue

                curve_name = f"{opt_name}_g{gi}"

                if curve_name not in self.traces:
                    self.traces[curve_name] = ([self.epoch], [lr])
                else:
                    xs, ys = self.traces[curve_name]
                    xs.append(self.epoch)
                    ys.append(lr)

                updated = True

        # Refresh plot if any LR was collected
        if updated:
            self.ax.clear()
            self.ax.set_xlabel("Epoch")
            self.ax.set_ylabel("Learning Rate")
            self.ax.set_title("Live Learning Rate (All Optimizers)")

            for curve_name, (xs, ys) in self.traces.items():
                self.ax.plot(xs, ys, label=curve_name)

            self.ax.legend(fontsize="x-small", frameon=False)

            clear_output(wait=True)
            display(self.fig)

        self.epoch += 1


class OneCycle(AbsOptSchedule):
    """
    Implement Smith's 1-cycle learning rate policy with momentum annealing.

    Based on "A disciplined approach to neural network hyper-parameters"
    (Smith, 2018, https://arxiv.org/abs/1803.09820).

    The schedule consists of two phases:

    1. Warmup phase:
       - Learning rate increases from init_lr to mid_lr
       - Momentum decreases from init_mom to mid_mom
       - Stabilizes training with high learning rates

    2. Convergence phase:
       - Learning rate decreases from mid_lr to final_lr
       - Momentum increases from mid_mom to final_mom
       - Allows fine-tuning with stable gradients

    Both phases use cosine annealing for smooth transitions.

    Args:
        opt_name: Name of optimizer affected by this scheduler
        warmup_length: Number of epochs for warmup phase
        init_lr: Initial learning rate (low) or None to use optimizer default
        init_mom: Initial momentum (high) or None to use optimizer default
        mid_lr: Peak learning rate (high)
        mid_mom: Mid-cycle momentum (moderate)
        final_lr: Final learning rate (low), defaults to init_lr if None
        final_mom: Final momentum (high), defaults to init_mom if None

    Note:
        Setting learning rate or momentum here overrides values specified
        when instantiating the optimizer. Use None to avoid overriding.
    """

    def __init__(
        self,
        opt_name: str,
        warmup_length: int,
        init_lr: Optional[float] = None,
        init_mom: Optional[float] = None,
        mid_lr: Optional[float] = None,
        mid_mom: Optional[float] = None,
        final_lr: Optional[float] = None,
        final_mom: Optional[float] = None,
    ) -> None:
        """Initialize 1-cycle scheduler with specified hyperparameters."""
        super().__init__(opt_name=opt_name, init_lr=init_lr, init_mom=init_mom)

        self.warmup_length = warmup_length
        self.mid_lr = mid_lr
        self.mid_mom = mid_mom
        self.final_lr = final_lr if final_lr is not None else init_lr
        self.final_mom = final_mom if final_mom is not None else init_mom

    def on_epoch_end(self) -> None:
        """
        Handle phase transition at end of warmup period.

        Resets iteration counter and recalculates schedule scale
        when transitioning from warmup to convergence phase.
        """
        if self.wrapper.fit_params.epoch - 1 == self.warmup_length:
            self.iter_cnt = 0
            self.warming_up = False
            self.length = self.n_epochs_expected - self.wrapper.fit_params.epoch + 1
            self.scale = self.length * self.n_iters_per_epoch

    def _activate(self) -> None:
        """
        Initialize schedule parameters when callback activates.

        Sets up warmup phase parameters and calculates iteration scale
        based on expected number of epochs and iterations per epoch.
        """
        super()._activate()
        self.warming_up = True
        self.length = self.warmup_length
        self.scale = self.length * self.n_iters_per_epoch

    def schedule(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute current learning rate and momentum values.

        Uses cosine annealing within current phase (warmup or convergence)
        to smoothly interpolate between phase start and end values.

        Returns:
            Tuple of (learning_rate, momentum), either value may be None
            if that parameter is not being scheduled
        """
        # Cosine annealing factor: 0 at start, 2 at end of phase
        x = np.cos(np.pi * self.iter_cnt / self.scale) + 1

        def get_param(init: float, mid: float, final: float) -> float:
            """Interpolate parameter based on current phase."""
            if self.warming_up:
                params = (init, mid)
            else:
                params = (mid, final)
            dx = (params[1] - params[0]) * x / 2
            return params[1] - dx

        # Compute scheduled values (None if not being scheduled)
        if self.init_lr is None:
            lr = None
        else:
            lr = get_param(self.init_lr, self.mid_lr, self.final_lr)

        if self.init_mom is None:
            mom = None
        else:
            mom = get_param(self.init_mom, self.mid_mom, self.final_mom)

        return lr, mom


class ReduceLROnPlateauCallback(Callback):
    """
    Reduce learning rate when a monitored metric stops improving.

    Implements a simple plateau detection strategy: if the monitored metric
    does not improve for a specified number of epochs (patience), reduce
    the learning rate by a multiplicative factor. Only affects parameter
    groups with non-zero learning rates.

    Live plot visualization shows learning rate evolution for all tracked
    parameter groups throughout training.

    Args:
        monitor: Name of metric to monitor (e.g., 'train_loss', 'val_loss')
        factor: Multiplicative factor for LR reduction (0 < factor < 1)
        patience: Number of epochs with no improvement before reducing LR
        min_lr: Minimum learning rate threshold
        verbose: Whether to print LR reduction messages

    Attributes:
        best: Best observed value of monitored metric
        num_bad_epochs: Counter for epochs without improvement
        traces: Dictionary storing LR history for plotting
        epoch: Current epoch counter
        fig, ax: Matplotlib figure and axes for live plotting
    """

    def __init__(self, monitor: str = "train_loss", factor: float = 0.5, patience: int = 3, min_lr: float = 1e-6, verbose: bool = True) -> None:
        """Initialize plateau-based LR reduction callback."""
        super().__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose

    def on_train_begin(self) -> None:
        """
        Initialize tracking and visualization at start of training.

        Sets up metric tracking, LR history storage, and creates
        initial live plot display.
        """
        self.best: float = float("inf")
        self.num_bad_epochs: int = 0
        self.traces: Dict[str, Tuple[List[int], List[float]]] = {}
        self.epoch: int = 0

        # Setup live plot
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Learning Rate")
        self.ax.set_title("Live LR (ReduceLROnPlateau)")
        display(self.fig)

    def on_epoch_end(self) -> None:
        """
        Check for improvement and potentially reduce learning rate.

        Monitors the specified metric after each epoch. If no improvement
        is seen for `patience` epochs, reduces learning rate for all
        parameter groups with non-zero LR. Updates live plot visualization.
        """
        if self.wrapper.fit_params.state != "train":
            return

        # Get current metric value
        val = self.wrapper.history.get(self.monitor, [float("inf")])[-1]  # type: ignore[attr-defined]

        # Check if metric improved
        if val < self.best:
            self.best = val
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        # Reduce LR if patience exceeded
        if self.num_bad_epochs >= self.patience:
            for opt_name, opt in self.wrapper.opts.items():
                for gi, pg in enumerate(opt.param_groups):
                    lr = pg.get("lr", 0.0)
                    if lr <= 0:
                        continue  # skip zero-LR params

                    new_lr = max(lr * self.factor, self.min_lr)
                    pg["lr"] = new_lr

                    if self.verbose:
                        print(f"Epoch {self.epoch}: reducing {opt_name}_g{gi} " f"lr {lr:.6f} → {new_lr:.6f}")
            self.num_bad_epochs = 0  # reset patience

        # Update traces and plot (only for lr > 0)
        for opt_name, opt in self.wrapper.opts.items():
            for gi, pg in enumerate(opt.param_groups):
                lr = pg.get("lr", 0.0)
                if lr <= 0:
                    continue

                curve_name = f"{opt_name}_g{gi}"
                if curve_name not in self.traces:
                    self.traces[curve_name] = ([self.epoch], [lr])
                else:
                    xs, ys = self.traces[curve_name]
                    xs.append(self.epoch)
                    ys.append(lr)

        # Refresh plot
        self.ax.clear()
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Learning Rate")
        self.ax.set_title("Live LR (ReduceLROnPlateau)")
        for curve_name, (xs, ys) in self.traces.items():
            self.ax.plot(xs, ys, label=curve_name)
        self.ax.legend(fontsize="x-small", frameon=False)
        clear_output(wait=True)
        display(self.fig)

        self.epoch += 1


class ReduceLRResponsible(Callback):
    """
    Reduce learning rate for parameter groups contributing most to the loss.

    Uses gradient norms to identify which parameter groups are "responsible"
    for the current loss value. When the monitored metric plateaus, only
    reduces learning rate for parameter groups whose gradient norms are
    above a threshold (fraction of maximum gradient norm).

    This targeted approach can be more effective than reducing all learning
    rates uniformly, especially when some parameters are already well-optimized.

    Args:
        monitor: Name of metric to monitor (default: 'val_loss')
        factor: Multiplicative factor for LR reduction (0 < factor < 1)
        min_lr: Minimum learning rate threshold
        patience: Number of epochs with no improvement before reducing
        verbose: Whether to print LR reduction messages
        top_frac: Fraction of max gradient norm; groups >= top_frac*max_norm
                  are considered responsible and will have LR reduced

    Attributes:
        epoch: Current epoch counter
        num_bad_epochs: Counter for epochs without improvement
        best: Best observed value of monitored metric
        traces: Dictionary storing LR history for plotting
        fig, ax: Matplotlib figure and axes for live plotting
    """

    def __init__(
        self, monitor: str = "val_loss", factor: float = 0.5, min_lr: float = 1e-6, patience: int = 3, verbose: bool = True, top_frac: float = 0.5
    ) -> None:
        """Initialize responsibility-based LR reduction callback."""
        super().__init__()
        self.monitor = monitor
        self.factor = factor
        self.min_lr = min_lr
        self.patience = patience
        self.verbose = verbose
        self.top_frac = top_frac

        # Setup live figure
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Learning Rate")
        self.ax.set_title("Live LR (Responsible Groups)")
        display(self.fig)

    def on_train_begin(self) -> None:
        """
        Initialize tracking at start of training.

        Sets up metric tracking and LR history storage for visualization.
        """
        self.epoch: int = 0
        self.num_bad_epochs: int = 0
        self.best: float = float("inf")
        self.traces: Dict[str, Tuple[List[int], List[float]]] = {}

    def on_epoch_end(self) -> None:
        """
        Check for improvement and reduce LR for responsible parameter groups.

        After validation phase, monitors the specified metric. If no improvement
        for `patience` epochs, computes gradient norms for all parameter groups
        and reduces learning rate only for those with norms above the threshold.
        Updates live plot visualization.
        """
        if self.wrapper.fit_params.state != "valid":
            return

        # Get current metric value
        metric_history = self.wrapper.history.get(self.monitor, [float("inf")])  # type: ignore[attr-defined]
        val = metric_history[-1]

        # Check for improvement
        if val < self.best:
            self.best = val
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        # Only reduce if patience exceeded
        if self.num_bad_epochs >= self.patience:
            # Compute gradient norms per parameter group
            grad_norms: List[Tuple[str, int, float, float]] = []

            for opt_name, opt in self.wrapper.opts.items():
                for gi, pg in enumerate(opt.param_groups):
                    lr = pg.get("lr", 0.0)
                    if lr <= 0:
                        continue

                    # Sum gradient norms for all parameters in group
                    group_norm = sum(p.grad.detach().norm().item() for p in pg["params"] if p.grad is not None)
                    grad_norms.append((opt_name, gi, group_norm, lr))

            # Reduce LR for groups with high gradient norms
            if grad_norms:
                max_norm = max(gn for _, _, gn, _ in grad_norms)
                threshold = self.top_frac * max_norm

                for opt_name, gi, gn, lr in grad_norms:
                    if gn >= threshold:
                        pg = self.wrapper.opts[opt_name].param_groups[gi]
                        new_lr = max(lr * self.factor, self.min_lr)
                        pg["lr"] = new_lr

                        if self.verbose:
                            print(f"Epoch {self.epoch}: reducing {opt_name}_g{gi} " f"lr {lr:.6f} → {new_lr:.6f} " f"(grad_norm={gn:.4f})")

            self.num_bad_epochs = 0  # reset patience

        # Update traces for plotting
        for opt_name, opt in self.wrapper.opts.items():
            for gi, pg in enumerate(opt.param_groups):
                lr = pg.get("lr", 0.0)
                if lr <= 0:
                    continue

                curve_name = f"{opt_name}_g{gi}"
                if curve_name not in self.traces:
                    self.traces[curve_name] = ([self.epoch], [lr])
                else:
                    xs, ys = self.traces[curve_name]
                    xs.append(self.epoch)
                    ys.append(lr)

        # Refresh plot
        self.ax.clear()
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Learning Rate")
        self.ax.set_title("Live LR (Responsible Groups)")
        for curve_name, (xs, ys) in self.traces.items():
            self.ax.plot(xs, ys, label=curve_name)
        self.ax.legend(fontsize="x-small", frameon=False)
        clear_output(wait=True)
        display(self.fig)

        self.epoch += 1


class PlotLoss(Callback):
    """
    Save loss plot to file after each epoch.

    Automatically detects training vs validation phases and maintains
    separate traces for train and validation loss. Saves a single
    updated plot file after each epoch showing both losses over time.

    Args:
        save_path: File path for saving the loss plot (default: 'loss_plot.png')

    Attributes:
        train_losses: List of training loss values per epoch
        val_losses: List of validation loss values per epoch
    """

    def __init__(self, save_path: str = "loss_plot.png") -> None:
        """
        Initialize loss plotting callback.

        Args:
            save_path: Path where loss plot will be saved
        """
        super().__init__()
        self.save_path = save_path
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)

    def on_epoch_end(self) -> None:
        """
        Update and save loss plot after each epoch.

        Detects current phase (train/validation), retrieves corresponding
        loss value from history, updates the appropriate trace, and saves
        a new plot showing both training and validation loss evolution.
        """
        # Determine current phase
        is_train = self.wrapper.fit_params.state == "train"

        # Select appropriate history key
        key = "train_loss" if is_train else "val_loss"

        if key not in self.wrapper.history:  # type: ignore[attr-defined]
            print("Warning: No loss recorded, skipping loss plot update.")
            return

        # Get most recent loss value
        print(self.wrapper.history[key])  # type: ignore[attr-defined]
        loss = self.wrapper.history[key][-1]  # type: ignore[attr-defined]

        # Append to appropriate trace
        if is_train:
            self.train_losses.append(loss)
        else:
            self.val_losses.append(loss)

        # Create and save plot
        plt.figure(figsize=(6, 4))

        if self.train_losses:
            plt.plot(self.train_losses, label="train", marker="o", markersize=3)
        if self.val_losses:
            plt.plot(self.val_losses, label="val", marker="s", markersize=3)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend(frameon=False, fontsize="small")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_path, dpi=150)
        plt.close()


class ClipSigmaSoftGrad(Callback):
    """
    Clip gradients of the sigma_soft parameter to prevent exploding updates.

    Applies gradient clipping specifically to the log_sigma_sw parameter
    in the loss function. This is useful when the software smoothing
    parameter's gradient can become unstable during optimization.

    Args:
        min_val: Minimum allowed gradient value (default: -1.0)
        max_val: Maximum allowed gradient value (default: 1.0)

    Note:
        Only applies if the loss function has a `log_sigma_sw` attribute
        with a gradient. Clipping is performed in-place.
    """

    def __init__(self, min_val: float = -1.0, max_val: float = 1.0) -> None:
        """
        Initialize gradient clipping callback.

        Args:
            min_val: Lower bound for gradient clipping
            max_val: Upper bound for gradient clipping
        """
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def on_backwards_end(self) -> None:
        """
        Clip log_sigma_sw gradient after backward pass.

        Called after gradients have been computed via backpropagation.
        Checks if the loss function has a log_sigma_sw parameter with
        a gradient, and if so, clips it to the specified range.
        """
        loss_func = self.wrapper.loss_func

        if hasattr(loss_func, "log_sigma_sw") and loss_func.log_sigma_sw.grad is not None:
            # Clip gradient in-place
            torch.clamp_(loss_func.log_sigma_sw.grad, min=self.min_val, max=self.max_val)
