import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass
class PlotData:
    """Container for data needed for diagnostic plots.

    bins: bins for T_bkg histogram and KDE
    Tc: threshold value for JSD statistic
    T_bkg: background JSD statistics
    pdf: KDE values for background JSD
    jsd_sig: signal JSD statistics
    jsd_norm: normalized signal JSD statistics
    Tc_norm: normalized threshold value
    sigmoid_det: sigmoid detection values for signal JSD
    tau_detect: steepness parameter for sigmoid detection
    epoch: current training epoch
    """

    bins: torch.Tensor
    Tc: torch.Tensor
    T_bkg: torch.Tensor
    pdf: torch.Tensor
    jsd_sig: torch.Tensor
    jsd_norm: torch.Tensor
    Tc_norm: torch.Tensor
    sigmoid_det: torch.Tensor
    tau_detect: float
    epoch: int

    def to_numpy(self) -> Dict[str, Union[np.ndarray, float]]:
        """Convert all tensors to numpy arrays."""
        return {
            "bins": self.bins.detach().cpu().numpy(),
            "Tc": self.Tc.detach().cpu().item(),
            "T_bkg": self.T_bkg.detach().cpu().numpy(),
            "pdf": self.pdf.squeeze().detach().cpu().numpy(),
            "jsd_sig": self.jsd_sig.detach().cpu().numpy(),
            "jsd_norm": self.jsd_norm.detach().cpu().numpy(),
            "Tc_norm": self.Tc_norm.detach().cpu().item(),
            "sigmoid_det": self.sigmoid_det.detach().cpu().numpy(),
            "tau_detect": self.tau_detect,
            "epoch": self.epoch,
        }


class DiagnosticPlotter:
    """Handles all plotting functionality."""

    def __init__(self, save_dir: str = "diagnostic_plots"):
        self.save_dir = Path(save_dir)

    def _ensure_arrays(self, data_dict: dict) -> dict:
        """Convert all entries that can be floats to at least 1D arrays for safe plotting."""
        safe_dict = {}
        for k, v in data_dict.items():
            if isinstance(v, (float, int)):
                safe_dict[k] = np.atleast_1d(v)
            else:
                safe_dict[k] = v
        return safe_dict

    def plot_background_distribution(self, plot_data: PlotData) -> None:
        """Plot background JSD histogram with KDE and threshold."""
        data = self._ensure_arrays(plot_data.to_numpy())
        epoch_path = self.save_dir / f"epoch_{int(data['epoch'][0]):03d}"
        epoch_path.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 5))

        # Background histogram
        ax.hist(data["T_bkg"], bins=data["bins"], alpha=0.5, color="blue", label="Background JSD", density=True)

        # KDE curve
        ax.plot(data["bins"], data["pdf"], color="red", lw=2, label="KDE")

        # Threshold line
        ax.axvline(data["Tc"][0], color="black", linestyle="--", lw=2, label="Tc")

        ax.set_xlabel("JSD Statistic")
        ax.set_ylabel("Density")
        ax.set_title(f'Background JSD with KDE and Tc (n={len(data["T_bkg"])}) samples')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(epoch_path / "background_jsd.png", dpi=150)
        plt.close(fig)

    def plot_signal_detection(self, plot_data: PlotData) -> None:
        """Plot signal JSD histogram with sigmoid detection curve."""
        data = self._ensure_arrays(plot_data.to_numpy())
        epoch_path = self.save_dir / f"epoch_{int(data['epoch'][0]):03d}"
        epoch_path.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 5))

        # Compute sigmoid detection curve over dense grid
        jsd_grid = np.linspace(data["jsd_norm"].min(), data["jsd_norm"].max(), 50)

        # Signal histogram
        ax.hist(data["jsd_norm"], bins=jsd_grid, range=(jsd_grid[0], jsd_grid[-1]), alpha=0.5, color="orange", label="Signal JSD", density=True)

        sigmoid_curve = 1 / (1 + np.exp(-data["tau_detect"] * (jsd_grid - data["Tc_norm"][0])))

        ax.plot(jsd_grid, sigmoid_curve, color="green", lw=2, label="Sigmoid detection (scaled)")

        # Threshold line
        ax.axvline(data["Tc_norm"][0], color="black", linestyle="--", lw=2, label="Tc (normalized)")

        ax.set_xlabel("JSD Statistic (normalized)")
        ax.set_ylabel("Density / Detection")
        ax.set_title(f'Signal JSD and Sigmoid Detection (n={len(data["jsd_sig"])}) samples')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(epoch_path / "signal_jsd_sigmoid.png", dpi=150)
        plt.close(fig)

    def plot_all(self, plot_data: PlotData) -> None:
        """Generate all diagnostic plots."""
        self.plot_background_distribution(plot_data)
        self.plot_signal_detection(plot_data)


class GradientDebugPlotter:
    """
    Track and plot values and gradients of debug_tensors across epochs with separate y-axes.
    """

    def __init__(self, save_path: str = "grad_debug"):
        self.tensor_vals: Dict[str, list] = defaultdict(list)
        self.tensor_grads: Dict[str, list] = defaultdict(list)
        self.epochs: list = []
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def update(self, debug_tensors: dict, epoch: int) -> None:
        self.epochs.append(epoch)
        for name, tensor in debug_tensors.items():
            if tensor is None or not torch.is_tensor(tensor):
                continue

            val_mean = tensor.detach().mean().item()
            self.tensor_vals[name].append(val_mean)
            grad_mean = tensor.grad.detach().mean().item() if tensor.grad is not None else 0.0
            self.tensor_grads[name].append(grad_mean)

    def plot(self, epoch: int = None) -> None:
        n_tensors = len(self.tensor_vals)
        if n_tensors == 0:
            return
        fig, axes = plt.subplots(n_tensors, 1, figsize=(10, 3 * n_tensors), sharex=True)
        if n_tensors == 1:
            axes = [axes]
        for ax, name in zip(axes, self.tensor_vals.keys()):
            # left axis: tensor value
            ax.plot(self.epochs, self.tensor_vals[name], color="blue", lw=2, label=f"{name} mean")
            ax.set_ylabel(name, color="blue")
            ax.tick_params(axis="y", labelcolor="blue")
            ax.grid(alpha=0.3)
            # right axis: gradient
            ax2 = ax.twinx()
            ax2.plot(self.epochs, self.tensor_grads[name], color="red", lw=1.5, linestyle="--", label=f"{name} grad mean")
            ax2.set_ylabel(f"{name} grad", color="red")
            ax2.tick_params(axis="y", labelcolor="red")
            # legends
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="upper right")
        axes[-1].set_xlabel("Epoch")
        plt.tight_layout()
        # Save figure
        fname = f"grad_debug_epoch_{epoch:03d}.png" if epoch is not None else "grad_debug.png"
        plt.savefig(os.path.join(self.save_path, fname))
        plt.close()


class SigmaHWGradPlotter:
    def __init__(self, save_path: str = "sigma_hw_grad"):
        self.epochs: List[int] = []
        self.grads: List[float] = []
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def update(self, sigma_hw: torch.Tensor, epoch: int) -> None:
        self.epochs.append(epoch)
        if sigma_hw.grad is None:
            self.grads.append(0.0)
        else:
            self.grads.append(sigma_hw.grad.detach().cpu().item())

    def plot(self, fname: str = "sigma_hw_grad.png") -> None:
        import os

        import matplotlib.pyplot as plt
        import numpy as np

        mean_grad = np.mean(self.grads)

        plt.figure(figsize=(6, 4))
        plt.plot(self.epochs, self.grads, lw=2, label=f"mean grad = {mean_grad:.3e}")
        plt.axhline(0.0, lw=1, ls="--")
        plt.xlabel("Epoch")
        plt.ylabel("grad(sigma_hw)")
        plt.legend()
        plt.tight_layout()

        out = os.path.join(self.save_path, fname)
        plt.savefig(out)
        plt.close()
