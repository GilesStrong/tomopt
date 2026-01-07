import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from diagnostics import DiagnosticPlotter, PlotData  # type: ignore[import] # noqa: E402

__all__ = ["HypothesisTestLoss", "HypothesisTestLossInner"]


class HypothesisTestLoss(nn.Module):
    """
    JSD-based differentiable hypothesis test utility with learnable software smoothing.

    Args:
        device: torch device
        alpha: significance level for threshold Tc
        n_bins: number of bins for soft histogram or KDE
        init_sigma_sw: initial value for learnable software smoothing parameter
        kde_scale: scale factor for KDE bandwidth (Silverman's rule)
        beta: steepness parameter for soft Tc computation
        tau_q: steepness of quantile approximation
        detection_mode: mode of power detection ('sigmoid', 'mann_whitney', 'p_value', or None)
        tau_detect: steepness of detection sigmoid
        tau_rank: temperature for pairwise rank (AUC-style) detection
        tau_p: steepness of p-value detection
        debug: enable debug printing and plotting
        plot_dir: directory for saving diagnostic plots
    """

    def __init__(
        self,
        device: torch.device = None,
        alpha: float = 0.05,
        n_bins: int = 50,
        init_sigma_sw: float = 0.01,
        kde_scale: float = 1.06,
        beta: float = 200.0,
        tau_q: float = 10.0,
        detection_mode: Optional[str] = None,
        tau_detect: float = 10.0,
        tau_rank: float = 2.0,
        tau_p: float = 0.02,
        debug: bool = True,
        plot_dir: str = "diagnostic_plots",
    ) -> None:
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Hyperparameters
        self.alpha = alpha
        self.n_bins = n_bins
        self.kde_scale = kde_scale
        self.beta = beta
        self.tau_q = tau_q
        self.detection_mode = detection_mode
        self.tau_detect = tau_detect
        self.tau_rank = tau_rank
        self.tau_p = tau_p
        self.debug = debug

        # Learnable parameters
        self.log_sigma_sw = nn.Parameter(torch.log(torch.tensor(init_sigma_sw, device=self.device)))

        # Buffers
        self.register_buffer("sigma_floor", torch.tensor(init_sigma_sw * 0.001, device=self.device))
        self.register_buffer("Tc_ema", torch.tensor(0.0))
        self.register_buffer("jsd_sig_ema", torch.tensor(0.0))
        self.register_buffer("T_bkg_ema", torch.tensor(0.0))
        self.register_buffer("ema_initialized", torch.tensor(0))

        # Runtime data (set via callback)
        self.scatters_bkg: Optional[List[torch.Tensor]] = None
        self.scatters_sig: Optional[List[torch.Tensor]] = None
        self.sigma_hw: Optional[torch.Tensor] = None
        self.epoch: Optional[int] = None

        # Debug storage
        self.debug_tensors: Dict[str, Optional[torch.Tensor]] = {
            "T_bkg": None,
            "jsd_sig": None,
            "Tc": None,
            "sigma_sw": None,
            "log_sigma_sw": None,
            "sigma_hw": None,
        }

        # Plotter
        self.plotter = DiagnosticPlotter(save_dir=plot_dir)

    def set_scatter_data(self, scatters_bkg: List[torch.Tensor], scatters_sig: List[torch.Tensor], sigma_hw: torch.Tensor, epoch: int) -> None:
        """
        Set scatter tensors and epoch information for current optimization step.
        To be called via callbacks during training.

        Args:
            scatters_bkg: background scatter tensors
            scatters_sig: signal scatter tensors
            sigma_hw: angular resolution (hardware)
            epoch: current epoch number
        """
        self.scatters_bkg = scatters_bkg
        self.scatters_sig = scatters_sig
        self.sigma_hw = sigma_hw
        self.epoch = epoch

    @property
    def sigma_sw(self) -> torch.Tensor:
        """Software smoothing parameter (always positive)."""
        return self.log_sigma_sw.exp()

    def soft_hist(self, data: torch.Tensor, bins: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute soft histogram using Gaussian kernel density estimation.

        Args:
            data: [N] tensor of scattering angles
            bins: [n_bins] tensor of bin centers
            sigma: bandwidth for Gaussian kernel

        Returns:
            Normalized histogram [n_bins]
        """
        sigma = sigma.clamp(min=1e-6)
        data = data.unsqueeze(-1)  # [N, 1]
        bins = bins.unsqueeze(0)  # [1, n_bins]

        weights = torch.exp(-0.5 * ((data - bins) / sigma) ** 2)
        weights = weights / (sigma * (2 * torch.pi) ** 0.5)

        hist = weights.sum(dim=0)
        hist = hist + 1e-10
        hist = hist / hist.sum()

        return hist

    def jsd_torch(self, P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """
        Compute Jensen-Shannon divergence between distributions.

        Args:
            P: [B, n_bins] probability distributions
            Q: [B, n_bins] probability distributions

        Returns:
            JSD values [B]
        """
        M = 0.5 * (P + Q)
        kl_pm = F.kl_div(P.log(), M, reduction="none").sum(dim=-1)
        kl_qm = F.kl_div(Q.log(), M, reduction="none").sum(dim=-1)
        jsd = 0.5 * (kl_pm + kl_qm)
        return jsd

    def compute_sigma_kde(self, T_bkg: torch.Tensor) -> torch.Tensor:
        """
        Compute KDE bandwidth using Silverman's rule.

        Args:
            T_bkg: [N] tensor of background JSD values

        Returns:
            Bandwidth scalar
        """
        n = T_bkg.numel()
        std = T_bkg.std().clamp_min(1e-6)
        return self.kde_scale * std * (n ** (-1 / 5))

    def compute_threshold(self, T_bkg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute differentiable critical value Tc using KDE-based CDF.

        Args:
            T_bkg: [N] tensor of background JSD values

        Returns:
            Tuple of (Tc, pdf, bins)
        """
        device = T_bkg.device
        sigma_kde = self.compute_sigma_kde(T_bkg).detach()

        # Extended grid (3σ padding)
        tmin, tmax = T_bkg.min(), T_bkg.max()
        lo = tmin - 3 * sigma_kde
        hi = tmax + 3 * sigma_kde
        bins = torch.linspace(lo, hi, self.n_bins, device=device)
        dx = (hi - lo) / (self.n_bins - 1)

        # KDE: sum of Gaussians normalized over dx
        data = T_bkg.unsqueeze(-1)  # [N, 1]
        diff = data - bins.unsqueeze(0)  # [N, n_bins]
        weights = torch.exp(-0.5 * (diff / sigma_kde) ** 2)
        pdf = weights.sum(dim=0)
        pdf = pdf / (pdf.sum() * dx + 1e-8)

        # CDF
        cdf = torch.cumsum(pdf, dim=0) * dx
        cdf = cdf / (cdf[-1] + 1e-8)

        # Find Tc via soft argmin of |CDF - (1 - α)|
        q = 1.0 - self.alpha
        err = (cdf - q).abs()
        w = torch.softmax(-self.beta * err, dim=0)
        Tc = (w * bins).sum()

        # Clamp inside true data range
        Tc = Tc.clamp(min=tmin, max=tmax)

        return Tc, pdf, bins

    def compute_detection(self, jsd_sig: torch.Tensor, Tc: torch.Tensor, T_bkg: torch.Tensor) -> torch.Tensor:
        """
        Compute detection power using specified detection mode.

        Args:
            jsd_sig: [N_sig] signal JSD values
            Tc: threshold value
            T_bkg: [N_bkg] background JSD values

        Returns:
            Detection power (scalar)
        """
        device = self.device
        s_b = torch.std(T_bkg).clamp(min=1e-6)

        if self.detection_mode == "sigmoid":
            tau_eff = torch.clamp(torch.tensor(self.tau_detect, device=device), min=0.1, max=10.0)
            z = (jsd_sig - Tc) / (s_b + 1e-8)
            z = torch.clamp(z, min=-8.0, max=8.0)

            pos = F.softplus(tau_eff * z)
            neg = F.softplus(-tau_eff * z)
            detections = pos / (pos + neg + 1e-8)

        elif self.detection_mode == "mann_whitney":
            delta = (jsd_sig.unsqueeze(1) - T_bkg.unsqueeze(0)) / (s_b + 1e-8)
            delta = torch.clamp(delta, min=-8.0, max=8.0)
            detections = torch.sigmoid(delta / self.tau_rank)

        elif self.detection_mode == "p_value":
            sigma_kde = self.compute_sigma_kde(T_bkg).detach()
            tmin, tmax = T_bkg.min(), T_bkg.max()
            bins_p = torch.linspace(tmin, tmax, self.n_bins, device=device)

            pdf_p = self.soft_hist(T_bkg, bins_p, sigma_kde)
            pdf_p = pdf_p / (pdf_p.sum() + 1e-8)
            cdf_p = torch.cumsum(pdf_p, dim=0)
            cdf_p = cdf_p / (cdf_p[-1] + 1e-8)

            w = torch.exp(-0.5 * ((jsd_sig.unsqueeze(-1) - bins_p) / (sigma_kde + 1e-8)) ** 2)
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
            p_sig = (w * cdf_p).sum(dim=-1)

            p_tail = 1.0 - p_sig
            detections = torch.sigmoid((self.alpha - p_tail) / self.tau_p)

        else:
            # Default: normalized sigmoid detection
            jsd_norm = (jsd_sig - jsd_sig.min()) / (jsd_sig.max() - jsd_sig.min() + 1e-8)
            Tc_norm = (Tc - jsd_sig.min()) / (jsd_sig.max() - jsd_sig.min() + 1e-8)
            detections = torch.sigmoid(self.tau_detect * (jsd_norm - Tc_norm))

        return detections.mean()

    def prepare_plot_data(
        self,
        bins: torch.Tensor,
        Tc: torch.Tensor,
        T_bkg: torch.Tensor,
        pdf: torch.Tensor,
        jsd_sig: torch.Tensor,
    ) -> PlotData:
        """
        Prepare all data needed for diagnostic plots.

        Args:
            bins: bin centers used for KDE
            Tc: computed threshold
            T_bkg: background JSD values
            pdf: background PDF from KDE
            jsd_sig: signal JSD values

        Returns:
            PlotData container with all necessary information
        """
        # Normalized values for sigmoid visualization
        jsd_norm = (jsd_sig - jsd_sig.min()) / (jsd_sig.max() - jsd_sig.min() + 1e-8)
        Tc_norm = (Tc - jsd_sig.min()) / (jsd_sig.max() - jsd_sig.min() + 1e-8)
        sigmoid_det = torch.sigmoid(self.tau_detect * (jsd_norm - Tc_norm))

        return PlotData(
            bins=bins, Tc=Tc, T_bkg=T_bkg, pdf=pdf, jsd_sig=jsd_sig, jsd_norm=jsd_norm, Tc_norm=Tc_norm, sigmoid_det=sigmoid_det, epoch=self.epoch or 0
        )

    def update_ema(self, T_bkg: torch.Tensor, jsd_sig: torch.Tensor) -> None:
        """Update exponential moving averages for monitoring."""
        with torch.no_grad():
            if self.ema_initialized.item() == 0:
                self.T_bkg_ema.copy_(T_bkg.mean())
                self.jsd_sig_ema.copy_(jsd_sig.mean())
                self.ema_initialized.fill_(1)
            else:
                self.T_bkg_ema.mul_(0.9).add_(0.1 * T_bkg.mean())
                self.jsd_sig_ema.mul_(0.9).add_(0.1 * jsd_sig.mean())

    def store_debug_tensors(self, T_bkg: torch.Tensor, jsd_sig: torch.Tensor, Tc: torch.Tensor) -> None:
        """Store intermediate tensors for debugging and retain gradients."""
        self.debug_tensors.update(
            {
                "T_bkg": T_bkg,
                "jsd_sig": jsd_sig,
                "Tc": Tc,
                "sigma_sw": self.sigma_sw,
                "log_sigma_sw": self.log_sigma_sw,
                "sigma_hw": self.sigma_hw,
            }
        )

        # Retain gradients on non-leaf tensors
        for tensor in self.debug_tensors.values():
            if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                try:
                    tensor.retain_grad()
                except RuntimeError:
                    pass

    def forward(self) -> torch.Tensor:
        """
        Compute hypothesis test loss.

        Returns:
            Loss value (1 - detection_power)
        """
        if self.scatters_bkg is None or self.scatters_sig is None or self.sigma_hw is None:
            raise ValueError("Scatter tensors and sigma_hw must be set before calling forward.")

        device = self.device

        # Compute global bins
        all_bkg_min = min(v.min().item() for v in self.scatters_bkg)
        all_sig_min = min(v.min().item() for v in self.scatters_sig)
        all_bkg_max = max(v.max().item() for v in self.scatters_bkg)
        all_sig_max = max(v.max().item() for v in self.scatters_sig)

        global_min = min(all_bkg_min, all_sig_min)
        global_max = max(all_bkg_max, all_sig_max)
        bins = torch.linspace(global_min, global_max, self.n_bins, device=device)

        # Background histograms
        bkg_hists = torch.stack([self.soft_hist(v.to(device), bins, self.sigma_sw) for v in self.scatters_bkg])
        avg_bkg_hist = bkg_hists.mean(dim=0, keepdim=True)

        # Signal histograms
        sig_hists = torch.stack([self.soft_hist(v.to(device), bins, self.sigma_sw) for v in self.scatters_sig])

        # Compute JSDs
        T_bkg = self.jsd_torch(bkg_hists, avg_bkg_hist)
        jsd_sig = self.jsd_torch(sig_hists, avg_bkg_hist)

        # Update EMAs
        self.update_ema(T_bkg, jsd_sig)

        # Compute threshold
        Tc, pdf, kde_bins = self.compute_threshold(T_bkg)

        # Compute detection power
        raw_power = self.compute_detection(jsd_sig, Tc, T_bkg)

        # Compute loss with optional penalty
        k_penalty = 0.1
        penalty = torch.exp(-k_penalty * ((self.sigma_sw - self.sigma_hw) / self.sigma_hw).pow(2))
        penalized_power = raw_power * penalty
        loss = 1 - penalized_power

        # Store debug information
        self.store_debug_tensors(T_bkg, jsd_sig, Tc)

        # Debug output
        if self.debug:
            print(
                f"Loss: {loss.item():.6f} | Power: {raw_power.item():.4f} | "
                f"Penalized power: {penalized_power.item():.4f} | "
                f"Sigma_sw: {self.sigma_sw.item():.6f} | Sigma_hw: {self.sigma_hw:.6f} | "
                f"Tc: {Tc.item():.6f}"
            )

            # Generate diagnostic plots
            plot_data = self.prepare_plot_data(kde_bins, Tc, T_bkg, pdf, jsd_sig)
            self.plotter.plot_all(plot_data, self.tau_detect)

        return loss


"""
Modified HypothesisTestLoss with inner loop optimization for software parameters.
"""


class HypothesisTestLossInner(nn.Module):
    """
    JSD-based hypothesis test with separated hardware/software optimization.

    Key changes:
    - Software parameter (log_sigma_sw) optimized in inner loop
    - Hardware parameters updated in outer loop (via wrapper)
    - Cached scatter samples for inner loop consistency
    """

    def __init__(
        self,
        device: torch.device = None,
        alpha: float = 0.05,
        n_bins: int = 50,
        init_sigma_sw: float = 0.01,
        kde_scale: float = 1.06,
        beta: float = 200.0,
        tau_q: float = 10.0,
        detection_mode: Optional[str] = None,
        tau_detect: float = 10.0,
        tau_rank: float = 2.0,
        tau_p: float = 0.02,
        debug: bool = True,
        plot_dir: str = "diagnostic_plots",
        n_inner_steps: int = 100,  # NEW: number of inner optimization steps
        inner_lr: float = 0.001,  # NEW: learning rate for inner optimization
    ) -> None:
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Hyperparameters
        self.alpha = alpha
        self.n_bins = n_bins
        self.kde_scale = kde_scale
        self.beta = beta
        self.tau_q = tau_q
        self.detection_mode = detection_mode
        self.tau_detect = tau_detect
        self.tau_rank = tau_rank
        self.tau_p = tau_p
        self.debug = debug

        # Inner loop settings
        self.n_inner_steps = n_inner_steps
        self.inner_lr = inner_lr

        # Learnable parameters
        self.log_sigma_sw = nn.Parameter(torch.log(torch.tensor(init_sigma_sw, device=self.device)))

        # Buffers
        self.register_buffer("sigma_floor", torch.tensor(init_sigma_sw * 0.001, device=self.device))
        self.register_buffer("Tc_ema", torch.tensor(0.0))
        self.register_buffer("jsd_sig_ema", torch.tensor(0.0))
        self.register_buffer("T_bkg_ema", torch.tensor(0.0))
        self.register_buffer("ema_initialized", torch.tensor(0))

        # Runtime data
        self.scatters_bkg: Optional[List[torch.Tensor]] = None
        self.scatters_sig: Optional[List[torch.Tensor]] = None
        self.sigma_hw: Optional[torch.Tensor] = None
        self.epoch: Optional[int] = None

        # NEW: Cache for fixed scatter samples during inner loop
        self.cached_scatters_bkg: Optional[List[torch.Tensor]] = None
        self.cached_scatters_sig: Optional[List[torch.Tensor]] = None
        self.cache_valid: bool = False

        # Debug storage
        self.debug_tensors: Dict[str, Optional[torch.Tensor]] = {
            "T_bkg": None,
            "jsd_sig": None,
            "Tc": None,
            "sigma_sw": None,
            "log_sigma_sw": None,
            "sigma_hw": None,
        }

        # Plotter
        self.plotter = DiagnosticPlotter(save_dir=plot_dir)

    def cache_scatter_samples(self) -> None:
        """
        Cache current scatter samples for inner loop optimization.
        Call this after hardware parameters change and new samples are generated.
        """
        if self.scatters_bkg is not None and self.scatters_sig is not None:
            # Detach to prevent gradients flowing back through sampling
            self.cached_scatters_bkg = [s.detach() for s in self.scatters_bkg]
            self.cached_scatters_sig = [s.detach() for s in self.scatters_sig]
            self.cache_valid = True
            if self.debug:
                print("[Cache] Scatter samples cached for inner loop")

    def invalidate_cache(self) -> None:
        """
        Invalidate cache when hardware parameters change.
        Call this before hardware optimization step.
        """
        self.cache_valid = False
        if self.debug:
            print("[Cache] Cache invalidated - hardware parameters changed")

    def optimize_software_inner_loop(self) -> torch.Tensor:
        """
        Perform inner loop optimization of software parameter.

        Uses cached scatter samples to optimize log_sigma_sw without
        the noise from resampling. Returns a fresh loss computation with
        full gradient graph for hardware optimization.

        Returns:
            Loss tensor with gradients for hardware parameters
        """
        if not self.cache_valid:
            raise ValueError("Cache invalid. Call cache_scatter_samples() first.")

        # Create optimizer for software parameter only
        inner_optimizer = torch.optim.Adam([self.log_sigma_sw], lr=self.inner_lr)

        if self.debug:
            print("\n{'='*60}")
            print("Starting inner loop optimization (σ_sw only)")
            print("{'='*60}")

        # Store initial value for monitoring
        initial_sigma_sw = self.sigma_sw.item()

        for step in range(self.n_inner_steps):
            inner_optimizer.zero_grad()

            loss = self._compute_loss_from_cache()

            loss.backward()
            inner_optimizer.step()

            if self.debug:
                print(f"  Inner step {step:3d}/{self.n_inner_steps}: " f"Loss={loss.item():.6f}, σ_sw={self.sigma_sw.item():.6f}")

        final_sigma_sw = self.sigma_sw.item()

        # NOW: Compute fresh loss with FULL computation graph
        # This allows gradients to flow back to hardware parameters
        final_loss = self._compute_loss()

        if self.debug or True:
            print(f"{'='*60}")
            print(f"Inner loop complete: σ_sw {initial_sigma_sw:.6f} → {final_sigma_sw:.6f}")
            print(f"Final loss: {final_loss.item():.6f}, σ_hw={self.sigma_hw.item():.6f}")
            print(f"{'='*60}\n")

        return final_loss

    def _compute_loss_from_cache(self) -> torch.Tensor:
        """
        Compute loss using cached scatter samples.
        Used during inner loop optimization.
        """
        if not self.cache_valid:
            raise ValueError("Cache invalid")

        device = self.device

        # Compute global bins
        all_bkg = torch.cat(self.cached_scatters_bkg)
        all_sig = torch.cat(self.cached_scatters_sig)
        global_min = min(all_bkg.min().item(), all_sig.min().item())
        global_max = max(all_bkg.max().item(), all_sig.max().item())
        bins = torch.linspace(global_min, global_max, self.n_bins, device=device)

        # Background histograms
        bkg_hists = torch.stack([self.soft_hist(v.to(device), bins, self.sigma_sw) for v in self.cached_scatters_bkg])
        avg_bkg_hist = bkg_hists.mean(dim=0, keepdim=True)

        # Signal histograms
        sig_hists = torch.stack([self.soft_hist(v.to(device), bins, self.sigma_sw) for v in self.cached_scatters_sig])

        # Compute JSDs
        T_bkg = self.jsd_torch(bkg_hists, avg_bkg_hist)
        jsd_sig = self.jsd_torch(sig_hists, avg_bkg_hist)

        # Compute threshold
        Tc, _, _ = self.compute_threshold(T_bkg)

        # Compute detection power
        raw_power = self.compute_detection(jsd_sig, Tc, T_bkg)

        # Compute loss with penalty
        k_penalty = 0.1
        penalty = torch.exp(-k_penalty * ((self.sigma_sw - self.sigma_hw.detach()) / self.sigma_hw.detach()).pow(2))
        penalized_power = raw_power * penalty
        loss = 1 - penalized_power

        # Store debug info
        self.debug_tensors.update(
            {
                "T_bkg": T_bkg,
                "jsd_sig": jsd_sig,
                "Tc": Tc,
                "sigma_sw": self.sigma_sw,
                "log_sigma_sw": self.log_sigma_sw,
                "sigma_hw": self.sigma_hw,
            }
        )

        print("sigma_sw grad:", self.sigma_sw.grad)

        return loss

    def _compute_loss(self) -> torch.Tensor:
        """
        Compute loss using scatter samples.
        Used during inner loop optimization.
        """
        device = self.device

        # Compute global bins
        all_bkg = torch.cat(self.scatters_bkg)
        all_sig = torch.cat(self.scatters_sig)
        global_min = min(all_bkg.min().item(), all_sig.min().item())
        global_max = max(all_bkg.max().item(), all_sig.max().item())
        bins = torch.linspace(global_min, global_max, self.n_bins, device=device)

        # Background histograms
        bkg_hists = torch.stack([self.soft_hist(v.to(device), bins, self.sigma_sw) for v in self.scatters_bkg])
        avg_bkg_hist = bkg_hists.mean(dim=0, keepdim=True)

        # Signal histograms
        sig_hists = torch.stack([self.soft_hist(v.to(device), bins, self.sigma_sw) for v in self.scatters_sig])

        # Compute JSDs
        T_bkg = self.jsd_torch(bkg_hists, avg_bkg_hist)
        jsd_sig = self.jsd_torch(sig_hists, avg_bkg_hist)

        # Compute threshold
        Tc, pdf, kde_bins = self.compute_threshold(T_bkg)

        # Compute detection power
        raw_power = self.compute_detection(jsd_sig, Tc, T_bkg)

        # Compute loss with penalty
        k_penalty = 0.1
        penalty = torch.exp(-k_penalty * ((self.sigma_sw - self.sigma_hw) / self.sigma_hw).pow(2))
        penalized_power = raw_power * penalty
        loss = 1 - penalized_power

        # Store debug info
        self.debug_tensors.update(
            {
                "T_bkg": T_bkg,
                "jsd_sig": jsd_sig,
                "Tc": Tc,
                "sigma_sw": self.sigma_sw,
                "log_sigma_sw": self.log_sigma_sw,
                "sigma_hw": self.sigma_hw,
            }
        )

        # Debug output
        if self.debug:
            print(
                f"Loss: {loss.item():.6f} | Power: {raw_power.item():.4f} | "
                f"Penalized power: {penalized_power.item():.4f} | "
                f"Sigma_sw: {self.sigma_sw.item():.6f} | Sigma_hw: {self.sigma_hw:.6f} | "
                f"Tc: {Tc.item():.6f}"
            )

            # Generate diagnostic plots
            plot_data = self.prepare_plot_data(kde_bins, Tc, T_bkg, pdf, jsd_sig)
            self.plotter.plot_all(plot_data, self.tau_detect)

        return loss

    @property
    def sigma_sw(self) -> torch.Tensor:
        """Software smoothing parameter (always positive)."""
        return self.log_sigma_sw.exp()

    def soft_hist(self, data: torch.Tensor, bins: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Compute soft histogram using Gaussian KDE."""
        sigma = sigma.clamp(min=1e-6)
        data = data.unsqueeze(-1)  # [N, 1]
        bins = bins.unsqueeze(0)  # [1, n_bins]

        weights = torch.exp(-0.5 * ((data - bins) / sigma) ** 2)
        weights = weights / (sigma * (2 * torch.pi) ** 0.5)

        hist = weights.sum(dim=0)
        hist = hist + 1e-10
        hist = hist / hist.sum()

        return hist

    def jsd_torch(self, P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """Compute Jensen-Shannon divergence."""
        M = 0.5 * (P + Q)
        kl_pm = F.kl_div(P.log(), M, reduction="none").sum(dim=-1)
        kl_qm = F.kl_div(Q.log(), M, reduction="none").sum(dim=-1)
        jsd = 0.5 * (kl_pm + kl_qm)
        return jsd

    def compute_sigma_kde(self, T_bkg: torch.Tensor) -> torch.Tensor:
        """Compute KDE bandwidth using Silverman's rule."""
        n = T_bkg.numel()
        std = T_bkg.std().clamp_min(1e-6)
        return self.kde_scale * std * (n ** (-1 / 5))

    def compute_threshold(self, T_bkg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute differentiable critical value Tc."""
        device = T_bkg.device
        sigma_kde = self.compute_sigma_kde(T_bkg).detach()

        tmin, tmax = T_bkg.min(), T_bkg.max()
        lo = tmin - 3 * sigma_kde
        hi = tmax + 3 * sigma_kde
        bins = torch.linspace(lo, hi, self.n_bins, device=device)
        dx = (hi - lo) / (self.n_bins - 1)

        data = T_bkg.unsqueeze(-1)
        diff = data - bins.unsqueeze(0)
        weights = torch.exp(-0.5 * (diff / sigma_kde) ** 2)
        pdf = weights.sum(dim=0)
        pdf = pdf / (pdf.sum() * dx + 1e-8)

        cdf = torch.cumsum(pdf, dim=0) * dx
        cdf = cdf / (cdf[-1] + 1e-8)

        q = 1.0 - self.alpha
        err = (cdf - q).abs()
        w = torch.softmax(-self.beta * err, dim=0)
        Tc = (w * bins).sum()

        Tc = Tc.clamp(min=tmin, max=tmax)

        return Tc, pdf, bins

    def compute_detection(self, jsd_sig: torch.Tensor, Tc: torch.Tensor, T_bkg: torch.Tensor) -> torch.Tensor:
        """Compute detection power."""
        device = self.device
        s_b = torch.std(T_bkg).clamp(min=1e-6)

        if self.detection_mode == "sigmoid":
            tau_eff = torch.clamp(torch.tensor(self.tau_detect, device=device), min=0.1, max=10.0)
            z = (jsd_sig - Tc) / (s_b + 1e-8)
            z = torch.clamp(z, min=-8.0, max=8.0)

            pos = F.softplus(tau_eff * z)
            neg = F.softplus(-tau_eff * z)
            detections = pos / (pos + neg + 1e-8)

        else:
            # Default detection
            jsd_norm = (jsd_sig - jsd_sig.min()) / (jsd_sig.max() - jsd_sig.min() + 1e-8)
            Tc_norm = (Tc - jsd_sig.min()) / (jsd_sig.max() - jsd_sig.min() + 1e-8)
            detections = torch.sigmoid(self.tau_detect * (jsd_norm - Tc_norm))

        return detections.mean()

    def set_scatter_data(self, scatters_bkg: List[torch.Tensor], scatters_sig: List[torch.Tensor], sigma_hw: torch.Tensor, epoch: int) -> None:
        """Set scatter tensors and epoch information."""
        self.scatters_bkg = scatters_bkg
        self.scatters_sig = scatters_sig
        self.sigma_hw = sigma_hw
        self.epoch = epoch

    def forward(self) -> torch.Tensor:
        """
        Compute loss with separated inner/outer optimization.

        Flow:
        1. Cache scatter samples (detached from hardware graph)
        2. Inner loop: Optimize sigma_sw on cached samples
        3. Recompute loss with original samples for hardware gradients

        Returns:
            Loss tensor with gradients to hardware parameters
        """
        if self.scatters_bkg is None or self.scatters_sig is None:
            raise ValueError("Scatter tensors must be set before calling forward.")

        # Cache samples if not already cached
        if not self.cache_valid:
            self.cache_scatter_samples()

        # Perform inner loop optimization (updates sigma_sw)
        # Returns fresh loss with hardware gradients
        loss = self.optimize_software_inner_loop()

        return loss
