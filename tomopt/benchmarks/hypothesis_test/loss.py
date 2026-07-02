from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .plotting import DiagnosticPlotter, PlotData

__all__ = ["HypothesisTestLoss", "HypothesisTestLossInner"]


class HypothesisTestLoss(nn.Module):
    """
    JSD-based differentiable hypothesis test loss, constructed for the task of detecting the presence of anomalous
    high-Z (signal) in a background material (null hypothesis). For each evaluation of the loss, the passive volume batch
    consists of equal numbers of background and signal samples for which the PoCA reconstruction is performed. A KDE of the
    PoCA scattering angle distribution is then built for each sample, and the JSD between each sample's distribution and the
    average background distribution is computed. The loss is then defined as 1 - detection_power, where detection_power is the
    fraction of signal samples whose JSD exceeds the threshold Tc, which is computed as the (1 - alpha) quantile of the
    background JSD distribution. The bandwidth used for the KDE of the scattering distributions is defined as a learnable parameter
    (sigma_sw) that is optimized during training, while the hardware resolution (sigma_hw), obtained from the current detector
    configuration, can be used for penalizing the loss. All operations are differentiable, allowing for end-to-end optimization
    of the detector configuration and software parameters.

    Args:
        alpha: significance level for threshold Tc (false positive rate)
        n_bins: number of bins for building scattering histograms
        init_sigma_sw: initial value for learnable software smoothing parameter
        kde_scale: scale factor for KDE bandwidth used for extraction of the critical threshold Tc (default: Silverman's rule)
        beta: steepness parameter used in soft argmin for differentiable quantile approximation in Tc computation
        detection_mode: mode of power detection ('sigmoid', 'p_value', or None (default))
        tau_detect: steepness of detection sigmoid
        tau_p: steepness of p-value detection
        use_sigma_match_penalty: whether to apply penalty for mismatch between learned sigma_sw and hardware sigma_hw
        use_sigma_hw_penalty: whether to apply penalty for large hardware resolution
        lambda_hw: weight for hardware resolution penalty
        k_penalty: weight for sigma mismatch penalty
        debug: enable debug printing and plotting
        plot_dir: directory for saving diagnostic plots
        device: torch device
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_bins: int = 50,
        init_sigma_sw: float = 0.01,
        kde_scale: float = 1.06,
        beta: float = 200.0,
        detection_mode: Optional[str] = None,
        tau_detect: float = 10.0,
        tau_p: float = 0.02,
        use_sigma_match_penalty: bool = False,
        use_sigma_hw_penalty: bool = True,
        lambda_hw: float = 0.1,
        k_penalty: float = 0.1,
        debug: bool = True,
        plot_dir: str = "diagnostic_plots",
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.n_bins = n_bins
        self.kde_scale = kde_scale
        self.beta = beta
        self.detection_mode = detection_mode
        self.tau_detect = tau_detect
        self.tau_p = tau_p
        self.use_sigma_match_penalty = use_sigma_match_penalty
        self.use_sigma_hw_penalty = use_sigma_hw_penalty
        self.lambda_hw = lambda_hw
        self.k_penalty = k_penalty
        self.debug = debug
        self.device = device

        self.log_sigma_sw = nn.Parameter(torch.log(torch.tensor(init_sigma_sw, device=self.device)))

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
            sigma_hw: angular resolution with the current hardware configuration
            epoch: current epoch number
        """
        self.scatters_bkg = scatters_bkg
        self.scatters_sig = scatters_sig
        self.sigma_hw = sigma_hw
        self.epoch = epoch

    def soft_hist(self, data: torch.Tensor, bins: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute soft histogram using Gaussian kernel density estimation.

        Args:
            data: [N] tensor of scattering angles
            bins: [n_bins] tensor of bin centers
            sigma: bandwidth for Gaussian kernel

        Returns:
            Normalized soft histogram [n_bins]
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
        To be used for threshold Tc computation.

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
        Compute differentiable critical value Tc from background JSD values using KDE-based CDF.

        Args:
            T_bkg: [N] tensor of background JSD values

        Returns:
            Tuple of (Tc, pdf, bins)
        """
        device = T_bkg.device
        sigma_kde = self.compute_sigma_kde(T_bkg).detach()

        # Extended grid (3-sigma padding)
        tmin, tmax = T_bkg.min(), T_bkg.max()
        lo = tmin - 3 * sigma_kde
        hi = tmax + 3 * sigma_kde
        bins = torch.linspace(lo, hi, self.n_bins, device=device)
        dx = (hi - lo) / (self.n_bins - 1)

        # gaussian KDE of background JSD
        data = T_bkg.unsqueeze(-1)  # [N, 1]
        diff = data - bins.unsqueeze(0)  # [N, n_bins]
        weights = torch.exp(-0.5 * (diff / sigma_kde) ** 2)
        pdf = weights.sum(dim=0)
        pdf = pdf / (pdf.sum() * dx + 1e-8)

        # CDF
        cdf = torch.cumsum(pdf, dim=0) * dx
        cdf = cdf / (cdf[-1] + 1e-8)

        # Tc via soft argmin of |CDF - (1 - alpha)|
        q = 1.0 - self.alpha
        err = (cdf - q).abs()
        w = torch.softmax(-self.beta * err, dim=0)
        Tc = (w * bins).sum()

        # Clamp within 3-sigma range of background JSD values
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

        # Compute global bins
        all_bkg_min = min(v.min().item() for v in self.scatters_bkg)
        all_sig_min = min(v.min().item() for v in self.scatters_sig)
        all_bkg_max = max(v.max().item() for v in self.scatters_bkg)
        all_sig_max = max(v.max().item() for v in self.scatters_sig)

        global_min = min(all_bkg_min, all_sig_min)
        global_max = max(all_bkg_max, all_sig_max)
        bins = torch.linspace(global_min, global_max, self.n_bins, device=self.device)

        # Background histograms
        bkg_hists = torch.stack([self.soft_hist(v.to(self.device), bins, self.sigma_sw) for v in self.scatters_bkg])
        avg_bkg_hist = bkg_hists.mean(dim=0, keepdim=True)

        # Signal histograms
        sig_hists = torch.stack([self.soft_hist(v.to(self.device), bins, self.sigma_sw) for v in self.scatters_sig])

        # Compute JSDs
        T_bkg = self.jsd_torch(bkg_hists, avg_bkg_hist)
        jsd_sig = self.jsd_torch(sig_hists, avg_bkg_hist)

        # Compute threshold
        Tc, pdf, kde_bins = self.compute_threshold(T_bkg)

        # Compute detection power
        raw_power = self.compute_detection(jsd_sig, Tc, T_bkg)

        # Compute loss with penalty
        loss = 1.0 - raw_power

        # optional penalties for sigma_sw and sigma_hw mismatch
        if self.use_sigma_match_penalty:
            rel_diff = (self.sigma_sw - self.sigma_hw) / (self.sigma_hw + 1e-8)
            penalty = torch.exp(-self.k_penalty * rel_diff.pow(2))
            loss = 1.0 - raw_power * penalty

        # optional penalty on large hardware resolution
        if self.use_sigma_hw_penalty:
            loss = loss + self.lambda_hw * self.sigma_hw

        # Store debug information
        self.store_debug_tensors(T_bkg, jsd_sig, Tc)

        # Debug output
        if self.debug:
            print(
                f"Loss: {loss.item():.6f} | Raw loss: {1 - raw_power.item():.4f} | "
                f"Sigma_sw: {self.sigma_sw.item():.6f} | Sigma_hw: {self.sigma_hw:.6f} | "
                f"Tc: {Tc.item():.6f}"
            )

            # Diagnostic plots of differentiable Tc and detection power computation
            plot_data = self.prepare_plot_data(kde_bins, Tc, T_bkg, pdf, jsd_sig)
            self.plotter.plot_all(plot_data)

        return loss

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
            pdf: KDE of background JSD, used in Tc computation
            jsd_sig: signal JSD values

        Returns:
            PlotData container with all necessary information
        """
        # Normalized values for sigmoid visualization
        jsd_norm = (jsd_sig - jsd_sig.min()) / (jsd_sig.max() - jsd_sig.min() + 1e-8)
        Tc_norm = (Tc - jsd_sig.min()) / (jsd_sig.max() - jsd_sig.min() + 1e-8)
        sigmoid_det = torch.sigmoid(self.tau_detect * (jsd_norm - Tc_norm))

        return PlotData(
            bins=bins,
            Tc=Tc,
            T_bkg=T_bkg,
            pdf=pdf,
            jsd_sig=jsd_sig,
            jsd_norm=jsd_norm,
            Tc_norm=Tc_norm,
            sigmoid_det=sigmoid_det,
            tau_detect=self.tau_detect,
            epoch=self.epoch or 0,
        )

    @property
    def sigma_sw(self) -> torch.Tensor:
        """Software parameter (bandwidth)."""
        return self.log_sigma_sw.exp()


class HypothesisTestLossInner(HypothesisTestLoss):
    """
    Modified HypothesisTestLoss with inner loop optimization for software parameter.

    Key changes:
    - Software parameter (log_sigma_sw) optimized in inner loop
    - Hardware parameters updated in outer loop (via wrapper)
    - Cached scatter samples for inner loop consistency

    Additional Args:
        n_inner_steps: number of inner optimization steps for software parameter
        inner_lr: learning rate for inner optimization of software parameter
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_bins: int = 50,
        init_sigma_sw: float = 0.01,
        kde_scale: float = 1.06,
        beta: float = 200.0,
        detection_mode: Optional[str] = None,
        tau_detect: float = 10.0,
        tau_p: float = 0.02,
        use_sigma_match_penalty: bool = False,
        use_sigma_hw_penalty: bool = True,
        lambda_hw: float = 0.1,
        k_penalty: float = 0.1,
        debug: bool = True,
        plot_dir: str = "diagnostic_plots",
        n_inner_steps: int = 100,
        inner_lr: float = 0.001,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(
            alpha=alpha,
            n_bins=n_bins,
            init_sigma_sw=init_sigma_sw,
            kde_scale=kde_scale,
            beta=beta,
            detection_mode=detection_mode,
            tau_detect=tau_detect,
            tau_p=tau_p,
            use_sigma_match_penalty=use_sigma_match_penalty,
            use_sigma_hw_penalty=use_sigma_hw_penalty,
            lambda_hw=lambda_hw,
            k_penalty=k_penalty,
            debug=debug,
            plot_dir=plot_dir,
            device=device,
        )

        self.n_inner_steps = n_inner_steps
        self.inner_lr = inner_lr

        # Cache for fixed scatter samples during inner loop
        self.cached_scatters_bkg: Optional[List[torch.Tensor]] = None
        self.cached_scatters_sig: Optional[List[torch.Tensor]] = None
        self.cache_valid: bool = False

        # Optimizer for software parameter
        self.inner_optimizer = torch.optim.Adam([self.log_sigma_sw], lr=self.inner_lr)

        # Scheduler for inner optimizer to reduce learning rate on plateau
        self.inner_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.inner_optimizer,
            mode="min",  # we want to reduce LR when loss stops decreasing
            factor=0.5,  # LR reduction factor
            patience=5,  # number of epochs to wait before reducing
            threshold=1e-4,  # minimum change to count as improvement
            cooldown=2,  # cooldown after LR reduction
            min_lr=1e-6,  # floor for LR
            verbose=self.debug,
        )

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
                print("[Cache] Scatter samples cached for inner loop", self.cache_valid)

    def invalidate_cache(self) -> None:
        """
        Invalidate cache when hardware parameters change.
        Call this before hardware optimization step.
        Used inside the wrapper.
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

        if self.debug:
            print("\n{'='*60}")
            print("Starting inner loop optimization (σ_sw only)")
            print("{'='*60}")

        # Store initial value for monitoring
        initial_sigma_sw = self.sigma_sw.item()

        initial_loss = self._compute_loss_from_cache()

        for step in range(self.n_inner_steps):
            self.inner_optimizer.zero_grad()

            loss = self._compute_loss_from_cache()

            loss.backward()
            self.inner_optimizer.step()

            if self.debug:
                print(
                    f"  Inner step {step:3d}/{self.n_inner_steps}: "
                    f"Loss={loss.item():.6f}, σ_sw={self.sigma_sw.item():.6f}, grad={self.log_sigma_sw.grad.item():.6f}"
                )

            self.inner_optimizer.step()

        # Step the scheduler once per outer epoch
        inner_metric = self._compute_loss_from_cache().detach()
        self.inner_scheduler.step(inner_metric)

        final_sigma_sw = self.sigma_sw.item()

        with torch.no_grad():
            self.log_sigma_sw.copy_(self.log_sigma_sw.detach())

        # Compute fresh loss with full computation graph
        # This allows gradients to flow back to hardware parameters
        final_loss = self._compute_loss()

        if (initial_loss - final_loss) < 1e-4:
            print("very small improvement in inner loss")

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

        # Compute global bins
        all_bkg = torch.cat(self.cached_scatters_bkg)
        all_sig = torch.cat(self.cached_scatters_sig)
        global_min = min(all_bkg.min().item(), all_sig.min().item())
        global_max = max(all_bkg.max().item(), all_sig.max().item())
        bins = torch.linspace(global_min, global_max, self.n_bins, device=self.device)

        # Background histograms
        bkg_hists = torch.stack([self.soft_hist(v.to(self.device), bins, self.sigma_sw) for v in self.cached_scatters_bkg])
        avg_bkg_hist = bkg_hists.mean(dim=0, keepdim=True)

        # Signal histograms
        sig_hists = torch.stack([self.soft_hist(v.to(self.device), bins, self.sigma_sw) for v in self.cached_scatters_sig])

        # Compute JSDs
        T_bkg = self.jsd_torch(bkg_hists, avg_bkg_hist)
        jsd_sig = self.jsd_torch(sig_hists, avg_bkg_hist)

        # Compute threshold
        Tc, _, _ = self.compute_threshold(T_bkg)

        # Compute detection power
        power = self.compute_detection(jsd_sig, Tc, T_bkg)

        # Compute loss
        loss = 1 - power

        return loss

    def _compute_loss(self) -> torch.Tensor:
        """
        Compute loss using scatter samples.
        This is the full loss computation used for hardware optimization, after inner loop optimization of software parameter.
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
        loss = 1.0 - raw_power

        # optional penalties for sigma_sw and sigma_hw mismatch
        if self.use_sigma_match_penalty:
            rel_diff = (self.sigma_sw - self.sigma_hw) / (self.sigma_hw + 1e-8)
            penalty = torch.exp(-self.k_penalty * rel_diff.pow(2))
            loss = 1.0 - raw_power * penalty

        # optional penalty on large hardware resolution
        if self.use_sigma_hw_penalty:
            loss = loss + self.lambda_hw * self.sigma_hw

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
        # Retain gradients on non-leaf tensors
        for tensor in self.debug_tensors.values():
            if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                try:
                    tensor.retain_grad()
                except RuntimeError:
                    pass

        # Debug output
        if self.debug:
            print(
                f"Loss: {loss.item():.6f} | Raw loss: {1 - raw_power.item():.4f} | "
                f"Sigma_sw: {self.sigma_sw.item():.6f} | Sigma_hw: {self.sigma_hw:.6f} | "
                f"Tc: {Tc.item():.6f}"
            )

            # Generate diagnostic plots
            plot_data = self.prepare_plot_data(kde_bins, Tc, T_bkg, pdf, jsd_sig)
            self.plotter.plot_all(plot_data)

        return loss

    def compute_threshold(self, T_bkg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute differentiable critical value Tc."""
        device = T_bkg.device
        sigma_kde = self.compute_sigma_kde(T_bkg)  # .detach()

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

    @torch.no_grad()
    def predict_current_utility(self, hw_sigma_hw: Optional[torch.Tensor] = None, sw_sigma: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict the utility loss for the current configuration of hardware/software.
        Does NOT run inner-loop optimization; only evaluates the loss.

        Args:
            hw_sigma_hw: Optional hardware resolution to override self.sigma_hw
            sw_sigma: Optional software smoothing parameter to override sigma_sw

        Returns:
            Predicted utility loss (scalar tensor)
        """

        # Override parameters if provided
        if hw_sigma_hw is not None:
            self.sigma_hw = hw_sigma_hw
        if sw_sigma is not None:
            self.log_sigma_sw = nn.Parameter(torch.log(torch.tensor(sw_sigma, device=self.device)))

        # Compute loss using current scatter samples
        if self.scatters_bkg is None or self.scatters_sig is None:
            raise ValueError("Scatter tensors must be set before calling predict_current_utility.")

        loss = self._compute_loss()

        return loss

    @torch.no_grad()
    def compute_reference_roc_auc(self, ref_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reference-based ROC/AUC evaluation.

        Protocol:
        - Use first volume (index 0) as reference (query)
        - Compute JSD(query, volume_i) for all i != 0
        - Use known sig/bkg identity only as labels
        - Return FPR, TPR, AUC

        Returns:
            fpr (Tensor), tpr (Tensor), auc (Tensor scalar)
        """

        if self.scatters_bkg is None or self.scatters_sig is None:
            raise ValueError("Scatter data not set")

        device = self.device

        # Combine volumes and labels
        volumes = self.scatters_bkg + self.scatters_sig
        labels = [0] * len(self.scatters_bkg) + [1] * len(self.scatters_sig)

        # Global bins (same as loss)
        all_vals = torch.cat(volumes)
        bins = torch.linspace(
            all_vals.min(),
            all_vals.max(),
            self.n_bins,
            device=device,
        )

        # Query volume (index 0)
        query_hist = self.soft_hist(volumes[ref_idx].to(device), bins, self.sigma_sw)

        jsd_scores = []
        gt_labels = []

        for i in range(1, len(volumes)):
            hist_i = self.soft_hist(volumes[i].to(device), bins, self.sigma_sw)

            jsd = self.jsd_torch(
                hist_i.unsqueeze(0),
                query_hist.unsqueeze(0),
            ).squeeze()

            jsd_scores.append(jsd)
            gt_labels.append(labels[i])

        jsd_scores_tensor = torch.stack(jsd_scores)
        gt_labels_tensor = torch.tensor(gt_labels, device=device, dtype=torch.float32)

        # Sort by score
        scores, idx = torch.sort(jsd_scores_tensor, descending=True)
        labels_sorted = gt_labels_tensor[idx]

        # ROC
        P = labels_sorted.sum()
        N = (1 - labels_sorted).sum()

        tpr = torch.cumsum(labels_sorted, dim=0) / (P + 1e-8)
        fpr = torch.cumsum(1 - labels_sorted, dim=0) / (N + 1e-8)

        # AUC (trapezoidal rule)
        auc = torch.trapz(tpr, fpr)

        return fpr, tpr, auc

    @torch.no_grad()
    def compute_background_averaged_auc(self, n_refs: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Average ROC AUC over background reference volumes.
        """

        n_bkg = len(self.scatters_bkg)
        ref_indices = range(n_bkg) if n_refs is None else range(min(n_refs, n_bkg))

        aucs = []

        for ref_idx in ref_indices:
            fpr, tpr, auc = self.compute_reference_roc_auc(ref_idx=ref_idx)
            aucs.append(auc)

        aucs_tensor = torch.stack(aucs)
        return aucs_tensor.mean(), aucs_tensor.std()
