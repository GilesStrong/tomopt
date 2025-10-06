from typing import Any, Tuple

import torch
from scipy import stats  # type: ignore
from torch import Tensor


class AnomalyUtility(torch.nn.Module):
    """
    Utility function for anomaly detection based on KL divergence between signal and background distributions.
    The sigma parameter, controlling histogram smoothing, is optimized to maximize the KL divergence.
    An optional null distribution can be provided to compute a null KL divergence for spurious anomaly control.
    A regularization term, dependent on the angular resolution, can be used to penalize the utility

    """

    def __init__(self, ang_res: float):
        """
        Args:
            ang_res (float): angular resolution of detector
        """
        super().__init__()
        self.ang_res = torch.tensor(ang_res, dtype=torch.float32)
        self.log_scale = torch.nn.Parameter(torch.tensor(-5.0, requires_grad=True))  # smoothing factor
        self.log_alpha = torch.nn.Parameter(torch.tensor(-2.0, requires_grad=True))  # regularization strength (currently unused)

    def soft_hist(self, data: Tensor, bins: Tensor) -> Tensor:
        """
        Compute a differentiable (soft) histogram of input data using Gaussian kernel smoothing.

        Each data point contributes to all bins with a Gaussian weight centered on the bin.
        This yields a smooth, differentiable approximation of a discrete histogram.

        Args:
            data (Tensor): 1D tensor of scattering angles or other scalar values (shape: [N]).
            bins (Tensor): 1D tensor of bin centers (shape: [B]).

        Returns:
            Tensor: Normalized soft histogram over bins (shape: [B]).
        """
        # Convert log-scale parameter to standard deviation and ensure numerical stability
        sigma = torch.exp(self.log_scale).clamp(min=1e-4)

        # Broadcast data and bins to form pairwise differences: (N, 1) - (1, B) --> (N, B)
        data = data.unsqueeze(-1)
        bins = bins.unsqueeze(0)

        # Compute Gaussian kernel weights for each (data, bin) pair
        weights = torch.exp(-0.5 * ((data - bins) / sigma) ** 2)
        weights = weights / (sigma * (2 * torch.pi) ** 0.5)  # Normalize each Gaussian

        # Sum contributions of all data points per bin --> soft counts
        hist = weights.sum(dim=0)

        # Avoid zeros and normalize to obtain a valid probability distribution
        hist = hist + 1e-8
        hist = hist / hist.sum()

        return hist

    def forward(self, data_bkg: Tensor, data_sig: Tensor, bins: Tensor, data_null: Tensor = None) -> Tuple[Any, ...]:
        hist_bkg = self.soft_hist(data_bkg, bins)
        hist_sig = self.soft_hist(data_sig, bins)
        kl_signal = torch.sum(hist_sig * torch.log(hist_sig / hist_bkg))

        # penalize by KL of null distribution, if available
        if data_null is not None:
            hist_null = self.soft_hist(data_null, bins)
            kl_null = torch.sum(hist_null * torch.log(hist_null / hist_bkg))

        # regularization term to account for large angular resolutions
        alpha = torch.exp(self.log_alpha)
        bin_width = bins[1] - bins[0]
        reg = (alpha * self.ang_res) * (1.0 / bin_width)

        U = kl_signal - kl_null  # - reg

        return U, hist_bkg, hist_sig, kl_signal, kl_null, reg


class DetectionPowerUtility(AnomalyUtility):
    """
    Extension of AnomalyUtility that optimizes for detection power at 5% significance.

    Converts KL divergence to detection power using statistical power analysis.
    The sigma parameter is optimized to maximize the power of detecting uranium.
    """

    def __init__(self, ang_res: float, significance_level: float = 0.05):
        super().__init__(ang_res)
        self.significance_level = significance_level

        # Compute z-critical from significance level using inverse normal CDF
        # For one-sided test: z_critical = Φ^(-1)(1 - α)
        # Common values: α=0.05 → z=1.645, α=0.01 → z=2.326
        self.z_critical = torch.tensor(stats.norm.ppf(1 - significance_level), dtype=torch.float32)
        print(f"Significance level: {significance_level*100:.1f}% → z_critical = {self.z_critical.item():.3f}")

    def kl_to_power(self, kl_divergence: Tensor, n_samples: int) -> Tuple[Tensor, Tensor]:
        r"""
        Convert KL divergence to detection power at the specified significance level.

        The relationship:
        1. KL divergence measures distributional difference
        2. Effect size ≈ 2 * :math:`\sqrt{KL}`  (standardized measure)
        3. Power = Φ(effect_size × :math:`\sqrt{n} - z_critical)

        Where z_critical is determined by the significance level:
        - α = 0.05 → z_critical = 1.645 (5% false positive rate)
        - α = 0.01 → z_critical = 2.326 (1% false positive rate)

        Args:
            kl_divergence: KL(signal || background)
            n_samples: Number of samples

        Returns:
            power: Probability of detection at specified significance level
            effect_size: Standardized effect size
        """
        # Convert KL to effect size (Cohen's d-like measure)
        effect_size = 2.0 * torch.sqrt(kl_divergence + 1e-8)

        # Non-centrality parameter
        ncp = effect_size * torch.sqrt(torch.tensor(n_samples, dtype=torch.float32))

        # Z-score for power calculation
        # Higher z_critical (stricter significance) --> lower power for same effect
        z_power = ncp - self.z_critical

        # Power using tanh-based normal CDF approximation
        # Φ(x) ≈ 0.5 * (1 + tanh(0.8x))
        power = 0.5 * (1.0 + torch.tanh(0.8 * z_power))

        # Clamp to maintain gradients
        power = torch.clamp(power, 0.01, 0.99)

        return power, effect_size

    def forward(
        self, data_bkg: Tensor, data_sig: Tensor, bins: Tensor, data_null: Tensor = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Compute detection power utility at the specified significance level.

        Uses parent class to compute KL divergence, then converts to power.
        Optimizes sigma to maximize detection power at the given significance level.

        The significance level controls the trade-off:
        - Lower α (e.g., 1%) → stricter test → lower power but fewer false alarms
        - Higher α (e.g., 10%) → lenient test → higher power but more false alarms

        Returns:
            power: Detection power at specified significance level
            hist_bkg: Background histogram
            hist_sig: Signal histogram
            kl_signal: KL divergence (signal vs background)
            kl_null: KL divergence (null vs background)
            reg: Regularization term
            effect_size: Standardized effect size
        """
        # Get KL divergence from parent class
        _, hist_bkg, hist_sig, kl_signal, kl_null, reg = super().forward(data_bkg, data_sig, bins, data_null)

        # Convert KL divergence to detection power at specified significance
        n_samples = len(data_sig)
        print(f"Number of signal samples: {n_samples}")
        power_signal, effect_signal = self.kl_to_power(kl_signal, n_samples)

        # If null data provided, compute null power (should be ~significance_level)
        if data_null is not None:
            power_null, effect_null = self.kl_to_power(kl_null, len(data_null))
            # Utility: maximize signal power, minimize null power
            power = power_signal - reg  # - power_null # removed to trust our null
        else:
            power = power_signal

        # Return power as the utility to maximize
        return power, hist_bkg, hist_sig, kl_signal, kl_null, reg, effect_signal
