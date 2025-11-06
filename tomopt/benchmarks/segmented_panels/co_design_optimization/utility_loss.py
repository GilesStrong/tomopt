# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class HypothesisTestLoss(nn.Module):
#     """
#     JSD-based hypothesis test utility with learnable software smoothing.
#     """

#     def __init__(self, alpha: float=0.05, n_bins: int=50, c: float=2e-3, gamma: float=0.5, init_sigma: float=0.001, debug:bool = False) -> None:
#         super().__init__()
#         self.alpha = alpha
#         self.n_bins = n_bins
#         self.c = c
#         self.gamma = gamma
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.debug = debug

#         # Learnable software smoothing (σ_soft > 0)
#         self.log_sigma_soft = nn.Parameter(torch.log(torch.tensor(init_sigma, device=self.device)))

#         # Placeholders for callback
#         self.scatters_bkg = None
#         self.scatters_sig = None
#         self.avg_scatter_bkg = None
#         self.angular_res = None  # effective sigma_hw


#     def soft_hist(self, data: torch.Tensor, bins: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
#         """
#         Soft histograms of the scattering angle distributions. A smoothing parameter is defined as a free software parameter.

#         Args:
#             data (torch.Tensor): scattering distribution of the volume
#             bins (torch.Tensor): (hard) histogram bins
#             sigma (torch.Tensor): free software smoothing parameter

#         Returns:
#             torch.Tensor: soft histogram of the input data
#         """
#         sigma = sigma.clamp(min=1e-6)
#         #data = data.unsqueeze(-1)
#         bins = bins.unsqueeze(0)
#         weights = torch.exp(-0.5 * ((data - bins) / sigma) ** 2)
#         weights = weights / (sigma * (2 * torch.pi) ** 0.5)
#         hist = weights.sum(dim=0)
#         hist = hist + 1e-10
#         hist = hist / hist.sum()
#         return hist


#     def jsd_torch(self, P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
#         """
#         P, Q: [B, n_bins]
#         Returns JSD per batch element: [B]
#         """
#         M = 0.5 * (P + Q)
#         jsd = 0.5 * (F.kl_div(P.log(), M, reduction='none').sum(dim=1) +
#                     F.kl_div(Q.log(), M, reduction='none').sum(dim=1))
#         return jsd

#     def soft_quantile(self, x: torch.Tensor, q: float = 0.95, tau: float = 50.0) -> torch.Tensor:
#         """
#         Differentiable approximation of the q-th quantile using a sigmoid weighting.

#         Args:
#             x: Tensor of shape (N,)
#             q: Quantile to compute (0..1)
#             tau: Steepness of the sigmoid (higher = closer to hard quantile)

#         Returns:
#             Approximate quantile (scalar)
#         """
#         # x_sorted = x.unsqueeze(0)  # [1, N]
#         # # Compute pairwise differences: x_i - x_j
#         # diffs = x_sorted - x_sorted.T  # [1, N] - [N, 1] broadcasted
#         # # Soft indicator: sigmoid(tau * (x_i - x_j))
#         # weights = torch.sigmoid(tau * diffs)  # [N, N]
#         # # Sum over rows to approximate rank
#         # ranks = weights.sum(dim=1)
#         # # Soft quantile: interpolate where rank crosses q * N
#         # q_value = (x * torch.sigmoid(tau * (ranks - q * x.shape[0]))).sum() / (torch.sigmoid(tau * (ranks - q * x.shape[0])).sum() + 1e-8)
#         # return q_value

#     def soft_quantile(self, x: torch.Tensor, q: float = 0.95, tau: float = 50.0) -> torch.Tensor:
#         """
#         Differentiable soft quantile via weighted combination without O(N^2) memory.
#         """
#         # x_sorted, _ = torch.sort(x)
#         # N = x.shape[0]
#         # ranks = torch.arange(1, N + 1, device=x.device) / N  # [0,1] approximate ranks
#         # weights = torch.sigmoid(tau * (ranks - q))          # [N]
#         # weights = weights / (weights.sum() + 1e-8)         # normalize
#         # return (weights * x_sorted).sum()

#         x_sorted, _ = torch.sort(x)
#         N = x.shape[0]
#         ranks = torch.linspace(0, 1, N, device=x.device)
#         # Weight emphasizes values near the q-th percentile
#         weights = torch.exp(-tau * (ranks - q) ** 2)
#         weights = weights / (weights.sum() + 1e-8)
#         return (weights * x_sorted).sum()


#     def forward(self) -> torch.Tensor:
#         if self.scatters_bkg is None or self.scatters_sig is None or self.angular_res is None:
#             raise ValueError("Scatter tensors and angular_res must be set before calling forward.")

#         device = self.device
#         sigma_soft = self.sigma_soft
#         sigma_hw = self.angular_res  # Use precomputed angular resolution

#         # # --- Background PDFs ---
#         # bkg_angles = self.scatters_bkg.to(device)
#         # sig_angles = self.scatters_sig.to(device)


#         # if bkg_angles is None or bkg_angles.numel() == 0:
#         #     raise RuntimeError("Background scatters are empty")
#         # if sig_angles is None or sig_angles.numel() == 0:
#         #     raise RuntimeError("Signal scatters are empty")

#         # bins = torch.linspace(
#         #     torch.min(torch.cat([bkg_angles, sig_angles])),
#         #     torch.max(torch.cat([bkg_angles, sig_angles])),
#         #     self.n_bins,
#         #     device=device
#         # )

#         all_bkg_min = min([v.min() for v in self.scatters_bkg])
#         all_sig_min = min([v.min() for v in self.scatters_sig])
#         all_bkg_max = max([v.max() for v in self.scatters_bkg])
#         all_sig_max = max([v.max() for v in self.scatters_sig])

#         global_min = min(all_bkg_min, all_sig_min)
#         global_max = max(all_bkg_max, all_sig_max)

#         bins = torch.linspace(global_min, global_max, self.n_bins, device=self.device)


#         avg_bkg_hist = self.soft_hist(self.avg_scatter_bkg, bins, sigma_soft)

#         # # Compute threshold Tc
#         # # Split background into pseudo-volumes for thresholding
#         # n_vols = 5  # for example
#         # split_size = bkg_angles.shape[0] // n_vols
#         # T_bkg = []
#         # for i in range(n_vols):
#         #     chunk = bkg_angles[i*split_size:(i+1)*split_size]
#         #     chunk_hist = self.soft_hist(chunk, bins, sigma_soft)
#         #     T_bkg.append(self.jsd_torch(chunk_hist, bkg_hist))
#         # # Tc = torch.quantile(torch.stack(T_bkg), 1 - self.alpha)
#         # Tc = self.soft_quantile(torch.stack(T_bkg), q=1-self.alpha, tau=50.0)

#         # Compute per-sample JSDs
#         ####################
#         ### another
#         # bkg_hists = self.soft_hist(bkg_angles, bins, sigma_soft)
#         # T_bkg = (self.jsd_torch(bkg_hists, avg_bkg_hist))
#         # print(T_bkg.shape)
#         # Tc = self.soft_quantile(T_bkg, q=1-self.alpha, tau=50.0)

#         # # --- Signal PDFs and detections ---
#         # sig_hist = self.soft_hist(sig_angles, bins, sigma_soft)
#         # jsd = self.jsd_torch(sig_hist, avg_bkg_hist)
#         # detections = (jsd > Tc).float()
#         # raw_power = detections  # already scalar 0 or 1

#         # Assume bkg_angles: [B_bkg, N_i], sig_angles: [B_sig, N_i]
#         # bkg_hists = self.soft_hist(bkg_angles, bins, sigma_soft)  # [B_bkg, n_bins]


#         bkg_hists = torch.stack([self.soft_hist(v.to(self.device), bins, sigma_soft)
#                          for v in self.scatters_bkg])  # [B_bkg, n_bins]
#         avg_bkg_hist = bkg_hists.mean(dim=0, keepdim=True)        # [1, n_bins]
#         sig_hists = torch.stack([self.soft_hist(v.to(self.device), bins, sigma_soft)
#                                 for v in self.scatters_sig])  # [B_sig, n_bins]


#         # Per-sample JSD
#         T_bkg = self.jsd_torch(bkg_hists, avg_bkg_hist)           # [B_bkg]

#         # Threshold
#         Tc = self.soft_quantile(T_bkg, q=1 - self.alpha, tau=2.0)

#         # Signal
#         # sig_hists = self.soft_hist(sig_angles, bins, sigma_soft)  # [B_sig, n_bins]
#         jsd_sig = self.jsd_torch(sig_hists, avg_bkg_hist)         # [B_sig]

#         # detections = (jsd_sig > Tc).float()
#         tau_detect = 2.0  # "temperature" — higher = sharper threshold
#         detections = torch.sigmoid(tau_detect * (jsd_sig - Tc))

#         raw_power = detections.mean()  # scalar

#         # --- Parabolic penalty ---
#         penalty = 1 - ((sigma_soft - sigma_hw) / sigma_hw) ** 2
#         penalty = torch.clamp(penalty, min=0)
#         utility = raw_power * penalty

#         # --- Loss to minimize ---
#         # loss = 1 - utility
#         reg = 1e-3 * (sigma_soft - sigma_hw) ** 2
#         loss = 1 - utility + reg


#         print(f"Utility: {utility.item():.4f}, Reg: {reg.item():.6f}, Sigma_soft: {sigma_soft.item():.6f}, Sigma_hw: {sigma_hw.item():.6f}")

#         if self.debug:
#             self.debug_plot_jsd(T_bkg, Tc, jsd_sig)
#             self.debug_plot_soft_quantile(T_bkg, q=1 - self.alpha, tau=50.0)
#         return loss


#     @torch.no_grad()
#     def debug_plot_jsd(self, T_bkg: torch.Tensor, Tc: torch.Tensor, jsd_sig: torch.Tensor, q: float = 0.95, tau: float = 2.0):
#         """
#         Visualize background and signal JSD distributions together with:
#         - the differentiable quantile weighting curve
#         - the soft threshold Tc
#         """
#         import matplotlib.pyplot as plt
#         import numpy as np

#         # --- Convert tensors to numpy ---
#         T_bkg_np = T_bkg.detach().cpu().numpy()
#         jsd_sig_np = jsd_sig.detach().cpu().numpy()
#         Tc_val = Tc.detach().cpu().item()

#         # --- Plot distributions ---
#         plt.figure(figsize=(7,5))
#         plt.hist(T_bkg_np, bins=40, alpha=0.5, density=True, color='C0', label='Background JSD')
#         plt.hist(jsd_sig_np, bins=40, alpha=0.5, density=True, color='C1', label='Signal JSD')

#         # Vertical line for Tc
#         plt.axvline(Tc_val, color='r', ls='--', lw=2, label=fr'$T_c$ (α={self.alpha})')

#         plt.xlabel("JSD value")
#         plt.ylabel("PDF / normalized weight")
#         plt.title("Differentiable Quantile and Hypothesis Test Distributions")
#         plt.legend()
#         plt.grid(alpha=0.3)
#         plt.tight_layout()
#         plt.show()


#     @torch.no_grad()
#     def debug_plot_soft_quantile(self, x: torch.Tensor, q: float = 0.95, tau: float = 50.0):
#         """
#         Visualize the soft quantile contributions for a given tensor x.
#         """
#         import matplotlib.pyplot as plt
#         import numpy as np

#         # Compute sorted values and soft weights (same as soft_quantile)
#         x_sorted, _ = torch.sort(x)
#         N = len(x_sorted)
#         ranks = torch.linspace(0, 1, N, device=x.device)
#         weights = torch.exp(-tau * (ranks - q) ** 2)
#         weights /= weights.sum() + 1e-8  # normalize

#         # Scale weights for visualization over histogram
#         x_np = x_sorted.detach().cpu().numpy()
#         weights_np = weights.detach().cpu().numpy()
#         hist_vals, _ = np.histogram(x.detach().cpu().numpy(), bins=40, density=True)
#         weights_scaled = weights_np * hist_vals.max() / weights_np.max()

#         # Plot
#         plt.figure(figsize=(6,4))
#         plt.hist(x.detach().cpu().numpy(), bins=40, density=True, alpha=0.5, label='Data')
#         plt.plot(x_np, weights_scaled, color='r', lw=2, label=f'Soft quantile weights (q={q}, tau={tau})')
#         plt.xlabel("Value")
#         plt.ylabel("PDF / normalized weight")
#         plt.title("Soft Quantile Contributions")
#         plt.legend()
#         plt.grid(alpha=0.3)
#         plt.tight_layout()
#         plt.show()


#     @property
#     def sigma_soft(self) -> torch.Tensor:
#         return self.log_sigma_soft.exp()


from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HypothesisTestLoss(nn.Module):
    """
    JSD-based hypothesis test utility with learnable software smoothing.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_bins: int = 50,
        c: float = 2e-3,
        gamma: float = 0.5,
        init_sigma: float = 0.0005,
        tau_detect: float = 10.0,
        threshold_method: str = "soft_sort",  # 'soft_sort', 'differentiable_topk', 'smooth_max'
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.n_bins = n_bins
        self.c = c
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.debug = debug
        self.threshold_method = threshold_method

        # Hyperparameters for differentiable operations
        self.tau_detect = tau_detect  # Steepness of detection threshold

        # Learnable software smoothing (σ_soft > 0)
        self.log_sigma_soft = nn.Parameter(torch.log(torch.tensor(init_sigma, device=self.device)))

        # Placeholders for callback
        self.scatters_bkg: Optional[List[torch.Tensor]] = None
        self.scatters_sig: Optional[List[torch.Tensor]] = None
        self.avg_scatter_bkg: Optional[torch.Tensor] = None
        self.angular_res: Optional[torch.Tensor] = None  # effective sigma_hw

    def set_scatter_data(
        self, scatters_bkg: List[torch.Tensor], scatters_sig: List[torch.Tensor], avg_scatter_bkg: torch.Tensor, angular_res: torch.Tensor
    ) -> None:
        self.scatters_bkg = scatters_bkg
        self.scatters_sig = scatters_sig
        self.avg_scatter_bkg = avg_scatter_bkg
        self.angular_res = angular_res

    def soft_hist(self, data: torch.Tensor, bins: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Soft histograms of the scattering angle distributions.
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
        P, Q: [B, n_bins] or compatible broadcast shapes
        Returns JSD per batch element: [B]
        """
        M = 0.5 * (P + Q)
        jsd = 0.5 * (F.kl_div(P.log(), M, reduction="none").sum(dim=-1) + F.kl_div(Q.log(), M, reduction="none").sum(dim=-1))
        return jsd

    def soft_quantile_softsort(self, x: torch.Tensor, q: float = 0.95, tau: float = 1.0) -> torch.Tensor:
        """
        Differentiable quantile using continuous relaxation of sorting (NeuralSort-style).
        More stable gradients than Gaussian weighting.

        Args:
            x: [N] tensor
            q: quantile (0-1)
            tau: temperature for softmax (lower = softer, higher = closer to hard sort)
        """
        N = x.shape[0]
        # Compute pairwise differences
        diff = x.unsqueeze(1) - x.unsqueeze(0)  # [N, N]

        # Soft comparison: P[i,j] ≈ probability that x[i] > x[j]
        P = torch.sigmoid(diff / tau)  # [N, N]

        # Soft ranks: sum of "wins" against other elements
        soft_ranks = P.sum(dim=1)  # [N]

        # Normalize to [0, 1]
        soft_ranks_normalized = soft_ranks / (N - 1)

        # Soft selection of elements near the quantile
        weights = torch.exp(-((soft_ranks_normalized - q) ** 2) / (2 * 0.01))  # Gaussian window
        weights = weights / (weights.sum() + 1e-8)

        return (weights * x).sum()

    def differentiable_topk(self, x: torch.Tensor, k: int, tau: float = 0.1) -> torch.Tensor:
        """
        Differentiable approximation of top-k selection using softmax.
        Returns soft-weighted average of top-k elements.

        Args:
            x: [N] tensor
            k: number of top elements
            tau: temperature (lower = closer to hard top-k)
        """
        # Apply softmax with temperature to emphasize larger values
        weights = F.softmax(x / tau, dim=0)

        # To focus on top-k, we can use a sharpened distribution
        # Option: use top-k mask with straight-through estimator
        _, topk_indices = torch.topk(x, k)
        mask = torch.zeros_like(x)
        mask[topk_indices] = 1.0

        # Soft version: use weights but renormalize over top region
        soft_weights = weights * mask
        soft_weights = soft_weights / (soft_weights.sum() + 1e-8)

        return (soft_weights * x).sum()

    def smooth_max_quantile(self, x: torch.Tensor, q: float = 0.95, alpha: float = 10.0) -> torch.Tensor:
        """
        LogSumExp-based smooth maximum to approximate quantile.
        More principled than Gaussian weighting.

        Args:
            x: [N] tensor
            q: quantile
            alpha: sharpness parameter (higher = closer to true max)
        """
        # Get approximate top-k values using smooth max
        # LogSumExp: smooth_max(x) ≈ (1/α) * log(Σ exp(α * x))
        weights = F.softmax(alpha * x, dim=0)
        smooth_quantile = (weights * x).sum()

        return smooth_quantile

    def differentiable_quantile_regression(self, x: torch.Tensor, q: float = 0.95, tau: float = 1.0) -> torch.Tensor:
        """
        Uses a differentiable approximation based on quantile regression loss.
        This is numerically stable and has well-behaved gradients.

        Args:
            x: [N] tensor
            q: desired quantile
            tau: smoothing parameter for the check function
        """
        # For each potential threshold value, compute smooth quantile loss
        # We'll use a learnable threshold approach
        # Sample candidate thresholds from the data
        x_sorted, _ = torch.sort(x)

        # Use smooth approximation of the quantile check function
        # Standard: ρ_q(u) = u * (q - I(u < 0))
        # Smooth: use sigmoid instead of indicator

        # Vectorized over sorted values as candidates
        residuals = x.unsqueeze(0) - x_sorted.unsqueeze(1)  # [N, N]
        indicator = torch.sigmoid(-residuals / tau)  # smooth I(u < 0)

        # Quantile check function
        check = residuals * (q - indicator)
        quantile_loss = check.sum(dim=1)  # [N]

        # The quantile is where loss is minimized
        # Use soft argmin via softmax
        weights = F.softmax(-quantile_loss / tau, dim=0)

        return (weights * x_sorted).sum()

    def get_threshold(self, T_bkg: torch.Tensor) -> torch.Tensor:
        """
        Compute differentiable threshold based on selected method.
        """
        if self.threshold_method == "soft_sort":
            return self.soft_quantile_softsort(T_bkg, q=1 - self.alpha, tau=0.1)
        elif self.threshold_method == "differentiable_topk":
            k = max(1, int(len(T_bkg) * self.alpha))
            return self.differentiable_topk(T_bkg, k=k, tau=0.05)
        elif self.threshold_method == "smooth_max":
            return self.smooth_max_quantile(T_bkg, q=1 - self.alpha, alpha=20.0)
        elif self.threshold_method == "quantile_regression":
            return self.differentiable_quantile_regression(T_bkg, q=1 - self.alpha, tau=0.1)
        else:
            raise ValueError(f"Unknown threshold method: {self.threshold_method}")

    def forward(self) -> torch.Tensor:
        if self.scatters_bkg is None or self.scatters_sig is None or self.angular_res is None:
            raise ValueError("Scatter tensors and angular_res must be set before calling forward.")

        device = self.device
        sigma_soft = self.sigma_soft
        sigma_hw = self.angular_res

        # --- Compute global bins ---
        all_bkg_min = min([v.min().item() for v in self.scatters_bkg])
        all_sig_min = min([v.min().item() for v in self.scatters_sig])
        all_bkg_max = max([v.max().item() for v in self.scatters_bkg])
        all_sig_max = max([v.max().item() for v in self.scatters_sig])

        global_min = min(all_bkg_min, all_sig_min)
        global_max = max(all_bkg_max, all_sig_max)
        bins = torch.linspace(global_min, global_max, self.n_bins, device=device)

        # --- Background histograms ---
        bkg_hists = torch.stack([self.soft_hist(v.to(device), bins, sigma_soft) for v in self.scatters_bkg])  # [B_bkg, n_bins]

        avg_bkg_hist = bkg_hists.mean(dim=0, keepdim=True)  # [1, n_bins]

        # --- Signal histograms ---
        sig_hists = torch.stack([self.soft_hist(v.to(device), bins, sigma_soft) for v in self.scatters_sig])  # [B_sig, n_bins]

        # --- Compute JSDs ---
        T_bkg = self.jsd_torch(bkg_hists, avg_bkg_hist)  # [B_bkg]
        jsd_sig = self.jsd_torch(sig_hists, avg_bkg_hist)  # [B_sig]

        # --- Differentiable threshold ---
        Tc = self.get_threshold(T_bkg)

        # --- Soft detection ---
        detections = torch.sigmoid(self.tau_detect * (jsd_sig - Tc))
        raw_power = detections.mean()  # Average detection rate

        penalty = 1 - ((sigma_soft - sigma_hw) / sigma_hw) ** 2
        penalty = torch.clamp(penalty, min=0)
        utility = raw_power * penalty

        # # --- Regularization term ---
        # sigma_reg = self.c * ((sigma_soft - sigma_hw) / sigma_hw) ** 2

        # # --- Combined objective (simple version without penalty) ---
        # loss = 1 - raw_power + sigma_reg

        loss = -utility

        if self.debug or True:  # Always print for debugging
            print(
                f"Loss: {loss.item():.6f} | Power: {raw_power.item():.4f} | "
                f"Sigma_soft: {sigma_soft.item():.6f} | Sigma_hw: {sigma_hw.item():.6f} | "
                f"Tc: {Tc.item():.6f} | Method: {self.threshold_method}"
            )

        if self.debug:
            self.debug_plot_jsd(T_bkg, Tc, jsd_sig)

        return loss

    @torch.no_grad()
    def debug_plot_jsd(self, T_bkg: torch.Tensor, Tc: torch.Tensor, jsd_sig: torch.Tensor) -> None:
        """
        Visualize background and signal JSD distributions with threshold.
        """
        import matplotlib.pyplot as plt

        T_bkg_np = T_bkg.detach().cpu().numpy()
        jsd_sig_np = jsd_sig.detach().cpu().numpy()
        Tc_val = Tc.detach().cpu().item()

        plt.figure(figsize=(7, 5))
        plt.hist(T_bkg_np, bins=40, alpha=0.5, density=True, color="C0", label="Background JSD")
        plt.hist(jsd_sig_np, bins=40, alpha=0.5, density=True, color="C1", label="Signal JSD")
        plt.axvline(Tc_val, color="r", ls="--", lw=2, label=rf"$T_c$ (α={self.alpha})")
        plt.xlabel("JSD value")
        plt.ylabel("Density")
        plt.title(f"JSD Distributions ({self.threshold_method})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    @property
    def sigma_soft(self) -> torch.Tensor:
        return self.log_sigma_soft.exp()
