import os
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


def plot_detection_power_analysis(all_histories: dict, resolutions: list, panel_seps: list, output_dir: str = "./plots", use_std: bool = True) -> None:
    """
    Generate and save plots for detection power optimization analysis.
    Each subplot is also saved as an individual figure (with uncertainty bands).
    """

    os.makedirs(output_dir, exist_ok=True)

    print("\nSummary Statistics (5% Significance Level):")
    print(f"{'Panel Sep':<12} {'Ang Res':<15} {'Mean Power':<12} {'Std Power':<12} {'Mean Effect':<15}")
    print("-" * 75)

    for ang_res, panel_sep in zip(resolutions, panel_seps):
        histories = all_histories[(ang_res, panel_sep)]
        final_powers = [h["power"][-1] for h in histories]
        final_effects = [h["effect_size"][-1] for h in histories]
        mean_power = np.mean(final_powers)
        std_power = np.std(final_powers)
        mean_effect = np.mean(final_effects)
        print(f"{panel_sep:<12.2f} {ang_res:<15.6f} {mean_power:<12.4f} " f"{std_power:<12.4f} {mean_effect:<15.4f}")

    fig, axes = plt.subplots(3, 3, figsize=(16, 10))

    plots: List[Tuple[str, Union[str, None], str, str, bool, Optional[Tuple[int, int]]]] = []  # (filename, key, title, ylabel, use_std, ylim)

    # Helper: generic plot with fill_between
    def make_plot(
        ax: plt.Axes,
        histories_key: str,
        title: str,
        ylabel: str,
        filename: str,
        ylim: Union[tuple, None] = None,
    ) -> None:
        for (ang_res, panel_sep), histories in all_histories.items():
            vals = np.array([h[histories_key] for h in histories])
            mean, std = vals.mean(axis=0), vals.std(axis=0)
            steps = np.arange(len(mean))
            label = f"Sep={panel_sep:.2f}"
            ax.plot(steps, mean, label=label, linewidth=2)
            if use_std:
                ax.fill_between(steps, mean - std, mean + std, alpha=0.2)
        ax.set_xlabel("Optimization Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plots.append((filename, histories_key, title, ylabel, use_std, ylim))

    # Main plots
    make_plot(axes[0, 0], "power", "Detection Power (5% Significance)", "Detection Power", "detection_power_vs_steps.png", ylim=(0, 1))
    make_plot(axes[0, 1], "effect_size", "Standardized Effect Size Evolution", "Effect Size (Cohen's d)", "effect_size_evolution.png")
    make_plot(axes[0, 2], "kl_signal", "KL(Signal || Background)", "KL Divergence", "kl_signal_evolution.png")
    make_plot(axes[1, 0], "sigma", "Sigma Optimized for Maximum Power", "Sigma (Smoothing Parameter)", "sigma_evolution.png")
    make_plot(axes[1, 1], "kl_null", "KL(Null || Background)", "KL Divergence", "kl_null_evolution.png")
    make_plot(axes[2, 0], "reg", "Regularization Term Evolution", "Regularization Term", "regularization_evolution.png")

    # Final power comparison bar plot
    ax = axes[1, 2]
    panel_sep_vals, mean_powers, std_powers = [], [], []
    for ang_res, panel_sep in zip(resolutions, panel_seps):
        histories = all_histories[(ang_res, panel_sep)]
        final_powers = [h["power"][-1] for h in histories]
        panel_sep_vals.append(panel_sep)
        mean_powers.append(np.mean(final_powers))
        std_powers.append(np.std(final_powers))
    x_pos = np.arange(len(panel_sep_vals))
    bars = ax.bar(x_pos, mean_powers, yerr=std_powers, capsize=5, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{ps:.2f}" for ps in panel_sep_vals])
    ax.set_xlabel("Panel Separation")
    ax.set_ylabel("Final Detection Power")
    ax.set_title("Detection Power by Configuration")
    ax.axhline(0.8, color="green", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val, err in zip(bars, mean_powers, std_powers):
        ax.text(bar.get_x() + bar.get_width() / 2, val + err + 0.02, f"{val:.3f}", ha="center", va="bottom")
    plots.append(("final_detection_power.png", None, "Detection Power by Configuration", "Detection Power", False, (0, 1)))

    plt.tight_layout()
    combined_path = os.path.join(output_dir, "detection_power_optimization.png")
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")
    print(f"\nCombined figure saved as: {combined_path}")

    # === Save each plot individually (with uncertainty bands) ===
    for filename, key, title, ylabel, use_std, ylim in plots:
        fig_i, ax_i = plt.subplots(figsize=(6, 4))
        if key is not None:
            for (ang_res, panel_sep), histories in all_histories.items():
                vals = np.array([h[key] for h in histories])
                mean, std = vals.mean(axis=0), vals.std(axis=0)
                steps = np.arange(len(mean))
                label = f"Sep={panel_sep:.2f}"
                ax_i.plot(steps, mean, label=label, linewidth=2)
                if use_std:
                    ax_i.fill_between(steps, mean - std, mean + std, alpha=0.2)
        else:
            # bar plot for final detection power
            ax_i.bar(x_pos, mean_powers, yerr=std_powers, capsize=5, alpha=0.7)
            ax_i.set_xticks(x_pos)
            ax_i.set_xticklabels([f"{ps:.2f}" for ps in panel_sep_vals])
            ax_i.axhline(0.8, color="green", linestyle="--", alpha=0.5)
            ax_i.set_ylim(ylim)

        ax_i.set_title(title)
        ax_i.set_xlabel("Optimization Step" if key else "Panel Separation")
        ax_i.set_ylabel(ylabel)
        ax_i.legend()
        ax_i.grid(True, alpha=0.3)

        path = os.path.join(output_dir, filename)
        fig_i.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig_i)
        print(f"Saved: {path}")

    plt.show()
