import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ...plotting import LBL_COL, LBL_SZ, LEG_SZ, STYLE, TK_COL, TK_SZ

__all__ = ["compare_raw_init_to_bias_corrected_init", "compare_init_to_optimised", "compare_init_optimised_2", "compare_optimised_to_baselines"]


def compare_raw_init_to_bias_corrected_init(df_start: pd.DataFrame, NAME: str) -> None:
    with sns.axes_style(**STYLE):
        _ = plt.figure(figsize=(12, 8))
        sdf_start = df_start.sort_values("gen_target")
        grps = sdf_start.groupby("gen_target")
        agg = grps.agg({f: ["mean", "std"] for f in ["pred", "new_pred"]})
        agg.columns = ["_".join(c).strip() for c in agg.columns.values]
        agg.reset_index(inplace=True)
        plt.errorbar(agg.gen_target, agg.pred_mean, yerr=agg.pred_std, label="Initial raw predictions", alpha=0.7)
        plt.errorbar(agg.gen_target, agg.new_pred_mean, yerr=agg.new_pred_std, label="Initial bias-corrected predictions", alpha=0.7)
        plt.xticks(fontsize=TK_SZ, color=TK_COL)
        plt.yticks(fontsize=TK_SZ, color=TK_COL)
        plt.xlabel("True fill-height [m]", fontsize=LBL_SZ, color=LBL_COL)
        plt.ylabel("Predicted fill-height [m]", fontsize=LBL_SZ, color=LBL_COL)
        plt.legend(fontsize=LEG_SZ)
        plt.savefig(f"{NAME}_plots/initial_predictions.pdf", bbox_inches="tight")
        plt.show()

        _ = plt.figure(figsize=(12, 8))
        sdf_start = df_start.sort_values("gen_target")
        grps = sdf_start.groupby("gen_target")
        agg = grps.agg({f: ["mean", "std"] for f in ["bias", "new_bias"]})
        agg.columns = ["_".join(c).strip() for c in agg.columns.values]
        agg.reset_index(inplace=True)
        plt.errorbar(
            agg.gen_target, agg.bias_mean, yerr=agg.bias_std, label=f"Initial raw predictions, mean |bias| {df_start.bias.abs().mean():.2E}", alpha=0.7
        )
        plt.errorbar(
            agg.gen_target,
            agg.new_bias_mean,
            yerr=agg.new_bias_std,
            label=f"Initial bias-corrected predictions, mean |bias| {df_start.new_bias.abs().mean():.2E}",
            alpha=0.7,
        )

        plt.xticks(fontsize=TK_SZ, color=TK_COL)
        plt.yticks(fontsize=TK_SZ, color=TK_COL)
        plt.xlabel("Target height [m]", fontsize=LBL_SZ, color=LBL_COL)
        plt.ylabel("Bias = true-pred. [m]", fontsize=LBL_SZ, color=LBL_COL)
        plt.legend(fontsize=LEG_SZ)
        plt.savefig(f"{NAME}_plots/initial_bias.pdf", bbox_inches="tight")
        plt.show()

        _ = plt.figure(figsize=(12, 8))
        sdf_start = df_start.sort_values("gen_target")
        grps = sdf_start.groupby("gen_target")
        agg = grps.agg({f: ["mean", "std"] for f in ["mse", "new_mse"]})
        agg.columns = ["_".join(c).strip() for c in agg.columns.values]
        agg.reset_index(inplace=True)
        plt.errorbar(agg.gen_target, agg.mse_mean, yerr=agg.mse_std, label=f"Initial raw predictions, MSE {df_start.mse.mean():.2E}", alpha=0.7)
        plt.errorbar(
            agg.gen_target, agg.new_mse_mean, yerr=agg.new_mse_std, label=f"Initial bias-corrected predictions, MSE {df_start.new_mse.mean():.2E}", alpha=0.7
        )

        plt.xticks(fontsize=TK_SZ, color=TK_COL)
        plt.yticks(fontsize=TK_SZ, color=TK_COL)
        plt.xlabel("Target height [m]", fontsize=LBL_SZ, color=LBL_COL)
        plt.ylabel("Squared error [$m^2$]", fontsize=LBL_SZ, color=LBL_COL)
        plt.legend(fontsize=LEG_SZ)
        plt.savefig(f"{NAME}_plots/initial_mse.pdf", bbox_inches="tight")
        plt.show()


def compare_init_to_optimised(df_start: pd.DataFrame, df_opt: pd.DataFrame, NAME: str) -> None:
    with sns.axes_style(**STYLE):
        _ = plt.figure(figsize=(12, 8))
        sdf_start = df_start.sort_values("gen_target")
        grps = sdf_start.groupby("gen_target")
        agg = grps.agg({f: ["mean", "std"] for f in ["pred", "new_pred"]})
        agg.columns = ["_".join(c).strip() for c in agg.columns.values]
        agg.reset_index(inplace=True)
        plt.errorbar(agg.gen_target, agg.new_pred_mean, yerr=agg.new_pred_std, label="Initial bias-corrected predictions", alpha=0.7)

        sdf_start = df_opt.sort_values("gen_target")
        grps = sdf_start.groupby("gen_target")
        agg = grps.agg({f: ["mean", "std"] for f in ["pred", "new_pred"]})
        agg.columns = ["_".join(c).strip() for c in agg.columns.values]
        agg.reset_index(inplace=True)
        plt.errorbar(agg.gen_target, agg.new_pred_mean, yerr=agg.new_pred_std, label="Optimised bias-corrected predictions", alpha=0.7)

        plt.xticks(fontsize=TK_SZ, color=TK_COL)
        plt.yticks(fontsize=TK_SZ, color=TK_COL)
        plt.xlabel("True fill-height [m]", fontsize=LBL_SZ, color=LBL_COL)
        plt.ylabel("Predicted fill-height [m]", fontsize=LBL_SZ, color=LBL_COL)
        plt.legend(fontsize=LEG_SZ)
        plt.savefig(f"{NAME}_plots/optimised_predictions.pdf", bbox_inches="tight")
        plt.show()

        _ = plt.figure(figsize=(12, 8))
        sdf_start = df_start.sort_values("gen_target")
        grps = sdf_start.groupby("gen_target")
        agg = grps.agg({f: ["mean", "std"] for f in ["bias", "new_bias"]})
        agg.columns = ["_".join(c).strip() for c in agg.columns.values]
        agg.reset_index(inplace=True)
        plt.errorbar(
            agg.gen_target,
            agg.new_bias_mean,
            yerr=agg.new_bias_std,
            label=f"Initial bias-corrected predictions, mean |bias| {df_start.new_bias.abs().mean():.2E}",
            alpha=0.7,
        )

        sdf_start = df_opt.sort_values("gen_target")
        grps = sdf_start.groupby("gen_target")
        agg = grps.agg({f: ["mean", "std"] for f in ["new_bias"]})
        agg.columns = ["_".join(c).strip() for c in agg.columns.values]
        agg.reset_index(inplace=True)
        plt.errorbar(
            agg.gen_target,
            agg.new_bias_mean,
            yerr=agg.new_bias_std,
            label=f"Optimised bias-corrected predictions, mean |bias| {df_opt.new_bias.abs().mean():.2E}",
            alpha=0.7,
        )

        plt.xticks(fontsize=TK_SZ, color=TK_COL)
        plt.yticks(fontsize=TK_SZ, color=TK_COL)
        plt.xlabel("Target height [m]", fontsize=LBL_SZ, color=LBL_COL)
        plt.ylabel("Bias = true-pred. [m]", fontsize=LBL_SZ, color=LBL_COL)
        plt.legend(fontsize=LEG_SZ)
        plt.savefig(f"{NAME}_plots/optimised_bias.pdf", bbox_inches="tight")
        plt.show()

        _ = plt.figure(figsize=(12, 8))
        sdf_start = df_start.sort_values("gen_target")
        grps = sdf_start.groupby("gen_target")
        agg = grps.agg({f: ["mean", "std"] for f in ["mse", "new_mse"]})
        agg.columns = ["_".join(c).strip() for c in agg.columns.values]
        agg.reset_index(inplace=True)
        plt.errorbar(
            agg.gen_target, agg.new_mse_mean, yerr=agg.new_mse_std, label=f"Initial bias-corrected predictions, MSE {df_start.new_mse.mean():.2E}", alpha=0.7
        )

        sdf_start = df_opt.sort_values("gen_target")
        grps = sdf_start.groupby("gen_target")
        agg = grps.agg({f: ["mean", "std"] for f in ["new_mse"]})
        agg.columns = ["_".join(c).strip() for c in agg.columns.values]
        agg.reset_index(inplace=True)
        plt.errorbar(
            agg.gen_target, agg.new_mse_mean, yerr=agg.new_mse_std, label=f"Optimised bias-corrected predictions, MSE {df_opt.new_mse.mean():.2E}", alpha=0.7
        )

        plt.xticks(fontsize=TK_SZ, color=TK_COL)
        plt.yticks(fontsize=TK_SZ, color=TK_COL)
        plt.xlabel("Target height [m]", fontsize=LBL_SZ, color=LBL_COL)
        plt.ylabel("Squared error [$m^2$]", fontsize=LBL_SZ, color=LBL_COL)
        plt.legend(fontsize=LEG_SZ)
        plt.yscale("log")
        plt.savefig(f"{NAME}_plots/optimised_mse.pdf", bbox_inches="tight")
        plt.show()


def compare_init_optimised_2(df_start: pd.DataFrame, df_opt_2: pd.DataFrame, NAME: str) -> None:
    with sns.axes_style(**STYLE):
        _ = plt.figure(figsize=(12, 8))
        sdf_start = df_start.sort_values("gen_target")
        grps = sdf_start.groupby("gen_target")
        agg = grps.agg({f: ["mean", "std"] for f in ["pred", "new_pred"]})
        agg.columns = ["_".join(c).strip() for c in agg.columns.values]
        agg.reset_index(inplace=True)
        plt.errorbar(agg.gen_target, agg.new_pred_mean, yerr=agg.new_pred_std, label="Initial bias-corrected predictions", alpha=0.7)

        sdf_start = df_opt_2.sort_values("gen_target")
        grps = sdf_start.groupby("gen_target")
        agg = grps.agg({f: ["mean", "std"] for f in ["pred", "new_pred"]})
        agg.columns = ["_".join(c).strip() for c in agg.columns.values]
        agg.reset_index(inplace=True)
        plt.errorbar(agg.gen_target, agg.new_pred_mean, yerr=agg.new_pred_std, label="Optimised bias-corrected predictions", alpha=0.7)

        plt.xticks(fontsize=TK_SZ, color=TK_COL)
        plt.yticks(fontsize=TK_SZ, color=TK_COL)
        plt.xlabel("True fill-height [m]", fontsize=LBL_SZ, color=LBL_COL)
        plt.ylabel("Predicted fill-height [m]", fontsize=LBL_SZ, color=LBL_COL)
        plt.legend(fontsize=LEG_SZ)
        plt.savefig(f"{NAME}_plots/optimised_2_predictions.pdf", bbox_inches="tight")
        plt.show()

        _ = plt.figure(figsize=(12, 8))
        sdf_start = df_start.sort_values("gen_target")
        grps = sdf_start.groupby("gen_target")
        agg = grps.agg({f: ["mean", "std"] for f in ["bias", "new_bias"]})
        agg.columns = ["_".join(c).strip() for c in agg.columns.values]
        agg.reset_index(inplace=True)
        plt.errorbar(
            agg.gen_target,
            agg.new_bias_mean,
            yerr=agg.new_bias_std,
            label=f"Initial bias-corrected predictions, mean |bias| {df_start.new_bias.abs().mean():.2E}",
            alpha=0.7,
        )

        sdf_start = df_opt_2.sort_values("gen_target")
        grps = sdf_start.groupby("gen_target")
        agg = grps.agg({f: ["mean", "std"] for f in ["new_bias"]})
        agg.columns = ["_".join(c).strip() for c in agg.columns.values]
        agg.reset_index(inplace=True)
        plt.errorbar(
            agg.gen_target,
            agg.new_bias_mean,
            yerr=agg.new_bias_std,
            label=f"Optimised bias-corrected predictions, mean |bias| {df_opt_2.new_bias.abs().mean():.2E}",
            alpha=0.7,
        )

        plt.xticks(fontsize=TK_SZ, color=TK_COL)
        plt.yticks(fontsize=TK_SZ, color=TK_COL)
        plt.xlabel("Target height [m]", fontsize=LBL_SZ, color=LBL_COL)
        plt.ylabel("Bias = true-pred. [m]", fontsize=LBL_SZ, color=LBL_COL)
        plt.legend(fontsize=LEG_SZ)
        plt.savefig(f"{NAME}_plots/optimised_2_bias.pdf", bbox_inches="tight")
        plt.show()

        _ = plt.figure(figsize=(12, 8))
        sdf_start = df_start.sort_values("gen_target")
        grps = sdf_start.groupby("gen_target")
        agg = grps.agg({f: ["mean", "std"] for f in ["mse", "new_mse"]})
        agg.columns = ["_".join(c).strip() for c in agg.columns.values]
        agg.reset_index(inplace=True)
        plt.errorbar(
            agg.gen_target, agg.new_mse_mean, yerr=agg.new_mse_std, label=f"Initial bias-corrected predictions, MSE {df_start.new_mse.mean():.2E}", alpha=0.7
        )

        sdf_start = df_opt_2.sort_values("gen_target")
        grps = sdf_start.groupby("gen_target")
        agg = grps.agg({f: ["mean", "std"] for f in ["new_mse"]})
        agg.columns = ["_".join(c).strip() for c in agg.columns.values]
        agg.reset_index(inplace=True)
        plt.errorbar(
            agg.gen_target, agg.new_mse_mean, yerr=agg.new_mse_std, label=f"Optimised bias-corrected predictions, MSE {df_opt_2.new_mse.mean():.2E}", alpha=0.7
        )

        plt.xticks(fontsize=TK_SZ, color=TK_COL)
        plt.yticks(fontsize=TK_SZ, color=TK_COL)
        plt.xlabel("Target height [m]", fontsize=LBL_SZ, color=LBL_COL)
        plt.ylabel("Squared error [$m^2$]", fontsize=LBL_SZ, color=LBL_COL)
        plt.legend(fontsize=LEG_SZ)
        plt.yscale("log")
        plt.savefig(f"{NAME}_plots/optimised_2_mse.pdf", bbox_inches="tight")
        plt.show()


def compare_optimised_to_baselines(df_bl_1: pd.DataFrame, df_bl_2: pd.DataFrame, df_opt_2: pd.DataFrame, NAME: str) -> None:
    with sns.axes_style(**STYLE):
        _ = plt.figure(figsize=(12, 8))
        sdf_start = df_bl_1.sort_values("gen_target")
        grps = sdf_start.groupby("gen_target")
        agg = grps.agg({f: ["mean", "std"] for f in ["new_pred"]})
        agg.columns = ["_".join(c).strip() for c in agg.columns.values]
        agg.reset_index(inplace=True)
        plt.errorbar(agg.gen_target, agg.new_pred_mean, yerr=agg.new_pred_std, label="Baseline 1 bias-corrected predictions", alpha=0.7)

        sdf_start = df_bl_2.sort_values("gen_target")
        grps = sdf_start.groupby("gen_target")
        agg = grps.agg({f: ["mean", "std"] for f in ["new_pred"]})
        agg.columns = ["_".join(c).strip() for c in agg.columns.values]
        agg.reset_index(inplace=True)
        plt.errorbar(agg.gen_target, agg.new_pred_mean, yerr=agg.new_pred_std, label="Baseline 2 bias-corrected predictions", alpha=0.7)

        sdf_start = df_opt_2.sort_values("gen_target")
        grps = sdf_start.groupby("gen_target")
        agg = grps.agg({f: ["mean", "std"] for f in ["new_pred"]})
        agg.columns = ["_".join(c).strip() for c in agg.columns.values]
        agg.reset_index(inplace=True)
        plt.errorbar(agg.gen_target, agg.new_pred_mean, yerr=agg.new_pred_std, label="Optimised bias-corrected predictions", alpha=0.7)

        plt.xticks(fontsize=TK_SZ, color=TK_COL)
        plt.yticks(fontsize=TK_SZ, color=TK_COL)
        plt.xlabel("True fill-height [m]", fontsize=LBL_SZ, color=LBL_COL)
        plt.ylabel("Predicted fill-height [m]", fontsize=LBL_SZ, color=LBL_COL)
        plt.legend(fontsize=LEG_SZ)
        plt.savefig(f"{NAME}_plots/baseline_predictions.pdf", bbox_inches="tight")
        plt.show()

        _ = plt.figure(figsize=(12, 8))
        sdf_start = df_bl_1.sort_values("gen_target")
        grps = sdf_start.groupby("gen_target")
        agg = grps.agg({f: ["mean", "std"] for f in ["new_bias"]})
        agg.columns = ["_".join(c).strip() for c in agg.columns.values]
        agg.reset_index(inplace=True)
        plt.errorbar(
            agg.gen_target,
            agg.new_bias_mean,
            yerr=agg.new_bias_std,
            label=f"Baseline 1 bias-corrected predictions, mean |bias| {df_bl_1.new_bias.abs().mean():.2E}",
            alpha=0.7,
        )

        sdf_start = df_bl_2.sort_values("gen_target")
        grps = sdf_start.groupby("gen_target")
        agg = grps.agg({f: ["mean", "std"] for f in ["new_bias"]})
        agg.columns = ["_".join(c).strip() for c in agg.columns.values]
        agg.reset_index(inplace=True)
        plt.errorbar(
            agg.gen_target,
            agg.new_bias_mean,
            yerr=agg.new_bias_std,
            label=f"Baseline 2 bias-corrected predictions, mean |bias| {df_bl_2.new_bias.abs().mean():.2E}",
            alpha=0.7,
        )

        sdf_start = df_opt_2.sort_values("gen_target")
        grps = sdf_start.groupby("gen_target")
        agg = grps.agg({f: ["mean", "std"] for f in ["new_bias"]})
        agg.columns = ["_".join(c).strip() for c in agg.columns.values]
        agg.reset_index(inplace=True)
        plt.errorbar(
            agg.gen_target,
            agg.new_bias_mean,
            yerr=agg.new_bias_std,
            label=f"Optimised bias-corrected predictions, mean |bias| {df_opt_2.new_bias.abs().mean():.2E}",
            alpha=0.7,
        )

        plt.xticks(fontsize=TK_SZ, color=TK_COL)
        plt.yticks(fontsize=TK_SZ, color=TK_COL)
        plt.xlabel("Target height [m]", fontsize=LBL_SZ, color=LBL_COL)
        plt.ylabel("Bias = true-pred. [m]", fontsize=LBL_SZ, color=LBL_COL)
        plt.legend(fontsize=LEG_SZ)
        plt.savefig(f"{NAME}_plots/baseline_bias.pdf", bbox_inches="tight")
        plt.show()

        _ = plt.figure(figsize=(12, 8))
        sdf_start = df_bl_1.sort_values("gen_target")
        grps = sdf_start.groupby("gen_target")
        agg = grps.agg({f: ["mean", "std"] for f in ["new_mse"]})
        agg.columns = ["_".join(c).strip() for c in agg.columns.values]
        agg.reset_index(inplace=True)
        plt.errorbar(
            agg.gen_target, agg.new_mse_mean, yerr=agg.new_mse_std, label=f"Baseline 1 bias-corrected predictions, MSE {df_bl_1.new_mse.mean():.2E}", alpha=0.7
        )

        sdf_start = df_bl_2.sort_values("gen_target")
        grps = sdf_start.groupby("gen_target")
        agg = grps.agg({f: ["mean", "std"] for f in ["new_mse"]})
        agg.columns = ["_".join(c).strip() for c in agg.columns.values]
        agg.reset_index(inplace=True)
        plt.errorbar(
            agg.gen_target, agg.new_mse_mean, yerr=agg.new_mse_std, label=f"Baseline 2 bias-corrected predictions, MSE {df_bl_2.new_mse.mean():.2E}", alpha=0.7
        )

        sdf_start = df_opt_2.sort_values("gen_target")
        grps = sdf_start.groupby("gen_target")
        agg = grps.agg({f: ["mean", "std"] for f in ["new_mse"]})
        agg.columns = ["_".join(c).strip() for c in agg.columns.values]
        agg.reset_index(inplace=True)
        plt.errorbar(
            agg.gen_target, agg.new_mse_mean, yerr=agg.new_mse_std, label=f"Optimised bias-corrected predictions, MSE {df_opt_2.new_mse.mean():.2E}", alpha=0.7
        )

        plt.xticks(fontsize=TK_SZ, color=TK_COL)
        plt.yticks(fontsize=TK_SZ, color=TK_COL)
        plt.xlabel("Target height [m]", fontsize=LBL_SZ, color=LBL_COL)
        plt.ylabel("Squared error [$m^2$]", fontsize=LBL_SZ, color=LBL_COL)
        plt.legend(fontsize=LEG_SZ)
        #     plt.yscale('log')
        plt.savefig(f"{NAME}_plots/baseline_mse.pdf", bbox_inches="tight")
        plt.show()
