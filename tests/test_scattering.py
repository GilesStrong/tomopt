from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from collections import defaultdict
from prettytable import PrettyTable
import h5py
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pytest
import os
from fastcore.all import Path

import torch
from torch import Tensor

from tomopt.volume import PGEANT_SCATTER_MODEL, PassiveLayer
from tomopt.muon import MuonBatch
from tomopt.core import X0

PKG_DIR = Path(os.path.dirname(os.path.abspath(__file__)))  # How robust is this? Could create hidden dir in home and download resources


def cut_tail(distrib, cut: float):

    Q1 = (100 - cut) / 2
    Q2 = 100 - Q1

    q1 = np.percentile(distrib, Q1)
    q2 = np.percentile(distrib, Q2)
    return distrib[(distrib > q1) & (distrib < q2)]


def get_scatters(grp: h5py.Group, n_steps: int, plots: bool, verbose: bool) -> Dict[str, Any]:
    r"""
    file example Fe_1cm_5GeV_normal.txt
    """

    # Get data & settings
    settings = json.loads(grp["settings"][()])
    df = pd.DataFrame(grp["data"][()], columns=settings["columns"])

    # Get settings
    mat = settings["mat"]
    dz = float(settings["dz"])
    mom = float(settings["mom"])
    theta = float(settings["theta"])
    phi = float(settings["phi"])
    n_muons = int(settings["n_muons"])

    if verbose:
        print(f"Making sims for {mat=}, {dz=}, {mom=}, {theta=}, {phi=}, {n_muons=}")

    # Init layer
    step_sz = dz / (n_steps * np.cos(theta))
    layer = PassiveLayer(lw=Tensor([dz * 100, dz * 100]), z=1, size=dz, step_sz=step_sz)
    layer.rad_length = torch.ones(list((layer.lw / layer.size).long())) * X0[mat]

    PGEANT_SCATTER_MODEL.load_data()

    # Get sims
    xy_m_t_p = torch.ones(n_muons, 5)
    xy_m_t_p[:, :2] = 50 * dz
    xy_m_t_p[:, 2] = mom
    xy_m_t_p[:, 3] = theta
    xy_m_t_p[:, 4] = phi
    muons = MuonBatch(xy_m_t_p, init_z=1)
    start = muons.copy()

    # # New param model
    # pgeant_scattering = layer._pgeant_scatter(x0=x0, theta=muons.theta, theta_x=muons.theta_x, theta_y=muons.theta_y, mom=muons.mom)
    # pgeant_scattering["dangle_vol"] = np.sqrt((pgeant_scattering["dtheta_vol"] ** 2) + (pgeant_scattering["dphi_vol"] ** 2))
    # pgeant_scattering["dspace_vol"] = np.sqrt((pgeant_scattering["dx_vol"] ** 2) + (pgeant_scattering["dy_vol"] ** 2))

    layer(muons)
    start.propagate_dz(dz)  # Propagate without scattering
    pdg_scattering = {}
    pdg_scattering["dtheta_x_vol"] = muons.theta_x - start.theta_x
    pdg_scattering["dtheta_y_vol"] = muons.theta_y - start.theta_y
    pdg_scattering["dx_vol"] = muons.x - start.x
    pdg_scattering["dy_vol"] = muons.y - start.y
    pdg_scattering["dangle_vol"] = np.sqrt((pdg_scattering["dtheta_x_vol"] ** 2) + (pdg_scattering["dtheta_y_vol"] ** 2))
    pdg_scattering["dspace_vol"] = np.sqrt((pdg_scattering["dx_vol"] ** 2) + (pdg_scattering["dy_vol"] ** 2))

    tests = make_comparison(df, pdg_scattering, pgeant_scattering=None, plots=plots, verbose=verbose)

    return {"data": {"df": df, "pdg": pdg_scattering}, "tests": tests}


def make_comparison(
    geant_df: pd.DataFrame, pdg_scattering: Dict[str, Tensor], pgeant_scattering: Optional[Dict[str, Tensor]] = None, plots: bool = True, verbose: bool = True
) -> Dict[str, Dict[str, float]]:

    # Make plots and tests
    tests = defaultdict(lambda: defaultdict(dict))
    var = "dangle_vol"
    if verbose:
        print("\n\n", var)
    tab: PrettyTable = None
    for name, sim in (("pdg", pdg_scattering), ("param_geant", pgeant_scattering)):
        if sim is None:
            continue
        tests[var][name]["mean"] = sim[var].mean()
        tests[var][name]["std_fdiff"] = np.abs(sim[var].std() - geant_df[var].std()) / geant_df[var].std()
        cut95 = np.percentile(geant_df[var].abs(), 95)
        cut98 = np.percentile(geant_df[var].abs(), 98)
        tests[var][name]["bulk95_mean_fdiff"] = (
            np.abs(sim[var][sim[var] <= cut95].mean() - geant_df.loc[geant_df[var] <= cut95, var].mean()) / geant_df.loc[geant_df[var] <= cut95, var].mean()
        )
        tests[var][name]["bulk98_std_fdiff"] = (
            np.abs(sim[var][sim[var] <= cut98].std() - geant_df.loc[geant_df[var] <= cut98, var].std()) / geant_df.loc[geant_df[var] <= cut98, var].std()
        )
        cut68 = np.percentile(geant_df[var].abs(), 68)
        if tab is None:
            tab = PrettyTable(["Sim"] + [f for f in tests[var][name]])
        tab.add_row([name] + [v for k, v in tests[var][name].items()])

    if plots:
        sns.distplot(geant_df[var], label="True GEANT")
        if pgeant_scattering is not None:
            sns.distplot(pgeant_scattering[var], label="Param GEANT")
        sns.distplot(pdg_scattering[var], label="PDG")
        cut99 = np.percentile(geant_df[var], 99.9)
        plt.xlim([-cut99, cut99])
        plt.xlabel(var)
        plt.legend()
        plt.show()

        sns.distplot(geant_df.loc[geant_df[var].abs() > cut68, var].abs(), label="True GEANT")
        if pgeant_scattering is not None:
            sns.distplot(pgeant_scattering[var][pgeant_scattering[var].abs() > cut68].abs(), label="Param GEANT")
        sns.distplot(pdg_scattering[var][pdg_scattering[var].abs() > cut68].abs(), label="PDG")
        plt.xlabel(f"|{var}|")
        plt.legend()
        plt.yscale("log")
        plt.ylim(1e-3, 1e4)
        plt.show()

    if verbose:
        print(tab)

    var = "dtheta_x_vol"
    if verbose:
        print("\n\n", var)
    tab: PrettyTable = None
    for name, sim in (("pdg", pdg_scattering), ("param_geant", pgeant_scattering)):
        if sim is None:
            continue
        tests[var][name]["mean"] = sim[var].mean()
        tests[var][name]["std_fdiff"] = np.abs(sim[var].std() - geant_df[var].std()) / geant_df[var].std()
        cut95 = np.percentile(geant_df[var].abs(), 95)
        cut98 = np.percentile(geant_df[var].abs(), 98)
        tests[var][name]["bulk98_std_fdiff"] = np.abs(sim[var].std() - cut_tail(geant_df[var], 98).std()) / cut_tail(geant_df[var], 98).std()
        cut68 = np.percentile(geant_df[var].abs(), 68)
        if tab is None:
            tab = PrettyTable(["Sim"] + [f for f in tests[var][name]])
        tab.add_row([name] + [v for k, v in tests[var][name].items()])

    if plots:
        sns.distplot(geant_df[var], label="True GEANT")
        if pgeant_scattering is not None:
            sns.distplot(pgeant_scattering[var], label="Param GEANT")
        sns.distplot(pdg_scattering[var], label="PDG")
        cut99 = np.percentile(geant_df[var], 99.9)
        plt.xlim([-cut99, cut99])
        plt.xlabel(var)
        plt.legend()
        plt.show()

        sns.distplot(geant_df.loc[geant_df[var].abs() > cut68, var].abs(), label="True GEANT")
        if pgeant_scattering is not None:
            sns.distplot(pgeant_scattering[var][pgeant_scattering[var].abs() > cut68].abs(), label="Param GEANT")
        sns.distplot(pdg_scattering[var][pdg_scattering[var].abs() > cut68].abs(), label="PDG")
        plt.xlabel(f"|{var}|")
        plt.legend()
        plt.yscale("log")
        plt.ylim(1e-3, 1e4)
        plt.show()

    if verbose:
        print(tab)

    var = "dtheta_y_vol"
    if verbose:
        print("\n\n", var)
    tab: PrettyTable = None
    for name, sim in (("pdg", pdg_scattering), ("param_geant", pgeant_scattering)):
        if sim is None:
            continue
        tests[var][name]["mean"] = sim[var].mean()
        tests[var][name]["std_fdiff"] = np.abs(sim[var].std() - geant_df[var].std()) / geant_df[var].std()
        cut95 = np.percentile(geant_df[var].abs(), 95)
        cut98 = np.percentile(geant_df[var].abs(), 98)
        tests[var][name]["bulk98_std_fdiff"] = np.abs(sim[var].std() - cut_tail(geant_df[var], 98).std()) / cut_tail(geant_df[var], 98).std()
        cut68 = np.percentile(geant_df[var].abs(), 68)
        if tab is None:
            tab = PrettyTable(["Sim"] + [f for f in tests[var][name]])
        tab.add_row([name] + [v for k, v in tests[var][name].items()])

    if plots:
        sns.distplot(geant_df[var], label="True GEANT")
        if pgeant_scattering is not None:
            sns.distplot(pgeant_scattering[var], label="Param GEANT")
        sns.distplot(pdg_scattering[var], label="PDG")
        cut99 = np.percentile(geant_df[var], 99.9)
        plt.xlim([-cut99, cut99])
        plt.xlabel(var)
        plt.legend()
        plt.show()

        sns.distplot(geant_df.loc[geant_df[var].abs() > cut68, var].abs(), label="True GEANT")
        if pgeant_scattering is not None:
            sns.distplot(pgeant_scattering[var][pgeant_scattering[var].abs() > cut68].abs(), label="Param GEANT")
        sns.distplot(pdg_scattering[var][pdg_scattering[var].abs() > cut68].abs(), label="PDG")
        plt.xlabel(f"|{var}|")
        plt.legend()
        plt.yscale("log")
        plt.ylim(1e-3, 1e4)
        plt.show()

    if verbose:
        print(tab)

    var = "dspace_vol"
    if verbose:
        print("\n\n", var)
    tab: PrettyTable = None
    for name, sim in (("pdg", pdg_scattering), ("param_geant", pgeant_scattering)):
        if sim is None:
            continue
        tests[var][name]["mean_fdiff"] = np.abs(sim[var].mean() - geant_df[var].mean()) / geant_df[var].mean()
        tests[var][name]["std_fdiff"] = np.abs(sim[var].std() - geant_df[var].std()) / geant_df[var].std()
        cut95 = np.percentile(geant_df[var], 95)
        cut98 = np.percentile(geant_df[var].abs(), 98)
        tests[var][name]["bulk95_mean_fdiff"] = (
            np.abs(sim[var][sim[var] <= cut95].mean() - geant_df.loc[geant_df[var] <= cut95, var].mean()) / geant_df.loc[geant_df[var] <= cut95, var].mean()
        )
        tests[var][name]["bulk98_std_fdiff"] = (
            np.abs(sim[var][sim[var] <= cut98].std() - geant_df.loc[geant_df[var] <= cut98, var].std()) / geant_df.loc[geant_df[var] <= cut98, var].std()
        )
        cut68 = np.percentile(geant_df[var], 68)
        if tab is None:
            tab = PrettyTable(["Sim"] + [f for f in tests[var][name]])
        tab.add_row([name] + [v for k, v in tests[var][name].items()])

    if plots:
        sns.distplot(geant_df[var], label="True GEANT")
        if pgeant_scattering is not None:
            sns.distplot(pgeant_scattering[var], label="Param GEANT")
        sns.distplot(pdg_scattering[var], label="PDG")
        cut99 = np.percentile(geant_df[var], 99.9)
        plt.xlim([0, cut99])
        plt.xlabel(var)
        plt.legend()
        plt.show()

        sns.distplot(geant_df.loc[geant_df[var].abs() > cut68, var].abs(), label="True GEANT")
        if pgeant_scattering is not None:
            sns.distplot(pgeant_scattering[var][pgeant_scattering[var].abs() > cut68].abs(), label="Param GEANT")
        sns.distplot(pdg_scattering[var][pdg_scattering[var].abs() > cut68].abs(), label="PDG")
        plt.xlabel(f"|{var}|")
        plt.legend()
        plt.yscale("log")
        plt.ylim(1e-1, 1e6)
        plt.show()

    if verbose:
        print(tab)

    var = "dx_vol"
    if verbose:
        print("\n\n", var)
    tab: PrettyTable = None
    for name, sim in (("pdg", pdg_scattering), ("param_geant", pgeant_scattering)):
        if sim is None:
            continue
        tests[var][name]["mean"] = sim[var].mean()
        tests[var][name]["std_fdiff"] = np.abs(sim[var].std() - geant_df[var].std()) / geant_df[var].std()
        cut95 = np.percentile(geant_df[var].abs(), 95)
        cut98 = np.percentile(geant_df[var].abs(), 98)
        tests[var][name]["bulk98_std_fdiff"] = np.abs(sim[var].std() - cut_tail(geant_df[var], 98).std()) / cut_tail(geant_df[var], 98).std()
        cut68 = np.percentile(geant_df[var].abs(), 68)
        if tab is None:
            tab = PrettyTable(["Sim"] + [f for f in tests[var][name]])
        tab.add_row([name] + [v for k, v in tests[var][name].items()])

    if plots:
        sns.distplot(geant_df[var], label="True GEANT")
        if pgeant_scattering is not None:
            sns.distplot(pgeant_scattering[var], label="Param GEANT")
        sns.distplot(pdg_scattering[var], label="PDG")
        cut99 = np.percentile(geant_df[var], 99.9)
        plt.xlim([-cut99, cut99])
        plt.xlabel(var)
        plt.legend()
        plt.show()

        sns.distplot(geant_df.loc[geant_df[var].abs() > cut68, var].abs(), label="True GEANT")
        if pgeant_scattering is not None:
            sns.distplot(pgeant_scattering[var][pgeant_scattering[var].abs() > cut68].abs(), label="Param GEANT")
        sns.distplot(pdg_scattering[var][pdg_scattering[var].abs() > cut68].abs(), label="PDG")
        plt.xlabel(f"|{var}|")
        plt.legend()
        plt.yscale("log")
        plt.ylim(1e-1, 1e6)
        plt.show()

    if verbose:
        print(tab)

    var = "dy_vol"
    if verbose:
        print("\n\n", var)
    tab: PrettyTable = None
    for name, sim in (("pdg", pdg_scattering), ("param_geant", pgeant_scattering)):
        if sim is None:
            continue
        tests[var][name]["mean"] = sim[var].mean()
        tests[var][name]["std_fdiff"] = np.abs(sim[var].std() - geant_df[var].std()) / geant_df[var].std()
        cut98 = np.percentile(geant_df[var].abs(), 98)
        cut95 = np.percentile(geant_df[var].abs(), 95)
        tests[var][name]["bulk98_std_fdiff"] = np.abs(sim[var].std() - cut_tail(geant_df[var], 98).std()) / cut_tail(geant_df[var], 98).std()
        cut68 = np.percentile(geant_df[var].abs(), 68)
        if tab is None:
            tab = PrettyTable(["Sim"] + [f for f in tests[var][name]])
        tab.add_row([name] + [v for k, v in tests[var][name].items()])

    if plots:
        sns.distplot(geant_df[var], label="True GEANT")
        if pgeant_scattering is not None:
            sns.distplot(pgeant_scattering[var], label="Param GEANT")
        sns.distplot(pdg_scattering[var], label="PDG")
        cut99 = np.percentile(geant_df[var], 99.9)
        plt.xlim([-cut99, cut99])
        plt.xlabel(var)
        plt.legend()
        plt.show()

        sns.distplot(geant_df.loc[geant_df[var].abs() > cut68, var].abs(), label="True GEANT")
        if pgeant_scattering is not None:
            sns.distplot(pgeant_scattering[var][pgeant_scattering[var].abs() > cut68].abs(), label="Param GEANT")
        sns.distplot(pdg_scattering[var][pdg_scattering[var].abs() > cut68].abs(), label="PDG")
        plt.xlabel(f"|{var}|")
        plt.legend()
        plt.yscale("log")
        plt.ylim(1e-1, 1e6)
        plt.show()

    if verbose:
        print(tab)

    return tests


def check_scatter_tests(
    tests: Dict[str, Dict[str, Dict[str, float]]],
    pdg_pmin: Optional[float],
    pgeant_pmin: Optional[float],
    test: str = "bulk98_std_fdiff",
    ignore_vars: Optional[List[str]] = None,
) -> bool:
    if ignore_vars is None:
        ignore_vars = []
    else:
        print(f"Ignoring {ignore_vars}")
    for var in [v for v in tests if v not in ignore_vars]:
        if pdg_pmin is not None:
            if tests[var]["pdg"][test] > pdg_pmin:
                print(f'{var}, failed for PDG {test} = {tests[var]["pdg"][test]}')
                return False
        if pgeant_pmin is not None:
            if tests[var]["param_geant"][test] < pgeant_pmin:
                print(f'{var}, failed for parameterised GEANT {test} = {tests[var]["param_geant"][test]}')
                return False
    return True


# Single step tests, perpendicular beam
@pytest.mark.flaky(max_runs=5, min_passes=1)
def test_scatter_models_Al_1cm_1GeV_normal():
    with h5py.File(PKG_DIR / "data/geant_scatter_validation.hdf5", "r") as geant_data:
        results = get_scatters(geant_data["Al_1cm_1GeV_normal"], plots=False, verbose=True, n_steps=1)
        assert check_scatter_tests(results["tests"], pdg_pmin=0.20, pgeant_pmin=None, ignore_vars=["dspace_vol", "dangle_vol"])
        assert check_scatter_tests(
            results["tests"], pdg_pmin=0.15, pgeant_pmin=None, test="bulk95_mean_fdiff", ignore_vars=["dx_vol", "dy_vol", "dtheta_x_vol", "dtheta_y_vol"]
        )


@pytest.mark.flaky(max_runs=5, min_passes=1)
def test_scatter_models_Fe_1cm_1GeV_normal():
    with h5py.File(PKG_DIR / "data/geant_scatter_validation.hdf5", "r") as geant_data:
        results = get_scatters(geant_data["Fe_1cm_1GeV_normal"], plots=False, verbose=True, n_steps=1)
        assert check_scatter_tests(results["tests"], pdg_pmin=0.15, pgeant_pmin=None, ignore_vars=["dspace_vol", "dangle_vol"])
        assert check_scatter_tests(
            results["tests"], pdg_pmin=0.1, pgeant_pmin=None, test="bulk95_mean_fdiff", ignore_vars=["dx_vol", "dy_vol", "dtheta_x_vol", "dtheta_y_vol"]
        )


@pytest.mark.flaky(max_runs=5, min_passes=1)
def test_scatter_models_Pb_1cm_1GeV_normal():
    with h5py.File(PKG_DIR / "data/geant_scatter_validation.hdf5", "r") as geant_data:
        results = get_scatters(geant_data["Pb_1cm_1GeV_normal"], plots=False, verbose=True, n_steps=1)
        assert check_scatter_tests(results["tests"], pdg_pmin=0.15, pgeant_pmin=None, ignore_vars=["dspace_vol", "dangle_vol"])
        assert check_scatter_tests(
            results["tests"], pdg_pmin=0.1, pgeant_pmin=None, test="bulk95_mean_fdiff", ignore_vars=["dx_vol", "dy_vol", "dtheta_x_vol", "dtheta_y_vol"]
        )


@pytest.mark.flaky(max_runs=5, min_passes=1)
def test_scatter_models_Cu_1cm_1GeV_normal():
    with h5py.File(PKG_DIR / "data/geant_scatter_validation.hdf5", "r") as geant_data:
        results = get_scatters(geant_data["Cu_1cm_1GeV_normal"], plots=False, verbose=True, n_steps=1)
        assert check_scatter_tests(results["tests"], pdg_pmin=0.15, pgeant_pmin=None, ignore_vars=["dspace_vol", "dangle_vol"])
        assert check_scatter_tests(
            results["tests"], pdg_pmin=0.1, pgeant_pmin=None, test="bulk95_mean_fdiff", ignore_vars=["dx_vol", "dy_vol", "dtheta_x_vol", "dtheta_y_vol"]
        )


@pytest.mark.flaky(max_runs=5, min_passes=1)
def test_scatter_models_U_1cm_1GeV_normal():
    with h5py.File(PKG_DIR / "data/geant_scatter_validation.hdf5", "r") as geant_data:
        results = get_scatters(geant_data["U_1cm_1GeV_normal"], plots=False, verbose=True, n_steps=1)
        assert check_scatter_tests(results["tests"], pdg_pmin=0.15, pgeant_pmin=None, ignore_vars=["dspace_vol", "dangle_vol"])
        assert check_scatter_tests(
            results["tests"], pdg_pmin=0.1, pgeant_pmin=None, test="bulk95_mean_fdiff", ignore_vars=["dx_vol", "dy_vol", "dtheta_x_vol", "dtheta_y_vol"]
        )


# Single step tests, theta_x = pi/4, theta_y = o
@pytest.mark.flaky(max_runs=5, min_passes=1)
def test_scatter_models_Al_1cm_1GeV_theta_x_pi4():
    with h5py.File(PKG_DIR / "data/geant_scatter_validation.hdf5", "r") as geant_data:
        results = get_scatters(geant_data["Al_1cm_1GeV_ZenithAngle=pi4"], plots=False, verbose=True, n_steps=1)
        assert check_scatter_tests(results["tests"], pdg_pmin=0.20, pgeant_pmin=None, ignore_vars=["dspace_vol", "dangle_vol"])
        assert check_scatter_tests(
            results["tests"], pdg_pmin=0.15, pgeant_pmin=None, test="bulk95_mean_fdiff", ignore_vars=["dx_vol", "dy_vol", "dtheta_x_vol", "dtheta_y_vol"]
        )


@pytest.mark.flaky(max_runs=5, min_passes=1)
def test_scatter_models_Fe_1cm_1GeV_theta_x_pi4():
    with h5py.File(PKG_DIR / "data/geant_scatter_validation.hdf5", "r") as geant_data:
        results = get_scatters(geant_data["Fe_1cm_1GeV_ZenithAngle=pi4"], plots=False, verbose=True, n_steps=1)
        assert check_scatter_tests(results["tests"], pdg_pmin=0.15, pgeant_pmin=None, ignore_vars=["dspace_vol", "dangle_vol"])
        assert check_scatter_tests(
            results["tests"], pdg_pmin=0.1, pgeant_pmin=None, test="bulk95_mean_fdiff", ignore_vars=["dx_vol", "dy_vol", "dtheta_x_vol", "dtheta_y_vol"]
        )


@pytest.mark.flaky(max_runs=5, min_passes=1)
def test_scatter_models_Pb_1cm_1GeV_theta_x_pi4():
    with h5py.File(PKG_DIR / "data/geant_scatter_validation.hdf5", "r") as geant_data:
        results = get_scatters(geant_data["Pb_1cm_1GeV_ZenithAngle=pi4"], plots=False, verbose=True, n_steps=1)
        assert check_scatter_tests(results["tests"], pdg_pmin=0.15, pgeant_pmin=None, ignore_vars=["dspace_vol", "dangle_vol"])
        assert check_scatter_tests(
            results["tests"], pdg_pmin=0.1, pgeant_pmin=None, test="bulk95_mean_fdiff", ignore_vars=["dx_vol", "dy_vol", "dtheta_x_vol", "dtheta_y_vol"]
        )


@pytest.mark.flaky(max_runs=5, min_passes=1)
def test_scatter_models_Cu_1cm_1GeV_theta_x_pi4():
    with h5py.File(PKG_DIR / "data/geant_scatter_validation.hdf5", "r") as geant_data:
        results = get_scatters(geant_data["Cu_1cm_1GeV_ZenithAngle=pi4"], plots=False, verbose=True, n_steps=1)
        assert check_scatter_tests(results["tests"], pdg_pmin=0.15, pgeant_pmin=None, ignore_vars=["dspace_vol", "dangle_vol"])
        assert check_scatter_tests(
            results["tests"], pdg_pmin=0.1, pgeant_pmin=None, test="bulk95_mean_fdiff", ignore_vars=["dx_vol", "dy_vol", "dtheta_x_vol", "dtheta_y_vol"]
        )


@pytest.mark.flaky(max_runs=5, min_passes=1)
def test_scatter_models_U_1cm_1GeV_theta_x_pi4():
    with h5py.File(PKG_DIR / "data/geant_scatter_validation.hdf5", "r") as geant_data:
        results = get_scatters(geant_data["U_1cm_1GeV_ZenithAngle=pi4"], plots=False, verbose=True, n_steps=1)
        assert check_scatter_tests(results["tests"], pdg_pmin=0.15, pgeant_pmin=None, ignore_vars=["dspace_vol", "dangle_vol"])
        assert check_scatter_tests(
            results["tests"], pdg_pmin=0.1, pgeant_pmin=None, test="bulk95_mean_fdiff", ignore_vars=["dx_vol", "dy_vol", "dtheta_x_vol", "dtheta_y_vol"]
        )


# Multiple step tests
@pytest.mark.flaky(max_runs=5, min_passes=1)
def test_scatter_models_Al_50cm_50GeV_theta_x_pi4():
    with h5py.File(PKG_DIR / "data/geant_scatter_validation.hdf5", "r") as geant_data:
        results = get_scatters(geant_data["Al_50cm_50GeV_ZenithAngle=pi4"], plots=False, verbose=True, n_steps=50)
        assert check_scatter_tests(results["tests"], pdg_pmin=0.20, pgeant_pmin=None, ignore_vars=["dspace_vol", "dangle_vol"])
        assert check_scatter_tests(
            results["tests"], pdg_pmin=0.15, pgeant_pmin=None, test="bulk95_mean_fdiff", ignore_vars=["dx_vol", "dy_vol", "dtheta_x_vol", "dtheta_y_vol"]
        )


@pytest.mark.flaky(max_runs=5, min_passes=1)
def test_scatter_models_Fe_50cm_50GeV_theta_x_pi4():
    with h5py.File(PKG_DIR / "data/geant_scatter_validation.hdf5", "r") as geant_data:
        results = get_scatters(geant_data["Fe_50cm_50GeV_ZenithAngle=pi4"], plots=False, verbose=True, n_steps=50)
        assert check_scatter_tests(results["tests"], pdg_pmin=0.15, pgeant_pmin=None, ignore_vars=["dspace_vol", "dangle_vol"])
        assert check_scatter_tests(
            results["tests"], pdg_pmin=0.1, pgeant_pmin=None, test="bulk95_mean_fdiff", ignore_vars=["dx_vol", "dy_vol", "dtheta_x_vol", "dtheta_y_vol"]
        )


@pytest.mark.flaky(max_runs=5, min_passes=1)
def test_scatter_models_Pb_50cm_50GeV_theta_x_pi4():
    with h5py.File(PKG_DIR / "data/geant_scatter_validation.hdf5", "r") as geant_data:
        results = get_scatters(geant_data["Pb_50cm_50GeV_ZenithAngle=pi4"], plots=False, verbose=True, n_steps=50)
        assert check_scatter_tests(results["tests"], pdg_pmin=0.15, pgeant_pmin=None, ignore_vars=["dspace_vol", "dangle_vol"])
        assert check_scatter_tests(
            results["tests"], pdg_pmin=0.15, pgeant_pmin=None, test="bulk95_mean_fdiff", ignore_vars=["dx_vol", "dy_vol", "dtheta_x_vol", "dtheta_y_vol"]
        )


@pytest.mark.flaky(max_runs=5, min_passes=1)
def test_scatter_models_Cu_50cm_50GeV_theta_x_pi4():
    with h5py.File(PKG_DIR / "data/geant_scatter_validation.hdf5", "r") as geant_data:
        results = get_scatters(geant_data["Cu_50cm_50GeV_ZenithAngle=pi4"], plots=False, verbose=True, n_steps=50)
        assert check_scatter_tests(results["tests"], pdg_pmin=0.15, pgeant_pmin=None, ignore_vars=["dspace_vol", "dangle_vol"])
        assert check_scatter_tests(
            results["tests"], pdg_pmin=0.1, pgeant_pmin=None, test="bulk95_mean_fdiff", ignore_vars=["dx_vol", "dy_vol", "dtheta_x_vol", "dtheta_y_vol"]
        )
