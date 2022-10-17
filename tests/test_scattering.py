from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from collections import defaultdict
from prettytable import PrettyTable
from scipy.stats import ks_2samp
import h5py
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pytest
import os
from fastcore.all import Path

import torch
from torch import Tensor

from tomopt.volume import SCATTER_MODEL, PassiveLayer
from tomopt.muon import MuonBatch
from tomopt.core import X0

PKG_DIR = Path(os.path.dirname(os.path.abspath(__file__)))  # How robust is this? Could create hidden dir in home and download resources


def get_scatters(grp: h5py.Group, plots: bool, verbose: bool) -> Dict[str, Any]:
    r"""
    file example Fe_1cm_5GeV_normal.txt
    """

    # Init layer
    layer = PassiveLayer(lw=Tensor([1, 1]), z=1, size=0.1)
    SCATTER_MODEL.load_data()

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

    # Get sims
    xy_m_t_p = torch.ones(n_muons, 5)
    xy_m_t_p[:, 2] = mom
    xy_m_t_p[:, 3] = theta
    xy_m_t_p[:, 4] = phi
    muons = MuonBatch(xy_m_t_p, init_z=1)
    x0 = torch.ones(len(muons)) * X0[mat]

    # New param model
    pgeant_scattering = layer._pgeant_scatter(x0=x0, deltaz=0.01, theta=muons.theta, theta_x=muons.theta_x, theta_y=muons.theta_y, mom=muons.mom)
    pgeant_scattering["dangle_vol"] = np.sqrt((pgeant_scattering["dtheta_vol"] ** 2) + (pgeant_scattering["dphi_vol"] ** 2))
    pgeant_scattering["dspace_vol"] = np.sqrt((pgeant_scattering["dx_vol"] ** 2) + (pgeant_scattering["dy_vol"] ** 2))

    # New PDG model
    pdg_scattering = layer._pdg_scatter(x0=x0, deltaz=0.01, theta=muons.theta, theta_x=muons.theta_x, theta_y=muons.theta_y, mom=muons.mom)
    pdg_scattering["dangle_vol"] = np.sqrt((pdg_scattering["dtheta_vol"] ** 2) + (pdg_scattering["dphi_vol"] ** 2))
    pdg_scattering["dspace_vol"] = np.sqrt((pdg_scattering["dx_vol"] ** 2) + (pdg_scattering["dy_vol"] ** 2))

    # Make plots and tests
    tests = defaultdict(lambda: defaultdict(dict))

    def get_ks(a, b) -> float:
        return ks_2samp(a, b).pvalue

    var = "dtheta_vol"
    if verbose:
        print("\n\n", var)
    tab: PrettyTable = None
    for name, sim in (("pdg", pdg_scattering), ("param_geant", pgeant_scattering)):
        tests[var][name]["mean"] = sim[var].mean()
        tests[var][name]["std_fdiff"] = np.abs(sim[var].std() - df[var].std()) / df[var].std()
        tests[var][name]["ks_p"] = get_ks(sim[var], df[var])
        cut95 = np.percentile(df[var].abs(), 95)
        tests[var][name]["bulk95_ks_p"] = get_ks(sim[var][sim[var].abs() <= cut95], df.loc[df[var].abs() <= cut95, var])
        cut68 = np.percentile(df[var].abs(), 68)
        tests[var][name]["bulk68_ks_p"] = get_ks(sim[var][sim[var].abs() <= cut68], df.loc[df[var].abs() <= cut68, var])
        if tab is None:
            tab = PrettyTable(["Sim"] + [f for f in tests[var][name]])
        tab.add_row([name] + [v for k, v in tests[var][name].items()])

    if plots:
        sns.distplot(df[var], label="True GEANT")
        sns.distplot(pgeant_scattering[var], label="Param GEANT")
        sns.distplot(pdg_scattering[var], label="PDG")
        cut99 = np.percentile(df[var], 99.9)
        plt.xlim([-cut99, cut99])
        plt.xlabel(var)
        plt.legend()
        plt.show()

        sns.distplot(df.loc[df[var].abs() > cut68, var].abs(), label="True GEANT")
        sns.distplot(pgeant_scattering[var][pgeant_scattering[var].abs() > cut68].abs(), label="Param GEANT")
        sns.distplot(pdg_scattering[var][pdg_scattering[var].abs() > cut68].abs(), label="PDG")
        plt.xlabel(f"|{var}|")
        plt.legend()
        plt.yscale("log")
        plt.ylim(1e-3, 1e4)
        plt.show()

    if verbose:
        print(tab)

    var = "dphi_vol"
    if verbose:
        print("\n\n", var)
    tab: PrettyTable = None
    for name, sim in (("pdg", pdg_scattering), ("param_geant", pgeant_scattering)):
        tests[var][name]["mean"] = sim[var].mean()
        tests[var][name]["std_fdiff"] = np.abs(sim[var].std() - df[var].std()) / df[var].std()
        tests[var][name]["ks_p"] = get_ks(sim[var], df[var])
        cut95 = np.percentile(df[var].abs(), 95)
        tests[var][name]["bulk95_ks_p"] = get_ks(sim[var][sim[var].abs() <= cut95], df.loc[df[var].abs() <= cut95, var])
        cut68 = np.percentile(df[var].abs(), 68)
        tests[var][name]["bulk68_ks_p"] = get_ks(sim[var][sim[var].abs() <= cut68], df.loc[df[var].abs() <= cut68, var])
        if tab is None:
            tab = PrettyTable(["Sim"] + [f for f in tests[var][name]])
        tab.add_row([name] + [v for k, v in tests[var][name].items()])

    if plots:
        sns.distplot(df[var], label="True GEANT")
        sns.distplot(pgeant_scattering[var], label="Param GEANT")
        sns.distplot(pdg_scattering[var], label="PDG")
        cut99 = np.percentile(df[var], 99.9)
        plt.xlim([-cut99, cut99])
        plt.xlabel(var)
        plt.legend()
        plt.show()

        sns.distplot(df.loc[df[var].abs() > cut68, var].abs(), label="True GEANT")
        sns.distplot(pgeant_scattering[var][pgeant_scattering[var].abs() > cut68].abs(), label="Param GEANT")
        sns.distplot(pdg_scattering[var][pdg_scattering[var].abs() > cut68].abs(), label="PDG")
        plt.xlabel(f"|{var}|")
        plt.legend()
        plt.yscale("log")
        plt.ylim(1e-3, 1e4)
        plt.show()

    if verbose:
        print(tab)

    var = "dx_vol"
    if verbose:
        print("\n\n", var)
    tab: PrettyTable = None
    for name, sim in (("pdg", pdg_scattering), ("param_geant", pgeant_scattering)):
        tests[var][name]["mean"] = sim[var].mean()
        tests[var][name]["std_fdiff"] = np.abs(sim[var].std() - df[var].std()) / df[var].std()
        tests[var][name]["ks_p"] = get_ks(sim[var], df[var])
        cut95 = np.percentile(df[var].abs(), 95)
        tests[var][name]["bulk95_ks_p"] = get_ks(sim[var][sim[var].abs() <= cut95], df.loc[df[var].abs() <= cut95, var])
        cut68 = np.percentile(df[var].abs(), 68)
        tests[var][name]["bulk68_ks_p"] = get_ks(sim[var][sim[var].abs() <= cut68], df.loc[df[var].abs() <= cut68, var])
        if tab is None:
            tab = PrettyTable(["Sim"] + [f for f in tests[var][name]])
        tab.add_row([name] + [v for k, v in tests[var][name].items()])

    if plots:
        sns.distplot(df[var], label="True GEANT")
        sns.distplot(pgeant_scattering[var], label="Param GEANT")
        sns.distplot(pdg_scattering[var], label="PDG")
        cut99 = np.percentile(df[var], 99.9)
        plt.xlim([-cut99, cut99])
        plt.xlabel(var)
        plt.legend()
        plt.show()

        sns.distplot(df.loc[df[var].abs() > cut68, var].abs(), label="True GEANT")
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
        tests[var][name]["mean"] = sim[var].mean()
        tests[var][name]["std_fdiff"] = np.abs(sim[var].std() - df[var].std()) / df[var].std()
        tests[var][name]["ks_p"] = get_ks(sim[var], df[var])
        cut95 = np.percentile(df[var].abs(), 95)
        tests[var][name]["bulk95_ks_p"] = get_ks(sim[var][sim[var].abs() <= cut95], df.loc[df[var].abs() <= cut95, var])
        cut68 = np.percentile(df[var].abs(), 68)
        tests[var][name]["bulk68_ks_p"] = get_ks(sim[var][sim[var].abs() <= cut68], df.loc[df[var].abs() <= cut68, var])
        if tab is None:
            tab = PrettyTable(["Sim"] + [f for f in tests[var][name]])
        tab.add_row([name] + [v for k, v in tests[var][name].items()])

    if plots:
        sns.distplot(df[var], label="True GEANT")
        sns.distplot(pgeant_scattering[var], label="Param GEANT")
        sns.distplot(pdg_scattering[var], label="PDG")
        cut99 = np.percentile(df[var], 99.9)
        plt.xlim([-cut99, cut99])
        plt.xlabel(var)
        plt.legend()
        plt.show()

        sns.distplot(df.loc[df[var].abs() > cut68, var].abs(), label="True GEANT")
        sns.distplot(pgeant_scattering[var][pgeant_scattering[var].abs() > cut68].abs(), label="Param GEANT")
        sns.distplot(pdg_scattering[var][pdg_scattering[var].abs() > cut68].abs(), label="PDG")
        plt.xlabel(f"|{var}|")
        plt.legend()
        plt.yscale("log")
        plt.ylim(1e-1, 1e6)
        plt.show()

    if verbose:
        print(tab)

    return {"data": {"df": df, "param_geant": pgeant_scattering, "pdg": pdg_scattering}, "tests": tests}


def check_scatter_tests(
    tests: Dict[str, Dict[str, Dict[str, float]]],
    pdg_pmin: Optional[float],
    pgeant_pmin: Optional[float],
    test: str = "bulk68_ks_p",
    ignore_vars: Optional[List[str]] = None,
) -> bool:
    if ignore_vars is None:
        ignore_vars = []
    else:
        print(f"Ignoring {ignore_vars}")
    for var in [v for v in tests if v not in ignore_vars]:
        if pdg_pmin is not None:
            if tests[var]["pdg"][test] < pdg_pmin:
                print(f'{var}, failed for PDG {test} = {tests[var]["pdg"][test]}')
                return False
        if pgeant_pmin is not None:
            if tests[var]["param_geant"][test] < pgeant_pmin:
                print(f'{var}, failed for parameterised GEANT {test} = {tests[var]["param_geant"][test]}')
                return False
    return True


@pytest.mark.flaky(max_runs=5, min_passes=1)
def test_scatter_models_Al_1cm_5GeV_normal():
    with h5py.File(PKG_DIR / "data/geant_scatter_validation.hdf5", "r") as geant_data:
        results = get_scatters(geant_data["Al_1cm_5GeV_normal"], plots=False, verbose=False)
        assert check_scatter_tests(results["tests"], pdg_pmin=0.01, pgeant_pmin=None, ignore_vars=["dx_vol"])  # dx is strange in the geant sample


@pytest.mark.flaky(max_runs=5, min_passes=1)
def test_scatter_models_Cu_1cm_5GeV_normal():
    with h5py.File(PKG_DIR / "data/geant_scatter_validation.hdf5", "r") as geant_data:
        results = get_scatters(geant_data["Cu_1cm_5GeV_normal"], plots=False, verbose=False)
        assert check_scatter_tests(results["tests"], pdg_pmin=0.01, pgeant_pmin=None)


@pytest.mark.flaky(max_runs=5, min_passes=1)
def test_scatter_models_Fe_1cm_1GeV_normal():
    with h5py.File(PKG_DIR / "data/geant_scatter_validation.hdf5", "r") as geant_data:
        results = get_scatters(geant_data["Fe_1cm_1GeV_normal"], plots=False, verbose=False)
        assert check_scatter_tests(results["tests"], pdg_pmin=0.01, pgeant_pmin=None)


@pytest.mark.flaky(max_runs=5, min_passes=1)
def test_scatter_models_Fe_1cm_5GeV_normal():
    with h5py.File(PKG_DIR / "data/geant_scatter_validation.hdf5", "r") as geant_data:
        results = get_scatters(geant_data["Fe_1cm_5GeV_normal"], plots=False, verbose=False)
        assert check_scatter_tests(results["tests"], pdg_pmin=0.01, pgeant_pmin=None)


@pytest.mark.flaky(max_runs=5, min_passes=1)
def test_scatter_models_Fe_1cm_50GeV_normal():
    with h5py.File(PKG_DIR / "data/geant_scatter_validation.hdf5", "r") as geant_data:
        results = get_scatters(geant_data["Fe_1cm_50GeV_normal"], plots=False, verbose=False)
        assert check_scatter_tests(results["tests"], pdg_pmin=0.01, pgeant_pmin=None)


@pytest.mark.flaky(max_runs=5, min_passes=1)
def test_scatter_models_U_1cm_5GeV_normal():
    with h5py.File(PKG_DIR / "data/geant_scatter_validation.hdf5", "r") as geant_data:
        results = get_scatters(geant_data["U_1cm_5GeV_normal"], plots=False, verbose=False)
        assert check_scatter_tests(results["tests"], pdg_pmin=0.01, pgeant_pmin=None)


@pytest.mark.flaky(max_runs=5, min_passes=1)
def test_scatter_models_Fe_1cm_5GeV_piby4():
    with h5py.File(PKG_DIR / "data/geant_scatter_validation.hdf5", "r") as geant_data:
        results = get_scatters(geant_data["Fe_1cm_5GeV_ZenithAngle=pi4"], plots=False, verbose=False)
        assert check_scatter_tests(results["tests"], pdg_pmin=0.01, pgeant_pmin=None)
