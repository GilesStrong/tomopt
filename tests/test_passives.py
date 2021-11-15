import pytest
import numpy as np

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from tomopt.core import X0
from tomopt.optimisation.data.passives import (
    PassiveYielder,
    VoxelPassiveGenerator,
    RandomBlockPassiveGenerator,
    BlockPresentPassiveGenerator,
    AbsPassiveGenerator,
)
from tomopt.volume import Volume, PassiveLayer, VoxelDetectorLayer, PanelDetectorLayer, DetectorPanel

LW = Tensor([1, 1])
SZ = 0.1
N = 100
Z = 1
INIT_RES = 1e4
INIT_EFF = 0.5
N_PANELS = 4


def arb_rad_length(*, z: float, lw: Tensor, size: float) -> float:
    rad_length = torch.ones(list((lw / size).long())) * X0["aluminium"]
    if z >= 0.5:
        rad_length[3:7, 3:7] = X0["lead"]
    return rad_length


def eff_cost(x: Tensor) -> Tensor:
    return torch.expm1(3 * F.relu(x))


def res_cost(x: Tensor) -> Tensor:
    return F.relu(x / 100) ** 2


def area_cost(x: Tensor) -> Tensor:
    return F.relu(x) ** 2


def get_voxel_layers() -> nn.ModuleList:
    layers = []

    pos = "above"
    for z, d in zip(np.arange(Z, 0, -SZ), [1, 1, 0, 0, 0, 0, 0, 0, 1, 1]):
        if d:
            layers.append(
                VoxelDetectorLayer(pos=pos, init_eff=INIT_EFF, init_res=INIT_RES, lw=LW, z=z, size=SZ, eff_cost_func=eff_cost, res_cost_func=res_cost)
            )
        else:
            pos = "below"
            layers.append(PassiveLayer(rad_length_func=arb_rad_length, lw=LW, z=z, size=SZ))

    return nn.ModuleList(layers)


def get_panel_layers() -> nn.ModuleList:
    layers = []
    layers.append(
        PanelDetectorLayer(
            pos="above",
            lw=LW,
            z=1,
            size=2 * SZ,
            panels=[
                DetectorPanel(res=INIT_RES, eff=INIT_EFF, init_xyz=[0.5, 0.5, 1 - (i * (2 * SZ) / N_PANELS)], init_xy_span=[0.5, 0.5], area_cost_func=area_cost)
                for i in range(N_PANELS)
            ],
        )
    )
    for z in [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        layers.append(PassiveLayer(rad_length_func=arb_rad_length, lw=LW, z=z, size=SZ))
    layers.append(
        PanelDetectorLayer(
            pos="below",
            lw=LW,
            z=0.2,
            size=2 * SZ,
            panels=[
                DetectorPanel(
                    res=INIT_RES, eff=INIT_EFF, init_xyz=[0.5, 0.5, 0.2 - (i * (2 * SZ) / N_PANELS)], init_xy_span=[0.5, 0.5], area_cost_func=area_cost
                )
                for i in range(N_PANELS)
            ],
        )
    )

    return nn.ModuleList(layers)


@pytest.fixture
def volume() -> Volume:
    return Volume(get_voxel_layers())


def test_passive_yielder_list():
    passives = range(10)
    py = PassiveYielder(passives=list(passives), shuffle=False)
    assert len(py) == 10
    for i, p in enumerate(py):
        assert p == passives[i]
    py.shuffle = True
    sp = [p for p in py]
    assert sp != passives


def test_passive_yielder_generator(volume):
    i = 0

    class PG(AbsPassiveGenerator):
        def _generate(self):
            return lambda: i, None

    pg = PG(volume)
    py = PassiveYielder(passives=pg, n_passives=3)
    assert len(py) == 3
    for j, p in enumerate(py):
        assert p() == j
        i += 1


def test_abs_passive_generator(volume):
    class PG(AbsPassiveGenerator):
        def _generate(self):
            pass

    pg = PG(volume)
    assert set(pg.materials) == set([m for m in X0])

    pg = PG(volume, ["iron", "carbon"])
    assert pg.materials == ["iron", "carbon"]


@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_voxel_passive_generator(volume):
    pg = VoxelPassiveGenerator(volume)
    passive = pg.generate()
    layer = passive(z=0.3, lw=LW, size=0.1)
    assert len(set(layer.flatten().tolist())) >= len(X0) / 2
    assert pg.get_data()[1] is None

    pg = VoxelPassiveGenerator(volume, materials=["lead", "iron"])
    passive = pg.generate()
    layer = passive(z=0.3, lw=LW, size=0.1)
    assert len(set(layer.flatten().tolist())) == 2

    assert (pg.generate()(z=0.3, lw=LW, size=0.1) != layer).any()


def test_block_passive_generator(volume):
    mats = ["lead", "carbon"]
    x0s = [X0[m] for m in mats]
    pg = RandomBlockPassiveGenerator(volume, block_size=[0.6, 0.4, 0.2], sort_x0=True, enforce_diff_mat=True, materials=mats)
    for _ in range(10):
        passive = pg.generate()
        vol = 0
        for z in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            layer = passive(z=z, lw=LW, size=0.1)
            print(z, layer)
            vol += (layer == x0s[0]).sum()
            print((layer == x0s[0]).sum(), vol)
        assert vol == 48  # Entirety of the block is present
        assert pg.get_data()[1] == X0["lead"]
