import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tomopt.core import DENSITIES, X0, A, B, Z, mean_excitation_E
from tomopt.optimisation.data.passives import (
    AbsPassiveGenerator,
    BlockPresentPassiveGenerator,
    PassiveYielder,
    RandomBlockPassiveGenerator,
    VoxelPassiveGenerator,
)
from tomopt.volume import DetectorPanel, PanelDetectorLayer, PassiveLayer, Volume

LW = Tensor([1, 1])
SZ = 0.1
N = 100
# Z = 1
INIT_RES = 1e4
INIT_EFF = 0.5
N_PANELS = 4


def arb_properties(*, z: float, lw: Tensor, size: float) -> Tensor:
    props = [X0, B, Z, A, DENSITIES, mean_excitation_E]  # noqa F405
    prop = lw.new_empty((6, int(lw[0].item() / size), int(lw[1].item() / size)))
    for i, p in enumerate(props):
        prop[i] = torch.ones(list((lw / size).long())) * p["alumnium"]
        if z >= 0.5:
            prop[i][3:7, 3:7] = p["lead"]
    return prop


def eff_cost(x: Tensor) -> Tensor:
    return torch.expm1(3 * F.relu(x))


def res_cost(x: Tensor) -> Tensor:
    return F.relu(x / 100) ** 2


def get_panel_layers() -> nn.ModuleList:
    layers = []
    layers.append(
        PanelDetectorLayer(
            pos="above",
            lw=LW,
            z=1,
            size=2 * SZ,
            panels=[
                DetectorPanel(res=INIT_RES, eff=INIT_EFF, init_xyz=[0.5, 0.5, 1 - (i * (2 * SZ) / N_PANELS)], init_xy_span=[1.0, 1.0]) for i in range(N_PANELS)
            ],
        )
    )
    for z in [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        layers.append(PassiveLayer(properties_func=arb_properties, lw=LW, z=z, size=SZ))
    layers.append(
        PanelDetectorLayer(
            pos="below",
            lw=LW,
            z=0.2,
            size=2 * SZ,
            panels=[
                DetectorPanel(res=INIT_RES, eff=INIT_EFF, init_xyz=[0.5, 0.5, 0.2 - (i * (2 * SZ) / N_PANELS)], init_xy_span=[1.0, 1.0])
                for i in range(N_PANELS)
            ],
        )
    )

    return nn.ModuleList(layers)


@pytest.fixture
def volume() -> Volume:
    return Volume(get_panel_layers())


def test_passive_yielder_list():
    passives = range(10)
    py = PassiveYielder(passives=list(passives), shuffle=False)
    assert len(py) == 10
    for i, p in enumerate(py):
        assert p[0] == passives[i]
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
    for j, (p, t) in enumerate(py):
        assert p() == j
        i += 1
        assert t is None


def test_abs_passive_generator(volume):
    class PG(AbsPassiveGenerator):
        def _generate(self):
            pass

    pg = PG(volume)
    assert set(pg.materials) == set([m for m in X0])

    pg = PG(volume, ["iron", "graphite"])
    assert pg.materials == ["iron", "graphite"]


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


def test_random_block_passive_generator(volume):
    mats = ["lead", "graphite"]
    x0s = [X0[m] for m in mats]
    pg = RandomBlockPassiveGenerator(volume, block_size=[0.6, 0.4, 0.2], sort_x0=True, enforce_diff_mat=True, materials=mats)
    for _ in range(10):
        passive, targ = pg.get_data()
        vol = 0
        for z in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            layer = passive(z=z, lw=LW, size=0.1)
            print(z, layer)
            vol += (layer == x0s[0]).sum()
            print((layer == x0s[0]).sum(), vol)
        assert vol == 48  # Entirety of the block is present
        assert targ == X0["lead"]

    pg = RandomBlockPassiveGenerator(volume, block_size=None, sort_x0=True, enforce_diff_mat=True, materials=mats, block_size_max_half=True)
    for _ in range(10):
        passive, targ = pg.get_data()
        vol = 0
        for z in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            layer = passive(z=z, lw=LW, size=0.1)
            print(z, layer)
            vol += (layer == x0s[0]).sum()
            print((layer == x0s[0]).sum(), vol)
        assert 0 <= vol <= 75
        assert targ == X0["lead"]


def test_block_present_passive_generator(volume):
    mats = ["lead", "graphite"]
    x0s = [X0[m] for m in mats]
    pg = BlockPresentPassiveGenerator(volume, block_size=[0.6, 0.4, 0.2], materials=mats)
    for _ in range(10):
        passive, targ = pg.get_data()
        vol = 0
        for z in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            layer = passive(z=z, lw=LW, size=0.1)
            print(z, layer)
            vol += (layer == x0s[1]).sum()
            print((layer == x0s[1]).sum(), vol)
        assert (vol == 48 and targ == x0s[1]) or (vol == 0 and targ == x0s[0])
