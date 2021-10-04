import pytest

from torch import Tensor

from tomopt.core import X0
from tomopt.optimisation.data.passives import PassiveYielder, VoxelPassiveGenerator, BlockPassiveGenerator, AbsPassiveGenerator


def test_passive_yielder_list():
    passives = range(10)
    py = PassiveYielder(passives=list(passives), shuffle=False)
    assert len(py) == 10
    for i, p in enumerate(py):
        assert p == passives[i]
    py.shuffle = True
    sp = [p for p in py]
    assert sp != passives


def test_passive_yielder_generator():
    i = 0

    class PG(AbsPassiveGenerator):
        def generate(self):
            return lambda: i

    pg = PG()
    py = PassiveYielder(passives=pg, n_passives=3)
    assert len(py) == 3
    for j, p in enumerate(py):
        assert p() == j
        i += 1


def test_abs_passive_generator():
    class PG(AbsPassiveGenerator):
        def generate(self):
            pass

    pg = PG()
    assert set(pg.materials) == set([m for m in X0])

    pg = PG(["iron", "carbon"])
    assert pg.materials == ["iron", "carbon"]


@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_voxel_passive_generator():
    pg = VoxelPassiveGenerator()
    passive = pg.generate()
    layer = passive(z=0.3, lw=Tensor([1.0, 1.0]), size=0.1)
    assert len(set(layer.flatten().tolist())) >= len(X0) / 2

    pg = VoxelPassiveGenerator(materials=["lead", "iron"])
    passive = pg.generate()
    layer = passive(z=0.3, lw=Tensor([1.0, 1.0]), size=0.1)
    assert len(set(layer.flatten().tolist())) == 2

    assert (pg.generate()(z=0.3, lw=Tensor([1.0, 1.0]), size=0.1) != layer).any()


def test_block_passive_generator():
    lw = Tensor([1.0, 1.0])
    z_range = [0.2, 0.8]
    mats = ["lead", "carbon"]
    x0s = [X0[m] for m in mats]
    pg = BlockPassiveGenerator(lw=lw, z_range=z_range, block_size=[0.6, 0.4, 0.2], sort_x0=True, materials=mats)
    for _ in range(10):
        passive = pg.generate()
        vol = 0
        for z in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            layer = passive(z=z, lw=lw, size=0.1)
            print(z, layer)
            vol += (layer == x0s[0]).sum()
            print((layer == x0s[0]).sum(), vol)
        assert vol == 48  # Entirety of the block is present
