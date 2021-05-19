from tomopt.optimisation.data.passives import PassiveYielder


def test_passive_yielder():
    passives = range(10)
    py = PassiveYielder(passives=list(passives), shuffle=False)
    assert len(py) == 10
    for i, p in enumerate(py):
        assert p == passives[i]
    py.shuffle = True
    sp = [p for p in py]
    assert sp != passives
