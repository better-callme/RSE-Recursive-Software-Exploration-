from forgemind.core.config import ForgeMindConfig, ScoringWeights

import dataclasses


def test_config_immutable():
    cfg = ForgeMindConfig()
    try:
        cfg.beam_width = 10
        raised = False
    except dataclasses.FrozenInstanceError:
        raised = True
    assert raised


def test_config_validation():
    import pytest
    with pytest.raises(ValueError):
        ForgeMindConfig(beam_width=0)
    with pytest.raises(ValueError):
        ForgeMindConfig(branching_factor=-1)


def test_config_serialization_deterministic():
    a = ForgeMindConfig(random_seed=1).as_dict()
    b = ForgeMindConfig(random_seed=1).as_dict()
    assert dict(a) == dict(b)
    assert "beam_width" in a and "scoring_weights" in a


def test_scoring_weights_defaults():
    w = ScoringWeights()
    assert 0 < w.test_pass_rate <= 2.0
