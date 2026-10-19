import pytest

from somerra_openadmet.data_sources.common import allowed_auxiliary, potency


def test_potency_reverses_censoring_and_units():
    assert potency(100, "nM", ">") == (7.0, "<")
    assert potency(0.1, "uM", "<=") == (7.0, ">=")
    with pytest.raises(ValueError):
        potency(0, "nM")


def test_auxiliary_excludes_heldout_parent_and_close_analog():
    assert allowed_auxiliary(["CCO", "c1ccccc1", "CCCCCCCCO"], ["CCO"]).tolist() == [
        False,
        True,
        True,
    ]
    assert allowed_auxiliary(["CCO"], []).tolist() == [True]
