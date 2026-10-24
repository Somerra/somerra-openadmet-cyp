from somerra_openadmet.data_sources.common import allowed_auxiliary


def test_auxiliary_excludes_heldout_parent_and_close_analog():
    assert allowed_auxiliary(["CCO", "c1ccccc1", "CCCCCCCCO"], ["CCO"]).tolist() == [
        False,
        True,
        True,
    ]
    assert allowed_auxiliary(["CCO"], []).tolist() == [True]
