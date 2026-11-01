from datetime import UTC, datetime, timedelta

import pytest

from somerra_openadmet.cyp_research import write
from somerra_openadmet.cyp_strategy import authorized_extension


def test_extension_preserves_frozen_allocation_and_requires_time(tmp_path):
    root = tmp_path / "research"
    campaign = root / "campaigns" / "CYP1A2"
    campaign.mkdir(parents=True)
    write(campaign / "campaign.json", {"trial_seconds": 580, "budget_gpu_hours": 0.5})
    before = (campaign / "campaign.json").read_bytes()
    path = tmp_path / "authorization.json"
    authorization = {
        "root": str(root),
        "deadline_utc": (datetime.now(UTC) + timedelta(hours=3)).isoformat(),
        "additional_budget_hours": {"CYP1A2": 3},
    }
    write(path, authorization)
    hours, digest = authorized_extension(path, root, "CYP1A2")
    assert hours == 3
    assert len(digest) == 64
    assert (campaign / "campaign.json").read_bytes() == before
    with pytest.raises(ValueError, match="another research root"):
        authorized_extension(path, tmp_path / "other", "CYP1A2")
    authorization["deadline_utc"] = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
    write(path, authorization)
    with pytest.raises(ValueError, match="Insufficient time"):
        authorized_extension(path, root, "CYP1A2")
    assert authorized_extension(None, root, "CYP1A2") == (0.0, None)
