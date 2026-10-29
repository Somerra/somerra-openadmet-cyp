import sys
import time

from somerra_openadmet.briford import read
from somerra_openadmet.briford_parallel import execute, wave


def test_wave_really_runs_concurrently_and_accounts_both_workers(tmp_path):
    script = tmp_path / "barrier.py"
    script.write_text(
        "import pathlib,sys,time\n"
        "root=pathlib.Path(sys.argv[1]); fold=sys.argv[2]\n"
        "(root/f'{fold}.started').touch()\n"
        "deadline=time.monotonic()+3\n"
        "while len(list(root.glob('*.started'))) < 2:\n"
        " if time.monotonic()>deadline: sys.exit(2)\n"
        " time.sleep(.01)\n"
        "time.sleep(.05)\n"
    )
    series = {"limit_seconds": 10.0, "used_seconds": 1.0, "attempts": []}

    def command(root, member, fold, seconds):
        assert seconds == 4.5
        return [sys.executable, str(script), str(root), str(fold)]

    assert wave(tmp_path, [("test", 0), ("test", 1)], series, command)
    saved = read(tmp_path / "series.json")
    assert "active_parallel" not in saved
    assert len(saved["attempts"]) == 2
    assert saved["used_seconds"] == 1 + sum(a["used_seconds"] for a in saved["attempts"])
    assert all(a["returncode"] == 0 for a in saved["attempts"])


def test_timeout_is_bounded_and_charged(tmp_path):
    started = time.monotonic()
    result = execute([sys.executable, "-c", "import time; time.sleep(10)"], tmp_path / "log", 0.1)
    assert result["returncode"] == 124
    assert 0.1 <= result["used_seconds"] < 3
    assert time.monotonic() - started < 3


def test_failed_wave_preserves_success_and_sets_failure(tmp_path):
    series = {"limit_seconds": 10.0, "used_seconds": 0.0, "attempts": []}

    def command(root, member, fold, seconds):
        return [sys.executable, "-c", f"raise SystemExit({fold})"]

    assert not wave(tmp_path, [("test", 0), ("test", 1)], series, command)
    saved = read(tmp_path / "series.json")
    assert saved["status"] == "failed"
    assert sorted(a["returncode"] for a in saved["attempts"]) == [0, 1]
    assert saved["used_seconds"] > 0
