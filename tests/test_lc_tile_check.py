"""lc_tile_check flags a lifted tile and names the state frame that caused it."""

import numpy as np
import tifffile

from microscope_command_server.diagnostics import lc_tile_check


def _make_run(tmp_path, n_tiles=40, bad_tile=7, bad_state="State2"):
    rng = np.random.default_rng(0)
    states = [f"State{i}" for i in range(5)]
    levels = {"State0": 300.0, "State1": 900.0, "State2": 900.0, "State3": 900.0, "State4": 900.0}
    for s in states + ["retardance"]:
        (tmp_path / s).mkdir()
    lines = ["dim = 2"]
    for t in range(n_tiles):
        name = f"{t}.tif"
        lines.append(f"{name}; ; ({t * 100.0}, {-t * 50.0})")
        scene = rng.uniform(0.95, 1.05)  # tissue changes all states together
        for s in states:
            level = levels[s] * scene * rng.normal(1.0, 0.002)
            if t == bad_tile and s == bad_state:
                level *= 0.94
            img = np.full((16, 16), level, dtype=np.float32)
            tifffile.imwrite(str(tmp_path / s / name), img.astype(np.uint16))
        floor = 2.0 + rng.normal(0, 0.1) + (6.0 if t == bad_tile else 0.0)
        tifffile.imwrite(str(tmp_path / "retardance" / name), np.full((16, 16), floor, np.float32))
    (tmp_path / "TileConfiguration_Stage.txt").write_text("\n".join(lines))
    return states


def test_flags_the_lifted_tile_and_names_its_state(tmp_path):
    states = _make_run(tmp_path)
    assert lc_tile_check.find_states(tmp_path) == states

    rows = lc_tile_check.measure(tmp_path, states)
    flagged = lc_tile_check.analyse(rows, states)

    assert [r["tile"] for r in flagged] == ["7.tif"]
    assert flagged[0]["worst_state"] == "State2"
    assert flagged[0]["worst_dev_pct"] < -4.0
    assert flagged[0]["stage_x_um"] == 700.0


def test_main_writes_csv_and_summary(tmp_path, capsys):
    _make_run(tmp_path)
    assert lc_tile_check.main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "1 flagged" in out
    assert "State2 x1" in out
    assert (tmp_path / "lc_tile_check.csv").is_file()


def test_no_state_folders(tmp_path):
    assert lc_tile_check.main([str(tmp_path)]) == 2
