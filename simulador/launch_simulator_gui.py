from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from telemetria.plot_dataset_on_guide import build_guide_centerline

SIMULATOR_PATH = PROJECT_ROOT / "externals" / "Ciberfisica_IA" / "Simulador_line_follower" / "line_follower_pyqt.py"
TRACK_PATH = PROJECT_ROOT / "digital_twin" / "assets" / "track_user_1200.png"
PRESET_PATH = PROJECT_ROOT / "digital_twin" / "presets" / "test_3_final_realmatch.json"


def load_module():
    spec = importlib.util.spec_from_file_location("lf_sim", SIMULATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def maybe_attach_guide(simulator, track, config) -> None:
    guide_cfg = config.get("guide", {})
    if not guide_cfg.get("enabled", False):
        return
    gx, gy, *_ = build_guide_centerline(track.w, track.h)
    points_m = [(float(x) / 1000.0, float(y) / 1000.0) for x, y in zip(gx, gy)]
    simulator.set_guide_path(points_m)


def main() -> None:
    sim_mod = load_module()
    app = sim_mod.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(sim_mod.QFont("Sans Serif", 9))
    win = sim_mod.MainWindow()

    preset_path = PRESET_PATH
    if len(sys.argv) > 1:
        candidate = Path(sys.argv[1]).expanduser()
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        preset_path = candidate

    preset = json.loads(preset_path.read_text(encoding="utf-8"))
    merged = copy.deepcopy(sim_mod.DEFAULT_CONFIG)
    for key, value in preset["config"].items():
        if isinstance(value, dict) and key in merged:
            merged[key].update(value)
        else:
            merged[key] = value

    win.config = merged
    win._populate_widgets()
    win._sync_config()
    win.track = sim_mod.Track(str(TRACK_PATH.resolve()))
    win.track_view.set_track(win.track)
    win.simulator = sim_mod.Simulator(win.config, win.track)
    maybe_attach_guide(win.simulator, win.track, win.config)
    win.track_view.set_simulator(win.simulator)
    win._update_track_info()
    win.statusBar().showMessage(
        f"Gemelo digital listo con preset {preset_path.name} y pista local: {TRACK_PATH.name}"
    )
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
