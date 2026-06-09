from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from telemetria.plot_dataset_on_guide import build_guide_centerline

SIMULATOR_PATH = PROJECT_ROOT / "externals" / "Ciberfisica_IA" / "Simulador_line_follower" / "line_follower_pyqt.py"
TRACK_PATH = PROJECT_ROOT / "digital_twin" / "assets" / "track_user_1200.png"
REAL_REFERENCE_PATH = PROJECT_ROOT / "digital_twin" / "real_robot_reference.json"
OUTPUT_DIR = PROJECT_ROOT / "digital_twin" / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta el gemelo digital en modo headless y exporta evidencias.")
    parser.add_argument("--preset", required=True, help="Archivo JSON del preset a ejecutar.")
    parser.add_argument("--duration-s", type=float, default=None, help="Duracion de la prueba en segundos. Si se omite, usa la referencia real.")
    parser.add_argument("--tag", default="", help="Prefijo corto para nombres de salida.")
    return parser.parse_args()


def load_simulator_module():
    spec = importlib.util.spec_from_file_location("lf_sim", SIMULATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_track_length_m(track) -> float:
    guide_x, guide_y, *_ = build_guide_centerline(track.w, track.h)
    total_px = 0.0
    for x0, y0, x1, y1 in zip(guide_x[:-1], guide_y[:-1], guide_x[1:], guide_y[1:]):
        total_px += math.hypot(x1 - x0, y1 - y0)
    return total_px / 1000.0


def maybe_attach_guide(simulator, track, config) -> None:
    guide_cfg = config.get("guide", {})
    if not guide_cfg.get("enabled", False):
        return
    guide_x, guide_y, *_ = build_guide_centerline(track.w, track.h)
    points_m = [(float(x) / 1000.0, float(y) / 1000.0) for x, y in zip(guide_x, guide_y)]
    simulator.set_guide_path(points_m)


def export_csv(history: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def build_overlay(track_path: Path, history: list[dict], output_png: Path, title: str) -> None:
    img = plt.imread(track_path)
    fig, ax = plt.subplots(figsize=(8.2, 8.2), constrained_layout=True)
    ax.imshow(img, origin="upper")
    if history:
        xs = [item["x"] * 1000.0 for item in history]
        ys = [item["y"] * 1000.0 for item in history]
        ax.plot(xs, ys, color="#00a651", linewidth=2.0, label="trayectoria simulada")
        ax.scatter([xs[0]], [ys[0]], s=90, color="#1d4ed8", edgecolor="white", linewidth=1.0, label="inicio")
        ax.scatter([xs[-1]], [ys[-1]], s=90, color="#dc2626", edgecolor="white", linewidth=1.0, label="fin")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower right")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def export_metrics(sim_metrics: dict, real_reference: dict, history: list[dict], track_length_m: float, output_json: Path) -> dict:
    real_metrics = real_reference["test_metrics_real"]
    real_avg_lap_time = float(real_metrics["average_lap_time_s"])
    real_avg_speed = track_length_m / max(real_avg_lap_time, 1e-6)
    sim_time_s = float(sim_metrics.get("tiempo_s", 0.0))
    sim_distance_m = float(sim_metrics.get("distancia_m", 0.0))
    sim_speed = float(sim_metrics.get("vel_media_mps", 0.0))
    sim_laps_equivalent = sim_distance_m / max(track_length_m, 1e-6)
    sim_osc = int(sim_metrics.get("oscilaciones", 0))
    sim_on_line = float(sim_metrics.get("porc_en_linea", 0.0)) / 100.0
    sim_line_lost_ratio = 1.0 - sim_on_line
    payload = {
        "real_reference": real_reference,
        "simulation_metrics": {
            "time_s": sim_time_s,
            "distance_m": sim_distance_m,
            "track_length_m_approx": track_length_m,
            "laps_equivalent": sim_laps_equivalent,
            "average_lap_time_s_estimated": sim_time_s / max(sim_laps_equivalent, 1e-6),
            "average_speed_mps": sim_speed,
            "rms_error": float(sim_metrics.get("rms_error", 0.0)),
            "max_error": float(sim_metrics.get("max_error", 0.0)),
            "oscillations_total": sim_osc,
            "oscillations_per_second": sim_osc / max(sim_time_s, 1e-6),
            "on_line_ratio": sim_on_line,
            "line_lost_ratio": sim_line_lost_ratio,
            "history_rows": len(history),
        },
        "comparison": {
            "real_average_speed_mps_estimated": real_avg_speed,
            "delta_speed_mps": sim_speed - real_avg_speed,
            "delta_laps_equivalent_vs_real_completed": sim_laps_equivalent - float(real_metrics["completed_laps"]),
            "delta_average_lap_time_s": (sim_time_s / max(sim_laps_equivalent, 1e-6)) - float(real_metrics["average_lap_time_s"]),
            "delta_line_lost_ratio": sim_line_lost_ratio - float(real_metrics["line_lost_ratio"]),
            "delta_oscillations_per_second": (sim_osc / max(sim_time_s, 1e-6)) - float(real_metrics["oscillations_per_second"]),
        },
    }
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    preset_path = Path(args.preset).expanduser().resolve()
    if not preset_path.exists():
        raise SystemExit(f"No existe el preset: {preset_path}")

    sim_mod = load_simulator_module()
    preset_obj = load_json(preset_path)
    config = preset_obj["config"]
    real_reference = load_json(REAL_REFERENCE_PATH)
    track = sim_mod.Track(str(TRACK_PATH.resolve()))
    simulator = sim_mod.Simulator(config, track)
    maybe_attach_guide(simulator, track, config)

    duration_s = (
        float(args.duration_s)
        if args.duration_s is not None
        else float(real_reference["test_metrics_real"]["total_time_s_completed_laps"])
    )
    steps = int(duration_s / 0.02)
    for _ in range(steps):
        simulator.step(0.02)

    metrics = simulator.metrics()
    tag = args.tag.strip() or preset_path.stem
    csv_path = OUTPUT_DIR / f"{tag}_telemetry.csv"
    metrics_path = OUTPUT_DIR / f"{tag}_metrics.json"
    overlay_path = OUTPUT_DIR / f"{tag}_overlay.png"

    export_csv(simulator.history, csv_path)
    track_length_m = compute_track_length_m(track)
    export_metrics(metrics, real_reference, simulator.history, track_length_m, metrics_path)
    build_overlay(TRACK_PATH, simulator.history, overlay_path, f"Gemelo digital - {tag}")

    print(f"Preset: {preset_path.name}")
    print(f"CSV: {csv_path}")
    print(f"Overlay: {overlay_path}")
    print(f"Metricas: {metrics_path}")
    print(f"Resumen: tiempo={float(metrics.get('tiempo_s', 0.0)):.2f}s, vel={float(metrics.get('vel_media_mps', 0.0)):.3f}m/s, oscilaciones={int(metrics.get('oscilaciones', 0))}")


if __name__ == "__main__":
    main()
