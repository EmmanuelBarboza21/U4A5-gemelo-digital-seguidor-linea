from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_dataset_on_guide import GUIDE_AREA_CM, build_guide_centerline, clamp, interpolate_series


SESSION_NAME = "session_20260423_154428"
CAR_WIDTH_CM = 18.0
SENSOR_WEIGHTS = np.array([-1.5, -0.5, 0.5, 1.5], dtype=float)


@dataclass
class AnalysisOutputs:
    clean_csv: Path
    trajectory_png: Path
    comparison_png: Path
    sensor_plot_png: Path
    metrics_json: Path
    rows_raw_running: int
    rows_clean: int
    rows_removed_tail: int
    duplicates_removed: int
    missing_removed: int
    r2_x: float
    r2_y: float
    rmse_x_cm: float
    rmse_y_cm: float


def load_session_rows(dataset_path: Path, session_name: str) -> list[dict]:
    rows: list[dict] = []
    for line_no, raw in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            continue
        if str(item.get("dataset_session") or "").strip() != session_name:
            continue
        if str(item.get("state") or "").strip().lower() not in {"run", "running"}:
            continue
        rows.append(item)
    if not rows:
        raise SystemExit(f"No se encontraron muestras running para la sesion {session_name}.")
    return rows


def trim_to_completed_manual_laps(rows: list[dict]) -> tuple[list[dict], int]:
    target_laps = max(int(float(row.get("dataset_manual_lap_count") or 0)) for row in rows)
    if target_laps <= 0:
        return list(rows), 0

    keep: list[dict] = []
    completed = 0
    for row in rows:
        keep.append(row)
        if int(float(row.get("dataset_manual_lap_marked") or 0)) != 0:
            completed += 1
            if completed >= target_laps:
                break
    removed_tail = max(0, len(rows) - len(keep))
    return keep, removed_tail


def clean_dataframe(rows: list[dict]) -> tuple[pd.DataFrame, int, int]:
    df = pd.DataFrame(rows).copy()
    required_numeric = [
        "dataset_row_index",
        "uptime_ms",
        "run_elapsed_ms",
        "norm0",
        "norm1",
        "norm2",
        "norm3",
        "line_pos",
        "motor_a_pwm",
        "motor_b_pwm",
        "line_lost",
        "dataset_manual_lap_count",
        "dataset_manual_lap_marked",
    ]
    for column in required_numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    before_missing = len(df)
    df = df.dropna(subset=required_numeric).copy()
    missing_removed = before_missing - len(df)

    before_duplicates = len(df)
    df = df.drop_duplicates(subset=["uptime_ms", "run_elapsed_ms", "dataset_row_index"]).copy()
    duplicates_removed = before_duplicates - len(df)

    df = df.sort_values(["dataset_row_index", "uptime_ms"]).reset_index(drop=True)
    return df, duplicates_removed, missing_removed


def compute_weighted_line_position(df: pd.DataFrame) -> np.ndarray:
    sensors = df[["norm0", "norm1", "norm2", "norm3"]].to_numpy(dtype=float)
    denominator = np.clip(sensors.sum(axis=1), 1e-6, None)
    return (sensors * SENSOR_WEIGHTS).sum(axis=1) / denominator


def build_reference_coordinates(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    guide_width_px = 1664
    guide_height_px = 1664
    px_per_cm = guide_width_px / GUIDE_AREA_CM
    car_width_px = CAR_WIDTH_CM * px_per_cm
    guide_x, guide_y, normal_x, normal_y = build_guide_centerline(guide_width_px, guide_height_px)

    mark_indices = [
        index for index, value in enumerate(df["dataset_manual_lap_marked"].to_numpy(dtype=int)) if value != 0
    ]
    total_laps = int(df["dataset_manual_lap_count"].max()) if len(df) else 0
    if total_laps <= 0 or len(mark_indices) < total_laps:
        raise SystemExit("La sesion no tiene suficientes marcas manuales para reconstruir la practica.")

    segment_starts = [0] + [index + 1 for index in mark_indices[:-1]]
    segment_ends = mark_indices[:total_laps]
    lap_segments = list(zip(segment_starts, segment_ends))

    ref_x_px: list[float] = []
    ref_y_px: list[float] = []
    lap_phase_list: list[float] = []
    global_progress_list: list[float] = []

    for index, row in df.iterrows():
        lap_number = 0
        lap_phase = 0.0
        for segment_index, (start_idx, end_idx) in enumerate(lap_segments):
            if start_idx <= index <= end_idx:
                lap_number = segment_index
                segment_len = max(1, end_idx - start_idx)
                lap_phase = (index - start_idx) / float(segment_len)
                if index == end_idx:
                    lap_phase = 0.999
                break

        guide_pos = lap_phase * float(len(guide_x) - 1)
        base_x = interpolate_series(guide_x, guide_pos)
        base_y = interpolate_series(guide_y, guide_pos)
        nx = interpolate_series(normal_x, guide_pos)
        ny = interpolate_series(normal_y, guide_pos)

        line_offset = clamp(float(row["line_pos"]) / 1500.0, -1.0, 1.0) * car_width_px * 0.18
        ref_x_px.append(base_x + nx * line_offset)
        ref_y_px.append(base_y + ny * line_offset)
        lap_phase_list.append(lap_phase)
        global_progress_list.append((lap_number + lap_phase) / float(total_laps))

    ref_x_cm = np.array(ref_x_px, dtype=float) / px_per_cm
    ref_y_cm = np.array(ref_y_px, dtype=float) / px_per_cm
    return ref_x_cm, ref_y_cm, np.array(lap_phase_list, dtype=float), np.array(global_progress_list, dtype=float)


def build_feature_matrix(
    df: pd.DataFrame,
    weighted_position: np.ndarray,
    lap_phase: np.ndarray,
    global_progress: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    features = [
        np.ones(len(df), dtype=float),
        global_progress,
        lap_phase,
        weighted_position,
        df["norm0"].to_numpy(dtype=float) / 100.0,
        df["norm1"].to_numpy(dtype=float) / 100.0,
        df["norm2"].to_numpy(dtype=float) / 100.0,
        df["norm3"].to_numpy(dtype=float) / 100.0,
        df["motor_a_pwm"].to_numpy(dtype=float) / 255.0,
        df["motor_b_pwm"].to_numpy(dtype=float) / 255.0,
        df["line_lost"].to_numpy(dtype=float),
    ]
    names = [
        "bias",
        "global_progress",
        "lap_phase",
        "weighted_line_position",
        "sensor_1",
        "sensor_2",
        "sensor_3",
        "sensor_4",
        "motor_a_pwm",
        "motor_b_pwm",
        "line_lost",
    ]

    for harmonic in range(1, 5):
        features.append(np.sin(2.0 * math.pi * harmonic * lap_phase))
        features.append(np.cos(2.0 * math.pi * harmonic * lap_phase))
        names.append(f"sin_{harmonic}")
        names.append(f"cos_{harmonic}")

    return np.column_stack(features), names


def fit_linear_regression(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
    predictions = X @ coefficients
    return coefficients, predictions


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    rmse = np.sqrt((residual ** 2).mean(axis=0))
    ss_res = (residual ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / ss_tot
    return {
        "r2_x": float(r2[0]),
        "r2_y": float(r2[1]),
        "rmse_x_cm": float(rmse[0]),
        "rmse_y_cm": float(rmse[1]),
    }


def save_clean_csv(
    df: pd.DataFrame,
    weighted_position: np.ndarray,
    ref_x_cm: np.ndarray,
    ref_y_cm: np.ndarray,
    predicted_xy: np.ndarray,
    output_csv: Path,
) -> None:
    export_df = pd.DataFrame(
        {
            "sample_index": np.arange(len(df), dtype=int),
            "time_s": df["run_elapsed_ms"].to_numpy(dtype=float) / 1000.0,
            "sensor_1": df["norm0"].to_numpy(dtype=float),
            "sensor_2": df["norm1"].to_numpy(dtype=float),
            "sensor_3": df["norm2"].to_numpy(dtype=float),
            "sensor_4": df["norm3"].to_numpy(dtype=float),
            "line_position_weighted": weighted_position,
            "line_pos_firmware": df["line_pos"].to_numpy(dtype=float),
            "motor_a_pwm": df["motor_a_pwm"].to_numpy(dtype=float),
            "motor_b_pwm": df["motor_b_pwm"].to_numpy(dtype=float),
            "line_lost": df["line_lost"].to_numpy(dtype=int),
            "local_mode": df.get("local_mode", pd.Series(["unknown"] * len(df))).astype(str),
            "manual_lap_count": df["dataset_manual_lap_count"].to_numpy(dtype=int),
            "manual_lap_marked": df["dataset_manual_lap_marked"].to_numpy(dtype=int),
            "guide_x_cm": ref_x_cm,
            "guide_y_cm": ref_y_cm,
            "predicted_x_cm": predicted_xy[:, 0],
            "predicted_y_cm": predicted_xy[:, 1],
        }
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(output_csv, index=False, encoding="utf-8")


def save_sensor_plot(df: pd.DataFrame, weighted_position: np.ndarray, output_path: Path) -> None:
    time_s = df["run_elapsed_ms"].to_numpy(dtype=float) / 1000.0
    fig, (ax_sensors, ax_position) = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    ax_sensors.plot(time_s, df["norm0"], label="sensor 1", linewidth=1.2)
    ax_sensors.plot(time_s, df["norm1"], label="sensor 2", linewidth=1.2)
    ax_sensors.plot(time_s, df["norm2"], label="sensor 3", linewidth=1.2)
    ax_sensors.plot(time_s, df["norm3"], label="sensor 4", linewidth=1.2)
    ax_sensors.set_title("Lecturas de sensores analogicos durante la corrida", fontsize=13, fontweight="bold")
    ax_sensors.set_xlabel("Tiempo de corrida (s)")
    ax_sensors.set_ylabel("Lectura normalizada")
    ax_sensors.grid(True, alpha=0.25)
    ax_sensors.legend(loc="upper right")

    ax_position.plot(time_s, weighted_position, color="#111111", linewidth=1.6, label="posicion ponderada")
    ax_position.plot(time_s, df["line_pos"] / 1000.0, color="#d62728", linewidth=1.0, alpha=0.8, label="line_pos firmware / 1000")
    ax_position.set_title("Estimacion de posicion de linea con pesos [-1.5, -0.5, 0.5, 1.5]", fontsize=12, fontweight="bold")
    ax_position.set_xlabel("Tiempo de corrida (s)")
    ax_position.set_ylabel("Posicion relativa")
    ax_position.grid(True, alpha=0.25)
    ax_position.legend(loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_trajectory_plot(ref_x_cm: np.ndarray, ref_y_cm: np.ndarray, pred_xy: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 8.5), constrained_layout=True)
    ax.plot(ref_x_cm, ref_y_cm, color="#bbbbbb", linewidth=5.0, alpha=0.9, label="trayectoria de referencia")
    ax.plot(pred_xy[:, 0], pred_xy[:, 1], color="#111111", linewidth=2.2, label="regresion lineal")
    ax.scatter([pred_xy[0, 0]], [pred_xy[0, 1]], s=120, color="#00a651", edgecolor="white", linewidth=1.2, label="inicio")
    ax.scatter([pred_xy[-1, 0]], [pred_xy[-1, 1]], s=120, color="#d62728", edgecolor="white", linewidth=1.2, label="fin")
    ax.set_title("Trayectoria reconstruida con regresion lineal base", fontsize=14, fontweight="bold")
    ax.set_xlabel("X estimada (cm)")
    ax.set_ylabel("Y estimada (cm)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.20)
    ax.legend(loc="lower left")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_comparison_plot(
    guide_image_path: Path,
    comparison_reference_png: Path,
    trajectory_png: Path,
    output_path: Path,
) -> None:
    guide_img = plt.imread(guide_image_path)
    reference_img = plt.imread(comparison_reference_png)
    trajectory_img = plt.imread(trajectory_png)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.5), constrained_layout=True)
    axes[0].imshow(guide_img)
    axes[0].set_title("Pista usada como referencia", fontsize=12, fontweight="bold")
    axes[1].imshow(reference_img)
    axes[1].set_title("Camino promedio previo", fontsize=12, fontweight="bold")
    axes[2].imshow(trajectory_img)
    axes[2].set_title("Salida de regresion lineal", fontsize=12, fontweight="bold")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset_path = root / "telemetria" / "dataset" / "linefollower_ai_dataset.jsonl"
    guide_image_path = root / "telemetria" / "pista_limpia_1664x1664.png"
    comparison_reference_png = root / "telemetria" / "output" / f"dataset_guide_{SESSION_NAME}_average_only.png"
    output_dir = root / "telemetria" / "report"
    clean_csv = output_dir / f"{SESSION_NAME}_clean_regression_dataset.csv"
    trajectory_png = output_dir / f"{SESSION_NAME}_regression_trajectory.png"
    comparison_png = output_dir / f"{SESSION_NAME}_comparison.png"
    sensor_plot_png = output_dir / f"{SESSION_NAME}_sensor_position.png"
    metrics_json = output_dir / f"{SESSION_NAME}_regression_metrics.json"

    raw_running_rows = load_session_rows(dataset_path, SESSION_NAME)
    trimmed_rows, rows_removed_tail = trim_to_completed_manual_laps(raw_running_rows)
    df, duplicates_removed, missing_removed = clean_dataframe(trimmed_rows)
    weighted_position = compute_weighted_line_position(df)
    ref_x_cm, ref_y_cm, lap_phase, global_progress = build_reference_coordinates(df)
    X, feature_names = build_feature_matrix(df, weighted_position, lap_phase, global_progress)
    y = np.column_stack([ref_x_cm, ref_y_cm])
    coefficients, predictions = fit_linear_regression(X, y)
    metrics = compute_metrics(y, predictions)

    save_clean_csv(df, weighted_position, ref_x_cm, ref_y_cm, predictions, clean_csv)
    save_sensor_plot(df, weighted_position, sensor_plot_png)
    save_trajectory_plot(ref_x_cm, ref_y_cm, predictions, trajectory_png)
    save_comparison_plot(guide_image_path, comparison_reference_png, trajectory_png, comparison_png)

    metrics_payload = {
        "session": SESSION_NAME,
        "rows_raw_running": len(raw_running_rows),
        "rows_clean": int(len(df)),
        "rows_removed_tail": int(rows_removed_tail),
        "duplicates_removed": int(duplicates_removed),
        "missing_removed": int(missing_removed),
        "feature_names": feature_names,
        "coefficients_shape": list(coefficients.shape),
        **metrics,
    }
    metrics_json.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"CSV limpio: {clean_csv}")
    print(f"Grafica trayectoria: {trajectory_png}")
    print(f"Grafica comparacion: {comparison_png}")
    print(f"Grafica sensores: {sensor_plot_png}")
    print(f"Metricas: {metrics_json}")
    print(
        "Resumen: "
        f"raw_running={len(raw_running_rows)}, clean={len(df)}, tail_removed={rows_removed_tail}, "
        f"R2x={metrics['r2_x']:.4f}, R2y={metrics['r2_y']:.4f}"
    )


if __name__ == "__main__":
    main()
