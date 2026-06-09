from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches, transforms
from matplotlib.collections import LineCollection


MODE_COLORS = {
    "straight": "#1f77b4",
    "left_soft": "#2ca02c",
    "left_hard": "#0b6e3d",
    "right_soft": "#ff7f0e",
    "right_hard": "#d62728",
    "recover": "#9467bd",
    "unknown": "#7f7f7f",
}

MODE_TURN_BIAS = {
    "straight": 0.00,
    "left_soft": -0.18,
    "left_hard": -0.38,
    "right_soft": 0.18,
    "right_hard": 0.38,
    "recover": 0.00,
    "unknown": 0.00,
}

MODE_SPEED_FACTOR = {
    "straight": 1.00,
    "left_soft": 0.92,
    "left_hard": 0.80,
    "right_soft": 0.92,
    "right_hard": 0.80,
    "recover": 0.40,
    "unknown": 0.74,
}

INPUT_CANDIDATES = [
    "telemetria/dataset/linefollower_ai_dataset.jsonl",
    "telemetria/dataset/linefollower_ai_dataset_auto_labeled_basic3.csv",
    "telemetria/dataset/linefollower_ai_dataset_auto_labeled.csv",
    "telemetria/dataset/linefollower_ai_dataset_hybrid_labeled_basic3.csv",
    "telemetria/dataset/linefollower_ai_dataset_hybrid_labeled.csv",
    "telemetria/dataset/linefollower_ai_dataset_labeled.csv",
]

GUIDE_AREA_CM = 120.0
DEFAULT_CAR_WIDTH_CM = 18.0
DEFAULT_CAR_LENGTH_CM = 20.0
DEFAULT_MARGIN_CM = 6.0
GUIDE_START_TARGET_FRACTION = (0.83, 0.92)

GUIDE_TEMPLATE_POINTS = [
    (470.0, 440.0),
    (370.0, 440.0),
    (280.0, 430.0),
    (205.0, 360.0),
    (205.0, 250.0),
    (250.0, 185.0),
    (360.0, 185.0),
    (430.0, 250.0),
    (450.0, 355.0),
    (470.0, 440.0),
    (475.0, 610.0),
    (540.0, 695.0),
    (690.0, 705.0),
    (860.0, 705.0),
    (935.0, 770.0),
    (930.0, 860.0),
    (810.0, 905.0),
    (550.0, 905.0),
    (185.0, 905.0),
    (80.0, 825.0),
    (75.0, 195.0),
    (135.0, 105.0),
    (780.0, 105.0),
    (880.0, 165.0),
    (885.0, 330.0),
    (835.0, 410.0),
    (740.0, 440.0),
    (470.0, 440.0),
]


@dataclass
class Sample:
    dataset_session: str
    dataset_row_index: int
    uptime_ms: int
    run_elapsed_ms: int
    manual_lap_count: int
    manual_lap_marked: int
    line_pos: float
    motor_a_pwm: float
    motor_b_pwm: float
    base_eff: float
    line_lost: int
    local_mode: str
    dataset_label: str
    last_dir: str
    confidence_avg: float
    curve_intensity: float
    pid_correction: float
    run_lap_estimate: int


@dataclass
class SessionSummary:
    name: str
    rows: int
    avg_dt_ms: float
    lost_ratio: float
    label_counts: Counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grafica una sesion del dataset sobre la imagen guia de la pista."
    )
    parser.add_argument(
        "--input",
        default="",
        help="CSV o JSONL del dataset. Si se omite, se busca uno compatible en telemetria/dataset.",
    )
    parser.add_argument(
        "--session",
        default="",
        help="Sesion concreta a graficar. Si se omite, se usa la sesion con mas muestras.",
    )
    parser.add_argument(
        "--guide",
        default="telemetria/pista_limpia_1664x1664.png",
        help="Imagen guia de la pista.",
    )
    parser.add_argument(
        "--output-dir",
        default="telemetria/output",
        help="Carpeta donde se guardan PNG/CSV/JSON.",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="Lista las sesiones detectadas y termina.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Cantidad maxima de sesiones a listar con --list-sessions.",
    )
    parser.add_argument(
        "--car-width-cm",
        type=float,
        default=DEFAULT_CAR_WIDTH_CM,
        help="Ancho aproximado del carro en cm.",
    )
    parser.add_argument(
        "--car-length-cm",
        type=float,
        default=DEFAULT_CAR_LENGTH_CM,
        help="Largo aproximado del carro en cm.",
    )
    parser.add_argument(
        "--margin-cm",
        type=float,
        default=DEFAULT_MARGIN_CM,
        help="Margen libre dentro del area guia de 120x120 cm.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Si es mayor que 0, usa solo los primeros N segundos de la corrida.",
    )
    return parser.parse_args()


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def sign_or(value: float, fallback: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return fallback


def is_running_state(value: object) -> bool:
    return str(value or "").strip().lower() in {"run", "running"}


def map_label_to_mode(label: str, line_pos: float, last_dir: str) -> str:
    raw = str(label or "").strip().lower()
    if raw == "straight":
        return "straight"
    if raw in {"recovery", "recover"}:
        return "recover"
    if raw in {"left_soft", "left_hard", "right_soft", "right_hard"}:
        return raw
    if raw in {"curve", "curve_soft", "curve_hard"}:
        direction = 0.0
        if last_dir == "right":
            direction = 1.0
        elif last_dir == "left":
            direction = -1.0
        direction = sign_or(line_pos, direction)
        family = "soft" if raw != "curve_hard" else "hard"
        return f"{'right' if direction >= 0.0 else 'left'}_{family}"
    return "unknown"


def normalize_mode(raw: object, *, line_pos: float = 0.0, last_dir: str = "") -> str:
    value = str(raw or "").strip().lower()
    if value in MODE_COLORS:
        return value
    return map_label_to_mode(value, line_pos, last_dir)


def read_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict] = []
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"JSON invalido en linea {line_no}: {exc}") from exc
            if isinstance(item, dict):
                rows.append(item)
        return rows

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_input_path(raw_input: str) -> Path:
    if raw_input.strip():
        path = Path(raw_input).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"No existe el dataset de entrada: {path}")
        return path

    workspace_root = Path(__file__).resolve().parents[1]
    for candidate in INPUT_CANDIDATES:
        path = (workspace_root / candidate).resolve()
        if path.exists():
            return path
    raise SystemExit("No se encontro ningun dataset compatible en telemetria/dataset.")


def build_samples(rows: list[dict]) -> dict[str, list[Sample]]:
    sessions: dict[str, list[Sample]] = defaultdict(list)
    for row in rows:
        if not is_running_state(row.get("state")):
            continue
        session = str(row.get("dataset_session") or "").strip()
        if not session:
            continue

        line_pos = float(row.get("line_pos") or 0.0)
        last_dir = str(row.get("last_dir") or "").strip().lower()
        dataset_label = str(row.get("dataset_label") or "unknown").strip().lower()
        local_mode = normalize_mode(
            row.get("local_mode") or dataset_label,
            line_pos=line_pos,
            last_dir=last_dir,
        )
        sessions[session].append(
            Sample(
                dataset_session=session,
                dataset_row_index=int(float(row.get("dataset_row_index") or 0)),
                uptime_ms=int(float(row.get("uptime_ms") or 0)),
                run_elapsed_ms=int(float(row.get("run_elapsed_ms") or 0)),
                manual_lap_count=int(float(row.get("dataset_manual_lap_count") or 0)),
                manual_lap_marked=int(float(row.get("dataset_manual_lap_marked") or 0)),
                line_pos=line_pos,
                motor_a_pwm=float(row.get("motor_a_pwm") or 0.0),
                motor_b_pwm=float(row.get("motor_b_pwm") or 0.0),
                base_eff=float(
                    row.get("effective_base_cmd")
                    or row.get("base_eff")
                    or row.get("adaptive_base_cmd")
                    or 0.0
                ),
                line_lost=int(float(row.get("line_lost") or 0)),
                local_mode=local_mode,
                dataset_label=dataset_label,
                last_dir=last_dir,
                confidence_avg=float(
                    row.get("confidence_avg")
                    or row.get("confidence")
                    or row.get("confidence_fast")
                    or 0.0
                ),
                curve_intensity=float(row.get("curve_intensity") or 0.0),
                pid_correction=float(row.get("pid_correction") or 0.0),
                run_lap_estimate=int(float(row.get("run_lap_estimate") or row.get("lap_estimate") or 0)),
            )
        )

    for items in sessions.values():
        items.sort(key=lambda sample: (sample.dataset_row_index, sample.uptime_ms))
    return dict(sessions)


def compute_avg_dt_ms(samples: list[Sample]) -> float:
    deltas: list[int] = []
    prev = None
    for sample in samples:
        if prev is not None:
            delta = sample.uptime_ms - prev
            if 20 <= delta <= 1000:
                deltas.append(delta)
        prev = sample.uptime_ms
    if not deltas:
        return 150.0
    return float(sum(deltas) / len(deltas))


def summarize_sessions(sessions: dict[str, list[Sample]]) -> list[SessionSummary]:
    result: list[SessionSummary] = []
    for name, samples in sessions.items():
        rows = len(samples)
        lost_ratio = (
            sum(1 for sample in samples if sample.line_lost != 0) / float(rows)
            if rows
            else 0.0
        )
        result.append(
            SessionSummary(
                name=name,
                rows=rows,
                avg_dt_ms=compute_avg_dt_ms(samples),
                lost_ratio=lost_ratio,
                label_counts=Counter(sample.dataset_label for sample in samples),
            )
        )
    return sorted(result, key=lambda item: (-item.rows, item.name))


def trim_samples_to_seconds(samples: list[Sample], max_seconds: float) -> list[Sample]:
    if max_seconds <= 0.0:
        return list(samples)

    max_ms = int(round(max_seconds * 1000.0))
    trimmed = [sample for sample in samples if sample.run_elapsed_ms > 0 and sample.run_elapsed_ms <= max_ms]
    if trimmed:
        return trimmed

    start_uptime = samples[0].uptime_ms
    return [sample for sample in samples if (sample.uptime_ms - start_uptime) <= max_ms]


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 2:
        return list(values)
    radius = max(1, window // 2)
    output: list[float] = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        output.append(sum(values[start:end]) / float(end - start))
    return output


def subdivide_polyline(
    points: list[tuple[float, float]], iterations: int = 3
) -> list[tuple[float, float]]:
    output = list(points)
    for _ in range(iterations):
        if len(output) < 2:
            break
        nxt = [output[0]]
        for p0, p1 in zip(output[:-1], output[1:]):
            q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
            r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
            nxt.extend([q, r])
        nxt.append(output[-1])
        output = nxt
    return output


def resample_polyline(
    points: list[tuple[float, float]], target_count: int = 1800
) -> tuple[list[float], list[float]]:
    distances = [0.0]
    total = 0.0
    for p0, p1 in zip(points[:-1], points[1:]):
        total += math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        distances.append(total)
    if total <= 1e-9:
        return [point[0] for point in points], [point[1] for point in points]

    xs: list[float] = []
    ys: list[float] = []
    idx = 0
    for i in range(target_count):
        target = total * i / float(target_count - 1)
        while idx < len(distances) - 2 and distances[idx + 1] < target:
            idx += 1
        d0 = distances[idx]
        d1 = distances[idx + 1]
        p0 = points[idx]
        p1 = points[idx + 1]
        alpha = 0.0 if d1 <= d0 else (target - d0) / (d1 - d0)
        xs.append(p0[0] + alpha * (p1[0] - p0[0]))
        ys.append(p0[1] + alpha * (p1[1] - p0[1]))
    return xs, ys


def compute_normals(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    nx: list[float] = []
    ny: list[float] = []
    for index in range(len(xs)):
        if index == 0:
            dx = xs[1] - xs[0]
            dy = ys[1] - ys[0]
        elif index == len(xs) - 1:
            dx = xs[-1] - xs[-2]
            dy = ys[-1] - ys[-2]
        else:
            dx = xs[index + 1] - xs[index - 1]
            dy = ys[index + 1] - ys[index - 1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            nx.append(0.0)
            ny.append(-1.0)
        else:
            nx.append(-dy / length)
            ny.append(dx / length)
    return nx, ny


def rotate_closed_path(
    xs: list[float],
    ys: list[float],
    *,
    target_x: float,
    target_y: float,
) -> tuple[list[float], list[float]]:
    if len(xs) != len(ys) or len(xs) < 4:
        return xs, ys

    best_index = 0
    best_distance = None
    for index, (x, y) in enumerate(zip(xs, ys)):
        distance = (x - target_x) ** 2 + (y - target_y) ** 2
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_index = index

    rotated_x = xs[best_index:] + xs[1:best_index + 1]
    rotated_y = ys[best_index:] + ys[1:best_index + 1]
    return rotated_x, rotated_y


def reverse_closed_path_preserve_start(
    xs: list[float],
    ys: list[float],
) -> tuple[list[float], list[float]]:
    if len(xs) != len(ys) or len(xs) < 4:
        return xs, ys
    if abs(xs[0] - xs[-1]) < 1e-9 and abs(ys[0] - ys[-1]) < 1e-9:
        core_x = list(reversed(xs[1:-1]))
        core_y = list(reversed(ys[1:-1]))
        return [xs[0]] + core_x + [xs[0]], [ys[0]] + core_y + [ys[0]]
    return [xs[0]] + list(reversed(xs[1:])), [ys[0]] + list(reversed(ys[1:]))


def build_guide_centerline(width_px: int, height_px: int) -> tuple[list[float], list[float], list[float], list[float]]:
    sx = width_px / 1000.0
    sy = height_px / 1000.0
    scaled = [(x * sx, y * sy) for x, y in GUIDE_TEMPLATE_POINTS]
    smooth = subdivide_polyline(scaled, iterations=3)
    xs, ys = resample_polyline(smooth, target_count=1800)
    target_x = width_px * GUIDE_START_TARGET_FRACTION[0]
    target_y = height_px * GUIDE_START_TARGET_FRACTION[1]
    xs, ys = rotate_closed_path(xs, ys, target_x=target_x, target_y=target_y)
    xs, ys = reverse_closed_path_preserve_start(xs, ys)
    nx, ny = compute_normals(xs, ys)
    return xs, ys, nx, ny


def interpolate_series(values: list[float], position: float) -> float:
    if not values:
        return 0.0
    if position <= 0.0:
        return values[0]
    if position >= len(values) - 1:
        return values[-1]
    left = int(math.floor(position))
    right = min(len(values) - 1, left + 1)
    alpha = position - left
    return values[left] + alpha * (values[right] - values[left])


def estimate_total_laps(samples: list[Sample]) -> int:
    max_marker = max((sample.run_lap_estimate for sample in samples), default=0)
    if max_marker > 0:
        return max(1, int(round((max_marker + 1) / 2.0)))
    return 1


def manual_lap_mark_indices(samples: list[Sample]) -> list[int]:
    return [index for index, sample in enumerate(samples) if sample.manual_lap_marked != 0]


def trim_samples_to_manual_laps(samples: list[Sample]) -> tuple[list[Sample], list[int], int, list[Sample]]:
    mark_indices = manual_lap_mark_indices(samples)
    manual_total = max((sample.manual_lap_count for sample in samples), default=0)
    if manual_total <= 0 or len(mark_indices) < manual_total:
        return list(samples), mark_indices, 0, []

    last_keep = mark_indices[manual_total - 1]
    trimmed = list(samples[: last_keep + 1])
    trimmed_marks = [index for index in mark_indices[:manual_total]]
    tail_samples = list(samples[last_keep + 1 :])
    return trimmed, trimmed_marks, manual_total, tail_samples


def project_samples_on_guide(
    samples: list[Sample],
    *,
    guide_width_px: int,
    guide_height_px: int,
    car_width_px: float,
) -> dict[str, object]:
    guide_x, guide_y, normal_x, normal_y = build_guide_centerline(guide_width_px, guide_height_px)
    trimmed_samples, manual_mark_indices, manual_total, tail_samples = trim_samples_to_manual_laps(samples)
    working_samples = trimmed_samples
    total_laps = manual_total if manual_total > 0 else estimate_total_laps(working_samples)
    max_elapsed = max((sample.run_elapsed_ms for sample in working_samples), default=0)
    if max_elapsed <= 0:
        start_uptime = working_samples[0].uptime_ms
        max_elapsed = max((sample.uptime_ms - start_uptime for sample in working_samples), default=1)

    img_x: list[float] = []
    img_y: list[float] = []
    ref_x: list[float] = []
    ref_y: list[float] = []
    lap_index_values: list[int] = []
    completed_mark_points: list[tuple[float, float, int]] = []
    completed_mark_ref_points: list[tuple[float, float, int]] = []

    if manual_total > 0:
        segment_starts = [0] + [index + 1 for index in manual_mark_indices[:-1]]
        segment_ends = manual_mark_indices[:manual_total]
        lap_segments = list(zip(segment_starts, segment_ends))
    else:
        lap_segments = []

    for index, sample in enumerate(working_samples):
        if manual_total > 0:
            lap_index = 0
            lap_phase = 0.0
            for segment_index, (start_idx, end_idx) in enumerate(lap_segments):
                if start_idx <= index <= end_idx:
                    lap_index = segment_index
                    segment_len = max(1, end_idx - start_idx)
                    lap_phase = (index - start_idx) / float(segment_len)
                    if index == end_idx:
                        lap_phase = 0.999
                    break
        else:
            elapsed = sample.run_elapsed_ms
            if elapsed <= 0:
                elapsed = sample.uptime_ms - working_samples[0].uptime_ms
            global_progress = clamp(elapsed / float(max_elapsed), 0.0, 1.0) * float(total_laps)
            lap_index = min(total_laps - 1, max(0, int(math.floor(global_progress))))
            lap_phase = global_progress - float(lap_index)
            if lap_index == total_laps - 1 and elapsed >= max_elapsed:
                lap_phase = 0.999

        guide_pos = lap_phase * float(len(guide_x) - 1)
        base_x = interpolate_series(guide_x, guide_pos)
        base_y = interpolate_series(guide_y, guide_pos)
        nx = interpolate_series(normal_x, guide_pos)
        ny = interpolate_series(normal_y, guide_pos)

        line_offset = clamp(sample.line_pos / 1500.0, -1.0, 1.0) * car_width_px * 0.18
        lap_spread = (lap_index - (total_laps - 1) / 2.0) * car_width_px * 0.24
        ref_px = base_x + nx * line_offset
        ref_py = base_y + ny * line_offset
        px = ref_px + nx * lap_spread
        py = ref_py + ny * lap_spread
        img_x.append(px)
        img_y.append(py)
        ref_x.append(ref_px)
        ref_y.append(ref_py)
        lap_index_values.append(lap_index)

        if sample.manual_lap_marked != 0:
            completed_mark_points.append((px, py, sample.manual_lap_count))
            completed_mark_ref_points.append((ref_px, ref_py, sample.manual_lap_count))

    return {
        "working_samples": working_samples,
        "img_x": img_x,
        "img_y": img_y,
        "ref_x": ref_x,
        "ref_y": ref_y,
        "lap_index": lap_index_values,
        "total_laps": total_laps,
        "projection": "manual_lap_progress" if manual_total > 0 else "guide_progress",
        "guide_x": guide_x,
        "guide_y": guide_y,
        "manual_total_laps": manual_total,
        "completed_mark_points": completed_mark_points,
        "completed_mark_ref_points": completed_mark_ref_points,
        "tail_samples": tail_samples,
    }


def reconstruct_path(samples: list[Sample]) -> dict[str, object]:
    if len(samples) < 2:
        raise SystemExit("La sesion no tiene suficientes muestras para graficarse.")

    avg_dt_ms = compute_avg_dt_ms(samples)
    theta = 0.0
    x = 0.0
    y = 0.0
    xs = [x]
    ys = [y]
    line_positions = [samples[0].line_pos]
    motor_a = [samples[0].motor_a_pwm]
    motor_b = [samples[0].motor_b_pwm]
    modes = [samples[0].local_mode]
    lost_flags = [samples[0].line_lost != 0]

    prev_uptime = samples[0].uptime_ms
    prev_line_pos = samples[0].line_pos
    for sample in samples[1:]:
        raw_dt_ms = sample.uptime_ms - prev_uptime
        prev_uptime = sample.uptime_ms
        dt_ms = raw_dt_ms if 20 <= raw_dt_ms <= 1000 else avg_dt_ms
        dt_ratio = clamp(dt_ms / avg_dt_ms, 0.45, 2.50)

        mode = normalize_mode(
            sample.local_mode or sample.dataset_label,
            line_pos=sample.line_pos,
            last_dir=sample.last_dir,
        )
        lost = sample.line_lost != 0 or mode == "recover"

        forward_pwm = max(
            abs(sample.base_eff),
            0.5 * (abs(sample.motor_a_pwm) + abs(sample.motor_b_pwm)),
        )
        speed_norm = clamp(forward_pwm / 255.0, 0.0, 1.0)
        diff_norm = clamp(
            (sample.motor_b_pwm - sample.motor_a_pwm) / max(forward_pwm, 85.0),
            -1.20,
            1.20,
        )
        line_norm = clamp(sample.line_pos / 1500.0, -1.0, 1.0)
        dline_norm = clamp((sample.line_pos - prev_line_pos) / 620.0, -1.0, 1.0)
        prev_line_pos = sample.line_pos

        mode_turn = MODE_TURN_BIAS.get(mode, 0.0)
        pid_turn = clamp(sample.pid_correction, -1.2, 1.2)
        curve_turn = clamp(sample.curve_intensity, 0.0, 1.2) * sign_or(line_norm, mode_turn)
        conf = clamp(sample.confidence_avg, 0.0, 1.0)

        recovery_dir = 0.0
        if sample.last_dir == "right":
            recovery_dir = 1.0
        elif sample.last_dir == "left":
            recovery_dir = -1.0
        recovery_dir = sign_or(line_norm, recovery_dir)

        turn_input = (
            0.82 * diff_norm
            + 0.58 * line_norm
            + 0.26 * dline_norm
            + 0.22 * pid_turn
            + 0.42 * curve_turn
            + 0.75 * mode_turn
        )
        if lost:
            turn_input = 0.35 * turn_input + 0.65 * recovery_dir

        step_size = (0.26 + 0.94 * speed_norm) * MODE_SPEED_FACTOR.get(mode, 0.74)
        step_size *= 0.70 + 0.30 * max(conf, 0.25)
        if lost:
            step_size *= 0.28

        theta += 0.29 * turn_input * dt_ratio
        x += math.cos(theta) * step_size * 10.0 * dt_ratio
        y += math.sin(theta) * step_size * 10.0 * dt_ratio

        xs.append(x)
        ys.append(y)
        line_positions.append(sample.line_pos)
        motor_a.append(sample.motor_a_pwm)
        motor_b.append(sample.motor_b_pwm)
        modes.append(mode)
        lost_flags.append(lost)

    smooth_window = max(7, int(round(len(samples) * 0.024)))
    if smooth_window % 2 == 0:
        smooth_window += 1

    return {
        "avg_dt_ms": avg_dt_ms,
        "raw_x": xs,
        "raw_y": ys,
        "track_x": moving_average(xs, smooth_window),
        "track_y": moving_average(ys, smooth_window),
        "line_positions": line_positions,
        "motor_a": motor_a,
        "motor_b": motor_b,
        "modes": modes,
        "lost_flags": lost_flags,
        "smooth_window": smooth_window,
    }


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "session"


def load_image_gray(path: Path) -> tuple[object, list[list[float]], int, int]:
    image = plt.imread(path)
    height = int(image.shape[0])
    width = int(image.shape[1])
    gray: list[list[float]] = []
    if len(image.shape) == 2:
        for row in image:
            gray.append([float(pixel) for pixel in row.tolist()])
    else:
        for row in image:
            gray_row: list[float] = []
            for pixel in row.tolist():
                if len(pixel) >= 3:
                    gray_row.append((float(pixel[0]) + float(pixel[1]) + float(pixel[2])) / 3.0)
                else:
                    gray_row.append(float(pixel[0]))
            gray.append(gray_row)
    return image, gray, width, height


def apply_symmetry(xs: list[float], ys: list[float], rotation: int, mirror_x: bool) -> tuple[list[float], list[float], str]:
    tx = list(xs)
    ty = list(ys)
    if mirror_x:
        tx = [-value for value in tx]

    for _ in range(rotation % 4):
        tx, ty = [-value for value in ty], list(tx)

    label = f"rot{rotation * 90}"
    if mirror_x:
        label += "_mirrorX"
    return tx, ty, label


def fit_to_image(
    xs: list[float],
    ys: list[float],
    *,
    width_px: int,
    height_px: int,
    margin_px: float,
) -> tuple[list[float], list[float], float]:
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    usable_w = max(1.0, width_px - 2.0 * margin_px)
    usable_h = max(1.0, height_px - 2.0 * margin_px)
    scale = min(usable_w / span_x, usable_h / span_y)
    offset_x = 0.5 * (width_px - scale * (min_x + max_x))
    offset_y = 0.5 * (height_px - scale * (min_y + max_y))
    fit_x = [offset_x + value * scale for value in xs]
    fit_y = [offset_y + value * scale for value in ys]
    return fit_x, fit_y, scale


def darkness_score(gray: list[list[float]], xs: list[float], ys: list[float]) -> float:
    height = len(gray)
    width = len(gray[0]) if height else 0
    if width == 0:
        return 0.0
    total = 0.0
    count = 0
    for x, y in zip(xs, ys):
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= xi < width and 0 <= yi < height:
            total += 1.0 - gray[yi][xi]
            count += 1
    return total / float(count) if count else 0.0


def align_path_to_guide(
    xs: list[float],
    ys: list[float],
    *,
    gray: list[list[float]],
    width_px: int,
    height_px: int,
    margin_px: float,
) -> dict[str, object]:
    centered_x = [value - sum(xs) / float(len(xs)) for value in xs]
    centered_y = [value - sum(ys) / float(len(ys)) for value in ys]
    best: dict[str, object] | None = None

    for rotation in range(4):
        for mirror_x in (False, True):
            sym_x, sym_y, label = apply_symmetry(centered_x, centered_y, rotation, mirror_x)
            fit_x, fit_y, scale = fit_to_image(
                sym_x,
                sym_y,
                width_px=width_px,
                height_px=height_px,
                margin_px=margin_px,
            )
            score = darkness_score(gray, fit_x, fit_y)
            candidate = {
                "img_x": fit_x,
                "img_y": fit_y,
                "transform": label,
                "score": score,
                "pixels_per_unit": scale,
            }
            if best is None or float(candidate["score"]) > float(best["score"]):
                best = candidate

    if best is None:
        raise SystemExit("No se pudo alinear la trayectoria con la imagen guia.")
    return best


def build_segments(xs: list[float], ys: list[float], modes: list[str]) -> tuple[list[list[tuple[float, float]]], list[str]]:
    segments: list[list[tuple[float, float]]] = []
    colors: list[str] = []
    for index in range(len(xs) - 1):
        segments.append([(xs[index], ys[index]), (xs[index + 1], ys[index + 1])])
        colors.append(MODE_COLORS.get(modes[min(index + 1, len(modes) - 1)], MODE_COLORS["unknown"]))
    return segments, colors


def contiguous_lap_segments(lap_indices: list[int]) -> list[tuple[int, int, int]]:
    if not lap_indices:
        return []
    segments: list[tuple[int, int, int]] = []
    start = 0
    current = lap_indices[0]
    for index, value in enumerate(lap_indices[1:], start=1):
        if value != current:
            segments.append((current, start, index - 1))
            start = index
            current = value
    segments.append((current, start, len(lap_indices) - 1))
    return segments


def resample_path(xs: list[float], ys: list[float], target_count: int) -> tuple[list[float], list[float]]:
    points = list(zip(xs, ys))
    if len(points) <= 1:
        return xs, ys
    return resample_polyline(points, target_count=target_count)


def build_average_lap(
    *,
    img_x: list[float],
    img_y: list[float],
    samples: list[Sample],
    lap_indices: list[int],
    tail_samples: list[Sample],
) -> dict[str, object]:
    segments = contiguous_lap_segments(lap_indices)
    if not segments:
        return {
            "segments": [],
            "average_x": [],
            "average_y": [],
            "lap_scores": [],
            "suspect_lap": 0,
            "tail_failed": False,
            "tail_rows": 0,
            "lap_stats": [],
        }

    target_count = 180
    per_lap_x: list[list[float]] = []
    per_lap_y: list[list[float]] = []
    lap_stats: list[dict[str, object]] = []

    for lap_index, start, end in segments:
        seg_x = img_x[start : end + 1]
        seg_y = img_y[start : end + 1]
        rs_x, rs_y = resample_path(seg_x, seg_y, target_count)
        per_lap_x.append(rs_x)
        per_lap_y.append(rs_y)

        seg_samples = samples[start : end + 1]
        lap_stats.append(
            {
                "lap_number": lap_index + 1,
                "rows": len(seg_samples),
                "start_elapsed_ms": seg_samples[0].run_elapsed_ms,
                "end_elapsed_ms": seg_samples[-1].run_elapsed_ms,
                "lost_count": sum(sample.line_lost != 0 for sample in seg_samples),
                "recover_count": sum(sample.local_mode == "recover" for sample in seg_samples),
                "mean_abs_line_pos": round(
                    sum(abs(sample.line_pos) for sample in seg_samples) / float(len(seg_samples)),
                    3,
                ),
            }
        )

    average_x = [sum(values[index] for values in per_lap_x) / float(len(per_lap_x)) for index in range(target_count)]
    average_y = [sum(values[index] for values in per_lap_y) / float(len(per_lap_y)) for index in range(target_count)]

    lap_scores: list[float] = []
    for lap_x, lap_y in zip(per_lap_x, per_lap_y):
        dist_sum = 0.0
        for index in range(target_count):
            dist_sum += math.hypot(lap_x[index] - average_x[index], lap_y[index] - average_y[index])
        lap_scores.append(dist_sum / float(target_count))

    suspect_lap = 0
    if lap_scores:
        suspect_lap = 1 + max(range(len(lap_scores)), key=lambda index: lap_scores[index])

    tail_failed = False
    if tail_samples:
        tail_lost = sum(sample.line_lost != 0 for sample in tail_samples)
        tail_recover = sum(sample.local_mode == "recover" for sample in tail_samples)
        tail_failed = tail_lost > 0 or tail_recover > 0

    return {
        "segments": segments,
        "average_x": average_x,
        "average_y": average_y,
        "lap_scores": lap_scores,
        "suspect_lap": suspect_lap,
        "tail_failed": tail_failed,
        "tail_rows": len(tail_samples),
        "lap_stats": lap_stats,
    }


def save_plot_csv(
    path: Path,
    *,
    samples: list[Sample],
    img_x: list[float],
    img_y: list[float],
    guide_width_px: int,
    guide_height_px: int,
) -> None:
    px_per_cm_x = guide_width_px / GUIDE_AREA_CM
    px_per_cm_y = guide_height_px / GUIDE_AREA_CM
    fieldnames = [
        "dataset_session",
        "dataset_row_index",
        "uptime_ms",
        "img_x_px",
        "img_y_px",
        "guide_x_cm",
        "guide_y_cm",
        "line_pos",
        "motor_a_pwm",
        "motor_b_pwm",
        "local_mode",
        "dataset_label",
        "line_lost",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, sample in enumerate(samples):
            writer.writerow(
                {
                    "dataset_session": sample.dataset_session,
                    "dataset_row_index": sample.dataset_row_index,
                    "uptime_ms": sample.uptime_ms,
                    "img_x_px": f"{img_x[index]:.2f}",
                    "img_y_px": f"{img_y[index]:.2f}",
                    "guide_x_cm": f"{img_x[index] / px_per_cm_x:.2f}",
                    "guide_y_cm": f"{img_y[index] / px_per_cm_y:.2f}",
                    "line_pos": f"{sample.line_pos:.2f}",
                    "motor_a_pwm": f"{sample.motor_a_pwm:.2f}",
                    "motor_b_pwm": f"{sample.motor_b_pwm:.2f}",
                    "local_mode": sample.local_mode,
                    "dataset_label": sample.dataset_label,
                    "line_lost": sample.line_lost,
                }
            )


def save_summary_json(
    path: Path,
    *,
    input_path: Path,
    guide_path: Path,
    session: str,
    samples: list[Sample],
    reconstruction: dict[str, object],
    alignment: dict[str, object],
    average_lap: dict[str, object],
    car_width_cm: float,
    car_length_cm: float,
) -> None:
    summary = {
        "input_path": str(input_path),
        "guide_path": str(guide_path),
        "session": session,
        "samples": len(samples),
        "avg_dt_ms": round(float(reconstruction["avg_dt_ms"]), 3),
        "smooth_window": int(reconstruction["smooth_window"]),
        "lost_ratio": round(
            sum(1 for item in reconstruction["lost_flags"] if item) / float(len(samples)),
            5,
        ),
        "transform": str(alignment.get("transform", "guide_progress")),
        "guide_match_score": round(float(alignment.get("score", 0.0)), 6),
        "projection": str(alignment.get("projection", "guide_progress")),
        "estimated_total_laps": int(alignment.get("total_laps", 1)),
        "manual_total_laps": int(alignment.get("manual_total_laps", 0)),
        "guide_area_cm": GUIDE_AREA_CM,
        "car_width_cm": car_width_cm,
        "car_length_cm": car_length_cm,
        "suspect_lap": int(average_lap.get("suspect_lap", 0)),
        "tail_failed_after_last_mark": bool(average_lap.get("tail_failed", False)),
        "tail_rows_after_last_mark": int(average_lap.get("tail_rows", 0)),
        "lap_scores": [round(float(value), 4) for value in average_lap.get("lap_scores", [])],
        "lap_stats": average_lap.get("lap_stats", []),
        "mode_counts": dict(sorted(Counter(reconstruction["modes"]).items())),
        "label_counts": dict(sorted(Counter(sample.dataset_label for sample in samples).items())),
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")


def plot_session(
    *,
    session: str,
    input_path: Path,
    guide_path: Path,
    image: object,
    guide_width_px: int,
    guide_height_px: int,
    samples: list[Sample],
    reconstruction: dict[str, object],
    alignment: dict[str, object],
    average_lap: dict[str, object],
    output_png: Path,
    car_width_cm: float,
    car_length_cm: float,
) -> None:
    img_x = alignment["img_x"]
    img_y = alignment["img_y"]
    line_positions = reconstruction["line_positions"]
    motor_a = reconstruction["motor_a"]
    motor_b = reconstruction["motor_b"]
    lost_flags = reconstruction["lost_flags"]
    modes = reconstruction["modes"]
    lap_indices = alignment.get("lap_index", [0] * len(img_x))
    total_laps = int(alignment.get("total_laps", 1))
    completed_mark_points = alignment.get("completed_mark_points", [])
    average_x = average_lap.get("average_x", [])
    average_y = average_lap.get("average_y", [])
    suspect_lap = int(average_lap.get("suspect_lap", 0))

    px_per_cm = guide_width_px / GUIDE_AREA_CM
    car_w_px = car_width_cm * px_per_cm
    car_l_px = car_length_cm * px_per_cm

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 0.72])
    ax_track = fig.add_subplot(grid[:, 0])
    ax_line = fig.add_subplot(grid[0, 1])
    ax_motor = fig.add_subplot(grid[1, 1])

    fig.suptitle(
        "Dataset del seguidor sobre la pista guia\n"
        "Area de referencia: 120 x 120 cm | Carro aprox: 18 x 20 cm",
        fontsize=15,
        fontweight="bold",
    )

    ax_track.imshow(image, origin="upper")
    segments, colors = build_segments(img_x, img_y, modes)
    overlay = LineCollection(segments, colors=colors, linewidths=3.0, alpha=0.92)
    ax_track.add_collection(overlay)

    ax_track.scatter(
        [img_x[0]],
        [img_y[0]],
        s=130,
        color="#1a9850",
        edgecolor="white",
        linewidth=1.2,
        zorder=6,
        label="Inicio",
    )
    ax_track.scatter(
        [img_x[-1]],
        [img_y[-1]],
        s=130,
        color="#111111",
        edgecolor="white",
        linewidth=1.2,
        zorder=6,
        label="Fin",
    )

    lost_x = [img_x[index] for index, lost in enumerate(lost_flags) if lost]
    lost_y = [img_y[index] for index, lost in enumerate(lost_flags) if lost]
    if lost_x:
        ax_track.scatter(
            lost_x,
            lost_y,
            s=18,
            color="#ff3b30",
            alpha=0.85,
            label="linea perdida",
            zorder=5,
        )

    if completed_mark_points:
        ax_track.scatter(
            [item[0] for item in completed_mark_points],
            [item[1] for item in completed_mark_points],
            s=55,
            color="#ffd600",
            edgecolor="#8a6d00",
            linewidth=1.0,
            marker="D",
            zorder=7,
            label="marca vuelta",
        )
        for x, y, lap_no in completed_mark_points:
            ax_track.text(x + 6, y - 6, f"V{lap_no}", fontsize=8, color="#8a6d00", weight="bold")

    if average_x and average_y:
        ax_track.plot(
            average_x,
            average_y,
            color="#ffffff",
            linewidth=6.0,
            alpha=0.72,
            zorder=5,
        )
        ax_track.plot(
            average_x,
            average_y,
            color="#111111",
            linewidth=2.6,
            alpha=0.96,
            zorder=6,
            label="promedio",
        )

    palette = ["#00c853", "#00b0ff", "#ff6d00", "#d500f9", "#ff1744", "#aeea00"]
    for lap_number in range(total_laps):
        lap_points = [
            (img_x[index], img_y[index])
            for index, lap_index in enumerate(lap_indices)
            if lap_index == lap_number
        ]
        if len(lap_points) < 2:
            continue
        ax_track.plot(
            [point[0] for point in lap_points],
            [point[1] for point in lap_points],
            color=palette[lap_number % len(palette)],
            linewidth=1.2,
            alpha=0.60,
            label=f"vuelta {lap_number + 1}",
            zorder=4,
        )

    if len(img_x) >= 2:
        dx = img_x[-1] - img_x[-2]
        dy = img_y[-1] - img_y[-2]
        angle_deg = math.degrees(math.atan2(dy, dx))
        car_patch = patches.Rectangle(
            (img_x[-1] - 0.5 * car_l_px, img_y[-1] - 0.5 * car_w_px),
            car_l_px,
            car_w_px,
            linewidth=1.4,
            edgecolor="#111111",
            facecolor="#ffd54f",
            alpha=0.58,
            zorder=7,
        )
        car_patch.set_transform(
            transforms.Affine2D().rotate_deg_around(img_x[-1], img_y[-1], angle_deg) + ax_track.transData
        )
        ax_track.add_patch(car_patch)

    info_text = (
        f"Sesion: {session}\n"
        f"Entrada: {input_path.name}\n"
        f"Muestras: {len(samples)}\n"
        f"dt medio: {reconstruction['avg_dt_ms']:.1f} ms\n"
        f"Proyeccion: {alignment.get('projection', 'guide_progress')}\n"
        f"Vueltas estimadas: {total_laps}\n"
        f"Vueltas manuales: {alignment.get('manual_total_laps', 0)}\n"
        f"Vuelta mas rara: {suspect_lap or 'n/a'}\n"
        f"Fallo tras ultima marca: {1 if average_lap.get('tail_failed', False) else 0}\n"
        f"Escala de guia: 120 x 120 cm\n"
        f"Carro: {car_width_cm:.0f} x {car_length_cm:.0f} cm"
    )
    ax_track.text(
        0.015,
        0.015,
        info_text,
        transform=ax_track.transAxes,
        fontsize=9,
        color="#111111",
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
    )
    ax_track.set_title("Trayectoria aproximada sobre imagen guia", fontsize=12, fontweight="bold")
    ax_track.set_xlim(0, guide_width_px)
    ax_track.set_ylim(guide_height_px, 0)
    ax_track.set_xticks([])
    ax_track.set_yticks([])
    ax_track.legend(loc="lower right", fontsize=8, framealpha=0.92)

    progress = list(range(len(samples)))
    ax_line.plot(progress, line_positions, color="#3366cc", linewidth=1.35)
    ax_line.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_line.set_title("line_pos por muestra", fontsize=12, fontweight="bold")
    ax_line.set_xlabel("muestra")
    ax_line.set_ylabel("line_pos")
    ax_line.grid(True, alpha=0.20)

    ax_motor.plot(progress, motor_a, color="#00897b", linewidth=1.25, label="motor A")
    ax_motor.plot(progress, motor_b, color="#ef6c00", linewidth=1.25, label="motor B")
    ax_motor.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_motor.set_title("PWM de motores", fontsize=12, fontweight="bold")
    ax_motor.set_xlabel("muestra")
    ax_motor.set_ylabel("PWM")
    ax_motor.grid(True, alpha=0.20)
    ax_motor.legend(loc="upper right", fontsize=8)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_average_reference(
    *,
    guide_width_px: int,
    guide_height_px: int,
    session: str,
    average_lap: dict[str, object],
    alignment: dict[str, object],
    output_png: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
    ax.set_facecolor("#f7f5ef")
    base_x = alignment.get("ref_x", alignment["img_x"])
    base_y = alignment.get("ref_y", alignment["img_y"])

    palette = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c", "#9b59b6", "#16a085", "#d35400", "#34495e"]
    for lap_number, start, end in average_lap.get("segments", []):
        xs = base_x[start : end + 1]
        ys = base_y[start : end + 1]
        ax.plot(xs, ys, color=palette[lap_number % len(palette)], linewidth=1.8, alpha=0.60, label=f"vuelta {lap_number + 1}")

    average_x = average_lap.get("average_x", [])
    average_y = average_lap.get("average_y", [])
    if average_x and average_y:
        ax.plot(average_x, average_y, color="#ffffff", linewidth=8, alpha=0.85)
        ax.plot(average_x, average_y, color="#111111", linewidth=3.2, label="camino promedio")

    marks = alignment.get("completed_mark_ref_points", alignment.get("completed_mark_points", []))
    if marks:
        ax.scatter(
            [item[0] for item in marks],
            [item[1] for item in marks],
            s=58,
            color="#ffd600",
            edgecolor="#8a6d00",
            linewidth=1.0,
            marker="D",
            zorder=5,
            label="marca vuelta",
        )

    ax.scatter(
        [base_x[0]],
        [base_y[0]],
        s=140,
        color="#00a651",
        edgecolor="white",
        linewidth=1.2,
        zorder=6,
        label="inicio",
    )
    ax.scatter(
        [base_x[-1]],
        [base_y[-1]],
        s=140,
        color="#111111",
        edgecolor="white",
        linewidth=1.2,
        zorder=6,
        label="fin",
    )

    title = (
        f"Camino promedio estimado - {session}\n"
        f"Vueltas manuales={alignment.get('manual_total_laps', 0)} | "
        f"vuelta mas rara={average_lap.get('suspect_lap', 0)}"
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(0, guide_width_px)
    ax.set_ylim(guide_height_px, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower left", fontsize=8, framealpha=0.95)

    note = (
        "Referencia limpia sin pista de fondo\n"
        "La linea negra marca el camino promedio\n"
        "Las lineas de color muestran cada vuelta"
    )
    ax.text(
        0.015,
        0.015,
        note,
        transform=ax.transAxes,
        fontsize=9,
        color="#111111",
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.94},
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_average_only_reference(
    *,
    guide_width_px: int,
    guide_height_px: int,
    session: str,
    average_lap: dict[str, object],
    alignment: dict[str, object],
    output_png: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
    ax.set_facecolor("#f7f5ef")

    average_x = average_lap.get("average_x", [])
    average_y = average_lap.get("average_y", [])
    if not average_x or not average_y:
        raise SystemExit("No hay trayectoria promedio suficiente para generar la referencia simple.")

    ax.plot(average_x, average_y, color="#ffffff", linewidth=10, alpha=0.92, solid_capstyle="round")
    ax.plot(average_x, average_y, color="#111111", linewidth=4.0, solid_capstyle="round")

    ax.scatter(
        [average_x[0]],
        [average_y[0]],
        s=170,
        color="#00a651",
        edgecolor="white",
        linewidth=1.5,
        zorder=5,
    )
    ax.scatter(
        [average_x[-1]],
        [average_y[-1]],
        s=170,
        color="#111111",
        edgecolor="white",
        linewidth=1.5,
        zorder=5,
    )

    ax.set_title(f"Camino promedio limpio - {session}", fontsize=16, fontweight="bold")
    ax.set_xlim(0, guide_width_px)
    ax.set_ylim(guide_height_px, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])

    note = (
        f"Vueltas manuales: {alignment.get('manual_total_laps', 0)}\n"
        "Solo se muestra la linea promedio estimada"
    )
    ax.text(
        0.015,
        0.015,
        note,
        transform=ax.transAxes,
        fontsize=10,
        color="#111111",
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.95},
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = resolve_input_path(args.input)
    guide_path = Path(args.guide).expanduser().resolve()
    if not guide_path.exists():
        raise SystemExit(f"No existe la imagen guia: {guide_path}")

    rows = read_rows(input_path)
    sessions = build_samples(rows)
    if not sessions:
        raise SystemExit("No se encontraron filas de corrida en el dataset.")

    summary = summarize_sessions(sessions)
    if args.list_sessions:
        print("Sesiones disponibles:")
        for item in summary[: max(1, args.top)]:
            print(
                f"- {item.name}: rows={item.rows}, avg_dt_ms={item.avg_dt_ms:.1f}, "
                f"lost_ratio={item.lost_ratio:.3f}, labels={dict(item.label_counts)}"
            )
        return

    selected_session = args.session.strip() if args.session else summary[0].name
    if selected_session not in sessions:
        raise SystemExit(f"La sesion no existe en el dataset: {selected_session}")

    selected_samples = trim_samples_to_seconds(sessions[selected_session], args.max_seconds)
    if len(selected_samples) < 2:
        raise SystemExit("No quedaron suficientes muestras despues de aplicar el recorte de tiempo.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    safe_name = sanitize_name(selected_session)
    suffix = f"_first_{int(args.max_seconds)}s" if args.max_seconds > 0 else ""
    output_png = output_dir / f"dataset_guide_{safe_name}{suffix}.png"
    output_csv = output_dir / f"dataset_guide_{safe_name}{suffix}.csv"
    output_json = output_dir / f"dataset_guide_{safe_name}{suffix}.json"
    output_clean_png = output_dir / f"dataset_guide_{safe_name}{suffix}_average_clean.png"
    output_average_only_png = output_dir / f"dataset_guide_{safe_name}{suffix}_average_only.png"

    image, gray, guide_width_px, guide_height_px = load_image_gray(guide_path)
    alignment = project_samples_on_guide(
        selected_samples,
        guide_width_px=guide_width_px,
        guide_height_px=guide_height_px,
        car_width_px=args.car_width_cm * (guide_width_px / GUIDE_AREA_CM),
    )
    output_samples = alignment.get("working_samples", selected_samples)
    reconstruction = reconstruct_path(output_samples)
    average_lap = build_average_lap(
        img_x=alignment.get("ref_x", alignment["img_x"]),
        img_y=alignment.get("ref_y", alignment["img_y"]),
        samples=output_samples,
        lap_indices=alignment["lap_index"],
        tail_samples=alignment.get("tail_samples", []),
    )

    plot_session(
        session=selected_session,
        input_path=input_path,
        guide_path=guide_path,
        image=image,
        guide_width_px=guide_width_px,
        guide_height_px=guide_height_px,
        samples=output_samples,
        reconstruction=reconstruction,
        alignment=alignment,
        average_lap=average_lap,
        output_png=output_png,
        car_width_cm=args.car_width_cm,
        car_length_cm=args.car_length_cm,
    )
    plot_average_reference(
        guide_width_px=guide_width_px,
        guide_height_px=guide_height_px,
        session=selected_session,
        average_lap=average_lap,
        alignment=alignment,
        output_png=output_clean_png,
    )
    plot_average_only_reference(
        guide_width_px=guide_width_px,
        guide_height_px=guide_height_px,
        session=selected_session,
        average_lap=average_lap,
        alignment=alignment,
        output_png=output_average_only_png,
    )
    save_plot_csv(
        output_csv,
        samples=output_samples,
        img_x=alignment["img_x"],
        img_y=alignment["img_y"],
        guide_width_px=guide_width_px,
        guide_height_px=guide_height_px,
    )
    save_summary_json(
        output_json,
        input_path=input_path,
        guide_path=guide_path,
        session=selected_session,
        samples=output_samples,
        reconstruction=reconstruction,
        alignment=alignment,
        average_lap=average_lap,
        car_width_cm=args.car_width_cm,
        car_length_cm=args.car_length_cm,
    )

    print(f"Sesion: {selected_session}")
    print(f"Entrada: {input_path}")
    print(f"Imagen guia: {guide_path}")
    print(f"PNG: {output_png}")
    print(f"Referencia limpia: {output_clean_png}")
    print(f"Promedio solo: {output_average_only_png}")
    print(f"CSV: {output_csv}")
    print(f"Resumen: {output_json}")


if __name__ == "__main__":
    main()
