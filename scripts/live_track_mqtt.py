from __future__ import annotations

import argparse
import json
import math
import threading
import time
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt
from matplotlib.animation import FuncAnimation
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

GUIDE_AREA_CM = 120.0
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


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def sign_or(value: float, fallback: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return fallback


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


def subdivide_polyline(
    points: list[tuple[float, float]],
    iterations: int = 3,
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
    points: list[tuple[float, float]],
    target_count: int = 1800,
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

    rotated_x = xs[best_index:] + xs[1 : best_index + 1]
    rotated_y = ys[best_index:] + ys[1 : best_index + 1]
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


@dataclass
class TelemetrySample:
    session: str
    uptime_ms: int
    run_elapsed_ms: int
    seq_fast: int
    state: str
    line_pos: float
    motor_a_pwm: float
    motor_b_pwm: float
    base_eff: float
    line_lost: int
    local_mode: str
    last_dir: str
    confidence_avg: float
    curve_intensity: float
    pid_correction: float
    run_lap_estimate: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visor en vivo por MQTT para estimar la pista del seguidor en tiempo real."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host del broker MQTT.")
    parser.add_argument("--port", type=int, default=1883, help="Puerto del broker MQTT.")
    parser.add_argument(
        "--topic",
        default="robot/linefollower/telemetry/fast",
        help="Topic MQTT de telemetria rapida.",
    )
    parser.add_argument(
        "--guide",
        default="telemetria/pista_limpia_1664x1664.png",
        help="Imagen guia de la pista.",
    )
    parser.add_argument(
        "--car-width-cm",
        type=float,
        default=18.0,
        help="Ancho aproximado del carro en cm.",
    )
    parser.add_argument(
        "--nominal-lap-seconds",
        type=float,
        default=10.8,
        help="Tiempo medio estimado por vuelta para avanzar sobre la guia.",
    )
    parser.add_argument(
        "--refresh-ms",
        type=int,
        default=150,
        help="Refresco de la interfaz en milisegundos.",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=250,
        help="Cantidad de muestras a mostrar en las graficas laterales.",
    )
    parser.add_argument(
        "--path-limit",
        type=int,
        default=5000,
        help="Cantidad maxima de puntos acumulados en la trayectoria en vivo.",
    )
    return parser.parse_args()


def is_running_state(value: object) -> bool:
    return str(value or "").strip().lower() in {"run", "running"}


def parse_telemetry(payload: dict) -> TelemetrySample:
    line_pos = float(payload.get("line_pos") or 0.0)
    last_dir = str(payload.get("last_dir") or "").strip().lower()
    local_mode = normalize_mode(
        payload.get("local_mode") or payload.get("dataset_label") or "unknown",
        line_pos=line_pos,
        last_dir=last_dir,
    )
    return TelemetrySample(
        session=str(payload.get("dataset_session") or "live").strip() or "live",
        uptime_ms=int(float(payload.get("uptime_ms") or 0)),
        run_elapsed_ms=int(float(payload.get("run_elapsed_ms") or 0)),
        seq_fast=int(float(payload.get("seq_fast") or 0)),
        state=str(payload.get("state") or "").strip().lower(),
        line_pos=line_pos,
        motor_a_pwm=float(payload.get("motor_a_pwm") or 0.0),
        motor_b_pwm=float(payload.get("motor_b_pwm") or 0.0),
        base_eff=float(
            payload.get("effective_base_cmd")
            or payload.get("base_eff")
            or payload.get("adaptive_base_cmd")
            or 0.0
        ),
        line_lost=int(float(payload.get("line_lost") or 0)),
        local_mode=local_mode,
        last_dir=last_dir,
        confidence_avg=float(
            payload.get("confidence_avg")
            or payload.get("confidence")
            or payload.get("confidence_fast")
            or 0.0
        ),
        curve_intensity=float(payload.get("curve_intensity") or 0.0),
        pid_correction=float(payload.get("pid_correction") or 0.0),
        run_lap_estimate=int(float(payload.get("run_lap_estimate") or 0)),
    )


def compute_cumulative_distances(xs: list[float], ys: list[float]) -> tuple[list[float], float]:
    cumulative = [0.0]
    total = 0.0
    for x0, y0, x1, y1 in zip(xs[:-1], ys[:-1], xs[1:], ys[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
        cumulative.append(total)
    return cumulative, total


def interpolate_track_position(
    *,
    target: float,
    guide_x: list[float],
    guide_y: list[float],
    normal_x: list[float],
    normal_y: list[float],
    cumulative: list[float],
    total_length: float,
) -> tuple[float, float, float, float]:
    if total_length <= 1e-9:
        return guide_x[0], guide_y[0], normal_x[0], normal_y[0]

    wrapped = target % total_length
    index = bisect_right(cumulative, wrapped) - 1
    index = max(0, min(index, len(cumulative) - 2))
    d0 = cumulative[index]
    d1 = cumulative[index + 1]
    alpha = 0.0 if d1 <= d0 else (wrapped - d0) / (d1 - d0)

    x = guide_x[index] + alpha * (guide_x[index + 1] - guide_x[index])
    y = guide_y[index] + alpha * (guide_y[index + 1] - guide_y[index])
    nx = normal_x[index] + alpha * (normal_x[index + 1] - normal_x[index])
    ny = normal_y[index] + alpha * (normal_y[index + 1] - normal_y[index])
    normal_length = math.hypot(nx, ny)
    if normal_length > 1e-9:
        nx /= normal_length
        ny /= normal_length
    return x, y, nx, ny


class LiveTrackState:
    def __init__(
        self,
        *,
        guide_width_px: int,
        guide_height_px: int,
        car_width_cm: float,
        nominal_lap_seconds: float,
        history_limit: int,
        path_limit: int,
    ) -> None:
        self.lock = threading.Lock()
        self.nominal_lap_seconds = max(3.0, nominal_lap_seconds)
        self.history_limit = max(60, history_limit)
        self.path_limit = max(200, path_limit)
        self.px_per_cm = guide_width_px / GUIDE_AREA_CM
        self.car_width_px = car_width_cm * self.px_per_cm

        self.guide_x, self.guide_y, self.normal_x, self.normal_y = build_guide_centerline(
            guide_width_px,
            guide_height_px,
        )
        self.cumulative, self.track_length_px = compute_cumulative_distances(self.guide_x, self.guide_y)

        self.connected = False
        self.last_error = ""
        self.last_message_at = 0.0
        self.message_count = 0
        self.current_session = "esperando"
        self.current_state = "idle"

        self.path_x: list[float] = []
        self.path_y: list[float] = []
        self.path_modes: list[str] = []
        self.lost_points: list[tuple[float, float]] = []
        self.line_history: deque[float] = deque(maxlen=self.history_limit)
        self.motor_a_history: deque[float] = deque(maxlen=self.history_limit)
        self.motor_b_history: deque[float] = deque(maxlen=self.history_limit)
        self.elapsed_history: deque[float] = deque(maxlen=self.history_limit)

        self.current_x = self.guide_x[0]
        self.current_y = self.guide_y[0]
        self.current_mode = "unknown"
        self.current_line_pos = 0.0
        self.current_lost = False
        self.progress_px = 0.0
        self.estimated_laps = 0.0
        self.run_elapsed_ms = 0
        self.last_sample: TelemetrySample | None = None
        self.reset_run("esperando")

    def reset_run(self, session_name: str) -> None:
        self.current_session = session_name or "live"
        self.path_x = [self.guide_x[0]]
        self.path_y = [self.guide_y[0]]
        self.path_modes = ["straight"]
        self.lost_points = []
        self.line_history = deque([0.0], maxlen=self.history_limit)
        self.motor_a_history = deque([0.0], maxlen=self.history_limit)
        self.motor_b_history = deque([0.0], maxlen=self.history_limit)
        self.elapsed_history = deque([0.0], maxlen=self.history_limit)
        self.current_x = self.guide_x[0]
        self.current_y = self.guide_y[0]
        self.current_mode = "straight"
        self.current_line_pos = 0.0
        self.current_lost = False
        self.progress_px = 0.0
        self.estimated_laps = 0.0
        self.run_elapsed_ms = 0
        self.last_sample = None

    def ingest(self, sample: TelemetrySample) -> None:
        with self.lock:
            self.message_count += 1
            self.last_message_at = time.time()
            self.current_state = sample.state or "idle"

            previous = self.last_sample
            must_reset = False
            if previous is None:
                must_reset = True
            elif sample.session != previous.session:
                must_reset = True
            elif sample.run_elapsed_ms + 250 < previous.run_elapsed_ms:
                must_reset = True
            elif sample.seq_fast > 0 and previous.seq_fast > 0 and sample.seq_fast + 5 < previous.seq_fast:
                must_reset = True

            if must_reset:
                self.reset_run(sample.session)

            self.current_session = sample.session

            if not is_running_state(sample.state):
                self.last_sample = sample
                return

            if previous is None or not is_running_state(previous.state):
                dt_ms = 150.0
            else:
                dt_ms = float(sample.uptime_ms - previous.uptime_ms)
                if dt_ms < 20.0 or dt_ms > 1000.0:
                    alt_dt = float(sample.run_elapsed_ms - previous.run_elapsed_ms)
                    dt_ms = alt_dt if 20.0 <= alt_dt <= 1000.0 else 150.0

            forward_pwm = max(
                abs(sample.base_eff),
                0.5 * (abs(sample.motor_a_pwm) + abs(sample.motor_b_pwm)),
                40.0,
            )
            speed_norm = clamp(forward_pwm / 255.0, 0.0, 1.0)
            line_norm = clamp(sample.line_pos / 1500.0, -1.0, 1.0)
            conf = clamp(sample.confidence_avg, 0.0, 1.0)
            mode = sample.local_mode
            lost = sample.line_lost != 0 or mode == "recover"

            base_px_per_sec = self.track_length_px / self.nominal_lap_seconds
            dt_sec = dt_ms / 1000.0
            progress_gain = (0.50 + 0.85 * speed_norm) * MODE_SPEED_FACTOR.get(mode, 0.74)
            progress_gain *= 0.72 + 0.28 * max(conf, 0.20)
            if lost:
                progress_gain *= 0.42
            self.progress_px += base_px_per_sec * dt_sec * progress_gain

            base_x, base_y, nx, ny = interpolate_track_position(
                target=self.progress_px,
                guide_x=self.guide_x,
                guide_y=self.guide_y,
                normal_x=self.normal_x,
                normal_y=self.normal_y,
                cumulative=self.cumulative,
                total_length=self.track_length_px,
            )

            direction = 0.0
            if sample.last_dir == "right":
                direction = 1.0
            elif sample.last_dir == "left":
                direction = -1.0
            direction = sign_or(line_norm, direction)

            lateral_offset = line_norm * self.car_width_px * 0.18
            curve_bias = clamp(sample.curve_intensity, 0.0, 1.4) * direction * self.car_width_px * 0.06
            pid_bias = clamp(sample.pid_correction, -1.2, 1.2) * self.car_width_px * 0.04
            px = base_x + nx * (lateral_offset + curve_bias + pid_bias)
            py = base_y + ny * (lateral_offset + curve_bias + pid_bias)

            self.current_x = px
            self.current_y = py
            self.current_mode = mode
            self.current_line_pos = sample.line_pos
            self.current_lost = lost
            self.run_elapsed_ms = sample.run_elapsed_ms
            self.estimated_laps = self.progress_px / max(self.track_length_px, 1.0)

            self.path_x.append(px)
            self.path_y.append(py)
            self.path_modes.append(mode)
            if lost:
                self.lost_points.append((px, py))

            if len(self.path_x) > self.path_limit:
                extra = len(self.path_x) - self.path_limit
                self.path_x = self.path_x[extra:]
                self.path_y = self.path_y[extra:]
                self.path_modes = self.path_modes[extra:]

            if len(self.lost_points) > max(40, self.history_limit):
                self.lost_points = self.lost_points[-max(40, self.history_limit) :]

            self.line_history.append(sample.line_pos)
            self.motor_a_history.append(sample.motor_a_pwm)
            self.motor_b_history.append(sample.motor_b_pwm)
            self.elapsed_history.append(sample.run_elapsed_ms / 1000.0)
            self.last_sample = sample

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            age = time.time() - self.last_message_at if self.last_message_at else None
            return {
                "connected": self.connected,
                "last_error": self.last_error,
                "message_count": self.message_count,
                "session": self.current_session,
                "state": self.current_state,
                "path_x": list(self.path_x),
                "path_y": list(self.path_y),
                "path_modes": list(self.path_modes),
                "lost_points": list(self.lost_points),
                "current_x": self.current_x,
                "current_y": self.current_y,
                "current_mode": self.current_mode,
                "current_line_pos": self.current_line_pos,
                "current_lost": self.current_lost,
                "run_elapsed_ms": self.run_elapsed_ms,
                "estimated_laps": self.estimated_laps,
                "line_history": list(self.line_history),
                "motor_a_history": list(self.motor_a_history),
                "motor_b_history": list(self.motor_b_history),
                "elapsed_history": list(self.elapsed_history),
                "last_message_age_s": age,
            }


def build_segments(
    path_x: list[float],
    path_y: list[float],
    path_modes: list[str],
) -> tuple[list[list[tuple[float, float]]], list[str]]:
    segments: list[list[tuple[float, float]]] = []
    colors: list[str] = []
    for index in range(len(path_x) - 1):
        segments.append([(path_x[index], path_y[index]), (path_x[index + 1], path_y[index + 1])])
        colors.append(MODE_COLORS.get(path_modes[min(index + 1, len(path_modes) - 1)], MODE_COLORS["unknown"]))
    return segments, colors


def main() -> None:
    args = parse_args()
    guide_path = Path(args.guide).expanduser().resolve()
    if not guide_path.exists():
        raise SystemExit(f"No existe la imagen guia: {guide_path}")

    image = plt.imread(guide_path)
    guide_height_px = int(image.shape[0])
    guide_width_px = int(image.shape[1])

    state = LiveTrackState(
        guide_width_px=guide_width_px,
        guide_height_px=guide_height_px,
        car_width_cm=args.car_width_cm,
        nominal_lap_seconds=args.nominal_lap_seconds,
        history_limit=args.history_limit,
        path_limit=args.path_limit,
    )

    def on_connect(client: mqtt.Client, userdata: object, flags: dict, rc: int) -> None:
        with state.lock:
            state.connected = rc == 0
            state.last_error = "" if rc == 0 else f"connect rc={rc}"
        if rc == 0:
            client.subscribe(args.topic)

    def on_disconnect(client: mqtt.Client, userdata: object, rc: int) -> None:
        with state.lock:
            state.connected = False
            if rc != 0:
                state.last_error = f"disconnect rc={rc}"

    def on_message(client: mqtt.Client, userdata: object, msg: mqtt.MQTTMessage) -> None:
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            if not isinstance(payload, dict):
                return
            sample = parse_telemetry(payload)
            state.ingest(sample)
        except Exception as exc:  # pragma: no cover - best effort during live view
            with state.lock:
                state.last_error = f"{type(exc).__name__}: {exc}"

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    client.connect(args.host, args.port, keepalive=30)
    client.loop_start()

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 0.72])
    ax_track = fig.add_subplot(grid[:, 0])
    ax_line = fig.add_subplot(grid[0, 1])
    ax_motor = fig.add_subplot(grid[1, 1])

    fig.suptitle(
        "Seguimiento en vivo por MQTT\n"
        "Proyeccion aproximada sobre la pista guia con inicio fijo abajo a la derecha",
        fontsize=15,
        fontweight="bold",
    )

    def render(_: int) -> None:
        snap = state.snapshot()

        ax_track.clear()
        ax_line.clear()
        ax_motor.clear()

        ax_track.imshow(image, origin="upper")
        path_x = snap["path_x"]
        path_y = snap["path_y"]
        path_modes = snap["path_modes"]
        if len(path_x) >= 2:
            segments, colors = build_segments(path_x, path_y, path_modes)
            overlay = LineCollection(segments, colors=colors, linewidths=3.0, alpha=0.94)
            ax_track.add_collection(overlay)

        if path_x:
            ax_track.scatter(
                [path_x[0]],
                [path_y[0]],
                s=130,
                color="#00a651",
                edgecolor="white",
                linewidth=1.2,
                zorder=6,
                label="inicio",
            )
        ax_track.scatter(
            [snap["current_x"]],
            [snap["current_y"]],
            s=170,
            color="#111111",
            edgecolor="white",
            linewidth=1.4,
            zorder=7,
            label="carro",
        )

        lost_points = snap["lost_points"]
        if lost_points:
            ax_track.scatter(
                [point[0] for point in lost_points],
                [point[1] for point in lost_points],
                s=20,
                color="#ff3b30",
                alpha=0.85,
                zorder=5,
                label="linea perdida",
            )

        age = snap["last_message_age_s"]
        age_text = "sin datos"
        if age is not None:
            age_text = f"{age:.2f} s"

        info = (
            f"Broker: {args.host}:{args.port}\n"
            f"Topic: {args.topic}\n"
            f"Sesion: {snap['session']}\n"
            f"Estado: {snap['state']}\n"
            f"Mensajes: {snap['message_count']}\n"
            f"Edad ultimo dato: {age_text}\n"
            f"run_elapsed: {snap['run_elapsed_ms'] / 1000.0:.2f} s\n"
            f"vueltas estimadas: {snap['estimated_laps']:.2f}\n"
            f"line_pos: {snap['current_line_pos']:.0f}\n"
            f"modo: {snap['current_mode']}\n"
            f"conectado: {1 if snap['connected'] else 0}"
        )
        if snap["last_error"]:
            info += f"\nerror: {snap['last_error']}"

        ax_track.text(
            0.015,
            0.015,
            info,
            transform=ax_track.transAxes,
            fontsize=9,
            color="#111111",
            va="bottom",
            ha="left",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.94},
        )
        ax_track.set_title("Pista en vivo estimada", fontsize=12, fontweight="bold")
        ax_track.set_xlim(0, guide_width_px)
        ax_track.set_ylim(guide_height_px, 0)
        ax_track.set_xticks([])
        ax_track.set_yticks([])
        ax_track.legend(loc="lower right", fontsize=8, framealpha=0.92)

        elapsed = snap["elapsed_history"]
        line_history = snap["line_history"]
        motor_a_history = snap["motor_a_history"]
        motor_b_history = snap["motor_b_history"]

        ax_line.plot(elapsed, line_history, color="#3366cc", linewidth=1.35)
        ax_line.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--", alpha=0.6)
        ax_line.set_title("line_pos en vivo", fontsize=12, fontweight="bold")
        ax_line.set_xlabel("tiempo de corrida (s)")
        ax_line.set_ylabel("line_pos")
        ax_line.grid(True, alpha=0.20)

        ax_motor.plot(elapsed, motor_a_history, color="#00897b", linewidth=1.25, label="motor A")
        ax_motor.plot(elapsed, motor_b_history, color="#ef6c00", linewidth=1.25, label="motor B")
        ax_motor.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--", alpha=0.6)
        ax_motor.set_title("PWM de motores en vivo", fontsize=12, fontweight="bold")
        ax_motor.set_xlabel("tiempo de corrida (s)")
        ax_motor.set_ylabel("PWM")
        ax_motor.grid(True, alpha=0.20)
        ax_motor.legend(loc="upper right", fontsize=8)

    animation = FuncAnimation(fig, render, interval=max(80, args.refresh_ms), cache_frame_data=False)
    fig._live_animation = animation

    try:
        plt.show()
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
