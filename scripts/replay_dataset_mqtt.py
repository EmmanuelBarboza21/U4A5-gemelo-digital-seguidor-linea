from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import paho.mqtt.client as mqtt


INPUT_CANDIDATES = [
    "telemetria/dataset/linefollower_ai_dataset.jsonl",
    "telemetria/dataset/linefollower_ai_dataset_auto_labeled_basic3.jsonl",
    "telemetria/dataset/linefollower_ai_dataset_auto_labeled_basic3.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce una sesion del dataset hacia MQTT para simular el carro en vivo."
    )
    parser.add_argument(
        "--input",
        default="",
        help="Archivo JSONL o CSV de entrada. Si se omite, se usa el dataset crudo de telemetria.",
    )
    parser.add_argument(
        "--session",
        default="",
        help="Sesion concreta a reproducir. Si se omite, se usa la sesion con mas muestras.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host del broker MQTT.")
    parser.add_argument("--port", type=int, default=1883, help="Puerto del broker MQTT.")
    parser.add_argument(
        "--topic",
        default="robot/linefollower/telemetry/fast",
        help="Topic MQTT donde se publican las muestras.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Multiplicador de velocidad. 2.0 reproduce al doble, 0.5 a la mitad.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Repite la sesion en bucle hasta que detengas el programa.",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="Lista sesiones disponibles y termina.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Maximo de sesiones a mostrar con --list-sessions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No publica nada; solo muestra el plan de reproduccion.",
    )
    parser.add_argument(
        "--include-idle",
        action="store_true",
        help="Incluye tambien muestras idle si existen en el dataset.",
    )
    return parser.parse_args()


def resolve_input_path(raw_input: str) -> Path:
    if raw_input.strip():
        path = Path(raw_input).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"No existe el archivo de entrada: {path}")
        return path

    workspace_root = Path(__file__).resolve().parents[1]
    for candidate in INPUT_CANDIDATES:
        path = (workspace_root / candidate).resolve()
        if path.exists():
            return path
    raise SystemExit("No se encontro ningun dataset compatible para reproducir.")


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


def normalize_row(row: dict, *, index: int, previous_uptime: int, previous_elapsed: int) -> dict:
    payload = dict(row)
    payload["dataset_session"] = str(payload.get("dataset_session") or "replay").strip() or "replay"
    payload["state"] = str(payload.get("state") or "running").strip().lower() or "running"
    payload["seq_fast"] = int(float(payload.get("seq_fast") or index))
    payload["dataset_row_index"] = int(float(payload.get("dataset_row_index") or index))
    payload["uptime_ms"] = int(float(payload.get("uptime_ms") or previous_uptime + 150))

    if "run_elapsed_ms" in payload and str(payload.get("run_elapsed_ms")).strip():
        payload["run_elapsed_ms"] = int(float(payload.get("run_elapsed_ms") or 0))
    else:
        if index == 0:
            payload["run_elapsed_ms"] = 0
        else:
            delta = payload["uptime_ms"] - previous_uptime
            delta = delta if 20 <= delta <= 1000 else 150
            payload["run_elapsed_ms"] = previous_elapsed + delta

    numeric_defaults = {
        "line_pos": 0.0,
        "motor_a_pwm": 0.0,
        "motor_b_pwm": 0.0,
        "base_eff": 130.0,
        "line_lost": 0,
        "confidence_avg": 0.7,
        "curve_intensity": 0.0,
        "pid_correction": 0.0,
        "run_lap_estimate": 0,
    }
    for key, default in numeric_defaults.items():
        value = payload.get(key, default)
        if isinstance(default, int):
            payload[key] = int(float(value or default))
        else:
            payload[key] = float(value or default)

    payload["local_mode"] = str(payload.get("local_mode") or payload.get("dataset_label") or "unknown").strip().lower()
    payload["last_dir"] = str(payload.get("last_dir") or "").strip().lower()
    return payload


def is_running_state(value: object) -> bool:
    return str(value or "").strip().lower() in {"run", "running"}


def group_sessions(rows: list[dict], *, include_idle: bool) -> dict[str, list[dict]]:
    sessions: dict[str, list[dict]] = {}
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        session = str(row.get("dataset_session") or "").strip()
        if not session:
            continue
        if not include_idle and "state" in row and not is_running_state(row.get("state")):
            continue
        grouped.setdefault(session, []).append(row)

    for session, items in grouped.items():
        normalized: list[dict] = []
        prev_uptime = 0
        prev_elapsed = 0
        for index, row in enumerate(items):
            payload = normalize_row(row, index=index, previous_uptime=prev_uptime, previous_elapsed=prev_elapsed)
            normalized.append(payload)
            prev_uptime = int(payload["uptime_ms"])
            prev_elapsed = int(payload["run_elapsed_ms"])
        sessions[session] = normalized
    return sessions


def summarize_sessions(sessions: dict[str, list[dict]]) -> list[tuple[str, int]]:
    counts = Counter({name: len(rows) for name, rows in sessions.items()})
    return counts.most_common()


def publish_session(
    *,
    client: mqtt.Client,
    topic: str,
    session_name: str,
    rows: list[dict],
    speed: float,
) -> None:
    if not rows:
        return

    previous_uptime = None
    for index, row in enumerate(rows):
        payload = dict(row)
        payload["dataset_session"] = session_name
        client.publish(topic, json.dumps(payload, ensure_ascii=True), qos=0, retain=False)

        current_uptime = int(payload.get("uptime_ms") or 0)
        if previous_uptime is not None:
            delta_ms = current_uptime - previous_uptime
            delta_ms = delta_ms if 20 <= delta_ms <= 1000 else 150
            time.sleep(max(0.0, delta_ms / 1000.0 / max(speed, 0.05)))
        previous_uptime = current_uptime

    final_payload = dict(rows[-1])
    final_payload["dataset_session"] = session_name
    final_payload["state"] = "idle"
    client.publish(topic, json.dumps(final_payload, ensure_ascii=True), qos=0, retain=False)


def main() -> None:
    args = parse_args()
    input_path = resolve_input_path(args.input)
    rows = read_rows(input_path)
    sessions = group_sessions(rows, include_idle=args.include_idle)
    if not sessions:
        raise SystemExit("No se encontraron sesiones validas en el archivo.")

    summary = summarize_sessions(sessions)
    if args.list_sessions:
        print("Sesiones disponibles:")
        for name, count in summary[: max(1, args.top)]:
            print(f"- {name}: rows={count}")
        return

    selected_session = args.session.strip() if args.session else summary[0][0]
    if selected_session not in sessions:
        raise SystemExit(f"La sesion no existe en el dataset: {selected_session}")

    selected_rows = sessions[selected_session]
    duration_ms = 0
    if selected_rows:
        duration_ms = int(selected_rows[-1].get("run_elapsed_ms") or selected_rows[-1].get("uptime_ms") or 0)

    print(f"Entrada: {input_path}")
    print(f"Sesion: {selected_session}")
    print(f"Muestras: {len(selected_rows)}")
    print(f"Duracion aproximada: {duration_ms / 1000.0:.2f} s")
    print(f"Broker: {args.host}:{args.port}")
    print(f"Topic: {args.topic}")
    print(f"Velocidad replay: x{args.speed:.2f}")

    if args.dry_run:
        return

    client = mqtt.Client()
    client.connect(args.host, args.port, keepalive=30)
    client.loop_start()
    try:
        iteration = 1
        while True:
            replay_session_name = selected_session if not args.loop else f"{selected_session}_replay"
            print(f"Reproduciendo iteracion {iteration}...")
            publish_session(
                client=client,
                topic=args.topic,
                session_name=replay_session_name,
                rows=selected_rows,
                speed=args.speed,
            )
            if not args.loop:
                break
            iteration += 1
            time.sleep(1.0)
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
