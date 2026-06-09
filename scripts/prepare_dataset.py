from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


PRIORITY_FIELDS = [
    "dataset_label",
    "dataset_session",
    "dataset_row_index",
    "dataset_manual_lap_count",
    "dataset_manual_lap_marked",
    "dataset_manual_lap_mark_at_ms",
    "dataset_manual_lap_mark_age_ms",
    "timestamp_unix_ms",
    "uptime_ms",
    "state",
    "local_mode",
    "local_mode_candidate",
    "ai_source",
    "ai_track_mode",
    "line_pos",
    "line_lost",
    "lost_ms",
    "balance_lr",
    "confidence_fast",
    "confidence_avg",
    "sensor_sum",
    "sensor_max",
    "norm0",
    "norm1",
    "norm2",
    "norm3",
    "raw0",
    "raw1",
    "raw2",
    "raw3",
    "pos_norm",
    "trend_pos",
    "trend_pos_avg",
    "curve_intensity",
    "pid_error",
    "pid_integral",
    "pid_d_filt",
    "pid_correction",
    "motor_a_pwm",
    "motor_b_pwm",
    "base_eff",
    "ai_confidence",
    "ai_blend",
    "ai_speed_factor",
    "ai_kp_scale",
    "ai_ki_scale",
    "ai_kd_scale",
    "ai_pivot_threshold",
    "ai_pivot_cap",
    "ai_recovery_bias",
    "dominant_sensor",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepara el dataset etiquetado del seguidor IA."
    )
    parser.add_argument(
        "--input",
        default="telemetria/dataset/linefollower_ai_dataset.jsonl",
        help="Ruta del JSONL crudo.",
    )
    parser.add_argument(
        "--label-mode",
        choices=("manual", "auto", "hybrid"),
        default="manual",
        help=(
            "manual: usa solo dataset_label del dashboard. "
            "auto: infiere etiquetas desde local_mode/ai_track_mode. "
            "hybrid: prioriza manual y completa lo demas en automatico."
        ),
    )
    parser.add_argument(
        "--keep-nonrun",
        action="store_true",
        help="Conserva filas fuera de estado run. Por defecto se descartan.",
    )
    parser.add_argument(
        "--recovery-lost-ms",
        type=int,
        default=120,
        help="Umbral de lost_ms para marcar recovery en modo auto.",
    )
    parser.add_argument(
        "--class-profile",
        choices=("full", "basic3"),
        default="full",
        help=(
            "full: conserva straight/curve_soft/curve_hard/recovery. "
            "basic3: colapsa curve_soft y curve_hard en curve."
        ),
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help=(
            "Si es mayor que 0, conserva solo las filas cuya corrida este dentro "
            "de los primeros N segundos."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"JSON invalido en linea {line_no}: {exc}") from exc
        if not isinstance(item, dict):
            continue
        rows.append(item)
    return rows


def is_labeled(row: dict) -> bool:
    label = str(row.get("dataset_label", "unknown"))
    valid = int(row.get("dataset_label_valid", 1 if label != "unknown" else 0))
    return valid == 1 and label != "unknown"


def is_run_row(row: dict) -> bool:
    return str(row.get("state", "")).strip().lower() in {"run", "running"}


def within_time_limit(row: dict, max_seconds: float) -> bool:
    if max_seconds <= 0.0:
        return True

    max_ms = max_seconds * 1000.0
    run_elapsed = row.get("run_elapsed_ms")
    if run_elapsed not in (None, ""):
        try:
            return float(run_elapsed) <= max_ms
        except (TypeError, ValueError):
            pass

    uptime = row.get("uptime_ms")
    if uptime not in (None, ""):
        try:
            return float(uptime) <= max_ms
        except (TypeError, ValueError):
            pass

    return True


def normalize_track_label(value: object) -> str:
    mode = str(value or "").strip().lower()
    if mode == "straight":
        return "straight"
    if mode in {"left_soft", "right_soft"}:
        return "curve_soft"
    if mode in {"left_hard", "right_hard"}:
        return "curve_hard"
    if mode == "recover":
        return "recovery"
    return "unknown"


def get_manual_label(row: dict) -> tuple[str, str]:
    if is_labeled(row):
        return str(row.get("dataset_label", "unknown")), "manual"
    return "unknown", "manual"


def get_auto_label(row: dict, recovery_lost_ms: int) -> tuple[str, str]:
    if int(row.get("line_lost", 0)) == 1:
        return "recovery", "auto:line_lost"

    if float(row.get("lost_ms", 0) or 0) >= recovery_lost_ms:
        return "recovery", "auto:lost_ms"

    for field in ("ai_track_mode", "local_mode", "local_mode_candidate"):
        label = normalize_track_label(row.get(field, ""))
        if label != "unknown":
            return label, f"auto:{field}"

    return "unknown", "auto:unknown"


def build_labeled_rows(
    raw_rows: list[dict],
    *,
    label_mode: str,
    keep_nonrun: bool,
    recovery_lost_ms: int,
    class_profile: str,
    max_seconds: float,
) -> tuple[list[dict], Counter, Counter]:
    labeled_rows: list[dict] = []
    label_sources: Counter = Counter()
    dropped: Counter = Counter()

    for row in raw_rows:
        if not keep_nonrun and not is_run_row(row):
            dropped["nonrun"] += 1
            continue

        if not within_time_limit(row, max_seconds):
            dropped["time_limit"] += 1
            continue

        manual_label, manual_source = get_manual_label(row)
        auto_label, auto_source = get_auto_label(row, recovery_lost_ms)

        final_label = "unknown"
        final_source = "unknown"
        if label_mode == "manual":
            final_label, final_source = manual_label, manual_source
        elif label_mode == "auto":
            final_label, final_source = auto_label, auto_source
        elif manual_label != "unknown":
            final_label, final_source = manual_label, manual_source
        else:
            final_label, final_source = auto_label, auto_source

        if final_label == "unknown":
            dropped["unknown"] += 1
            continue

        if class_profile == "basic3" and final_label in {"curve_soft", "curve_hard"}:
            final_label = "curve"

        item = dict(row)
        item["dataset_label"] = final_label
        item["dataset_label_valid"] = 1
        item["dataset_label_source"] = final_source
        labeled_rows.append(item)
        label_sources[final_source] += 1

    return labeled_rows, label_sources, dropped


def build_field_order(rows: list[dict]) -> list[str]:
    all_fields = set()
    for row in rows:
        all_fields.update(row.keys())
    ordered = [field for field in PRIORITY_FIELDS if field in all_fields]
    remainder = sorted(field for field in all_fields if field not in ordered)
    return ordered + remainder


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(
    path: Path,
    *,
    raw_rows: list[dict],
    labeled_rows: list[dict],
    fields: list[str],
    label_mode: str,
    class_profile: str,
    label_sources: Counter,
    dropped: Counter,
) -> None:
    labels = Counter(str(row.get("dataset_label", "unknown")) for row in labeled_rows)
    sessions = Counter(str(row.get("dataset_session", "")) for row in labeled_rows)
    summary = {
        "label_mode": label_mode,
        "class_profile": class_profile,
        "raw_rows": len(raw_rows),
        "labeled_rows": len(labeled_rows),
        "labels": dict(sorted(labels.items())),
        "label_sources": dict(sorted(label_sources.items())),
        "dropped": dict(sorted(dropped.items())),
        "sessions": dict(sorted(sessions.items())),
        "fields": fields,
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise SystemExit(f"No existe el archivo de entrada: {input_path}")

    raw_rows = read_jsonl(input_path)
    labeled_rows, label_sources, dropped = build_labeled_rows(
        raw_rows,
        label_mode=args.label_mode,
        keep_nonrun=args.keep_nonrun,
        recovery_lost_ms=args.recovery_lost_ms,
        class_profile=args.class_profile,
        max_seconds=args.max_seconds,
    )
    if not labeled_rows:
        raise SystemExit(
            "No hay filas etiquetadas validas con la configuracion actual."
        )

    output_dir = input_path.parent
    name_prefix = (
        "labeled" if args.label_mode == "manual" else f"{args.label_mode}_labeled"
    )
    if args.class_profile != "full":
        suffix = f"{name_prefix}_{args.class_profile}"
        summary_name = f"dataset_summary_{args.label_mode}_{args.class_profile}.json"
    else:
        suffix = name_prefix
        summary_name = (
            "dataset_summary.json"
            if args.label_mode == "manual"
            else f"dataset_summary_{args.label_mode}.json"
        )
    labeled_jsonl = output_dir / f"linefollower_ai_dataset_{suffix}.jsonl"
    labeled_csv = output_dir / f"linefollower_ai_dataset_{suffix}.csv"
    summary_json = output_dir / summary_name

    fields = build_field_order(labeled_rows)
    write_jsonl(labeled_jsonl, labeled_rows)
    write_csv(labeled_csv, labeled_rows, fields)
    write_summary(
        summary_json,
        raw_rows=raw_rows,
        labeled_rows=labeled_rows,
        fields=fields,
        label_mode=args.label_mode,
        class_profile=args.class_profile,
        label_sources=label_sources,
        dropped=dropped,
    )

    print(f"Entrada: {input_path}")
    print(f"Modo de etiquetado: {args.label_mode}")
    print(f"Perfil de clases: {args.class_profile}")
    print(f"Filas crudas: {len(raw_rows)}")
    print(f"Filas etiquetadas: {len(labeled_rows)}")
    if dropped:
        print(f"Filas descartadas: {dict(sorted(dropped.items()))}")
    print(f"JSONL etiquetado: {labeled_jsonl}")
    print(f"CSV etiquetado: {labeled_csv}")
    print(f"Resumen: {summary_json}")


if __name__ == "__main__":
    main()
