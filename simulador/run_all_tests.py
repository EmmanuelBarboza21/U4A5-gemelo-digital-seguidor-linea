from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRESET_DIR = PROJECT_ROOT / "digital_twin" / "presets"
OUTPUT_DIR = PROJECT_ROOT / "digital_twin" / "outputs"

TESTS = [
    ("test_1", PRESET_DIR / "test_1_initial_firmware_scaled.json"),
    ("test_2", PRESET_DIR / "test_2_tuned_stage1.json"),
    ("test_3", PRESET_DIR / "test_3_final_realmatch.json"),
]


def main() -> None:
    for tag, preset_path in TESTS:
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "digital_twin" / "run_headless_twin.py"), "--preset", str(preset_path), "--tag", tag],
            check=True,
            cwd=str(PROJECT_ROOT),
        )

    summary_rows = [
        [
            "test",
            "time_s",
            "laps_equivalent",
            "average_lap_time_s_estimated",
            "average_speed_mps",
            "line_lost_ratio",
            "oscillations_per_second",
        ]
    ]
    for tag, _ in TESTS:
        metrics_path = OUTPUT_DIR / f"{tag}_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        sim = metrics["simulation_metrics"]
        summary_rows.append(
            [
                tag,
                f"{sim['time_s']:.3f}",
                f"{sim['laps_equivalent']:.3f}",
                f"{sim['average_lap_time_s_estimated']:.3f}",
                f"{sim['average_speed_mps']:.3f}",
                f"{sim['line_lost_ratio']:.4f}",
                f"{sim['oscillations_per_second']:.3f}",
            ]
        )

    summary_path = OUTPUT_DIR / "simulation_tests_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(summary_rows)
    print(f"Resumen: {summary_path}")


if __name__ == "__main__":
    main()
