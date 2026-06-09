from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier


LOST_STOP_MS = 3000.0
DEFAULT_DATASET = "telemetria/dataset/linefollower_ai_dataset_auto_labeled_basic3.csv"
DEFAULT_HEADER = "include/ai_model_hook.h"
DEFAULT_REPORT = "telemetria/model/ai_model_basic3_report.json"
DEFAULT_EVAL_REPORT = "telemetria/model/ai_model_basic3_external_eval.json"
DEFAULT_EVAL_PREDICTIONS = "telemetria/model/ai_model_basic3_external_predictions.csv"

LABEL_TO_INT = {
    "straight": 0,
    "curve": 1,
    "recovery": 2,
}
INT_TO_LABEL = {value: key for key, value in LABEL_TO_INT.items()}

LOCAL_MODE_TO_INT = {
    "unknown": 0,
    "straight": 1,
    "left_soft": 2,
    "left_hard": 3,
    "right_soft": 4,
    "right_hard": 5,
    "recover": 6,
}

FEATURE_NAMES = [
    "last_s0",
    "last_s1",
    "last_s2",
    "last_s3",
    "mean_s0",
    "mean_s1",
    "mean_s2",
    "mean_s3",
    "last_pos_norm",
    "mean_abs_pos",
    "mean_trend",
    "mean_conf",
    "mean_curve",
    "mean_balance",
    "mean_speed",
    "mean_lost_ms",
    "lost_ratio",
    "mean_sensor_sum",
    "mean_sensor_max",
    "mean_abs_pid_error",
    "mean_pid_d",
    "last_local_mode",
    "last_line_lost",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena y exporta un modelo pequeno para el supervisor IA."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_DATASET,
        help="CSV etiquetado a usar como dataset de entrenamiento.",
    )
    parser.add_argument(
        "--window-len",
        type=int,
        default=8,
        help="Tamano de ventana temporal usado para construir features.",
    )
    parser.add_argument(
        "--output-header",
        default=DEFAULT_HEADER,
        help="Ruta del header C++ generado para el hook del modelo.",
    )
    parser.add_argument(
        "--report",
        default=DEFAULT_REPORT,
        help="Ruta del reporte JSON con metricas y configuracion.",
    )
    parser.add_argument(
        "--eval-input",
        default="",
        help=(
            "CSV etiquetado externo para evaluar el modelo ya entrenado, "
            "por ejemplo una pista nueva."
        ),
    )
    parser.add_argument(
        "--eval-report",
        default=DEFAULT_EVAL_REPORT,
        help="Ruta del reporte JSON de la evaluacion externa.",
    )
    parser.add_argument(
        "--eval-predictions",
        default=DEFAULT_EVAL_PREDICTIONS,
        help="Ruta del CSV con predicciones fila por fila de la evaluacion externa.",
    )
    return parser.parse_args()


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def encode_local_mode(value: object) -> float:
    return float(LOCAL_MODE_TO_INT.get(str(value).strip().lower(), 0))


def build_speed_norm(row: pd.Series) -> float:
    return clamp01(
        0.5
        * (
            abs(float(row.get("motor_a_pwm", 0.0))) / 255.0
            + abs(float(row.get("motor_b_pwm", 0.0))) / 255.0
        )
    )


def build_row_snapshot(row: pd.Series) -> dict[str, float]:
    return {
        "s0": clamp01(float(row.get("norm0", 0.0)) / 1000.0),
        "s1": clamp01(float(row.get("norm1", 0.0)) / 1000.0),
        "s2": clamp01(float(row.get("norm2", 0.0)) / 1000.0),
        "s3": clamp01(float(row.get("norm3", 0.0)) / 1000.0),
        "pos_norm": float(row.get("pos_norm", 0.0)),
        "trend": float(row.get("trend_pos", 0.0)),
        "confidence": clamp01(float(row.get("confidence_fast", 0.0))),
        "curve": clamp01(float(row.get("curve_intensity", 0.0))),
        "balance": float(row.get("balance_lr", 0.0)),
        "speed": build_speed_norm(row),
        "lost_ms": clamp01(float(row.get("lost_ms", 0.0)) / LOST_STOP_MS),
        "line_lost": float(int(row.get("line_lost", 0) or 0)),
        "sensor_sum": clamp01(float(row.get("sensor_sum", 0.0)) / 4000.0),
        "sensor_max": clamp01(float(row.get("sensor_max", 0.0)) / 1000.0),
        "pid_error_abs": clamp01(abs(float(row.get("pid_error", 0.0))) / 2.0),
        "pid_d_abs": clamp01(abs(float(row.get("pid_d_filt", 0.0))) / 25.0),
        "local_mode": encode_local_mode(row.get("local_mode", "unknown")),
    }


def build_window_features(window_rows: list[dict[str, float]]) -> list[float]:
    last = window_rows[-1]
    count = float(len(window_rows))

    def avg(key: str) -> float:
        return sum(item[key] for item in window_rows) / count

    return [
        last["s0"],
        last["s1"],
        last["s2"],
        last["s3"],
        avg("s0"),
        avg("s1"),
        avg("s2"),
        avg("s3"),
        last["pos_norm"],
        sum(abs(item["pos_norm"]) for item in window_rows) / count,
        avg("trend"),
        avg("confidence"),
        avg("curve"),
        avg("balance"),
        avg("speed"),
        avg("lost_ms"),
        avg("line_lost"),
        avg("sensor_sum"),
        avg("sensor_max"),
        avg("pid_error_abs"),
        avg("pid_d_abs"),
        last["local_mode"],
        last["line_lost"],
    ]


def build_dataset_records(
    frame: pd.DataFrame, window_len: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int], list[dict[str, object]]]:
    frame = frame.copy()
    frame["dataset_session"] = frame["dataset_session"].astype(str)
    frame = frame.sort_values(["dataset_session", "dataset_row_index", "seq_fast"]).reset_index(drop=True)

    features: list[list[float]] = []
    labels: list[int] = []
    groups: list[str] = []
    records: list[dict[str, object]] = []
    class_counts: Counter = Counter()

    for session, session_df in frame.groupby("dataset_session", sort=False):
        window: list[dict[str, float]] = []
        for row in session_df.to_dict("records"):
            snapshot = build_row_snapshot(pd.Series(row))
            window.append(snapshot)
            if len(window) > window_len:
                window.pop(0)
            label = str(row["dataset_label"])
            features.append(build_window_features(window))
            labels.append(LABEL_TO_INT[label])
            groups.append(session)
            class_counts[label] += 1
            records.append(
                {
                    "dataset_session": session,
                    "dataset_row_index": int(row.get("dataset_row_index", len(records))),
                    "seq_fast": int(row.get("seq_fast", 0) or 0),
                    "timestamp_unix_ms": int(row.get("timestamp_unix_ms", 0) or 0),
                    "state": str(row.get("state", "")),
                    "local_mode": str(row.get("local_mode", "")),
                    "ai_track_mode": str(row.get("ai_track_mode", "")),
                    "line_lost": int(row.get("line_lost", 0) or 0),
                    "lost_ms": float(row.get("lost_ms", 0.0) or 0.0),
                    "source_label": label,
                }
            )

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
        np.asarray(groups, dtype=object),
        dict(sorted(class_counts.items())),
        records,
    )


def load_dataset(
    path: Path, window_len: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int], list[dict[str, object]]]:
    frame = pd.read_csv(path)
    frame = frame[frame["dataset_label"].isin(LABEL_TO_INT.keys())].copy()
    if frame.empty:
        raise SystemExit("El dataset no contiene etiquetas compatibles para entrenar.")
    return build_dataset_records(frame, window_len)


def split_groups(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_classes = set(int(v) for v in np.unique(y))
    seeds = range(42, 200)
    for seed in seeds:
        outer = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        train_val_idx, test_idx = next(outer.split(x, y, groups))
        if set(int(v) for v in np.unique(y[test_idx])) != unique_classes:
            continue
        inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1000)
        rel_train_idx, rel_val_idx = next(
            inner.split(x[train_val_idx], y[train_val_idx], groups[train_val_idx])
        )
        train_idx = train_val_idx[rel_train_idx]
        val_idx = train_val_idx[rel_val_idx]
        if set(int(v) for v in np.unique(y[val_idx])) != unique_classes:
            continue
        return train_idx, val_idx, test_idx
    raise SystemExit("No se pudo crear una division por sesiones que cubra todas las clases.")


def choose_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[DecisionTreeClassifier, dict[str, float]]:
    candidates: list[tuple[float, int, int, int, DecisionTreeClassifier]] = []
    for depth in (3, 4, 5, 6):
        for min_leaf in (8, 12, 20, 32):
            model = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_leaf=min_leaf,
                class_weight="balanced",
                random_state=42,
            )
            model.fit(x_train, y_train)
            pred_val = model.predict(x_val)
            val_macro_f1 = f1_score(y_val, pred_val, average="macro")
            candidates.append(
                (val_macro_f1, -model.tree_.node_count, depth, min_leaf, model)
            )

    best_f1, _, best_depth, best_min_leaf, best_model = max(candidates)
    return best_model, {
        "val_macro_f1": float(best_f1),
        "max_depth": int(best_depth),
        "min_samples_leaf": int(best_min_leaf),
        "node_count": int(best_model.tree_.node_count),
    }


def evaluate_model(model: DecisionTreeClassifier, x: np.ndarray, y: np.ndarray) -> dict:
    pred = model.predict(x)
    report = classification_report(
        y,
        pred,
        labels=list(INT_TO_LABEL.keys()),
        target_names=[INT_TO_LABEL[i] for i in INT_TO_LABEL],
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "confusion_matrix": confusion_matrix(
            y, pred, labels=list(INT_TO_LABEL.keys())
        ).tolist(),
        "classification_report": report,
    }


def build_feature_importance_report(model: DecisionTreeClassifier) -> list[dict[str, object]]:
    pairs = []
    for name, importance in zip(FEATURE_NAMES, model.feature_importances_):
        pairs.append({"feature": name, "importance": float(importance)})
    return sorted(pairs, key=lambda item: item["importance"], reverse=True)


def build_prediction_rows(
    model: DecisionTreeClassifier,
    x: np.ndarray,
    y: np.ndarray,
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    predictions = model.predict(x)
    probabilities = model.predict_proba(x)
    class_indices = list(model.classes_)
    class_labels = [INT_TO_LABEL[int(index)] for index in class_indices]

    rows: list[dict[str, object]] = []
    for idx, (record, truth, pred) in enumerate(zip(records, y.tolist(), predictions.tolist())):
        prob_row = probabilities[idx]
        item = dict(record)
        item["label_true"] = INT_TO_LABEL[int(truth)]
        item["label_pred"] = INT_TO_LABEL[int(pred)]
        item["correct"] = int(int(truth) == int(pred))
        item["pred_confidence"] = float(np.max(prob_row))
        for class_label, prob in zip(class_labels, prob_row.tolist()):
            item[f"prob_{class_label}"] = float(prob)
        rows.append(item)
    return rows


def c_float(value: float) -> str:
    return f"{float(value):.7f}f"


def c_int_list(values: list[int], per_line: int = 12) -> str:
    chunks = []
    for start in range(0, len(values), per_line):
        chunk = ", ".join(str(v) for v in values[start : start + per_line])
        chunks.append(f"  {chunk}")
    return ",\n".join(chunks)


def c_float_list(values: list[float], per_line: int = 8) -> str:
    chunks = []
    for start in range(0, len(values), per_line):
        chunk = ", ".join(c_float(v) for v in values[start : start + per_line])
        chunks.append(f"  {chunk}")
    return ",\n".join(chunks)


def build_header(
    model: DecisionTreeClassifier,
    *,
    dataset_path: Path,
    report_path: Path,
    feature_names: list[str],
    metrics: dict,
    params: dict,
) -> str:
    tree = model.tree_
    children_left = tree.children_left.tolist()
    children_right = tree.children_right.tolist()
    features = tree.feature.tolist()
    thresholds = [float(value) for value in tree.threshold.tolist()]
    leaf_class: list[int] = []
    leaf_conf: list[float] = []
    for node_index in range(tree.node_count):
        values = tree.value[node_index][0]
        total = float(values.sum())
        if children_left[node_index] == children_right[node_index]:
            predicted = int(np.argmax(values))
            confidence = float(values[predicted] / total) if total > 0.0 else 0.0
            leaf_class.append(predicted)
            leaf_conf.append(confidence)
        else:
            leaf_class.append(-1)
            leaf_conf.append(0.0)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    feature_comment = ", ".join(feature_names)

    return f"""#pragma once

#include <Arduino.h>
#include "ai_supervisor_contract.h"

// Auto-generated by telemetria/train_ai_supervisor_model.py
// Generated at: {generated_at}
// Dataset: {dataset_path.as_posix()}
// Report: {report_path.as_posix()}
// Features: {feature_comment}
// Test accuracy: {metrics["test"]["accuracy"]:.4f}
// Test macro F1: {metrics["test"]["macro_f1"]:.4f}
// Tree nodes: {params["node_count"]}

static constexpr uint8_t AI_TRACK_UNKNOWN = 0;
static constexpr uint8_t AI_TRACK_STRAIGHT = 1;
static constexpr uint8_t AI_TRACK_LEFT_SOFT = 2;
static constexpr uint8_t AI_TRACK_LEFT_HARD = 3;
static constexpr uint8_t AI_TRACK_RIGHT_SOFT = 4;
static constexpr uint8_t AI_TRACK_RIGHT_HARD = 5;
static constexpr uint8_t AI_TRACK_RECOVER = 6;

namespace linefollower_ai_model {{

static constexpr int kFeatureCount = {len(feature_names)};
static constexpr int kNodeCount = {tree.node_count};

static const int16_t kChildrenLeft[kNodeCount] = {{
{c_int_list(children_left)}
}};

static const int16_t kChildrenRight[kNodeCount] = {{
{c_int_list(children_right)}
}};

static const int8_t kFeatureIndex[kNodeCount] = {{
{c_int_list(features)}
}};

static const float kThreshold[kNodeCount] = {{
{c_float_list(thresholds)}
}};

static const int8_t kLeafClass[kNodeCount] = {{
{c_int_list(leaf_class)}
}};

static const float kLeafConfidence[kNodeCount] = {{
{c_float_list(leaf_conf)}
}};

static inline float clamp01f(float value) {{
  if (value < 0.0f) return 0.0f;
  if (value > 1.0f) return 1.0f;
  return value;
}}

static inline float clampfLocal(float value, float lo, float hi) {{
  if (value < lo) return lo;
  if (value > hi) return hi;
  return value;
}}

static inline float avgAbsPos(const AISupervisorFrame* window, int count) {{
  float total = 0.0f;
  for (int i = 0; i < count; ++i) total += fabsf(window[i].posNorm);
  return total / (float)count;
}}

static inline float avgFieldTrend(const AISupervisorFrame* window, int count) {{
  float total = 0.0f;
  for (int i = 0; i < count; ++i) total += window[i].trend;
  return total / (float)count;
}}

static inline float avgFieldConfidence(const AISupervisorFrame* window, int count) {{
  float total = 0.0f;
  for (int i = 0; i < count; ++i) total += window[i].confidence;
  return total / (float)count;
}}

static inline float avgFieldCurve(const AISupervisorFrame* window, int count) {{
  float total = 0.0f;
  for (int i = 0; i < count; ++i) total += window[i].curveIntensity;
  return total / (float)count;
}}

static inline float avgFieldBalance(const AISupervisorFrame* window, int count) {{
  float total = 0.0f;
  for (int i = 0; i < count; ++i) total += window[i].balanceLR;
  return total / (float)count;
}}

static inline float avgFieldSpeed(const AISupervisorFrame* window, int count) {{
  float total = 0.0f;
  for (int i = 0; i < count; ++i) total += clamp01f(window[i].speedNorm);
  return total / (float)count;
}}

static inline float avgFieldLost(const AISupervisorFrame* window, int count) {{
  float total = 0.0f;
  for (int i = 0; i < count; ++i) total += clamp01f(window[i].lostMsNorm);
  return total / (float)count;
}}

static inline float avgFieldLostRatio(const AISupervisorFrame* window, int count) {{
  float total = 0.0f;
  for (int i = 0; i < count; ++i) total += (window[i].lineLost ? 1.0f : 0.0f);
  return total / (float)count;
}}

static inline float avgFieldSensorSum(const AISupervisorFrame* window, int count) {{
  float total = 0.0f;
  for (int i = 0; i < count; ++i) total += clamp01f((float)window[i].sensorSum / 4000.0f);
  return total / (float)count;
}}

static inline float avgFieldSensorMax(const AISupervisorFrame* window, int count) {{
  float total = 0.0f;
  for (int i = 0; i < count; ++i) total += clamp01f((float)window[i].sensorMax / 1000.0f);
  return total / (float)count;
}}

static inline float avgFieldPidError(const AISupervisorFrame* window, int count) {{
  float total = 0.0f;
  for (int i = 0; i < count; ++i) total += clamp01f(fabsf(window[i].pidError) / 2.0f);
  return total / (float)count;
}}

static inline float avgFieldPidD(const AISupervisorFrame* window, int count) {{
  float total = 0.0f;
  for (int i = 0; i < count; ++i) total += clamp01f(fabsf(window[i].pidD) / 25.0f);
  return total / (float)count;
}}

static inline void buildFeatures(const AISupervisorFrame* window, int count, float* out) {{
  const AISupervisorFrame& last = window[count - 1];
  out[0] = clamp01f(last.sensors[0]);
  out[1] = clamp01f(last.sensors[1]);
  out[2] = clamp01f(last.sensors[2]);
  out[3] = clamp01f(last.sensors[3]);

  float sumS0 = 0.0f;
  float sumS1 = 0.0f;
  float sumS2 = 0.0f;
  float sumS3 = 0.0f;
  for (int i = 0; i < count; ++i) {{
    sumS0 += clamp01f(window[i].sensors[0]);
    sumS1 += clamp01f(window[i].sensors[1]);
    sumS2 += clamp01f(window[i].sensors[2]);
    sumS3 += clamp01f(window[i].sensors[3]);
  }}
  const float invCount = 1.0f / (float)count;
  out[4] = sumS0 * invCount;
  out[5] = sumS1 * invCount;
  out[6] = sumS2 * invCount;
  out[7] = sumS3 * invCount;

  out[8] = last.posNorm;
  out[9] = avgAbsPos(window, count);
  out[10] = avgFieldTrend(window, count);
  out[11] = avgFieldConfidence(window, count);
  out[12] = avgFieldCurve(window, count);
  out[13] = avgFieldBalance(window, count);
  out[14] = avgFieldSpeed(window, count);
  out[15] = avgFieldLost(window, count);
  out[16] = avgFieldLostRatio(window, count);
  out[17] = avgFieldSensorSum(window, count);
  out[18] = avgFieldSensorMax(window, count);
  out[19] = avgFieldPidError(window, count);
  out[20] = avgFieldPidD(window, count);
  out[21] = (float)last.localMode;
  out[22] = last.lineLost ? 1.0f : 0.0f;
}}

static inline int runTree(const float* features) {{
  int node = 0;
  while (kFeatureIndex[node] >= 0) {{
    const int featureIndex = kFeatureIndex[node];
    node = (features[featureIndex] <= kThreshold[node]) ? kChildrenLeft[node] : kChildrenRight[node];
    if (node < 0 || node >= kNodeCount) return 0;
  }}
  return node;
}}

static inline uint8_t inferCurveTrackMode(const AISupervisorFrame* window, int count, float meanBalance) {{
  const AISupervisorFrame& last = window[count - 1];
  const uint8_t lastMode = last.localMode;
  if (lastMode == AI_TRACK_LEFT_SOFT || lastMode == AI_TRACK_LEFT_HARD) return lastMode;
  if (lastMode == AI_TRACK_RIGHT_SOFT || lastMode == AI_TRACK_RIGHT_HARD) return lastMode;
  const bool hardCurve = (fabsf(last.posNorm) > 0.42f) || (fabsf(last.trend) > 0.12f);
  if (meanBalance < 0.0f) return hardCurve ? AI_TRACK_LEFT_HARD : AI_TRACK_LEFT_SOFT;
  return hardCurve ? AI_TRACK_RIGHT_HARD : AI_TRACK_RIGHT_SOFT;
}}

static inline float computeRecoveryBias(const AISupervisorFrame* window, int count, float meanTrend, float meanBalance) {{
  float bias = clampfLocal(0.60f * meanTrend + 0.40f * meanBalance, -1.0f, 1.0f);
  if (fabsf(bias) < 0.12f) {{
    const float fallback = (window[count - 1].balanceLR >= 0.0f) ? 0.25f : -0.25f;
    bias = fallback;
  }}
  return bias;
}}

static inline bool runAIModelHook(const AISupervisorFrame* window, int count, AISupervisorOutput& out) {{
  if (!window || count <= 0) return false;

  float features[kFeatureCount] = {{0}};
  buildFeatures(window, count, features);
  const int leaf = runTree(features);
  const int predictedClass = (leaf >= 0 && leaf < kNodeCount) ? kLeafClass[leaf] : -1;
  if (predictedClass < 0) return false;

  const float meanTrend = avgFieldTrend(window, count);
  const float meanBalance = avgFieldBalance(window, count);
  const float meanCurve = avgFieldCurve(window, count);
  const float lostRatio = avgFieldLostRatio(window, count);
  const float recoveryBias = computeRecoveryBias(window, count, meanTrend, meanBalance);

  out.valid = true;
  out.modelSuggested = true;
  out.blended = false;
  out.source = AI_SOURCE_MODEL;
  out.confidence = clampfLocal(0.88f * kLeafConfidence[leaf] + 0.12f * (1.0f - lostRatio), 0.25f, 0.92f);
  out.recoveryBias = recoveryBias;

  switch (predictedClass) {{
    case 0:
      out.trackMode = AI_TRACK_STRAIGHT;
      out.speedFactor = 1.03f;
      out.kpScale = 0.98f;
      out.kiScale = 1.00f;
      out.kdScale = 0.97f;
      out.pivotThreshold = 0.60f;
      out.pivotCap = 210.0f;
      break;
    case 1: {{
      const bool hardHint =
        (meanCurve > 0.40f) ||
        (fabsf(meanTrend) > 0.10f) ||
        (fabsf(window[count - 1].posNorm) > 0.38f);
      out.trackMode = inferCurveTrackMode(window, count, meanBalance);
      out.speedFactor = hardHint ? 0.90f : 0.95f;
      out.kpScale = hardHint ? 1.10f : 1.05f;
      out.kiScale = hardHint ? 0.88f : 0.93f;
      out.kdScale = hardHint ? 1.16f : 1.10f;
      out.pivotThreshold = hardHint ? 0.46f : 0.50f;
      out.pivotCap = hardHint ? 228.0f : 220.0f;
      break;
    }}
    case 2:
      out.trackMode = AI_TRACK_RECOVER;
      out.speedFactor = 0.66f;
      out.kpScale = 1.04f;
      out.kiScale = 0.00f;
      out.kdScale = 1.36f;
      out.pivotThreshold = 0.34f;
      out.pivotCap = 244.0f;
      break;
    default:
      return false;
  }}

  if (lostRatio > 0.18f) {{
    out.trackMode = AI_TRACK_RECOVER;
    out.speedFactor = min(out.speedFactor, 0.68f);
    out.kiScale = 0.0f;
    out.kdScale = max(out.kdScale, 1.36f);
    out.pivotThreshold = min(out.pivotThreshold, 0.38f);
    out.pivotCap = max(out.pivotCap, 242.0f);
  }}

  return true;
}}

}}  // namespace linefollower_ai_model

using linefollower_ai_model::runAIModelHook;
"""


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_header = Path(args.output_header).resolve()
    report_path = Path(args.report).resolve()
    eval_input_path = Path(args.eval_input).resolve() if args.eval_input else None
    eval_report_path = Path(args.eval_report).resolve()
    eval_predictions_path = Path(args.eval_predictions).resolve()

    if not input_path.exists():
        raise SystemExit(f"No existe el dataset de entrada: {input_path}")
    if eval_input_path and not eval_input_path.exists():
        raise SystemExit(f"No existe el dataset externo de evaluacion: {eval_input_path}")

    x, y, groups, class_counts, _records = load_dataset(input_path, args.window_len)
    train_idx, val_idx, test_idx = split_groups(x, y, groups)

    best_model, best_params = choose_model(
        x[train_idx], y[train_idx], x[val_idx], y[val_idx]
    )

    final_model = DecisionTreeClassifier(
        max_depth=best_params["max_depth"],
        min_samples_leaf=best_params["min_samples_leaf"],
        class_weight="balanced",
        random_state=42,
    )
    train_val_idx = np.concatenate([train_idx, val_idx])
    final_model.fit(x[train_val_idx], y[train_val_idx])

    metrics = {
        "train_val": evaluate_model(final_model, x[train_val_idx], y[train_val_idx]),
        "test": evaluate_model(final_model, x[test_idx], y[test_idx]),
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": input_path.as_posix(),
        "window_len": args.window_len,
        "feature_names": FEATURE_NAMES,
        "feature_importances": build_feature_importance_report(final_model),
        "class_distribution": class_counts,
        "split": {
            "train_val_sessions": sorted({str(g) for g in groups[train_val_idx]}),
            "test_sessions": sorted({str(g) for g in groups[test_idx]}),
            "train_val_rows": int(len(train_val_idx)),
            "test_rows": int(len(test_idx)),
        },
        "model": {
            "type": "DecisionTreeClassifier",
            "params": {
                "max_depth": int(best_params["max_depth"]),
                "min_samples_leaf": int(best_params["min_samples_leaf"]),
                "class_weight": "balanced",
            },
            "node_count": int(final_model.tree_.node_count),
            "max_depth_trained": int(final_model.tree_.max_depth),
        },
        "metrics": metrics,
    }

    external_eval_report: dict[str, object] | None = None
    if eval_input_path:
        eval_x, eval_y, eval_groups, eval_class_counts, eval_records = load_dataset(
            eval_input_path, args.window_len
        )
        eval_metrics = evaluate_model(final_model, eval_x, eval_y)
        prediction_rows = build_prediction_rows(final_model, eval_x, eval_y, eval_records)
        external_eval_report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "trained_on_dataset": input_path.as_posix(),
            "evaluated_dataset": eval_input_path.as_posix(),
            "window_len": args.window_len,
            "feature_names": FEATURE_NAMES,
            "feature_importances": report["feature_importances"],
            "class_distribution": eval_class_counts,
            "sessions": sorted({str(g) for g in eval_groups}),
            "rows": int(len(eval_y)),
            "metrics": eval_metrics,
        }

    header = build_header(
        final_model,
        dataset_path=input_path,
        report_path=report_path,
        feature_names=FEATURE_NAMES,
        metrics=metrics,
        params=report["model"],
    )

    ensure_parent(output_header)
    ensure_parent(report_path)
    output_header.write_text(header, encoding="utf-8", newline="\n")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    if external_eval_report is not None:
        ensure_parent(eval_report_path)
        ensure_parent(eval_predictions_path)
        eval_report_path.write_text(
            json.dumps(external_eval_report, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        pd.DataFrame(build_prediction_rows(final_model, eval_x, eval_y, eval_records)).to_csv(
            eval_predictions_path, index=False
        )

    print(f"Dataset: {input_path}")
    print(f"Filas: {len(y)}")
    print(f"Clases: {class_counts}")
    print(f"Tree nodes: {final_model.tree_.node_count}")
    print(f"Test accuracy: {metrics['test']['accuracy']:.4f}")
    print(f"Test macro F1: {metrics['test']['macro_f1']:.4f}")
    print(f"Header generado: {output_header}")
    print(f"Reporte generado: {report_path}")
    if external_eval_report is not None:
        print(f"Eval dataset: {eval_input_path}")
        print(f"Eval accuracy: {external_eval_report['metrics']['accuracy']:.4f}")
        print(f"Eval macro F1: {external_eval_report['metrics']['macro_f1']:.4f}")
        print(f"Eval report: {eval_report_path}")
        print(f"Eval predictions CSV: {eval_predictions_path}")


if __name__ == "__main__":
    main()
