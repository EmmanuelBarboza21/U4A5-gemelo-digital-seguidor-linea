#pragma once

#include <Arduino.h>

enum AISupervisorSource : uint8_t {
  AI_SOURCE_DISABLED = 0,
  AI_SOURCE_HEURISTIC,
  AI_SOURCE_MODEL,
  AI_SOURCE_MODEL_FALLBACK
};

struct AISupervisorFrame {
  uint32_t sampleMs;
  float sensors[4];
  float posNorm;
  float trend;
  float confidence;
  float curveIntensity;
  float balanceLR;
  float pidError;
  float pidD;
  float speedNorm;
  float lostMsNorm;
  uint16_t sensorSum;
  uint16_t sensorMax;
  uint8_t localMode;
  uint8_t lineLost;
};

struct AISupervisorOutput {
  bool valid;
  bool modelSuggested;
  bool blended;
  uint8_t source;
  uint8_t trackMode;
  float confidence;
  float speedFactor;
  float kpScale;
  float kiScale;
  float kdScale;
  float pivotThreshold;
  float pivotCap;
  float recoveryBias;
};
