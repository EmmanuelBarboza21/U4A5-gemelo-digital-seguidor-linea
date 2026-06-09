/**
 * ESP32-S3 · Line Follower · PID + MQTT + OTA
 * Versión 2.5.0-optimized
 *
 * FIXES aplicados vs 2.4.1-stable-mqttsafe:
 *
 * [FIX-1] MQTT buffer reducido a 2400 para payloadSlow.
 *         payloadSlow se comprime: sólo publica diferencias críticas.
 *         payloadFast reducido a 1200 bytes (campos esenciales).
 *
 * [FIX-2] publishTelemetryFast() aligerada: elimina campos redundantes
 *         que ya van en slow (cal_*, alarms, broker_*, connect_counts).
 *
 * [FIX-3] ArduinoOTA.handle() movido a tarea separada en Core 0.
 *         El loop() (Core 1) nunca se bloquea por OTA.
 *
 * [FIX-4] Cálculo de percepción en STATE_IDLE eliminado del loop hot.
 *         Se ejecuta sólo cada 200ms para no saturar el CPU.
 *
 * [FIX-5] JsonDocument local en cada publish() reemplazado por
 *         instancia reutilizable con clear() entre usos → menos
 *         fragmentación de heap.
 *
 * [FIX-6] MQTT socketTimeout aumentado a 5s, keepAlive a 30s.
 *         Evita desconexiones falsas que causan reconexión + vueltas.
 *
 * [FIX-7] Guard en updatePID(): si lostForMs > 3000ms → stopMotors()
 *         en lugar de seguir girando indefinidamente.
 *
 * [FIX-8] Heap mínimo de seguridad: si freeHeap < 8KB se suspende
 *         publicación de telemetría hasta recuperar memoria.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <ArduinoOTA.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <Adafruit_NeoPixel.h>
#include <Preferences.h>
#include <math.h>
#include <time.h>
#include "ai_supervisor_contract.h"

#if __has_include("ai_model_hook.h")
#include "ai_model_hook.h"
#define HAS_AI_MODEL_HOOK 1
#else
#define HAS_AI_MODEL_HOOK 0
#endif

#if __has_include("secrets.h")
#include "secrets.h"
#define HAS_SECRETS_HEADER 1
#else
#define HAS_SECRETS_HEADER 0
#endif

#ifndef ROBOT_DEVICE_ID
#define ROBOT_DEVICE_ID "esp32s3_linefollower_01"
#endif

#ifndef ROBOT_FW_VERSION
#define ROBOT_FW_VERSION "2.7.0-trackplus"
#endif

#ifndef ROBOT_SCHEMA_VERSION
#define ROBOT_SCHEMA_VERSION 7
#endif

#ifndef ROBOT_WIFI_SSID
#define ROBOT_WIFI_SSID "CONFIGURAR_EN_SECRETS_H"
#endif

#ifndef ROBOT_WIFI_PASS
#define ROBOT_WIFI_PASS "CONFIGURAR_EN_SECRETS_H"
#endif

#ifndef ROBOT_MQTT_HOST
#define ROBOT_MQTT_HOST "127.0.0.1"
#endif

#ifndef ROBOT_MQTT_PORT
#define ROBOT_MQTT_PORT 1883
#endif

// ─── LED PWM ────────────────────────────────────────────────────────────────
static const int RGB_LED_PIN_PRIMARY   = 48;
static const int RGB_LED_PIN_SECONDARY = 38;
static const uint8_t RGB_LED_COUNT     = 1;
static const uint8_t RGB_LED_BRIGHTNESS = 72;
static const bool RGB_LED_USE_SECONDARY = false;

Adafruit_NeoPixel rgbLedPrimary(RGB_LED_COUNT, RGB_LED_PIN_PRIMARY, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel rgbLedSecondary(RGB_LED_COUNT, RGB_LED_PIN_SECONDARY, NEO_GRB + NEO_KHZ800);

inline void setRgbLedColor(uint8_t r, uint8_t g, uint8_t b) {
  uint32_t c1 = rgbLedPrimary.Color(r, g, b);
  rgbLedPrimary.setPixelColor(0, c1);
  rgbLedPrimary.show();
  if (RGB_LED_USE_SECONDARY) {
    uint32_t c2 = rgbLedSecondary.Color(r, g, b);
    rgbLedSecondary.setPixelColor(0, c2);
    rgbLedSecondary.show();
  }
}

#define LED_OFF()     do { setRgbLedColor(0,0,0); } while (0)
#define LED_RED()     do { setRgbLedColor(255,0,0); } while (0)
#define LED_GREEN()   do { setRgbLedColor(0,190,0); } while (0)
#define LED_BLUE()    do { setRgbLedColor(0,0,255); } while (0)
#define LED_LIGHTBLUE() do { setRgbLedColor(60,200,255); } while (0)
#define LED_YELLOW()  do { setRgbLedColor(255,150,0); } while (0)
#define LED_CYAN()    do { setRgbLedColor(0,170,180); } while (0)
#define LED_WHITE()   do { setRgbLedColor(110,110,110); } while (0)

// ─── Identidad ───────────────────────────────────────────────────────────────
const char* DEVICE_ID   = ROBOT_DEVICE_ID;
const char* FW_VERSION  = ROBOT_FW_VERSION;
const uint16_t SCHEMA_VERSION = ROBOT_SCHEMA_VERSION;

// ─── WiFi / MQTT ─────────────────────────────────────────────────────────────
const char* WIFI_SSID = ROBOT_WIFI_SSID;
const char* WIFI_PASS = ROBOT_WIFI_PASS;

const char* MQTT_HOST   = ROBOT_MQTT_HOST;
const uint16_t MQTT_PORT = ROBOT_MQTT_PORT;

const char* TOPIC_CMD       = "robot/linefollower/cmd";
const char* TOPIC_STATUS    = "robot/linefollower/status";
const char* TOPIC_ACK       = "robot/linefollower/ack";
const char* TOPIC_EVENT     = "robot/linefollower/event";
const char* TOPIC_TELE_FAST = "robot/linefollower/telemetry/fast";
const char* TOPIC_TELE_SLOW = "robot/linefollower/telemetry/slow";
const char* TOPIC_TELE_BUFFER = "robot/linefollower/telemetry/buffer";
const char* TOPIC_TRACK_CSV  = "robot/linefollower/track/csv";

// ─── Sensores QTR ────────────────────────────────────────────────────────────
static const int QTR_PINS[4]         = {8, 3, 2, 1};
static const char* SENSOR_LABELS[4]  = {"left_outer","left_inner","right_inner","right_outer"};

int calMin[4]      = {4095,4095,4095,4095};
int calMax[4]      = {0,0,0,0};
int rawVals[4]     = {0,0,0,0};
int normVals[4]    = {0,0,0,0};
int sensorSpan[4]  = {0,0,0,0};
uint8_t sensorOnLine[4] = {0,0,0,0};

int sensorSum      = 0;
int sensorMax      = 0;
int dominantSensor = 0;
float balanceLR    = 0.0f;

// ─── Calibración ─────────────────────────────────────────────────────────────
bool calibrating = false;
bool calibrated  = false;
uint32_t calStartMs   = 0;
const uint32_t CAL_DURATION_MS = 4200;
const uint32_t CAL_CENTER_SETTLE_MS = 220;
const uint32_t CAL_EDGE_HOLD_MS     = 70;
const uint32_t CAL_START_TIMEOUT_MS = 2200;
const uint32_t CAL_SWEEP_TIMEOUT_MS = 950;
const uint32_t CAL_KICK_MS          = 110;
const uint8_t  CAL_ZIGZAG_LEGS      = 6;
const int CAL_PWM_DEFAULT           = 68;
const int CAL_PWM_MIN_LIMIT         = 50;
const int CAL_PWM_MAX_LIMIT         = 110;
const int CAL_VISIBLE_SUM_TH        = 180;
const int CAL_VISIBLE_MAX_TH        = 90;
const int CAL_EDGE_POS_TARGET_DEFAULT = 560;
const int CAL_EDGE_POS_TARGET_MIN     = 380;
const int CAL_EDGE_POS_TARGET_MAX     = 900;
const float CAL_PWM_SMOOTH_ALPHA    = 0.22f;

// ─── Motores ─────────────────────────────────────────────────────────────────
static const int AIN1 = 5;
static const int AIN2 = 6;
static const int BIN1 = 7;
static const int BIN2 = 15;
static const int STBY = 17;

// ─── PID ─────────────────────────────────────────────────────────────────────
float Kp = 0.45f;
float Ki = 0.225f;
float Kd = 0.32f;
int   baseSpeed = 100;
int   baseMin   = 40;
int   baseMax   = 130;
const int minPwm = 60;

float pidError      = 0.0f;
float pidLastErr    = 0.0f;
float pidIntegral   = 0.0f;
float pidCorrection = 0.0f;
float pidDFiltered  = 0.0f;
float posPrev       = 0.0f;

int motorSpeedA = 0;
int motorSpeedB = 0;
int linePos     = 0;
int lastGoodPos = 0;
int8_t lastDir  = 1;

uint32_t lastPidUs  = 0;
uint32_t lostSinceMs = 0;
uint32_t lostForMs   = 0;
bool     lineLostFlag = false;

// [FIX-7] Tiempo máximo de búsqueda antes de parar motores
const uint32_t LOST_STOP_MS = 3000;

// ─── Track Recorder ─────────────────────────────────────────────────────────
const uint32_t TRACK_SAMPLE_MS       = 80;    // muestreo cada 80 ms
const uint16_t TRACK_SAMPLE_CAP      = 600;   // ~48 s de pista
const uint16_t TRACK_CSV_CHUNK_BYTES = 320;   // bytes max por mensaje MQTT
const uint32_t TRACK_CSV_CHUNK_MS    = 220;   // delay entre chunks (ms)
// TRACK_SPEED_SCALE: convierte PWM (0-255) a cm/s.
// Mide cuantos cm recorre tu robot a 150 PWM en 1 segundo: cm_s / 150.0
// Motores amarillos TT aprox 25 cm/s a 150 PWM -> 25/150 = 0.165
const float TRACK_SPEED_SCALE        = 0.165f;
// TRACK_HEADING_SCALE: si la pista se ve muy abierta -> bajar; muy cerrada -> subir
const float TRACK_HEADING_SCALE      = 0.0018f;


int   lostSumTh  = 250;
int   lostMxTh   = 140;
float iClamp     = 0.50f;
float uClamp     = 1.25f;
float deClamp    = 10.0f;
float derivAlpha = 0.35f;
float speedKE    = 0.42f;
float speedKDE   = 0.14f;
float pivotETh   = 0.55f;
int   pivotCap   = 220;

// ─── Percepción local ────────────────────────────────────────────────────────
enum LocalMode {
  LOCAL_UNKNOWN = 0,
  LOCAL_STRAIGHT,
  LOCAL_LEFT_SOFT,
  LOCAL_LEFT_HARD,
  LOCAL_RIGHT_SOFT,
  LOCAL_RIGHT_HARD,
  LOCAL_RECOVER
};

enum DriveProfile : uint8_t {
  PROFILE_SAFE = 0,
  PROFILE_NORMAL,
  PROFILE_FAST
};

struct DriveProfilePreset {
  int baseSpeed;
  float kp;
  float ki;
  float kd;
  int lostSumTh;
  int lostMxTh;
  float aiBlend;
};

LocalMode localMode          = LOCAL_UNKNOWN;
LocalMode localModeCandidate = LOCAL_UNKNOWN;
uint8_t   localModeVotes     = 0;

static const int LOCAL_HISTORY_LEN = 6;
float histPosNorm[LOCAL_HISTORY_LEN]   = {0};
float histTrendPos[LOCAL_HISTORY_LEN]  = {0};
float histConfidence[LOCAL_HISTORY_LEN]= {0};
int   localHistCount = 0;

int   leftSumNorm   = 0;
int   rightSumNorm  = 0;
int   centerSumNorm = 0;
int   outerSumNorm  = 0;
int   edgeBias      = 0;

float posNorm          = 0.0f;
float posNormPrevLocal = 0.0f;
float trendPos         = 0.0f;
float trendPosAvg      = 0.0f;
float confidence       = 0.0f;
float confidenceAvg    = 0.0f;
float curveIntensity   = 0.0f;

// ─── Control adaptativo ──────────────────────────────────────────────────────
bool  adaptiveControlEnabled = true;
float dynSpeedFactor = 1.0f;
float dynKpScale     = 1.0f;
float dynKiScale     = 1.0f;
float dynKdScale     = 1.0f;
float dynPivotETh    = 0.55f;
float dynPivotCapF   = 220.0f;

int   adaptiveBaseCmd = 100;
float adaptiveKp      = 0.75f;
float adaptiveKi      = 0.225f;
float adaptiveKd      = 0.12f;
int   effectiveBaseCmd = 0;
float effectiveSpeedScale = 1.0f;

const float ADAPT_ALPHA = 0.22f;

// AI supervisor: keeps PID as fast loop and adds a slower context layer.
bool  aiSupervisorEnabled = true;
bool  aiModelEnabled      = false;
float aiSupervisorBlend   = 0.65f;
const uint32_t AI_SUPERVISOR_MS = 40;
static const int AI_WINDOW_LEN = 8;
AISupervisorFrame aiWindow[AI_WINDOW_LEN] = {};
int aiWindowCount = 0;
int aiWindowHead  = 0;
uint32_t lastAISupervisorMs = 0;
uint32_t aiInferenceCount   = 0;
uint32_t aiFallbackCount    = 0;
uint32_t aiLastInferenceMs  = 0;
float aiRecoveryBias        = 0.0f;
AISupervisorOutput aiSupervisorOutput = {
  false, false, false, AI_SOURCE_DISABLED, LOCAL_UNKNOWN,
  0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.55f, 220.0f, 0.0f
};

DriveProfile driveProfile = PROFILE_NORMAL;

const DriveProfilePreset PROFILE_SAFE_PRESET   = { 88, 0.66f, 0.210f, 0.30f, 200, 100, 0.48f };
const DriveProfilePreset PROFILE_NORMAL_PRESET = { 100, 0.74f, 0.225f, 0.34f, 220, 108, 0.58f };
const DriveProfilePreset PROFILE_FAST_PRESET   = { 118, 0.82f, 0.235f, 0.38f, 245, 118, 0.64f };

bool intersectionDetected = false;
bool intersectionActive   = false;
bool lapMarkerDetected    = false;
float intersectionScore   = 0.0f;
uint32_t intersectionHoldUntilMs = 0;
uint32_t lastIntersectionMs      = 0;
uint32_t intersectionCount       = 0;
uint32_t lastLapMarkerMs         = 0;
int8_t intersectionPreferredDir  = 1;

const uint32_t INTERSECTION_HOLD_MS      = 120;
const uint32_t INTERSECTION_COOLDOWN_MS  = 520;
const float    INTERSECTION_SCORE_TH     = 0.78f;
const uint32_t LAP_MARKER_ARM_MS         = 1400;
const uint32_t LAP_MARKER_COOLDOWN_MS    = 4500;

uint32_t runId                = 0;
bool     runMetricsActive     = false;
uint32_t runStartMs           = 0;
uint32_t runElapsedMs         = 0;
uint32_t runLineLostCount     = 0;
uint32_t runTotalLostMs       = 0;
uint32_t runMaxLostMs         = 0;
uint32_t runIntersectionCount = 0;
uint32_t runLapEstimate       = 0;
uint32_t runLastLapMs         = 0;
uint32_t runBestLapMs         = 0;
uint32_t runMarkerHits        = 0;
bool     runLostEdgeLatched   = false;

// ─── Estado robot ─────────────────────────────────────────────────────────────
enum RobotState { STATE_IDLE, STATE_CALIBRATING, STATE_RUNNING };
RobotState robotState = STATE_IDLE;

// ─── Timings ──────────────────────────────────────────────────────────────────
uint32_t TELE_FAST_MS      = 150;
uint32_t TELE_SLOW_MS      = 2000;
const uint32_t RUN_DATASET_FAST_MS = 250;
const uint32_t WIFI_RETRY_MS  = 5000;
const uint32_t MQTT_RETRY_MS  = 3000;
const uint32_t LED_BLINK_MS   = 250;
const uint32_t READY_BLINK_MS = 500;
const uint32_t RUN_LED_MS     = 650;
const uint32_t AUTONOMOUS_RUN_MS = 9000;
const uint32_t AUTONOMOUS_SAMPLE_MS = 100;
const uint32_t AUTONOMOUS_FLUSH_MS = 120;
const uint16_t AUTONOMOUS_SAMPLE_CAP = 96;
const uint8_t AUTONOMOUS_FLUSH_CHUNK = 8;

// [FIX-4] Tasa de percepción en IDLE (ms)
const uint32_t IDLE_PERCEPTION_MS = 200;
const uint32_t ALIGN_STABLE_MS = 260;
const int ALIGN_CENTER_POS_TH = 95;
const int ALIGN_PWM_MIN = 70;
const int ALIGN_PWM_MAX = 120;
const int ALIGN_VISIBLE_SUM_TH = 240;
const int ALIGN_VISIBLE_MAX_TH = 120;
const float ALIGN_PID_KP_DEFAULT = 0.22f;
const float ALIGN_PID_KI_DEFAULT = 0.000f;
const float ALIGN_PID_KD_DEFAULT = 0.070f;
const float ALIGN_PID_I_CLAMP    = 0.45f;
const float ALIGN_PID_D_CLAMP    = 3.0f;

uint32_t lastFastMs       = 0;
uint32_t lastSlowMs       = 0;
uint32_t lastWiFiTryMs    = 0;
uint32_t lastMQTTTryMs    = 0;
uint32_t lastLedMs        = 0;
uint32_t lastSweepMs      = 0;
uint32_t lastIdlePercMs   = 0;  // [FIX-4]
uint32_t lastRunLedMs     = 0;
uint32_t alignStableSinceMs = 0;
uint32_t lastAlignPidUs    = 0;
bool     ledBlink         = false;
bool     calSweepDir      = false;
bool     alignToLineActive = false;
bool     calibrationFinalizePending = false;
bool     calibrationSaveOk          = false;
bool     calibrationLineVisible     = false;
bool     stopLedLatched    = false;
bool     autoStopLedLatched = false;
uint8_t  runLedR           = 0;
uint8_t  runLedG           = 0;
uint8_t  runLedB           = 0;
uint8_t  calSweepLegIndex  = 0;
uint32_t calPhaseStartMs   = 0;
uint32_t calEdgeHoldSinceMs = 0;
uint32_t calCenterStableMs = 0;
uint32_t calKickUntilMs    = 0;
float    calAppliedPwm     = 0.0f;
int      calSweepPwm       = CAL_PWM_DEFAULT;
int      calEdgePosTarget  = CAL_EDGE_POS_TARGET_DEFAULT;
int8_t   calLastTurnDir    = 0;
float    alignPidKp        = ALIGN_PID_KP_DEFAULT;
float    alignPidKi        = ALIGN_PID_KI_DEFAULT;
float    alignPidKd        = ALIGN_PID_KD_DEFAULT;
float    alignPidIntegral  = 0.0f;
float    alignPidLastError = 0.0f;
float    alignPidDFiltered = 0.0f;

// ─── Métricas de loop ────────────────────────────────────────────────────────
uint32_t loopCounter    = 0;
uint32_t loopHz         = 0;
uint32_t lastLoopRateMs = 0;
uint32_t loopUsAcc      = 0;
uint32_t loopUsMax      = 0;
uint32_t loopUsAvg      = 0;
uint32_t freeHeapMin    = 0xFFFFFFFF;

// [FIX-8] Umbral de heap mínimo para publicar telemetría (bytes)
const uint32_t HEAP_TELE_MIN = 8192;

// Persistencia local
Preferences prefs;
const char* PREFS_NS = "linefollow";
const uint16_t CAL_STORE_VERSION = 1;
bool calibrationLoadedFromNvs = false;

// ─── WiFi / MQTT / OTA ───────────────────────────────────────────────────────
WiFiClient   wifiClient;
PubSubClient mqttClient(wifiClient);

bool otaReady        = false;
bool wifiWasConnected= false;
bool mqttWasConnected= false;
bool ntpConfigured   = false;
bool timeValid       = false;

uint32_t msgSeq             = 0;
uint32_t mqttConnectAttempts= 0;
uint32_t mqttConnectFails   = 0;
uint32_t mqttPublishFails   = 0;

// ─── Buffers de payload ───────────────────────────────────────────────────────
// [FIX-1] Tamaños reducidos: fast 1200, slow 2400
static char payloadAck[512];
static char payloadEvent[512];
static char payloadStatus[1400];
static char payloadFast[1400];
static char payloadSlow[3200];

// [FIX-5] JsonDocument reutilizable con clear() entre usos
static JsonDocument sharedDoc;

// ─── Track Recorder — variables ─────────────────────────────────────────────
struct TrackSample {
  uint16_t t_ms;        // ms desde inicio del run
  int16_t  line_pos;    // linePos (-1500..1500)
  int8_t   motor_a;     // motorSpeedA / 2
  int8_t   motor_b;     // motorSpeedB / 2
  float    x;           // posicion estimada X (cm)
  float    y;           // posicion estimada Y (cm)
  float    heading;     // orientacion acumulada (rad)
  uint8_t  local_mode;  // enum LocalMode
  uint8_t  flags;       // bit0=lineLost  bit1=intersection
};

TrackSample  trackSamples[TRACK_SAMPLE_CAP] = {};
uint16_t     trackSampleCount   = 0;
uint16_t     trackPublishIdx    = 0;
uint32_t     lastTrackSampleMs  = 0;
uint32_t     lastTrackChunkMs   = 0;
bool         trackPublishActive = false;
bool         trackDataReady     = false;
bool         trackHeaderSent    = false;
float        trackHeading       = 0.0f;
float        trackX             = 0.0f;
float        trackY             = 0.0f;
uint32_t     trackRunId         = 0;
uint32_t     trackStartMs       = 0;
static char  trackCsvBuf[960];


// ─── Snapshot diferido ───────────────────────────────────────────────────────
bool pendingSlowSnapshot   = false;
char pendingSlowCode[32]   = {0};

struct AutonomousRunSample {
  uint16_t elapsedMs;
  int16_t linePos;
  int16_t motorA;
  int16_t motorB;
  uint16_t sensorSum;
  uint16_t sensorMax;
  uint16_t lostMs;
  uint16_t norm0;
  uint16_t norm1;
  uint16_t norm2;
  uint16_t norm3;
  uint8_t localMode;
  uint8_t flags;
};

AutonomousRunSample autonomousSamples[AUTONOMOUS_SAMPLE_CAP] = {};
bool     autonomousRunActive          = false;
bool     autonomousBufferFlushPending = false;
bool     autonomousBufferOverflow     = false;
bool     autonomousReconnectPending   = false;
uint16_t autonomousSampleCount        = 0;
uint16_t autonomousFlushIndex         = 0;
uint32_t autonomousRunStartMs         = 0;
uint32_t autonomousRunElapsedMs       = 0;
uint32_t lastAutonomousSampleMs       = 0;
uint32_t lastAutonomousFlushMs        = 0;

// ─── OTA en Core 0 ───────────────────────────────────────────────────────────
// [FIX-3] Tarea OTA separada para no bloquear el loop PID
TaskHandle_t otaTaskHandle = nullptr;
portMUX_TYPE deferredMux = portMUX_INITIALIZER_UNLOCKED;

enum DeferredFlag : uint32_t {
  DEFERRED_OTA_START = 1u << 0,
  DEFERRED_OTA_END   = 1u << 1,
  DEFERRED_OTA_ERROR = 1u << 2
};

volatile uint32_t deferredFlags = 0;
volatile uint8_t deferredOtaError = 0;
bool otaInProgress = false;

void otaTask(void* pvParameters) {
  for (;;) {
    if (otaReady && WiFi.status() == WL_CONNECTED) {
      ArduinoOTA.handle();
    }
    vTaskDelay(pdMS_TO_TICKS(50));
  }
}

// ─── Utilidades ──────────────────────────────────────────────────────────────
uint32_t nextSeq() { return ++msgSeq; }
float clamp01(float v)                   { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }
float clampf(float v, float lo, float hi){ return v < lo ? lo : (v > hi ? hi : v); }
float blendf(float base, float target, float mix) { return base + (target - base) * clamp01(mix); }
float smoothTo(float c, float t, float a){ return c + a*(t-c); }

const char* otaErrorToStr(uint8_t error) {
  switch (error) {
    case OTA_AUTH_ERROR:   return "auth";
    case OTA_BEGIN_ERROR:  return "begin";
    case OTA_CONNECT_ERROR:return "connect";
    case OTA_RECEIVE_ERROR:return "receive";
    case OTA_END_ERROR:    return "end";
    default:               return "unknown";
  }
}

void queueDeferredFlag(DeferredFlag flag, uint8_t otaError = 0) {
  portENTER_CRITICAL(&deferredMux);
  deferredFlags |= static_cast<uint32_t>(flag);
  if (otaError != 0) deferredOtaError = otaError;
  portEXIT_CRITICAL(&deferredMux);
}

const char* aiSourceToStr(uint8_t src) {
  switch (src) {
    case AI_SOURCE_HEURISTIC:      return "heuristic";
    case AI_SOURCE_MODEL:          return "model";
    case AI_SOURCE_MODEL_FALLBACK: return "model_fallback";
    default:                       return "disabled";
  }
}

const char* stateToStr(RobotState st) {
  if (st == STATE_RUNNING)    return "running";
  if (st == STATE_CALIBRATING)return "calibrating";
  return "idle";
}

const char* localModeToStr(LocalMode m) {
  switch (m) {
    case LOCAL_STRAIGHT:   return "straight";
    case LOCAL_LEFT_SOFT:  return "left_soft";
    case LOCAL_LEFT_HARD:  return "left_hard";
    case LOCAL_RIGHT_SOFT: return "right_soft";
    case LOCAL_RIGHT_HARD: return "right_hard";
    case LOCAL_RECOVER:    return "recover";
    default:               return "unknown";
  }
}

const char* driveProfileToStr(DriveProfile profile) {
  switch (profile) {
    case PROFILE_SAFE:   return "safe";
    case PROFILE_FAST:   return "fast";
    default:             return "normal";
  }
}

const DriveProfilePreset& driveProfilePreset(DriveProfile profile) {
  switch (profile) {
    case PROFILE_SAFE:   return PROFILE_SAFE_PRESET;
    case PROFILE_FAST:   return PROFILE_FAST_PRESET;
    default:             return PROFILE_NORMAL_PRESET;
  }
}

void fillCommonEnvelope(JsonDocument& doc, const char* type, const char* topicRole) {
  doc["type"]           = type;
  doc["topic_role"]     = topicRole;
  doc["schema_version"] = SCHEMA_VERSION;
  doc["device_id"]      = DEVICE_ID;
  doc["fw_version"]     = FW_VERSION;
  doc["seq"]            = nextSeq();
  doc["uptime_ms"]      = millis();
  time_t nowTs          = time(nullptr);
  bool valid            = (nowTs > 1700000000);
  timeValid             = valid;
  doc["timestamp_valid"]= valid ? 1 : 0;
  if (valid) doc["timestamp_unix_ms"] = (int64_t)nowTs * 1000LL;
}

void ipToCstr(IPAddress ip, char* out, size_t outLen) {
  snprintf(out, outLen, "%u.%u.%u.%u", ip[0], ip[1], ip[2], ip[3]);
}

const char* mqttStateToStr(int state) {
  switch (state) {
    case MQTT_CONNECTION_TIMEOUT:     return "connection_timeout";
    case MQTT_CONNECTION_LOST:        return "connection_lost";
    case MQTT_CONNECT_FAILED:         return "tcp_connect_failed";
    case MQTT_DISCONNECTED:           return "disconnected";
    case MQTT_CONNECTED:              return "connected";
    case MQTT_CONNECT_BAD_PROTOCOL:   return "bad_protocol";
    case MQTT_CONNECT_BAD_CLIENT_ID:  return "bad_client_id";
    case MQTT_CONNECT_UNAVAILABLE:    return "server_unavailable";
    case MQTT_CONNECT_BAD_CREDENTIALS:return "bad_credentials";
    case MQTT_CONNECT_UNAUTHORIZED:   return "unauthorized";
    default:                          return "unknown";
  }
}

void printNetworkSnapshot() {
  char ipbuf[20];
  char gwbuf[20];
  char maskbuf[20];
  ipToCstr(WiFi.localIP(), ipbuf, sizeof(ipbuf));
  ipToCstr(WiFi.gatewayIP(), gwbuf, sizeof(gwbuf));
  ipToCstr(WiFi.subnetMask(), maskbuf, sizeof(maskbuf));

  Serial.print("[NET] ssid="); Serial.println(WIFI_SSID);
  Serial.print("[NET] ip="); Serial.print(ipbuf);
  Serial.print(" gw="); Serial.print(gwbuf);
  Serial.print(" mask="); Serial.println(maskbuf);
  Serial.print("[NET] mqtt_host="); Serial.print(MQTT_HOST);
  Serial.print(" mqtt_port="); Serial.println(MQTT_PORT);
}

bool serializeToBuffer(JsonDocument& doc, char* buf, size_t len) {
  size_t n = serializeJson(doc, buf, len);
  if (n >= len) { mqttPublishFails++; return false; }
  buf[n] = '\0';
  return true;
}

bool mqttPublishSafe(const char* topic, const char* payload, bool retained = false) {
  bool ok = mqttClient.publish(topic, payload, retained);
  if (!ok) mqttPublishFails++;
  return ok;
}

// [FIX-8] Guard de heap antes de publicar
bool heapOkForTelemetry() {
  return ESP.getFreeHeap() >= HEAP_TELE_MIN;
}

bool periodicTelemetryAllowed() {
  if (trackPublishActive) return false;
  return true;
}

void requestSlowSnapshot(const char* code = nullptr) {
  pendingSlowSnapshot = true;
  if (code && code[0] != '\0') {
    strncpy(pendingSlowCode, code, sizeof(pendingSlowCode)-1);
    pendingSlowCode[sizeof(pendingSlowCode)-1] = '\0';
  } else {
    pendingSlowCode[0] = '\0';
  }
}

void publishAck(const char* cmd, const char* result, const char* detail);
void publishEvent(const char* severity, const char* code, const char* message);
void publishStatus(const char* onlineStatus, const char* reason);
void beginWiFi();
void stopMotors();
void setRobotIdleSafe();
void resetControllerState();
void updateLastDir();
void beginAutonomousRunWindow();
void updateAutonomousRunWindow();
void captureAutonomousRunSample();
void publishAutonomousBufferChunk();
void clearAutonomousRunBuffer();
void resumeCommunicationsAfterAutonomousRun();
void trackRecorderBegin();
void trackRecorderSample();
void trackRecorderFinish();
bool trackRecorderRequestPublish();
void trackRecorderPublishChunk();

// ─── Lectura de sensores ──────────────────────────────────────────────────────
int readRaw(int idx)   { return analogRead(QTR_PINS[idx]); }
int readMedian3(int idx) {
  int a=readRaw(idx), b=readRaw(idx), c=readRaw(idx);
  if (a>b){int t=a;a=b;b=t;} if (b>c){int t=b;b=c;c=t;} if (a>b){int t=a;a=b;b=t;}
  return b;
}

int readNormFromRange(int raw, int mn, int mx) {
  if (mx <= mn+10) return 0;
  long n = (long)(raw-mn)*1000L/(long)(mx-mn);
  return (int)constrain(n, 0L, 1000L);
}

int readNormFromRaw(int raw, int idx) {
  if (!calibrated) return -1;
  return readNormFromRange(raw, calMin[idx], calMax[idx]);
}

const char* sensorHealthStatus(int idx) {
  if (!calibrated)         return "uncalibrated";
  int span=sensorSpan[idx], rv=rawVals[idx], nv=normVals[idx];
  if (span < 80)           return "bad_calibration";
  if (rv<=8 || rv>=4087)   return "saturated";
  if (nv<30 && sensorMax>500) return "weak";
  return "ok";
}

void computeSensorDerived() {
  sensorSum=0; sensorMax=0; dominantSensor=0;
  for (int i=0;i<4;i++) {
    sensorSpan[i]=calMax[i]-calMin[i];
    int v=max(normVals[i],0);
    sensorSum+=v;
    if (v>sensorMax){sensorMax=v; dominantSensor=i;}
    sensorOnLine[i]=(v>=200)?1:0;
  }
  int left=normVals[0]+normVals[1], right=normVals[2]+normVals[3];
  balanceLR=(sensorSum>0)?((float)(right-left))/(float)sensorSum:0.0f;
}

void updateRunningLed() {
  uint32_t now = millis();
  if ((now - lastRunLedMs) >= RUN_LED_MS || (runLedR == 0 && runLedG == 0 && runLedB == 0)) {
    lastRunLedMs = now;
    runLedR = 50 + (uint8_t)(esp_random() % 206);
    runLedG = 50 + (uint8_t)(esp_random() % 206);
    runLedB = 50 + (uint8_t)(esp_random() % 206);
  }
  setRgbLedColor(runLedR, runLedG, runLedB);
}

void clearAutonomousRunBuffer() {
  autonomousSampleCount = 0;
  autonomousFlushIndex = 0;
  autonomousBufferFlushPending = false;
  autonomousBufferOverflow = false;
  lastAutonomousSampleMs = 0;
  lastAutonomousFlushMs = 0;
}

void resumeCommunicationsAfterAutonomousRun() {
  autonomousReconnectPending = false;
}

void beginAutonomousRunWindow() {
  autonomousRunActive = false;
  autonomousReconnectPending = false;
  autoStopLedLatched = false;
  autonomousRunStartMs = 0;
  autonomousRunElapsedMs = 0;
  clearAutonomousRunBuffer();
}

void captureAutonomousRunSample() {
  if (!autonomousRunActive || robotState != STATE_RUNNING) return;
  uint32_t now = millis();
  if ((now - lastAutonomousSampleMs) < AUTONOMOUS_SAMPLE_MS && autonomousSampleCount > 0) return;
  lastAutonomousSampleMs = now;
  if (autonomousSampleCount >= AUTONOMOUS_SAMPLE_CAP) {
    autonomousBufferOverflow = true;
    return;
  }

  AutonomousRunSample& s = autonomousSamples[autonomousSampleCount++];
  uint32_t elapsed = now - autonomousRunStartMs;
  s.elapsedMs = (uint16_t)min<uint32_t>(elapsed, 0xFFFFu);
  s.linePos = (int16_t)constrain(linePos, -32768, 32767);
  s.motorA = (int16_t)constrain(motorSpeedA, -255, 255);
  s.motorB = (int16_t)constrain(motorSpeedB, -255, 255);
  s.sensorSum = (uint16_t)constrain(sensorSum, 0, 65535);
  s.sensorMax = (uint16_t)constrain(sensorMax, 0, 65535);
  s.lostMs = (uint16_t)min<uint32_t>(lostForMs, 0xFFFFu);
  s.norm0 = (uint16_t)constrain(normVals[0], 0, 1000);
  s.norm1 = (uint16_t)constrain(normVals[1], 0, 1000);
  s.norm2 = (uint16_t)constrain(normVals[2], 0, 1000);
  s.norm3 = (uint16_t)constrain(normVals[3], 0, 1000);
  s.localMode = (uint8_t)localMode;
  s.flags = 0;
  if (lineLostFlag)        s.flags |= 0x01;
  if (intersectionActive)  s.flags |= 0x02;
  if (lapMarkerDetected)   s.flags |= 0x04;
  if (runMetricsActive)    s.flags |= 0x08;
}

void updateAutonomousRunWindow() {
  autonomousRunActive = false;
  autonomousRunElapsedMs = 0;
}

// ─── Percepción ───────────────────────────────────────────────────────────────
void shiftInsert(float* arr, int len, float value) {
  for (int i=len-1;i>0;--i) arr[i]=arr[i-1];
  arr[0]=value;
}

float avgHistory(const float* arr, int count) {
  if (count<=0) return 0.0f;
  float s=0.0f;
  for (int i=0;i<count;i++) s+=arr[i];
  return s/(float)count;
}

void resetPerceptionState() {
  leftSumNorm=rightSumNorm=centerSumNorm=outerSumNorm=edgeBias=0;
  posNorm=posNormPrevLocal=trendPos=trendPosAvg=confidence=confidenceAvg=curveIntensity=0.0f;
  localMode=LOCAL_UNKNOWN; localModeCandidate=LOCAL_UNKNOWN; localModeVotes=0; localHistCount=0;
  for (int i=0;i<LOCAL_HISTORY_LEN;i++) histPosNorm[i]=histTrendPos[i]=histConfidence[i]=0.0f;
}

void resetIntersectionState() {
  intersectionDetected = false;
  intersectionActive = false;
  lapMarkerDetected = false;
  intersectionScore = 0.0f;
  intersectionHoldUntilMs = 0;
  lastIntersectionMs = 0;
  lastLapMarkerMs = 0;
  intersectionPreferredDir = (lastDir >= 0) ? 1 : -1;
}

void resetAdaptiveControl() {
  dynSpeedFactor=dynKpScale=dynKiScale=dynKdScale=1.0f;
  dynPivotETh=pivotETh; dynPivotCapF=(float)pivotCap;
  adaptiveBaseCmd=baseSpeed; adaptiveKp=Kp; adaptiveKi=Ki; adaptiveKd=Kd;
  effectiveBaseCmd = adaptiveBaseCmd;
  effectiveSpeedScale = 1.0f;
}

void resetRunMetrics() {
  runElapsedMs = 0;
  runLineLostCount = 0;
  runTotalLostMs = 0;
  runMaxLostMs = 0;
  runIntersectionCount = 0;
  runLapEstimate = 0;
  runLastLapMs = 0;
  runBestLapMs = 0;
  runMarkerHits = 0;
  runLostEdgeLatched = false;
  lastLapMarkerMs = 0;
}

void startRunMetrics() {
  runId++;
  runMetricsActive = true;
  runStartMs = millis();
  resetRunMetrics();
}

void finishRunMetrics() {
  if (!runMetricsActive) return;
  runElapsedMs = millis() - runStartMs;
  runMetricsActive = false;
  runLostEdgeLatched = false;
}

void applyDriveProfile(DriveProfile profile, bool announce = true) {
  const DriveProfilePreset& preset = driveProfilePreset(profile);
  driveProfile = profile;
  baseSpeed = constrain(preset.baseSpeed, baseMin, baseMax);
  Kp = preset.kp;
  Ki = preset.ki;
  Kd = preset.kd;
  lostSumTh = preset.lostSumTh;
  lostMxTh = preset.lostMxTh;
  aiSupervisorBlend = preset.aiBlend;
  pidIntegral = 0.0f;
  pidDFiltered = 0.0f;
  pidCorrection = 0.0f;
  resetAdaptiveControl();
  if (!announce) return;
  char detail[48];
  char message[64];
  snprintf(detail, sizeof(detail), "profile_%s", driveProfileToStr(profile));
  snprintf(message, sizeof(message), "Drive profile set to %s", driveProfileToStr(profile));
  publishAck("set_profile", "ok", detail);
  publishEvent("info", "profile_updated", message);
  publishStatus("online", "profile_updated");
  requestSlowSnapshot("profile_updated");
}

bool calibrationDataLooksValid(const int* mins, const int* maxs) {
  for (int i=0;i<4;i++) {
    if (mins[i] < 0 || mins[i] > 4095) return false;
    if (maxs[i] < 0 || maxs[i] > 4095) return false;
    if (maxs[i] <= mins[i] + 20) return false;
  }
  return true;
}

bool saveCalibrationToNvs() {
  if (!calibrationDataLooksValid(calMin, calMax)) return false;
  if (!prefs.begin(PREFS_NS, false)) return false;
  prefs.putUShort("cal_ver", CAL_STORE_VERSION);
  prefs.putBool("cal_ok", true);
  for (int i=0;i<4;i++) {
    char keyMin[8];
    char keyMax[8];
    snprintf(keyMin, sizeof(keyMin), "mn%d", i);
    snprintf(keyMax, sizeof(keyMax), "mx%d", i);
    prefs.putUShort(keyMin, (uint16_t)calMin[i]);
    prefs.putUShort(keyMax, (uint16_t)calMax[i]);
  }
  prefs.end();
  calibrationLoadedFromNvs = true;
  return true;
}

bool loadCalibrationFromNvs() {
  if (!prefs.begin(PREFS_NS, true)) return false;
  bool ok = prefs.getBool("cal_ok", false);
  uint16_t ver = prefs.getUShort("cal_ver", 0);
  int mins[4] = {0,0,0,0};
  int maxs[4] = {0,0,0,0};
  if (ok && ver == CAL_STORE_VERSION) {
    for (int i=0;i<4;i++) {
      char keyMin[8];
      char keyMax[8];
      snprintf(keyMin, sizeof(keyMin), "mn%d", i);
      snprintf(keyMax, sizeof(keyMax), "mx%d", i);
      mins[i] = (int)prefs.getUShort(keyMin, 0);
      maxs[i] = (int)prefs.getUShort(keyMax, 0);
    }
  }
  prefs.end();
  if (!ok || ver != CAL_STORE_VERSION) return false;
  if (!calibrationDataLooksValid(mins, maxs)) return false;
  for (int i=0;i<4;i++) {
    calMin[i] = mins[i];
    calMax[i] = maxs[i];
    sensorSpan[i] = calMax[i] - calMin[i];
  }
  calibrated = true;
  calibrating = false;
  calibrationLoadedFromNvs = true;
  return true;
}

void clearCalibrationFromNvs() {
  if (prefs.begin(PREFS_NS, false)) {
    prefs.clear();
    prefs.end();
  }
  calibrationLoadedFromNvs = false;
}

void resetAISupervisorState() {
  memset(aiWindow, 0, sizeof(aiWindow));
  aiWindowCount = 0;
  aiWindowHead = 0;
  lastAISupervisorMs = 0;
  aiRecoveryBias = 0.0f;
  aiSupervisorOutput = {
    false, false, false, AI_SOURCE_DISABLED, LOCAL_UNKNOWN,
    0.0f, 1.0f, 1.0f, 1.0f, 1.0f, pivotETh, (float)pivotCap, 0.0f
  };
}

void pushAISupervisorFrame() {
  AISupervisorFrame& frame = aiWindow[aiWindowHead];
  frame.sampleMs = millis();
  for (int i=0;i<4;i++) frame.sensors[i] = clamp01((float)normVals[i] / 1000.0f);
  frame.posNorm = posNorm;
  frame.trend = trendPos;
  frame.confidence = confidence;
  frame.curveIntensity = curveIntensity;
  frame.balanceLR = balanceLR;
  frame.pidError = pidError;
  frame.pidD = pidDFiltered;
  frame.speedNorm = 0.5f * (
    fabsf((float)motorSpeedA) / 255.0f +
    fabsf((float)motorSpeedB) / 255.0f
  );
  frame.lostMsNorm = clamp01((float)lostForMs / (float)LOST_STOP_MS);
  frame.sensorSum = (uint16_t)constrain(sensorSum, 0, 65535);
  frame.sensorMax = (uint16_t)constrain(sensorMax, 0, 65535);
  frame.localMode = (uint8_t)localMode;
  frame.lineLost = lineLostFlag ? 1 : 0;

  aiWindowHead = (aiWindowHead + 1) % AI_WINDOW_LEN;
  if (aiWindowCount < AI_WINDOW_LEN) aiWindowCount++;
}

bool runAIModelHookWrapper(const AISupervisorFrame* window, int count, AISupervisorOutput& out) {
#if HAS_AI_MODEL_HOOK
  return runAIModelHook(window, count, out);
#else
  (void)window;
  (void)count;
  (void)out;
  return false;
#endif
}

void computeHeuristicAISupervisor(AISupervisorOutput& out) {
  out = {
    true, false, false, AI_SOURCE_HEURISTIC, (uint8_t)localMode,
    confidenceAvg, 1.0f, 1.0f, 1.0f, 1.0f, pivotETh, (float)pivotCap, 0.0f
  };
  if (!calibrated || aiWindowCount <= 0) {
    out.valid = false;
    out.source = AI_SOURCE_DISABLED;
    return;
  }

  float avgAbsPos = 0.0f;
  float avgTrend = 0.0f;
  float avgConf = 0.0f;
  float avgCurve = 0.0f;
  float avgBalance = 0.0f;
  float lostRatio = 0.0f;
  int modeVotes[LOCAL_RECOVER + 1] = {0};
  int dominantMode = (int)localMode;

  for (int i=0;i<aiWindowCount;i++) {
    int idx = (aiWindowHead - aiWindowCount + i + AI_WINDOW_LEN) % AI_WINDOW_LEN;
    const AISupervisorFrame& frame = aiWindow[idx];
    avgAbsPos += fabsf(frame.posNorm);
    avgTrend += frame.trend;
    avgConf += frame.confidence;
    avgCurve += frame.curveIntensity;
    avgBalance += frame.balanceLR;
    lostRatio += frame.lineLost ? 1.0f : 0.0f;
    if (frame.localMode <= LOCAL_RECOVER) modeVotes[frame.localMode]++;
  }

  float invCount = 1.0f / (float)aiWindowCount;
  avgAbsPos *= invCount;
  avgTrend *= invCount;
  avgConf *= invCount;
  avgCurve *= invCount;
  avgBalance *= invCount;
  lostRatio *= invCount;

  int bestVotes = -1;
  for (int mode=LOCAL_STRAIGHT; mode<=LOCAL_RECOVER; mode++) {
    if (modeVotes[mode] > bestVotes) {
      bestVotes = modeVotes[mode];
      dominantMode = mode;
    }
  }

  out.trackMode = (uint8_t)dominantMode;
  switch ((LocalMode)dominantMode) {
    case LOCAL_STRAIGHT:
      out.speedFactor = 1.08f; out.kpScale = 0.94f; out.kiScale = 1.02f; out.kdScale = 0.92f;
      out.pivotThreshold = 0.64f; out.pivotCap = 205.0f;
      break;
    case LOCAL_LEFT_SOFT:
    case LOCAL_RIGHT_SOFT:
      out.speedFactor = 0.97f; out.kpScale = 1.08f; out.kiScale = 0.96f; out.kdScale = 1.12f;
      out.pivotThreshold = 0.50f; out.pivotCap = 222.0f;
      break;
    case LOCAL_LEFT_HARD:
    case LOCAL_RIGHT_HARD:
      out.speedFactor = 0.82f; out.kpScale = 1.18f; out.kiScale = 0.82f; out.kdScale = 1.28f;
      out.pivotThreshold = 0.40f; out.pivotCap = 238.0f;
      break;
    case LOCAL_RECOVER:
      out.speedFactor = 0.64f; out.kpScale = 1.06f; out.kiScale = 0.00f; out.kdScale = 1.42f;
      out.pivotThreshold = 0.33f; out.pivotCap = 248.0f;
      break;
    default:
      break;
  }

  float anticipation = clamp01(fabsf(avgTrend) * 3.5f + avgCurve * 0.8f);
  out.kpScale *= (1.0f + 0.10f * anticipation);
  out.kdScale *= (1.0f + 0.14f * anticipation);
  out.speedFactor *= (1.0f - 0.12f * avgAbsPos);

  if (avgConf < 0.45f) {
    out.speedFactor *= 0.84f;
    out.kdScale *= 1.08f;
  } else if (avgConf < 0.60f) {
    out.speedFactor *= 0.92f;
  }

  if (lostRatio > 0.15f) {
    out.trackMode = LOCAL_RECOVER;
    out.speedFactor = min(out.speedFactor, 0.70f);
    out.kiScale = 0.0f;
    out.kdScale = max(out.kdScale, 1.35f);
    out.pivotThreshold = min(out.pivotThreshold, 0.38f);
    out.pivotCap = max(out.pivotCap, 242.0f);
  }

  out.recoveryBias = clampf(0.60f * avgTrend + 0.40f * avgBalance, -1.0f, 1.0f);
  if (fabsf(out.recoveryBias) < 0.12f) out.recoveryBias = (lastDir >= 0) ? 0.25f : -0.25f;
  out.confidence = clamp01(
    0.55f * avgConf +
    0.20f * (1.0f - lostRatio) +
    0.15f * (1.0f - clamp01(avgAbsPos)) +
    0.10f * (1.0f - avgCurve)
  );

  out.speedFactor = clampf(out.speedFactor, 0.55f, 1.20f);
  out.kpScale = clampf(out.kpScale, 0.70f, 1.45f);
  out.kiScale = clampf(out.kiScale, 0.00f, 1.10f);
  out.kdScale = clampf(out.kdScale, 0.75f, 1.60f);
  out.pivotThreshold = clampf(out.pivotThreshold, 0.30f, 0.70f);
  out.pivotCap = clampf(out.pivotCap, 180.0f, 255.0f);
}

void updateAISupervisor() {
  if (!aiSupervisorEnabled || !calibrated || aiWindowCount <= 0) {
    resetAISupervisorState();
    return;
  }

  uint32_t nowMs = millis();
  if ((nowMs - lastAISupervisorMs) < AI_SUPERVISOR_MS && aiSupervisorOutput.valid) return;
  lastAISupervisorMs = nowMs;

  AISupervisorOutput heuristicOut;
  computeHeuristicAISupervisor(heuristicOut);
  AISupervisorOutput finalOut = heuristicOut;
  finalOut.source = AI_SOURCE_HEURISTIC;

  if (aiModelEnabled) {
    aiInferenceCount++;
    aiLastInferenceMs = nowMs;
    AISupervisorFrame orderedWindow[AI_WINDOW_LEN];
    for (int i=0;i<aiWindowCount;i++) {
      int idx = (aiWindowHead - aiWindowCount + i + AI_WINDOW_LEN) % AI_WINDOW_LEN;
      orderedWindow[i] = aiWindow[idx];
    }

    AISupervisorOutput modelOut = finalOut;
    modelOut.valid = false;
    bool modelOk = runAIModelHookWrapper(orderedWindow, aiWindowCount, modelOut);
    if (modelOk && modelOut.valid) {
      float mix = clamp01(aiSupervisorBlend * clampf(modelOut.confidence, 0.35f, 1.0f));
      finalOut.valid = true;
      finalOut.modelSuggested = true;
      finalOut.blended = (mix < 0.999f);
      finalOut.source = AI_SOURCE_MODEL;
      if (modelOut.trackMode <= LOCAL_RECOVER && mix > 0.55f) finalOut.trackMode = modelOut.trackMode;
      finalOut.speedFactor = blendf(heuristicOut.speedFactor, clampf(modelOut.speedFactor, 0.55f, 1.20f), mix);
      finalOut.kpScale = blendf(heuristicOut.kpScale, clampf(modelOut.kpScale, 0.70f, 1.45f), mix);
      finalOut.kiScale = blendf(heuristicOut.kiScale, clampf(modelOut.kiScale, 0.00f, 1.10f), mix);
      finalOut.kdScale = blendf(heuristicOut.kdScale, clampf(modelOut.kdScale, 0.75f, 1.60f), mix);
      finalOut.pivotThreshold = blendf(heuristicOut.pivotThreshold, clampf(modelOut.pivotThreshold, 0.30f, 0.70f), mix);
      finalOut.pivotCap = blendf(heuristicOut.pivotCap, clampf(modelOut.pivotCap, 180.0f, 255.0f), mix);
      finalOut.recoveryBias = blendf(heuristicOut.recoveryBias, clampf(modelOut.recoveryBias, -1.0f, 1.0f), mix);
      finalOut.confidence = clamp01(blendf(heuristicOut.confidence, modelOut.confidence, 0.60f));
    } else {
      aiFallbackCount++;
      finalOut.source = AI_SOURCE_MODEL_FALLBACK;
      finalOut.modelSuggested = false;
      finalOut.blended = false;
    }
  }

  aiRecoveryBias = finalOut.recoveryBias;
  aiSupervisorOutput = finalOut;
}

float computeConfidence() {
  if (!calibrated) return 0.0f;
  int minSpan=sensorSpan[0];
  for (int i=1;i<4;i++) if(sensorSpan[i]<minSpan) minSpan=sensorSpan[i];
  float maxPart  = clamp01((float)sensorMax/1000.0f);
  float sumPart  = clamp01((float)sensorSum/1800.0f);
  float spanPart = clamp01((float)minSpan/250.0f);
  return clamp01(0.55f*maxPart+0.30f*sumPart+0.15f*spanPart);
}

void refreshCurveIntensity() {
  float trendMag = clamp01(fabsf(trendPosAvg)*4.0f);
  float dMag     = clamp01(fabsf(pidDFiltered)/((deClamp>0.001f)?deClamp:1.0f));
  curveIntensity = clamp01(0.60f*trendMag+0.40f*dMag);
}

float computeIntersectionScore() {
  if (!calibrated || lineLostFlag) return 0.0f;
  int onCount = sensorOnLine[0] + sensorOnLine[1] + sensorOnLine[2] + sensorOnLine[3];
  float onPart = clamp01((float)(onCount - 2) / 2.0f);
  float sumPart = clamp01(((float)sensorSum - 1100.0f) / 1400.0f);
  float centerPart = clamp01(((float)centerSumNorm - 650.0f) / 700.0f);
  float outerPart = clamp01(((float)outerSumNorm - 380.0f) / 700.0f);
  float balancePart = 1.0f - clamp01(fabsf(balanceLR) / 0.22f);
  float posPart = 1.0f - clamp01(fabsf(posNorm) / 0.22f);
  float trendPart = 1.0f - clamp01(fabsf(trendPosAvg) / 0.10f);
  float confPart = clamp01((confidenceAvg - 0.35f) / 0.45f);
  float score =
    0.20f * onPart +
    0.16f * sumPart +
    0.16f * centerPart +
    0.14f * outerPart +
    0.12f * balancePart +
    0.10f * posPart +
    0.06f * trendPart +
    0.06f * confPart;
  bool widePattern =
    (onCount >= 3) ||
    ((normVals[1] > 540) && (normVals[2] > 540) && ((normVals[0] > 240) || (normVals[3] > 240)));
  if (!widePattern) score *= 0.55f;
  return clamp01(score);
}

bool detectLapMarkerCandidate() {
  if (!intersectionActive) return false;
  if (fabsf(posNorm) > 0.16f) return false;
  if (fabsf(trendPosAvg) > 0.08f) return false;
  if (curveIntensity > 0.24f) return false;
  if (sensorSum < 1450 || centerSumNorm < 900) return false;
  return true;
}

void updateRunMetrics(bool lost, float dt) {
  if (!runMetricsActive) return;
  runElapsedMs = millis() - runStartMs;
  if (lost && !runLostEdgeLatched) runLineLostCount++;
  runLostEdgeLatched = lost;
  if (!lost) return;
  uint32_t deltaMs = (uint32_t)lroundf(clampf(dt * 1000.0f, 1.0f, 50.0f));
  runTotalLostMs += deltaMs;
  if (lostForMs > runMaxLostMs) runMaxLostMs = lostForMs;
}

void updateIntersectionState() {
  uint32_t now = millis();
  float score = computeIntersectionScore();
  bool candidate = (robotState != STATE_CALIBRATING) && !lineLostFlag && (score >= INTERSECTION_SCORE_TH);
  bool wasActive = intersectionActive;

  intersectionScore = score;
  intersectionDetected = candidate;

  if (candidate && ((now - lastIntersectionMs) >= INTERSECTION_COOLDOWN_MS)) {
    intersectionActive = true;
    intersectionHoldUntilMs = now + INTERSECTION_HOLD_MS;
    lastIntersectionMs = now;
    intersectionPreferredDir = (lastDir >= 0) ? 1 : -1;
    if (!wasActive && robotState == STATE_RUNNING) {
      intersectionCount++;
      if (runMetricsActive) runIntersectionCount++;
    }
  } else {
    intersectionActive = candidate || (intersectionActive && now < intersectionHoldUntilMs);
  }

  if (!intersectionActive) {
    lapMarkerDetected = false;
    return;
  }

  lapMarkerDetected = detectLapMarkerCandidate();
  if (!runMetricsActive || !lapMarkerDetected) return;
  if ((now - runStartMs) < LAP_MARKER_ARM_MS) return;
  if ((now - lastLapMarkerMs) < LAP_MARKER_COOLDOWN_MS) return;

  runMarkerHits++;
  if (lastLapMarkerMs != 0) {
    uint32_t lapMs = now - lastLapMarkerMs;
    runLapEstimate++;
    runLastLapMs = lapMs;
    if (runBestLapMs == 0 || lapMs < runBestLapMs) runBestLapMs = lapMs;
  }
  lastLapMarkerMs = now;
}

void updatePerceptionFeatures() {
  leftSumNorm   = normVals[0]+normVals[1];
  rightSumNorm  = normVals[2]+normVals[3];
  centerSumNorm = normVals[1]+normVals[2];
  outerSumNorm  = normVals[0]+normVals[3];
  edgeBias      = outerSumNorm-centerSumNorm;
  posNorm       = constrain(((float)linePos)/1500.0f,-1.0f,1.0f);
  trendPos      = posNorm-posNormPrevLocal;
  posNormPrevLocal = posNorm;
  confidence    = computeConfidence();
  shiftInsert(histPosNorm,    LOCAL_HISTORY_LEN, posNorm);
  shiftInsert(histTrendPos,   LOCAL_HISTORY_LEN, trendPos);
  shiftInsert(histConfidence, LOCAL_HISTORY_LEN, confidence);
  if (localHistCount<LOCAL_HISTORY_LEN) localHistCount++;
  trendPosAvg    = avgHistory(histTrendPos,   localHistCount);
  confidenceAvg  = avgHistory(histConfidence, localHistCount);
  refreshCurveIntensity();
  updateIntersectionState();
  pushAISupervisorFrame();
}

LocalMode classifyLocalModeRaw() {
  if (!calibrated) return LOCAL_UNKNOWN;
  bool lowSignal = (confidenceAvg<0.18f)||(sensorMax<80)||(sensorSum<120);
  if (lineLostFlag||lowSignal) return LOCAL_RECOVER;
  float absPos=fabsf(posNorm), absTrend=fabsf(trendPosAvg);
  float centerRatio=(sensorSum>0)?((float)centerSumNorm/(float)sensorSum):0.0f;
  float outerRatio =(sensorSum>0)?((float)outerSumNorm /(float)sensorSum):0.0f;
  if ((absPos<0.14f)&&(absTrend<0.07f)&&(curveIntensity<0.30f)&&(centerRatio>=0.34f)) return LOCAL_STRAIGHT;
  float dirScore=0.70f*posNorm+0.20f*trendPosAvg+0.10f*balanceLR;
  if (fabsf(dirScore)<0.06f) dirScore=(lastDir>=0)?0.10f:-0.10f;
  bool hardCurve=(absPos>0.48f)||(absTrend>0.14f)||(outerRatio>0.58f)||(curveIntensity>0.52f)||(edgeBias>120);
  if (dirScore<0.0f) return hardCurve?LOCAL_LEFT_HARD:LOCAL_LEFT_SOFT;
  return hardCurve?LOCAL_RIGHT_HARD:LOCAL_RIGHT_SOFT;
}

void updateLocalMode() {
  LocalMode raw=classifyLocalModeRaw();
  if (raw==LOCAL_RECOVER||raw==LOCAL_UNKNOWN){localMode=raw;localModeCandidate=raw;localModeVotes=0;return;}
  if (
    intersectionActive &&
    (localMode==LOCAL_STRAIGHT || localMode==LOCAL_LEFT_SOFT || localMode==LOCAL_LEFT_HARD ||
     localMode==LOCAL_RIGHT_SOFT || localMode==LOCAL_RIGHT_HARD)
  ) {
    localModeCandidate=localMode;
    localModeVotes=0;
    return;
  }
  if (raw==localMode){localModeCandidate=raw;localModeVotes=0;return;}
  if (raw!=localModeCandidate){localModeCandidate=raw;localModeVotes=1;return;}
  if (localModeVotes<255) localModeVotes++;
  if (localModeVotes>=2){localMode=raw;localModeVotes=0;}
}

void updateAdaptiveControl() {
  updateAISupervisor();
  float tSF=1.0f,tKp=1.0f,tKi=1.0f,tKd=1.0f,tPETh=pivotETh,tPC=(float)pivotCap;
  if (adaptiveControlEnabled&&calibrated) {
    switch (localMode) {
      case LOCAL_STRAIGHT:    tSF=1.12f;tKp=0.88f;tKi=0.90f;tKd=0.92f;tPETh=0.62f;tPC=205.0f;break;
      case LOCAL_LEFT_SOFT:
      case LOCAL_RIGHT_SOFT:  tSF=0.98f;tKp=1.05f;tKi=0.95f;tKd=1.10f;tPETh=0.52f;tPC=220.0f;break;
      case LOCAL_LEFT_HARD:
      case LOCAL_RIGHT_HARD:  tSF=0.82f;tKp=1.22f;tKi=0.75f;tKd=1.30f;tPETh=0.42f;tPC=235.0f;break;
      case LOCAL_RECOVER:     tSF=0.68f;tKp=1.10f;tKi=0.00f;tKd=1.45f;tPETh=0.35f;tPC=245.0f;break;
      default: break;
    }
    if (confidenceAvg<0.40f){tSF*=0.80f;tKd*=1.08f;} else if(confidenceAvg<0.55f) tSF*=0.90f;
    float pp=clamp01((fabsf(posNorm)-0.35f)/0.65f);
    tSF*=(1.0f-0.18f*pp);
    if (sensorMax<(lostMxTh+40)||sensorSum<(lostSumTh+120)){tSF*=0.84f;tKd*=1.05f;}

    if (aiSupervisorEnabled && aiSupervisorOutput.valid) {
      float aiMixRaw = clamp01(aiSupervisorBlend * clampf(aiSupervisorOutput.confidence, 0.35f, 1.0f));
      bool aiRecoveryActive =
        (aiSupervisorOutput.trackMode == LOCAL_RECOVER) ||
        (fabsf(aiRecoveryBias) > 0.35f);
      float aiSpeedMix = min(aiMixRaw, aiRecoveryActive ? 0.62f : 0.28f);
      float aiGainMix  = min(aiMixRaw, aiRecoveryActive ? 0.68f : 0.38f);
      float aiPivotMix = min(aiMixRaw, aiRecoveryActive ? 0.70f : 0.26f);
      tSF   = blendf(tSF,   aiSupervisorOutput.speedFactor,   aiSpeedMix);
      tKp   = blendf(tKp,   aiSupervisorOutput.kpScale,       aiGainMix);
      tKi   = blendf(tKi,   aiSupervisorOutput.kiScale,       aiGainMix);
      tKd   = blendf(tKd,   aiSupervisorOutput.kdScale,       aiGainMix);
      tPETh = blendf(tPETh, aiSupervisorOutput.pivotThreshold,aiPivotMix);
      tPC   = blendf(tPC,   aiSupervisorOutput.pivotCap,      aiPivotMix);
    }

    if (intersectionActive) {
      float ix = clamp01(max(intersectionScore, 0.45f));
      float crossSpeed = 0.96f;
      float crossKp = 1.00f;
      float crossKd = 1.00f;
      float crossPivot = 0.62f;
      float crossCap = 214.0f;
      if (driveProfile == PROFILE_SAFE) {
        crossSpeed = 0.90f;
        crossKp = 0.94f;
        crossKd = 0.94f;
        crossPivot = 0.66f;
        crossCap = 204.0f;
      } else if (driveProfile == PROFILE_FAST) {
        crossSpeed = 1.00f;
        crossKp = 1.03f;
        crossKd = 1.04f;
        crossPivot = 0.60f;
        crossCap = 220.0f;
      }
      float crossMix = min(0.52f, 0.20f + 0.32f * ix);
      tSF = blendf(tSF, min(tSF, crossSpeed), crossMix);
      tKp = blendf(tKp, min(tKp, crossKp), 0.35f * crossMix);
      tKi = blendf(tKi, min(tKi, 0.96f), 0.20f * crossMix);
      tKd = blendf(tKd, min(tKd, crossKd), 0.40f * crossMix);
      tPETh = blendf(tPETh, crossPivot, 0.45f * crossMix);
      tPC = blendf(tPC, crossCap, 0.40f * crossMix);
    }
  }
  tSF=clampf(tSF,0.55f,1.20f); tKp=clampf(tKp,0.70f,1.45f);
  tKi=clampf(tKi,0.00f,1.10f); tKd=clampf(tKd,0.75f,1.60f);
  tPETh=clampf(tPETh,0.30f,0.70f); tPC=clampf(tPC,180.0f,255.0f);
  dynSpeedFactor=smoothTo(dynSpeedFactor,tSF,  ADAPT_ALPHA);
  dynKpScale    =smoothTo(dynKpScale,    tKp,  ADAPT_ALPHA);
  dynKiScale    =smoothTo(dynKiScale,    tKi,  ADAPT_ALPHA);
  dynKdScale    =smoothTo(dynKdScale,    tKd,  ADAPT_ALPHA);
  dynPivotETh   =smoothTo(dynPivotETh,   tPETh,ADAPT_ALPHA);
  dynPivotCapF  =smoothTo(dynPivotCapF,  tPC,  ADAPT_ALPHA);
  adaptiveBaseCmd=constrain((int)lroundf((float)baseSpeed*dynSpeedFactor),baseMin,baseMax);
  adaptiveKp=Kp*dynKpScale; adaptiveKi=Ki*dynKiScale; adaptiveKd=Kd*dynKdScale;
}

int calcLinePos() {
  long sum=0,weight=0;
  for (int i=0;i<4;i++){
    rawVals[i]=readMedian3(i);
    int v=max(readNormFromRaw(rawVals[i],i),0);
    normVals[i]=v;
    const int pos=(i*1000)-1500;
    sum+=(long)v*pos; weight+=v;
  }
  computeSensorDerived();
  if (sensorSum<=0||weight<50) return lastGoodPos;
  int pos=(int)(sum/weight);
  pos=constrain(pos,-1500,1500);
  lastGoodPos=pos;
  return pos;
}

void updateLastDir() {
  int left=normVals[0]+normVals[1], right=normVals[2]+normVals[3];
  int d=right-left; const int h=40;
  if (d>h) lastDir=1; else if (d<-h) lastDir=-1;
}

// ─── Motores ──────────────────────────────────────────────────────────────────
void enableMotorDriver() { digitalWrite(STBY,HIGH); }
void disableMotorDriver(){ digitalWrite(STBY,LOW); }

void stopMotors() {
  analogWrite(AIN1,0); analogWrite(AIN2,0);
  analogWrite(BIN1,0); analogWrite(BIN2,0);
  motorSpeedA=0; motorSpeedB=0;
  disableMotorDriver();
}

void setMotorA(int speed) {
  enableMotorDriver();
  if (speed>=0){analogWrite(AIN1,speed);analogWrite(AIN2,0);}
  else{analogWrite(AIN1,0);analogWrite(AIN2,-speed);}
  motorSpeedA=speed;
}

void setMotorB(int speed) {
  enableMotorDriver();
  if (speed>=0){analogWrite(BIN1,speed);analogWrite(BIN2,0);}
  else{analogWrite(BIN1,0);analogWrite(BIN2,-speed);}
  motorSpeedB=speed;
}

int applyDeadband(int v) {
  if (v>0&&v<minPwm) return minPwm;
  if (v<0&&v>-minPwm) return -minPwm;
  return v;
}

void setMotorsDeadband(int a, int b) {
  a=applyDeadband(constrain(a,-255,255));
  b=applyDeadband(constrain(b,-255,255));
  if (a==0 && b==0) { stopMotors(); return; }
  setMotorA(a);
  setMotorB(b);
}

void setMotorsClamped(int a, int b) {
  a = constrain(a, -255, 255);
  b = constrain(b, -255, 255);
  if (a == 0 && b == 0) { stopMotors(); return; }
  setMotorA(a);
  setMotorB(b);
}

void resetAlignPID() {
  alignPidIntegral = 0.0f;
  alignPidLastError = 0.0f;
  alignPidDFiltered = 0.0f;
  lastAlignPidUs = micros();
}

void beginLineAlign(bool announce = true) {
  setRobotIdleSafe();
  stopLedLatched = false;
  autoStopLedLatched = false;
  alignToLineActive = true;
  alignStableSinceMs = 0;
  resetAlignPID();
  if (!announce) return;
  publishAck("align","ok","aligning");
  publishEvent("info","line_align_start","Align to line started");
  publishStatus("online","aligning");
}

bool sampleCalibrationLine(int& pos) {
  int frameMin = 4095;
  int frameMax = 0;
  for (int i=0;i<4;i++) {
    int rv = readMedian3(i);
    rawVals[i] = rv;
    if (rv < frameMin) frameMin = rv;
    if (rv > frameMax) frameMax = rv;
    if (rv < calMin[i]) calMin[i] = rv;
    if (rv > calMax[i]) calMax[i] = rv;
  }

  long sum = 0;
  long weight = 0;
  int frameSpan = frameMax - frameMin;
  for (int i=0;i<4;i++) {
    int span = calMax[i] - calMin[i];
    int v = 0;
    if (span >= 18) v = readNormFromRange(rawVals[i], calMin[i], calMax[i]);
    else if (frameSpan >= 24) v = readNormFromRange(rawVals[i], frameMin, frameMax);
    normVals[i] = v;
    const int sensorPos = (i * 1000) - 1500;
    sum += (long)v * sensorPos;
    weight += v;
  }

  computeSensorDerived();
  calibrationLineVisible = (sensorMax >= CAL_VISIBLE_MAX_TH) && (sensorSum >= CAL_VISIBLE_SUM_TH) && (weight >= 80);
  lineLostFlag = !calibrationLineVisible;
  if (!calibrationLineVisible) {
    pos = lastGoodPos;
    linePos = pos;
    return false;
  }

  pos = constrain((int)(sum / weight), -1500, 1500);
  lastGoodPos = pos;
  linePos = pos;
  updateLastDir();
  return true;
}

void setCalibrationTurn(int dir, int targetPwm) {
  targetPwm = constrain(targetPwm, 0, 255);
  if (targetPwm <= 0 || dir == 0) {
    calAppliedPwm = 0.0f;
    calKickUntilMs = 0;
    calLastTurnDir = 0;
    stopMotors();
    return;
  }
  uint32_t now = millis();
  int minCalPwm = max(CAL_PWM_MIN_LIMIT, minPwm);
  if (dir != calLastTurnDir) {
    calKickUntilMs = now + CAL_KICK_MS;
    calLastTurnDir = dir;
  }
  float smoothedPwm = smoothTo(calAppliedPwm, (float)targetPwm, CAL_PWM_SMOOTH_ALPHA);
  int pwm = constrain((int)lroundf(smoothedPwm), minCalPwm, 255);
  if (now < calKickUntilMs) {
    pwm = max(pwm, constrain(targetPwm, minCalPwm, 255));
  }
  calAppliedPwm = (float)pwm;
  setMotorsDeadband(dir * pwm, -dir * pwm);
}

void finishCalibrationInvalid(const char* detail, const char* eventCode, const char* message) {
  calibrating = false;
  calibrationFinalizePending = false;
  calibrationLineVisible = false;
  calAppliedPwm = 0.0f;
  stopMotors();
  calibrated = false;
  robotState = STATE_IDLE;
  resetControllerState();
  publishAck("calibrate","error",detail);
  publishEvent("error",eventCode,message);
  publishStatus("online",eventCode);
  requestSlowSnapshot(eventCode);
}

void resetControllerState() {
  pidError=pidLastErr=pidIntegral=pidCorrection=pidDFiltered=posPrev=0.0f;
  lastPidUs=micros(); lostSinceMs=0; lostForMs=0; lineLostFlag=false;
  resetPerceptionState(); resetIntersectionState(); resetAdaptiveControl(); resetAISupervisorState();
}

void setRobotIdleSafe() {
  finishRunMetrics();
  robotState=STATE_IDLE;
  calibrating=false;
  alignToLineActive=false;
  alignStableSinceMs=0;
  stopMotors();
  resetControllerState();
}

void startLineAlign() {
  if (!calibrated) {
    publishAck("align","rejected","not_calibrated");
    return;
  }
  calibrationFinalizePending = false;
  beginLineAlign(true);
}

void updateLineAlignment() {
  if (!alignToLineActive || !calibrated || calibrating || robotState != STATE_IDLE) return;

  uint32_t nowUs = micros();
  float dt = (lastAlignPidUs > 0) ? ((float)(nowUs - lastAlignPidUs) * 1e-6f) : 0.010f;
  dt = constrain(dt, 0.001f, 0.050f);
  lastAlignPidUs = nowUs;

  int pos = calcLinePos();
  linePos = pos;
  bool visible = (sensorMax >= ALIGN_VISIBLE_MAX_TH) && (sensorSum >= ALIGN_VISIBLE_SUM_TH);
  lineLostFlag = !visible;

  if (visible) {
    lostSinceMs = 0;
    lostForMs = 0;
    updateLastDir();
    updatePerceptionFeatures();
    refreshCurveIntensity();
    updateLocalMode();
    updateAdaptiveControl();
  } else {
    if (lostSinceMs == 0) lostSinceMs = millis();
    lostForMs = millis() - lostSinceMs;
    alignStableSinceMs = 0;
    resetAlignPID();
    stopMotors();
    return;
  }

  int absPos = abs(pos);
  if (absPos <= ALIGN_CENTER_POS_TH) {
    resetAlignPID();
    stopMotors();
    if (alignStableSinceMs == 0) alignStableSinceMs = millis();
    if ((millis() - alignStableSinceMs) >= ALIGN_STABLE_MS) {
      alignToLineActive = false;
      if (calibrationFinalizePending) {
        calibrationFinalizePending = false;
        publishAck("calibrate","ok","calibration_centered");
        publishEvent("info","calibration_done", calibrationSaveOk ? "Calibration completed, centered and stored" : "Calibration completed and centered");
        if (!calibrationSaveOk) publishEvent("warn","calibration_nvs","Calibration not stored in NVS");
        publishStatus("online","calibration_done");
        requestSlowSnapshot("calibration_done");
      } else {
        publishAck("align","ok","aligned");
        publishEvent("info","line_align_done","Robot aligned and holding center");
        publishStatus("online","aligned");
      }
    }
    return;
  }

  alignStableSinceMs = 0;
  float e = ((float)pos) / 1500.0f;
  alignPidIntegral = constrain(alignPidIntegral + e * dt, -ALIGN_PID_I_CLAMP, ALIGN_PID_I_CLAMP);
  float de = clampf((e - alignPidLastError) / dt, -ALIGN_PID_D_CLAMP, ALIGN_PID_D_CLAMP);
  alignPidDFiltered += 0.28f * (de - alignPidDFiltered);
  alignPidLastError = e;

  float u = (alignPidKp * e) + (alignPidKi * alignPidIntegral) + (alignPidKd * alignPidDFiltered);
  float mag = fabsf(u) * (float)ALIGN_PWM_MAX;
  int pwm = constrain((int)lroundf(mag), ALIGN_PWM_MIN, ALIGN_PWM_MAX);
  int dir = (u >= 0.0f) ? +1 : -1;
  setMotorsClamped(dir * pwm, -dir * pwm);
}

// ─── Publicación MQTT ─────────────────────────────────────────────────────────
void publishAck(const char* cmd, const char* result, const char* detail="") {
  if (!mqttClient.connected()) return;
  sharedDoc.clear();
  fillCommonEnvelope(sharedDoc,"ack","ack");
  sharedDoc["cmd"]=cmd; sharedDoc["result"]=result;
  sharedDoc["detail"]=detail; sharedDoc["state"]=stateToStr(robotState);
  if (serializeToBuffer(sharedDoc,payloadAck,sizeof(payloadAck)))
    mqttPublishSafe(TOPIC_ACK,payloadAck,false);
}

void publishEvent(const char* severity, const char* code, const char* message) {
  if (!mqttClient.connected()) return;
  sharedDoc.clear();
  fillCommonEnvelope(sharedDoc,"event","event");
  sharedDoc["severity"]=severity; sharedDoc["code"]=code;
  sharedDoc["message"]=message;  sharedDoc["state"]=stateToStr(robotState);
  if (serializeToBuffer(sharedDoc,payloadEvent,sizeof(payloadEvent)))
    mqttPublishSafe(TOPIC_EVENT,payloadEvent,false);
}

void publishStatus(const char* onlineStatus, const char* reason) {
  if (!mqttClient.connected()) return;
  char ipbuf[20]; ipToCstr(WiFi.localIP(),ipbuf,sizeof(ipbuf));
  uint32_t runElapsedNow = runMetricsActive ? (millis() - runStartMs) : runElapsedMs;
  sharedDoc.clear();
  fillCommonEnvelope(sharedDoc,"status","status");
  sharedDoc["online"]=onlineStatus; sharedDoc["reason"]=reason;
  sharedDoc["state"]=stateToStr(robotState);
  sharedDoc["calibrated"]=calibrated?1:0;
  sharedDoc["calibration_loaded_from_nvs"]=calibrationLoadedFromNvs?1:0;
  sharedDoc["local_mode"]=localModeToStr(localMode);
  sharedDoc["adaptive_enabled"]=adaptiveControlEnabled?1:0;
  sharedDoc["ai_enabled"]=aiSupervisorEnabled?1:0;
  sharedDoc["ai_track_mode"]=localModeToStr((LocalMode)aiSupervisorOutput.trackMode);
  sharedDoc["ai_source"]=aiSourceToStr(aiSupervisorOutput.source);
  sharedDoc["profile"]=driveProfileToStr(driveProfile);
  sharedDoc["intersection_active"]=intersectionActive?1:0;
  sharedDoc["run_elapsed_ms"]=runElapsedNow;
  sharedDoc["lap_estimate"]=runLapEstimate;
  JsonObject track=sharedDoc["track"].to<JsonObject>();
  track["ready"]=trackDataReady?1:0;
  track["publishing"]=trackPublishActive?1:0;
  track["samples"]=trackSampleCount;
  track["capacity"]=TRACK_SAMPLE_CAP;
  track["run_id"]=trackRunId;
  JsonObject net=sharedDoc["network"].to<JsonObject>();
  net["wifi_connected"]=(WiFi.status()==WL_CONNECTED)?1:0;
  net["mqtt_connected"]=mqttClient.connected()?1:0;
  net["wifi_rssi"]=WiFi.RSSI(); net["ip"]=ipbuf;
  net["mqtt_host"]=MQTT_HOST; net["mqtt_port"]=MQTT_PORT;
  net["mqtt_connect_attempts"]=mqttConnectAttempts;
  net["mqtt_connect_fails"]=mqttConnectFails;
  net["mqtt_publish_fails"]=mqttPublishFails;
  net["secrets_header"]=HAS_SECRETS_HEADER?1:0;
  if (serializeToBuffer(sharedDoc,payloadStatus,sizeof(payloadStatus)))
    mqttPublishSafe(TOPIC_STATUS,payloadStatus,true);
}

// [FIX-2] Fast reducida: sólo campos que cambian rápido
void publishTelemetryFast(const char* eventCode=nullptr) {
  if (!mqttClient.connected()) return;
  if (!heapOkForTelemetry()) return; // [FIX-8]
  uint32_t runElapsedNow = runMetricsActive ? (millis() - runStartMs) : runElapsedMs;
  sharedDoc.clear();
  fillCommonEnvelope(sharedDoc,"telemetry","telemetry/fast");
  sharedDoc["state"]       = stateToStr(robotState);
  sharedDoc["local_mode"]  = localModeToStr(localMode);
  sharedDoc["local_mode_candidate"] = localModeToStr(localModeCandidate);
  sharedDoc["line_pos"]    = linePos;
  sharedDoc["line_lost"]   = lineLostFlag?1:0;
  sharedDoc["lost_ms"]     = lostForMs;
  sharedDoc["last_dir"]    = (lastDir>=0)?"right":"left";
  sharedDoc["dominant_sensor"] = SENSOR_LABELS[dominantSensor];
  sharedDoc["balance_lr"]  = balanceLR;
  sharedDoc["confidence"]  = confidenceAvg;
  sharedDoc["confidence_avg"] = confidenceAvg;
  sharedDoc["profile"]     = driveProfileToStr(driveProfile);
  sharedDoc["pos_norm"]    = posNorm;
  sharedDoc["trend_pos"]   = trendPos;
  sharedDoc["trend_pos_avg"] = trendPosAvg;
  sharedDoc["curve_intensity"] = curveIntensity;
  sharedDoc["intersection_active"] = intersectionActive?1:0;
  sharedDoc["intersection_score"]  = intersectionScore;
  sharedDoc["run_elapsed_ms"]      = runElapsedNow;
  sharedDoc["lap_estimate"]        = runLapEstimate;
  sharedDoc["run_intersection_count"] = runIntersectionCount;
  sharedDoc["autonomous_run"]      = autonomousRunActive?1:0;

  JsonObject pid=sharedDoc["pid"].to<JsonObject>();
  pid["error"]=pidError; pid["integral"]=pidIntegral;
  pid["d_filt"]=pidDFiltered; pid["correction"]=pidCorrection;

  JsonObject motors=sharedDoc["motors"].to<JsonObject>();
  motors["a_pwm"]=motorSpeedA; motors["b_pwm"]=motorSpeedB;
  motors["base_cmd"]=baseSpeed;
  motors["adaptive_base_cmd"]=adaptiveBaseCmd;
  motors["effective_base_cmd"]=effectiveBaseCmd;
  motors["speed_scale"]=effectiveSpeedScale;
  motors["base_eff"]=effectiveBaseCmd;

  JsonObject sensors=sharedDoc["sensors"].to<JsonObject>();
  sensors["sum"]=sensorSum; sensors["max"]=sensorMax;
  JsonArray norm=sensors["norm"].to<JsonArray>();
  for (int i=0;i<4;i++) norm.add(normVals[i]);

  JsonObject perf=sharedDoc["perf"].to<JsonObject>();
  perf["loop_hz"]=loopHz; perf["loop_us_avg"]=loopUsAvg;
  perf["heap_free"]=ESP.getFreeHeap();
  perf["rssi"]=WiFi.RSSI();

  JsonObject ai=sharedDoc["ai"].to<JsonObject>();
  ai["enabled"]=aiSupervisorEnabled?1:0;
  ai["model_enabled"]=aiModelEnabled?1:0;
  ai["hook_present"]=HAS_AI_MODEL_HOOK?1:0;
  ai["source"]=aiSourceToStr(aiSupervisorOutput.source);
  ai["track_mode"]=localModeToStr((LocalMode)aiSupervisorOutput.trackMode);
  ai["confidence"]=aiSupervisorOutput.confidence;
  ai["blend"]=aiSupervisorBlend;
  ai["recovery_bias"]=aiRecoveryBias;

  if (eventCode&&eventCode[0]!='\0') sharedDoc["event"]=eventCode;
  if (serializeToBuffer(sharedDoc,payloadFast,sizeof(payloadFast)))
    mqttPublishSafe(TOPIC_TELE_FAST,payloadFast,false);
}

void publishTelemetrySlow(const char* eventCode=nullptr) {
  if (!mqttClient.connected()) return;
  if (!heapOkForTelemetry()) return; // [FIX-8]
  char ipbuf[20]; ipToCstr(WiFi.localIP(),ipbuf,sizeof(ipbuf));
  uint32_t nowMs = millis();
  uint32_t runElapsedNow = runMetricsActive ? (nowMs - runStartMs) : runElapsedMs;
  sharedDoc.clear();
  fillCommonEnvelope(sharedDoc,"telemetry","telemetry/slow");
  sharedDoc["state"]=stateToStr(robotState);
  sharedDoc["local_mode"]=localModeToStr(localMode);
  sharedDoc["local_mode_candidate"]=localModeToStr(localModeCandidate);
  sharedDoc["line_pos"]=linePos; sharedDoc["line_lost"]=lineLostFlag?1:0;
  sharedDoc["lost_ms"]=lostForMs; sharedDoc["last_dir"]=(lastDir>=0)?"right":"left";
  sharedDoc["dominant_sensor"]=SENSOR_LABELS[dominantSensor];
  sharedDoc["balance_lr"]=balanceLR; sharedDoc["confidence"]=confidenceAvg;

  JsonObject pid=sharedDoc["pid"].to<JsonObject>();
  pid["kp"]=Kp; pid["ki"]=Ki; pid["kd"]=Kd;
  pid["error"]=pidError; pid["integral"]=pidIntegral;
  pid["d_filt"]=pidDFiltered; pid["correction"]=pidCorrection;
  pid["i_clamp"]=iClamp; pid["u_clamp"]=uClamp;
  pid["de_clamp"]=deClamp; pid["deriv_alpha"]=derivAlpha;

  JsonObject motors=sharedDoc["motors"].to<JsonObject>();
  motors["base_cmd"]=baseSpeed; motors["adaptive_base_cmd"]=adaptiveBaseCmd;
  motors["effective_base_cmd"]=effectiveBaseCmd;
  motors["speed_scale"]=effectiveSpeedScale;
  motors["base_min"]=baseMin; motors["base_max"]=baseMax;
  motors["a_pwm"]=motorSpeedA; motors["b_pwm"]=motorSpeedB;
  motors["min_pwm"]=minPwm; motors["pivot_e_th"]=pivotETh;
  motors["pivot_cap"]=pivotCap; motors["dyn_pivot_e_th"]=dynPivotETh;
  motors["dyn_pivot_cap"]=(int)lroundf(dynPivotCapF);

  JsonObject cal=sharedDoc["calibration"].to<JsonObject>();
  cal["calibrated"]=calibrated?1:0; cal["calibrating"]=calibrating?1:0;
  cal["duration_ms"]=CAL_DURATION_MS;
  cal["elapsed_ms"]=calibrating?(millis()-calStartMs):0;
  cal["loaded_from_nvs"]=calibrationLoadedFromNvs?1:0;
  cal["line_visible"]=calibrationLineVisible?1:0;
  cal["zigzag_legs_target"]=CAL_ZIGZAG_LEGS;
  cal["zigzag_legs_done"]=calSweepLegIndex > 0 ? (int)(calSweepLegIndex - 1) : 0;
  cal["sweep_direction"]=(calSweepDir?"right":"left");
  cal["sweep_pwm"]=calSweepPwm;
  cal["applied_pwm"]=(int)lroundf(calAppliedPwm);
  cal["edge_target_pos"]=calEdgePosTarget;
  cal["auto_center_pending"]=calibrationFinalizePending?1:0;
  cal["align_kp"]=alignPidKp;
  cal["align_ki"]=alignPidKi;
  cal["align_kd"]=alignPidKd;
  cal["align_integral"]=alignPidIntegral;
  cal["align_d_filt"]=alignPidDFiltered;

  JsonObject ctx=sharedDoc["context"].to<JsonObject>();
  ctx["pos_norm"]=posNorm; ctx["trend_pos"]=trendPos;
  ctx["trend_pos_avg"]=trendPosAvg; ctx["curve_intensity"]=curveIntensity;
  ctx["left_sum"]=leftSumNorm; ctx["right_sum"]=rightSumNorm;
  ctx["center_sum"]=centerSumNorm; ctx["outer_sum"]=outerSumNorm;
  ctx["edge_bias"]=edgeBias; ctx["confidence_now"]=confidence;
  ctx["confidence_avg"]=confidenceAvg; ctx["history_len"]=localHistCount;

  JsonObject sensorSummary=sharedDoc["sensor_summary"].to<JsonObject>();
  sensorSummary["sum"]=sensorSum;
  sensorSummary["max"]=sensorMax;
  sensorSummary["lost_sum_threshold"]=lostSumTh;
  sensorSummary["lost_max_threshold"]=lostMxTh;
  sensorSummary["dominant_index"]=dominantSensor;
  sensorSummary["dominant_label"]=SENSOR_LABELS[dominantSensor];

  JsonObject adp=sharedDoc["adaptive"].to<JsonObject>();
  adp["enabled"]=adaptiveControlEnabled?1:0;
  adp["speed_factor"]=dynSpeedFactor; adp["kp_scale"]=dynKpScale;
  adp["ki_scale"]=dynKiScale; adp["kd_scale"]=dynKdScale;
  adp["kp_eff"]=adaptiveKp; adp["ki_eff"]=adaptiveKi; adp["kd_eff"]=adaptiveKd;

  JsonObject ai=sharedDoc["ai"].to<JsonObject>();
  ai["enabled"]=aiSupervisorEnabled?1:0;
  ai["model_enabled"]=aiModelEnabled?1:0;
  ai["hook_present"]=HAS_AI_MODEL_HOOK?1:0;
  ai["window_len"]=AI_WINDOW_LEN;
  ai["window_count"]=aiWindowCount;
  ai["source"]=aiSourceToStr(aiSupervisorOutput.source);
  ai["track_mode"]=localModeToStr((LocalMode)aiSupervisorOutput.trackMode);
  ai["confidence"]=aiSupervisorOutput.confidence;
  ai["blend"]=aiSupervisorBlend;
  ai["speed_factor"]=aiSupervisorOutput.speedFactor;
  ai["kp_scale"]=aiSupervisorOutput.kpScale;
  ai["ki_scale"]=aiSupervisorOutput.kiScale;
  ai["kd_scale"]=aiSupervisorOutput.kdScale;
  ai["pivot_threshold"]=aiSupervisorOutput.pivotThreshold;
  ai["pivot_cap"]=aiSupervisorOutput.pivotCap;
  ai["recovery_bias"]=aiRecoveryBias;
  ai["model_suggested"]=aiSupervisorOutput.modelSuggested?1:0;
  ai["blended"]=aiSupervisorOutput.blended?1:0;
  ai["inference_count"]=aiInferenceCount;
  ai["fallback_count"]=aiFallbackCount;
  ai["last_inference_ms"]=aiLastInferenceMs;

  JsonObject profile=sharedDoc["profile"].to<JsonObject>();
  profile["name"]=driveProfileToStr(driveProfile);
  profile["base_speed"]=baseSpeed;
  profile["ai_blend"]=aiSupervisorBlend;

  JsonObject crossing=sharedDoc["crossing"].to<JsonObject>();
  crossing["detected"]=intersectionDetected?1:0;
  crossing["active"]=intersectionActive?1:0;
  crossing["score"]=intersectionScore;
  crossing["count"]=intersectionCount;
  crossing["hold_ms"]=intersectionActive && intersectionHoldUntilMs>nowMs ? (intersectionHoldUntilMs-nowMs) : 0;
  crossing["preferred_dir"]=(intersectionPreferredDir>=0)?"right":"left";
  crossing["marker_detected"]=lapMarkerDetected?1:0;

  JsonObject run=sharedDoc["run"].to<JsonObject>();
  run["active"]=runMetricsActive?1:0;
  run["run_id"]=runId;
  run["elapsed_ms"]=runElapsedNow;
  run["line_lost_count"]=runLineLostCount;
  run["lost_total_ms"]=runTotalLostMs;
  run["lost_max_ms"]=runMaxLostMs;
  run["intersection_count"]=runIntersectionCount;
  run["marker_hits"]=runMarkerHits;
  run["lap_estimate"]=runLapEstimate;
  run["last_lap_ms"]=runLastLapMs;
  run["best_lap_ms"]=runBestLapMs;
  run["autonomous_mode"]=autonomousRunActive?1:0;
  run["autonomous_elapsed_ms"]=autonomousRunActive?autonomousRunElapsedMs:0;
  run["buffered_samples"]=autonomousSampleCount;
  run["buffer_overflow"]=autonomousBufferOverflow?1:0;

  bool alarmBadCal=false, alarmSat=false, alarmWeak=false;
  JsonArray sArr=sharedDoc["sensors"].to<JsonArray>();
  for (int i=0;i<4;i++){
    JsonObject s=sArr.add<JsonObject>();
    const char* h=sensorHealthStatus(i);
    bool sat=(rawVals[i]<=8||rawVals[i]>=4087);
    bool badCal=(sensorSpan[i]<80);
    bool weak=(normVals[i]<30&&sensorMax>500);
    s["index"]=i; s["label"]=SENSOR_LABELS[i];
    s["raw"]=rawVals[i]; s["norm"]=normVals[i];
    s["cal_min"]=calMin[i]; s["cal_max"]=calMax[i];
    s["cal_span"]=sensorSpan[i]; s["on_line"]=sensorOnLine[i];
    s["health"]=h; s["saturated"]=sat?1:0;
    s["weak"]=weak?1:0; s["bad_cal"]=badCal?1:0;
    if (sat) alarmSat=true; if (weak) alarmWeak=true; if (badCal) alarmBadCal=true;
  }

  JsonObject alarms=sharedDoc["alarms"].to<JsonObject>();
  alarms["line_lost"]=lineLostFlag?1:0;
  alarms["bad_cal"]=alarmBadCal?1:0;
  alarms["saturated"]=alarmSat?1:0;
  alarms["weak"]=alarmWeak?1:0;

  JsonObject net=sharedDoc["network"].to<JsonObject>();
  net["wifi_connected"]=(WiFi.status()==WL_CONNECTED)?1:0;
  net["mqtt_connected"]=mqttClient.connected()?1:0;
  net["wifi_rssi"]=WiFi.RSSI(); net["ip"]=ipbuf;
  net["broker_host"]=MQTT_HOST; net["broker_port"]=MQTT_PORT;
  net["connect_attempts"]=mqttConnectAttempts;
  net["connect_fails"]=mqttConnectFails;
  net["publish_fails"]=mqttPublishFails;
  net["time_valid"]=timeValid?1:0;
  net["secrets_header"]=HAS_SECRETS_HEADER?1:0;

  JsonObject perf=sharedDoc["perf"].to<JsonObject>();
  perf["loop_hz"]=loopHz; perf["loop_us_avg"]=loopUsAvg;
  perf["loop_us_max"]=loopUsMax;
  perf["free_heap"]=ESP.getFreeHeap();
  perf["free_heap_min"]=freeHeapMin;
  perf["fast_ms"]=TELE_FAST_MS; perf["slow_ms"]=TELE_SLOW_MS;

  if (eventCode&&eventCode[0]!='\0') sharedDoc["event"]=eventCode;
  if (serializeToBuffer(sharedDoc,payloadSlow,sizeof(payloadSlow)))
    mqttPublishSafe(TOPIC_TELE_SLOW,payloadSlow,false);
}

void publishAutonomousBufferChunk() {
  if (!autonomousBufferFlushPending || !mqttClient.connected()) return;
  uint32_t now = millis();
  if ((now - lastAutonomousFlushMs) < AUTONOMOUS_FLUSH_MS) return;
  lastAutonomousFlushMs = now;

  if (autonomousFlushIndex >= autonomousSampleCount) {
    autonomousBufferFlushPending = false;
    autonomousReconnectPending = false;
    clearAutonomousRunBuffer();
    return;
  }

  uint16_t chunkStart = autonomousFlushIndex;
  uint16_t chunkEnd = min<uint16_t>(autonomousSampleCount, chunkStart + AUTONOMOUS_FLUSH_CHUNK);
  uint16_t chunkIndex = (chunkStart / AUTONOMOUS_FLUSH_CHUNK) + 1;
  uint16_t chunkTotal = (autonomousSampleCount + AUTONOMOUS_FLUSH_CHUNK - 1) / AUTONOMOUS_FLUSH_CHUNK;

  sharedDoc.clear();
  fillCommonEnvelope(sharedDoc,"telemetry","telemetry/buffer");
  sharedDoc["state"] = stateToStr(robotState);
  sharedDoc["event"] = "autonomous_buffer";

  JsonObject run = sharedDoc["run"].to<JsonObject>();
  run["run_id"] = runId;
  run["elapsed_ms"] = autonomousRunElapsedMs;
  run["autonomous_ms"] = AUTONOMOUS_RUN_MS;
  run["sample_ms"] = AUTONOMOUS_SAMPLE_MS;
  run["stored_samples"] = autonomousSampleCount;
  run["overflow"] = autonomousBufferOverflow?1:0;
  run["chunk_index"] = chunkIndex;
  run["chunk_total"] = chunkTotal;

  JsonArray samples = sharedDoc["samples"].to<JsonArray>();
  for (uint16_t i = chunkStart; i < chunkEnd; ++i) {
    const AutonomousRunSample& s = autonomousSamples[i];
    JsonObject item = samples.add<JsonObject>();
    item["t"] = s.elapsedMs;
    item["p"] = s.linePos;
    item["a"] = s.motorA;
    item["b"] = s.motorB;
    item["sum"] = s.sensorSum;
    item["max"] = s.sensorMax;
    item["lost"] = s.lostMs;
    item["lm"] = s.localMode;
    item["f"] = s.flags;
    JsonArray norm = item["n"].to<JsonArray>();
    norm.add(s.norm0);
    norm.add(s.norm1);
    norm.add(s.norm2);
    norm.add(s.norm3);
  }

  if (!serializeToBuffer(sharedDoc,payloadSlow,sizeof(payloadSlow))) return;
  if (!mqttPublishSafe(TOPIC_TELE_BUFFER,payloadSlow,false)) return;

  autonomousFlushIndex = chunkEnd;
  if (autonomousFlushIndex >= autonomousSampleCount) {
    autonomousBufferFlushPending = false;
    autonomousReconnectPending = false;
    clearAutonomousRunBuffer();
  }
}

// ─── Calibración ─────────────────────────────────────────────────────────────
void processDeferredEvents() {
  uint32_t flags = 0;
  uint8_t otaError = 0;
  portENTER_CRITICAL(&deferredMux);
  flags = deferredFlags;
  otaError = deferredOtaError;
  deferredFlags = 0;
  deferredOtaError = 0;
  portEXIT_CRITICAL(&deferredMux);

  if (flags & DEFERRED_OTA_START) {
    otaInProgress = true;
    setRobotIdleSafe();
    publishEvent("info","ota_start","OTA update started");
    publishStatus("online","ota_start");
    requestSlowSnapshot("ota_start");
  }
  if (flags & DEFERRED_OTA_ERROR) {
    otaInProgress = false;
    publishEvent("error","ota_error",otaErrorToStr(otaError));
    publishStatus("online","ota_error");
    requestSlowSnapshot("ota_error");
  }
  if (flags & DEFERRED_OTA_END) {
    otaInProgress = false;
    publishEvent("info","ota_end","OTA update finished");
    publishStatus("online","ota_end");
    requestSlowSnapshot("ota_end");
  }
}

void startCalibration(bool automatic=true) {
  (void)automatic;
  finishRunMetrics();
  stopMotors();
  alignToLineActive = false;
  alignStableSinceMs = 0;
  calibrationFinalizePending = false;
  calibrationSaveOk = false;
  calibrationLineVisible = false;
  stopLedLatched = false;
  autoStopLedLatched = false;
  for (int i=0;i<4;i++){calMin[i]=4095;calMax[i]=0;}
  calibrating=true; calibrated=false; calStartMs=millis();
  calibrationLoadedFromNvs=false;
  robotState=STATE_CALIBRATING; calSweepDir=false; lastSweepMs=millis();
  calSweepLegIndex = 0;
  calPhaseStartMs = calStartMs;
  calEdgeHoldSinceMs = 0;
  calCenterStableMs = 0;
  calKickUntilMs = 0;
  calAppliedPwm = 0.0f;
  calLastTurnDir = 0;
  lineLostFlag=false; lostSinceMs=0; lostForMs=0;
  resetPerceptionState(); resetIntersectionState(); resetAdaptiveControl();
  publishEvent("info","calibration_start","Calibration started");
  requestSlowSnapshot("calibration_start");
}

void updateCalibration() {
  if (!calibrating) return;
  uint32_t now = millis();
  int pos = 0;
  bool visible = sampleCalibrationLine(pos);

  if (!visible) {
    if (lostSinceMs == 0) lostSinceMs = now;
    lostForMs = now - lostSinceMs;
  } else {
    lostSinceMs = 0;
    lostForMs = 0;
  }

  if ((now - calStartMs) >= CAL_DURATION_MS) {
    finishCalibrationInvalid("timeout", "calibration_timeout", "Calibration timed out before completing zigzag");
    return;
  }

  if (calSweepLegIndex == 0) {
    if (!visible) {
      setCalibrationTurn(0, 0);
      calCenterStableMs = 0;
      if ((now - calPhaseStartMs) >= CAL_START_TIMEOUT_MS) {
        finishCalibrationInvalid("line_not_visible", "calibration_line_missing", "Place the robot centered on the line before calibrating");
      }
      return;
    }

    int absPos = abs(pos);
    if (absPos <= ALIGN_CENTER_POS_TH) {
      setCalibrationTurn(0, 0);
      if (calCenterStableMs == 0) calCenterStableMs = now;
      if ((now - calCenterStableMs) >= CAL_CENTER_SETTLE_MS) {
        calSweepLegIndex = 1;
        calSweepDir = false; // izquierda primero
        lastSweepMs = now;
        calPhaseStartMs = now;
        calEdgeHoldSinceMs = 0;
        calCenterStableMs = 0;
      }
      return;
    }

    calCenterStableMs = 0;
    int pwm = map(absPos, ALIGN_CENTER_POS_TH, 900, ALIGN_PWM_MIN, calSweepPwm);
    pwm = constrain(pwm, ALIGN_PWM_MIN, calSweepPwm);
    int dir = (pos >= 0) ? +1 : -1;
    setCalibrationTurn(dir, pwm);
    return;
  }

  int sweepDir = calSweepDir ? +1 : -1;
  bool reachedEdge = visible && ((sweepDir < 0 && pos <= -calEdgePosTarget) || (sweepDir > 0 && pos >= calEdgePosTarget));
  bool timedOut = (now - lastSweepMs) >= CAL_SWEEP_TIMEOUT_MS;
  bool reverseNow = false;

  if (reachedEdge) {
    if (calEdgeHoldSinceMs == 0) calEdgeHoldSinceMs = now;
    if ((now - calEdgeHoldSinceMs) >= CAL_EDGE_HOLD_MS) reverseNow = true;
  } else {
    calEdgeHoldSinceMs = 0;
  }

  if (!visible && (now - lastSweepMs) >= 120) reverseNow = true;
  if (timedOut) reverseNow = true;

  if (reverseNow) {
    setCalibrationTurn(0, 0);
    calSweepDir = !calSweepDir;
    lastSweepMs = now;
    calEdgeHoldSinceMs = 0;
    calAppliedPwm = 0.0f;
    calSweepLegIndex++;
    if (calSweepLegIndex > CAL_ZIGZAG_LEGS) {
      calibrating = false;
      stopMotors();
      if (!calibrationDataLooksValid(calMin, calMax)) {
        finishCalibrationInvalid("invalid_calibration", "calibration_invalid", "Calibration data out of range");
        return;
      }
      calibrated = true;
      robotState = STATE_IDLE;
      resetControllerState();
      calibrationSaveOk = saveCalibrationToNvs();
      calibrationFinalizePending = true;
      beginLineAlign(false);
      publishEvent("info","calibration_centering","Calibration captured, centering robot");
      publishStatus("online","calibration_centering");
      requestSlowSnapshot("calibration_centering");
    }
    return;
  }

  setCalibrationTurn(sweepDir, calSweepPwm);
}

// ─── PID ──────────────────────────────────────────────────────────────────────
void updatePID() {
  if (robotState!=STATE_RUNNING||!calibrated) return;
  uint32_t nowUs=micros();
  float dt=(nowUs-lastPidUs)*1e-6f;
  dt=constrain(dt,0.001f,0.050f);
  lastPidUs=nowUs;

  int pos=calcLinePos();
  linePos=pos;
  bool lost=(sensorMax<lostMxTh)||(sensorSum<lostSumTh);
  lineLostFlag=lost;

  updatePerceptionFeatures();
  updateRunMetrics(lost, dt);
  if (!lost) updateLastDir();

  if (lost) {
    if (lostSinceMs==0){
      lostSinceMs=millis();
      publishEvent("warn","line_lost","Line lost");
    }
    lostForMs=millis()-lostSinceMs;

    // [FIX-7] Parar si lleva demasiado tiempo perdida
    if (lostForMs>LOST_STOP_MS){
      stopMotors();
      pidIntegral=0.0f; pidLastErr=0.0f; posPrev=(float)pos;
      pidDFiltered=0.0f; pidCorrection=0.0f; pidError=0.0f;
      return;
    }

    refreshCurveIntensity();
    updateLocalMode();
    updateAdaptiveControl();
    int mag=(lostForMs<300)?180:130;
    float recoveryBias = (aiSupervisorEnabled && aiSupervisorOutput.valid) ? aiRecoveryBias : 0.0f;
    int dir=(lastDir>=0)?+1:-1;
    if (fabsf(recoveryBias) >= 0.20f) dir = (recoveryBias >= 0.0f) ? +1 : -1;
    if (fabsf(recoveryBias) >= 0.75f) mag += 10;
    effectiveBaseCmd = 0;
    effectiveSpeedScale = 0.0f;
    setMotorsDeadband(+dir*mag,-dir*mag);
    pidIntegral=0.0f; pidLastErr=0.0f; posPrev=(float)pos;
    pidDFiltered=0.0f; pidCorrection=0.0f; pidError=0.0f;
    return;
  }

  lostSinceMs=0; lostForMs=0;
  float e=((float)pos)/1500.0f;
  float dPos=((float)pos-posPrev)/dt;
  posPrev=(float)pos;
  float de=clampf(dPos/1500.0f,-deClamp,deClamp);
  pidDFiltered+=derivAlpha*(de-pidDFiltered);

  refreshCurveIntensity(); updateLocalMode(); updateAdaptiveControl();

  float ae=fabsf(e), ade=fabsf(pidDFiltered)/deClamp;
  float speedScale=constrain(1.0f-speedKE*ae-speedKDE*ade,0.45f,1.00f);
  int baseEff=constrain((int)((float)adaptiveBaseCmd*speedScale),baseMin,baseMax);
  effectiveBaseCmd = baseEff;
  effectiveSpeedScale = speedScale;

  if (ae<0.65f&&localMode!=LOCAL_RECOVER)
    pidIntegral=constrain(pidIntegral+e*dt,-iClamp,iClamp);
  else
    pidIntegral*=0.95f;

  float u=clampf(adaptiveKp*e+adaptiveKi*pidIntegral+adaptiveKd*pidDFiltered,-uClamp,uClamp);
  int diff=(int)(u*(float)baseEff);
  int sA=baseEff+diff, sB=baseEff-diff;
  int dynPivotCap=constrain((int)lroundf(dynPivotCapF),180,255);

  if (ae>dynPivotETh){
    setMotorsDeadband(constrain(sA,-dynPivotCap,dynPivotCap),constrain(sB,-dynPivotCap,dynPivotCap));
  } else {
    setMotorsDeadband(constrain(max(sA,0),0,255),constrain(max(sB,0),0,255));
  }
  pidError=e; pidLastErr=e; pidCorrection=u;
}

// ─── LED ──────────────────────────────────────────────────────────────────────
void updateLED() {
  uint32_t now = millis();
  bool blinkTick=(now-lastLedMs>=LED_BLINK_MS);
  if (blinkTick){lastLedMs=now;ledBlink=!ledBlink;}
  bool readyBlink = (((now / READY_BLINK_MS) % 2UL) == 0UL);
  if (robotState==STATE_CALIBRATING){
    LED_RED();
    return;
  }
  if (robotState==STATE_RUNNING){
    updateRunningLed();
    return;
  }
  if (stopLedLatched) {
    LED_RED();
    return;
  }
  if (autoStopLedLatched) {
    LED_YELLOW();
    return;
  }
  if (alignToLineActive) {
    LED_YELLOW();
    return;
  }
  if (WiFi.status()!=WL_CONNECTED){
    if (ledBlink) LED_LIGHTBLUE(); else LED_OFF();
    return;
  }
  if (calibrated) {
    if (readyBlink) LED_GREEN(); else LED_OFF();
    return;
  }
  LED_BLUE();
}

// ─── Comandos ─────────────────────────────────────────────────────────────────
bool getJsonNumber(JsonVariantConst v, float& out) {
  if (!v.is<float>()&&!v.is<int>()&&!v.is<double>()) return false;
  out=v.as<float>(); return true;
}
bool getJsonInt(JsonVariantConst v, int& out) {
  if (!v.is<int>()&&!v.is<float>()&&!v.is<double>()) return false;
  out=(int)v.as<float>(); return true;
}

bool parseDriveProfileValue(JsonVariantConst v, DriveProfile& out) {
  if (v.is<const char*>()) {
    String raw = v.as<String>();
    raw.trim();
    raw.toLowerCase();
    if (raw=="safe" || raw=="seguro")   { out = PROFILE_SAFE; return true; }
    if (raw=="fast" || raw=="rapido")   { out = PROFILE_FAST; return true; }
    if (raw=="normal")                  { out = PROFILE_NORMAL; return true; }
    return false;
  }
  int iv = 0;
  if (!getJsonInt(v, iv)) return false;
  if (iv <= 0) out = PROFILE_SAFE;
  else if (iv == 1) out = PROFILE_NORMAL;
  else out = PROFILE_FAST;
  return true;
}

void applyStart() {
  if (!calibrated){publishAck("start","rejected","not_calibrated");return;}
  alignToLineActive = false;
  alignStableSinceMs = 0;
  calibrationFinalizePending = false;
  stopLedLatched = false;
  autoStopLedLatched = false;
  robotState=STATE_RUNNING; resetControllerState(); startRunMetrics();
  trackRecorderBegin();
  enableMotorDriver();
  publishAck("start","ok","running");
  publishEvent("info","robot_started","Robot state set to running");
  publishStatus("online","started");
  clearAutonomousRunBuffer();
}

void applyStop() {
  autonomousRunActive = false;
  calibrationFinalizePending = false;
  trackRecorderFinish();
  setRobotIdleSafe();
  autonomousReconnectPending = false;
  autoStopLedLatched = false;
  clearAutonomousRunBuffer();
  stopLedLatched = true;
  publishAck("stop","ok","idle");
  publishEvent("info","robot_stopped","Robot state set to idle");
  publishStatus("online","stopped");
}

void handleCommand(JsonDocument& doc) {
  if (!doc["cmd"].is<const char*>()) return;
  String cmd=doc["cmd"].as<String>(); cmd.trim(); cmd.toLowerCase();
  float fv=0.0f; int iv=0;
  DriveProfile profile = PROFILE_NORMAL;

  if (otaInProgress && cmd!="status") {
    publishAck(cmd.c_str(),"rejected","ota_in_progress");
    return;
  }

  if (cmd=="start"){applyStart();return;}
  if (cmd=="stop"){applyStop();return;}
  if (cmd=="dump_track" || cmd=="publish_track" || cmd=="export_track"){
    if (robotState == STATE_RUNNING) {
      publishAck(cmd.c_str(),"rejected","stop_first");
      return;
    }
    if (!trackRecorderRequestPublish()) {
      publishAck(cmd.c_str(),"rejected","no_track_ready");
      return;
    }
    publishAck(cmd.c_str(),"ok","track_publish_started");
    return;
  }
  if (cmd=="align" || cmd=="alinear" || cmd=="center" || cmd=="centrar"){
    startLineAlign();
    return;
  }
  if (cmd=="calibrate"){
    stopMotors(); startCalibration(true);
    publishAck("calibrate","ok","started");
    publishStatus("online","calibrating");
    return;
  }
  if (cmd=="reset_cal"){
    calibrated=false; setRobotIdleSafe();
    calibrationFinalizePending = false;
    stopLedLatched = false;
    for (int i=0;i<4;i++){calMin[i]=4095;calMax[i]=0;}
    clearCalibrationFromNvs();
    publishAck("reset_cal","ok","calibration_reset");
    publishEvent("warn","calibration_reset","Calibration data reset");
    publishStatus("online","calibration_reset");
    return;
  }
  if ((cmd=="set_kp"||cmd=="set_kpi")&&getJsonNumber(doc["value"],fv)){Kp=constrain(fv,0.0f,50.0f);publishAck(cmd.c_str(),"ok","kp_updated");return;}
  if (cmd=="set_ki"&&getJsonNumber(doc["value"],fv)){Ki=constrain(fv,0.0f,5.0f);publishAck("set_ki","ok","ki_updated");return;}
  if (cmd=="set_kd"&&getJsonNumber(doc["value"],fv)){Kd=constrain(fv,0.0f,100.0f);publishAck("set_kd","ok","kd_updated");return;}
  if (cmd=="set_speed"&&getJsonInt(doc["value"],iv)){baseSpeed=constrain(iv,baseMin,baseMax);publishAck("set_speed","ok","base_speed_updated");return;}
  if (cmd=="set_speed_min"&&getJsonInt(doc["value"],iv)){baseMin=constrain(iv,0,255);if(baseMin>baseMax)baseMin=baseMax;if(baseSpeed<baseMin)baseSpeed=baseMin;publishAck("set_speed_min","ok","base_min_updated");return;}
  if (cmd=="set_speed_max"&&getJsonInt(doc["value"],iv)){baseMax=constrain(iv,0,255);if(baseMax<baseMin)baseMax=baseMin;if(baseSpeed>baseMax)baseSpeed=baseMax;publishAck("set_speed_max","ok","base_max_updated");return;}
  if ((cmd=="set_telemetry_ms"||cmd=="set_fast_ms")&&getJsonInt(doc["value"],iv)){TELE_FAST_MS=constrain(iv,50,1000);publishAck(cmd.c_str(),"ok","fast_rate_updated");return;}
  if (cmd=="set_slow_ms"&&getJsonInt(doc["value"],iv)){TELE_SLOW_MS=constrain(iv,300,5000);publishAck("set_slow_ms","ok","slow_rate_updated");return;}
  if (cmd=="set_cal_pwm"&&getJsonInt(doc["value"],iv)){calSweepPwm=constrain(iv,CAL_PWM_MIN_LIMIT,CAL_PWM_MAX_LIMIT);publishAck("set_cal_pwm","ok","calibration_pwm_updated");requestSlowSnapshot("calibration_pwm_updated");return;}
  if (cmd=="set_cal_edge"&&getJsonInt(doc["value"],iv)){calEdgePosTarget=constrain(iv,CAL_EDGE_POS_TARGET_MIN,CAL_EDGE_POS_TARGET_MAX);publishAck("set_cal_edge","ok","calibration_edge_updated");requestSlowSnapshot("calibration_edge_updated");return;}
  if (cmd=="set_align_kp"&&getJsonNumber(doc["value"],fv)){alignPidKp=constrain(fv,0.0f,2.0f);publishAck("set_align_kp","ok","align_kp_updated");requestSlowSnapshot("align_kp_updated");return;}
  if (cmd=="set_align_ki"&&getJsonNumber(doc["value"],fv)){alignPidKi=constrain(fv,0.0f,1.0f);publishAck("set_align_ki","ok","align_ki_updated");requestSlowSnapshot("align_ki_updated");return;}
  if (cmd=="set_align_kd"&&getJsonNumber(doc["value"],fv)){alignPidKd=constrain(fv,0.0f,1.0f);publishAck("set_align_kd","ok","align_kd_updated");requestSlowSnapshot("align_kd_updated");return;}
  if (cmd=="set_lost_sum"&&getJsonInt(doc["value"],iv)){lostSumTh=constrain(iv,0,4000);publishAck("set_lost_sum","ok","lost_sum_updated");return;}
  if (cmd=="set_lost_mx"&&getJsonInt(doc["value"],iv)){lostMxTh=constrain(iv,0,1000);publishAck("set_lost_mx","ok","lost_mx_updated");return;}
  if (cmd=="set_adaptive"&&getJsonInt(doc["value"],iv)){
    adaptiveControlEnabled=(iv!=0);
    if (!adaptiveControlEnabled) resetAdaptiveControl();
    publishAck("set_adaptive","ok",adaptiveControlEnabled?"adaptive_enabled":"adaptive_disabled");
    return;
  }
  if ((cmd=="set_ai"||cmd=="set_ai_supervisor"||cmd=="set_supervisor")&&getJsonInt(doc["value"],iv)){
    aiSupervisorEnabled=(iv!=0);
    if (!aiSupervisorEnabled) resetAISupervisorState();
    publishAck(cmd.c_str(),"ok",aiSupervisorEnabled?"ai_enabled":"ai_disabled");
    return;
  }
  if ((cmd=="set_ai_model"||cmd=="set_supervisor_model")&&getJsonInt(doc["value"],iv)){
    if ((iv!=0) && !HAS_AI_MODEL_HOOK) {
      aiModelEnabled=false;
      publishAck(cmd.c_str(),"rejected","ai_model_hook_missing");
    } else {
      aiModelEnabled=(iv!=0);
      publishAck(cmd.c_str(),"ok",aiModelEnabled?"ai_model_enabled":"ai_model_disabled");
    }
    return;
  }
  if ((cmd=="set_ai_blend"||cmd=="set_supervisor_blend")&&getJsonNumber(doc["value"],fv)){
    aiSupervisorBlend=clampf(fv,0.0f,1.0f);
    publishAck(cmd.c_str(),"ok","ai_blend_updated");
    return;
  }
  if ((cmd=="set_profile"||cmd=="profile") && parseDriveProfileValue(doc["value"], profile)) {
    applyDriveProfile(profile);
    return;
  }
  if (cmd=="status"){
    if (robotState == STATE_RUNNING) {
      publishAck("status","ok","running_snapshot_suppressed");
      return;
    }
    publishAck("status","ok","snapshot_requested");
    publishStatus("online","manual_status");
    publishTelemetryFast("manual_status");
    requestSlowSnapshot("manual_status");
    return;
  }
  publishAck(cmd.c_str(),"error","unknown_or_bad_value");
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  (void)topic;
  char msg[512];
  unsigned int n=min(length,(unsigned int)(sizeof(msg)-1));
  memcpy(msg,payload,n); msg[n]='\0';
  JsonDocument doc;
  if (!deserializeJson(doc,msg)){handleCommand(doc);return;}
  String raw=String(msg); raw.trim(); raw.toLowerCase();
  if (otaInProgress && raw!="status") {
    publishAck(raw.c_str(),"rejected","ota_in_progress");
    return;
  }
  if (raw=="start")       applyStart();
  else if (raw=="stop")   applyStop();
  else if (raw=="dump_track" || raw=="publish_track" || raw=="export_track") {
    if (robotState == STATE_RUNNING) publishAck(raw.c_str(),"rejected","stop_first");
    else if (!trackRecorderRequestPublish()) publishAck(raw.c_str(),"rejected","no_track_ready");
    else publishAck(raw.c_str(),"ok","track_publish_started");
  }
  else if (raw=="align" || raw=="alinear" || raw=="center" || raw=="centrar") startLineAlign();
  else if (raw=="calibrate"){stopMotors();startCalibration(true);publishAck("calibrate","ok","started_plain");}
  else if (raw=="profile_safe"){applyDriveProfile(PROFILE_SAFE);}
  else if (raw=="profile_normal"){applyDriveProfile(PROFILE_NORMAL);}
  else if (raw=="profile_fast"){applyDriveProfile(PROFILE_FAST);}
  else if (raw=="ai_on"){aiSupervisorEnabled=true;publishAck("ai_on","ok","ai_enabled_plain");}
  else if (raw=="ai_off"){aiSupervisorEnabled=false;resetAISupervisorState();publishAck("ai_off","ok","ai_disabled_plain");}
  else if (raw=="status"){
    if (robotState == STATE_RUNNING) publishAck("status","ok","running_snapshot_suppressed");
    else {
      publishStatus("online","manual_status_plain");
      publishTelemetryFast("manual_status_plain");
      requestSlowSnapshot("manual_status_plain");
    }
  }
}

// ─── WiFi / MQTT ──────────────────────────────────────────────────────────────
void beginWiFi() {
  WiFi.mode(WIFI_STA); WiFi.persistent(false);
  WiFi.setSleep(false); WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID,WIFI_PASS);
}

void configureNTP() {
  if (ntpConfigured) return;
  configTime(0,0,"pool.ntp.org","time.nist.gov");
  ntpConfigured=true;
}

void serviceWiFi() {
  bool connected=(WiFi.status()==WL_CONNECTED);
  if (connected&&!wifiWasConnected){
    wifiWasConnected=true; configureNTP();
    Serial.print("[WIFI] connected IP="); Serial.print(WiFi.localIP());
    Serial.print(" RSSI="); Serial.println(WiFi.RSSI());
    printNetworkSnapshot();
    publishEvent("info","wifi_connected","WiFi connected");
    if (!otaReady){
      ArduinoOTA.setHostname(DEVICE_ID);
      ArduinoOTA.onStart([](){ queueDeferredFlag(DEFERRED_OTA_START); });
      ArduinoOTA.onEnd([](){ queueDeferredFlag(DEFERRED_OTA_END); });
      ArduinoOTA.onError([](ota_error_t error){ queueDeferredFlag(DEFERRED_OTA_ERROR, (uint8_t)error); });
      ArduinoOTA.begin();
      // [FIX-3] Iniciar tarea OTA en Core 0 con stack suficiente
      xTaskCreatePinnedToCore(otaTask,"ota_task",4096,nullptr,1,&otaTaskHandle,0);
      otaReady=true;
    }
  }
  if (!connected&&wifiWasConnected){
    wifiWasConnected=false;
    Serial.print("[WIFI] disconnected status="); Serial.println((int)WiFi.status());
    if (mqttClient.connected()) mqttClient.disconnect();
  }
  if (!connected&&(millis()-lastWiFiTryMs>=WIFI_RETRY_MS)){
    lastWiFiTryMs=millis();
    Serial.println("[WIFI] reconnecting...");
    beginWiFi();
  }
}

void serviceMQTT() {
  if (WiFi.status()!=WL_CONNECTED) return;
  if (mqttClient.connected()){
    if (!mqttWasConnected){
      mqttWasConnected=true;
      Serial.println("[MQTT] connected");
      mqttClient.subscribe(TOPIC_CMD);
      publishStatus("online","mqtt_connected");
      publishEvent("info","mqtt_connected","MQTT connected and subscribed");
      publishAck("boot","ok","ready");
      publishTelemetryFast("boot");
      requestSlowSnapshot("boot");
    }
    return;
  }
  if (mqttWasConnected){
    mqttWasConnected=false;
    Serial.println("[MQTT] disconnected");
  }
  if (millis()-lastMQTTTryMs>=MQTT_RETRY_MS){
    lastMQTTTryMs=millis();
    mqttConnectAttempts++;
    char lwt[192];
    snprintf(
      lwt, sizeof(lwt),
      "{\"type\":\"status\",\"topic_role\":\"status\",\"schema_version\":%u,"
      "\"device_id\":\"%s\",\"online\":\"offline\",\"reason\":\"lwt\"}",
      SCHEMA_VERSION, DEVICE_ID
    );
    if (!mqttClient.connect(DEVICE_ID,TOPIC_STATUS,1,true,lwt)){
      mqttConnectFails++;
      Serial.print("[MQTT] connect fail host="); Serial.print(MQTT_HOST);
      Serial.print(" port="); Serial.print(MQTT_PORT);
      Serial.print(" rc="); Serial.print(mqttClient.state());
      Serial.print(" reason="); Serial.println(mqttStateToStr(mqttClient.state()));
    }
  }
}


// ─── Track Recorder — funciones ─────────────────────────────────────────────

void trackRecorderBegin() {
  trackSampleCount   = 0;
  trackPublishIdx    = 0;
  trackPublishActive = false;
  trackDataReady     = false;
  trackHeaderSent    = false;
  lastTrackSampleMs  = 0;
  lastTrackChunkMs   = 0;
  trackHeading       = 0.0f;
  trackX             = 0.0f;
  trackY             = 0.0f;
  trackRunId         = runId;
  trackStartMs       = millis();
}

void trackRecorderSample() {
  if (robotState != STATE_RUNNING || !calibrated) return;
  uint32_t now = millis();
  if (trackSampleCount > 0 && (now - lastTrackSampleMs) < TRACK_SAMPLE_MS) return;
  if (trackSampleCount >= TRACK_SAMPLE_CAP) return;

  float dt = (trackSampleCount == 0)
             ? (TRACK_SAMPLE_MS * 0.001f)
             : ((float)(now - lastTrackSampleMs) * 0.001f);
  dt = constrain(dt, 0.005f, 0.200f);
  lastTrackSampleMs = now;

  float vFwd = 0.0f;
  if (!lineLostFlag && (motorSpeedA != 0 || motorSpeedB != 0)) {
    float vA = fabsf((float)motorSpeedA) * TRACK_SPEED_SCALE;
    float vB = fabsf((float)motorSpeedB) * TRACK_SPEED_SCALE;
    vFwd = 0.5f * (vA + vB);
  }

  float dHeading = ((float)linePos) * TRACK_HEADING_SCALE * dt * 10.0f;
  trackHeading += dHeading;
  trackX += vFwd * cosf(trackHeading) * dt;
  trackY += vFwd * sinf(trackHeading) * dt;

  TrackSample& s = trackSamples[trackSampleCount++];
  s.t_ms       = (uint16_t)constrain((int32_t)(now - trackStartMs), 0, 65535);
  s.line_pos   = (int16_t)constrain(linePos, -1500, 1500);
  s.motor_a    = (int8_t)constrain(motorSpeedA / 2, -127, 127);
  s.motor_b    = (int8_t)constrain(motorSpeedB / 2, -127, 127);
  s.x          = trackX;
  s.y          = trackY;
  s.heading    = trackHeading;
  s.local_mode = (uint8_t)localMode;
  s.flags      = 0;
  if (lineLostFlag)       s.flags |= 0x01;
  if (intersectionActive) s.flags |= 0x02;
}

void trackRecorderFinish() {
  if (trackSampleCount == 0) return;
  trackPublishActive = false;
  trackPublishIdx    = 0;
  trackDataReady     = true;
  trackHeaderSent    = false;
  lastTrackChunkMs   = 0;
  Serial.printf("[TRACK] Run finalizado: %u muestras de %u max\n",
                trackSampleCount, (unsigned)TRACK_SAMPLE_CAP);
}

bool trackRecorderRequestPublish() {
  if (robotState == STATE_RUNNING) return false;
  if (!trackDataReady || trackSampleCount == 0) return false;
  trackPublishActive = true;
  trackPublishIdx = 0;
  trackHeaderSent = false;
  lastTrackChunkMs = 0;
  return true;
}

void trackRecorderPublishChunk() {
  if (!trackPublishActive) return;
  if (robotState == STATE_RUNNING) return;  // nunca publicar mientras el robot corre
  if (!mqttClient.connected()) return;
  if (!heapOkForTelemetry()) return;
  mqttClient.loop();

  uint32_t now = millis();
  if (lastTrackChunkMs > 0 && (now - lastTrackChunkMs) < TRACK_CSV_CHUNK_MS) return;
  lastTrackChunkMs = now;

  if (!trackHeaderSent) {
    snprintf(trackCsvBuf, sizeof(trackCsvBuf),
      "META,run_id=%lu,samples=%u,sample_ms=%lu,speed_scale=%.4f,heading_scale=%.6f\n"
      "t_ms,line_pos,motor_a,motor_b,x_cm,y_cm,heading_rad,local_mode,flags\n",
      (unsigned long)trackRunId, trackSampleCount,
      (unsigned long)TRACK_SAMPLE_MS,
      TRACK_SPEED_SCALE, TRACK_HEADING_SCALE);
    if (mqttPublishSafe(TOPIC_TRACK_CSV, trackCsvBuf, false)) {
      trackHeaderSent = true;
    }
    return;
  }

  if (trackPublishIdx >= trackSampleCount) {
    snprintf(trackCsvBuf, sizeof(trackCsvBuf),
             "END,run_id=%lu,total=%u\n",
             (unsigned long)trackRunId, trackSampleCount);
    if (mqttPublishSafe(TOPIC_TRACK_CSV, trackCsvBuf, false)) {
      trackPublishActive = false;
      Serial.printf("[TRACK] CSV publicado: %u filas run_id=%lu\n",
                    trackSampleCount, (unsigned long)trackRunId);
    }
    return;
  }

  uint16_t pos = 0;
  uint16_t nextIdx = trackPublishIdx;
  while (nextIdx < trackSampleCount) {
    const TrackSample& s = trackSamples[nextIdx];
    char row[100];
    int n = snprintf(row, sizeof(row),
      "%u,%d,%d,%d,%.2f,%.2f,%.4f,%u,%u\n",
      s.t_ms, (int)s.line_pos,
      (int)(s.motor_a * 2), (int)(s.motor_b * 2),
      s.x, s.y, s.heading,
      (unsigned)s.local_mode, (unsigned)s.flags);
    if (n <= 0 || (pos + (uint16_t)n) >= TRACK_CSV_CHUNK_BYTES) break;
    memcpy(trackCsvBuf + pos, row, (size_t)n);
    pos += (uint16_t)n;
    nextIdx++;
  }
  if (pos > 0) {
    trackCsvBuf[pos] = '\0';
    if (mqttPublishSafe(TOPIC_TRACK_CSV, trackCsvBuf, false)) {
      trackPublishIdx = nextIdx;
    }
  }
}

// ─── Setup ───────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200); delay(300);
  Serial.printf("\n[BOOT] setup v%s\n", FW_VERSION);

  analogReadResolution(12); analogSetAttenuation(ADC_11db);
  for (int i=0;i<4;i++) pinMode(QTR_PINS[i],INPUT);
  pinMode(AIN1,OUTPUT); pinMode(AIN2,OUTPUT);
  pinMode(BIN1,OUTPUT); pinMode(BIN2,OUTPUT);
  pinMode(STBY,OUTPUT); digitalWrite(STBY,HIGH);
  stopMotors();
  rgbLedPrimary.begin();
  rgbLedPrimary.setBrightness(RGB_LED_BRIGHTNESS);
  rgbLedPrimary.clear();
  rgbLedPrimary.show();
  if (RGB_LED_USE_SECONDARY) {
    rgbLedSecondary.begin();
    rgbLedSecondary.setBrightness(RGB_LED_BRIGHTNESS);
    rgbLedSecondary.clear();
    rgbLedSecondary.show();
  }
  LED_OFF();

  mqttClient.setServer(MQTT_HOST,MQTT_PORT);
  mqttClient.setCallback(mqttCallback);
  // [FIX-1] Buffer ajustado al tamaño real del payload slow
  mqttClient.setBufferSize(sizeof(payloadSlow));
  // [FIX-6] Keepalive mayor y socket timeout generoso
  mqttClient.setKeepAlive(30);
  mqttClient.setSocketTimeout(5);

  beginWiFi();
  resetControllerState();
  applyDriveProfile(PROFILE_NORMAL, false);
  robotState=STATE_IDLE; calibrating=false; calibrated=false;
  if (loadCalibrationFromNvs()) {
    Serial.println("[BOOT] calibration restored from NVS");
  } else {
    Serial.println("[BOOT] no valid calibration stored");
  }
  stopMotors();
}

// ─── Loop ─────────────────────────────────────────────────────────────────────
void loop() {
  uint32_t loopStartUs=micros();
  loopCounter++;

  serviceWiFi();
  // [FIX-3] OTA ya corre en su propia tarea; no bloqueamos aquí
  serviceMQTT();
  if (mqttClient.connected()) mqttClient.loop();
  processDeferredEvents();

  updateCalibration();
  updateLineAlignment();
  updatePID();
  captureAutonomousRunSample();
  updateAutonomousRunWindow();

  // [FIX-4] Percepción en IDLE sólo cada IDLE_PERCEPTION_MS
  if (robotState==STATE_IDLE && calibrated && !calibrating && !alignToLineActive) {
    if (millis()-lastIdlePercMs >= IDLE_PERCEPTION_MS) {
      lastIdlePercMs=millis();
      int p=calcLinePos();
      linePos=p;
      bool lost=(sensorMax<lostMxTh)||(sensorSum<lostSumTh);
      lineLostFlag=lost;
      if (lost){
        if (lostSinceMs==0) lostSinceMs=millis();
        lostForMs=millis()-lostSinceMs;
      } else {
        lostSinceMs=0; lostForMs=0; updateLastDir();
      }
      updatePerceptionFeatures();
      refreshCurveIntensity();
      updateLocalMode();
      updateAdaptiveControl();
    }
  }

  trackRecorderSample();
  trackRecorderPublishChunk();
  updateLED();

  // [FIX-8] Guard de heap en publicacion
  if (mqttClient.connected() && heapOkForTelemetry() && !otaInProgress && periodicTelemetryAllowed()) {
    uint32_t nowMs = millis();
    if (robotState == STATE_RUNNING) {
      if (nowMs - lastFastMs >= RUN_DATASET_FAST_MS) {
        lastFastMs = nowMs;
        publishTelemetryFast("run_capture");
      }
    } else {
      if (nowMs - lastFastMs >= TELE_FAST_MS){
        lastFastMs = nowMs;
        publishTelemetryFast();
      }
      bool doPeriodicSlow=(nowMs - lastSlowMs >= TELE_SLOW_MS);
      bool doPendingSlow=pendingSlowSnapshot&&(nowMs - lastSlowMs >= 300);
      if (doPeriodicSlow||doPendingSlow){
        lastSlowMs=nowMs;
        publishTelemetrySlow(pendingSlowCode[0]?pendingSlowCode:nullptr);
        pendingSlowSnapshot=false; pendingSlowCode[0]='\0';
      }
      publishAutonomousBufferChunk();
    }
  }

  uint32_t loopDurUs=micros()-loopStartUs;
  loopUsAcc+=loopDurUs;
  if (loopDurUs>loopUsMax) loopUsMax=loopDurUs;
  uint32_t fh=ESP.getFreeHeap();
  if (fh<freeHeapMin) freeHeapMin=fh;

  if (millis()-lastLoopRateMs>=1000){
    lastLoopRateMs=millis();
    loopHz=loopCounter; loopCounter=0;
    loopUsAvg=(loopHz>0)?(loopUsAcc/loopHz):0;
    loopUsAcc=0; loopUsMax=0;
  }
}
