#pragma once

#include "ai_supervisor_contract.h"

// Copia este archivo como ai_model_hook.h cuando tengas un modelo real.
// Debe devolver true si produjo una sugerencia valida.
static inline bool runAIModelHook(const AISupervisorFrame* window,
                                  int count,
                                  AISupervisorOutput& out) {
  (void)window;
  (void)count;
  (void)out;
  return false;
}
