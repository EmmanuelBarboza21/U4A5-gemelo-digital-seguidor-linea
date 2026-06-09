# Gemelo Digital Ajustado

Esta carpeta integra el simulador de `Simulador_line_follower` dentro del proyecto `Seguidor IA`
para compararlo contra el robot real usando como referencia principal la sesion:

- `session_20260423_154428`

La geometria actual del gemelo digital ya usa las medidas reales reportadas del robot:

- masa: `0.43 kg`
- diametro de rueda: `0.07 m`
- ancho de rueda: `0.026 m`
- distancia entre ruedas: `0.148 m`
- separacion entre sensores: `0.009 m`
- distancia sensores-eje: `0.145 m`
- bateria: `LiPo 2S 2200 mAh 45C (7.4 V)`
- motor: `TT 48:1`
- chasis: `0.20 m x 0.175 m`

## Contenido

- `assets/track_real_1200.png`
  - pista adaptada al simulador con escala `10 px = 1 cm`
- `presets/test_1_initial_firmware_scaled.json`
  - valores iniciales tomados del firmware real ya puestos sobre la geometria real
- `presets/test_2_tuned_stage1.json`
  - ajuste intermedio, mas rapido pero aun nervioso
- `presets/test_3_final_realmatch.json`
  - preset final recomendado
- `real_robot_reference.json`
  - resumen de medidas, observaciones y metricas del robot real
- `run_headless_twin.py`
  - ejecuta una prueba sin abrir GUI y exporta `CSV + JSON + PNG`
- `run_all_tests.py`
  - corre las tres pruebas y genera una tabla resumen
- `launch_simulator_gui.py`
  - abre la GUI del simulador con la pista y preset final ya cargados

## Uso rapido

Abrir el simulador con el preset final:

```powershell
python .\digital_twin\launch_simulator_gui.py
```

Correr las tres pruebas automaticas:

```powershell
python .\digital_twin\run_all_tests.py
```

Ejecutar una sola prueba:

```powershell
python .\digital_twin\run_headless_twin.py --preset .\digital_twin\presets\test_3_final_realmatch.json --tag final
```

## Referencia real usada

La validacion cuantitativa principal sigue la telemetria exportada desde Node-RED:

- vueltas completas analizadas: `7`
- tiempo total: `76.068 s`
- tiempo promedio por vuelta: `10.867 s`
- razon de perdida de linea: `0.0658`
- PWM medio normalizado: `0.4051`
- oscilaciones por segundo: `1.56`

Ademas, se documento como observacion manual reciente una vuelta cercana a `8.8 s`.
Ese valor se conserva en `real_robot_reference.json`, pero no se usa como metrica principal
porque la sesion telemetrica tiene marcas de vuelta y tiempos consistentes.

## Limitacion conocida

El simulador sigue siendo mas ideal que el robot real en el manejo del cruce y en las perdidas
transitorias de linea. Por eso, el ajuste final prioriza:

- tiempo por vuelta
- velocidad media
- forma del recorrido
- estabilidad general

sin forzar una coincidencia exacta en todos los eventos de oscilacion o perdida de linea.
