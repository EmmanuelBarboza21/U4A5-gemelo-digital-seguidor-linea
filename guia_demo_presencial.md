# Guia de grabacion y demostracion

## Que codigo usar

- Firmware del seguidor: `firmware_sanitizado/src/main.cpp`
- Flujo de adquisicion: `dashboard_node_red_sanitizado/node-red.json`
- Preprocesado: `scripts/prepare_dataset.py`
- Modelo e inferencia: `scripts/train_ai_supervisor_model.py`
- Gemelo digital: `simulador/launch_simulator_gui.py`
- Reproduccion de telemetria: `scripts/replay_dataset_mqtt.py`

## Orden recomendado para grabar el video

1. Muestra el robot fisico y la pista.
2. Abre Node-RED y enseña el flujo importado.
3. Pulsa `NUEVA SESION` y activa `Grabar dataset`.
4. Enciende el robot y deja visible la telemetria entrando.
5. Muestra el CSV o el dataset limpio para probar el preprocesado.
6. Abre el archivo `modelo/resumen_modelo.json` y enseña:
   - baseline accuracy
   - model accuracy
   - F1 macro
7. Abre el gemelo digital y enseña la trayectoria simulada.
8. Cierra con una conclusion breve: el PID controla el movimiento y la IA clasifica estados, no reemplaza el PID.

## Comandos utiles

### Captura y modelo

```powershell
python .\scripts\prepare_dataset.py --label-mode auto --class-profile basic3
python .\scripts\train_ai_supervisor_model.py
```

### Gemelo digital

```powershell
python .\simulador\run_headless_twin.py --preset .\simulador\presets\test_3_final_realmatch.json
python .\simulador\launch_simulator_gui.py
```

### Telemetria en vivo

```powershell
python .\scripts\live_track_mqtt.py --host 127.0.0.1 --port 1883
python .\scripts\replay_dataset_mqtt.py --session session_20260423_154428 --speed 1.0
```

## Que debe verse en el video

- Robot real funcionando.
- Node-RED recibiendo datos.
- Dataset o CSV limpio.
- Resultado del modelo con accuracy y F1.
- Gemelo digital mostrando la pista.
- Cierre con una frase sobre la diferencia entre PID e IA.

## Pendiente real

- Agregar el video final grabado por el equipo.

