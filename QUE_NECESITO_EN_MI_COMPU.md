# Que necesito tener listo en mi computadora

## 1. Programas que debes tener abiertos o instalados

- PlatformIO / VS Code
- Python 3
- Node-RED
- Browser para abrir el dashboard de Node-RED
- Microsoft Word
- Simulador del gemelo digital

## 2. Archivos que ya deje organizados

- `documentacion/U4A5_Barboza_Bryan_sin_portada.docx`
- `README.md`
- `requirements.txt`
- `guia_demo_presencial.md`
- `dataset/linefollower_ai_dataset.jsonl`
- `dataset/linefollower_ai_dataset_auto_labeled_basic3.csv`
- `modelo/resumen_modelo.json`
- `modelo/knn_u4a3.joblib`
- `simulador/presets/test_3_final_realmatch.json`
- `simulador/presets/test_4_visual_linehug.json`
- `simulador/outputs/test_4_path_follow_metrics.json`

## 3. Codigo que debes mostrar en el video

- Firmware del seguidor: `firmware_sanitizado/src/main.cpp`
- Flujo de adquisicion: `dashboard_node_red_sanitizado/node-red.json`
- Preprocesado: `scripts/prepare_dataset.py`
- Entrenamiento e inferencia: `scripts/train_ai_supervisor_model.py`
- Gemelo digital: `simulador/launch_simulator_gui.py`
- Reproduccion de telemetria: `scripts/replay_dataset_mqtt.py`

## 4. Lo que debes tener visible durante la demo

1. El robot fisico encendido.
2. Node-RED recibiendo datos.
3. El CSV limpio o el resumen del modelo.
4. El gemelo digital abierto.
5. Las metricas del modelo: baseline, accuracy y F1.
6. La comparacion real vs simulada.

## 5. Comandos utiles

```powershell
python .\scripts\prepare_dataset.py --label-mode auto --class-profile basic3
python .\scripts\train_ai_supervisor_model.py
python .\simulador\run_all_tests.py
python .\simulador\launch_simulator_gui.py
python .\scripts\replay_dataset_mqtt.py --session session_20260423_154428 --speed 1.0
```

## 6. Pendientes reales

- Agregar el video real de demostracion.
- Agregar la portada oficial al Word.
- Si vas a publicar fuera de tu PC, subir el repositorio remoto.

