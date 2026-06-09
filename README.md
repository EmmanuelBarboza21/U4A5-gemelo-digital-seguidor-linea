# U4A5 - Gemelo Digital CPS+IA Para Robot Seguidor De Linea

Repositorio final de la actividad U4A5: reporte tecnico final, demo y documentacion.

## Objetivo

Demostrar el prototipo final del gemelo digital del robot seguidor de linea, integrando:

- robot fisico;
- adquisicion de datos;
- preprocesado;
- modelo de inferencia;
- salida accionable;
- evidencias y video de demostracion.

## Contenido Principal

- `dataset/`: dataset usado en la actividad y version limpia.
- `modelo/`: resumen del modelo y artefacto entrenado.
- `evidencias/`: capturas, graficas y resultados del prototipo.
- `firmware_sanitizado/`: firmware sin credenciales reales.
- `dashboard_node_red_sanitizado/`: flujo Node-RED listo para importar.
- `simulador/`: scripts, presets y recursos del gemelo digital.
- `scripts/`: scripts de preprocesado, entrenamiento y analisis.
- `documentacion/`: reporte tecnico en Word.
- `video/`: video real de demostracion del robot.

## Ejecucion Rapida

Desde la raiz del repositorio:

```powershell
pip install -r requirements.txt
python scripts/prepare_dataset.py --label-mode auto --class-profile basic3
python scripts/train_ai_supervisor_model.py
python simulador/run_all_tests.py
python simulador/launch_simulator_gui.py
```

## Evidencia De Video

El video real de demostracion se encuentra disponible en Google Drive:

```text
https://drive.google.com/drive/folders/1bBfNXvskMC9Q5mihdVCJGszS6gVZNWpv?usp=drive_link
```

## Ciberhigiene

- No se incluye `secrets.h` real.
- Las credenciales reales fueron retiradas del firmware.
- `firmware_sanitizado/include/secrets.h.example` se conserva solo como plantilla.
- No publicar contrasenas, IP privadas, tokens ni datos personales.

## Resultado General

El prototipo integra el flujo CPS+IA completo: robot fisico, dataset, modelo, salida accionable y gemelo digital. El modelo KNN mejora la linea base y el gemelo digital permite comparar el comportamiento del robot con una pista simulada de referencia.
