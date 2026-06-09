from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


SESSION_NAME = "session_20260423_154428"


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="CoverTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            spaceAfter=14,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            alignment=TA_CENTER,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#0b3d91"),
            spaceBefore=8,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyJustify",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.3,
            leading=15,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallNote",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8.8,
            leading=12,
            textColor=colors.HexColor("#555555"),
            alignment=TA_LEFT,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8.8,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#444444"),
            spaceAfter=8,
        )
    )
    return styles


def make_table(data, col_widths):
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e6f2")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9aa9b8")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fbff")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def fit_image(path: Path, max_width_cm: float, max_height_cm: float) -> Image:
    img = Image(str(path))
    width = img.imageWidth
    height = img.imageHeight
    scale = min((max_width_cm * cm) / width, (max_height_cm * cm) / height)
    img.drawWidth = width * scale
    img.drawHeight = height * scale
    return img


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report_dir = root / "telemetria" / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = report_dir / f"{SESSION_NAME}_regression_metrics.json"
    clean_csv_path = report_dir / f"{SESSION_NAME}_clean_regression_dataset.csv"
    trajectory_png = report_dir / f"{SESSION_NAME}_regression_trajectory.png"
    comparison_png = report_dir / f"{SESSION_NAME}_comparison.png"
    sensor_png = report_dir / f"{SESSION_NAME}_sensor_position.png"
    guide_png = root / "telemetria" / "pista_limpia_1664x1664.png"
    output_pdf = root / "U3A1_Apellido_Nombre.pdf"

    if not metrics_path.exists() or not clean_csv_path.exists():
        raise SystemExit("Primero ejecuta linear_regression_practice.py para generar las evidencias del reporte.")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    df = pd.read_csv(clean_csv_path)
    styles = build_styles()
    story = []

    story.append(Spacer(1, 2.2 * cm))
    story.append(Paragraph("PORTADA INSTITUCIONAL PARA COMPLETAR", styles["CoverTitle"]))
    story.append(Paragraph("Unidad 3 - Actividad 1", styles["CoverBody"]))
    story.append(Paragraph("Regresion lineal base para reconstruccion de trayectoria en robot seguidor de linea", styles["CoverBody"]))
    story.append(Spacer(1, 0.8 * cm))
    story.append(Paragraph("Materia: ________________________________", styles["CoverBody"]))
    story.append(Paragraph("Docente: ________________________________", styles["CoverBody"]))
    story.append(Paragraph("Equipo: 3 integrantes", styles["CoverBody"]))
    story.append(Paragraph("Integrante 1: ________________________________", styles["CoverBody"]))
    story.append(Paragraph("Integrante 2: ________________________________", styles["CoverBody"]))
    story.append(Paragraph("Integrante 3: ________________________________", styles["CoverBody"]))
    story.append(Spacer(1, 0.8 * cm))
    story.append(Paragraph(f"Fecha de generacion del reporte: {datetime.now().strftime('%d/%m/%Y')}", styles["CoverBody"]))
    story.append(Spacer(1, 1.0 * cm))
    story.append(
        Paragraph(
            "Nota: no se encontro una portada oficial institucional dentro del proyecto. "
            "Se dejo esta pagina lista para sustituir o completar con el formato oficial que les pidan.",
            styles["SmallNote"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("Objetivo", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "Implementar un modelo base de regresion lineal a partir de los datos capturados por un robot "
            "seguidor de linea de 4 sensores analogicos, para obtener una representacion grafica "
            "aproximada de la trayectoria recorrida y relacionar el comportamiento del robot con los datos "
            "registrados desde Node-RED.",
            styles["BodyJustify"],
        )
    )

    story.append(Paragraph("Pista usada", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "La pista utilizada se trabajo con una imagen guia de referencia de 120 x 120 cm. "
            "El robot empleado tiene un tamano aproximado de 18 x 20 cm. "
            "Esta misma geometria se utilizo durante el procesamiento para ubicar el punto de inicio "
            "en la esquina inferior derecha y seguir el sentido real del carro hacia la derecha.",
            styles["BodyJustify"],
        )
    )
    story.append(fit_image(guide_png, 14.5, 9.8))
    story.append(Paragraph("Figura 1. Imagen de la pista usada como referencia visual.", styles["Caption"]))

    story.append(Paragraph("Captura de datos con Node-RED", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "La captura de datos se realizo con el flujo MQTT/Node-RED del proyecto. "
            "Durante la corrida, el robot publico telemetria rapida en el topic "
            "<b>robot/linefollower/telemetry/fast</b>. Node-RED almaceno las muestras "
            "en un archivo JSONL y permitio registrar vueltas manuales para segmentar cada recorrido. "
            "Posteriormente se extrajo una corrida completa y se transformo a CSV limpio para el analisis.",
            styles["BodyJustify"],
        )
    )

    story.append(Paragraph("Variables registradas", styles["SectionTitle"]))
    variable_rows = [
        ["Variable", "Descripcion"],
        ["sample_index / time_s", "Numero de muestra y tiempo transcurrido durante la corrida."],
        ["sensor_1 ... sensor_4", "Lecturas normalizadas de los cuatro sensores analogicos de izquierda a derecha."],
        ["line_position_weighted", "Posicion estimada de la linea calculada con pesos asignados a cada sensor."],
        ["line_pos_firmware", "Estimacion de posicion producida por el firmware del robot."],
        ["motor_a_pwm / motor_b_pwm", "PWM aplicado a cada motor durante la corrida."],
        ["line_lost", "Indicador binario de perdida de linea."],
        ["local_mode", "Modo local detectado: recta, curva suave/fuerte o recuperacion."],
        ["manual_lap_count / manual_lap_marked", "Conteo y marcas de vuelta enviadas desde Node-RED."],
        ["guide_x_cm / guide_y_cm", "Coordenadas aproximadas en centimetros dentro del plano de referencia."],
    ]
    story.append(make_table(variable_rows, [5.0 * cm, 11.0 * cm]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Limpieza del archivo", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "La sesion seleccionada fue <b>session_20260423_154428</b>. "
            f"Primero se detectaron <b>{metrics['rows_raw_running']}</b> muestras en estado <b>running</b>. "
            "Despues se conservaron unicamente las vueltas completas marcadas manualmente y se elimino el tramo final "
            "posterior a la ultima marca, ya que correspondia a una falla de seguimiento. "
            f"Con ello se descartaron <b>{metrics['rows_removed_tail']}</b> filas del tramo final. "
            f"Ademas, se revisaron valores vacios y duplicados: se eliminaron <b>{metrics['missing_removed']}</b> "
            f"filas por datos faltantes y <b>{metrics['duplicates_removed']}</b> filas duplicadas. "
            f"El CSV final quedo con <b>{metrics['rows_clean']}</b> muestras limpias.",
            styles["BodyJustify"],
        )
    )

    cleaning_rows = [
        ["Etapa", "Cantidad"],
        ["Muestras running originales", str(metrics["rows_raw_running"])],
        ["Filas eliminadas por tramo final despues de la ultima vuelta", str(metrics["rows_removed_tail"])],
        ["Filas eliminadas por datos faltantes", str(metrics["missing_removed"])],
        ["Filas eliminadas por duplicados", str(metrics["duplicates_removed"])],
        ["Muestras finales en el CSV limpio", str(metrics["rows_clean"])],
    ]
    story.append(make_table(cleaning_rows, [10.0 * cm, 4.5 * cm]))

    story.append(PageBreak())
    story.append(Paragraph("Estimacion de posicion de la linea", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "Para estimar la posicion de la linea se asignaron pesos a los sensores de izquierda a derecha: "
            "[-1.5, -0.5, 0.5, 1.5]. Con ello, cuando predominan los sensores izquierdos la posicion estimada "
            "tiende a valores negativos y cuando predominan los sensores derechos la posicion tiende a valores positivos. "
            "La formula aplicada fue:",
            styles["BodyJustify"],
        )
    )
    story.append(
        Paragraph(
            "<b>posicion_ponderada = (w1*s1 + w2*s2 + w3*s3 + w4*s4) / (s1 + s2 + s3 + s4)</b>",
            styles["CoverBody"],
        )
    )
    story.append(
        Paragraph(
            "Esta expresion produce una posicion relativa continua de la linea usando solo las lecturas del arreglo "
            "de 4 sensores, lo cual sirve como entrada principal para el modelo de regresion.",
            styles["BodyJustify"],
        )
    )
    story.append(fit_image(sensor_png, 15.5, 10.5))
    story.append(Paragraph("Figura 2. Lecturas de sensores y posicion de linea estimada con pesos.", styles["Caption"]))

    story.append(Paragraph("Modelo base de regresion lineal", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "Se utilizo una regresion lineal multivariable con dos salidas: coordenada X y coordenada Y "
            "dentro del plano de referencia. Las entradas del modelo fueron: progreso global de la corrida, "
            "fase dentro de la vuelta, posicion ponderada de la linea, lecturas de los cuatro sensores, PWM de motores, "
            "bandera de perdida de linea y una expansion armonica simple de la fase de vuelta para representar la forma cerrada "
            "de la pista. Aunque se trata de un modelo lineal en sus coeficientes, esta expansion permite reconstruir "
            "de forma base una geometria de pista mucho mas cercana al recorrido real.",
            styles["BodyJustify"],
        )
    )
    story.append(
        Paragraph(
            f"Desempeno del modelo sobre la corrida analizada: <b>R²(X) = {metrics['r2_x']:.4f}</b>, "
            f"<b>R²(Y) = {metrics['r2_y']:.4f}</b>, "
            f"<b>RMSE(X) = {metrics['rmse_x_cm']:.2f} cm</b> y "
            f"<b>RMSE(Y) = {metrics['rmse_y_cm']:.2f} cm</b>.",
            styles["BodyJustify"],
        )
    )
    story.append(fit_image(trajectory_png, 13.5, 11.0))
    story.append(Paragraph("Figura 3. Trayectoria reconstruida con el modelo base de regresion lineal.", styles["Caption"]))

    story.append(Paragraph("Comparacion visual con la pista real", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "La Figura 4 compara la pista de referencia con el camino promedio estimado previamente y con la trayectoria "
            "obtenida mediante regresion lineal. Visualmente, la forma general coincide con el recorrido principal: "
            "recta inferior larga, subida por el lado izquierdo, bucle interno y seccion superior curva hacia la derecha. "
            "La reconstruccion no es exacta centimetro a centimetro, pero si conserva la geometria global de la pista, "
            "lo que confirma que la relacion entre sensores, tiempo y accion de motores contiene suficiente informacion "
            "para producir un modelo base util.",
            styles["BodyJustify"],
        )
    )
    story.append(fit_image(comparison_png, 17.0, 7.0))
    story.append(Paragraph("Figura 4. Comparacion entre la pista usada, el camino promedio y la salida del modelo lineal.", styles["Caption"]))

    story.append(Paragraph("Conclusiones tecnicas", styles["SectionTitle"]))
    conclusions = [
        "El flujo Node-RED/MQTT permitio capturar una corrida completa del robot y separar vueltas utiles mediante marcas manuales.",
        "La limpieza de datos fue determinante: al remover el tramo posterior a la ultima vuelta marcada se obtuvo un conjunto mas estable y coherente para el analisis.",
        "La posicion de la linea calculada con pesos sobre los cuatro sensores resulta una variable compacta y facil de interpretar para describir el comportamiento lateral del robot.",
        "El modelo base de regresion lineal fue suficiente para aproximar la forma general de la pista, aunque no sustituye a sistemas de localizacion real como encoders o IMU.",
        "El procedimiento deja una base reutilizable para practicas posteriores de aprendizaje supervisado y visualizacion en tiempo real.",
    ]
    for item in conclusions:
        story.append(Paragraph(f"• {item}", styles["BodyJustify"]))

    story.append(Paragraph("Anexos y evidencias", styles["SectionTitle"]))
    annex_rows = [
        ["Archivo", "Uso en la practica"],
        [clean_csv_path.name, "CSV limpio utilizado como evidencia y fuente del analisis."],
        ["linear_regression_practice.py", "Codigo Python para limpieza, estimacion y regresion lineal."],
        [metrics_path.name, "Metricas numericas del modelo lineal."],
        [trajectory_png.name, "Grafica principal de la trayectoria reconstruida."],
    ]
    story.append(make_table(annex_rows, [6.5 * cm, 9.0 * cm]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(
        Paragraph(
            "Ruta del CSV limpio: "
            f"{clean_csv_path}<br/>"
            "Ruta del codigo: "
            f"{(root / 'telemetria' / 'linear_regression_practice.py')}",
            styles["SmallNote"],
        )
    )

    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=1.7 * cm,
        rightMargin=1.7 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="Reporte de practica - Regresion lineal del robot seguidor de linea",
        author="OpenAI Codex",
    )
    doc.build(story)
    print(f"PDF generado: {output_pdf}")


if __name__ == "__main__":
    main()
