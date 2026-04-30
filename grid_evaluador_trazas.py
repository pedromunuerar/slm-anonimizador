import csv
import json
import os
import re
import time
import ollama

# ==========================================
# 🎛️ CONFIGURACIÓN DE LA MATRIZ DE PRUEBAS
# ==========================================
MODELOS_A_PROBAR = ["qwen2.5-coder:0.5b ", "qwen2.5-coder:3b","phi3.5"]
PROMPTS_A_PROBAR = "prompts.json"
TEMPERATURAS = [0.0]
CONTEXTOS = [4096]
DATASET_JIRAS = "dataset_jiras.json"
ARCHIVO_RESULTADOS = "resultados_evaluacion.csv"
CARPETA_TRAZAS = "trazas_evaluacion"  # 📂 Carpeta para las salidas detalladas

# Asegurar que la carpeta de trazas existe
os.makedirs(CARPETA_TRAZAS, exist_ok=True)


def normalizar_texto(texto):
    """Pipeline de normalización estricta."""
    if not texto:
        return ""
    texto = texto.lower()
    #quitar acentos
    texto = re.sub(r"[áàä]", "a", texto)
    texto = re.sub(r"[éèë]", "e", texto)
    texto = re.sub(r"[íìï]", "i", texto)
    texto = re.sub(r"[óòö]", "o", texto)
    texto = re.sub(r"[úùü]", "u", texto)

    texto = re.sub(r"[\n\r\t]+", " ", texto)
    texto = re.sub(r"[^\w\s\[\]]", "", texto)
    return re.sub(r"\s+", " ", texto).strip()


def cargar_json(archivo):
    try:
        with open(archivo, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: No se encuentra el archivo {archivo}")
        return None


def ejecutar_evaluacion():
    jiras = cargar_json(DATASET_JIRAS)
    prompts_dict = cargar_json(PROMPTS_A_PROBAR)

    if not jiras or not prompts_dict:
        return

    resumen_final = []

    for modelo in MODELOS_A_PROBAR:
        for prompt_id, prompt_texto in prompts_dict.items():
            for temp in TEMPERATURAS:
                for context in CONTEXTOS:
                
                    print(f"🤖 Evaluando: {modelo} | {prompt_id} | Temp: {temp} | Ctx: {context}" )
                    tiempos = []

                    aciertos = 0

                    # Lista para guardar la traza completa de esta combinación
                    traza_detallada = []

                    for item in jiras:
                        jira_id = item.get("id", "N/A")
                        texto_original = item["original"]
                        texto_esperado_limpio = normalizar_texto(item["esperado"])
                        prompt_usuario = f"<jira_text>\n{texto_original}\n</jira_text>"
                        texto_generado = ""
                        exito = False
                        inicio_tiempo = time.time()
                        try:
                            respuesta = ollama.generate(
                                model=modelo,
                                prompt=prompt_usuario,
                                system=prompt_texto,
                                options={
                                    "temperature": temp,
                                    "num_ctx": context,
                                    "seed": 42,
                                    "top_k": 1,
                                },
                            )
                            tiempo_empleado = time.time() - inicio_tiempo
                            tiempos.append(tiempo_empleado)

                            texto_generado = respuesta["response"]
                            texto_generado_limpio = normalizar_texto(texto_generado)

                            if texto_generado_limpio == texto_esperado_limpio:
                                aciertos += 1
                                exito = True

                        except Exception as e:
                            print(f"   ⚠️ Error en Ollama con {modelo}: {e}")
                            tiempos.append(0)
                            texto_generado = f"ERROR: {str(e)}"

                        # Guardamos la traza de este Jira específico
                        traza_detallada.append(
                            {
                                "id_jira": jira_id,
                                "original": texto_original,
                                "esperado": item["esperado"],
                                "generado_por_ia": texto_generado,
                                "resultado": "CORRECTO" if exito else "FALLO",
                            }
                        )

                    # 1. Calcular métricas de la combinación
                    precision = (
                        (aciertos / len(jiras)) * 100 if len(jiras) > 0 else 0
                    )
                    tiempo_medio = sum(tiempos) / len(tiempos) if tiempos else 0

                    resumen_final.append(
                        {
                            "Modelo": modelo,
                            "Prompt_ID": prompt_id,
                            "Temperatura": temp,
                            "Contexto": context,
                            "Precision": f"{precision:.1f}%",
                            "Aciertos": f"{aciertos}/{len(jiras)}",
                            "Velocidad_Media_Seg": f"{tiempo_medio:.2f}",
                        }
                    )

                    # 2. Guardar el archivo JSON de trazas para esta iteración
                    nombre_archivo_traza = (
                        f"{modelo}_{prompt_id}_t{temp}_c{context}".replace(":", "-") + ".json"
                    )
                    ruta_completa = os.path.join(
                        CARPETA_TRAZAS, nombre_archivo_traza
                    )

                    with open(ruta_completa, "w", encoding="utf-8") as f:
                        json.dump(traza_detallada, f, indent=2, ensure_ascii=False)

                    print(
                        f"   📊 {precision:.1f}% aciertos | Traza guardada en: {nombre_archivo_traza}\n"
                    )

    # 3. Guardar el CSV global
    guardar_en_csv(resumen_final)


def guardar_en_csv(datos):
    columnas = [
        "Modelo",
        "Prompt_ID",
        "Temperatura",
        "Contexto",
        "Precision",
        "Aciertos",
        "Velocidad_Media_Seg",
    ]
    try:
        with open(ARCHIVO_RESULTADOS, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columnas)
            writer.writeheader()
            writer.writerows(datos)
        print(
            f"🎉 ¡Proceso finalizado! Resumen guardado en: {ARCHIVO_RESULTADOS}"
        )
    except Exception as e:
        print(f"❌ Error al guardar el CSV: {e}")




if __name__ == "__main__":
    ejecutar_evaluacion()
