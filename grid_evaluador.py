import csv
import json
import re
import time
import ollama

# ==========================================
# 🎛️ CONFIGURACIÓN DE LA MATRIZ DE PRUEBAS
# ==========================================
# Añade aquí los modelos que tengas descargados en tu Ollama
MODELOS_A_PROBAR = ["qwen2.5-coder:0.5b ", "qwen2.5-coder:3b"]  #"gemma2:2b"
PROMPTS_A_PROBAR = "prompts.json"
TEMPERATURAS = [0.0]  # Mantener a 0.0 para mayor estabilidad
CONTEXTOS = [4096,6000]
DATASET_JIRAS = "dataset_jiras.json"
ARCHIVO_RESULTADOS = "resultados_evaluacion.csv"


def normalizar_texto(texto):
    """Pipeline de normalización para evitar falsos negativos."""
    if not texto: return ""
    texto = texto.lower()
    #quitar acentos
    texto = re.sub(r"[áàä]", "a", texto)
    texto = re.sub(r"[éèë]", "e", texto)
    texto = re.sub(r"[íìï]", "i", texto)
    texto = re.sub(r"[óòö]", "o", texto)
    texto = re.sub(r"[úùü]", "u", texto)

    # Cambia saltos de línea por espacios
    texto = re.sub(r"[\n\r\t]+", " ", texto)
    # Quita puntuación conflictiva dejando letras, números, espacios y corchetes []
    texto = re.sub(r"[^\w\s\[\]]", "", texto)
    # Colapsa múltiples espacios en uno solo
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

    total_combinaciones = len(MODELOS_A_PROBAR) * len(prompts_dict) * len(TEMPERATURAS)
    print(f"🚀 Iniciando Grid Search. Total de combinaciones: {total_combinaciones}\n")

    resumen_final = []

    for modelo in MODELOS_A_PROBAR:
        for prompt_id, prompt_texto in prompts_dict.items():
            for temp in TEMPERATURAS:
                for context in CONTEXTOS:
                
                    print(f"🤖 Evaluando: {modelo} | {prompt_id} | Temp: {temp} | Ctc: {context}" )
                    aciertos = 0
                    tiempos = []

                    for item in jiras:
                        texto_original = item["original"]
                        texto_esperado_limpio = normalizar_texto(item["esperado"])
                        prompt_usuario = f"<jira_text>\n{texto_original}\n</jira_text>"

                        inicio_tiempo = time.time()
                        try:
                            # LLAMADA A LA API CON PARÁMETROS DE ESTABILIDAD
                            respuesta = ollama.generate(
                                model=modelo,
                                prompt=prompt_usuario,
                                system=prompt_texto,
                                options={
                                    "temperature": temp,
                                    "num_ctx": context,
                                    "seed": 42,      # 🔐 Clave para fijar el determinismo
                                    "top_k": 1       # 🔐 Elige siempre la mejor opción
                                },
                            )
                            tiempo_empleado = time.time() - inicio_tiempo
                            tiempos.append(tiempo_empleado)

                            texto_generado_limpio = normalizar_texto(respuesta["response"])

                            if texto_generado_limpio == texto_esperado_limpio:
                                aciertos += 1
                        except Exception as e:
                            print(f"   ⚠️ Error en Ollama con {modelo}: {e}")
                            tiempos.append(0)

                    # Guardar métricas
                    precision = (aciertos / len(jiras)) * 100 if len(jiras) > 0 else 0
                    tiempo_medio = sum(tiempos) / len(tiempos) if tiempos else 0
                    resumen_final.append({
                        "Modelo": modelo,
                        "Prompt_ID": prompt_id,
                        "Temperatura": temp,
                        "Contexto": context,
                        "Precision": f"{precision:.1f}%",
                        "Aciertos": f"{aciertos}/{len(jiras)}",
                        "Velocidad_Media_Seg": f"{tiempo_medio:.2f}"
                    })
                    print(f"   📊 Resultado: {precision:.1f}% de precisión | {tiempo_medio:.2f}s de media.\n")

    # Guardar a CSV
    guardar_en_csv(resumen_final)


def guardar_en_csv(datos):
    columnas = ["Modelo", "Prompt_ID", "Temperatura", "Contexto", "Precision", "Aciertos", "Velocidad_Media_Seg"]
    try:
        with open(ARCHIVO_RESULTADOS, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columnas)
            writer.writeheader()
            writer.writerows(datos)
        print(f"🎉 ¡Proceso finalizado! Resultados en: {ARCHIVO_RESULTADOS}")
    except Exception as e:
        print(f"❌ Error al guardar el CSV: {e}")


if __name__ == "__main__":
    ejecutar_evaluacion()
