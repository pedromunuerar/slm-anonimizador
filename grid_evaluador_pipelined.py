import csv
import json
import os
import re
import time
import ollama

# ==========================================
# 🎛️ DEFINICIÓN DE PIPELINES (FUNCIONES PYTHON)
# ==========================================

def pipeline_extraccion(salida_chat, texto_original):
    """
    Simula el caso en que el LLM solo nos devuelve un JSON con datos,
    y nosotros en Python hacemos el reemplazo sobre el texto original.
    """
    try:
        # Intentamos parsear lo que escupió la IA como un JSON
        datos_extraidos = json.loads(salida_chat)
        resultado = texto_original
        
        # Hacemos el reemplazo determinista en Python (Estilo Oracle Dev)
        for nombre in datos_extraidos.get("nombres", []):
            resultado = resultado.replace(nombre, "[PERSONA]")
        for ip in datos_extraidos.get("ips", []):
            resultado = resultado.replace(ip, "[IP]")
            
        return resultado
    except Exception:
        # Si la IA no devolvió un JSON válido, devolvemos la salida en bruto
        return salida_chat

# ==========================================
# 🎛️ CONFIGURACIÓN DE LA MATRIZ DE PRUEBAS
# ==========================================
MODELOS_A_PROBAR = ["qwen2.5-coder:3b"]
TEMPERATURAS = [0.0]
CONTEXTOS = [4096, 6000]
DATASET_JIRAS = "dataset_jiras.json"
ARCHIVO_RESULTADOS = "resultados_evaluacion.csv"
CARPETA_TRAZAS = "trazas_evaluacion"

os.makedirs(CARPETA_TRAZAS, exist_ok=True)

# AQUÍ ESTÁ EL CAMBIO: El diccionario ahora admite objetos con funciones asignadas
PROMPTS_A_PROBAR = {
    "prompt_v1": {
        "sistema": "Eres un software de anonimización. Cambia nombres por [PERSONA] e IPs por [IP].",
        "pipeline": None # No hace falta procesar nada después
    },
    "prompt_v2_extractor": {
        "sistema": "Extrae nombres de personas, emails e IPs. Devuelve SOLO un JSON con las claves 'nombres' (lista) e 'ips' (lista). No digas nada más.",
        "pipeline": pipeline_extraccion #  Enganchamos la función de Python
    }
}

def normalizar_texto(texto):
    if not texto: return ""
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
    if not jiras: return

    resumen_final = []
    total_combinaciones = len(MODELOS_A_PROBAR) * len(PROMPTS_A_PROBAR) * len(TEMPERATURAS) * len(CONTEXTOS)
    combinacion_actual = 0
    inicio_global = time.time()

    for modelo in MODELOS_A_PROBAR:
        for prompt_id, config_prompt in PROMPTS_A_PROBAR.items():
            for temp in TEMPERATURAS:
                for context in CONTEXTOS:
                    combinacion_actual += 1
                    prompt_texto = config_prompt["sistema"]
                    pipeline_func = config_prompt["pipeline"]
                    tiempo_inicio_combinacion = time.time()

                    print(f"\n🤖 [{combinacion_actual}/{total_combinaciones}] Evaluando: {modelo} | {prompt_id} | Temp: {temp} | Ctx: {context}")
                    aciertos = 0
                    tiempos = []
                    traza_detallada = []

                    for idx, item in enumerate(jiras, 1):
                        jira_id = item.get("id", "N/A")
                        texto_original = item["original"]
                        texto_esperado_limpio = normalizar_texto(item["esperado"])
                        prompt_usuario = f"<jira_text>\n{texto_original}\n</jira_text>"

                        inicio_tiempo = time.time()
                        texto_evaluado = ""
                        exito = False

                        try:
                            respuesta = ollama.generate(
                                model=modelo,
                                prompt=prompt_usuario,
                                system=prompt_texto,
                                options={"temperature": temp, "num_ctx": context, "seed": 42, "top_k": 1},
                            )
                            tiempo_empleado = time.time() - inicio_tiempo
                            tiempos.append(tiempo_empleado)

                            salida_chat_bruta = respuesta["response"]

                            if pipeline_func is not None:
                                texto_evaluado = pipeline_func(salida_chat_bruta, texto_original)
                            else:
                                texto_evaluado = salida_chat_bruta

                            texto_generado_limpio = normalizar_texto(texto_evaluado)

                            if texto_generado_limpio == texto_esperado_limpio:
                                aciertos += 1
                                exito = True

                        except Exception as e:
                            print(f"   ⚠️ Error en Ollama con {modelo}: {e}")
                            tiempos.append(0)
                            texto_evaluado = f"ERROR: {str(e)}"

                        tiempo_item = tiempo_empleado if tiempos and tiempos[-1] > 0 else 0
                        icono = "✅" if exito else "❌"
                        print(f"   [{idx}/{len(jiras)}] {jira_id} ... {tiempo_item:.2f}s {icono}")

                        traza_detallada.append({
                            "id_jira": jira_id,
                            "original": texto_original,
                            "esperado": item["esperado"],
                            "procesado_final": texto_evaluado,
                            "resultado": "CORRECTO" if exito else "FALLO"
                        })

                    precision = (aciertos / len(jiras)) * 100 if len(jiras) > 0 else 0
                    tiempo_medio = sum(tiempos) / len(tiempos) if tiempos else 0
                    tiempo_combinacion = time.time() - tiempo_inicio_combinacion

                    resumen_final.append({
                        "Modelo": modelo,
                        "Prompt_ID": prompt_id,
                        "Temperatura": temp,
                        "Contexto": context,
                        "Precision": f"{precision:.1f}%",
                        "Aciertos": f"{aciertos}/{len(jiras)}",
                        "Velocidad_Media_Seg": f"{tiempo_medio:.2f}",
                        "Tiempo_Total_Combinacion_Seg": f"{tiempo_combinacion:.2f}"
                    })

                    nombre_archivo_traza = f"{modelo}_{prompt_id}_t{temp}_c{context}".replace(":", "-") + ".json"
                    with open(os.path.join(CARPETA_TRAZAS, nombre_archivo_traza), "w", encoding="utf-8") as f:
                        json.dump(traza_detallada, f, indent=2, ensure_ascii=False)

                    print(f"   📊 {precision:.1f}% aciertos | {tiempo_medio:.2f}s media | {tiempo_combinacion:.2f}s total | Traza: {nombre_archivo_traza}")

    tiempo_total = time.time() - inicio_global
    print(f"\n⏱️ Tiempo total de evaluación: {tiempo_total:.2f}s")
    guardar_en_csv(resumen_final)

def guardar_en_csv(datos):
    columnas = ["Modelo", "Prompt_ID", "Temperatura", "Contexto", "Precision", "Aciertos", "Velocidad_Media_Seg", "Tiempo_Total_Combinacion_Seg"]
    with open(ARCHIVO_RESULTADOS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columnas)
        writer.writeheader()
        writer.writerows(datos)
    print(f"🎉 ¡Proceso finalizado! Resumen guardado en: {ARCHIVO_RESULTADOS}")

if __name__ == "__main__":
    ejecutar_evaluacion()
