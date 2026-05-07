import csv
import json
import os
import re
import time
from typing import Optional
from difflib import SequenceMatcher
from pydantic import BaseModel, ValidationError
from ollama import chat, generate


# ==========================================
# 🎛️ CONFIGURACIÓN DE LA MATRIZ DE PRUEBAS
# ==========================================
MODELOS_A_PROBAR = ["qwen2.5-coder:7b"]
TEMPERATURAS = [0.0]
CONTEXTOS = [4096]
DATASET_JIRAS = "dataset_jiras.json"
ARCHIVO_RESULTADOS = "resultados_evaluacion.csv"
CARPETA_TRAZAS = "trazas_evaluacion"


# ==========================================
# 🎛️ MODELOS DE DATOS PARA JSON ESTRUCTURADO
# ==========================================

class DatosExtraidos(BaseModel):
    """Modelo para la extracción de datos personales"""
    nombres: list[str] = []
    emails: list[str] = []
    ips: list[str] = []
    urls: list[str] = []

# ==========================================
# 🎛️ DEFINICIÓN DE PIPELINES (FUNCIONES PYTHON)
# ==========================================

def calcular_similitud(texto1, texto2):
    """Calcula la similitud entre dos textos (0-1)"""
    return SequenceMatcher(None, texto1, texto2).ratio()

def pipeline_extraccion_avanzado(salida_chat, texto_original):
    """
    Pipeline avanzado con múltiples estrategias y validación mejorada
    """
    try:
        # Intentar validar con Pydantic primero
        try:
            datos_extraidos = DatosExtraidos.model_validate_json(salida_chat)
        except ValidationError:
            # Si falla, intentar extraer JSON de otras formas
            datos_extraidos = extraer_json_robusto(salida_chat)
        
        if not datos_extraidos:
            raise ValueError("No se pudo extraer JSON válido")
        
        # Si es un objeto Pydantic, convertir a dict
        if isinstance(datos_extraidos, DatosExtraidos):
            datos = datos_extraidos.model_dump()
        else:
            datos = datos_extraidos
        
        # Extraer y validar cada tipo de dato
        nombres = validar_lista_strings(datos.get("nombres", []))
        emails = validar_lista_strings(datos.get("emails", []))
        ips = validar_lista_strings(datos.get("ips", []))
        urls = validar_lista_strings(datos.get("urls", []))
        
        resultado = texto_original
        
        # Ordenar por longitud (más largos primero para evitar reemplazos parciales)
        emails_ordenados = sorted(emails, key=len, reverse=True)
        urls_ordenados = sorted(urls, key=len, reverse=True)
        ips_ordenados = sorted(ips, key=len, reverse=True)
        nombres_ordenados = sorted(nombres, key=len, reverse=True)
        
        # Reemplazar en orden: emails -> urls -> ips -> nombres
        for email in emails_ordenados:
            if email in resultado:
                resultado = resultado.replace(email, "[EMAIL]")
        
        for url in urls_ordenados:
            if url in resultado:
                resultado = resultado.replace(url, "[URL]")
        
        for ip in ips_ordenados:
            if ip in resultado:
                resultado = resultado.replace(ip, "[IP]")
        
        for nombre in nombres_ordenados:
            if nombre in resultado:
                resultado = resultado.replace(nombre, "[PERSONA]")
        
        return resultado
        
    except Exception as e:
        print(f"   ⚠️ Error en pipeline: {e}")
        return salida_chat

def extraer_json_robusto(texto):
    """Intenta extraer JSON de la respuesta usando múltiples estrategias"""
    # Estrategia 1: Ya viene validado de Pydantic
    try:
        return DatosExtraidos.model_validate_json(texto)
    except:
        pass
    
    # Estrategia 2: Buscar entre marcadores de código
    patrones = [
        r'```json\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
        r'\{[^{}]*\{.*?\}[^{}]*\}',  # JSON anidado simple
        r'\{.*?\}'  # Último recurso: primer objeto JSON
    ]
    
    for patron in patrones:
        try:
            match = re.search(patron, texto, re.DOTALL)
            if match:
                json_str = match.group(1) if match.lastindex else match.group(0)
                # Intentar parsear y validar
                datos = json.loads(json_str)
                return DatosExtraidos(**datos)
        except:
            continue
    
    return None

def validar_lista_strings(lista):
    """Valida y limpia una lista de strings"""
    if not isinstance(lista, list):
        return []
    
    resultado = []
    for item in lista:
        if isinstance(item, str):
            item_limpio = item.strip()
            if item_limpio and len(item_limpio) > 1:
                resultado.append(item_limpio)
    return resultado

def pipeline_con_postprocesado(salida_chat, texto_original):
    """
    Pipeline que aplica post-procesado para corregir errores comunes
    """
    # Primero aplicamos el pipeline normal
    resultado = pipeline_extraccion_avanzado(salida_chat, texto_original)
    
    # Si el resultado es igual a la salida original (falló el pipeline)
    if resultado == salida_chat:
        # Intentar correcciones comunes
        resultado = corregir_errores_comunes(salida_chat, texto_original)
    
    return resultado

def corregir_errores_comunes(texto_procesado, texto_original):
    """
    Corrige errores comunes como cambios en conjugaciones verbales,
    preposiciones, artículos, etc.
    """
    # Palabras que el modelo suele cambiar incorrectamente
    correcciones = {
        # Conjugaciones verbales comunes
        "realiza": ["realizó", "realizo", "realiza"],
        "realizó": ["realiza", "realizo"],
        "tiene": ["tenía", "tuvo", "tiene"],
        "tenía": ["tiene", "tuvo"],
        "tuvo": ["tiene", "tenía"],
        "hace": ["hizo", "hacía", "hace"],
        "hizo": ["hace", "hacía"],
        "va": ["fue", "iba", "va"],
        "fue": ["va", "iba"],
        "está": ["estaba", "estuvo", "está"],
        "estaba": ["está", "estuvo"],
        "puede": ["pudo", "podía", "puede"],
        "pudo": ["puede", "podía"],
        "dice": ["dijo", "decía", "dice"],
        "dijo": ["dice", "decía"],
        "da": ["dio", "daba", "da"],
        "dio": ["da", "daba"],
        
        # Preposiciones y artículos
        "de": ["del", "de"],
        "del": ["de"],
        "a": ["al", "a"],
        "al": ["a"],
        "en": ["en el", "en la", "en"],
        "por": ["para", "por"],
        "para": ["por", "para"],
        
        # Pronombres
        "se": ["le", "les", "se"],
        "le": ["se", "les"],
        "lo": ["la", "los", "las", "lo"],
        "la": ["lo", "las", "el"],
    }
    
    # Si el texto procesado es muy diferente al original, 
    # probablemente no vale la pena intentar corregir
    similitud = calcular_similitud(texto_procesado, texto_original)
    if similitud < 0.7:
        return texto_procesado
    
    # Intentar restaurar palabras del original que fueron cambiadas
    palabras_originales = set(texto_original.split())
    palabras_procesadas = set(texto_procesado.split())
    
    # Palabras que están en el original pero no en el procesado
    palabras_perdidas = palabras_originales - palabras_procesadas
    
    # Si hay pocas diferencias, intentar restaurar
    if len(palabras_perdidas) < len(texto_original.split()) * 0.1:  # Menos del 10% de cambio
        resultado = texto_procesado
        for palabra in palabras_perdidas:
            # Ver si la palabra fue reemplazada por una similar
            for palabra_procesada in list(palabras_procesadas):
                if palabra_procesada in correcciones:
                    if palabra in correcciones[palabra_procesada]:
                        # Encontramos una posible corrección
                        resultado = resultado.replace(palabra_procesada, palabra)
                        break
        
        return resultado
    
    return texto_procesado


os.makedirs(CARPETA_TRAZAS, exist_ok=True)

# PROMPTS MEJORADOS CON SOPORTE PARA JSON ESTRUCTURADO
PROMPTS_A_PROBAR = {
    "prompt_v1_copia_exacta": {
        "sistema": """Eres un software de enmascaramiento de datos. Tu tarea es EXACTAMENTE copiar el texto de entrada y sustituir SOLO los datos personales por etiquetas.

REGLAS ABSOLUTAS (NO NEGOCIABLES):
1. COPIA TEXTUAL: Cada palabra, espacio, coma, punto y salto de línea debe ser IDÉNTICO al original
2. NO MODIFIQUES: No cambies tiempos verbales, preposiciones, artículos ni ninguna palabra
3. SOLO SUSTITUYE: Únicamente cambia nombres por [PERSONA], emails por [EMAIL], IPs por [IP], URLs por [URL]
4. MANTÉN GRAMÁTICA: Conserva la conjugación verbal exacta del original
5. SIN INTERPRETACIÓN: No corrijas, mejores ni interpretes el texto

Si el texto dice "Juan realizó la tarea", debe quedar "[PERSONA] realizó la tarea" (NO "realiza" ni "realizaba")
Si el texto dice "María tenía un problema", debe quedar "[PERSONA] tenía un problema" (NO "tiene")""",
        "pipeline": None,
        "use_json_format": False,
        "post_procesar": True
    },
    
    "prompt_v2_extractor_preciso": {
        "sistema": """Eres un extractor de datos para enmascaramiento. Tu ÚNICA función es identificar y extraer datos personales del texto.

ANALIZA el texto y extrae SOLO los siguientes tipos de datos:
- nombres: Nombres completos de personas (Nombre Apellido, Apellidos o Nombre Apellidos)
- emails: Direcciones de correo electrónico completas
- ips: Direcciones IP (IPv4 o IPv6)
- urls: URLs completas con protocolo

REGLAS DE EXTRACCIÓN:
1. SOLO EXTRAE DATOS LITERALES: Copia exactamente como aparecen en el texto
2. NO INTERPRETES: Si no estás 100% seguro, no incluyas el dato
3. NOMBRES: Solo nombres completos de personas (Nombre Apellido, Apellidos o Nombre y Apellidos)
4. EMAILS: Deben contener @ y dominio (ej: usuario@dominio.com)
5. IPs: Direcciones IP completas (ej: 192.168.1.1)
6. URLs: Deben incluir protocolo (http:// o https://)

EJEMPLO CORRECTO:
Texto: "Juan Pérez envió un correo a maria@garcia.com desde 192.168.1.1"
Extracción: nombres=["Juan Pérez"], emails=["maria@garcia.com"], ips=["192.168.1.1"], urls=[]

ERRORES A EVITAR:
- NO extraigas palabras sueltas como nombres
- NO inventes datos que no aparecen literalmente
- NO incluyas signos de puntuación en los datos extraídos""",
        "pipeline": pipeline_con_postprocesado,
        "use_json_format": True,
        "post_procesar": True,
        "schema_class": DatosExtraidos
    },
    
    "prompt_v3_hibrido_preciso": {
        "sistema": """Eres un sistema de enmascaramiento de datos de alta precisión.

PROCESO:
1. Primero, copia EXACTAMENTE el texto de entrada (sin cambios)
2. Luego, en la copia, sustituye SOLO los datos personales

REGLAS CRÍTICAS:
- COPIA TEXTUAL: El 99% del texto debe ser idéntico al original
- NO MEJORES: No corrijas gramática, ortografía ni estilo
- NO INTERPRETES: Si una palabra puede ser nombre o no, déjala como está
- CONSERVA CONJUGACIONES: "realizó" sigue siendo "realizó", "tenía" sigue siendo "tenía"

SUSTITUCIONES EXACTAS:
- Nombres completos de personas → [PERSONA]
- Correos electrónicos → [EMAIL]  
- Direcciones IP → [IP]
- URLs → [URL]

EJEMPLO:
Entrada: "Juan Pérez realizó la configuración del servidor 192.168.1.1"
Salida: "[] realizó la configuración del servidor [IP]"
(NOTA: "realizó" se mantiene igual, NO se cambia a "realiza")""",
        "pipeline": None,
        "use_json_format": False,
        "post_procesar": True
    }
}

def normalizar_texto_mejorado(texto):
    """Normalización mejorada que preserva mejor la estructura verbal"""
    if not texto: 
        return ""
    
    texto = texto.lower().strip()
    
    # Quitar acentos (pero mantener la estructura de la palabra)
    reemplazos_acentos = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
        'ä': 'a', 'ë': 'e', 'ï': 'i', 'ö': 'o', 'ü': 'u',
        'â': 'a', 'ê': 'e', 'î': 'i', 'ô': 'o', 'û': 'u',
        'ñ': 'n', 'ç': 'c'
    }
    
    for acento, sin_acento in reemplazos_acentos.items():
        texto = texto.replace(acento, sin_acento)
    
    # Normalizar espacios pero mantener puntuación básica
    texto = re.sub(r'[\r\t]+', ' ', texto)
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    
    # Mantener puntos, comas y otros signos de puntuación importantes
    texto = re.sub(r'[^\w\s\[\]\n@.,;:!?¡¿()\-_/]', ' ', texto)
    
    # Normalizar espacios
    texto = re.sub(r' +', ' ', texto)

    #Eliminar tokens  <texto_a_procesar> </texto_a_procesar>

    texto = texto.replace(' texto_a_procesar ', '')
    texto = texto.replace(' /texto_a_procesar ', '')
    texto = texto.replace(' texto_a_procesado ', '')
    texto = texto.replace(' /texto_a_procesado ', '')

    lineas = [linea.strip() for linea in texto.split('\n')]
    texto = '\n'.join(linea for linea in lineas if linea)
    
    return texto.strip()

def cargar_json(archivo):
    """Carga un archivo JSON con manejo de errores"""
    try:
        with open(archivo, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: No se encuentra el archivo {archivo}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error: JSON inválido en {archivo}: {e}")
        return None

def ejecutar_evaluacion_mejorada():
    """Evaluación con métricas adicionales de similitud y soporte JSON estructurado"""
    jiras = cargar_json(DATASET_JIRAS)
    if not jiras: 
        return

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
                    use_json = config_prompt.get("use_json_format", False)
                    post_procesar = config_prompt.get("post_procesar", False)
                    schema_class = config_prompt.get("schema_class", None)
                    
                    tiempo_inicio_combinacion = time.time()

                    print(f"\n🤖 [{combinacion_actual}/{total_combinaciones}] {modelo} | {prompt_id}")
                    print(f"   ⚙️ Temp:{temp} | Ctx:{context} | JSON:{use_json} | PostPro:{post_procesar}")
                    
                    aciertos = 0
                    aciertos_parciales = 0  # Similitud > 95%
                    tiempos = []
                    similitudes = []
                    traza_detallada = []

                    for idx, item in enumerate(jiras, 1):
                        jira_id = item.get("id", "N/A")
                        texto_original = item["original"]
                        texto_esperado = item["esperado"]
                        texto_esperado_limpio = normalizar_texto_mejorado(texto_esperado)
                        
                        prompt_usuario = f"<texto_a_procesar>\n{texto_original}\n</texto_a_procesar>"

                        inicio_tiempo = time.time()
                        texto_evaluado = ""
                        exito = False
                        exito_parcial = False
                        similitud = 0
                        tiempo_empleado = 0

                        try:
                            opciones_ollama = {
                                "temperature": temp,
                                "num_ctx": context,
                                "seed": 42,
                                "top_k": 1,
                                "top_p": 0.9,
                                "repeat_penalty": 1.1,
                                "presence_penalty": 0.1
                            }
                            
                            # LLAMADA A OLLAMA CON O SIN ESQUEMA JSON
                            if use_json and schema_class:
                                # Usar chat() con el esquema JSON de Pydantic
                                respuesta = chat(
                                    model=modelo,
                                    messages=[
                                        {
                                            "role": "system",
                                            "content": prompt_texto
                                        },
                                        {
                                            "role": "user",
                                            "content": prompt_usuario
                                        }
                                    ],
                                    format=schema_class.model_json_schema(),
                                    options=opciones_ollama
                                )
                                salida_chat_bruta = respuesta.message.content.strip()
                                
                                # Validar que cumple con el esquema
                                try:
                                    datos_validados = schema_class.model_validate_json(salida_chat_bruta)
                                    print(f"   ✅ JSON validado: {len(datos_validados.nombres)} nombres, {len(datos_validados.emails)} emails, {len(datos_validados.ips)} IPs, {len(datos_validados.urls)} URLs")
                                except Exception as e:
                                    print(f"   ⚠️ JSON no cumple el esquema: {e}")
                                    
                            else:
                                # Usar generate() normal para texto
                                respuesta = generate(
                                    model=modelo,
                                    prompt=prompt_usuario,
                                    system=prompt_texto,
                                    options=opciones_ollama
                                )
                                salida_chat_bruta = respuesta.response.strip()
                            
                            tiempo_empleado = time.time() - inicio_tiempo
                            tiempos.append(tiempo_empleado)

                            # Aplicar pipeline
                            if pipeline_func is not None:
                                texto_evaluado = pipeline_func(salida_chat_bruta, texto_original)
                            else:
                                texto_evaluado = salida_chat_bruta
                                if post_procesar:
                                    texto_evaluado = corregir_errores_comunes(texto_evaluado, texto_original)

                            texto_generado_limpio = normalizar_texto_mejorado(texto_evaluado)
                            
                            # Calcular similitud
                            similitud = calcular_similitud(texto_generado_limpio, texto_esperado_limpio)
                            similitudes.append(similitud)
                            
                            # Evaluar éxito
                            if texto_generado_limpio == texto_esperado_limpio:
                                aciertos += 1
                                exito = True
                                exito_parcial = True
                            elif similitud >= 0.95:  # 95% de similitud
                                aciertos_parciales += 1
                                exito_parcial = True

                        except Exception as e:
                            print(f"   ⚠️ Error: {e}")
                            tiempos.append(0)
                            texto_evaluado = f"ERROR: {str(e)}"
                            similitudes.append(0)

                        # Mostrar progreso con más detalle
                        icono = "✅" if exito else ("🟡" if exito_parcial else "❌")
                        print(f"   [{idx}/{len(jiras)}] {jira_id} ... {tiempo_empleado:.2f}s {icono} (sim:{similitud:.3f})")

                        traza_detallada.append({
                            "id_jira": jira_id,
                            "original": texto_original,
                            "esperado": texto_esperado,
                            "procesado_final": texto_evaluado,
                            "resultado": "EXACTO" if exito else ("PARCIAL" if exito_parcial else "FALLO"),
                            "similitud": similitud,
                            "tiempo_segundos": tiempo_empleado
                        })

                    # Métricas ampliadas
                    precision = (aciertos / len(jiras)) * 100 if len(jiras) > 0 else 0
                    precision_parcial = ((aciertos + aciertos_parciales) / len(jiras)) * 100 if len(jiras) > 0 else 0
                    tiempo_medio = sum(tiempos) / len(tiempos) if tiempos else 0
                    similitud_media = sum(similitudes) / len(similitudes) if similitudes else 0
                    tiempo_combinacion = time.time() - tiempo_inicio_combinacion

                    resumen_final.append({
                        "Modelo": modelo,
                        "Prompt_ID": prompt_id,
                        "Temperatura": temp,
                        "Contexto": context,
                        "Precision_Exacta": f"{precision:.1f}%",
                        "Precision_Parcial": f"{precision_parcial:.1f}%",
                        "Similitud_Media": f"{similitud_media:.3f}",
                        "Aciertos": f"{aciertos}/{len(jiras)}",
                        "Velocidad_Media_Seg": f"{tiempo_medio:.2f}",
                        "Tiempo_Total_Seg": f"{tiempo_combinacion:.2f}",
                        "Modo_JSON": "Sí" if use_json else "No",
                        "PostProcesado": "Sí" if post_procesar else "No"
                    })

                    # Guardar traza detallada
                    nombre_archivo_traza = f"{modelo}_{prompt_id}_t{temp}_c{context}".replace(":", "-") + ".json"
                    ruta_traza = os.path.join(CARPETA_TRAZAS, nombre_archivo_traza)
                    with open(ruta_traza, "w", encoding="utf-8") as f:
                        json.dump(traza_detallada, f, indent=2, ensure_ascii=False)

                    print(f"   📊 Exacta:{precision:.1f}% | Parcial:{precision_parcial:.1f}% | Sim:{similitud_media:.3f}")
                    print(f"   💾 Traza: {nombre_archivo_traza}")

    tiempo_total = time.time() - inicio_global
    print(f"\n⏱️ Tiempo total: {tiempo_total:.2f}s")
    guardar_en_csv(resumen_final)

def guardar_en_csv(datos):
    """Guarda los resultados en CSV"""
    columnas = ["Modelo", "Prompt_ID", "Temperatura", "Contexto", "Precision_Exacta", 
                "Precision_Parcial", "Similitud_Media", "Aciertos", "Velocidad_Media_Seg", 
                "Tiempo_Total_Seg", "Modo_JSON", "PostProcesado"]
    
    with open(ARCHIVO_RESULTADOS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columnas)
        writer.writeheader()
        writer.writerows(datos)
    
    print(f"🎉 ¡Proceso finalizado! Resumen guardado en: {ARCHIVO_RESULTADOS}")

if __name__ == "__main__":
    ejecutar_evaluacion_mejorada()