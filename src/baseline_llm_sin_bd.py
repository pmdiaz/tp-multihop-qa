# pregunta al LLM sin usar la base de datos vectorial. 
# se guardan las respuestas a las preguntas en un JSON.

# Para usar Gemma o LLMs de HF:
# 1 Entrar a huggingface.co/settings/tokens.
# 2 Clic en "Create new token".
# 3 Tipo de token, "Read" (lectura). Nombre, crear y copiar.
# 4 En terminal: export HUGGINGFACE_API_KEY="token_aca"

# Para usar Gemini:
# 1 Entrar a https://aistudio.google.com/api-keys
# 2 Create "API key" y copiar.
# 3 En Terminal: export GEMINI_API_KEY="api_key_aca"

import json
import time
from datasets import load_dataset
from litellm import completion

# Carga del dataset reducido (validación) y muestra de 50 preguntas para el baseline (se pueden tomar mas)
muestra = load_dataset("nlp-udesa/hotpot_qa_3k", split="validation").select(range(50))

resultados_baseline = []

# Iteración y consulta a Gemma a través de HF
for fila in muestra:
    response = completion(
        model="gemini/gemini-3.1-flash-lite",
        messages=[
            {"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": fila['question']}
        ]
    )
    
    # Guardado de datos para métricas
    resultados_baseline.append({
        "id": fila['id'],
        "pregunta": fila['question'],
        "respuesta_real": fila['answer'],
        "respuesta_modelo": response.choices[0].message.content.strip()
    })

    time.sleep(4.5) # esperar 4.5 segundos para no superar el límite de 15 requests per minute

# JSON en disco para calcular métricas)
with open("resultados_baseline.json", "w", encoding="utf-8") as f:
    json.dump(resultados_baseline, f, indent=4, ensure_ascii=False)