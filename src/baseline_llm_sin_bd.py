import json
import time
from datasets import load_dataset
from litellm import completion
from tqdm import tqdm

def main():
    # carga del dataset reducido (validación) y muestra de 150 preguntas para el baseline
    muestra = load_dataset("nlp-udesa/hotpot_qa_3k", split="validation").shuffle(seed=42).select(range(150))

    resultados_baseline = []

    with tqdm(total=len(muestra), desc="Procesando preguntas") as pbar:
        # iteración y consulta a gemini
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
                "respuesta_modelo": response.choices[0].message.content.strip(),
                "type": fila['type'],
                "level": fila['level']
            })

            time.sleep(4.5) # esperar 4.5 segundos para no superar el límite de 15 requests per minute
            pbar.update(1)
    # JSON en disco para calcular métricas
    with open("resultados_baseline.json", "w", encoding="utf-8") as f:
        json.dump(resultados_baseline, f, indent=4, ensure_ascii=False)

    print("Respuestas guardadas en resultados_baseline.json")


if __name__ == "__main__":
    main()
