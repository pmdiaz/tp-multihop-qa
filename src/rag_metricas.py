import json
import re
import string
from collections import Counter


# Limpiar los textos antes de comparar
def normalizar_respuesta(s):
    def sacar_articulos(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def arreglar_espacios(text):
        return ' '.join(text.split())
    def sacar_puntuacion(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return arreglar_espacios(sacar_articulos(sacar_puntuacion(s.lower())))

def exact_match(prediccion, real):
    return int(normalizar_respuesta(prediccion) == normalizar_respuesta(real))

def f1_score(prediccion, real):
    pred_tokens = normalizar_respuesta(prediccion).split()
    real_tokens = normalizar_respuesta(real).split()
    comunes = Counter(pred_tokens) & Counter(real_tokens)
    num_iguales = sum(comunes.values())

    if num_iguales == 0:
        return 0
    precision = 1.0 * num_iguales / len(pred_tokens)
    recall = 1.0 * num_iguales / len(real_tokens)
    return (2 * precision * recall) / (precision + recall)


def main():
    # Cargar resultados RAG (JSON)
    with open("resultados_rag.json", "r", encoding="utf-8") as f:
        resultados = json.load(f)

    em_total = 0
    f1_total = 0
    total_preguntas = len(resultados)

    # Cálculo métricas pregunta por pregunta
    for item in resultados:
        em_total += exact_match(item["respuesta_modelo"], item["respuesta_real"])
        f1_total += f1_score(item["respuesta_modelo"], item["respuesta_real"])

    # Resultados finales
    print(f"Total de preguntas evaluadas: {total_preguntas}")
    print(f"Exact Match (EM): {(em_total / total_preguntas) * 100:.2f}%")
    print(f"F1-Score: {(f1_total / total_preguntas) * 100:.2f}%")


if __name__ == "__main__":
    main()
