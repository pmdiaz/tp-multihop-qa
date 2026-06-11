"""Calcula EM y F1 (normalizados según HotpotQA) para cualquier archivo de resultados."""

import argparse
import json
import re
import string
from collections import Counter, defaultdict


def normalizar_respuesta(s):
    if isinstance(s, list):
        if s and isinstance(s[0], dict) and "text" in s[0]:
            s = " ".join(bloque.get("text", "") for bloque in s)
        else:
            s = " ".join(str(x) for x in s)
    elif not isinstance(s, str):
        s = str(s)

    s = s.lower()
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())


def exact_match(prediccion, real):
    return int(normalizar_respuesta(prediccion) == normalizar_respuesta(real))


def f1_score(prediccion, real):
    pred_tokens = normalizar_respuesta(prediccion).split()
    real_tokens = normalizar_respuesta(real).split()
    comunes = Counter(pred_tokens) & Counter(real_tokens)
    num_iguales = sum(comunes.values())
    if num_iguales == 0:
        return 0.0
    precision = num_iguales / len(pred_tokens)
    recall = num_iguales / len(real_tokens)
    return (2 * precision * recall) / (precision + recall)


def calcular_metricas(items):
    em = sum(exact_match(i["respuesta_modelo"], i["respuesta_real"]) for i in items)
    f1 = sum(f1_score(i["respuesta_modelo"], i["respuesta_real"]) for i in items)
    n = len(items)
    return em / n * 100, f1 / n * 100, n


def main():
    parser = argparse.ArgumentParser(description="Calcula EM y F1 sobre un archivo de resultados JSON.")
    parser.add_argument("archivo", help="Ruta al JSON de resultados (ej: resultados_baseline.json)")
    args = parser.parse_args()

    with open(args.archivo, "r", encoding="utf-8") as f:
        resultados = json.load(f)

    em, f1, n = calcular_metricas(resultados)
    print(f"\nTotal preguntas: {n}")
    print(f"EM: {em:.2f}%   F1: {f1:.2f}%")

    # Breakdown por tipo (bridge / comparison)
    tiene_type = "type" in resultados[0] if resultados else False
    if tiene_type:
        por_tipo = defaultdict(list)
        for item in resultados:
            por_tipo[item["type"]].append(item)
        print("\nPor tipo:")
        for tipo in sorted(por_tipo):
            e, f, c = calcular_metricas(por_tipo[tipo])
            print(f"  {tipo:<12} ({c:>3})   EM: {e:.2f}%   F1: {f:.2f}%")

    #Desgloze por nivel (easy / medium / hard)
    tiene_level = "level" in resultados[0] if resultados else False
    if tiene_level:
        por_nivel = defaultdict(list)
        for item in resultados:
            por_nivel[item["level"]].append(item)
        orden = ["easy", "medium", "hard"]
        print("\nPor nivel:")
        for nivel in orden:
            if nivel in por_nivel:
                e, f, c = calcular_metricas(por_nivel[nivel])
                print(f"  {nivel:<8} ({c:>3})   EM: {e:.2f}%   F1: {f:.2f}%")

    #Uso de tools
    tiene_traza = False
    for i in resultados:
        if "traza" in i:
            tiene_traza = True
            break

    if tiene_traza:
        items_con_traza = []
        for i in resultados:
            if "traza" in i:
                items_con_traza.append(i)

        llamadas = []
        for i in items_con_traza:
            llamadas.append(i["traza"]["num_llamadas"])

        promedio = sum(llamadas) / len(llamadas)
        sin_llamadas = 0
        for x in llamadas:
            if x == 0:
                sin_llamadas += 1

        print(f"\nUso de herramientas ({len(items_con_traza)} preguntas con traza):")
        print(f"  Promedio llamadas/pregunta: {promedio:.2f}")
        print(f"  Sin llamadas (0):           {sin_llamadas} ({sin_llamadas / len(items_con_traza) * 100:.1f}%)")
        for k in sorted(set(llamadas)):
            cnt = llamadas.count(k)
            print(f"  {k} llamada{'s' if k != 1 else ''}:  {cnt:>4} preguntas ({cnt / len(llamadas) * 100:.1f}%)")

        if tiene_type:
            print("\n  Promedio llamadas por tipo:")
            for tipo in sorted(por_tipo):
                items_t = []
                for i in por_tipo[tipo]:
                    if "traza" in i:
                        items_t.append(i)
                if items_t:
                    total_llamadas = 0
                    for i in items_t:
                        total_llamadas += i["traza"]["num_llamadas"]
                    avg = total_llamadas / len(items_t)
                    print(f"    {tipo:<12} {avg:.2f}")
    print()


if __name__ == "__main__":
    main()
