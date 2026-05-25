"""Wrapper del LLM (Gemma 4) para generación de texto.

Usa AutoModelForImageTextToText + AutoTokenizer, que es la API correcta para
los modelos de la familia Gemma 4 en tareas puramente de texto.

Funciones principales:

- ``load_model``: carga modelo y tokenizer desde HuggingFace o caché local.
- ``generate``: recibe una lista de mensajes en formato chat y devuelve el texto
  generado por el modelo (solo la respuesta, sin el prompt de entrada).
"""

from __future__ import annotations

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from . import config


def _print_device_info() -> None:
    print("  Dispositivos disponibles:")
    cuda_ok = torch.cuda.is_available()
    mps_ok = torch.backends.mps.is_available()
    print(f"    CUDA : {'sí — ' + torch.cuda.get_device_name(0) if cuda_ok else 'no'}")
    print(f"    MPS  : {'sí (Apple Silicon)' if mps_ok else 'no'}")
    if not cuda_ok and not mps_ok:
        print("    → usando CPU")


def load_model(
    model_name: str = config.LLM_MODEL,
) -> tuple[AutoModelForImageTextToText, AutoTokenizer]:
    """Carga el modelo y el tokenizer. Usa bfloat16 y device_map automático.

    Usamos AutoTokenizer (no AutoProcessor) para evitar cargar los procesadores
    de imagen/video de Gemma 4, que no son necesarios para texto puro.
    """
    _print_device_info()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    actual_device = next(model.parameters()).device
    print(f"  Modelo cargado en: {actual_device}")
    return model, tokenizer


def generate(
    model: AutoModelForImageTextToText,
    tokenizer: AutoTokenizer,
    messages: list[dict],
    max_new_tokens: int = config.MAX_NEW_TOKENS,
    temperature: float = config.TEMPERATURE,
) -> str:
    """Genera una respuesta a partir de una lista de mensajes en formato chat.

    Devuelve únicamente el texto generado (sin repetir el prompt de entrada).
    """
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else None,
        )

    # Recortar el prompt de entrada para devolver solo la respuesta
    prompt_len = inputs["input_ids"].shape[-1]
    response = tokenizer.decode(
        output_ids[0][prompt_len:],
        skip_special_tokens=True,
    )
    return response.strip()
