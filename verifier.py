"""El VLM usado tal cual como verificador Yes/No.

No se añade ninguna capa: se hace un forward y se leen los logits de los tokens
"Yes" y "No" en la última posición. score = logit_yes - logit_no, p = sigmoid(score).
El fine-tuning (LoRA) entrena exactamente esa misma salida con BCE.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor

from common import resolve_model


def _first_token(tok, word: str) -> int:
    ids = tok.encode(word, add_special_tokens=False)
    if len(ids) != 1:
        print(f"[aviso] '{word}' se tokeniza en {ids}; se usa el primer token")
    return ids[0]


class VLMVerifier:
    def __init__(self, model: str, dtype: str = "bfloat16", device_map="auto",
                 load_4bit: bool = False, attn_impl: str | None = "sdpa",
                 min_pixels: int | None = None, max_pixels: int | None = None):
        mid = resolve_model(model)
        pkw = {}
        if min_pixels:
            pkw["min_pixels"] = min_pixels
        if max_pixels:
            pkw["max_pixels"] = max_pixels
        self.processor = AutoProcessor.from_pretrained(mid, **pkw)
        mkw = {"torch_dtype": getattr(torch, dtype), "device_map": device_map}
        if attn_impl:
            mkw["attn_implementation"] = attn_impl
        if load_4bit:  # para depurar en la RTX 2080 (8 GB)
            from transformers import BitsAndBytesConfig
            mkw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=getattr(torch, dtype), bnb_4bit_quant_type="nf4")
        self.model = AutoModelForImageTextToText.from_pretrained(mid, **mkw)
        tok = self.processor.tokenizer
        self.id_yes, self.id_no = _first_token(tok, "Yes"), _first_token(tok, "No")

    @property
    def device(self):
        return next(self.model.parameters()).device

    def scores(self, inputs: dict) -> torch.Tensor:
        """Devuelve logit_yes - logit_no por elemento del batch (B,)."""
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                 for k, v in inputs.items() if k not in ("labels_bin", "meta")}
        try:
            out = self.model(**batch, logits_to_keep=1, use_cache=False)
        except TypeError:  # versiones sin logits_to_keep
            out = self.model(**batch, use_cache=False)
        last = out.logits[:, -1, :].float()  # padding a la izquierda -> última posición válida
        return last[:, self.id_yes] - last[:, self.id_no]

    def loss(self, inputs: dict, pos_weight: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.scores(inputs)
        y = inputs["labels_bin"].to(s.device)
        pw = torch.tensor([pos_weight], device=s.device) if pos_weight else None
        return F.binary_cross_entropy_with_logits(s, y, pos_weight=pw), s.detach()


def add_lora(verifier: VLMVerifier, r=16, alpha=32, dropout=0.05, train_vision=False):
    """LoRA solo en las proyecciones lineales del LLM (la visión queda congelada por defecto)."""
    from peft import LoraConfig, get_peft_model

    names = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    targets = []
    for n, mod in verifier.model.named_modules():
        if not isinstance(mod, torch.nn.Linear) or not n.endswith(names):
            continue
        is_vision = any(k in n for k in ("visual", "vision"))
        if is_vision and not train_vision:
            continue
        targets.append(n)
    if not targets:
        raise RuntimeError("No se encontraron capas objetivo para LoRA; revisa los nombres del modelo")
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=targets, bias="none")
    verifier.model = get_peft_model(verifier.model, cfg)
    verifier.model.print_trainable_parameters()
    return verifier
