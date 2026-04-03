from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .rendering import apply_chat_template_with_spans

logger = logging.getLogger(__name__)


@dataclass
class StepOutput:
    generated_text: str
    prompt_text: str
    input_ids: torch.Tensor
    output_ids: torch.Tensor
    attentions: List[torch.Tensor]
    offset_mapping: List[tuple]
    message_char_spans: List[Dict]


class HFChatModel:
    def __init__(
        self,
        model_name_or_path: str,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        attn_implementation: Optional[str] = "flash_attention_2",
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_dtype = "auto" if torch_dtype == "auto" else getattr(torch, torch_dtype)
        kwargs = {
            "trust_remote_code": True,
            "device_map": device_map,
            "torch_dtype": model_dtype,
        }
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **kwargs
        )
        self.model.eval()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_gen_kwargs(
        temperature: float,
        pad_token_id: int,
        eos_token_id: int,
        max_new_tokens: int,
        **extra,
    ) -> dict:
        do_sample = temperature > 0
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            use_cache=True,
            **extra,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
        return gen_kwargs

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate_assistant_reply(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 48,
        temperature: float = 0.0,
    ) -> str:
        rendered = apply_chat_template_with_spans(self.tokenizer, messages)
        tok = self.tokenizer(
            rendered.text, return_tensors="pt", add_special_tokens=False
        )
        tok = {k: v.to(self.model.device) for k, v in tok.items()}

        gen_kwargs = self._build_gen_kwargs(
            temperature=temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
        )
        out = self.model.generate(**tok, **gen_kwargs)
        gen_ids = out[0, tok["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    @torch.inference_mode()
    def capture_final_answer_step(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        capture_last_k_steps: int = 1,
    ) -> StepOutput:
        if capture_last_k_steps < 1:
            raise ValueError(
                f"capture_last_k_steps must be >= 1, got {capture_last_k_steps}"
            )

        rendered = apply_chat_template_with_spans(self.tokenizer, messages)
        tok_for_offsets = self.tokenizer(
            rendered.text, return_offsets_mapping=True, add_special_tokens=False
        )
        offset_mapping = list(tok_for_offsets["offset_mapping"])

        tok = self.tokenizer(
            rendered.text, return_tensors="pt", add_special_tokens=False
        )
        input_ids = tok["input_ids"].to(self.model.device)
        attention_mask = tok["attention_mask"].to(self.model.device)

        gen_kwargs = self._build_gen_kwargs(
            temperature=temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=False,
        )
        output_ids = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs
        )
        gen_ids = output_ids[:, input_ids.shape[1] :]
        generated_text = self.tokenizer.decode(
            gen_ids[0], skip_special_tokens=True
        ).strip()

        # --- attention capture ------------------------------------------
        # We run a single forward pass of the full sequence (prompt + all
        # generated tokens except the last one) and extract the attention
        # maps for the last *k* query positions.
        saved_attn: List[torch.Tensor] = []
        num_gen = gen_ids.shape[1]
        if num_gen > 0:
            k = min(capture_last_k_steps, num_gen)
            teacher_ids = output_ids[:, :-1]
            teacher_mask = torch.ones_like(teacher_ids, device=teacher_ids.device)
            outputs = self.model(
                input_ids=teacher_ids,
                attention_mask=teacher_mask,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
            for layer_attn in outputs.attentions:
                # layer_attn: [batch, heads, seq_len, seq_len]
                saved_attn.append(
                    layer_attn[:, :, -k:, :].detach().float().cpu()
                )

        return StepOutput(
            generated_text=generated_text,
            prompt_text=rendered.text,
            input_ids=input_ids.detach().cpu(),
            output_ids=output_ids.detach().cpu(),
            attentions=saved_attn,
            offset_mapping=offset_mapping,
            message_char_spans=rendered.char_spans,
        )
