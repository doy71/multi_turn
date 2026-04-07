from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
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
        attn_implementation: Optional[str] = None,
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

    @staticmethod
    def _select_next_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        logits: [batch, vocab]
        returns: [batch, 1]
        """
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        return next_token

    @staticmethod
    def _stack_captured_attentions(
        per_step_attentions: List[List[torch.Tensor]],
        capture_last_k_steps: int,
    ) -> List[torch.Tensor]:
        """
        Convert a list of per-step attention tensors into the old output format:
        one tensor per layer with shape [batch, heads, query_steps, key_len].

        In incremental decoding the key length grows by one each step, so we
        right-pad earlier captured steps to the final key length before stacking.
        """
        if not per_step_attentions:
            return []

        total_steps = len(per_step_attentions)
        keep_from = max(0, total_steps - capture_last_k_steps)
        selected_steps = per_step_attentions[keep_from:]
        num_layers = len(selected_steps[0])

        stacked_layers: List[torch.Tensor] = []
        for layer_idx in range(num_layers):
            layer_steps = [step[layer_idx] for step in selected_steps]
            final_key_len = layer_steps[-1].shape[-1]
            padded = []
            for attn in layer_steps:
                pad = final_key_len - attn.shape[-1]
                if pad > 0:
                    attn = F.pad(attn, (0, pad))
                padded.append(attn)
            stacked_layers.append(torch.cat(padded, dim=2))
        return stacked_layers

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

        if input_ids.shape[1] < 1:
            raise ValueError("Prompt must contain at least one token.")

        # Retrieval_Head-style incremental decoding:
        # 1) prefill cache with the prompt prefix
        # 2) feed exactly one token at a time with past_key_values
        # 3) capture output_attentions=True at each real decoding step
        if input_ids.shape[1] == 1:
            past_key_values = None
            decode_input = input_ids
            running_attention_mask = attention_mask.clone()
        else:
            prefill_outputs = self.model(
                input_ids=input_ids[:, :-1],
                attention_mask=attention_mask[:, :-1],
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
            past_key_values = prefill_outputs.past_key_values
            decode_input = input_ids[:, -1:]
            running_attention_mask = attention_mask.clone()

        generated_steps: List[torch.Tensor] = []
        per_step_attentions: List[List[torch.Tensor]] = []

        eos_token_id = self.tokenizer.eos_token_id
        for _ in range(max_new_tokens):
            outputs = self.model(
                input_ids=decode_input,
                attention_mask=running_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values

            next_token = self._select_next_token(
                outputs.logits[:, -1, :], temperature=temperature
            )
            generated_steps.append(next_token.detach().cpu())

            # Store the actual attention returned at this decoding step.
            per_step_attentions.append(
                [layer_attn.detach().float().cpu() for layer_attn in outputs.attentions]
            )

            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break

            decode_input = next_token.to(self.model.device)
            next_mask = torch.ones(
                (running_attention_mask.shape[0], 1),
                dtype=running_attention_mask.dtype,
                device=running_attention_mask.device,
            )
            running_attention_mask = torch.cat(
                [running_attention_mask, next_mask], dim=1
            )

        if generated_steps:
            gen_ids = torch.cat(generated_steps, dim=1)
        else:
            gen_ids = torch.empty(
                (input_ids.shape[0], 0), dtype=input_ids.dtype, device=input_ids.device
            )

        output_ids = torch.cat([input_ids, gen_ids.to(input_ids.device)], dim=1)
        generated_text = self.tokenizer.decode(
            gen_ids[0], skip_special_tokens=True
        ).strip()

        saved_attn = self._stack_captured_attentions(
            per_step_attentions=per_step_attentions,
            capture_last_k_steps=capture_last_k_steps,
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
