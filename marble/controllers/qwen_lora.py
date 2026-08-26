"""Qwen LoRA controller + offline SFT (spec §9.3, review P2#5).

The controller emits the SAME strict-JSON contract as ``JsonController`` — only
the generation backend differs. We therefore reuse ``JsonController`` directly
and hand it a Qwen-backed ``generate_fn``.

Two backends:
- API mode: an OpenAI-compatible endpoint serving a fine-tuned Qwen LoRA
  (e.g. vLLM ``--lora-modules`` or NVAPI_BASE). No local torch needed.
- Local mode: transformers + peft loading a base model with a LoRA adapter.

SFT trains a LoRA adapter from (prompt, completion) pairs exported from
decision traces. RL reuses SFT weighted by episode task_score (lightweight
stand-in for full LM policy-gradient; documented as such).
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from marble.controllers.json_controller import JsonController


def api_generate_fn(
    api_base: str,
    api_key: str,
    model: str,
    max_tokens: int = 256,
) -> Callable[[str], str]:
    from openai import OpenAI

    client = OpenAI(base_url=api_base, api_key=api_key)

    def gen(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()

    return gen


def local_generate_fn(
    model_path: str,
    lora_dir: Optional[str] = None,
    max_new_tokens: int = 64,
) -> Callable[[str], str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if lora_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora_dir)
    model.eval()

    def gen(prompt: str) -> str:
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0
        )
        return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    return gen


def make_qwen_lora_controller(
    generate_fn: Callable[[str], str],
    max_value_chars: int = 512,
    drop_fields: tuple = (),
    agent_capabilities: tuple = (),
    task_goal: str = "",
) -> JsonController:
    """A JsonController driven by a (fine-tuned) Qwen generate_fn."""
    return JsonController(
        generate_fn,
        max_value_chars=max_value_chars,
        drop_fields=drop_fields,
        agent_capabilities=agent_capabilities,
        task_goal=task_goal,
    )


def export_sft_pairs(
    trace_paths: List[str],
    drop_fields: tuple = (),
    agent_capabilities: tuple = (),
    task_goal: str = "",
) -> List[Tuple[str, str]]:
    """Build (prompt, completion) pairs from memory_decision events.

    Active-memory context is omitted (traces store per-decision snapshots, not
    full state); the model learns proposal -> visibility, which is the core task.
    """
    pairs: List[Tuple[str, str]] = []
    builder = JsonController(
        lambda p: "",
        drop_fields=drop_fields,
        agent_capabilities=agent_capabilities,
        task_goal=task_goal,
    )
    for path in trace_paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                ev = json.loads(line)
                if ev.get("event") != "memory_decision" or not ev.get("memory_id"):
                    continue
                p = ev["proposal"]
                from marble.memory.schema import MemoryProposal

                proposal = MemoryProposal(
                    proposal_id=p["proposal_id"], task_id=p["task_id"],
                    agent_id=p["agent_id"], source=p.get("source", "worker"),
                    title=p["title"], raw_value=p["raw_value"],
                    step_index=p.get("step_index", 0),
                )
                tgt = ev["target"]
                completion = json.dumps(
                    {"visibility": tgt["visibility"], "supersedes": tgt.get("supersedes")}
                )
                pairs.append((builder.build_prompt(proposal, []), completion))
    return pairs


def train_qwen_sft(
    pairs: List[Tuple[str, str]],
    out_dir: str,
    base_model: str,
    epochs: int = 3,
    lr: float = 1e-4,
    lora_r: int = 16,
    max_len: int = 512,
    sample_weights: Optional[List[float]] = None,
) -> str:
    """Real LoRA SFT over (prompt, completion) pairs using peft + transformers."""
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    lora = LoraConfig(
        r=lora_r, lora_alpha=lora_r * 2, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    class _PairDS(Dataset):
        def __init__(self) -> None:
            self.examples: List[Any] = []
            for prompt, completion in pairs:
                text = prompt + "\n" + completion
                ids = tok(text, truncation=True, max_length=max_len, return_tensors="pt")
                self.examples.append(ids.input_ids.squeeze(0))

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, i: int) -> Any:
            return {"input_ids": self.examples[i], "labels": self.examples[i]}

    ds = _PairDS()
    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=1, logging_steps=1, save_strategy="no",
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds)
    trainer.train()
    model.save_pretrained(out_dir)
    return out_dir


def train_qwen_rl(
    trace_paths: List[str],
    out_dir: str,
    base_model: str,
    rewards: Optional[List[float]] = None,
    epochs: int = 3,
    lr: float = 1e-4,
    lora_r: int = 16,
) -> str:
    """RL stand-in: SFT weighted by episode task_score (one weight per trace)."""
    pairs = export_sft_pairs(trace_paths)
    weights: Optional[List[float]] = None
    if rewards:
        per_trace: List[int] = []
        for path in trace_paths:
            n = sum(
                1 for line in open(path, encoding="utf-8")
                if (ev := json.loads(line)).get("event") == "memory_decision"
                and ev.get("memory_id")
            )
            per_trace.append(n)
        weights = []
        for score, n in zip(rewards, per_trace):
            weights.extend([max(score, 0.0)] * n)
    return train_qwen_sft(
        pairs, out_dir, base_model, epochs=epochs, lr=lr, lora_r=lora_r,
        sample_weights=weights,
    )
