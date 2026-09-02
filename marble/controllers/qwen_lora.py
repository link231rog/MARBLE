"""Qwen LoRA controller, offline SFT, and trace-replay policy gradient.

The controller emits the SAME strict-JSON contract as ``JsonController`` — only
the generation backend differs. We therefore reuse ``JsonController`` directly
and hand it a Qwen-backed ``generate_fn``.

Two backends:
- API mode: an OpenAI-compatible endpoint serving a fine-tuned Qwen LoRA
  (e.g. vLLM ``--lora-modules`` or NVAPI_BASE). No local torch needed.
- Local mode: transformers + peft loading a base model with a LoRA adapter.

SFT trains a LoRA adapter from (prompt, completion) pairs exported from
decision traces. ``train_qwen_rl`` uses recorded controller completions and
episode rewards for completion-level REINFORCE.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from marble.controllers.json_controller import JsonController
from marble.llms.usage import record_successful_completion


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
        record_successful_completion(resp)
        return (resp.choices[0].message.content or "").strip()

    return gen


def local_generate_fn(
    model_path: str,
    lora_dir: Optional[str] = None,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
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
        sample = temperature > 0
        generation_kwargs = {"do_sample": sample}
        if sample:
            generation_kwargs["temperature"] = temperature
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **generation_kwargs,
        )
        return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    return gen


def make_qwen_lora_controller(
    generate_fn: Callable[[str], str],
    max_value_chars: int = 512,
    drop_fields: tuple = (),
    agent_capabilities: tuple = (),
    task_goal: str = "",
    agent_role_map: Optional[Mapping[str, str]] = None,
) -> JsonController:
    """A JsonController driven by a (fine-tuned) Qwen generate_fn."""
    return JsonController(
        generate_fn,
        max_value_chars=max_value_chars,
        drop_fields=drop_fields,
        agent_capabilities=agent_capabilities,
        task_goal=task_goal,
        agent_role_map=agent_role_map,
    )


def export_sft_pairs(
    trace_paths: List[str],
    drop_fields: tuple = (),
    agent_capabilities: tuple = (),
    task_goal: str = "",
    agent_role_map: Optional[Mapping[str, str]] = None,
) -> List[Tuple[str, str]]:
    """Build (prompt, completion) pairs from memory_decision events.

    Uses a recorded controller prompt when available. Older traces without a
    prompt snapshot fall back to reconstruction from proposal/target fields.
    """
    pairs: List[Tuple[str, str]] = []
    builder = JsonController(
        lambda p: "",
        drop_fields=drop_fields,
        agent_capabilities=agent_capabilities,
        task_goal=task_goal,
        agent_role_map=agent_role_map,
    )
    for path in trace_paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                ev = json.loads(line)
                if ev.get("event") != "memory_decision":
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
                prompt = ev.get("controller_prompt")
                if not isinstance(prompt, str) or not prompt.strip():
                    prompt = builder.build_prompt(proposal, [])
                pairs.append((prompt, completion))
    return pairs


def completion_only_example(
    tokenizer: Any,
    prompt: str,
    completion: str,
    max_len: int,
) -> Dict[str, List[int]]:
    """Encode one example while masking every prompt token from the loss."""
    if max_len < 2:
        raise ValueError("max_len must be at least 2 for prompt and completion")

    prompt_ids = list(
        tokenizer(prompt + "\n", add_special_tokens=False)["input_ids"]
    )
    completion_ids = list(
        tokenizer(completion, add_special_tokens=False)["input_ids"]
    )
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        completion_ids.append(eos_token_id)
    if not completion_ids:
        raise ValueError("completion must encode to at least one token")

    # Preserve the target whenever the full pair does not fit. Prompt context
    # can be shortened; silently dropping the supervised JSON cannot train a
    # controller.
    completion_ids = completion_ids[:max_len]
    prompt_budget = max_len - len(completion_ids)
    prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def completion_only_collator(
    features: List[Dict[str, Any]],
    pad_token_id: int,
) -> Dict[str, Any]:
    """Pad variable-length completion-only examples without supervising padding."""
    import torch

    max_batch_len = max(len(feature["input_ids"]) for feature in features)

    def pad(values: List[int], pad_value: int) -> List[int]:
        return values + [pad_value] * (max_batch_len - len(values))

    return {
        "input_ids": torch.tensor(
            [pad(feature["input_ids"], pad_token_id) for feature in features],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [pad(feature["attention_mask"], 0) for feature in features],
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            [pad(feature["labels"], -100) for feature in features],
            dtype=torch.long,
        ),
        "sample_weight": torch.tensor(
            [feature["sample_weight"] for feature in features],
            dtype=torch.float,
        ),
    }


def weighted_completion_loss(logits: Any, labels: Any, sample_weight: Any) -> Any:
    """Mean completion-token loss, weighted once per example."""
    import torch

    next_logits = logits[:, :-1, :].contiguous()
    next_labels = labels[:, 1:].contiguous()
    token_loss = torch.nn.functional.cross_entropy(
        next_logits.view(-1, next_logits.size(-1)),
        next_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(next_labels.shape)
    token_mask = next_labels.ne(-100)
    per_example = (token_loss * token_mask).sum(dim=1) / token_mask.sum(
        dim=1
    ).clamp_min(1)
    return (per_example * sample_weight).sum() / sample_weight.sum().clamp_min(1e-8)


def sequence_log_probs(logits: Any, labels: Any) -> Any:
    """Return summed completion-token log probabilities for each example."""
    import torch

    next_logits = logits[:, :-1, :].contiguous()
    next_labels = labels[:, 1:].contiguous()
    token_mask = next_labels.ne(-100)
    safe_labels = next_labels.masked_fill(~token_mask, 0)
    token_log_probs = torch.nn.functional.log_softmax(next_logits, dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    return (token_log_probs * token_mask).sum(dim=1)


def load_qwen_rl_samples(
    trace_paths: Sequence[str],
    rewards: Sequence[float],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Read replayable stored-memory decisions with formal per-memory credit."""
    if len(trace_paths) != len(rewards):
        raise ValueError("--rewards must provide one task_score per trace")

    from marble.memory.rewards import proposal_rewards

    episodes: List[Tuple[List[Dict[str, Any]], float, Tuple[str, str]]] = []
    scores_by_task: Dict[Tuple[str, str], List[float]] = {}
    for trace_path, reward in zip(trace_paths, rewards):
        with open(trace_path, encoding="utf-8") as fh:
            events = [json.loads(line) for line in fh if line.strip()]
        summary_path = Path(trace_path).with_name("summary.json")
        summary: Dict[str, Any] = {}
        if summary_path.is_file():
            with summary_path.open(encoding="utf-8") as fh:
                summary = json.load(fh)
        metadata = next(
            (event for event in events if "score_status" in event or "benchmark" in event),
            {},
        )
        if summary.get("score_status", metadata.get("score_status")) == "unavailable":
            continue
        decisions = [
            event for event in events if event.get("event") == "memory_decision"
        ]
        if not decisions:
            continue
        task_id = str(decisions[0].get("proposal", {}).get("task_id", ""))
        if not task_id:
            raise ValueError(f"trace {trace_path} has no decision task_id")
        benchmark = str(summary.get("benchmark", metadata.get("benchmark", "")))
        score = float(reward)
        task_key = (benchmark, task_id)
        episodes.append((events, score, task_key))
        scores_by_task.setdefault(task_key, []).append(score)
    undersized = sorted(task for task, scores in scores_by_task.items() if len(scores) < 2)
    if undersized:
        raise ValueError(
            "Qwen RL needs at least two comparable rollouts per task for "
            f"same-task advantage; undersized tasks: {', '.join(undersized)}"
        )

    samples: List[Dict[str, Any]] = []
    stats = {
        "traces": len(trace_paths),
        "decisions": 0,
        "samples": 0,
        "skipped_no_output": 0,
        "skipped_no_prompt": 0,
        "skipped_no_credit": 0,
    }
    for events, score, task_key in episodes:
        baseline = sum(scores_by_task[task_key]) / len(scores_by_task[task_key])
        credits = proposal_rewards(
            events, task_score=score, same_task_baseline=baseline
        )
        for event in events:
            if event.get("event") != "memory_decision":
                continue
            stats["decisions"] += 1
            memory_id = event.get("memory_id")
            credit = credits.get(memory_id)
            if credit is None:
                stats["skipped_no_credit"] += 1
                continue
            prompt = event.get("controller_prompt")
            completion = event.get("controller_output")
            if not isinstance(completion, str) or not completion.strip():
                stats["skipped_no_output"] += 1
                continue
            if not isinstance(prompt, str) or not prompt.strip():
                stats["skipped_no_prompt"] += 1
                continue
            samples.append(
                {"prompt": prompt, "completion": completion, "advantage": credit}
            )
    stats["samples"] = len(samples)
    return samples, stats


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
    if not pairs:
        raise ValueError("Qwen SFT needs at least one prompt/completion pair")
    if sample_weights is not None and len(sample_weights) != len(pairs):
        raise ValueError("sample_weights must have one value per SFT pair")
    if sample_weights is not None and any(weight < 0 for weight in sample_weights):
        raise ValueError("sample_weights must be non-negative")

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
    if tok.pad_token_id is None:
        if tok.eos_token is None:
            raise ValueError("tokenizer needs a pad_token or eos_token")
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.config.pad_token_id = tok.pad_token_id
    lora = LoraConfig(
        r=lora_r, lora_alpha=lora_r * 2, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    class _PairDS(Dataset):
        def __init__(self) -> None:
            self.examples: List[Dict[str, Any]] = []
            for index, (prompt, completion) in enumerate(pairs):
                item = completion_only_example(tok, prompt, completion, max_len)
                item["sample_weight"] = (
                    float(sample_weights[index]) if sample_weights is not None else 1.0
                )
                self.examples.append(item)

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, i: int) -> Any:
            return self.examples[i]

    class _Collator:
        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            return completion_only_collator(features, tok.pad_token_id)

    class _WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            sample_weight = inputs.pop("sample_weight")
            outputs = model(**inputs)
            loss = weighted_completion_loss(outputs.logits, labels, sample_weight)
            return (loss, outputs) if return_outputs else loss

    ds = _PairDS()
    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=1, logging_steps=1, save_strategy="no",
        report_to=[],
    )
    trainer = _WeightedTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=_Collator(),
    )
    trainer.train()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (Path(out_dir) / "adapter_metadata.json").write_text(
        json.dumps(
            {
                "base_model": base_model,
                "training_method": "completion_only_sft",
                "lora_r": lora_r,
                "max_len": max_len,
                "sample_weights": sample_weights is not None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_dir


def train_qwen_rl(
    trace_paths: Sequence[str],
    out_dir: str,
    base_model: str,
    rewards: Sequence[float],
    epochs: int = 3,
    lr: float = 1e-4,
    lora_r: int = 16,
    max_len: int = 512,
    init_checkpoint: Optional[str] = None,
    max_grad_norm: float = 1.0,
) -> str:
    """Trace-replay completion-level REINFORCE over a Qwen LoRA adapter.

    Each stored-memory action receives formal ``G_i`` credit from its trace.
    Each task needs two or more comparable traces for same-task baselines.
    Absent actions have no ``G_i`` and remain supervised by the SFT phase.
    This consumes recorded rollouts; caller must ensure they were sampled by
    the current policy before this update to claim on-policy training.
    """
    samples, stats = load_qwen_rl_samples(trace_paths, rewards)
    if not samples:
        raise ValueError(
            "Qwen RL found no replayable decisions "
            f"(skipped_no_output={stats['skipped_no_output']}, "
            f"skipped_no_prompt={stats['skipped_no_prompt']}, "
            f"skipped_no_credit={stats['skipped_no_credit']})"
        )

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("tokenizer needs a pad_token or eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16
    )
    if init_checkpoint:
        model = PeftModel.from_pretrained(model, init_checkpoint, is_trainable=True)
    else:
        lora = LoraConfig(
            r=lora_r, lora_alpha=lora_r * 2, target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
    model.config.pad_token_id = tokenizer.pad_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    class _ReplayDS(Dataset):
        def __init__(self) -> None:
            self.examples: List[Dict[str, Any]] = []
            for sample in samples:
                item = completion_only_example(
                    tokenizer, sample["prompt"], sample["completion"], max_len
                )
                item["sample_weight"] = sample["advantage"]
                self.examples.append(item)

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int) -> Dict[str, Any]:
            return self.examples[index]

    def collate(features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return completion_only_collator(features, tokenizer.pad_token_id)

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=lr)
    loader = DataLoader(_ReplayDS(), batch_size=1, shuffle=True, collate_fn=collate)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            labels = batch.pop("labels").to(device)
            advantages = batch.pop("sample_weight").to(device)
            batch = {name: value.to(device) for name, value in batch.items()}
            outputs = model(**batch)
            loss = -(advantages * sequence_log_probs(outputs.logits, labels)).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_grad_norm)
            optimizer.step()

    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    torch.save(optimizer.state_dict(), output_path / "optimizer.pt")
    metadata = {
        "base_model": base_model,
        "training_method": "trace_replay_reinforce",
        "trace_replay": True,
        "init_checkpoint": init_checkpoint,
        "lora_r": lora_r,
        "max_len": max_len,
        "max_grad_norm": max_grad_norm,
        "epochs": epochs,
        "learning_rate": lr,
        "reward_mean": sum(rewards) / len(rewards),
        "reward_count": len(rewards),
        **stats,
    }
    (output_path / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    return out_dir
