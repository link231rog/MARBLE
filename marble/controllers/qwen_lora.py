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
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from marble.controllers.json_controller import JsonController
from marble.llms.usage import record_successful_completion


def api_generate_fn(
    api_base: str,
    api_key: str,
    model: str,
    max_tokens: int = 256,
    timeout: Optional[float] = None,
    temperature: float = 0.0,
) -> Callable[[str], str]:
    import os

    from openai import OpenAI

    eff_timeout = (
        timeout
        if timeout is not None
        else float(os.environ.get("MARBLE_CONTROLLER_TIMEOUT", "180.0"))
    )
    eff_temperature = float(os.environ.get("MARBLE_CONTROLLER_TEMPERATURE", str(temperature)))
    client = OpenAI(base_url=api_base, api_key=api_key, timeout=eff_timeout)
    disable_thinking = (
        "siliconflow" in api_base
        or os.environ.get("MARBLE_CONTROLLER_DISABLE_THINKING", "") in ("1", "true", "True")
    )

    import logging
    import time

    logger = logging.getLogger(__name__)

    def gen(prompt: str) -> str:
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": min(max_tokens, 64),
            "temperature": eff_temperature,
            "logprobs": True,
        }
        if disable_thinking:
            kwargs["extra_body"] = {"enable_thinking": False}

        max_retries = int(os.environ.get("MARBLE_CONTROLLER_MAX_RETRIES", "5"))
        base_delay = float(os.environ.get("MARBLE_CONTROLLER_RETRY_DELAY", "2.0"))

        last_err: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                resp = client.chat.completions.create(**kwargs)
                record_successful_completion(resp)
                choice = resp.choices[0]
                content = (choice.message.content or "").strip()
                if content.startswith("```"):
                    lines = content.splitlines()
                    if lines and lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    content = "\n".join(lines).strip()

                total_log_prob = 0.0
                has_logprobs = False
                if hasattr(choice, "logprobs") and choice.logprobs:
                    content_logprobs = getattr(choice.logprobs, "content", None) or []
                    if content_logprobs:
                        total_log_prob = sum(
                            float(t.logprob)
                            for t in content_logprobs
                            if getattr(t, "logprob", None) is not None
                        )
                        has_logprobs = True
                gen.last_log_prob = float(total_log_prob) if has_logprobs else None
                return content
            except Exception as exc:
                last_err = exc
                err_msg = str(exc).lower()
                if "model does not exist" in err_msg or "20012" in err_msg or "model_not_found" in err_msg:
                    logger.error(
                        "Controller API fatal error: requested model '%s' does not exist on endpoint %s! (%s)",
                        model, api_base, exc
                    )
                    raise last_err
                if attempt < max_retries:
                    sleep_s = base_delay * (2 ** attempt)
                    logger.warning(
                        "Controller API attempt %d failed (%s); retrying in %.1fs...",
                        attempt + 1, exc, sleep_s
                    )
                    time.sleep(sleep_s)
                else:
                    raise last_err
        gen.last_log_prob = None
        return ""

    gen.last_log_prob = None
    return gen



def _hf_local_files_only() -> bool:
    """Return True unless MARBLE_ALLOW_HF_DOWNLOAD=1 is explicitly set.

    Guards against accidental multi-gigabyte weight downloads on local dev machines.
    """
    return os.environ.get("MARBLE_ALLOW_HF_DOWNLOAD") != "1"



def local_generate_fn(
    model_path: str,
    lora_dir: Optional[str] = None,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
) -> Callable[[str], str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local_only = _hf_local_files_only()
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=local_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=local_only,
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
            return_dict_in_generate=True,
            output_scores=True,
            **generation_kwargs,
        )
        gen_tokens = out.sequences[0][inputs.input_ids.shape[1]:]
        text = tok.decode(gen_tokens, skip_special_tokens=True).strip()
        total_lp = 0.0
        if hasattr(out, "scores") and out.scores:
            for i, score in enumerate(out.scores):
                if i < len(gen_tokens):
                    lp = torch.nn.functional.log_softmax(score[0], dim=-1)[gen_tokens[i]].item()
                    total_lp += lp
            gen.last_log_prob = float(total_lp)
        else:
            gen.last_log_prob = None
        return text

    gen.last_log_prob = None
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
                vis = tgt.get("visibility")
                if vis == "targeted":
                    recipients = list(tgt.get("target_recipients") or [])
                    vis_val = recipients if recipients else [proposal.agent_id]
                elif vis == "private":
                    recipients = list(tgt.get("target_recipients") or [])
                    vis_val = recipients if recipients else "private"
                else:
                    vis_val = vis
                update_key = "update" if "update" in tgt else "supersedes"
                update_val = tgt.get("update") if "update" in tgt else tgt.get("supersedes")
                completion = json.dumps(
                    {"visibility": vis_val, update_key: update_val}
                )
                prompt = ev.get("controller_prompt")
                if not isinstance(prompt, str) or not prompt.strip():
                    prompt = builder.build_prompt(proposal, [])
                pairs.append((prompt, completion))
    return pairs


def audit_sft_distribution(pairs: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
    """Audit distribution of SFT action labels without artificial quota forcing (spec §12.1 rule 8, §12.3 R11)."""
    counts = {"absent": 0, "private": 0, "targeted": 0, "global": 0, "invalid": 0}
    for _, completion in pairs:
        try:
            parsed = json.loads(completion)
            vis = parsed.get("visibility")
            if vis in ("absent", "global"):
                counts[vis] += 1
            elif isinstance(vis, list) and len(vis) > 0:
                counts["targeted"] += 1
                counts["private"] += 1
            elif vis in ("private", "targeted"):
                counts["targeted"] += 1
                counts["private"] += 1
            else:
                counts["invalid"] += 1
        except Exception:
            counts["invalid"] += 1
    total = len(pairs)
    proportions = {
        k: round(v / total, 4) if total > 0 else 0.0
        for k, v in counts.items()
    }
    return {
        "total_pairs": total,
        "counts": counts,
        "proportions": proportions,
    }


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
    # controller. Crucially, preserve the head containing [SYSTEM] instructions
    # and action space schema rather than discarding them.
    completion_ids = completion_ids[:max_len]
    prompt_budget = max_len - len(completion_ids)
    prompt_ids = prompt_ids[:prompt_budget] if prompt_budget else []
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
            [feature.get("sample_weight", 1.0) for feature in features],
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

    labels = labels.to(logits.device)
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
    advantage_mode: str = "task_grpo",
    beta: float | None = None,
    lambda_: float | None = None,
    target_bonus: float | None = None,
    target_card_budget: int | None = None,
    gamma_density: float | None = None,
    harmful_penalty: float | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Read replayable memory decisions with verified GRPO group advantage or shaped credit.

    advantage_mode options:
      - 'task_grpo' (default, strict Chapter 4 §5.1):
          A_e = (S_e - mean(S_group)) / (std(S_group) + eps)
          Assigned to all decisions in trajectory e. Correctly rewards valid absent
          decisions in winning trajectories and penalizes unhelpful actions in failing ones.
      - 'hybrid':
          A_e,d = task_advantage + credit
          Blends group task advantage baseline with collaboration credit shaping bonus.
      - 'credit':
          A_e,d = credit (legacy proposal credit)
    """
    if len(trace_paths) != len(rewards):
        raise ValueError("--rewards must provide one task_score per trace")

    from marble.memory.rewards import proposal_rewards

    episodes: List[Tuple[List[Dict[str, Any]], float, Tuple[str, str], Dict[str, Any]]] = []
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
        task_id = str(
            summary.get("task_id")
            or metadata.get("task_id")
            or (decisions[0].get("proposal", {}) or {}).get("task_id")
            or decisions[0].get("task_id", "")
        )
        if not task_id:
            raise ValueError(f"trace {trace_path} has no decision task_id")
        benchmark = str(
            summary.get("benchmark")
            or metadata.get("benchmark")
            or decisions[0].get("benchmark", "")
        )
        score = float(reward)
        task_key = (benchmark, task_id)
        episodes.append((events, score, task_key, summary))
        scores_by_task.setdefault(task_key, []).append(score)
    undersized = sorted(task for task, scores in scores_by_task.items() if len(scores) < 2)
    if undersized:
        raise ValueError(
            "Qwen RL needs at least two comparable rollouts per task for "
            f"same-task advantage; undersized tasks: {', '.join(f'{b}:{t}' for b, t in undersized)}"
        )

    group_stats: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for task_key, group_scores in scores_by_task.items():
        mean_s = sum(group_scores) / len(group_scores)
        var_s = sum((s - mean_s) ** 2 for s in group_scores) / len(group_scores)
        std_s = var_s ** 0.5
        group_stats[task_key] = (mean_s, std_s)

    samples: List[Dict[str, Any]] = []
    stats = {
        "traces": len(trace_paths),
        "decisions": 0,
        "samples": 0,
        "skipped_no_output": 0,
        "skipped_no_prompt": 0,
        "skipped_no_credit": 0,
    }
    for events, score, task_key, summary in episodes:
        mean_s, std_s = group_stats[task_key]
        # Chapter 4 §5.1: Group Advantage Normalization: A_e = (S_e - mean) / (std + eps)
        norm_advantage = (score - mean_s) / max(std_s, 1e-8) if std_s > 0 else 0.0

        raw_success = summary.get("task_success")
        if raw_success is not None:
            task_success = bool(raw_success) if not isinstance(raw_success, (int, float)) else (float(raw_success) >= 0.5)
        else:
            task_success = (score >= 0.5)

        credits = proposal_rewards(
            events,
            task_score=score,
            advantage=norm_advantage,
            same_task_baseline=mean_s,
            task_success=task_success,
            beta=0.25 if beta is None else beta,
            lambda_=0.05 if lambda_ is None else lambda_,
            target_bonus=0.35 if target_bonus is None else target_bonus,
            gamma_density=0.15 if gamma_density is None else gamma_density,
            target_card_budget=8 if target_card_budget is None else target_card_budget,
            harmful_penalty=0.25 if harmful_penalty is None else harmful_penalty,
        )
        for event in events:
            if event.get("event") != "memory_decision":
                continue
            stats["decisions"] += 1
            if event.get("parse_status") in ("format_error", "schema_error"):
                stats["skipped_format_error"] = stats.get("skipped_format_error", 0) + 1
                continue
            pid = event.get("proposal", {}).get("proposal_id")
            memory_id = event.get("memory_id")
            credit = credits.get(pid) if pid else None
            if credit is None and memory_id:
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
            old_lp = event.get("old_log_prob") if "old_log_prob" in event else event.get("controller_log_prob")

            # Calculate effective advantage based on mode
            if advantage_mode == "task_grpo":
                eff_adv = float(norm_advantage)
            elif advantage_mode == "hybrid":
                eff_adv = float(norm_advantage) + float(credit)
            else:  # credit
                eff_adv = float(credit)

            sample_entry: Dict[str, Any] = {
                "prompt": prompt,
                "completion": completion,
                "advantage": eff_adv,
                "task_advantage": float(norm_advantage),
                "credit": float(credit),
                "benchmark": task_key[0],
                "task_id": task_key[1],
            }
            if old_lp is not None:
                try:
                    sample_entry["old_log_prob"] = float(old_lp)
                except (TypeError, ValueError):
                    pass
            samples.append(sample_entry)
    stats["samples"] = len(samples)
    return samples, stats


def train_qwen_sft(
    pairs: List[Tuple[str, str]],
    out_dir: str,
    base_model: str,
    epochs: int = 3,
    lr: float = 1e-4,
    lora_r: int = 16,
    max_len: int = 4096,
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
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    local_only = _hf_local_files_only()
    tok = AutoTokenizer.from_pretrained(base_model, local_files_only=local_only)
    if tok.pad_token_id is None:
        if tok.eos_token is None:
            raise ValueError("tokenizer needs a pad_token or eos_token")
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=local_only,
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
        remove_unused_columns=False,
        report_to=[],
        seed=42,
        data_seed=42,
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
    epochs: int = 1,
    lr: float = 1e-4,
    lora_r: int = 16,
    max_len: int = 4096,
    init_checkpoint: Optional[str] = None,
    sft_reference_checkpoint: Optional[str] = None,
    max_grad_norm: float = 1.0,
    seed: Optional[int] = None,
    kl_coeff: float = 0.05,
    clip_eps: float = 0.2,
    group_size: int = 4,
    advantage_mode: str = "hybrid",
    beta: float = 0.75,
    lambda_: float = 0.005,
    target_bonus: float = 0.50,
    target_card_budget: int = 20,
    gamma_density: float = 0.02,
    harmful_penalty: float = 0.05,
) -> str:
    """Trajectory-level GRPO over Qwen LoRA adapter (spec Chapter 4 §4-§6).

    - Group-relative standardized advantage: A_i = (r_i - mean(r_group)) / (std(r_group) + eps)
    - Clipped policy-ratio loss: -min(r * A, clip(r, 1-eps, 1+eps) * A)
    - Relative KL penalty vs frozen SFT reference policy: D_KL(pi_theta || pi_SFT)
    - epochs=1 on fresh rollouts without multi-epoch replay drift.
    """
    samples, stats = load_qwen_rl_samples(
        trace_paths,
        rewards,
        advantage_mode=advantage_mode,
        beta=beta,
        lambda_=lambda_,
        target_bonus=target_bonus,
        target_card_budget=target_card_budget,
        gamma_density=gamma_density,
        harmful_penalty=harmful_penalty,
    )
    if not samples:
        raise ValueError(
            "Qwen RL found no replayable decisions "
            f"(skipped_no_output={stats['skipped_no_output']}, "
            f"skipped_no_prompt={stats['skipped_no_prompt']}, "
            f"skipped_no_credit={stats['skipped_no_credit']})"
        )

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    local_only = _hf_local_files_only()
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=local_only)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("tokenizer needs a pad_token or eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, local_files_only=local_only
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
    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception:
            pass
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    examples: List[Dict[str, Any]] = []
    for sample in samples:
        item = completion_only_example(
            tokenizer, sample["prompt"], sample["completion"], max_len
        )
        item["advantage"] = float(sample["advantage"])
        item["old_log_prob"] = sample.get("old_log_prob")
        item["benchmark"] = sample.get("benchmark", "")
        item["task_id"] = sample.get("task_id", "")
        examples.append(item)

    # 1. Ensure exact token-level alignment for old_log_prob under rollout policy.
    # When init_checkpoint is provided, always evaluate log \pi_{old}(completion | prompt)
    # directly over the exact masked label tokens to guarantee 100% token-by-token alignment
    # with curr_log_prob (eliminating markdown fence / whitespace mismatches).
    compute_aligned_old_lp = bool(init_checkpoint) or any(ex.get("old_log_prob") is None for ex in examples)
    if compute_aligned_old_lp:
        model.eval()
        with torch.no_grad():
            for ex in examples:
                in_ids = torch.tensor([ex["input_ids"]], dtype=torch.long, device=device)
                attn = torch.tensor([ex["attention_mask"]], dtype=torch.long, device=device)
                lbls = torch.tensor([ex["labels"]], dtype=torch.long, device=device)
                out = model(input_ids=in_ids, attention_mask=attn)
                if hasattr(out, "logits"):
                    ex["old_log_prob"] = float(sequence_log_probs(out.logits, lbls).item())
                else:
                    ex["old_log_prob"] = 0.0
                del in_ids, attn, lbls, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 2. Compute ref_log_prob under frozen SFT reference policy (Chapter 4 §6)
    ref_ckpt = sft_reference_checkpoint or init_checkpoint
    has_distinct_ref = bool(
        ref_ckpt and init_checkpoint and str(Path(ref_ckpt).resolve()) != str(Path(init_checkpoint).resolve())
    )
    if has_distinct_ref:
        if not hasattr(model, "load_adapter"):
            raise RuntimeError(
                f"Model architecture does not support load_adapter for SFT reference checkpoint: {ref_ckpt}"
            )
        try:
            model.load_adapter(ref_ckpt, adapter_name="sft_ref")
            model.set_adapter("sft_ref")
            model.eval()
            with torch.no_grad():
                for ex in examples:
                    in_ids = torch.tensor([ex["input_ids"]], dtype=torch.long, device=device)
                    attn = torch.tensor([ex["attention_mask"]], dtype=torch.long, device=device)
                    lbls = torch.tensor([ex["labels"]], dtype=torch.long, device=device)
                    out = model(input_ids=in_ids, attention_mask=attn)
                    if hasattr(out, "logits"):
                        ex["ref_log_prob"] = float(sequence_log_probs(out.logits, lbls).item())
                    else:
                        raise RuntimeError(f"Reference policy forward pass returned no logits for checkpoint: {ref_ckpt}")
                    del in_ids, attn, lbls, out
            model.set_adapter("default")
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error(
                "FATAL: Failed to load or evaluate SFT reference checkpoint '%s': %s", ref_ckpt, exc
            )
            raise RuntimeError(
                f"Failed to load or evaluate SFT reference checkpoint '{ref_ckpt}': {exc}. "
                "Aborting training to prevent unanchored KL drift."
            ) from exc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        for ex in examples:
            ex["ref_log_prob"] = float(ex["old_log_prob"])

    def grpo_collate(features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_batch_len = max(len(feature["input_ids"]) for feature in features)
        def pad(values: List[int], pad_value: int) -> List[int]:
            return values + [pad_value] * (max_batch_len - len(values))
        return {
            "input_ids": torch.tensor([pad(f["input_ids"], tokenizer.pad_token_id) for f in features], dtype=torch.long),
            "attention_mask": torch.tensor([pad(f["attention_mask"], 0) for f in features], dtype=torch.long),
            "labels": torch.tensor([pad(f["labels"], -100) for f in features], dtype=torch.long),
            "advantage": torch.tensor([f["advantage"] for f in features], dtype=torch.float),
            "old_log_prob": torch.tensor([f["old_log_prob"] for f in features], dtype=torch.float),
            "ref_log_prob": torch.tensor([f["ref_log_prob"] for f in features], dtype=torch.float),
        }

    # Group examples strictly by (benchmark, task_id) for true Group Batching
    task_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for ex in examples:
        key = (ex.get("benchmark", ""), str(ex.get("task_id", "")))
        task_groups.setdefault(key, []).append(ex)

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=lr)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import random
    model.train()
    chunk_size = int(os.environ.get("MARBLE_RL_CHUNK_SIZE", "1"))
    for _ in range(epochs):
        group_keys = list(task_groups.keys())
        if seed is not None:
            random.seed(seed)
        random.shuffle(group_keys)

        for g_key in group_keys:
            g_examples = task_groups[g_key]
            total_decisions = len(g_examples)
            if total_decisions == 0:
                continue

            optimizer.zero_grad()
            for c_idx in range(0, total_decisions, chunk_size):
                chunk = g_examples[c_idx : c_idx + chunk_size]
                batch = grpo_collate(chunk)
                labels = batch.pop("labels").to(device)
                advantages = batch.pop("advantage").to(device)
                old_log_probs = batch.pop("old_log_prob").to(device)
                ref_log_probs = batch.pop("ref_log_prob").to(device)
                batch = {name: value.to(device) for name, value in batch.items()}
                outputs = model(**batch)
                if hasattr(outputs, "logits"):
                    curr_log_probs = sequence_log_probs(outputs.logits, labels).to(device)
                    # Policy ratio r_{e,d} = exp(curr_log_prob - old_log_prob)
                    ratio = torch.exp(curr_log_probs - old_log_probs)
                    # Clipped surrogate loss: -min(r * A, clip(r, 1-eps, 1+eps) * A)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
                    policy_loss = -torch.minimum(surr1, surr2).sum() / total_decisions
                    # Non-negative reverse KL divergence: exp(log_ref - log_curr) - (log_ref - log_curr) - 1
                    log_diff = ref_log_probs - curr_log_probs
                    kl_div = torch.exp(log_diff) - log_diff - 1.0
                    loss = policy_loss + kl_coeff * (kl_div.sum() / total_decisions)
                else:
                    loss = torch.tensor(0.0, requires_grad=True, device=device)

                loss.backward()
                del loss, outputs, batch, labels, advantages, old_log_probs, ref_log_probs

            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_grad_norm)
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    torch.save(optimizer.state_dict(), output_path / "optimizer.pt")
    metadata = {
        "base_model": base_model,
        "training_method": "trajectory_grpo",
        "advantage_mode": advantage_mode,
        "beta": beta,
        "lambda": lambda_,
        "target_bonus": target_bonus,
        "target_card_budget": target_card_budget,
        "gamma_density": gamma_density,
        "harmful_penalty": harmful_penalty,
        "group_batching": True,
        "groups": len(task_groups),
        "group_size": group_size,
        "kl_anchor": True,
        "kl_coeff": kl_coeff,
        "clip_eps": clip_eps,
        "epochs": epochs,
        "init_checkpoint": init_checkpoint,
        "sft_reference_checkpoint": ref_ckpt,
        "lora_r": lora_r,
        "max_len": max_len,
        "max_grad_norm": max_grad_norm,
        "learning_rate": lr,
        "reward_mean": sum(rewards) / len(rewards),
        "reward_count": len(rewards),
        **stats,
    }
    (output_path / "adapter_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (output_path / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    return out_dir
