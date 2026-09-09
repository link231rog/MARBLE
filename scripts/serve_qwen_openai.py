import asyncio
import json
import os
import re
import time
import uuid
from typing import Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.environ.get("MARBLE_MODEL_PATH", "/data/home/huangzixuan/models/Qwen3.5-4B")
SERVED_MODEL_NAME = os.environ.get("MARBLE_SERVED_MODEL_NAME", "Qwen/Qwen3.5-4B")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"[SERVER] Loading base model from {MODEL_PATH} onto {DEVICE}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
).to(DEVICE)
base_model.eval()

# Detect current scale to strictly prevent loading adapters from different architecture scales
scale_env = os.environ.get("MARBLE_SCALE", "").lower()
if not scale_env:
    for s in ("0.8b", "2b", "4b", "9b"):
        if s in MODEL_PATH.lower() or s in SERVED_MODEL_NAME.lower():
            scale_env = s
            break

ADAPTER_MAP: Dict[str, str] = {}
if scale_env == "4b":
    ADAPTER_MAP = {
        "qwen_sft_v2": "/data/home/huangzixuan/MARBLE/runs/checkpoints/qwen_sft_v2",
        "qwen_sft": "/data/home/huangzixuan/MARBLE/runs/checkpoints/qwen_sft",
        "qwen_rl": "/data/home/huangzixuan/MARBLE/runs/checkpoints/qwen_rl",
        "qwen_4b_sft": "/data/home/huangzixuan/MARBLE/runs/checkpoints/controller_scaling/qwen_4b_sft",
        "qwen_4b_rl": "/data/home/huangzixuan/MARBLE/runs/checkpoints/controller_scaling/qwen_4b_rl",
    }
elif scale_env in ("0.8b", "2b", "9b"):
    ADAPTER_MAP = {
        f"qwen_{scale_env}_sft": f"/data/home/huangzixuan/MARBLE/runs/checkpoints/controller_scaling/qwen_{scale_env}_sft",
        f"qwen_{scale_env}_rl": f"/data/home/huangzixuan/MARBLE/runs/checkpoints/controller_scaling/qwen_{scale_env}_rl",
    }

ckpt_base = "/data/home/huangzixuan/MARBLE/runs/checkpoints"
if os.path.isdir(ckpt_base):
    for sub in sorted(os.listdir(ckpt_base)):
        sub_p = os.path.join(ckpt_base, sub)
        if os.path.isdir(sub_p) and os.path.isfile(os.path.join(sub_p, "adapter_config.json")):
            if sub not in ADAPTER_MAP:
                ADAPTER_MAP[sub] = sub_p

def _safe_adapter_name(name: str) -> str:
    return name.replace(".", "_")

peft_wrapper: Optional[PeftModel] = None
loaded_adapters: Dict[str, str] = {}
adapter_alias_map: Dict[str, str] = {}

for name, p in ADAPTER_MAP.items():
    if os.path.isdir(p) and os.path.isfile(os.path.join(p, "adapter_config.json")):
        safe_name = _safe_adapter_name(name)
        try:
            if peft_wrapper is None:
                print(f"[SERVER] Initializing PeftModel with adapter {safe_name!r} (alias {name!r}) from {p}...", flush=True)
                peft_wrapper = PeftModel.from_pretrained(base_model, p, adapter_name=safe_name)
            else:
                print(f"[SERVER] Loading additional adapter {safe_name!r} (alias {name!r}) from {p}...", flush=True)
                peft_wrapper.load_adapter(p, adapter_name=safe_name)
            loaded_adapters[name] = p
            loaded_adapters[safe_name] = p
            adapter_alias_map[name] = safe_name
            adapter_alias_map[safe_name] = safe_name
        except Exception as err:
            print(f"[SERVER] Warning: Failed to pre-load adapter {name!r} from {p}: {err}", flush=True)

if torch.cuda.is_available():
    vram = torch.cuda.memory_allocated() / (1024**3)
    print(f"[SERVER] Model & adapters loaded! VRAM allocated: {vram:.2f} GB", flush=True)

app = FastAPI(title="Qwen3.5 Local OpenAI Compatible Server")
generate_lock = asyncio.Lock()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = SERVED_MODEL_NAME
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    logprobs: Optional[bool] = False

@app.get("/health")
@app.get("/v1/health")
def health():
    return {
        "status": "ok",
        "base_model": SERVED_MODEL_NAME,
        "loaded_adapters": list(loaded_adapters.keys()),
    }

@app.get("/v1/models")
def list_models():
    models = [{"id": SERVED_MODEL_NAME, "object": "model", "owned_by": "qwen"}]
    for name in loaded_adapters:
        models.append({"id": name, "object": "model", "owned_by": "marble"})
    return {"object": "list", "data": models}

def _resolve_adapter(req_model: str) -> Optional[str]:
    global peft_wrapper
    if req_model in (SERVED_MODEL_NAME, "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B", ""):
        return "base"

    safe_req = _safe_adapter_name(req_model)
    if req_model in adapter_alias_map:
        target_name = adapter_alias_map[req_model]
        if peft_wrapper is not None and hasattr(peft_wrapper, "set_adapter"):
            peft_wrapper.set_adapter(target_name)
        return target_name
    if safe_req in adapter_alias_map:
        target_name = adapter_alias_map[safe_req]
        if peft_wrapper is not None and hasattr(peft_wrapper, "set_adapter"):
            peft_wrapper.set_adapter(target_name)
        return target_name

    adapter_path = ADAPTER_MAP.get(req_model) or ADAPTER_MAP.get(safe_req)
    if not adapter_path:
        search_dirs = [
            req_model,
            safe_req,
            os.path.join("/data/home/huangzixuan/MARBLE/runs/checkpoints/controller_scaling", req_model),
            os.path.join("/data/home/huangzixuan/MARBLE/runs/checkpoints/controller_scaling", safe_req),
            os.path.join("/data/home/huangzixuan/MARBLE/runs/checkpoints", req_model),
            os.path.join("/data/home/huangzixuan/MARBLE/runs/checkpoints", safe_req),
            os.path.join("/data/home/huangzixuan/MARBLE", req_model),
            os.path.join("/data/home/huangzixuan", req_model),
        ]
        for cand in search_dirs:
            if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "adapter_config.json")):
                adapter_path = cand
                break

    if not adapter_path or not os.path.isdir(adapter_path):
        raise HTTPException(
            status_code=404,
            detail=f"Adapter {req_model!r} not found! Registered adapters: {list(loaded_adapters.keys())}. Silent fallback to base model is disabled."
        )

    if safe_req not in loaded_adapters:
        if peft_wrapper is None:
            print(f"[SERVER] Initializing PeftModel with adapter {safe_req!r} from {adapter_path}...", flush=True)
            peft_wrapper = PeftModel.from_pretrained(base_model, adapter_path, adapter_name=safe_req)
        else:
            print(f"[SERVER] Loading adapter {safe_req!r} from {adapter_path}...", flush=True)
            peft_wrapper.load_adapter(adapter_path, adapter_name=safe_req)
        loaded_adapters[req_model] = adapter_path
        loaded_adapters[safe_req] = adapter_path
        adapter_alias_map[req_model] = safe_req
        adapter_alias_map[safe_req] = safe_req

    if peft_wrapper is not None and hasattr(peft_wrapper, "set_adapter"):
        peft_wrapper.set_adapter(safe_req)

    return safe_req

def _clean_output_text(text: str) -> str:
    text = text.strip()
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
        if m:
            return m.group(1).strip()
    m = re.search(r"(\{[\s\S]*\})", text)
    if m:
        try:
            json.loads(m.group(1))
            return m.group(1).strip()
        except Exception:
            pass
    return text

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    async with generate_lock:
        try:
            adapter_mode = _resolve_adapter(req.model)
            raw_msgs = [{"role": m.role, "content": m.content} for m in req.messages]
            
            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    prompt_text = tokenizer.apply_chat_template(
                        raw_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
                    )
                except Exception:
                    prompt_text = tokenizer.apply_chat_template(
                        raw_msgs, tokenize=False, add_generation_prompt=True
                    )
            else:
                prompt_text = "\n".join(f"{m['role']}: {m['content']}" for m in raw_msgs) + "\nassistant:"

            inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
            input_ids = inputs["input_ids"]
            prompt_len = input_ids.shape[1]

            do_sample = bool(req.temperature and req.temperature > 0.0)
            gen_kwargs = {
                "max_new_tokens": min(req.max_tokens or 64, 64),
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = req.temperature
                gen_kwargs["top_p"] = req.top_p
            if req.logprobs:
                gen_kwargs["return_dict_in_generate"] = True
                gen_kwargs["output_scores"] = True

            is_base = (adapter_mode == "base")
            active_model = base_model if peft_wrapper is None else peft_wrapper

            with torch.inference_mode():
                if is_base and peft_wrapper is not None:
                    with peft_wrapper.disable_adapter():
                        out_gen = active_model.generate(**inputs, **gen_kwargs)
                else:
                    if peft_wrapper is not None:
                        peft_wrapper.set_adapter(adapter_mode)
                    out_gen = active_model.generate(**inputs, **gen_kwargs)

            if hasattr(out_gen, "sequences"):
                out_ids = out_gen.sequences
                scores = getattr(out_gen, "scores", None)
            else:
                out_ids = out_gen
                scores = None

            new_tokens = out_ids[0, prompt_len:]
            raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            clean_output = _clean_output_text(raw_output)
            completion_len = len(new_tokens)

            choice_obj: Dict[str, Any] = {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": clean_output,
                },
                "finish_reason": "stop",
            }
            if req.logprobs and scores is not None:
                content_logprobs = []
                for i, score in enumerate(scores):
                    if i < len(new_tokens):
                        tok_id = int(new_tokens[i].item())
                        tok_str = tokenizer.decode([tok_id])
                        lp = float(torch.nn.functional.log_softmax(score[0], dim=-1)[tok_id].item())
                        content_logprobs.append({"token": tok_str, "logprob": lp})
                choice_obj["logprobs"] = {"content": content_logprobs}

            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [choice_obj],
                "usage": {
                    "prompt_tokens": prompt_len,
                    "completion_tokens": completion_len,
                    "total_tokens": prompt_len + completion_len,
                },
            }
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
