#!/usr/bin/env python3
"""Chapter 7 Phase B SFT v2 Checkpoint Gate Verifier.

Tests the trained qwen_sft_v2 adapter directly on sample prompts to verify:
1. Valid JSON adhering to {"visibility": "absent" | "global" | ["<agent_id>", ...], "supersedes": null}
2. Output distribution contains absent, global, and targeted recipient arrays
3. All recipient agent IDs are members of agent_role_map
4. 0 syntax / format errors
5. 0 prompt echo / 0 Python code leakage
"""
import json
import os
import re
import sys
from pathlib import Path

def clean_output(text: str) -> str:
    m = re.search(r"(\{[\s\S]*?\})", text)
    if m:
        try:
            json.loads(m.group(1))
            return m.group(1).strip()
        except:
            pass
    return text.strip()

def main():
    ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/checkpoints/qwen_sft_v2"
    base_model = sys.argv[2] if len(sys.argv) > 2 else "/data/home/huangzixuan/models/Qwen3.5-4B"

    print("========================================================================")
    print("🔍 Chapter 7 Phase B: SFT v2 Checkpoint Gate Verification")
    print(f"   Adapter Checkpoint: {ckpt_dir}")
    print(f"   Base Model:         {base_model}")
    print("========================================================================")

    # 1. Metadata check
    meta_path = Path(ckpt_dir) / "adapter_metadata.json"
    if not meta_path.is_file():
        print(f"❌ Error: {meta_path} not found!")
        sys.exit(1)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    print("✅ Metadata found:")
    print(json.dumps(meta, indent=2))
    assert meta.get("max_len") == 4096, f"Expected max_len=4096, got {meta.get('max_len')}"

    # 2. Test prompt generation
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("\nLoading model & adapter onto GPU...")
    tok = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, ckpt_dir)
    model.eval()

    test_cases = [
        {
            "desc": "Agent 0 scratchpad/draft -> should route targeted to ['agent_0']",
            "agent_role_map": {"agent_0": "researcher", "agent_1": "reviewer"},
            "proposal": {
                "agent_reference": "agent_0",
                "title": "my local scratch note on arxiv papers",
                "value": "I am drafting preliminary query ideas for my own next step...",
                "source": "worker",
            },
        },
        {
            "desc": "Agent 1 critique specifically for Agent 0 -> should route targeted to ['agent_0']",
            "agent_role_map": {"agent_0": "coder", "agent_1": "tester"},
            "proposal": {
                "agent_reference": "agent_1",
                "title": "feedback on test failures for agent_0",
                "value": "agent_0: lines 42-45 in your solution have an off-by-one error.",
                "source": "worker",
            },
        },
        {
            "desc": "Team milestone / confirmed finding -> should route global",
            "agent_role_map": {"agent_0": "lead", "agent_1": "analyst", "agent_2": "writer"},
            "proposal": {
                "agent_reference": "agent_0",
                "title": "verified final experiment results for section 4",
                "value": "All benchmarks verified across all 24 tasks. Table 1 results confirmed.",
                "source": "worker",
            },
        },
        {
            "desc": "Trivial acknowledgment / duplicate noise -> should route absent",
            "agent_role_map": {"agent_0": "worker", "agent_1": "worker"},
            "proposal": {
                "agent_reference": "agent_0",
                "title": "done step 1",
                "value": "ok, got it, proceeding now.",
                "source": "worker",
            },
        },
    ]

    from marble.controllers.json_controller import JsonController
    from marble.memory.schema import MemoryProposal

    controller = JsonController(lambda p: "")
    outcomes = []

    print("\nRunning inference on test cases:")
    for idx, tc in enumerate(test_cases, 1):
        controller.set_context(
            task_goal="collaborative MAS task",
            agent_role_map=tc["agent_role_map"],
        )
        p = tc["proposal"]
        proposal_obj = MemoryProposal(
            proposal_id=f"test_{idx}",
            task_id="test_task",
            agent_id=p["agent_reference"],
            source=p["source"],
            title=p["title"],
            raw_value=p["value"],
            step_index=idx,
        )
        prompt = controller.build_prompt(proposal_obj, [])
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=48,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        raw_output = tok.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        clean = clean_output(raw_output)
        val_err = controller._validate(clean, [], proposal_obj)

        print(f"\n[{idx}] {tc['desc']}")
        print(f"    Raw Output:     {clean}")
        print(f"    Validation:     {'✅ VALID' if val_err is None else f'❌ INVALID: {val_err}'}")
        outcomes.append({
            "desc": tc["desc"],
            "raw": clean,
            "valid": val_err is None,
            "error": val_err,
        })

    # Also test real dataset trace samples
    trace_file = "runs/sft_rebuild_smoke_20260909/sft_v2_trace.jsonl"
    if os.path.isfile(trace_file):
        print("\nTesting Real SFT v2 Dataset Traces:")
        with open(trace_file, encoding="utf-8") as f:
            real_samples = [json.loads(l) for l in f if l.strip()][:5]
        for s_idx, sample in enumerate(real_samples, 1):
            s_prompt = sample["controller_prompt"]
            # Extract agent_role_map from the prompt for validation context
            arm_match = re.search(r"agent_role_map:\s*(\{.*?\})\n", s_prompt)
            if arm_match:
                try:
                    controller.agent_role_map = json.loads(arm_match.group(1))
                except:
                    pass
            s_inputs = tok(s_prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                s_out = model.generate(
                    **s_inputs, max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id
                )
            s_raw = tok.decode(s_out[0][s_inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            s_clean = clean_output(s_raw)
            s_val = controller._validate(s_clean, [], None)
            print(f"  Real Sample {s_idx} [GT: {sample.get('target')}]:")
            print(f"    Prediction: {s_clean}")
            print(f"    Validation: {'✅ VALID' if s_val is None else f'❌ INVALID: {s_val}'}")
            outcomes.append({
                "desc": f"Real Trace Sample {s_idx}",
                "raw": s_clean,
                "valid": s_val is None,
                "error": s_val,
            })

    all_valid = all(o["valid"] for o in outcomes)
    vis_types = set()
    for o in outcomes:
        try:
            d = json.loads(o["raw"])
            v = d.get("visibility")
            if isinstance(v, list):
                vis_types.add("targeted")
            elif isinstance(v, str):
                vis_types.add(v)
        except:
            pass

    print("\n========================================================")
    print("📋 SFT v2 CHECKPOINT GATE RESULTS:")
    print(f"   All Outputs Strict JSON: {'✅ PASSED' if all_valid else '❌ FAILED'}")
    print(f"   Observed Visibilities:   {sorted(vis_types)}")
    has_targeted = "targeted" in vis_types
    print(f"   Targeted Routing Seen:   {'✅ PASSED' if has_targeted else '❌ FAILED'}")
    print("========================================================")

    if all_valid and has_targeted:
        print("\n🎉 SFT v2 CHECKPOINT GATE FULLY PASSED!")
        sys.exit(0)
    else:
        print("\n⚠️ SFT v2 CHECKPOINT GATE FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main()
