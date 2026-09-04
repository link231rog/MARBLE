#!/usr/bin/env python3
import os
import glob
import json

RUN_DIR = "runs/nvidia-gpt-oss-stratified-20260904"
MANIFEST_FILE = "configs/experiments/multiagentbench_stratified_frozen.json"

def load_manifest():
    with open(MANIFEST_FILE) as f:
        return json.load(f)

def get_split_task_ids(manifest):
    splits = manifest["splits"]
    train_db = set(t["task_id"] for t in splits["train"] if t["benchmark"] == "database")
    test_db = set(t["task_id"] for t in splits["test"] if t["benchmark"] == "database")
    train_res = set(t["task_id"] for t in splits["train"] if t["benchmark"] == "research")
    test_res = set(t["task_id"] for t in splits["test"] if t["benchmark"] == "research")

    return {
        ("database", "train"): train_db,
        ("database", "test"): test_db,
        ("research", "train"): train_res,
        ("research", "test"): test_res,
    }

def main():
    manifest = load_manifest()
    split_map = get_split_task_ids(manifest)
    
    summaries = glob.glob(f"{RUN_DIR}/**/summary.json", recursive=True)
    records = []
    for s in summaries:
        try:
            with open(s) as f:
                d = json.load(f)
            records.append(d)
        except Exception:
            continue

    baselines = ["single_agent", "no_memory", "global_add_all", "lts_style", "memory_r1_style"]

    print("=========================================================================================")
    print("                 PHASE 1 STRATIFIED BENCHMARK: PASS@1 & EFFICIENCY                       ")
    print(f"Directory: {RUN_DIR}")
    print(f"Total Completed Summaries: {len(records)}")
    print("=========================================================================================\n")

    for benchmark in ["database", "research"]:
        print(f"\n==================== BENCHMARK: {benchmark.upper()} ====================")
        for split_type in ["train", "test"]:
            print(f"\n--- Split: {split_type.upper()} ---")
            header = f"{'Method':<18} | {'Done':<5} | {'Mean Score':<10} | {'Pass@1 SR':<11} | {'Avg Tokens':<10} | {'Tok/Pass@1':<12} | {'Net Reward':<10}"
            print(header)
            print("-" * len(header))

            target_all = split_map[(benchmark, split_type)]

            for b in baselines:
                b_records = [r for r in records if r.get("method") == b and r.get("benchmark") == benchmark and r.get("task_id") in target_all]
                done_count = len(b_records)
                total_target = len(target_all)

                if done_count == 0:
                    print(f"{b:<18} | {done_count:>2}/{total_target:<2} | {'N/A':<10} | {'N/A':<11} | {'N/A':<10} | {'N/A':<12} | {'N/A':<10}")
                    continue

                scores = [r.get("task_score", 0.0) or 0.0 for r in b_records]
                successes = [1 if r.get("task_success") else 0 for r in b_records]
                tokens = [r.get("total_tokens", 0) or 0 for r in b_records]
                rewards = [r.get("episode_reward", 0.0) or 0.0 for r in b_records]

                mean_score = sum(scores) / done_count
                sr = (sum(successes) / done_count) * 100.0
                avg_tok = sum(tokens) / done_count
                tok_per_success = (sum(tokens) / sum(successes)) if sum(successes) > 0 else float("inf")
                mean_reward = sum(rewards) / done_count

                tok_succ_str = f"{tok_per_success/1000:.1f}k" if tok_per_success != float("inf") else "inf"

                print(f"{b:<18} | {done_count:>2}/{total_target:<2} | {mean_score:<10.3f} | {sr:>5.1f}%     | {avg_tok/1000:>6.1f}k   | {tok_succ_str:<12} | {mean_reward:<10.4f}")

if __name__ == "__main__":
    main()
