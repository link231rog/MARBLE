# runs_archive — 实验产物归档

本目录保存 MARBLE Phase-3 GovernedMemory 实验的可复核产物，使仓库可独立验证（评审 P3#6）。

## 产物来源（服务器 gpu1）

权威产物在 `cis_gpu1.utlab.ltd:~/MARBLE/`：

- `ckpt_sft.json` — SFT 预热后的 `LocalPolicyController` 线性权重
- `ckpt_rl.json` — RL（真实 task_score reward 回路）微调后的权重
- `runs/stageB/` — 固定基线扫描：5 基线（no_memory / global_always / private_only / heuristic / learned_controller）× coding+bargaining × train 任务 {1,2,4} = 30 episode，全部 `[ok]`
- `runs/stageE/` — 留持评估：learned_controller（ckpt_rl）在 coding test 任务 7、3 = 2 episode，task_score 0.2 / 0.55

每个任务目录含：`config.yaml` `summary.json` `memory_trace.jsonl` `reward.json` `errors.log`。

## 拉取命令（需可用 ssh）

```bash
ssh huangzixuan@cis_gpu1.utlab.ltd
# 在本地仓库根执行：
scp -r huangzixuan@cis_gpu1.utlab.ltd:~/MARBLE/runs/stageB ./runs_archive/stageB
scp -r huangzixuan@cis_gpu1.utlab.ltd:~/MARBLE/runs/stageE ./runs_archive/stageE
scp  huangzixuan@cis_gpu1.utlab.ltd:~/MARBLE/ckpt_sft.json ./runs_archive/
scp  huangzixuan@cis_gpu1.utlab.ltd:~/MARBLE/ckpt_rl.json  ./runs_archive/
```

> 注：本会话 ssh 别名不可用，物理拷贝待可用 ssh 时执行。训练权重为线性感知机（非 Qwen LoRA，见 P2#5）。

## 复现

```bash
uv run python -m marble.experiments.train_controller \
  --mode sft --traces runs_archive/stageB/*/learned_controller/*/*/memory_trace.jsonl \
  --out runs_archive/ckpt_sft.json
uv run python -m marble.experiments.train_controller \
  --mode rl --traces <同上> --rewards <对应 summary.json> \
  --init runs_archive/ckpt_sft.json --out runs_archive/ckpt_rl.json
```
