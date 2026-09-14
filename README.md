# π0.5 CALVIN LoRA 微调与跨场景闭环验证

这是基于 OpenPI 的个人实验项目：将 π0.5 VLA 适配到 CALVIN，并验证 CALVIN
ABC 训练场景到 D 场景的连续任务执行能力。

## 项目成果

使用 `pi05_libero` 作为初始化权重，在 CALVIN ABC 示范数据上完成 30,000
step LoRA SFT，并在相同的 500 条 CALVIN-D 五任务序列上与原始
`pi05_libero` 做对照。

| 指标 | 微调 π0.5 | 原始 pi05_libero | 提升 |
|---|---:|---:|---:|
| 平均完成任务数 | 1.926 | 0.002 | +1.924 |
| 至少完成 1 个任务 | 71.6% | 0.2% | +71.4 个百分点 |
| 至少完成 2 个任务 | 48.0% | 0.0% | +48.0 个百分点 |
| 至少完成 3 个任务 | 32.6% | 0.0% | +32.6 个百分点 |
| 至少完成 4 个任务 | 23.6% | 0.0% | +23.6 个百分点 |
| 连续完成 5 个任务 | 16.8%（84/500） | 0.0%（0/500） | +16.8 个百分点 |

评测条件：`seed=0`、500 条官方 D 场景序列、`replan_steps=5`，两组实验使用
相同的 CALVIN 相机输入、动作语义和归一化统计。

## 成功案例视频

以下 4 组视频来自同一批早期 100 条序列：左侧是微调模型完整完成 5 个任务
的成功回放，右侧是原始 `pi05_libero` 在相同序列上的失败回放。它们用于
直观展示微调前后的行为差异；500 条统计结果仍以 JSON 报告为准。

| 序列 | 任务链 | 微调成功 | 原始失败 |
|---:|---|---|---|
| 0 | lift blue block from slider → place in slider → turn on lightbulb → open drawer → push pink block left | [MP4](artifacts/calvin_d_500/videos/success_sequence_0000.mp4) | [MP4](artifacts/calvin_d_500/videos/original_failures_100/rejected/sequence_0000_attempt_01_0of5.mp4) |
| 16 | lift red block from slider → place in drawer → move slider left → turn on LED → close drawer | [MP4](artifacts/calvin_d_500/videos/success_sequence_0016.mp4) | [MP4](artifacts/calvin_d_500/videos/original_failures_100/rejected/sequence_0016_attempt_01_0of5.mp4) |
| 17 | push red block left → turn on lightbulb → open drawer → lift blue block from table → place in drawer | [MP4](artifacts/calvin_d_500/videos/success_sequence_0017.mp4) | [MP4](artifacts/calvin_d_500/videos/original_failures_100/rejected/sequence_0017_attempt_01_0of5.mp4) |
| 20 | lift red block from slider → place in drawer → turn on LED → move slider right → lift red block from drawer | [MP4](artifacts/calvin_d_500/videos/success_sequence_0020.mp4) | [MP4](artifacts/calvin_d_500/videos/original_failures_100/rejected/sequence_0020_attempt_01_0of5.mp4) |

这 8 个视频严格使用同一批 100 条序列中的相同 index 和任务链。由于策略推理
含随机采样，它们是可视化案例，不替代 500 条统计评测。

## 原始模型失败案例的复现

配对视频来自 100 条序列测试。若需重新生成同一批失败回放，先启动原始模型
服务，再用 `num_sequences=100` 生成结果，最后录制 `0,16,17,20`：

```bash
bash examples/calvin/run_pi05_libero_calvin_policy_server.sh
bash examples/calvin/run_calvin_eval.sh \
  --num-sequences 100 --seed 0 --replan-steps 5 \
  --observation-profile calvin --output outputs/calvin_eval/pi05_libero/results_original_100.json
bash examples/calvin/run_calvin_record_successes.sh \
  --results outputs/calvin_eval/pi05_libero/results_original_100.json \
  --sequence-indices 0,16,17,20 --min-completed 0 --max-completed 5 \
  --output-dir artifacts/calvin_d_500/videos/original_failures_100
```

## 代码与复现

- [CALVIN 微调与评测说明](examples/calvin/README_FINETUNE.md)
- [500 条对比 Markdown 报告](artifacts/calvin_d_500/comparison_finetuned_vs_pi05_libero_500.md)
- [完整对比 JSON](artifacts/calvin_d_500/comparison_finetuned_vs_pi05_libero_500.json)
- [成果目录说明](artifacts/calvin_d_500/README.md)

主要适配内容包括：

- CALVIN LeRobot v3 数据读取与 OpenPI transform 适配
- 15 维机器人状态和 7 维相对末端动作映射
- 10-step Action Chunk 与 episode 边界处理
- CALVIN 动作/状态归一化统计
- CALVIN D 场景闭环评测、任务 oracle 判定和 MP4 回放
- 微调权重与原始 `pi05_libero` 的配对成功率比较
