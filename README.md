# TAAC2026 PCVR 方案复盘

这个仓库是 TAAC2026 腾讯广告算法大赛学术赛道的一个 PCVR
方案。任务是点击后转化率预测：给定用户特征、目标 item 特征，以及
四个行为序列 domain，预测该点击样本是否会转化。训练标签来自
`label_type == 2`。

本项目从官方 baseline 出发，经过多轮特征工程、训练稳定性优化和模型
结构消融，最终线上最好成绩为：

```text
AUC: 0.831588
分支: codex/log-pair-6266-v1
提交: ad88c74
评估任务: 修改62-66pair epoch5_eval_1779544640
```

## 目录结构

```text
.
├── run.sh              # 官方训练平台入口脚本
├── train.py            # 参数解析、数据集、模型和 trainer 构建
├── dataset.py          # Parquet IterableDataset 与特征工程
├── model.py            # PCVRHyFormer 模型、序列编码器、tokenizer
├── trainer.py          # 训练循环、验证、EMA、checkpoint 保存
├── utils.py            # 日志、随机种子、loss、工具函数
├── eval/
│   ├── infer.py        # 官方评估平台推理入口
│   ├── dataset.py      # 评估侧数据代码副本
│   └── model.py        # 评估侧模型代码副本
├── ns_groups.json      # 可选 NS 语义分组配置示例
└── AGENTS.md           # 项目背景、平台约束和实验上下文
```

完整比赛训练数据不包含在仓库中。本地如果存在 `demo_1000.parquet`，它只是
官方提供的格式/demo 数据集，不代表完整训练集，并且被 git 忽略。

## 最终主线配置

当前 `run.sh` 对应最终 best 风格配置，主要包括：

- RankMixer NS tokenizer。
- Longer sequence encoder。
- 序列长度：`seq_a:256, seq_b:256, seq_c:1024, seq_d:1024`。
- 按 row group 时间中位数做验证集切分。
- AMP BF16 训练。
- Dense-only EMA。
- 每个 epoch 保存 checkpoint。
- BCE + focal blend loss。
- Target DIN 分支。
- 特征工程：
  - pair features；
  - calendar/time features；
  - sequence time features；
  - sequence truncation summary；
  - missing indicator；
  - aligned dense-int pair tokens。

最终最关键的特征改动是：只对 user feature id `62-66` 做 aligned
dense-int pair，并对 dense 权重做 `log1p` 变换；`89-91` 不再进入 pair
路径，而是保留在 raw dense 中。这个改动对应当前最高线上 AUC `0.831588`。

## 从 Baseline 到 Best 的主线

整体上分路径可以概括为：

```text
官方 baseline
→ NS output fusion
→ time split + AMP BF16
→ checkpoint / 训练稳定性建设
→ user dense 61/87 特殊处理
→ domain/calendar/time 特征工程
→ dense-only EMA
→ 62-66 log-weighted dense-int pair tokens
```

核心经验：

- 最大收益来自理解字段语义，而不是盲目加大模型。
- user dense/int 对齐特征是最关键的涨分来源。
- 时间特征有收益，但过多 shift/session/gap/window 特征容易冗余或过拟合。
- DIN 在最终特征栈里有贡献，但继续复杂化没有稳定收益。
- Transformer、OneTrans、pyramid、tokenizer 顺序调整等结构实验有信息量，
  但没有超过最终 LongerEncoder 主线。

## 官方平台训练

官方训练平台会自动执行根目录下的 `run.sh`。训练代码通过环境变量读取平台
路径，主要包括：

- `TRAIN_DATA_PATH`
- `TRAIN_CKPT_PATH`
- `TRAIN_LOG_PATH`
- `TRAIN_TF_EVENTS_PATH`

提交训练任务时，需要上传根目录的 Python 文件、`run.sh`、必要时的
`ns_groups.json`，以及 `eval/` 目录。

模型 checkpoint 会保存到 `TRAIN_CKPT_PATH`。训练时会同时保存
`train_config.json` 等 sidecar 文件，评估阶段会用这些信息重建完全一致的
模型结构。

## 本地检查

完整训练依赖官方平台数据。本地主要用于语法检查和小规模 smoke test。

常用检查命令：

```bash
python3 -m py_compile train.py trainer.py dataset.py model.py \
    eval/infer.py eval/dataset.py eval/model.py
bash -n run.sh
diff -q dataset.py eval/dataset.py
diff -q model.py eval/model.py
```

如果本地有 demo parquet 和 schema，可以临时改路径跑通流程，但 demo 分数
不能代表线上效果。

## 官方评估

官方评估使用 `eval/infer.py`。推理流程为：

1. 从 `MODEL_OUTPUT_PATH` 读取 checkpoint。
2. 根据 checkpoint 中的 sidecar config 重建模型。
3. 从 `EVAL_DATA_PATH` 读取测试数据。
4. 写出 `${EVAL_RESULT_PATH}/predictions.json`。

输出格式：

```json
{
  "predictions": {
    "user_id": 0.1234
  }
}
```

平台最终展示的评估指标是 AUC。

## 关键 Git 锚点

```text
main                         412a747  官方风格初始 baseline
codex/domain-calendar-emb-v1 6d67b13  强时间特征锚点
codex/dense-ema-v1           81c6181  Dense-only EMA 锚点
codex/log-pair-6266-v1       ad88c74  最终 best 代码锚点
```

当前 best 分支是：

```text
codex/log-pair-6266-v1 @ ad88c74
```

## 注意事项

- `model.py` 与 `eval/model.py` 必须保持同步。
- `dataset.py` 与 `eval/dataset.py` 必须保持同步。
- `eval_scores.csv` 是本地实验记录文件，被 git 忽略。
- `demo_1000.parquet` 是 demo 数据，被 git 忽略。
- 后续如果继续实验，建议从 `codex/log-pair-6266-v1` 拉新分支。

