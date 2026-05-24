# TAAC2026 PCVR Solution

This repository contains an experimental solution for the TAAC2026 Tencent Ads
Algorithm Competition academic track. The task is post-click conversion-rate
prediction: given user features, target item features, and four user behavior
sequence domains, predict whether a clicked sample converts.

The code started from the official baseline and was iteratively improved through
feature engineering, training stability fixes, and model-side ablations. The
best recorded online evaluation in this repository is:

```text
AUC: 0.831588
Branch: codex/log-pair-6266-v1
Commit: ad88c74
Eval: 修改62-66pair epoch5_eval_1779544640
```

## Repository Layout

```text
.
├── run.sh              # Official platform training entry point
├── train.py            # Argument parsing, dataset/model/trainer setup
├── dataset.py          # Parquet IterableDataset and feature engineering
├── model.py            # PCVRHyFormer model and sequence/token modules
├── trainer.py          # Training loop, validation, EMA, checkpoint saving
├── utils.py            # Logging, seed setup, losses, helpers
├── eval/
│   ├── infer.py        # Official platform inference entry point
│   ├── dataset.py      # Eval-side copy of dataset code
│   └── model.py        # Eval-side copy of model code
├── ns_groups.json      # Optional semantic NS grouping example
└── AGENTS.md           # Internal project notes and platform constraints
```

The full competition data is not included. `demo_1000.parquet`, when present
locally, is only a demo/format dataset and is ignored by git.

## Final Main Configuration

The active `run.sh` uses the final best-style configuration:

- RankMixer NS tokenizer.
- Longer sequence encoder.
- `seq_a:256, seq_b:256, seq_c:1024, seq_d:1024`.
- Time-based row-group validation split.
- AMP BF16 training.
- Dense-only EMA.
- Per-epoch checkpoint snapshots.
- BCE + focal blend loss.
- Target DIN branch.
- Feature-engineering flags for:
  - pair features,
  - calendar/time features,
  - sequence time features,
  - sequence truncation summaries,
  - missing indicators,
  - aligned dense-int pair tokens.

The most important final feature change is the aligned dense-int handling for
user feature ids `62-66`: their dense values are treated as weights for the
corresponding sparse ids, transformed with `log1p`, and represented through
dedicated aligned pair tokens. Feature ids `89-91` are excluded from this pair
path because their dense vectors behave more like fixed-basis coefficient
vectors than frequency/count weights.

## Main Improvement Path

The main research path was:

```text
official baseline
→ NS output fusion
→ time split + AMP BF16
→ sequence/checkpoint infrastructure
→ special dense handling for user dense 61/87
→ domain/time feature engineering
→ dense-only EMA
→ 62-66 log-weighted dense-int pair tokens
```

High-signal findings:

- Field semantics mattered more than blindly scaling the model.
- User dense/int aligned features were the largest source of improvement.
- Time features helped, but excessive shift/session/gap variants tended to be
  redundant or unstable.
- DIN was useful in the final feature stack, but making it more complex did not
  reliably help.
- Transformer and OneTrans-style structure experiments were informative but did
  not beat the final LongerEncoder branch in this codebase.

## Training on the Official Platform

The official platform executes `run.sh` automatically. The script reads platform
environment variables through `train.py`, especially:

- `TRAIN_DATA_PATH`
- `TRAIN_CKPT_PATH`
- `TRAIN_LOG_PATH`
- `TRAIN_TF_EVENTS_PATH`

To submit a training job, upload the root-level Python files, `run.sh`,
`ns_groups.json` if desired, and the `eval/` directory required for inference.

Checkpoints are written under `TRAIN_CKPT_PATH`. The code saves sidecar metadata
such as `train_config.json`, allowing `eval/infer.py` to rebuild the exact model
structure during official evaluation.

## Local Smoke Checks

The full training data is only available on the platform, but the code can be
syntax-checked locally:

```bash
python3 -m py_compile train.py trainer.py dataset.py model.py \
    eval/infer.py eval/dataset.py eval/model.py
bash -n run.sh
diff -q dataset.py eval/dataset.py
diff -q model.py eval/model.py
```

If a local demo parquet and schema are available, `run.sh` can be adapted to run
against that local path, but local scores should not be treated as competition
scores.

## Evaluation

Official evaluation runs `eval/infer.py`. The script:

1. Loads the checkpoint from `MODEL_OUTPUT_PATH`.
2. Rebuilds the model using checkpoint sidecar configuration.
3. Reads test data from `EVAL_DATA_PATH`.
4. Writes `${EVAL_RESULT_PATH}/predictions.json`.

The output JSON has the required format:

```json
{
  "predictions": {
    "user_id": 0.1234
  }
}
```

## Git Anchors

Important branches/commits:

```text
main                         412a747  Initial official-style baseline
codex/domain-calendar-emb-v1 6d67b13  Strong domain/time feature anchor
codex/dense-ema-v1           81c6181  Dense-only EMA anchor
codex/log-pair-6266-v1       ad88c74  Final best code anchor
```

For post-competition analysis, see `AGENTS.md` and local ignored notes under
`agent_workspace/` if they are present in your working copy.

## Notes

- `eval/model.py` and `eval/dataset.py` must stay synchronized with
  `model.py` and `dataset.py`.
- The competition metric shown by the platform is AUC.
- `eval_scores.csv` was used locally to track experiments and is intentionally
  ignored by git.

