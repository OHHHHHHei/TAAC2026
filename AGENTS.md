# TAAC2026 Code Context

## Project Goal

This repository is a TAAC2026 PCVR baseline. The task is post-click
conversion-rate prediction: given user features, target item features, and four
behavior-sequence domains, train a binary classifier that predicts whether a
clicked sample converts. The training label is derived from `label_type == 2`.

The competition platform trains and evaluates inside the official environment.
The local data file is `demo_1000.parquet`, a format/demo dataset rather than
the full competition training dataset.

## Current Repository State

The git tree has been reorganized for post-competition review:

- `baseline/original`: original platform-style baseline, commit `412a747`.
- `baseline-original`: lightweight tag pointing to the same original baseline.
- `main`: current best validated mainline, commit `8d142e2`.
- `codex/log-pair-6266-v1`: historical branch for the best experiment line,
  also at `8d142e2`.
- `codex/*` and `exp/*`: side experiment branches. Branches that did not beat
  the current mainline remain as side branches and should not be merged into
  `main` unless their platform eval result is later confirmed to improve the
  best score.
- `eval_scores.csv`: local CSV file used to record platform eval results. It is
  intentionally ignored by git.

The intended workflow is:

1. Keep `main` as the best confirmed solution line.
2. Start each new experiment from `main`.
3. Merge back only experiments with confirmed platform AUC improvement.
4. Keep failed or ambiguous experiments as side branches for review.

Sparse embedding restart behavior in the current code:

- `reinit_cardinality_threshold=0` means reset almost all non-empty sparse
  embeddings after each eligible epoch.
- The restart happens after validation and best-checkpoint saving, so it does
  not corrupt the just-saved best checkpoint.

## Platform Constraints

- The official training platform executes root-level `run.sh` automatically.
- The platform provides `USER_CACHE_PATH` as a 20GB user cache path shared by
  training and evaluation; it is an important resource for reusable artifacts
  such as preprocessing caches or feature statistics.
- The official training stage provides `TRAIN_DATA_PATH`, `TRAIN_CKPT_PATH`,
  `TRAIN_TF_EVENTS_PATH`, and related log/cache environment variables.
- Training code must save model checkpoints under `TRAIN_CKPT_PATH` for the
  platform to recognize them.
- Iterative checkpoint directories must be prefixed with `global_step`; names
  must be at most 300 characters and may contain letters, numbers,
  underscores, hyphens, equal signs, and periods.
- The platform TensorBoard view reads event files from `TRAIN_TF_EVENTS_PATH`
  and supports scalar metrics.
- The official eval path loads one checkpoint directory and writes
  `predictions.json`.
- The official evaluation stage provides `MODEL_OUTPUT_PATH`,
  `EVAL_DATA_PATH`, `EVAL_RESULT_PATH`, and `EVAL_INFER_PATH` environment
  variables.
- The official inference script must be named `infer.py` and define a
  zero-argument `main()` function.
- Evaluation output must be written to
  `${EVAL_RESULT_PATH}/predictions.json`.
- `predictions.json` must contain a top-level `predictions` mapping from
  test-set `user_id` strings to predicted conversion probabilities in `[0, 1]`.
- The platform eval records observed so far show one reported AUC value per eval
  job.
- `eval/infer.py` rebuilds the model from checkpoint sidecar files, especially
  `train_config.json`, and loads `model.pt` with `strict=True`.
- The training and eval trees contain separate copies of model/data code:
  `model.py` with `eval/model.py`, and `dataset.py` with `eval/dataset.py`.

## Important Files

- `run.sh`: platform training entry point.
- `train.py`: argument parsing, data loading, model construction, trainer setup.
- `dataset.py`: Parquet IterableDataset for user/item features and sequence
  domains.
- `model.py`: `PCVRHyFormer` model, sequence encoders, RankMixer, NS tokenizers.
- `trainer.py`: BCE/Focal training loop, validation AUC/logloss, best-checkpoint
  saving, sparse embedding restart.
- `utils.py`: logger, seed setup, EarlyStopping, focal loss.
- `eval/infer.py`: platform inference script.
- `eval/dataset.py`, `eval/model.py`: eval-side copies of training-side
  data/model code.
- `ns_groups.json`: example semantic grouping for NS tokens. Current active
  `run.sh` disables it with `--ns_groups_json ""`.

## Current Best Configuration

Active `run.sh` on `main` uses:

- RankMixer NS tokenizer with `--ns_groups_json ""`.
- `--user_ns_tokens 4`, `--item_ns_tokens 2`, `--num_queries 2`.
- Longer sequence encoder with `--seq_top_k 64`.
- `--seq_max_lens seq_a:256,seq_b:256,seq_c:1024,seq_d:1024`.
- Row-group time split by `timestamp` median.
- AMP BF16.
- Dense-only EMA with `--ema_decay 0.999`.
- Per-epoch checkpoint snapshots.
- BCE + focal blend loss.
- Target DIN.
- Calendar/time, sequence-time, sequence-truncation, missing-indicator, pair,
  and aligned dense-int features.

The best confirmed online result is:

- AUC `0.831588`
- Eval job `129141`
- Eval name `修改62-66pair epoch5_eval_1779544640`
- Code anchor `ad88c74` for the core model/data change
- Current documented mainline `8d142e2`

## Observed Platform Results

Observed platform AUC examples:

- `2026-04-26`, job `28899`, `Test_eval_1777175638028`, baseline, best
  checkpoint: `0.810704`.
- `2026-04-27`, job `30988`, `item_id_best_eval_1777259958985`,
  `item_id_hash_1m`, best checkpoint: `0.810335`.
- `2026-04-27`, job `30298`, `item_id_eval_1777227860671`,
  `item_id_hash_1m`, middle checkpoint: `0.811296`.

The platform eval UI reports the metric as `auc`.

Observed validation-curve screenshots:

- With platform-style per-epoch sparse embedding restart enabled, validation AUC
  stayed near `0.86` on the displayed run.
- With delayed/no early sparse embedding restart, validation AUC moved from about
  `0.86` down to about `0.71-0.74` on the displayed run.

## Implementation Notes

- `LongerEncoder` contains top-k/padding alignment logic for short sequences.
- Demo sequence timestamps are descending; list heads are the most recent
  actions in the inspected demo data.
- `RotaryEmbedding` defaults to `max_seq_len=2048`.
- `action_num > 1` is exposed in argument help text, while the trainer uses a
  binary BCE-style label path.
- `demo_1000.parquet` has one row group.

## Repository Hygiene

- `.gitignore` excludes local data/cache/output artifacts such as
  `demo_1000.parquet`, `__pycache__/`, logs, checkpoints, and event files.
- When a change modifies model/data/training logic, create a git anchor commit
  so the exploration path can be reconstructed. Pure `run.sh` parameter trials
  may be tracked in `eval_scores.csv` without a new code commit.
- Verification commands used for the current structural train/eval changes:

```bash
python3 -m py_compile train.py trainer.py dataset.py model.py eval/infer.py eval/dataset.py eval/model.py
diff -q dataset.py eval/dataset.py
diff -q model.py eval/model.py
bash -n run.sh
```
