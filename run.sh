#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- Active config: RankMixer NS tokenizer (no ns_groups.json required) ----
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 4 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --use_ns_output_fusion \
    --use_pair_features \
    --use_calendar_time_features \
    --use_calendar_bucket_features \
    --use_seq_time_features \
    --use_seq_domain_calendar_features \
    --use_seq_trunc_features \
    --use_missing_indicator_features \
    --use_aligned_dense_int \
    --aligned_dense_int_tokens 2 \
    --use_target_din \
    --dropout_rate 0.05 \
    --loss_type bce_focal_blend \
    --focal_alpha 0.5 \
    --focal_gamma 1.0 \
    --focal_blend_weight 0.3 \
    --seq_encoder_type longer \
    --seq_top_k 64 \
    --seq_max_lens seq_a:256,seq_b:256,seq_c:1024,seq_d:1024 \
    --split_mode time \
    --split_time_col timestamp \
    --split_time_stat median \
    --amp \
    --amp_dtype bf16 \
    --ema_decay 0.999 \
    --save_epoch_checkpoints \
    --num_workers 8 \
    "$@"

# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With d_model=64 and num_ns=12 (7 user_int + 1 user_dense + 4 item_int),
# only num_queries=1 satisfies d_model % T == 0 (T = num_queries*4 + num_ns).
# To switch, comment out the block above and uncomment the block below.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     "$@"
