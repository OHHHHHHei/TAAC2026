"""PCVR Parquet dataset module (performance-tuned).

Reads raw multi-column Parquet directly and obtains feature metadata from
``schema.json``.

Optimizations:
- Pre-allocated numpy buffers to eliminate ``np.zeros`` + ``np.stack`` overhead.
- Fused padding loop over sequence domains that writes directly into a 3D buffer.
- Pre-computed column-index lookup to avoid per-row string lookups.
- ``file_system`` tensor-sharing strategy to work around ``/dev/shm`` exhaustion
  when using many DataLoader workers.
"""

import os
import logging
import random
import json
import gc
import glob as _glob

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing
from torch.utils.data import IterableDataset, DataLoader
from typing import Any, Dict, Iterator, List, Optional, Tuple

# numpy.typing is available since numpy >= 1.20; on older numpy fall back to a
# no-op shim so that forward-referenced annotations like ``npt.NDArray[np.int64]``
# keep working as plain strings without raising at import time.
try:
    import numpy.typing as npt  # noqa: F401
except ImportError:  # pragma: no cover
    class _NptFallback:  # type: ignore[no-redef]
        NDArray = Any

    npt = _NptFallback()  # type: ignore[assignment]


# ─────────────────────────── Feature Schema ──────────────────────────────────


class FeatureSchema:
    """Records ``(feature_id, offset, length)`` for each feature so downstream
    code can locate the segment of the flattened tensor that belongs to a
    specific feature id.

    For int features:
      - int_value: length = 1
      - int_array: length = array length
      - int_array_and_float_array: int part length
    For dense features:
      - float_value: length = 1
      - float_array: length = array length
      - int_array_and_float_array: float part length
    """

    def __init__(self) -> None:
        # Ordered list of (feature_id, offset, length).
        self.entries: List[Tuple[int, int, int]] = []
        self.total_dim: int = 0
        # Quick lookup from fid to its (offset, length).
        self._fid_to_entry: Dict[int, Tuple[int, int]] = {}

    def add(self, feature_id: int, length: int) -> None:
        """Append a feature to the schema."""
        offset = self.total_dim
        self.entries.append((feature_id, offset, length))
        self._fid_to_entry[feature_id] = (offset, length)
        self.total_dim += length

    def get_offset_length(self, feature_id: int) -> Tuple[int, int]:
        """Get ``(offset, length)`` for a feature_id."""
        return self._fid_to_entry[feature_id]

    @property
    def feature_ids(self) -> List[int]:
        """Return all feature_ids in their insertion order."""
        return [fid for fid, _, _ in self.entries]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (for JSON dumping)."""
        return {
            'entries': self.entries,
            'total_dim': self.total_dim,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FeatureSchema':
        """Reconstruct a :class:`FeatureSchema` from its dict form."""
        schema = cls()
        for fid, offset, length in d['entries']:
            schema.entries.append((fid, offset, length))
            schema._fid_to_entry[fid] = (offset, length)
        schema.total_dim = d['total_dim']
        return schema

    def __repr__(self) -> str:
        lines = [f"FeatureSchema(total_dim={self.total_dim}, features=["]
        for fid, offset, length in self.entries:
            lines.append(f"  fid={fid}: offset={offset}, length={length}")
        lines.append("])")
        return "\n".join(lines)

# Use filesystem-based tensor sharing (instead of /dev/shm) to avoid running
# out of shared memory when many DataLoader workers are active.
torch.multiprocessing.set_sharing_strategy('file_system')


def _collect_row_groups(data_dir: str) -> List[Tuple[str, int, int]]:
    """Collect ``(file_path, row_group_index, num_rows)`` in file order."""
    pq_files = sorted(_glob.glob(os.path.join(data_dir, '*.parquet')))
    rg_info: List[Tuple[str, int, int]] = []
    for f in pq_files:
        pf = pq.ParquetFile(f)
        for i in range(pf.metadata.num_row_groups):
            rg_info.append((f, i, pf.metadata.row_group(i).num_rows))
    return rg_info


def _row_group_time_value(
    file_path: str,
    row_group_idx: int,
    time_col: str,
    stat: str,
) -> float:
    """Read one time column from a Row Group and return its summary value."""
    pf = pq.ParquetFile(file_path)
    table = pf.read_row_group(row_group_idx, columns=[time_col])
    values = table.column(time_col).to_numpy(zero_copy_only=False)
    values = values[~np.isnan(values)] if np.issubdtype(values.dtype, np.floating) else values
    if len(values) == 0:
        return 0.0

    if stat == 'median':
        return float(np.median(values))
    if stat == 'mean':
        return float(np.mean(values))
    if stat == 'min':
        return float(np.min(values))
    if stat == 'max':
        return float(np.max(values))
    raise ValueError(f"Unknown split_time_stat={stat!r}")


def _order_row_groups(
    rg_info: List[Tuple[str, int, int]],
    split_mode: str,
    split_time_col: str,
    split_time_stat: str,
) -> List[Tuple[str, int, int]]:
    """Return Row Groups ordered for train/valid splitting."""
    if split_mode == 'row_group':
        return rg_info
    if split_mode != 'time':
        raise ValueError(f"Unknown split_mode={split_mode!r}")

    timed = []
    for pos, (f, rg_idx, n_rows) in enumerate(rg_info):
        t = _row_group_time_value(f, rg_idx, split_time_col, split_time_stat)
        timed.append((t, pos, f, rg_idx, n_rows))
    timed.sort(key=lambda x: (x[0], x[1]))
    return [(f, rg_idx, n_rows) for _, _, f, rg_idx, n_rows in timed]

# Time-delta bucket boundaries (64 edges -> 65 buckets: 0=padding, 1..64).
BUCKET_BOUNDARIES = np.array([
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    120, 180, 240, 300, 360, 420, 480, 540, 600,
    900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
    5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600,
    32400, 43200, 54000, 64800, 75600, 86400,
    172800, 259200, 345600, 432000, 518400, 604800,
    1123200, 1641600, 2160000, 2592000,
    4320000, 6048000, 7776000,
    11664000, 15552000,
    31536000,
], dtype=np.int64)

# Total number of time-bucket embedding slots (= number of boundaries + 1, with
# padding=0 included).
#
# This constant is uniquely determined by the length of BUCKET_BOUNDARIES; on
# the model side, ``nn.Embedding(num_embeddings=NUM_TIME_BUCKETS)`` must match
# this value exactly, otherwise an IndexError may be raised at runtime.
#
# That is why ``train.py`` / ``infer.py`` only expose the boolean flag
# ``--use_time_buckets`` and derive the concrete bucket count from here.
NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES) + 1

# High-confidence item-history intersections observed on the demo parquet. Each
# spec creates five dense features: hit_any, hit_count_norm, hit_recency_score,
# hit_in_recent_20, hit_in_recent_100.
PAIR_FEATURE_SPECS: Tuple[Tuple[int, str, int], ...] = (
    (81, 'seq_a', 46),
    (81, 'seq_c', 32),
    (81, 'seq_a', 40),
    (13, 'seq_d', 25),
    (13, 'seq_a', 40),
    (81, 'seq_d', 17),
    (81, 'seq_c', 33),
    (13, 'seq_c', 32),
    (81, 'seq_b', 68),
    (81, 'seq_d', 25),
    (13, 'seq_a', 46),
    (13, 'seq_b', 68),
    (81, 'seq_b', 75),
    (83, 'seq_b', 75),
    (13, 'seq_a', 41),
    (83, 'seq_d', 25),
    (83, 'seq_b', 68),
    (83, 'seq_a', 41),
    (9, 'seq_b', 75),
    (9, 'seq_a', 46),
    (9, 'seq_d', 25),
    (83, 'seq_a', 46),
    (83, 'seq_c', 32),
    (83, 'seq_a', 40),
    (83, 'seq_c', 33),
    (83, 'seq_d', 17),
    (9, 'seq_a', 40),
    (9, 'seq_a', 41),
    (9, 'seq_b', 68),
    (9, 'seq_c', 32),
    (9, 'seq_c', 33),
    (9, 'seq_d', 17),
)
PAIR_FEATURES_PER_SPEC = 5
PAIR_TIME_FEATURES_PER_SPEC = 4
PAIR_FEATURE_ID_BASE = 900000
PAIR_RECENCY_CAP_SECONDS = float(BUCKET_BOUNDARIES[-1])
PAIR_RECENT_WINDOWS: Tuple[int, int] = (20, 100)
PAIR_TIME_WINDOWS_SECONDS: Tuple[int, int, int, int] = (
    1800,    # 30 minutes
    7200,    # 2 hours
    86400,   # 1 day
    604800,  # 7 days
)

CALENDAR_TIME_FEATURE_ID = 910000
CALENDAR_TIME_FEATURE_DIM = 7
LOCAL_TIME_OFFSET_SECONDS = 8 * 3600

CALENDAR_BUCKET_FEATURES: Tuple[Tuple[str, int, int], ...] = (
    ('hour', 940000, 24),
    ('weekday', 940001, 7),
    ('hour_weekday', 940002, 168),
    ('ten_minute', 940003, 144),
)

SEQ_LEN_BUCKET_BOUNDARIES = np.array([
    1, 2, 3, 5, 8, 13, 20, 32, 50,
    80, 128, 192, 256, 384, 512, 768, 1024,
], dtype=np.int64)
SEQ_TIME_BUCKET_FEATURE_ID_BASE = 950000
SEQ_TIME_BUCKET_FEATURES: Tuple[Tuple[str, int], ...] = (
    ('last_recency', len(BUCKET_BOUNDARIES)),
    ('full_span', len(BUCKET_BOUNDARIES)),
    ('valid_len', len(SEQ_LEN_BUCKET_BOUNDARIES)),
)

SEQ_TIME_FEATURE_ID_BASE = 920000
SEQ_TIME_FEATURES_PER_DOMAIN = 7
SEQ_TIME_RECENT_WINDOWS: Tuple[int, int, int] = (3600, 86400, 604800)

SEQ_TRUNC_FEATURE_ID_BASE = 930000
SEQ_TRUNC_FEATURES_PER_DOMAIN = 6
SEQ_TRUNC_LEN_CAP_MULT = 4

MISSING_INDICATOR_FEATURE_ID = 990000
TYPED_MISSING_INDICATOR_FEATURE_ID = 990001
INT_TYPED_MISSING_FEATURES_PER_SPEC = 3
DENSE_TYPED_MISSING_FEATURES_PER_SPEC = 2
TYPED_MISSING_SUMMARY_DIM = 8


def _bucketize_with_padding(
    values: "npt.NDArray[np.int64]",
    boundaries: "npt.NDArray[np.int64]",
    valid_mask: "npt.NDArray[np.bool_]",
) -> "npt.NDArray[np.int64]":
    """Map nonnegative values to 1-based buckets while preserving 0 padding."""
    out = np.zeros_like(values, dtype=np.int64)
    if not valid_mask.any():
        return out
    raw = np.searchsorted(boundaries, values[valid_mask])
    raw = np.clip(raw, 0, len(boundaries) - 1)
    out[valid_mask] = raw.astype(np.int64) + 1
    return out


class PCVRParquetDataset(IterableDataset):
    """PCVR dataset that reads raw multi-column Parquet directly.

    - int features: scalar or list (multi-hot); values <= 0 are mapped to 0 (padding).
    - dense features: ``list<float>``, variable-length padded up to ``max_dim``.
    - sequence features: ``list<int64>``, grouped by domain; includes side-info
      columns and an optional timestamp column (used for time-bucketing).
    - label: mapped from ``label_type == 2``.
    """

    def __init__(
        self,
        parquet_path: str,
        schema_path: str,
        batch_size: int = 256,
        seq_max_lens: Optional[Dict[str, int]] = None,
        shuffle: bool = True,
        buffer_batches: int = 20,
        row_group_range: Optional[Tuple[int, int]] = None,
        row_group_list: Optional[List[Tuple[str, int, int]]] = None,
        clip_vocab: bool = True,
        is_training: bool = True,
        use_pair_features: bool = False,
        use_pair_time_features: bool = False,
        use_calendar_time_features: bool = False,
        use_calendar_bucket_features: bool = False,
        use_seq_time_bucket_features: bool = False,
        use_seq_time_features: bool = False,
        use_seq_trunc_features: bool = False,
        use_missing_indicator_features: bool = False,
        use_typed_missing_indicator_features: bool = False,
        use_missing_sparse_buckets: bool = False,
    ) -> None:
        """
        Args:
            parquet_path: either a directory containing ``*.parquet`` files or
                a single parquet file path.
            schema_path: path of the schema JSON describing feature layouts.
            batch_size: fixed batch size used for the pre-allocated buffers.
            seq_max_lens: optional per-domain override of sequence truncation,
                e.g. ``{'seq_d': 256}``. Domains not listed fall back to the
                schema default of 256.
            shuffle: whether to shuffle within a ``buffer_batches``-sized window.
            buffer_batches: shuffle buffer size in units of batches.
            row_group_range: ``(start, end)`` slice of Row Groups; ``None`` to
                use all Row Groups.
            row_group_list: explicit Row Groups to read. When provided, this
                takes precedence over ``row_group_range``.
            clip_vocab: if True, clip out-of-bound ids to 0; if False, raise.
            is_training: if True, derive ``label`` from ``label_type == 2``;
                if False, return an all-zeros label column.
            use_pair_features: append item-history pair match statistics to
                ``user_dense_feats``.
            use_pair_time_features: append target-aware real-time-window match
                flags for each pair feature spec.
            use_calendar_time_features: append cyclic sample-level calendar
                features derived from the row ``timestamp`` to
                ``user_dense_feats``.
            use_calendar_bucket_features: append low-cardinality sample-level
                calendar buckets derived from the row ``timestamp`` to
                ``user_int_feats``.
            use_seq_time_bucket_features: append low-cardinality per-domain
                sequence time summary buckets to ``user_int_feats``.
            use_seq_time_features: append per-domain sequence timestamp
                summary features to ``user_dense_feats``.
            use_seq_trunc_features: append per-domain raw length and truncation
                summary features to ``user_dense_feats``.
            use_missing_indicator_features: append binary dense flags that
                distinguish raw missing/invalid non-sequence features from
                normal padding index 0.
            use_typed_missing_indicator_features: append finer-grained dense
                missing flags for absent/short, non-positive, OOB, and dense
                zero-like states, plus compact ratio summaries.
            use_missing_sparse_buckets: map raw missing/invalid non-sequence
                int ids to the per-feature spare bucket ``vocab_size`` instead
                of padding 0, letting embeddings learn the invalid state.
        """
        super().__init__()

        # Accept either a directory or a single file path.
        if os.path.isdir(parquet_path):
            import glob
            files = sorted(glob.glob(os.path.join(parquet_path, '*.parquet')))
            if not files:
                raise FileNotFoundError(f"No .parquet files in {parquet_path}")
            self._parquet_files = files
        else:
            self._parquet_files = [parquet_path]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_batches = buffer_batches
        self.clip_vocab = clip_vocab
        self.is_training = is_training
        self.use_pair_features = use_pair_features
        self.use_pair_time_features = use_pair_time_features
        self.use_calendar_time_features = use_calendar_time_features
        self.use_calendar_bucket_features = use_calendar_bucket_features
        self.use_seq_time_bucket_features = use_seq_time_bucket_features
        self.use_seq_time_features = use_seq_time_features
        self.use_seq_trunc_features = use_seq_trunc_features
        self.use_missing_indicator_features = use_missing_indicator_features
        self.use_typed_missing_indicator_features = use_typed_missing_indicator_features
        self.use_missing_sparse_buckets = use_missing_sparse_buckets
        self._pair_feature_specs = list(PAIR_FEATURE_SPECS) if use_pair_features else []
        self._pair_features_per_spec = (
            PAIR_FEATURES_PER_SPEC
            + (PAIR_TIME_FEATURES_PER_SPEC if use_pair_time_features else 0)
        )
        # Out-of-bound statistics:
        #   {(group, col_idx): {'count': N, 'max': M, 'min_oob': M, 'vocab': V}}
        self._oob_stats: Dict[Tuple[str, int], Dict[str, int]] = {}

        # Build the list of Row Groups.
        if row_group_list is not None:
            self._rg_list = list(row_group_list)
        else:
            self._rg_list = []
            for f in self._parquet_files:
                pf = pq.ParquetFile(f)
                for i in range(pf.metadata.num_row_groups):
                    self._rg_list.append((f, i, pf.metadata.row_group(i).num_rows))

            if row_group_range is not None:
                start, end = row_group_range
                self._rg_list = self._rg_list[start:end]

        self.num_rows = sum(r[2] for r in self._rg_list)

        # Load schema.json.
        self._load_schema(schema_path, seq_max_lens or {})

        # ---- Pre-compute column index lookup ----
        pf = pq.ParquetFile(self._parquet_files[0])
        schema_names = pf.schema_arrow.names
        self._col_idx = {name: i for i, name in enumerate(schema_names)}

        # ---- Pre-allocate numpy buffers ----
        B = batch_size
        self._buf_user_int = np.zeros((B, self.user_int_schema.total_dim), dtype=np.int64)
        self._buf_item_int = np.zeros((B, self.item_int_schema.total_dim), dtype=np.int64)
        self._buf_user_dense = np.zeros((B, self.user_dense_schema.total_dim), dtype=np.float32)
        self._buf_seq = {}
        self._buf_seq_tb = {}
        self._buf_seq_ts = {}
        self._buf_seq_lens = {}
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            n_feats = len(self.sideinfo_fids[domain])
            self._buf_seq[domain] = np.zeros((B, n_feats, max_len), dtype=np.int64)
            self._buf_seq_tb[domain] = np.zeros((B, max_len), dtype=np.int64)
            self._buf_seq_ts[domain] = np.zeros((B, max_len), dtype=np.int64)
            self._buf_seq_lens[domain] = np.zeros(B, dtype=np.int64)

        # ---- Pre-compute (col_idx, offset, vocab_size) plans for int columns ----
        self._user_int_plan = []  # [(col_idx, dim, offset, vocab_size), ...]
        offset = 0
        for fid, vs, dim in self._user_int_cols:
            ci = self._col_idx.get(f'user_int_feats_{fid}')
            self._user_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._item_int_plan = []
        offset = 0
        for fid, vs, dim in self._item_int_cols:
            ci = self._col_idx.get(f'item_int_feats_{fid}')
            self._item_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._user_dense_plan = []
        offset = 0
        for fid, dim in self._user_dense_cols:
            ci = self._col_idx.get(f'user_dense_feats_{fid}')
            self._user_dense_plan.append((ci, dim, offset))
            offset += dim

        self._missing_indicator_plan = []
        if self.use_missing_indicator_features:
            out_offset = self._missing_indicator_offset
            for fid, vs, dim in self._user_int_cols:
                ci = self._col_idx.get(f'user_int_feats_{fid}')
                self._missing_indicator_plan.append(('int', ci, dim, vs, out_offset))
                out_offset += 1
            for fid, vs, dim in self._item_int_cols:
                ci = self._col_idx.get(f'item_int_feats_{fid}')
                self._missing_indicator_plan.append(('int', ci, dim, vs, out_offset))
                out_offset += 1
            for fid, dim in self._user_dense_cols:
                ci = self._col_idx.get(f'user_dense_feats_{fid}')
                self._missing_indicator_plan.append(('dense', ci, dim, 0, out_offset))
                out_offset += 1

        self._typed_missing_indicator_plan = []
        if self.use_typed_missing_indicator_features:
            out_offset = self._typed_missing_indicator_offset
            for fid, vs, dim in self._user_int_cols:
                ci = self._col_idx.get(f'user_int_feats_{fid}')
                self._typed_missing_indicator_plan.append(
                    ('user_int', ci, dim, vs, out_offset))
                out_offset += INT_TYPED_MISSING_FEATURES_PER_SPEC
            for fid, vs, dim in self._item_int_cols:
                ci = self._col_idx.get(f'item_int_feats_{fid}')
                self._typed_missing_indicator_plan.append(
                    ('item_int', ci, dim, vs, out_offset))
                out_offset += INT_TYPED_MISSING_FEATURES_PER_SPEC
            for fid, dim in self._user_dense_cols:
                ci = self._col_idx.get(f'user_dense_feats_{fid}')
                self._typed_missing_indicator_plan.append(
                    ('user_dense', ci, dim, 0, out_offset))
                out_offset += DENSE_TYPED_MISSING_FEATURES_PER_SPEC

        # Sequence column plan: {domain: ([(col_idx, feat_slot, vocab_size), ...], ts_col_idx)}
        self._seq_plan = {}
        for domain in self.seq_domains:
            prefix = self._seq_prefix[domain]
            sideinfo_fids = self.sideinfo_fids[domain]
            ts_fid = self.ts_fids[domain]
            side_plan = []
            for slot, fid in enumerate(sideinfo_fids):
                ci = self._col_idx.get(f'{prefix}_{fid}')
                vs = self.seq_vocab_sizes[domain][fid]
                side_plan.append((ci, slot, vs))
            ts_ci = self._col_idx.get(f'{prefix}_{ts_fid}') if ts_fid is not None else None
            self._seq_plan[domain] = (side_plan, ts_ci)

        self._build_pair_feature_plan()

        logging.info(
            f"PCVRParquetDataset: {self.num_rows} rows from "
            f"{len(self._parquet_files)} file(s), batch_size={batch_size}, "
            f"buffer_batches={buffer_batches}, shuffle={shuffle}")

    def _load_schema(self, schema_path: str, seq_max_lens: Dict[str, int]) -> None:
        """Populate per-group schema information from ``schema_path``."""
        with open(schema_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # ---- user_int: [[fid, vocab_size, dim], ...] ----
        self._user_int_cols: List[List[int]] = raw['user_int']
        self.user_int_schema: FeatureSchema = FeatureSchema()
        self.user_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._user_int_cols:
            self.user_int_schema.add(fid, dim)
            self.user_int_vocab_sizes.extend([vs] * dim)
        self._calendar_bucket_plan: List[Tuple[str, int, int]] = []
        if self.use_calendar_bucket_features:
            for name, fid, vocab_size in CALENDAR_BUCKET_FEATURES:
                offset = self.user_int_schema.total_dim
                self.user_int_schema.add(fid, 1)
                self.user_int_vocab_sizes.append(vocab_size)
                self._calendar_bucket_plan.append((name, offset, vocab_size))
            logging.info(
                "Calendar bucket features enabled: %d user_int fids %s",
                len(CALENDAR_BUCKET_FEATURES),
                [name for name, _, _ in CALENDAR_BUCKET_FEATURES],
            )

        # ---- item_int ----
        self._item_int_cols: List[List[int]] = raw['item_int']
        self.item_int_schema: FeatureSchema = FeatureSchema()
        self.item_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._item_int_cols:
            self.item_int_schema.add(fid, dim)
            self.item_int_vocab_sizes.extend([vs] * dim)

        # ---- user_dense: [[fid, dim], ...] ----
        self._user_dense_cols: List[List[int]] = raw['user_dense']
        self.user_dense_schema: FeatureSchema = FeatureSchema()
        for fid, dim in self._user_dense_cols:
            self.user_dense_schema.add(fid, dim)
        self._pair_feature_start_offset = self.user_dense_schema.total_dim
        if self._pair_feature_specs:
            for i, _ in enumerate(self._pair_feature_specs):
                self.user_dense_schema.add(
                    PAIR_FEATURE_ID_BASE + i,
                    self._pair_features_per_spec,
                )
        self._pair_feature_dim = (
            len(self._pair_feature_specs) * self._pair_features_per_spec
        )
        self._calendar_time_feature_offset = self.user_dense_schema.total_dim
        self._calendar_time_feature_dim = (
            CALENDAR_TIME_FEATURE_DIM if self.use_calendar_time_features else 0
        )
        if self.use_calendar_time_features:
            self.user_dense_schema.add(
                CALENDAR_TIME_FEATURE_ID,
                CALENDAR_TIME_FEATURE_DIM,
            )
            logging.info(
                "Calendar time features enabled: +%d dims "
                "(day sin/cos, weekday sin/cos, month-day sin/cos, weekend)",
                CALENDAR_TIME_FEATURE_DIM)

        self._missing_indicator_offset = self.user_dense_schema.total_dim
        self._missing_indicator_dim = (
            len(self._user_int_cols) + len(self._item_int_cols) + len(self._user_dense_cols)
            if self.use_missing_indicator_features else 0
        )
        if self.use_missing_indicator_features:
            self.user_dense_schema.add(
                MISSING_INDICATOR_FEATURE_ID,
                self._missing_indicator_dim,
            )
            logging.info(
                "Missing indicator features enabled: user_int=%d, item_int=%d, "
                "user_dense=%d, +%d dense dims",
                len(self._user_int_cols),
                len(self._item_int_cols),
                len(self._user_dense_cols),
                self._missing_indicator_dim)

        self._typed_missing_indicator_offset = self.user_dense_schema.total_dim
        self._typed_missing_indicator_dim = 0
        if self.use_typed_missing_indicator_features:
            self._typed_missing_int_dim = (
                (len(self._user_int_cols) + len(self._item_int_cols))
                * INT_TYPED_MISSING_FEATURES_PER_SPEC
            )
            self._typed_missing_dense_dim = (
                len(self._user_dense_cols) * DENSE_TYPED_MISSING_FEATURES_PER_SPEC
            )
            self._typed_missing_summary_offset = (
                self._typed_missing_indicator_offset
                + self._typed_missing_int_dim
                + self._typed_missing_dense_dim
            )
            self._typed_missing_indicator_dim = (
                self._typed_missing_int_dim
                + self._typed_missing_dense_dim
                + TYPED_MISSING_SUMMARY_DIM
            )
            self.user_dense_schema.add(
                TYPED_MISSING_INDICATOR_FEATURE_ID,
                self._typed_missing_indicator_dim,
            )
            logging.info(
                "Typed missing indicator features enabled: user_int=%d x%d, "
                "item_int=%d x%d, user_dense=%d x%d, summary=%d, +%d dense dims",
                len(self._user_int_cols),
                INT_TYPED_MISSING_FEATURES_PER_SPEC,
                len(self._item_int_cols),
                INT_TYPED_MISSING_FEATURES_PER_SPEC,
                len(self._user_dense_cols),
                DENSE_TYPED_MISSING_FEATURES_PER_SPEC,
                TYPED_MISSING_SUMMARY_DIM,
                self._typed_missing_indicator_dim)
        if self.use_missing_sparse_buckets:
            logging.info(
                "Missing sparse buckets enabled: raw missing/invalid "
                "non-sequence int ids use per-feature bucket id=vocab_size")

        # ---- item_dense (empty) ----
        self.item_dense_schema: FeatureSchema = FeatureSchema()

        # ---- sequence domains ----
        self._seq_cfg: Dict[str, Dict[str, Any]] = raw['seq']
        self.seq_domains: List[str] = sorted(self._seq_cfg.keys())
        self.seq_feature_ids: Dict[str, List[int]] = {}
        self.seq_vocab_sizes: Dict[str, Dict[int, int]] = {}
        self.seq_domain_vocab_sizes: Dict[str, List[int]] = {}
        self.ts_fids: Dict[str, Optional[int]] = {}
        self.sideinfo_fids: Dict[str, List[int]] = {}
        self._seq_prefix: Dict[str, str] = {}
        self._seq_maxlen: Dict[str, int] = {}

        for domain in self.seq_domains:
            cfg = self._seq_cfg[domain]
            self._seq_prefix[domain] = cfg['prefix']
            ts_fid = cfg['ts_fid']
            self.ts_fids[domain] = ts_fid

            all_fids = [fid for fid, vs in cfg['features']]
            self.seq_feature_ids[domain] = all_fids
            self.seq_vocab_sizes[domain] = {fid: vs for fid, vs in cfg['features']}

            sideinfo = [fid for fid in all_fids if fid != ts_fid]
            self.sideinfo_fids[domain] = sideinfo
            self.seq_domain_vocab_sizes[domain] = [
                self.seq_vocab_sizes[domain][fid] for fid in sideinfo
            ]

            # max_len: from seq_max_lens arg; unspecified domains fall back to 256.
            self._seq_maxlen[domain] = seq_max_lens.get(domain, 256)

        self._seq_time_bucket_plan: Dict[str, Dict[str, Tuple[int, int]]] = {}
        if self.use_seq_time_bucket_features:
            for domain_idx, domain in enumerate(self.seq_domains):
                domain_plan: Dict[str, Tuple[int, int]] = {}
                for feature_idx, (name, vocab_size) in enumerate(SEQ_TIME_BUCKET_FEATURES):
                    fid = (
                        SEQ_TIME_BUCKET_FEATURE_ID_BASE
                        + domain_idx * len(SEQ_TIME_BUCKET_FEATURES)
                        + feature_idx
                    )
                    offset = self.user_int_schema.total_dim
                    self.user_int_schema.add(fid, 1)
                    self.user_int_vocab_sizes.append(vocab_size)
                    domain_plan[name] = (offset, vocab_size)
                self._seq_time_bucket_plan[domain] = domain_plan
            logging.info(
                "Sequence time bucket features enabled: %d domains x %d fids = +%d user_int fids",
                len(self.seq_domains),
                len(SEQ_TIME_BUCKET_FEATURES),
                len(self.seq_domains) * len(SEQ_TIME_BUCKET_FEATURES),
            )

        self._seq_time_feature_offset = self.user_dense_schema.total_dim
        self._seq_time_feature_dim = (
            len(self.seq_domains) * SEQ_TIME_FEATURES_PER_DOMAIN
            if self.use_seq_time_features else 0
        )
        if self.use_seq_time_features:
            for i, domain in enumerate(self.seq_domains):
                self.user_dense_schema.add(
                    SEQ_TIME_FEATURE_ID_BASE + i,
                    SEQ_TIME_FEATURES_PER_DOMAIN,
                )
            logging.info(
                "Sequence time features enabled: %d domains x %d dims = +%d dims",
                len(self.seq_domains),
                SEQ_TIME_FEATURES_PER_DOMAIN,
                self._seq_time_feature_dim)

        self._seq_trunc_feature_offset = self.user_dense_schema.total_dim
        self._seq_trunc_feature_dim = (
            len(self.seq_domains) * SEQ_TRUNC_FEATURES_PER_DOMAIN
            if self.use_seq_trunc_features else 0
        )
        if self.use_seq_trunc_features:
            for i, domain in enumerate(self.seq_domains):
                self.user_dense_schema.add(
                    SEQ_TRUNC_FEATURE_ID_BASE + i,
                    SEQ_TRUNC_FEATURES_PER_DOMAIN,
                )
            logging.info(
                "Sequence truncation features enabled: %d domains x %d dims = +%d dims",
                len(self.seq_domains),
                SEQ_TRUNC_FEATURES_PER_DOMAIN,
                self._seq_trunc_feature_dim)

    def _build_pair_feature_plan(self) -> None:
        """Resolve pair feature specs to item offsets and sequence slots."""
        self._pair_feature_plan: List[Tuple[int, str, int, int, int, int, int]] = []
        if not self._pair_feature_specs:
            return

        item_lookup = {
            fid: (offset, length)
            for fid, offset, length in self.item_int_schema.entries
        }
        seq_slot_lookup = {
            domain: {fid: slot for slot, fid in enumerate(self.sideinfo_fids[domain])}
            for domain in self.seq_domains
        }

        active = 0
        for i, (item_fid, domain, seq_fid) in enumerate(self._pair_feature_specs):
            out_offset = self._pair_feature_start_offset + i * self._pair_features_per_spec
            item_entry = item_lookup.get(item_fid)
            seq_slot = seq_slot_lookup.get(domain, {}).get(seq_fid)
            if item_entry is None or seq_slot is None:
                logging.warning(
                    "Pair feature spec skipped: item_int_feats_%s x %s_%s "
                    "is not present in schema",
                    item_fid, domain, seq_fid)
                continue
            item_offset, item_len = item_entry
            self._pair_feature_plan.append(
                (item_fid, domain, seq_fid, item_offset, item_len, seq_slot, out_offset)
            )
            active += 1

        logging.info(
            "Pair dense features enabled: %d specs, %d active, %d dims/spec, +%d dims%s",
            len(self._pair_feature_specs),
            active,
            self._pair_features_per_spec,
            self._pair_feature_dim,
            f", time_windows={PAIR_TIME_WINDOWS_SECONDS}"
            if self.use_pair_time_features else "")

    def _fill_calendar_time_features(
        self,
        user_dense: "npt.NDArray[np.float32]",
        timestamps: "npt.NDArray[np.int64]",
    ) -> None:
        """Append cyclic sample timestamp features in Beijing time."""
        if not self.use_calendar_time_features:
            return

        out = user_dense[:, self._calendar_time_feature_offset:
                         self._calendar_time_feature_offset + CALENDAR_TIME_FEATURE_DIM]
        local_ts = timestamps.astype(np.int64) + LOCAL_TIME_OFFSET_SECONDS
        seconds_in_day = np.mod(local_ts, 86400).astype(np.float32)
        day_phase = (2.0 * np.pi * seconds_in_day) / 86400.0

        day_index = np.floor_divide(local_ts, 86400)
        weekday = np.mod(day_index + 3, 7).astype(np.float32)
        week_phase = (2.0 * np.pi * weekday) / 7.0

        local_days = (
            timestamps.astype('datetime64[s]') + np.timedelta64(LOCAL_TIME_OFFSET_SECONDS, 's')
        ).astype('datetime64[D]')
        local_months = local_days.astype('datetime64[M]')
        month_day = (local_days - local_months).astype(np.int64).astype(np.float32)
        month_phase = (2.0 * np.pi * month_day) / 31.0

        out[:, 0] = np.sin(day_phase).astype(np.float32)
        out[:, 1] = np.cos(day_phase).astype(np.float32)
        out[:, 2] = np.sin(week_phase).astype(np.float32)
        out[:, 3] = np.cos(week_phase).astype(np.float32)
        out[:, 4] = np.sin(month_phase).astype(np.float32)
        out[:, 5] = np.cos(month_phase).astype(np.float32)
        out[:, 6] = (weekday >= 5).astype(np.float32)

    def _fill_calendar_bucket_features(
        self,
        user_int: "npt.NDArray[np.int64]",
        timestamps: "npt.NDArray[np.int64]",
    ) -> None:
        """Append low-cardinality sample timestamp bucket ids in Beijing time."""
        if not self.use_calendar_bucket_features:
            return

        local_ts = timestamps.astype(np.int64) + LOCAL_TIME_OFFSET_SECONDS
        seconds_in_day = np.mod(local_ts, 86400)
        hour = np.floor_divide(seconds_in_day, 3600).astype(np.int64)
        ten_minute = np.floor_divide(seconds_in_day, 600).astype(np.int64)

        day_index = np.floor_divide(local_ts, 86400)
        weekday = np.mod(day_index + 3, 7).astype(np.int64)
        hour_weekday = weekday * 24 + hour

        values = {
            'hour': hour + 1,
            'weekday': weekday + 1,
            'hour_weekday': hour_weekday + 1,
            'ten_minute': ten_minute + 1,
        }
        for name, offset, vocab_size in self._calendar_bucket_plan:
            vals = values[name]
            vals = np.clip(vals, 0, vocab_size).astype(np.int64)
            user_int[:, offset] = vals

    def _fill_seq_time_bucket_features(
        self,
        user_int: "npt.NDArray[np.int64]",
        seq_timestamps: Dict[str, "npt.NDArray[np.int64]"],
        seq_lengths: Dict[str, "npt.NDArray[np.int64]"],
        seq_full_time_stats: Dict[str, Dict[str, "npt.NDArray[np.int64]"]],
        timestamps: "npt.NDArray[np.int64]",
    ) -> None:
        """Append per-domain sequence time summary bucket ids."""
        if not self.use_seq_time_bucket_features:
            return

        for domain in self.seq_domains:
            plan = self._seq_time_bucket_plan.get(domain)
            if not plan:
                continue

            ts_matrix = seq_timestamps.get(domain)
            lengths = seq_lengths.get(domain)
            stats = seq_full_time_stats.get(domain)

            if ts_matrix is not None:
                valid = ts_matrix > 0
                retained_max_ts = np.where(valid, ts_matrix, 0).max(axis=1)
                has_retained_ts = retained_max_ts > 0
                recency = np.zeros_like(timestamps, dtype=np.int64)
                recency[has_retained_ts] = np.maximum(
                    timestamps[has_retained_ts] - retained_max_ts[has_retained_ts],
                    0,
                )
                offset, _ = plan['last_recency']
                user_int[:, offset] = _bucketize_with_padding(
                    recency,
                    BUCKET_BOUNDARIES,
                    has_retained_ts,
                )

            if stats:
                full_min = stats['full_min']
                full_max = stats['full_max']
                has_full_ts = (full_min > 0) & (full_max > 0) & (full_max >= full_min)
                full_span = np.zeros_like(timestamps, dtype=np.int64)
                full_span[has_full_ts] = full_max[has_full_ts] - full_min[has_full_ts]
                offset, _ = plan['full_span']
                user_int[:, offset] = _bucketize_with_padding(
                    full_span,
                    BUCKET_BOUNDARIES,
                    has_full_ts,
                )

            if lengths is not None:
                valid_len = lengths.astype(np.int64)
                offset, _ = plan['valid_len']
                user_int[:, offset] = _bucketize_with_padding(
                    valid_len,
                    SEQ_LEN_BUCKET_BOUNDARIES,
                    valid_len > 0,
                )

    def _fill_seq_time_features(
        self,
        user_dense: "npt.NDArray[np.float32]",
        seq_timestamps: Dict[str, "npt.NDArray[np.int64]"],
        timestamps: "npt.NDArray[np.int64]",
    ) -> None:
        """Append per-domain sequence timestamp summary features."""
        if not self.use_seq_time_features:
            return

        row_timestamps = timestamps.reshape(-1, 1)
        log_cap = np.log1p(PAIR_RECENCY_CAP_SECONDS)
        eps = 1e-6

        for domain_idx, domain in enumerate(self.seq_domains):
            ts_matrix = seq_timestamps.get(domain)
            if ts_matrix is None:
                continue

            out_offset = (
                self._seq_time_feature_offset
                + domain_idx * SEQ_TIME_FEATURES_PER_DOMAIN
            )
            out = user_dense[:, out_offset:out_offset + SEQ_TIME_FEATURES_PER_DOMAIN]

            valid = ts_matrix > 0
            counts = valid.sum(axis=1).astype(np.float32)
            has_valid = counts > 0

            age = np.where(
                valid,
                np.maximum(row_timestamps - ts_matrix, 0),
                PAIR_RECENCY_CAP_SECONDS,
            ).astype(np.float32)
            min_age = age.min(axis=1)
            recency = 1.0 - np.minimum(np.log1p(min_age) / log_cap, 1.0)
            recency[~has_valid] = 0.0

            valid_ts_for_max = np.where(valid, ts_matrix, 0)
            valid_ts_for_min = np.where(valid, ts_matrix, np.iinfo(np.int64).max)
            max_ts = valid_ts_for_max.max(axis=1)
            min_ts = valid_ts_for_min.min(axis=1)
            span = np.zeros_like(counts, dtype=np.float32)
            has_span = counts > 1
            span[has_span] = np.maximum(
                max_ts[has_span] - min_ts[has_span],
                0,
            ).astype(np.float32)
            span_norm = np.minimum(np.log1p(span) / log_cap, 1.0)

            age_sum = np.where(valid, age, 0.0).sum(axis=1)
            avg_age = np.divide(
                age_sum,
                np.maximum(counts, 1.0),
                out=np.full_like(age_sum, PAIR_RECENCY_CAP_SECONDS),
                where=counts > 0,
            )
            avg_recency = 1.0 - np.minimum(np.log1p(avg_age) / log_cap, 1.0)
            avg_recency[~has_valid] = 0.0

            max_len = max(1, self._seq_maxlen[domain])
            out[:, 0] = (
                np.log1p(counts) / np.log1p(float(max_len))
            ).astype(np.float32)
            out[:, 1] = recency.astype(np.float32)
            out[:, 2] = span_norm.astype(np.float32)
            out[:, 3] = avg_recency.astype(np.float32)

            for j, window in enumerate(SEQ_TIME_RECENT_WINDOWS):
                recent = valid & (age <= float(window))
                ratio = np.divide(
                    recent.sum(axis=1).astype(np.float32),
                    np.maximum(counts, 1.0),
                    out=np.zeros_like(counts),
                    where=counts > eps,
                )
                out[:, 4 + j] = ratio.astype(np.float32)

    def _fill_seq_trunc_features(
        self,
        user_dense: "npt.NDArray[np.float32]",
        seq_raw_lengths: Dict[str, "npt.NDArray[np.int64]"],
        seq_full_time_stats: Dict[str, Dict[str, "npt.NDArray[np.int64]"]],
    ) -> None:
        """Append per-domain raw-length and truncation summary features."""
        if not self.use_seq_trunc_features:
            return

        log_time_cap = np.log1p(PAIR_RECENCY_CAP_SECONDS)

        for domain_idx, domain in enumerate(self.seq_domains):
            raw_lengths = seq_raw_lengths.get(domain)
            if raw_lengths is None:
                continue

            out_offset = (
                self._seq_trunc_feature_offset
                + domain_idx * SEQ_TRUNC_FEATURES_PER_DOMAIN
            )
            out = user_dense[:, out_offset:out_offset + SEQ_TRUNC_FEATURES_PER_DOMAIN]

            max_len = max(1, self._seq_maxlen[domain])
            len_cap = float(max(max_len * SEQ_TRUNC_LEN_CAP_MULT, max_len + 1))
            raw = raw_lengths.astype(np.float32)
            overflow = np.maximum(raw - float(max_len), 0.0)

            out[:, 0] = np.minimum(
                np.log1p(raw) / np.log1p(len_cap),
                1.0,
            ).astype(np.float32)
            out[:, 1] = (overflow > 0).astype(np.float32)
            out[:, 2] = np.divide(
                overflow,
                np.maximum(raw, 1.0),
                out=np.zeros_like(raw),
                where=raw > 0,
            ).astype(np.float32)
            out[:, 3] = np.minimum(
                np.log1p(overflow) / np.log1p(len_cap),
                1.0,
            ).astype(np.float32)

            stats = seq_full_time_stats.get(domain)
            if not stats:
                continue

            full_min = stats['full_min']
            full_max = stats['full_max']
            retained_min = stats['retained_min']
            has_full_span = (full_min > 0) & (full_max > 0) & (full_max >= full_min)

            full_span = np.zeros_like(raw, dtype=np.float32)
            full_span[has_full_span] = (
                full_max[has_full_span] - full_min[has_full_span]
            ).astype(np.float32)
            out[:, 4] = np.minimum(
                np.log1p(full_span) / log_time_cap,
                1.0,
            ).astype(np.float32)

            has_tail_gap = (
                (overflow > 0)
                & (retained_min > 0)
                & (full_min > 0)
                & (retained_min >= full_min)
            )
            tail_gap = np.zeros_like(raw, dtype=np.float32)
            tail_gap[has_tail_gap] = (
                retained_min[has_tail_gap] - full_min[has_tail_gap]
            ).astype(np.float32)
            out[:, 5] = np.minimum(
                np.log1p(tail_gap) / log_time_cap,
                1.0,
            ).astype(np.float32)

    def __len__(self) -> int:
        # Ceiling per Row Group; this is an upper bound on the true batch count.
        return sum((n + self.batch_size - 1) // self.batch_size
                   for _, _, n in self._rg_list)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        rg_list = self._rg_list
        if worker_info is not None and worker_info.num_workers > 1:
            rg_list = [rg for i, rg in enumerate(rg_list)
                       if i % worker_info.num_workers == worker_info.id]

        buffer: List[Dict[str, Any]] = []
        for file_path, rg_idx, _ in rg_list:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(batch_size=self.batch_size, row_groups=[rg_idx]):
                batch_dict = self._convert_batch(batch)
                if self.shuffle and self.buffer_batches > 1:
                    buffer.append(batch_dict)
                    if len(buffer) >= self.buffer_batches:
                        yield from self._flush_buffer(buffer)
                        buffer = []
                else:
                    yield batch_dict

        if buffer:
            yield from self._flush_buffer(buffer)

        del buffer
        gc.collect()

    def _flush_buffer(
        self, buffer: List[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """Concatenate the buffered batches, shuffle at the row level, then
        re-slice and yield batch-sized chunks.
        """
        merged: Dict[str, torch.Tensor] = {}
        non_tensor_keys: Dict[str, Any] = {}
        for k in buffer[0].keys():
            if isinstance(buffer[0][k], torch.Tensor):
                merged[k] = torch.cat([b[k] for b in buffer], dim=0)
            else:
                non_tensor_keys[k] = buffer[0][k]
        total_rows = merged['label'].shape[0]
        rand_idx = torch.randperm(total_rows) if self.shuffle else torch.arange(total_rows)
        for i in range(0, total_rows, self.batch_size):
            end = min(i + self.batch_size, total_rows)
            batch: Dict[str, Any] = {k: v[rand_idx[i:end]] for k, v in merged.items()}
            batch.update(non_tensor_keys)
            yield batch
        del merged
        buffer.clear()

    # ---- Helpers ----

    def _record_oob(
        self,
        group: str,
        col_idx: int,
        arr: "npt.NDArray[np.int64]",
        vocab_size: int,
    ) -> None:
        """Record out-of-bound indices and (optionally) clip them to 0,
        without printing to the console.
        """
        oob_mask = arr >= vocab_size
        if not oob_mask.any():
            return
        key = (group, col_idx)
        oob_vals = arr[oob_mask]
        n = int(oob_mask.sum())
        mx = int(oob_vals.max())
        mn = int(oob_vals.min())
        if key in self._oob_stats:
            s = self._oob_stats[key]
            s['count'] += n
            s['max'] = max(s['max'], mx)
            s['min_oob'] = min(s['min_oob'], mn)
        else:
            self._oob_stats[key] = {
                'count': n, 'max': mx, 'min_oob': mn, 'vocab': vocab_size,
            }
        if self.clip_vocab:
            arr[oob_mask] = 0
        else:
            raise ValueError(
                f"{group} col_idx={col_idx}: {n} values out of range "
                f"[0, {vocab_size}), actual=[{mn}, {mx}]. "
                f"Use clip_vocab=True to clip or fix schema.json")

    def dump_oob_stats(self, path: Optional[str] = None) -> None:
        """Dump out-of-bound statistics to a file if ``path`` is provided,
        otherwise to ``logging.info``.
        """
        if not self._oob_stats:
            logging.info("No out-of-bound values detected.")
            return
        lines = ["=== Out-of-Bound Stats ==="]
        for (group, ci), s in sorted(self._oob_stats.items()):
            direction = "TOO_HIGH" if s['min_oob'] >= s['vocab'] else "TOO_LOW"
            lines.append(
                f"  {group} col_idx={ci}: vocab={s['vocab']}, "
                f"oob_count={s['count']}, range=[{s['min_oob']}, {s['max']}], "
                f"{direction}")
        msg = "\n".join(lines)
        if path:
            with open(path, 'w') as f:
                f.write(msg + "\n")
            logging.info(f"OOB stats written to {path}")
        else:
            logging.info(msg)

    def _pad_varlen_int_column(
        self,
        arrow_col: "pa.ListArray",
        max_len: int,
        B: int,
        invalid_bucket_id: Optional[int] = None,
        vocab_size: int = 0,
    ) -> Tuple["npt.NDArray[np.int64]", "npt.NDArray[np.int64]"]:
        """Pad an Arrow ``ListArray`` of ints to shape ``[B, max_len]``.

        Values <= 0 are mapped to 0 by default. When ``invalid_bucket_id`` is
        provided, raw missing/invalid values that were actually present in the
        source list are mapped to that learned bucket instead. Natural padding
        positions created by shorter lists remain 0.

        Returns:
            A tuple ``(padded, lengths)`` where ``padded`` has shape
            ``[B, max_len]`` and ``lengths`` has shape ``[B]``.
        """
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_len), dtype=np.int64)
        lengths = np.zeros(B, dtype=np.int64)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                if invalid_bucket_id is not None and max_len > 0:
                    padded[i, 0] = invalid_bucket_id
                    lengths[i] = 1
                continue
            use_len = min(raw_len, max_len)
            row = values[start:start + use_len]
            if invalid_bucket_id is not None:
                row = row.astype(np.int64, copy=True)
                invalid = row <= 0
                if vocab_size > 0:
                    invalid |= row >= vocab_size
                else:
                    invalid |= row != 0
                row[invalid] = invalid_bucket_id
            padded[i, :use_len] = row
            lengths[i] = use_len

        if invalid_bucket_id is None:
            padded[padded <= 0] = 0
        return padded, lengths

    # Backwards-compatible alias kept for bench_raw_dataset.py and other
    # external callers that pre-date the rename. New code should call
    # `_pad_varlen_int_column` directly.
    _pad_varlen_column = _pad_varlen_int_column

    def _map_scalar_int_column(
        self,
        arrow_col: Any,
        vocab_size: int,
        group: str,
        col_idx: int,
    ) -> "npt.NDArray[np.int64]":
        """Map a scalar int feature, optionally preserving invalid as a bucket."""
        arr = arrow_col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
        if vocab_size <= 0:
            arr[:] = 0
            return arr

        if self.use_missing_sparse_buckets:
            invalid = arr <= 0
            invalid |= arr >= vocab_size
            arr[invalid] = vocab_size
            return arr

        arr[arr <= 0] = 0
        self._record_oob(group, col_idx, arr, vocab_size)
        return arr

    def _pad_varlen_float_column(
        self,
        arrow_col: "pa.ListArray",
        max_dim: int,
        B: int,
    ) -> "npt.NDArray[np.float32]":
        """Pad an Arrow ``ListArray<float>`` to shape ``[B, max_dim]``."""
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_dim), dtype=np.float32)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_dim)
            padded[i, :use_len] = values[start:start + use_len]

        return padded

    def _int_missing_or_invalid_mask(
        self,
        arrow_col: Any,
        dim: int,
        B: int,
        vocab_size: int,
    ) -> "npt.NDArray[np.bool_]":
        """Return rows whose raw int feature is missing or mapped to padding."""
        null_mask = np.asarray(
            arrow_col.is_null().to_numpy(zero_copy_only=False),
            dtype=bool,
        )
        invalid = null_mask.copy()

        if dim == 1:
            arr = arrow_col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
            invalid |= arr <= 0
            if vocab_size > 0:
                invalid |= arr >= vocab_size
            else:
                invalid |= arr != 0
            return invalid

        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()
        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                invalid[i] = True
                continue
            if raw_len < dim:
                invalid[i] = True
            row = values[start:end]
            row_invalid = row <= 0
            if vocab_size > 0:
                row_invalid = row_invalid | (row >= vocab_size)
            else:
                row_invalid = np.ones_like(row, dtype=bool)
            if row_invalid.any():
                invalid[i] = True
        return invalid

    def _dense_missing_mask(
        self,
        arrow_col: Any,
        B: int,
        dim: int,
    ) -> "npt.NDArray[np.bool_]":
        """Return rows whose raw dense feature is null or shorter than schema."""
        invalid = np.asarray(
            arrow_col.is_null().to_numpy(zero_copy_only=False),
            dtype=bool,
        )
        if hasattr(arrow_col, 'offsets'):
            offsets = arrow_col.offsets.to_numpy()
            invalid |= (offsets[1:] - offsets[:-1]) < dim
        return invalid

    def _int_typed_missing_masks(
        self,
        arrow_col: Any,
        dim: int,
        B: int,
        vocab_size: int,
    ) -> Tuple["npt.NDArray[np.bool_]", "npt.NDArray[np.bool_]", "npt.NDArray[np.bool_]"]:
        """Split raw int invalid states into absent/short, non-positive, OOB."""
        null_mask = np.asarray(
            arrow_col.is_null().to_numpy(zero_copy_only=False),
            dtype=bool,
        )
        absent = null_mask.copy()
        nonpositive = np.zeros(B, dtype=bool)
        oob = np.zeros(B, dtype=bool)

        if dim == 1:
            arr = arrow_col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
            present = ~null_mask
            nonpositive |= present & (arr <= 0)
            if vocab_size > 0:
                oob |= present & (arr >= vocab_size)
            return absent, nonpositive, oob

        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()
        for i in range(B):
            if null_mask[i]:
                continue
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                absent[i] = True
                continue
            if raw_len < dim:
                absent[i] = True
            row = values[start:end]
            nonpositive[i] = bool((row <= 0).any())
            if vocab_size > 0:
                oob[i] = bool((row >= vocab_size).any())
        return absent, nonpositive, oob

    def _dense_typed_missing_masks(
        self,
        arrow_col: Any,
        B: int,
        dim: int,
    ) -> Tuple["npt.NDArray[np.bool_]", "npt.NDArray[np.bool_]"]:
        """Split raw dense invalid states into absent/short and zero-like."""
        null_mask = np.asarray(
            arrow_col.is_null().to_numpy(zero_copy_only=False),
            dtype=bool,
        )
        absent = null_mask.copy()
        zero_like = np.zeros(B, dtype=bool)

        if hasattr(arrow_col, 'offsets'):
            offsets = arrow_col.offsets.to_numpy()
            values = arrow_col.values.to_numpy()
            for i in range(B):
                if null_mask[i]:
                    continue
                start, end = int(offsets[i]), int(offsets[i + 1])
                raw_len = end - start
                if raw_len <= 0:
                    absent[i] = True
                    continue
                if raw_len < dim:
                    absent[i] = True
                row = values[start:min(end, start + dim)]
                zero_like[i] = bool(np.all(np.abs(row) <= 1e-8))
            return absent, zero_like

        arr = arrow_col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.float32)
        present = ~null_mask
        zero_like |= present & (np.abs(arr) <= 1e-8)
        return absent, zero_like

    def _fill_missing_indicator_features(
        self,
        user_dense: "npt.NDArray[np.float32]",
        batch: "pa.RecordBatch",
        B: int,
    ) -> None:
        """Append binary flags for raw non-sequence missing/invalid values."""
        if not self.use_missing_indicator_features:
            return

        out = user_dense[:, self._missing_indicator_offset:
                         self._missing_indicator_offset + self._missing_indicator_dim]
        out[:] = 0.0
        for kind, ci, dim, vocab_size, out_offset in self._missing_indicator_plan:
            if ci is None:
                user_dense[:, out_offset] = 1.0
                continue
            col = batch.column(ci)
            if kind == 'int':
                mask = self._int_missing_or_invalid_mask(col, dim, B, vocab_size)
            else:
                mask = self._dense_missing_mask(col, B, dim)
            user_dense[:, out_offset] = mask.astype(np.float32)

    def _fill_typed_missing_indicator_features(
        self,
        user_dense: "npt.NDArray[np.float32]",
        batch: "pa.RecordBatch",
        B: int,
    ) -> None:
        """Append typed missing flags plus compact side-level ratios."""
        if not self.use_typed_missing_indicator_features:
            return

        out = user_dense[:, self._typed_missing_indicator_offset:
                         self._typed_missing_indicator_offset
                         + self._typed_missing_indicator_dim]
        out[:] = 0.0
        summary = np.zeros((B, TYPED_MISSING_SUMMARY_DIM), dtype=np.float32)

        for kind, ci, dim, vocab_size, out_offset in self._typed_missing_indicator_plan:
            if ci is None:
                if kind == 'user_dense':
                    absent = np.ones(B, dtype=bool)
                    zero_like = np.zeros(B, dtype=bool)
                    user_dense[:, out_offset] = 1.0
                    user_dense[:, out_offset + 1] = 0.0
                    summary[:, 6] += absent.astype(np.float32)
                    summary[:, 7] += zero_like.astype(np.float32)
                else:
                    absent = np.ones(B, dtype=bool)
                    nonpositive = np.zeros(B, dtype=bool)
                    oob = np.zeros(B, dtype=bool)
                    user_dense[:, out_offset] = 1.0
                    user_dense[:, out_offset + 1] = 0.0
                    user_dense[:, out_offset + 2] = 0.0
                    base = 0 if kind == 'user_int' else 3
                    summary[:, base] += absent.astype(np.float32)
                    summary[:, base + 1] += nonpositive.astype(np.float32)
                    summary[:, base + 2] += oob.astype(np.float32)
                continue

            col = batch.column(ci)
            if kind in ('user_int', 'item_int'):
                absent, nonpositive, oob = self._int_typed_missing_masks(
                    col, dim, B, vocab_size)
                user_dense[:, out_offset] = absent.astype(np.float32)
                user_dense[:, out_offset + 1] = nonpositive.astype(np.float32)
                user_dense[:, out_offset + 2] = oob.astype(np.float32)
                base = 0 if kind == 'user_int' else 3
                summary[:, base] += absent.astype(np.float32)
                summary[:, base + 1] += nonpositive.astype(np.float32)
                summary[:, base + 2] += oob.astype(np.float32)
            else:
                absent, zero_like = self._dense_typed_missing_masks(col, B, dim)
                user_dense[:, out_offset] = absent.astype(np.float32)
                user_dense[:, out_offset + 1] = zero_like.astype(np.float32)
                summary[:, 6] += absent.astype(np.float32)
                summary[:, 7] += zero_like.astype(np.float32)

        denom = np.array([
            max(len(self._user_int_cols), 1),
            max(len(self._user_int_cols), 1),
            max(len(self._user_int_cols), 1),
            max(len(self._item_int_cols), 1),
            max(len(self._item_int_cols), 1),
            max(len(self._item_int_cols), 1),
            max(len(self._user_dense_cols), 1),
            max(len(self._user_dense_cols), 1),
        ], dtype=np.float32)
        summary /= denom.reshape(1, -1)
        user_dense[:, self._typed_missing_summary_offset:
                   self._typed_missing_summary_offset + TYPED_MISSING_SUMMARY_DIM] = summary

    def _fill_pair_features(
        self,
        user_dense: "npt.NDArray[np.float32]",
        item_int: "npt.NDArray[np.int64]",
        seq_values: Dict[str, "npt.NDArray[np.int64]"],
        seq_timestamps: Dict[str, "npt.NDArray[np.int64]"],
        timestamps: "npt.NDArray[np.int64]",
    ) -> None:
        """Append item-history match statistics into ``user_dense`` in place."""
        if not self._pair_feature_plan:
            return

        row_timestamps = timestamps.reshape(-1, 1)
        log_cap = np.log1p(PAIR_RECENCY_CAP_SECONDS)

        for _, domain, _, item_offset, item_len, seq_slot, out_offset in self._pair_feature_plan:
            domain_values = seq_values.get(domain)
            if domain_values is None:
                continue

            targets = item_int[:, item_offset:item_offset + item_len]
            seq_matrix = domain_values[:, seq_slot, :]
            valid_seq = seq_matrix > 0

            if item_len == 1:
                target = targets[:, 0]
                match = (
                    valid_seq
                    & (target.reshape(-1, 1) > 0)
                    & (seq_matrix == target.reshape(-1, 1))
                )
            else:
                target_valid = targets > 0
                match = (
                    valid_seq[:, :, None]
                    & target_valid[:, None, :]
                    & (seq_matrix[:, :, None] == targets[:, None, :])
                ).any(axis=2)

            counts = match.sum(axis=1).astype(np.float32)
            valid_counts = np.maximum(valid_seq.sum(axis=1).astype(np.float32), 1.0)
            user_dense[:, out_offset] = (counts > 0).astype(np.float32)
            user_dense[:, out_offset + 1] = (
                np.log1p(counts) / np.log1p(valid_counts)
            ).astype(np.float32)

            for j, window in enumerate(PAIR_RECENT_WINDOWS):
                user_dense[:, out_offset + 3 + j] = (
                    match[:, :window].any(axis=1).astype(np.float32)
                )

            ts_matrix = seq_timestamps.get(domain)
            if ts_matrix is None:
                user_dense[:, out_offset + 2] = 0.0
                continue

            valid_match_ts = match & (ts_matrix > 0)
            deltas = np.where(
                valid_match_ts,
                np.maximum(row_timestamps - ts_matrix, 0),
                PAIR_RECENCY_CAP_SECONDS,
            )
            min_delta = deltas.min(axis=1).astype(np.float32)
            recency = 1.0 - np.minimum(
                np.log1p(min_delta) / log_cap,
                1.0,
            )
            recency[counts <= 0] = 0.0
            user_dense[:, out_offset + 2] = recency.astype(np.float32)

            if self.use_pair_time_features:
                time_out_offset = out_offset + PAIR_FEATURES_PER_SPEC
                for j, window in enumerate(PAIR_TIME_WINDOWS_SECONDS):
                    user_dense[:, time_out_offset + j] = (
                        valid_match_ts
                        & (deltas <= float(window))
                    ).any(axis=1).astype(np.float32)

    def _convert_batch(self, batch: "pa.RecordBatch") -> Dict[str, Any]:
        """Convert an Arrow RecordBatch into a training-ready dict of tensors."""
        B = batch.num_rows

        # ---- meta ----
        timestamps = batch.column(self._col_idx['timestamp']).to_numpy().astype(np.int64)
        if self.is_training:
            labels = (batch.column(self._col_idx['label_type']).fill_null(0)
                      .to_numpy(zero_copy_only=False).astype(np.int64) == 2).astype(np.int64)
        else:
            labels = np.zeros(B, dtype=np.int64)
        user_ids = batch.column(self._col_idx['user_id']).to_pylist()

        # ---- user_int: write into pre-allocated buffer ----
        # By default null / <=0 / OOB values are mapped to padding 0. When
        # sparse missing buckets are enabled, raw invalid values use the
        # per-feature spare index ``vocab_size``; naturally padded list slots
        # remain 0 so multi-value mean pooling is not polluted by padding.
        user_int = self._buf_user_int[:B]
        user_int[:] = 0
        for ci, dim, offset, vs in self._user_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = self._map_scalar_int_column(
                    col, vs, group='user_int', col_idx=ci)
                user_int[:, offset] = arr
            else:
                if self.use_missing_sparse_buckets and vs > 0:
                    padded, _ = self._pad_varlen_int_column(
                        col, dim, B, invalid_bucket_id=vs, vocab_size=vs)
                else:
                    padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0 and not self.use_missing_sparse_buckets:
                    self._record_oob('user_int', ci, padded, vs)
                elif vs <= 0:
                    padded[:] = 0
                user_int[:, offset:offset + dim] = padded
        if self.use_calendar_bucket_features:
            self._fill_calendar_bucket_features(
                user_int=user_int,
                timestamps=timestamps,
            )

        # ---- item_int ----
        item_int = self._buf_item_int[:B]
        item_int[:] = 0
        for ci, dim, offset, vs in self._item_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = self._map_scalar_int_column(
                    col, vs, group='item_int', col_idx=ci)
                item_int[:, offset] = arr
            else:
                if self.use_missing_sparse_buckets and vs > 0:
                    padded, _ = self._pad_varlen_int_column(
                        col, dim, B, invalid_bucket_id=vs, vocab_size=vs)
                else:
                    padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0 and not self.use_missing_sparse_buckets:
                    self._record_oob('item_int', ci, padded, vs)
                elif vs <= 0:
                    padded[:] = 0
                item_int[:, offset:offset + dim] = padded

        # ---- user_dense ----
        user_dense = self._buf_user_dense[:B]
        user_dense[:] = 0
        for ci, dim, offset in self._user_dense_plan:
            col = batch.column(ci)
            padded = self._pad_varlen_float_column(col, dim, B)
            user_dense[:, offset:offset + dim] = padded
        if self.use_missing_indicator_features:
            self._fill_missing_indicator_features(
                user_dense=user_dense,
                batch=batch,
                B=B,
            )
        if self.use_typed_missing_indicator_features:
            self._fill_typed_missing_indicator_features(
                user_dense=user_dense,
                batch=batch,
                B=B,
            )

        result = {
            'user_int_feats': torch.from_numpy(user_int.copy()),
            'item_int_feats': torch.from_numpy(item_int.copy()),
            'item_dense_feats': torch.zeros(B, 0, dtype=torch.float32),
            'label': torch.from_numpy(labels),
            'timestamp': torch.from_numpy(timestamps),
            'user_id': user_ids,
            '_seq_domains': self.seq_domains,
        }

        # ---- Sequence features: fused padding directly into the 3D buffer ----
        seq_values: Dict[str, "npt.NDArray[np.int64]"] = {}
        seq_timestamps: Dict[str, "npt.NDArray[np.int64]"] = {}
        seq_lengths: Dict[str, "npt.NDArray[np.int64]"] = {}
        seq_raw_lengths: Dict[str, "npt.NDArray[np.int64]"] = {}
        seq_full_time_stats: Dict[str, Dict[str, "npt.NDArray[np.int64]"]] = {}
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            side_plan, ts_ci = self._seq_plan[domain]

            # Write directly into the pre-allocated 3D buffer.
            out = self._buf_seq[domain][:B]
            out[:] = 0
            lengths = self._buf_seq_lens[domain][:B]
            lengths[:] = 0
            raw_lengths = np.zeros(B, dtype=np.int64)

            # Fused path: first collect (offsets, values, vocab_size, col_idx)
            # for every side-info column, then fill the buffer in a single pass.
            col_data = []
            for ci, slot, vs in side_plan:
                col = batch.column(ci)
                col_data.append((col.offsets.to_numpy(), col.values.to_numpy(), vs, ci))

            for c, (offs, vals, vs, ci) in enumerate(col_data):
                for i in range(B):
                    s = int(offs[i])
                    e = int(offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    if rl > raw_lengths[i]:
                        raw_lengths[i] = rl
                    ul = min(rl, max_len)
                    out[i, c, :ul] = vals[s:s + ul]
                    if ul > lengths[i]:
                        lengths[i] = ul

            # Values <= 0 -> 0.
            out[out <= 0] = 0

            # Check out-of-bound values per feature's vocab_size.
            # vs==0 means no vocab info; force the whole slice to 0 so that
            # the model's 1-slot Embedding is never indexed out of range.
            for c, (_, _, vs, ci) in enumerate(col_data):
                slice_c = out[:, c, :]
                if vs > 0:
                    self._record_oob(f'seq_{domain}', ci, slice_c, vs)
                else:
                    slice_c[:] = 0

            result[domain] = torch.from_numpy(out.copy())
            result[f'{domain}_len'] = torch.from_numpy(lengths.copy())

            # Time bucketing.
            time_bucket = self._buf_seq_tb[domain][:B]
            time_bucket[:] = 0
            ts_padded = self._buf_seq_ts[domain][:B]
            ts_padded[:] = 0
            full_min_ts = np.zeros(B, dtype=np.int64)
            full_max_ts = np.zeros(B, dtype=np.int64)
            retained_min_ts = np.zeros(B, dtype=np.int64)
            if ts_ci is not None:
                ts_col = batch.column(ts_ci)
                ts_offs = ts_col.offsets.to_numpy()
                ts_vals = ts_col.values.to_numpy()
                # Pad timestamps into shape (B, max_len).
                for i in range(B):
                    s = int(ts_offs[i])
                    e = int(ts_offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    if rl > raw_lengths[i]:
                        raw_lengths[i] = rl
                    ul = min(rl, max_len)
                    raw_ts = ts_vals[s:e]
                    valid_raw_ts = raw_ts[raw_ts > 0]
                    if len(valid_raw_ts) > 0:
                        full_min_ts[i] = int(valid_raw_ts.min())
                        full_max_ts[i] = int(valid_raw_ts.max())
                    retained_ts = ts_vals[s:s + ul]
                    ts_padded[i, :ul] = retained_ts
                    valid_retained_ts = retained_ts[retained_ts > 0]
                    if len(valid_retained_ts) > 0:
                        retained_min_ts[i] = int(valid_retained_ts.min())

                ts_expanded = timestamps.reshape(-1, 1)
                time_diff = np.maximum(ts_expanded - ts_padded, 0)
                # np.searchsorted returns values in [0, len(BUCKET_BOUNDARIES)].
                # After +1 the nominal range is [1, len(BUCKET_BOUNDARIES)+1];
                # the upper bound only appears when time_diff exceeds the
                # largest boundary (~1 year) and would index past
                # nn.Embedding(NUM_TIME_BUCKETS=len(BUCKET_BOUNDARIES)+1).
                # Clip raw result to [0, len(BUCKET_BOUNDARIES)-1] so the final
                # bucket id (after +1) stays within [1, len(BUCKET_BOUNDARIES)]
                # and is always a valid Embedding index. Time-diffs beyond the
                # largest boundary collapse into the last bucket.
                raw_buckets = np.clip(
                    np.searchsorted(BUCKET_BOUNDARIES, time_diff.ravel()),
                    0, len(BUCKET_BOUNDARIES) - 1,
                )
                buckets = raw_buckets.reshape(B, max_len) + 1
                buckets[ts_padded == 0] = 0
                time_bucket[:] = buckets

            result[f'{domain}_time_bucket'] = torch.from_numpy(time_bucket.copy())
            seq_values[domain] = out
            seq_timestamps[domain] = ts_padded
            seq_lengths[domain] = lengths
            seq_raw_lengths[domain] = raw_lengths
            seq_full_time_stats[domain] = {
                'full_min': full_min_ts,
                'full_max': full_max_ts,
                'retained_min': retained_min_ts,
            }

        if self.use_pair_features:
            self._fill_pair_features(
                user_dense=user_dense,
                item_int=item_int,
                seq_values=seq_values,
                seq_timestamps=seq_timestamps,
                timestamps=timestamps,
            )
        if self.use_calendar_time_features:
            self._fill_calendar_time_features(
                user_dense=user_dense,
                timestamps=timestamps,
            )
        if self.use_seq_time_bucket_features:
            self._fill_seq_time_bucket_features(
                user_int=user_int,
                seq_timestamps=seq_timestamps,
                seq_lengths=seq_lengths,
                seq_full_time_stats=seq_full_time_stats,
                timestamps=timestamps,
            )
        if self.use_seq_time_features:
            self._fill_seq_time_features(
                user_dense=user_dense,
                seq_timestamps=seq_timestamps,
                timestamps=timestamps,
            )
        if self.use_seq_trunc_features:
            self._fill_seq_trunc_features(
                user_dense=user_dense,
                seq_raw_lengths=seq_raw_lengths,
                seq_full_time_stats=seq_full_time_stats,
            )
        result['user_int_feats'] = torch.from_numpy(user_int.copy())
        result['user_dense_feats'] = torch.from_numpy(user_dense.copy())

        return result


def get_pcvr_data(
    data_dir: str,
    schema_path: str,
    batch_size: int = 256,
    valid_ratio: float = 0.1,
    train_ratio: float = 1.0,
    num_workers: int = 16,
    buffer_batches: int = 20,
    shuffle_train: bool = True,
    seed: int = 42,
    clip_vocab: bool = True,
    seq_max_lens: Optional[Dict[str, int]] = None,
    split_mode: str = 'row_group',
    split_time_col: str = 'timestamp',
    split_time_stat: str = 'median',
    use_pair_features: bool = False,
    use_pair_time_features: bool = False,
    use_calendar_time_features: bool = False,
    use_calendar_bucket_features: bool = False,
    use_seq_time_bucket_features: bool = False,
    use_seq_time_features: bool = False,
    use_seq_trunc_features: bool = False,
    use_missing_indicator_features: bool = False,
    use_typed_missing_indicator_features: bool = False,
    use_missing_sparse_buckets: bool = False,
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader, PCVRParquetDataset]:
    """Create train / valid DataLoaders from raw multi-column Parquet files.

    The validation split is taken as the last ``valid_ratio`` fraction of Row
    Groups. ``split_mode='row_group'`` keeps the source file order, while
    ``split_mode='time'`` first sorts Row Groups by a per-Row-Group time
    statistic and therefore uses later Row Groups for validation.

    Returns:
        A tuple ``(train_loader, valid_loader, train_dataset)``. The third
        element is returned so the caller can access the feature schema
        (``user_int_schema``, ``item_int_schema``, ...) needed to construct
        the model.
    """
    random.seed(seed)

    rg_info = _collect_row_groups(data_dir)
    rg_info = _order_row_groups(
        rg_info,
        split_mode=split_mode,
        split_time_col=split_time_col,
        split_time_stat=split_time_stat,
    )
    total_rgs = len(rg_info)

    n_valid_rgs = max(1, int(total_rgs * valid_ratio))
    n_train_rgs = total_rgs - n_valid_rgs

    # train_ratio: use only the first N% of the training Row Groups.
    if train_ratio < 1.0:
        n_train_rgs = max(1, int(n_train_rgs * train_ratio))
        logging.info(f"train_ratio={train_ratio}: using {n_train_rgs} train Row Groups")

    train_rg_info = rg_info[:n_train_rgs]
    valid_rg_info = rg_info[n_train_rgs:]
    train_rows = sum(r[2] for r in train_rg_info)
    valid_rows = sum(r[2] for r in valid_rg_info)

    logging.info(f"Row Group split ({split_mode}): {len(train_rg_info)} train "
                 f"({train_rows} rows), {len(valid_rg_info)} valid ({valid_rows} rows)")
    if split_mode == 'time':
        first_train_time = _row_group_time_value(
            train_rg_info[0][0], train_rg_info[0][1],
            split_time_col, split_time_stat) if train_rg_info else None
        last_train_time = _row_group_time_value(
            train_rg_info[-1][0], train_rg_info[-1][1],
            split_time_col, split_time_stat) if train_rg_info else None
        first_valid_time = _row_group_time_value(
            valid_rg_info[0][0], valid_rg_info[0][1],
            split_time_col, split_time_stat) if valid_rg_info else None
        last_valid_time = _row_group_time_value(
            valid_rg_info[-1][0], valid_rg_info[-1][1],
            split_time_col, split_time_stat) if valid_rg_info else None
        logging.info(
            f"Time split by {split_time_col}/{split_time_stat}: "
            f"train_time=[{first_train_time}, {last_train_time}], "
            f"valid_time=[{first_valid_time}, {last_valid_time}]")

    train_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=shuffle_train,
        buffer_batches=buffer_batches,
        row_group_list=train_rg_info,
        clip_vocab=clip_vocab,
        use_pair_features=use_pair_features,
        use_pair_time_features=use_pair_time_features,
        use_calendar_time_features=use_calendar_time_features,
        use_calendar_bucket_features=use_calendar_bucket_features,
        use_seq_time_bucket_features=use_seq_time_bucket_features,
        use_seq_time_features=use_seq_time_features,
        use_seq_trunc_features=use_seq_trunc_features,
        use_missing_indicator_features=use_missing_indicator_features,
        use_typed_missing_indicator_features=use_typed_missing_indicator_features,
        use_missing_sparse_buckets=use_missing_sparse_buckets,
    )

    use_cuda = torch.cuda.is_available()
    _train_kw = {}
    if num_workers > 0:
        _train_kw['persistent_workers'] = True
        _train_kw['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_dataset, batch_size=None,
        num_workers=num_workers, pin_memory=use_cuda, **_train_kw,
    )

    valid_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        row_group_list=valid_rg_info,
        clip_vocab=clip_vocab,
        use_pair_features=use_pair_features,
        use_pair_time_features=use_pair_time_features,
        use_calendar_time_features=use_calendar_time_features,
        use_calendar_bucket_features=use_calendar_bucket_features,
        use_seq_time_bucket_features=use_seq_time_bucket_features,
        use_seq_time_features=use_seq_time_features,
        use_seq_trunc_features=use_seq_trunc_features,
        use_missing_indicator_features=use_missing_indicator_features,
        use_typed_missing_indicator_features=use_typed_missing_indicator_features,
        use_missing_sparse_buckets=use_missing_sparse_buckets,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=None,
        num_workers=0, pin_memory=use_cuda,
    )

    logging.info(f"Parquet train: {train_rows} rows, valid: {valid_rows} rows, "
                 f"batch_size={batch_size}, buffer_batches={buffer_batches}")

    return train_loader, valid_loader, train_dataset
