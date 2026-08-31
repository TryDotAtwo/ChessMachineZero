"""Offline stable-compaction circuit: prefix GEMMs, hard rank routing, GEMM.

No runtime sorting, list filtering or index extraction. Payloads can be real
matrices; eligibility must be hard 0/1. The caller must consume overflow: the
rows contain at most the first capacity eligible payloads, never a hidden tail.
"""

from dataclasses import dataclass
import math

import numpy as np

from .circuit import Circuit


@dataclass(frozen=True)
class Compacted:
    rows: int
    present: int
    overflow: int
    count: int
    rank: int


def stable_compact(circuit: Circuit, eligible: int, payload: int, *, capacity: int) -> Compacted:
    rows, width = circuit.shapes[eligible]
    if width != 1 or circuit.shapes[payload][0] != rows:
        raise ValueError("compaction requires [N,1] bits and [N,D] payloads")
    if not isinstance(capacity, int) or not 1 <= capacity <= 256 or rows > 16384:
        raise ValueError("compaction capacity or exact FP32 score bound exceeded")

    # Two small triangular matrices, not an N-by-N prefix matrix.
    block = min(64, rows)
    groups = math.ceil(rows / block)
    padded_rows = groups * block
    bits = eligible
    if padded_rows != rows:
        padding = circuit.constant("prefix_zero_padding", np.zeros((padded_rows - rows, 1)))
        bits = circuit.concat_rows((bits, padding))
    blocks = circuit.reshape(bits, groups, block)
    local = circuit.project(blocks, np.triu(np.ones((block, block), dtype=np.float32)), "within_block_prefix")
    totals = circuit.project(blocks, np.ones((block, 1), dtype=np.float32), "block_counts")
    preceding = circuit.project(circuit.transpose(totals),
                                np.triu(np.ones((groups, groups), dtype=np.float32), 1), "preceding_block_counts")
    prefix = circuit.add(local, circuit.transpose(preceding), "inclusive_prefix")
    rank = circuit.reshape(prefix, padded_rows, 1)
    if padded_rows != rows:
        rank = circuit.route(rank, 0, rows)

    count = circuit.project(circuit.transpose(totals), np.ones((groups, 1)), "eligible_count")
    overflow = circuit.threshold(count, capacity + .5, name="capacity_overflow")

    # 2*x*r-r*r is uniquely maximal at integer r=x in [0,capacity].
    # Out-of-capacity ranks are explicitly removed and reported via overflow.
    ranks = np.arange(capacity + 1, dtype=np.float32)[None, :]
    logits = circuit.project(rank, 2 * ranks, "rank_scores_linear")
    logits = circuit.bias(logits, -(ranks * ranks), "rank_scores_bias")
    routing = circuit.hardmax(logits, name="rank_one_hot")
    in_range = circuit.threshold(circuit.scale(rank, -1.), -capacity - .5)
    routing = circuit.logical_and(routing, circuit.logical_and(eligible, in_range), "eligible_rank_routes")
    routing = circuit.route(circuit.transpose(routing), 1, capacity + 1, name="slot_candidate_routes")
    packed = circuit.matmul(routing, payload, "stable_compacted_payload")
    ones = circuit.constant("candidate_unit_column", np.ones((rows, 1)))
    present = circuit.matmul(routing, ones, "occupied_output_slots")
    return Compacted(packed, present, overflow, count, rank)
