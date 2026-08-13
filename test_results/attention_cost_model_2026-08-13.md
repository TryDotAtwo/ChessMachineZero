# Attention cost model — 2026-08-13

```text
dense_score_pairs=15120240
useful_score_pairs=203861
reduction=74.1694x
```

Per-stage exact-route block counts range from 748 to 1004. Therefore exact
route grouping is rejected as a production GPU plan despite its minimal score
pair count. The next gate is an offline padded-block merge optimizer evaluated
on total FLOPs, bytes, block launches and measured latency.

Stage 2 budgeted merge result:

```text
useful=131820
one_block_padded=750992
sixteen_blocks_padded=171920
one_block_estimated_cost=755088
sixteen_blocks_estimated_cost=237456
```

The 16-block materialized prototype is byte-exact with dense output and global
hardmax winners. Dense selection matrices remain prototype-only, so this is not
yet performance evidence.

Frozen sparse COO routing then replaced dense selection matrices. Each route
matrix contains exactly one nonzero per row and retains byte-exact output and
winners. Five-run, warm, single-thread CPU diagnostic:

```text
stage2_cpu_dense_median_ms=39.6268
stage2_cpu_block_median_ms=27.5298
speedup=1.43941x
```

This result covers stage 2 only. Production transition and GPU speedup remain
unclaimed until all 15 plans and end-to-end gates pass.
