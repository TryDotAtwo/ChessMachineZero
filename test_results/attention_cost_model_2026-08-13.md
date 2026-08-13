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
