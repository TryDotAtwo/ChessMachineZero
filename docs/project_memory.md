# Project Memory

- active_branch=codex/pure-frozen-transformer-vm-clean
- origin=empty orphan branch; implementation is from scratch and must not migrate legacy runtime code
- input=[B,2048,128] hard one-hot; requested move occupies first 3 rows
- output=[B,2045,128] recurrent context
- context=1200 history rows + 768 LEGAL_SET rows + 1 status row + 76 service rows
- numeric=FP4 frozen weights, exact FP16/BF16 fallback, FP32 QK/critical accumulation, hard forward plus floating STE backward
- attention=all heads d_head=2; exact 2D HullKV forward and NestedHullTopK2D backward competitors
- production_boundary=generic C++ tensor runtime only; no chess types, board state object, move decoder, procedural replay, search, evaluation, Python, or host argmax
