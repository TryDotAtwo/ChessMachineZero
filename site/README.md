# Static full-VM inspector

The page has two explicit evidence layers.

`recurrent_trace.json` is the primary view. It contains all 2,877 operations of
one complete recurrent transition (`request + prior context -> next context`),
nine contiguous stages, bilingual operation/frozen-tensor semantics, compact
exact numeric windows and one scalar proof per operation. The fixture is the
initial `e2e4` transition and includes prior/output context hashes, status,
history length, legal count and exact output-to-next-input feedback identity.

`numeric_trace.json` is the nested 45-operation position microscope. It retains
complete zero-based COO for its fixed `e2e4,d7d5,e4d5` fixture, arbitrary-cell
lookup and producer navigation.

Both traces are generated from executable artifacts:

```powershell
python -m vm_compiler.recurrent_site_trace
python -m vm_compiler.site_trace
```

The recurrent numeric windows are the exact Python/PyTorch reference execution
of the same serialized graph. They are not a dump of every native CUDA
intermediate. Publishing all conservative batch-one retained SSA values would
be roughly 3 GiB. Native evidence separately compares full output contexts,
sequential feedback and backward. The browser validates stored topology,
producer order, feedback hashes and generic scalar arithmetic; it neither runs
`.cmz` nor implements chess.

Static asset versions are derived from normalized content by
`vm_compiler.site_trace`. Serve locally with:

```powershell
python -m http.server 4173 --directory site
```

See `docs/architecture.md` and
`test_results/full_recurrent_vm_and_site_2026-09-01.md` for the exact runtime
boundary, native commands, measurements and non-claims.
