# Matrix Trace Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Заменить текущий промо-сайт проверяемым интерактивным инспектором полной численной трассы fixed-attention inference.

**Architecture:** Native exporter формирует версионированный JSON из реального frozen `ProgramImage` и одного проверочного pawn-перехода. React-приложение только отображает этот артефакт, независимо пересчитывает выбранные матричные ячейки для проверки и не содержит шахматной логики.

**Tech Stack:** C++17/LibTorch, JSON trace artifact, React 19, TypeScript, Vite, Vitest, Python/pytest evidence gates, GitHub Pages.

## Global Constraints

- Runtime-семантика остаётся только fixed self-attention, matrix projections и deterministic hardmax.
- JavaScript не вычисляет шахматную легальность и не создаёт board transitions.
- Все численные данные имеют provenance из native trace artifact; декоративные матрицы запрещены.
- Полные шахматы явно помечены как не реализованные.
- На сайте нет истории запросов, ответов пользователю и маркетингового hero.
- Большие матрицы доступны полностью, но могут отображаться виртуализированно.
- Hardmax при равенстве выбирает минимальный глобальный индекс.
- `python-chess` разрешён только как test oracle и не входит в runtime/site.

---

### Task 1: Версионированная схема trace-артефакта

**Files:**
- Create: `site/src/trace-schema.ts`
- Create: `site/src/trace-schema.test.ts`
- Create: `site/public/traces/pawn-e2-e3.json`
- Modify: `site/package.json`

**Interfaces:**
- Produces: `MatrixData`, `StageTrace`, `InferenceTrace`, `parseInferenceTrace(input: unknown): InferenceTrace`.
- Matrix encoding: `{name, rows, cols, storage: "dense"|"coo", values?, entries?}`.

- [ ] **Step 1: Write failing schema tests**

Test exact schema version, commit provenance, 15 ordered stages, legal shapes,
COO bounds, finite numeric entries except mask `-Infinity`, winner bounds, and
board diff sourced from final matrix coordinates.

- [ ] **Step 2: Verify RED**

Run: `npm test -- --run site/src/trace-schema.test.ts`
Expected: FAIL because parser and artifact do not exist.

- [ ] **Step 3: Implement parser and smallest complete fixture**

Implement strict discriminated TypeScript validation without accepting missing
matrices. The checked-in artifact must include `X,Wq,Wk,Wv,Q,K,V,M,S,A,Y,R,C,XPrime`
for each stage, token labels, winners and changed coordinates.

- [ ] **Step 4: Verify GREEN and build**

Run: `npm test && npm run build` in `site/`.
Expected: all tests and TypeScript build pass.

- [ ] **Step 5: Commit**

`git commit -m "test: define matrix trace artifact schema"`

### Task 2: Native exporter and arithmetic provenance

**Files:**
- Create: `native/vm2/include/cmz_vm2/trace_exporter.h`
- Create: `native/vm2/src/trace_exporter.cpp`
- Create: `native/vm2/tools/export_chess1_trace.cpp`
- Create: `native/vm2/tests/test_trace_exporter.cpp`
- Modify: `native/vm2/include/cmz_vm2/attention.h`
- Modify: `native/vm2/src/attention.cpp`
- Modify: `native/vm2/CMakeLists.txt`
- Regenerate: `site/public/traces/pawn-e2-e3.json`

**Interfaces:**
- Produces: `StageSnapshot trace_stage(image, state, stage)` and
  `write_trace_json(path, image, initial_state, provenance)`.
- Consumes the exact schema from Task 1.

- [ ] **Step 1: Write failing native exact test**

Assert 15 snapshots; exact identities `Q=XWq`, `K=XWk`, `V=XWv`,
`S=QKᵀ+M`, `A=one_hot(argmax(S))`, `Y=AV`; exact stage write output; and
JSON winner/board diff equality with the native tensors.

- [ ] **Step 2: Verify RED**

Build and run `cmz_vm2_test_trace_exporter`; expect missing exporter symbols.

- [ ] **Step 3: Implement trace capture without changing transition semantics**

Expose attention intermediates already computed by the generic primitive and
serialize tensors deterministically. Dense small matrices serialize row-major;
large frozen tensors serialize COO entries plus explicit zero/default value.
Represent mask negative infinity with the string `"-inf"`.

- [ ] **Step 4: Generate the public artifact**

Run exporter for a white pawn on e2 choosing e2-e3. Embed current commit,
schema `cmz.matrix-trace.v1`, tensor dtype, token labels and stage names.

- [ ] **Step 5: Verify native test and TypeScript parser**

Run exporter test, then `npm test` and `npm run build` in `site/`.
Expected: artifact validates and exact native identities pass.

- [ ] **Step 6: Commit**

`git commit -m "feat: export exact native matrix traces"`

### Task 3: Matrix data model and independent verifier

**Files:**
- Create: `site/src/matrix.ts`
- Create: `site/src/matrix.test.ts`
- Create: `site/src/useTrace.ts`

**Interfaces:**
- Produces: `valueAt(matrix,row,col)`, `multiplyCell(left,right,row,col)`,
  `scoreCell(q,k,mask,row,col)`, `hardmaxRow(scores)`, `writeCell(x,y,r,c,row,col)`.
- `multiplyCell` returns `{terms: Term[], sum: number}` with every nonzero term.

- [ ] **Step 1: Write failing unit tests**

Cover dense/COO lookup, zero default, `-inf`, exact dot product terms, equal-score
lowest-index tie, AV lookup, and the two-step `R(Y-X)C` cell expansion.

- [ ] **Step 2: Verify RED**

Run: `npm test -- --run src/matrix.test.ts` from `site/`.

- [ ] **Step 3: Implement pure arithmetic helpers**

No chess concepts or square rules may occur in this module. Helpers consume
only dimensions and numbers from the artifact.

- [ ] **Step 4: Verify GREEN**

Run full `npm test`; expected all tests pass.

- [ ] **Step 5: Commit**

`git commit -m "feat: verify trace matrix arithmetic in browser"`

### Task 4: Proof-oriented application shell

**Files:**
- Replace: `site/src/App.tsx`
- Create: `site/src/components/TraceToolbar.tsx`
- Create: `site/src/components/Pipeline.tsx`
- Create: `site/src/components/BoardDiff.tsx`
- Create: `site/src/components/ProvenanceBar.tsx`
- Modify: `site/src/main.tsx`
- Delete: `site/src/trace.ts`

**Interfaces:**
- App state: `{stageIndex, operation, selectedRow, selectedCol, playing, viewMode}`.
- Components consume only parsed `InferenceTrace` and selection callbacks.

- [ ] **Step 1: Write component behavior tests**

Assert all 15 stages are selectable; back/forward/play/reset work; pipeline has
every required operation; provenance and incomplete-full-chess boundary are
visible; no legacy hero/request-response copy is rendered.

- [ ] **Step 2: Verify RED**

Run `npm test`; expect missing components/old page assertions.

- [ ] **Step 3: Implement the desktop shell**

Build a compact technical workspace: toolbar, pipeline rail, before/after board
derived solely from artifact coordinates, matrix viewport and inspector area.

- [ ] **Step 4: Implement keyboard and reduced-motion behavior**

Arrow keys change stages/cells; Space toggles play; visible focus states; live
region announces stage and winning key.

- [ ] **Step 5: Verify tests and build**

Run `npm test && npm run build`.

- [ ] **Step 6: Commit**

`git commit -m "feat: replace landing page with trace workspace"`

### Task 5: Full matrix viewer and scalar arithmetic inspector

**Files:**
- Create: `site/src/components/MatrixViewport.tsx`
- Create: `site/src/components/MatrixCell.tsx`
- Create: `site/src/components/ArithmeticInspector.tsx`
- Create: `site/src/components/TokenPath.tsx`
- Create: `site/src/components/MatrixViewport.test.tsx`
- Modify: `site/src/App.tsx`

**Interfaces:**
- `MatrixViewport` accepts `MatrixData`, selected cell and filters.
- `ArithmeticInspector` accepts operation plus row/column and returns the full
  formula/term list from Task 3 helpers.

- [ ] **Step 1: Write failing interaction tests**

Select a Q cell and show X×Wq terms; select score cell and show dot product plus
mask; select A cell and show winner/tie-break; select XPrime and show both write
products. Toggle full/all, active-query, winner and changed-cell modes.

- [ ] **Step 2: Verify RED**

Run the focused Vitest file; expect missing viewer.

- [ ] **Step 3: Implement virtualized full matrix grid**

Render only visible rows/columns while scroll ranges cover `rows×cols`; expose
exact row/column labels, value text, heat coloring and a full accessible cell
name. Never sample or fabricate values.

- [ ] **Step 4: Implement arithmetic and token-path inspectors**

List every nonzero scalar term, running/final sum, mask addition, winner reason,
selected V row, output features and write delta.

- [ ] **Step 5: Verify interaction tests and build**

Run full site test/build.

- [ ] **Step 6: Commit**

`git commit -m "feat: add exhaustive matrix and token inspectors"`

### Task 6: Visual system, responsive states and animations

**Files:**
- Replace: `site/src/styles.css`
- Modify: components from Tasks 4–5 only where semantic hooks are required.

**Interfaces:**
- Desktop: pipeline + matrix + arithmetic inspector simultaneously.
- Mobile: tabs `Путь`, `Матрица`, `Арифметика`, preserving exact data.

- [ ] **Step 1: Implement approved design tokens and layout**

Use near-black workspace, monospace numeric tables, cyan active flow, amber only
for hardmax winner, restrained borders, no decorative cards or hero treatment.

- [ ] **Step 2: Add data-flow motion**

Animate only active pipeline edges, current query row and winner-to-value route;
disable all motion under `prefers-reduced-motion`.

- [ ] **Step 3: Implement mobile tabs and overflow containment**

At 720px and below show one inspector region at a time; matrix remains fully
scrollable; toolbar actions remain reachable.

- [ ] **Step 4: Browser visual QA**

Verify desktop and mobile core flow, take screenshots, inspect clipping,
typography, values, focus states and selected/winner distinction.

- [ ] **Step 5: Commit**

`git commit -m "style: make matrix trace fully inspectable"`

### Task 7: Evidence gates and Pages deployment

**Files:**
- Modify: `tests/test_public_evidence_gate.py`
- Create: `tests/test_trace_artifact_evidence.py`
- Modify: `.github/workflows/pages.yml`
- Modify: `docs/project_memory.md`
- Modify: `docs/change_history.md`
- Modify: `docs/prompt_history.md`
- Create: `test_results/matrix_trace_explorer_2026-08-13.md`

**Interfaces:**
- CI blocks Pages unless source purity, public claims, artifact schema/provenance,
  selected arithmetic identities, site tests and site build all pass.

- [ ] **Step 1: Write failing Python evidence tests**

Assert exact schema/version, 15 stages from header, 256 candidates from header,
required matrices on every stage, source commit format, no legacy response copy,
explicit incomplete-full-chess claim and board diff coordinates present in X′.

- [ ] **Step 2: Verify RED then update workflow**

Add Python artifact test plus `npm test` to Pages evidence/build jobs and include
trace exporter/site paths in workflow triggers.

- [ ] **Step 3: Run complete scoped verification**

Run native exporter exact test, `pytest` purity/public/artifact gates, site
Vitest, site production build and browser desktop/mobile workflow.

- [ ] **Step 4: Record evidence honestly**

Document commands, exact pass counts, trace commit/hash, observed browser sizes,
and retain the explicit statement that only the pawn rule slice is implemented.

- [ ] **Step 5: Commit and push**

`git commit -m "ci: gate matrix explorer on native trace evidence"`, push
`codex/percepta-transformer-vm`, wait for Pages workflow, then verify live HTML,
artifact and assets return HTTP 200.

