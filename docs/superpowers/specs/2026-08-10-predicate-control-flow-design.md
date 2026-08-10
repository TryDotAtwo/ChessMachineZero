# Predicate-token control flow

## Goal

Extend the clean Transformer VM with conditional control flow while preserving
its semantic boundary: runtime execution is only repeated frozen
self-attention inference over program and state tokens.

This design supersedes the initial VM specification where the host stopped
after observing `HALT`. The revised runtime uses a fixed inference unroll and
never branches on semantic output tokens.

The new instructions are:

- `CMP_EQ predicate, left, right`
- `JUMP_IF predicate, target`

No native branch may inspect an opcode, predicate value, register value, or
program counter. It may not inspect `HALT` to decide whether to continue.

## Percepta model

Predicates are explicit state tokens with a finite value domain `FALSE/TRUE`.
`CMP_EQ` does not produce a host boolean. Frozen attention relations match the
two operand-value tokens and route the corresponding predicate-value token into
the selected predicate slot.

`JUMP_IF` uses the predicate token as a key. Frozen attention relations select
either the instruction's target token or the sequential `PC+1` token and route
that value into the control token. Thus control flow is produced by inference,
not by native control flow.

The semantic primitive remains:

```text
Q = X Wq
K = X Wk
V = X Wv
scores = Q K^T + mask
A = deterministic_hardmax(scores)
Y = A V
```

Fixed row/feature matrix projections and tensor addition combine stage outputs.
Elementwise write masks are forbidden at runtime. All data-dependent selection
occurs through attention and hardmax.

## Token schema

Add:

- sixteen fixed instruction slots, replacing the initial four-slot capacity;
- predicate slots `p0..p3`;
- immutable predicate-value tokens `FALSE` and `TRUE`;
- equality relation tokens over the finite integer domain `0..15`;
- branch relation tokens mapping `(predicate value, fallthrough, target)` to
  the selected next-PC value;
- instruction fields for predicate destination, predicate source, and branch
  target. A branch predicate source may name a predicate slot or the immutable
  `TRUE` token; both use the same attention routing path.

The compiler may construct these frozen tokens and matrices offline. It must
reject invalid predicate indices and branch targets before execution. Runtime
does not validate them by inspecting token values.

## Attention stages

The existing transition is extended to eleven fixed stages:

1. Decode instruction and read both operands.
2. Select the ALU/equality relation and perform register writeback.
3. Write the equality result into a predicate token.
4. Read the branch predicate.
5. Build target and fallthrough candidate tokens.
6. Select the candidate PC and write PC/HALT.

Stages run for every instruction. Opcode and type tokens make inactive writes
route the previous state back unchanged. The native executor does not skip
stages based on the current instruction.

## First accepted loop

The acceptance program is logically equivalent to:

```text
LOAD_CONST r0, 0
LOAD_CONST r1, 3
LOAD_CONST r2, 1
loop:
ADD r0, r0, r2
CMP_EQ p0, r0, r1
JUMP_IF p0, done
JUMP_IF TRUE, loop
done:
HALT
```

The compiler represents the unconditional back-edge with an immutable `TRUE`
predicate token; the runtime receives no special jump opcode or host loop
semantics. The expected final state is `r0=3`, `p0=TRUE`, and `HALT=TRUE`.

## Runtime boundary

Production runtime may only:

- invoke the fixed tensor graph exactly `inference_steps` times, where the
  count is fixed before execution and independent of all VM outputs;
- check tensor shapes and finiteness without decoding semantic fields;
- serialize tensors and traces.

It may not contain host comparison, branching by VM data, indirect register or
instruction lookup, data-dependent gather/scatter, or a scalar fallback.
Host repetition of the identical transition is a fixed unroll, not program
control flow.

`HALT` is an absorbing state token. Once active, frozen attention relations
route the complete previous VM state back unchanged on every remaining
inference step. The host never performs `if HALT then stop`; callers inspect
the final output only after the fixed unroll has completed. A run whose final
state is not halted is reported as incomplete by an out-of-runtime result
validator, never by substituting another execution path.

## Verification

Acceptance requires:

1. Exhaustive `CMP_EQ` tests for all `16 x 16` value pairs.
2. Exhaustive `JUMP_IF` tests for both predicate values and every valid target.
3. Exact loop trace and final state for the program above.
4. Exact proof that every post-HALT step is byte-identical to the HALT state.
5. Byte-identical repeated executions.
6. A program-token mutation that changes a branch target without rebuilding
   runtime code.
7. A hardmax tie-break test.
8. Source audit rejection of opcode-, predicate-, value-, PC-, and HALT-dependent
   native branches and data-dependent indexing.
9. Mutation tests proving that injected native predicate dispatch or
   `if HALT then stop` logic fails the audit.

Passing functional tests alone is not evidence of semantic purity; the source
and graph-boundary audits are mandatory acceptance gates.

## Deferred scope

Ordering comparisons, indirect memory, wider integers, chess token schemas,
legal-move generation, and CUDA optimization remain separate reviewed slices.
This slice establishes reusable conditional control flow needed to express
those later programs.
