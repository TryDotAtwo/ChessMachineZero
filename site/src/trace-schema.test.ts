import { describe, expect, it } from "vitest";
import { parseInferenceTrace } from "./trace-schema";

const matrixNames = ["X", "Wq", "Wk", "Wv", "Q", "K", "V", "M", "S", "A", "Y", "XPrime"];

function matrix(name: string, rows = 2, cols = 2) {
  return { name, rows, cols, storage: "dense", values: [1, 0, 0, 1] };
}

function validTrace() {
  return {
    schema: "cmz.matrix-trace.v1",
    provenance: { commit: "3bfe852d1234567890abcdef1234567890abcdef", generator: "cmz_vm2_export_chess1_trace", dtype: "float64" },
    claim: { scope: "pawn-rule-slice", fullChess: false },
    tokenLabels: ["SIDE", "SQUARE:e2"],
    stages: Array.from({ length: 15 }, (_, index) => ({
      index,
      name: `stage-${index}`,
      matrices: Object.fromEntries(matrixNames.map(name => [name, matrix(name)])),
      writes: Array.from({ length: 3 }, (_, component) => ({
        component,
        R: matrix("R"),
        C: matrix("C"),
        before: matrix("before"),
        after: matrix("after"),
      })),
      winners: [0, 1],
      changed: index === 14 ? [{ row: 1, col: 1, before: 0, after: 1 }] : [],
    })),
  };
}

describe("parseInferenceTrace", () => {
  it("accepts a complete 15-stage numeric trace", () => {
    const trace = parseInferenceTrace(validTrace());
    expect(trace.stages).toHaveLength(15);
    expect(trace.stages[14].matrices.XPrime.name).toBe("XPrime");
    expect(trace.claim.fullChess).toBe(false);
  });

  it("rejects missing required matrices", () => {
    const input = validTrace();
    delete input.stages[3].matrices.S;
    expect(() => parseInferenceTrace(input)).toThrow(/stage 3.*S/);
  });

  it("requires all three sequential write projections", () => {
    const input = validTrace();
    input.stages[2].writes.pop();
    expect(() => parseInferenceTrace(input)).toThrow(/write components/);
  });

  it("rejects out-of-bounds COO entries and winners", () => {
    const input = validTrace();
    input.stages[0].matrices.Q = { name: "Q", rows: 2, cols: 2, storage: "coo", default: 0, entries: [[2, 0, 1]] } as never;
    expect(() => parseInferenceTrace(input)).toThrow(/COO/);
    const winnerInput = validTrace();
    winnerInput.stages[0].winners[0] = 2;
    expect(() => parseInferenceTrace(winnerInput)).toThrow(/winner/);
  });

  it("rejects invalid provenance, stage order, and unbacked board diff", () => {
    const badCommit = validTrace();
    badCommit.provenance.commit = "latest";
    expect(() => parseInferenceTrace(badCommit)).toThrow(/commit/);
    const badOrder = validTrace();
    badOrder.stages[4].index = 5;
    expect(() => parseInferenceTrace(badOrder)).toThrow(/stage index/);
    const badDiff = validTrace();
    badDiff.stages[14].changed[0].after = 7;
    expect(() => parseInferenceTrace(badDiff)).toThrow(/XPrime/);
  });
});
