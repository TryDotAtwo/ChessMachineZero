import { describe, expect, it } from "vitest";
import { hardmaxRow, multiplyCell, scoreCell, valueAt } from "./matrix";
import type { MatrixData } from "./trace-schema";

const dense = (name: string, rows: number, cols: number, values: number[]): MatrixData =>
  ({ name, rows, cols, storage: "dense", values });

describe("matrix arithmetic", () => {
  it("reads dense and COO values without inventing missing entries", () => {
    expect(valueAt(dense("D", 2, 2, [1, 2, 3, 4]), 1, 0)).toBe(3);
    const sparse: MatrixData = { name: "C", rows: 2, cols: 3, storage: "coo", default: 0, entries: [[1, 2, 7]] };
    expect(valueAt(sparse, 1, 2)).toBe(7);
    expect(valueAt(sparse, 0, 1)).toBe(0);
  });

  it("exposes every nonzero scalar term of a matrix product", () => {
    const result = multiplyCell(dense("L", 1, 3, [2, 0, 4]), dense("R", 3, 1, [3, 8, 5]), 0, 0);
    expect(result.terms).toEqual([{ k: 0, left: 2, right: 3, product: 6 }, { k: 2, left: 4, right: 5, product: 20 }]);
    expect(result.sum).toBe(26);
  });

  it("adds the mask and uses lowest-index hardmax ties", () => {
    const q = dense("Q", 1, 2, [1, 2]);
    const k = dense("K", 2, 2, [3, 4, 3, 4]);
    const mask = dense("M", 1, 2, [0, 0]);
    expect(scoreCell(q, k, mask, 0, 1).sum).toBe(11);
    expect(hardmaxRow([11, 11, 9])).toBe(0);
  });
});
