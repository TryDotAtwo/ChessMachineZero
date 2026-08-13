import type { MatrixData, NumericValue } from "./trace-schema";

export type ProductTerm = { k: number; left: number; right: number; product: number };
export type CellCalculation = { terms: ProductTerm[]; sum: number };

function finite(value: NumericValue, where: string): number {
  if (value === "-inf") throw new Error(`${where} is -inf, not a multiplicand`);
  return value;
}

export function valueAt(matrix: MatrixData, row: number, col: number): NumericValue {
  if (!Number.isInteger(row) || !Number.isInteger(col) || row < 0 || col < 0 || row >= matrix.rows || col >= matrix.cols) {
    throw new RangeError(`${matrix.name}[${row},${col}] is out of bounds`);
  }
  if (matrix.storage === "dense") return matrix.values![row * matrix.cols + col];
  return matrix.entries!.find(entry => entry[0] === row && entry[1] === col)?.[2] ?? matrix.default!;
}

export function multiplyCell(left: MatrixData, right: MatrixData, row: number, col: number): CellCalculation {
  if (left.cols !== right.rows) throw new Error(`cannot multiply ${left.name} ${left.rows}x${left.cols} by ${right.name} ${right.rows}x${right.cols}`);
  const terms: ProductTerm[] = [];
  let sum = 0;
  for (let k = 0; k < left.cols; k += 1) {
    const l = finite(valueAt(left, row, k), `${left.name}[${row},${k}]`);
    const r = finite(valueAt(right, k, col), `${right.name}[${k},${col}]`);
    const product = l * r;
    if (product !== 0) terms.push({ k, left: l, right: r, product });
    sum += product;
  }
  return { terms, sum };
}

export function scoreCell(q: MatrixData, k: MatrixData, mask: MatrixData, row: number, col: number): CellCalculation {
  const dot = multiplyCell(q, { ...k, name: `${k.name}T`, rows: k.cols, cols: k.rows, storage: "dense", values: Array.from({ length: k.rows * k.cols }, (_, index) => {
    const transposedRow = Math.floor(index / k.rows);
    const transposedCol = index % k.rows;
    return valueAt(k, transposedCol, transposedRow);
  }) }, row, col);
  const maskValue = valueAt(mask, row, col);
  return { terms: dot.terms, sum: maskValue === "-inf" ? Number.NEGATIVE_INFINITY : dot.sum + maskValue };
}

export function hardmaxRow(scores: number[]): number {
  if (scores.length === 0) throw new Error("hardmax requires a non-empty row");
  let winner = 0;
  for (let index = 1; index < scores.length; index += 1) {
    if (scores[index] > scores[winner]) winner = index;
  }
  return winner;
}
