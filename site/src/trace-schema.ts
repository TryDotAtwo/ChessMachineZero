export const REQUIRED_MATRICES = [
  "X", "Wq", "Wk", "Wv", "Q", "K", "V", "M", "S", "A", "Y", "R", "C", "XPrime",
] as const;

export type MatrixName = typeof REQUIRED_MATRICES[number];
export type NumericValue = number | "-inf";
export type CooEntry = [row: number, col: number, value: NumericValue];

export type MatrixData = {
  name: string;
  rows: number;
  cols: number;
  storage: "dense" | "coo";
  values?: NumericValue[];
  default?: NumericValue;
  entries?: CooEntry[];
};

export type StageTrace = {
  index: number;
  name: string;
  matrices: Record<MatrixName, MatrixData>;
  winners: number[];
  changed: Array<{ row: number; col: number; before: NumericValue; after: NumericValue }>;
};

export type InferenceTrace = {
  schema: "cmz.matrix-trace.v1";
  provenance: { commit: string; generator: string; dtype: "float64" };
  claim: { scope: "pawn-rule-slice"; fullChess: false };
  tokenLabels: string[];
  stages: StageTrace[];
};

function object(value: unknown, where: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error(`${where} must be an object`);
  return value as Record<string, unknown>;
}

function integer(value: unknown, where: string): number {
  if (!Number.isInteger(value) || (value as number) < 0) throw new Error(`${where} must be a non-negative integer`);
  return value as number;
}

function numeric(value: unknown, where: string): NumericValue {
  if (value === "-inf") return value;
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${where} must be finite or -inf`);
  return value;
}

function parseMatrix(value: unknown, expectedName: MatrixName, stage: number): MatrixData {
  const input = object(value, `stage ${stage} matrix ${expectedName}`);
  if (input.name !== expectedName) throw new Error(`stage ${stage} matrix ${expectedName} has wrong name`);
  const rows = integer(input.rows, `${expectedName}.rows`);
  const cols = integer(input.cols, `${expectedName}.cols`);
  if (rows === 0 || cols === 0) throw new Error(`${expectedName} shape must be non-empty`);
  if (input.storage === "dense") {
    if (!Array.isArray(input.values) || input.values.length !== rows * cols) throw new Error(`${expectedName} dense size mismatch`);
    return { name: expectedName, rows, cols, storage: "dense", values: input.values.map((v, i) => numeric(v, `${expectedName}[${i}]`)) };
  }
  if (input.storage === "coo") {
    if (!Array.isArray(input.entries)) throw new Error(`${expectedName} COO entries missing`);
    const entries = input.entries.map((entry, index): CooEntry => {
      if (!Array.isArray(entry) || entry.length !== 3) throw new Error(`${expectedName} COO entry ${index} malformed`);
      const row = integer(entry[0], `${expectedName} COO row`);
      const col = integer(entry[1], `${expectedName} COO col`);
      if (row >= rows || col >= cols) throw new Error(`${expectedName} COO entry out of bounds`);
      return [row, col, numeric(entry[2], `${expectedName} COO value`)];
    });
    return { name: expectedName, rows, cols, storage: "coo", default: numeric(input.default ?? 0, `${expectedName}.default`), entries };
  }
  throw new Error(`${expectedName} storage must be dense or coo`);
}

function matrixValue(matrix: MatrixData, row: number, col: number): NumericValue {
  if (matrix.storage === "dense") return matrix.values![row * matrix.cols + col];
  return matrix.entries!.find(entry => entry[0] === row && entry[1] === col)?.[2] ?? matrix.default!;
}

export function parseInferenceTrace(value: unknown): InferenceTrace {
  const input = object(value, "trace");
  if (input.schema !== "cmz.matrix-trace.v1") throw new Error("unsupported trace schema");
  const provenance = object(input.provenance, "provenance");
  if (typeof provenance.commit !== "string" || !/^[0-9a-f]{40}$/.test(provenance.commit)) throw new Error("provenance commit must be a full SHA-1");
  if (provenance.generator !== "cmz_vm2_export_chess1_trace" || provenance.dtype !== "float64") throw new Error("invalid provenance generator or dtype");
  const claim = object(input.claim, "claim");
  if (claim.scope !== "pawn-rule-slice" || claim.fullChess !== false) throw new Error("trace must retain incomplete full-chess claim");
  if (!Array.isArray(input.tokenLabels) || !input.tokenLabels.every(label => typeof label === "string")) throw new Error("tokenLabels must be strings");
  if (!Array.isArray(input.stages) || input.stages.length !== 15) throw new Error("trace must contain exactly 15 stages");
  const stages = input.stages.map((stageValue, expectedIndex): StageTrace => {
    const stage = object(stageValue, `stage ${expectedIndex}`);
    if (stage.index !== expectedIndex) throw new Error(`stage index ${expectedIndex} is out of order`);
    if (typeof stage.name !== "string") throw new Error(`stage ${expectedIndex} name missing`);
    const rawMatrices = object(stage.matrices, `stage ${expectedIndex} matrices`);
    const matrices = Object.fromEntries(REQUIRED_MATRICES.map(name => {
      if (!(name in rawMatrices)) throw new Error(`stage ${expectedIndex} missing matrix ${name}`);
      return [name, parseMatrix(rawMatrices[name], name, expectedIndex)];
    })) as Record<MatrixName, MatrixData>;
    if (!Array.isArray(stage.winners) || stage.winners.length !== matrices.S.rows) throw new Error(`stage ${expectedIndex} winner count mismatch`);
    const winners = stage.winners.map((winner, row) => {
      const parsed = integer(winner, `stage ${expectedIndex} winner ${row}`);
      if (parsed >= matrices.S.cols) throw new Error(`stage ${expectedIndex} winner out of bounds`);
      return parsed;
    });
    if (!Array.isArray(stage.changed)) throw new Error(`stage ${expectedIndex} changed must be an array`);
    const changed = stage.changed.map((changeValue, changeIndex) => {
      const change = object(changeValue, `stage ${expectedIndex} change ${changeIndex}`);
      const row = integer(change.row, "change row");
      const col = integer(change.col, "change col");
      if (row >= matrices.XPrime.rows || col >= matrices.XPrime.cols) throw new Error("change outside XPrime");
      const before = numeric(change.before, "change before");
      const after = numeric(change.after, "change after");
      if (matrixValue(matrices.XPrime, row, col) !== after) throw new Error("changed value is not backed by XPrime");
      return { row, col, before, after };
    });
    return { index: expectedIndex, name: stage.name, matrices, winners, changed };
  });
  return {
    schema: "cmz.matrix-trace.v1",
    provenance: { commit: provenance.commit as string, generator: provenance.generator as string, dtype: "float64" },
    claim: { scope: "pawn-rule-slice", fullChess: false },
    tokenLabels: input.tokenLabels as string[],
    stages,
  };
}
