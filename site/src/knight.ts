export const WQ = [[0, 0], [1, 0], [0, 1]] as const;
export const WK = [[1, 0], [0, 1]] as const;
export const CLASS_KEYS = [[1, 0], [0, 1]] as const;
export const squareName = (square: number) => `${"abcdefgh"[square % 8]}${Math.floor(square / 8) + 1}`;
export const knightTargets = (source: number) => Array.from({length:64},(_,target)=>target)
  .filter(target => knightTrace(source, target).winner === 1);

const matVec = (x: readonly number[], w: readonly (readonly number[])[]) =>
  Array.from({ length: w[0].length }, (_, c) => x.reduce((sum, v, r) => sum + v * w[r][c], 0));

export function knightTrace(source: number, target: number) {
  const delta: [number, number] = [
    Math.abs(source % 8 - target % 8),
    Math.abs(Math.floor(source / 8) - Math.floor(target / 8)),
  ];
  const legal = (delta[0] === 1 && delta[1] === 2) || (delta[0] === 2 && delta[1] === 1);
  const token: [number, number, number] = [1, legal ? 0 : 1, legal ? 1 : 0];
  const q = matVec(token, WQ);
  const k = CLASS_KEYS.map(x => matVec(x, WK));
  const scores = k.map(row => row.reduce((sum, value, i) => sum + value * q[i], 0));
  const winner = scores[1] > scores[0] ? 1 : 0;
  return { source, target, delta, token, q, k, scores, winner };
}
