import { describe, expect, it } from "vitest";
import { knightTrace } from "./knight";

describe("knight matrix trace", () => {
  it("classifies a legal pair through matrix scores and hardmax", () => {
    const trace = knightTrace(1, 18);
    expect(trace.delta).toEqual([1, 2]);
    expect(trace.token).toEqual([1, 0, 1]);
    expect(trace.q).toEqual([0, 1]);
    expect(trace.scores).toEqual([0, 1]);
    expect(trace.winner).toBe(1);
  });

  it("classifies an illegal pair", () => {
    const trace = knightTrace(1, 17);
    expect(trace.delta).toEqual([0, 2]);
    expect(trace.token).toEqual([1, 1, 0]);
    expect(trace.scores).toEqual([1, 0]);
    expect(trace.winner).toBe(0);
  });
});
