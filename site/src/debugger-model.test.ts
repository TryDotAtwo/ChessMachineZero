import { describe, expect, it } from "vitest";
import { DEBUG_STEPS, boardForStage, heatmapWindow, stageExplanation } from "./debugger-model";

describe("animated full trace debugger model", () => {
  it("maps six microsteps to the native matrices in execution order", () => {
    expect(DEBUG_STEPS.map(step => step.matrix)).toEqual(["Q", "K", "S", "A", "V", "XPrime"]);
  });

  it("explains every native stage in human language", () => {
    expect(Array.from({length:15},(_,stage)=>stageExplanation(stage).length > 20).every(Boolean)).toBe(true);
  });

  it("animates the proven e2-e3 board transition only after board write", () => {
    expect(boardForStage(11).pawn).toBe(12);
    expect(boardForStage(12).pawn).toBe(20);
  });

  it("builds a normalized heatmap from the actual selected matrix window", () => {
    const matrix = { name:"S", rows:2, cols:2, storage:"dense" as const, values:[0, 2, "-inf" as const, 1] };
    const map = heatmapWindow(matrix, 0, 0, 2);
    expect(map.map(cell => cell.intensity)).toEqual([0, 1, 0, 0.5]);
    expect(map[2].value).toBe("-inf");
  });
});
