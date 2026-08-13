import { describe, expect, it } from "vitest";
import appSource from "./App.tsx?raw";
import laboratorySource from "./TraceDebugger.tsx?raw";

describe("unified landing layout", () => {
  it("uses one native trace laboratory without the fake knight lab", () => {
    expect(appSource).toContain("NativeTraceLaboratory");
    expect(appSource).not.toContain("KnightLab");
    expect(appSource).toContain("Арифметика ячейки");
    expect(laboratorySource).toContain("реальные float64");
    expect(laboratorySource).toContain("без логики правил во frontend");
    expect(laboratorySource).not.toContain("knightTrace");
  });
});
