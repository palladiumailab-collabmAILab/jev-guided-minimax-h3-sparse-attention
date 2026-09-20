import { describe, expect, it } from "vitest";

import { createSparsePlan, type AiRunner } from "../src/planner";

class FakeAI implements AiRunner {
  constructor(private readonly response: unknown) {}

  async run(): Promise<unknown> {
    return this.response;
  }
}

describe("createSparsePlan", () => {
  it("returns Jev-selected ratios when confidence is sufficient", async () => {
    const ai = new FakeAI({
      model: "jev-test",
      answers: {
        layer_0: { choice: "keep_01", confidence: 0.9 },
        layer_1: { choice: "keep_05", confidence: 0.8 },
        layer_2: { choice: "keep_10", confidence: 0.7 },
      },
      usage: { input_tokens: 10, output_tokens: 3 },
    });

    const plan = await createSparsePlan(ai, {
      step: 1,
      layer_count: 3,
      allowed_keep_percent: [1, 3, 5, 10],
      fallback_keep_percent: 10,
      min_confidence: 0.55,
    });

    expect(plan.keep_percent).toEqual([1, 5, 10]);
    expect(plan.fallback_layers).toEqual([]);
    expect(plan.average_keep_percent).toBeCloseTo(16 / 3);
    expect(plan.model).toBe("jev-test");
  });

  it("falls back on low confidence or invalid choices", async () => {
    const ai = new FakeAI({
      answers: {
        layer_0: { choice: "keep_01", confidence: 0.2 },
        layer_1: { choice: "keep_99", confidence: 0.99 },
      },
    });

    const plan = await createSparsePlan(ai, {
      step: 0,
      layer_count: 2,
      allowed_keep_percent: [1, 3, 5, 10],
      fallback_keep_percent: 10,
      min_confidence: 0.55,
    });

    expect(plan.keep_percent).toEqual([10, 10]);
    expect(plan.fallback_layers).toEqual([0, 1]);
  });

  it("uses the largest allowed ratio when the requested fallback is unavailable", async () => {
    const ai = new FakeAI({ answers: {} });

    const plan = await createSparsePlan(ai, {
      step: 0,
      layer_count: 2,
      allowed_keep_percent: [1, 3, 5],
      fallback_keep_percent: 10,
    });

    expect(plan.keep_percent).toEqual([5, 5]);
  });
});
