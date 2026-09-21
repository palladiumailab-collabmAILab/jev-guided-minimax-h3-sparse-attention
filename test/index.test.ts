import { describe, expect, it } from "vitest";

import worker, { type Env } from "../src/index";

const TOKEN = "test-shared-token-with-enough-entropy";

const plannerPayload = {
  step: 0,
  layer_count: 1,
  allowed_keep_percent: [1, 3, 5, 10],
  fallback_keep_percent: 10,
  min_confidence: 0.55,
  policy: "balanced",
  layer_metrics: [],
};

function makeEnv(overrides: Partial<Env> = {}): Env {
  return {
    AI: {
      run: async () => ({
        answers: { layer_0: { choice: "keep_10", confidence: 1 } },
      }),
    } as unknown as Env["AI"],
    ...overrides,
  };
}

async function request(
  path: string,
  env: Env,
  options: RequestInit = {},
): Promise<Response> {
  return worker.fetch(new Request(`https://worker.example${path}`, options), env);
}

describe("worker authentication", () => {
  it("keeps health public but fails closed for a remote plan without a token", async () => {
    const health = await request("/health", makeEnv());
    expect(health.status).toBe(200);

    const plan = await request("/plan", makeEnv({ ALLOW_INSECURE_LOCAL_DEV: "true" }), {
      method: "POST",
      body: JSON.stringify(plannerPayload),
    });
    expect(plan.status).toBe(401);
    expect(await plan.json()).toEqual({ error: "unauthorized" });
  });

  it("rejects missing and wrong credentials on both protected endpoints", async () => {
    const missing = await request("/mcp", makeEnv({ SHARED_TOKEN: TOKEN }), { method: "POST" });
    expect(missing.status).toBe(401);

    const wrong = await request("/plan", makeEnv({ SHARED_TOKEN: TOKEN }), {
      method: "POST",
      headers: { authorization: "Bearer wrong-token" },
      body: JSON.stringify(plannerPayload),
    });
    expect(wrong.status).toBe(401);
  });

  it("accepts a valid bearer token", async () => {
    const response = await request("/plan", makeEnv({ SHARED_TOKEN: TOKEN }), {
      method: "POST",
      headers: {
        authorization: `Bearer ${TOKEN}`,
        "content-type": "application/json",
      },
      body: JSON.stringify(plannerPayload),
    });

    expect(response.status).toBe(200);
    const payload = (await response.json()) as { keep_percent?: unknown };
    expect(payload.keep_percent).toEqual([10]);
  });

  it("allows an explicit loopback-only local development opt-in", async () => {
    const local = await worker.fetch(
      new Request("http://127.0.0.1:8787/plan", {
        method: "POST",
        body: JSON.stringify(plannerPayload),
      }),
      makeEnv({ ALLOW_INSECURE_LOCAL_DEV: "true" }),
    );
    expect(local.status).toBe(200);
  });
});
