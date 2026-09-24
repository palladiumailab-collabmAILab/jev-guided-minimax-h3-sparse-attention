import { describe, expect, it } from "vitest";

import worker, { type Env } from "../src/index";

function env(): Env {
  return {
    AI: {
      run: async () => ({
        answers: { layer_0: { choice: "keep_10", confidence: 1 } },
      }),
    } as unknown as Env["AI"],
    SHARED_TOKEN: "token",
  };
}

describe("worker HTTP boundaries", () => {
  it("rejects non-POST requests to /plan", async () => {
    const response = await worker.fetch(
      new Request("https://worker.example/plan", {
        method: "GET",
        headers: { authorization: "Bearer token" },
      }),
      env(),
    );

    expect(response.status).toBe(405);
    expect(await response.json()).toEqual({ error: "method_not_allowed" });
  });

  it("returns 400 for invalid planner payload", async () => {
    const response = await worker.fetch(
      new Request("https://worker.example/plan", {
        method: "POST",
        headers: {
          authorization: "Bearer token",
          "content-type": "application/json",
        },
        body: JSON.stringify({ layer_count: 0 }),
      }),
      env(),
    );

    expect(response.status).toBe(400);
    const body = (await response.json()) as { error?: string };
    expect(body.error).toBe("invalid_request");
  });

  it("keeps unknown routes non-successful", async () => {
    const response = await worker.fetch(
      new Request("https://worker.example/unknown"),
      env(),
    );

    expect(response.status).toBe(404);
  });
});
