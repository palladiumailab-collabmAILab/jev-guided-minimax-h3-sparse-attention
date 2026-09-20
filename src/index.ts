import { McpServer } from "@modelcontextprotocol/server";
import { createMcpHandler } from "agents/mcp/server";
import { ZodError } from "zod";

import { PlanInputSchema, createSparsePlan, type AiRunner } from "./planner";

interface Env {
  AI: Ai;
  SHARED_TOKEN?: string;
}

function json(data: unknown, status = 200): Response {
  return Response.json(data, {
    status,
    headers: {
      "cache-control": "no-store",
    },
  });
}

function authorized(request: Request, env: Env): boolean {
  if (!env.SHARED_TOKEN) return true;
  return request.headers.get("authorization") === `Bearer ${env.SHARED_TOKEN}`;
}

function createServer(env: Env): McpServer {
  const server = new McpServer({
    name: "jev-h3-sparse-attention",
    version: "0.1.0",
  });

  server.registerTool(
    "choose_sparse_plan",
    {
      description:
        "Choose a per-layer MiniMax H3 sparse-attention keep ratio. " +
        "Returns one of 1%, 3%, 5%, or 10% for each layer using Workers AI typesafe/jev.",
      inputSchema: PlanInputSchema.shape,
    },
    async (args) => {
      const result = await createSparsePlan(env.AI as unknown as AiRunner, args);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(result),
          },
        ],
      };
    },
  );

  return server;
}

async function handlePlan(request: Request, env: Env): Promise<Response> {
  if (request.method !== "POST") {
    return json({ error: "method_not_allowed" }, 405);
  }

  try {
    const body = PlanInputSchema.parse(await request.json());
    const result = await createSparsePlan(env.AI as unknown as AiRunner, body);
    return json(result);
  } catch (error) {
    if (error instanceof ZodError) {
      return json({ error: "invalid_request", issues: error.issues }, 400);
    }
    const message = error instanceof Error ? error.message : "unknown planner error";
    return json({ error: "planner_failed", message }, 502);
  }
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);

    if (url.pathname === "/health") {
      return json({ ok: true, service: "jev-h3-sparse-attention" });
    }

    if (url.pathname === "/plan" || url.pathname === "/mcp") {
      if (!authorized(request, env)) {
        return json({ error: "unauthorized" }, 401);
      }
    }

    if (url.pathname === "/plan") {
      return handlePlan(request, env);
    }

    if (url.pathname === "/mcp") {
      return createMcpHandler(() => createServer(env), { route: "/mcp" })(request, env, ctx);
    }

    return json(
      {
        service: "jev-h3-sparse-attention",
        endpoints: ["/health", "/plan", "/mcp"],
      },
      404,
    );
  },
} satisfies ExportedHandler<Env>;
