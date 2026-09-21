import { McpServer, createMcpHandler } from "@modelcontextprotocol/server";
import { ZodError } from "zod";

import { PlanInputSchema, createSparsePlan, type AiRunner } from "./planner";

export interface Env {
  AI: Ai;
  SHARED_TOKEN?: string;
  ALLOW_INSECURE_LOCAL_DEV?: string;
}

function json(data: unknown, status = 200): Response {
  return Response.json(data, {
    status,
    headers: {
      "cache-control": "no-store",
    },
  });
}

function tokensMatch(provided: string, expected: string): boolean {
  const length = Math.max(provided.length, expected.length);
  let difference = provided.length ^ expected.length;
  for (let index = 0; index < length; index += 1) {
    difference |= (provided.charCodeAt(index) || 0) ^ (expected.charCodeAt(index) || 0);
  }
  return difference === 0;
}

function isLoopbackHost(hostname: string): boolean {
  return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "[::1]" || hostname === "::1";
}

function authorized(request: Request, env: Env): boolean {
  const configuredToken = env.SHARED_TOKEN?.trim();
  if (!configuredToken) {
    const url = new URL(request.url);
    return env.ALLOW_INSECURE_LOCAL_DEV === "true" && isLoopbackHost(url.hostname);
  }

  const authorization = request.headers.get("authorization") ?? "";
  const separator = authorization.indexOf(" ");
  if (separator < 0 || authorization.slice(0, separator).toLowerCase() !== "bearer") {
    return false;
  }
  return tokensMatch(authorization.slice(separator + 1).trim(), configuredToken);
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
  async fetch(request: Request, env: Env): Promise<Response> {
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
      const handler = createMcpHandler(() => createServer(env));
      return handler.fetch(request);
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
