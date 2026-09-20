import { z } from "zod";

export const KEEP_OPTIONS = [1, 3, 5, 10] as const;
export type KeepPercent = (typeof KEEP_OPTIONS)[number];

const KeepPercentSchema = z.union([
  z.literal(1),
  z.literal(3),
  z.literal(5),
  z.literal(10),
]);

export const LayerMetricSchema = z.object({
  layer: z.number().int().min(0).max(127),
  activation_rms: z.number().finite().nonnegative().optional(),
  activation_variance: z.number().finite().nonnegative().optional(),
  delta_rms: z.number().finite().nonnegative().optional(),
});

export const PlanInputSchema = z.object({
  step: z.number().int().min(0),
  sigma: z.number().finite().optional(),
  sequence_length: z.number().int().positive().optional(),
  layer_count: z.number().int().min(1).max(64),
  allowed_keep_percent: z.array(KeepPercentSchema).min(1).default([...KEEP_OPTIONS]),
  fallback_keep_percent: KeepPercentSchema.optional(),
  min_confidence: z.number().min(0).max(1).default(0.55),
  policy: z.enum(["speed", "balanced", "quality"]).default("balanced"),
  layer_metrics: z.array(LayerMetricSchema).max(64).default([]),
});

export type PlanInput = z.infer<typeof PlanInputSchema>;

export interface AiRunner {
  run(model: string, input: unknown): Promise<unknown>;
}

export interface SparsePlan {
  keep_percent: KeepPercent[];
  confidence: number[];
  fallback_layers: number[];
  average_keep_percent: number;
  model: string | null;
  usage: unknown;
}

type JevChoiceAnswer = {
  type?: string;
  choice?: string;
  confidence?: number;
  probabilities?: Record<string, number>;
};

type JevResponse = {
  model?: string;
  answers?: Record<string, JevChoiceAnswer>;
  usage?: unknown;
};

function choiceKey(percent: KeepPercent): string {
  return `keep_${String(percent).padStart(2, "0")}`;
}

function policyInstruction(policy: PlanInput["policy"]): string {
  switch (policy) {
    case "speed":
      return "Prefer lower keep percentages when the available evidence does not indicate a sensitive layer.";
    case "quality":
      return "Prefer higher keep percentages unless the available evidence strongly supports more aggressive pruning.";
    default:
      return "Balance compute reduction against approximation risk; use aggressive pruning only when supported by the state.";
  }
}

function criteriaFor(options: KeepPercent[]): Record<string, string> {
  const descriptions: Record<KeepPercent, string> = {
    1: "Keep 1% of key blocks: maximum compute reduction and highest approximation risk.",
    3: "Keep 3% of key blocks: aggressive compute reduction.",
    5: "Keep 5% of key blocks: moderate compute reduction.",
    10: "Keep 10% of key blocks: most conservative option among the allowed sparse ratios.",
  };
  return Object.fromEntries(options.map((p) => [choiceKey(p), descriptions[p]]));
}

function asJevResponse(value: unknown): JevResponse {
  return value && typeof value === "object" ? (value as JevResponse) : {};
}

export async function createSparsePlan(ai: AiRunner, rawInput: unknown): Promise<SparsePlan> {
  const input = PlanInputSchema.parse(rawInput);
  const allowed = [...new Set(input.allowed_keep_percent)].sort((a, b) => a - b) as KeepPercent[];
  const fallback =
    input.fallback_keep_percent !== undefined && allowed.includes(input.fallback_keep_percent)
      ? input.fallback_keep_percent
      : allowed[allowed.length - 1];

  const metricsByLayer = Object.fromEntries(input.layer_metrics.map((m) => [String(m.layer), m]));
  const criteria = criteriaFor(allowed);
  const questions: Record<string, unknown> = {};

  for (let layer = 0; layer < input.layer_count; layer += 1) {
    questions[`layer_${layer}`] = {
      type: "choice",
      instructions:
        `Choose the sparse-attention keep ratio for transformer layer ${layer} of ${input.layer_count - 1}. ` +
        policyInstruction(input.policy) +
        " Use metrics for this exact layer when present. When evidence is weak, avoid unjustified aggressive pruning.",
      criteria,
    };
  }

  const state = {
    task: "Choose per-layer block-sparse attention keep percentages for a MiniMax H3 denoising step.",
    step: input.step,
    sigma: input.sigma ?? null,
    sequence_length: input.sequence_length ?? null,
    policy: input.policy,
    allowed_keep_percent: allowed,
    layer_metrics: metricsByLayer,
    constraints: [
      "Lower keep percentage reduces attention compute but increases approximation risk.",
      "The requested value is a routing decision, not a claim that output quality is unchanged.",
      "Do not infer unavailable activation statistics.",
    ],
  };

  const raw = asJevResponse(
    await ai.run("typesafe/jev", {
      state,
      questions,
    }),
  );

  const keepPercent: KeepPercent[] = [];
  const confidence: number[] = [];
  const fallbackLayers: number[] = [];

  for (let layer = 0; layer < input.layer_count; layer += 1) {
    const answer = raw.answers?.[`layer_${layer}`];
    const conf = typeof answer?.confidence === "number" ? answer.confidence : 0;
    const selected = allowed.find((p) => choiceKey(p) === answer?.choice);

    if (selected === undefined || conf < input.min_confidence) {
      keepPercent.push(fallback);
      fallbackLayers.push(layer);
    } else {
      keepPercent.push(selected);
    }
    confidence.push(conf);
  }

  const averageKeepPercent =
    keepPercent.reduce((sum, value) => sum + value, 0) / keepPercent.length;

  return {
    keep_percent: keepPercent,
    confidence,
    fallback_layers: fallbackLayers,
    average_keep_percent: averageKeepPercent,
    model: typeof raw.model === "string" ? raw.model : null,
    usage: raw.usage ?? null,
  };
}
