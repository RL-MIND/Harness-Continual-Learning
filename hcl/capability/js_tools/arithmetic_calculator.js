"use strict";

module.exports.metadata = {
  name: "js_arithmetic_calculator",
  description:
    "Evaluate a numeric arithmetic expression containing only numbers, parentheses, and basic arithmetic operators.",
  tags: ["arithmetic", "math", "numeric", "calculate"],
  parameters: {
    type: "object",
    properties: {
      expression: {
        type: "string",
        description: "Arithmetic expression, e.g. '(20 / 2) * 20'."
      },
      precision: {
        type: "integer",
        description: "Decimal places for non-integer results.",
        minimum: 0,
        maximum: 12
      }
    },
    required: ["expression", "precision"],
    additionalProperties: false
  }
};

module.exports.execute = async function execute(args) {
  const expression = String(args.expression || "");
  const precision = Math.max(0, Math.min(Number(args.precision ?? 6), 12));
  if (!expression.trim()) {
    throw new Error("expression is required");
  }
  if (!/^[0-9+\-*/%().\s]+$/.test(expression)) {
    throw new Error("expression contains unsupported characters");
  }
  const result = Function(`"use strict"; return (${expression});`)();
  if (!Number.isFinite(result)) {
    throw new Error("expression did not produce a finite number");
  }
  const rendered = Number.isInteger(result)
    ? String(result)
    : result.toFixed(precision).replace(/0+$/g, "").replace(/\.$/g, "");
  return {
    expression,
    result: rendered,
    numeric_result: result
  };
};
