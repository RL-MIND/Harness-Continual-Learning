"use strict";

const path = require("path");

async function main() {
  const mode = process.argv[2];
  const toolPath = process.argv[3];
  const rawArgs = process.argv[4] || "{}";
  if (!["metadata", "execute"].includes(mode)) {
    throw new Error("mode must be metadata or execute");
  }
  if (!toolPath) {
    throw new Error("tool path is required");
  }
  const resolvedToolPath = path.resolve(toolPath);
  const tool = require(resolvedToolPath);
  if (!tool || typeof tool !== "object") {
    throw new Error("tool module must export an object");
  }
  if (!tool.metadata || typeof tool.metadata !== "object") {
    throw new Error("tool module must export metadata");
  }
  if (mode === "metadata") {
    console.log(JSON.stringify({ metadata: tool.metadata }));
    return;
  }
  if (typeof tool.execute !== "function") {
    throw new Error("tool module must export execute(args)");
  }
  const args = JSON.parse(rawArgs);
  const result = await tool.execute(args);
  console.log(JSON.stringify({ result }));
}

main().catch((error) => {
  console.error(error && error.stack ? error.stack : String(error));
  process.exit(1);
});
