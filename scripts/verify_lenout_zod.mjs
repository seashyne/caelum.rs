#!/usr/bin/env node
/**
 * Verification Bridge: Caelum (Rust) -> @alenout/celestia (TypeScript Zod Validator)
 *
 * Reads a JSON proposal from stdin (e.g. piped from `caelum propose ...`).
 * When run inside the Cetia monorepo (where `../../dist/index.js` exists), validates strictly
 * against `@alenout/celestia` Zod schemas; otherwise performs standalone structural validation.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

function stripNullFields(val) {
  if (Array.isArray(val)) return val.map(stripNullFields);
  if (val && typeof val === "object") {
    const out = {};
    for (const [k, v] of Object.entries(val)) {
      if (v !== null) out[k] = stripNullFields(v);
    }
    return out;
  }
  return val;
}

const chunks = [];
for await (const chunk of process.stdin) {
  chunks.push(chunk);
}
const raw = Buffer.concat(chunks).toString("utf8").trim();

let parsed;
try {
  parsed = JSON.parse(raw);
} catch (err) {
  console.error("[Caelum Verify] FAILED: Invalid JSON syntax:", err.message);
  console.error("Raw output:\n", raw);
  process.exit(1);
}

const normalized = stripNullFields(parsed);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const zodDistCandidates = [
  path.resolve(__dirname, "../../packages/celestia-ts/dist/index.js"),
  path.resolve(__dirname, "../../dist/index.js"),
];
const zodDistPath = zodDistCandidates.find((p) => fs.existsSync(p));

if (zodDistPath) {
  const { validateProposal, sanitizeProposal } = await import(pathToFileURL(zodDistPath).href);
  const result = validateProposal(normalized);
  if (!result.ok) {
    console.error("[Zod Verify] FAILED Schema Validation:", result.error.message);
    console.error("Parsed JSON:\n", JSON.stringify(normalized, null, 2));
    process.exit(1);
  }
  const safe = sanitizeProposal(result.proposal);
  console.log("[Zod Verify] PASSED! Valid Lenout CelestiaProposal:");
  console.log(`  proposalId:           ${safe.proposalId}`);
  console.log(`  targetMode:           ${safe.targetMode}`);
  console.log(`  summary:              ${safe.summary}`);
  console.log(`  actions (${safe.actions.length}):          ${safe.actions.map((a) => a.type).join(", ")}`);
  console.log(`  requiresConfirmation: ${safe.requiresConfirmation}`);
} else {
  const required = ["proposalId", "requestId", "targetMode", "summary", "actions", "requiresConfirmation"];
  const missing = required.filter((k) => !(k in normalized));
  if (missing.length > 0 || !Array.isArray(normalized.actions) || normalized.actions.length === 0) {
    console.error("[Standalone Verify] FAILED: missing keys or empty actions:", missing);
    process.exit(1);
  }
  console.log("[Standalone Verify] PASSED! Valid JSON Proposal:", normalized.proposalId);
}
