#!/usr/bin/env node
// Runs the sensaix binary from this platform's package.
const { spawnSync } = require("node:child_process");
const path = require("node:path");

const pkg = `sensaix-${process.platform}-${process.arch}`;
let dir;
try {
  dir = path.dirname(require.resolve(`${pkg}/package.json`));
} catch {
  console.error(`sensaix: no build for ${process.platform}-${process.arch} (${pkg} is not installed)`);
  process.exit(1);
}
const run = spawnSync(path.join(dir, "bin", "sensaix"), process.argv.slice(2), { stdio: "inherit" });
if (run.error) {
  console.error(`sensaix: ${run.error.message}`);
  process.exit(1);
}
process.exit(run.status ?? 1);
