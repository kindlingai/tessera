#!/usr/bin/env node
// Runs the tesserax binary from this platform's package.
const { spawnSync } = require("node:child_process");
const path = require("node:path");

const pkg = `@kindlingai/tessera-${process.platform}-${process.arch}`;
let dir;
try {
  dir = path.dirname(require.resolve(`${pkg}/package.json`));
} catch {
  console.error(`tesserax: no build for ${process.platform}-${process.arch} (${pkg} is not installed)`);
  // npx keeps an install that ran before the platform package was served.
  const npx = __dirname.match(/^(.*[\\/]_npx[\\/][^\\/]+)[\\/]/);
  if (npx) {
    console.error(`tesserax: if ${pkg} exists for this version, delete ${npx[1]} and run npx again`);
  } else {
    console.error(`tesserax: if ${pkg} exists for this version, reinstall @kindlingai/tessera`);
  }
  process.exit(1);
}
const run = spawnSync(path.join(dir, "bin", "tesserax"), process.argv.slice(2), { stdio: "inherit" });
if (run.error) {
  console.error(`tesserax: ${run.error.message}`);
  process.exit(1);
}
process.exit(run.status ?? 1);
