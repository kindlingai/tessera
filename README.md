# tessera

tessera is an engine to build engines.

In tessera, you define the shape of your model in code, and the engine makes an optimized engine for your
hardware, automatically.

If the automatic optimizer isn't good enough, you can write GPU-native code that makes use of the platform
services to squeeze all of the performance out the hardware.

## State

tessera is not alpha or beta. It's an experiment right now and no numbers are available.

## Supported Hardware

 - CUDA (NVIDIA)
 - ROCm (AMD)
 - Metal (Apple)
 - CPU (any)

## Supported Models

 - GLM 5.3
 - Qwen 0.6B
 - EmbeddingGemma 2
 - spaCy

## Crates

<!-- crate table -->
<table>
<tr><th>Folder</th><th>Crate</th><th>Description</th></tr>
<tr><td colspan="3"><a href="engine"><b>engine/</b></a></td></tr>
<tr><td>&emsp;<a href="engine/engine">engine/</a></td><td>tessera-engine</td><td>Runs graphs: a decode loop over a stepper, or one forward launch</td></tr>
<tr><td colspan="3">&emsp;<a href="engine/gpu"><b>gpu/</b></a></td></tr>
<tr><td>&emsp;&emsp;<a href="engine/gpu/dl">dl/</a></td><td>tessera-dl</td><td>dlopen for the runtimes that load at run time</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/gpu/gpukit">gpukit/</a></td><td>tessera-gpukit</td><td>The GPU runtimes tessera loads at run time: pipelines and buffers</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/gpu/platform">platform/</a></td><td>tessera-platform</td><td>Plain hardware description: platforms, limits and the per-architecture table</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/gpu/work">work/</a></td><td>tessera-work</td><td>A launch's work: ops per trip and the load they make</td></tr>
<tr><td colspan="3">&emsp;<a href="engine/ir"><b>ir/</b></a></td></tr>
<tr><td>&emsp;&emsp;<a href="engine/ir/graph">graph/</a></td><td>tessera-graph</td><td>The model graph: typed nodes, symbolic shapes, a JSON form</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/ir/interp">interp/</a></td><td>tessera-interp</td><td>The CPU reference for every plan: runs a graph op by op</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/ir/kv">kv/</a></td><td>tessera-kv</td><td>Pages of kept rows, shared between runs by refcount</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/ir/mem">mem/</a></td><td>tessera-mem</td><td>Buffers in zones, typed views onto them, and storage forms</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/ir/testkit">testkit/</a></td><td>tessera-testkit</td><td>Helpers that tests share: noise, comparisons and scratch files</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/ir/trace">trace/</a></td><td>tessera-trace</td><td>Builds graphs by running model descriptions</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/ir/trace-derive">trace-derive/</a></td><td>tessera-trace-derive</td><td>#[derive(Weights)] for model descriptions</td></tr>
<tr><td>&emsp;<a href="engine/models">models/</a></td><td>tessera-models</td><td>Model descriptions, traced into graphs</td></tr>
<tr><td colspan="3">&emsp;<a href="engine/plan"><b>plan/</b></a></td></tr>
<tr><td>&emsp;&emsp;<a href="engine/plan/codegen">codegen/</a></td><td>tessera-codegen</td><td>Generates kernel text for every dialect, and counts what it costs</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/plan/comm">comm/</a></td><td>tessera-engine-comm</td><td>Moves bytes between ranks over sockets, shared memory or RDMA</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/plan/host">host/</a></td><td>tessera-host</td><td>Runs kernel bodies on the host, each group in lockstep</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/plan/jit">jit/</a></td><td>tessera-jit</td><td>Compiles kernel bodies for the host with Cranelift as plans are made</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/plan/kernel-macro">kernel-macro/</a></td><td>tessera-kernel-macro</td><td>#[kernel]: a Rust function body as a kernel tree</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/plan/lower">lower/</a></td><td>tessera-lower</td><td>Lowers a graph to a plan of kernel launches</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/plan/run">run/</a></td><td>tessera-run</td><td>Runs lowered plans on the host, Metal, CUDA or ROCm</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/plan/sim">sim/</a></td><td>tessera-sim</td><td>Analytic models of GPUs: what a launch should cost</td></tr>
<tr><td colspan="3">&emsp;<a href="engine/rules"><b>rules/</b></a></td></tr>
<tr><td>&emsp;&emsp;<a href="engine/rules/attention">attention/</a></td><td>tessera-attention</td><td>Attention kernels for the CPU and the GPU, registered as lowering rules</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/rules/cpu">cpu/</a></td><td>tessera-cpu</td><td>CPU kernels for matmuls: short bodies over a dot primitive</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/rules/device">device/</a></td><td>tessera-device</td><td>Device headers, import scanner, host proxy and launcher for recipe kernels</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/rules/device">device/</a></td><td>tessera-device</td><td>Device headers, import scanner, host proxy and launcher for recipe kernels</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/rules/device">device/</a></td><td>tessera-device</td><td>Device headers, import scanner, host proxy and launcher for recipe kernels</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/rules/guest">guest/</a></td><td>tessera-guest</td><td>The recipe module ABI: memory hand-off and the table format</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/rules/kernels">kernels/</a></td><td>tessera-kernels</td><td>Kernels and the lowering rules for each</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/rules/linear">linear/</a></td><td>tessera-linear</td><td>Generated matmul kernels, registered as lowering rules</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/rules/rules">rules/</a></td><td>tessera-rules</td><td>Links every lowering rule crate</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/rules/wasm">wasm/</a></td><td>tessera-wasm</td><td>Lowering rules a sandboxed wasm module proposes, over JSON</td></tr>
<tr><td>&emsp;<a href="engine/settings">settings/</a></td><td>tessera-engine-settings</td><td>Knobs the front end sets once and the engine reads</td></tr>
<tr><td colspan="3"><a href="examples"><b>examples/</b></a></td></tr>
<tr><td colspan="3">&emsp;<a href="examples/recipes"><b>recipes/</b></a></td></tr>
<tr><td colspan="3">&emsp;&emsp;<a href="examples/recipes/glm53-moe"><b>glm53-moe/</b></a></td></tr>
<tr><td>&emsp;&emsp;&emsp;<a href="examples/recipes/glm53-moe/model">model/</a></td><td>tessera-glm53-moe-module</td><td>GLM-5.3's routed block as one CUDA kernel, as a recipe module</td></tr>
<tr><td colspan="3">&emsp;&emsp;<a href="examples/recipes/tessera-small"><b>tessera-small/</b></a></td></tr>
<tr><td>&emsp;&emsp;&emsp;<a href="examples/recipes/tessera-small/model">model/</a></td><td>tessera-sprite-module</td><td>tessera-small's graph and tables, as a recipe module</td></tr>
<tr><td colspan="3"><a href="frontend"><b>frontend/</b></a></td></tr>
<tr><td>&emsp;<a href="frontend/bundle">bundle/</a></td><td>tessera-bundle</td><td>A Hugging Face model directory as graph, weights, tokenizer and template</td></tr>
<tr><td>&emsp;<a href="frontend/checkpoint">checkpoint/</a></td><td>tessera-checkpoint</td><td>Reads weights from safetensors files</td></tr>
<tr><td>&emsp;<a href="frontend/cli">cli/</a></td><td>tessera-cli</td><td>The tesserax command: opens a recipe, then serves, fetches or runs tasks</td></tr>
<tr><td>&emsp;<a href="frontend/comm">comm/</a></td><td>tessera-frontend-comm</td><td>Joins ranks over TCP or a shared file for the engine</td></tr>
<tr><td>&emsp;<a href="frontend/eval">eval/</a></td><td>tessera-eval</td><td>Measures a recipe's model: batch speed, and parity with a reference</td></tr>
<tr><td>&emsp;<a href="frontend/grammar">grammar/</a></td><td>tessera-grammar</td><td>Grammars that limit which tokens a decode step may pick</td></tr>
<tr><td>&emsp;<a href="frontend/recipe">recipe/</a></td><td>tessera-recipe</td><td>A model, weights, and plan and serve settings, from disk or git</td></tr>
<tr><td>&emsp;<a href="frontend/serve">serve/</a></td><td>tessera-serve</td><td>OpenAI-shaped HTTP routes over generation, chat and embedding</td></tr>
<tr><td>&emsp;<a href="frontend/settings">settings/</a></td><td>tessera-frontend-settings</td><td>Settings read once from the environment, passed down</td></tr>
<tr><td>&emsp;<a href="frontend/tasks">tasks/</a></td><td>tessera-tasks</td><td>Text in, answers out: generation and tagging over a loaded model</td></tr>
<tr><td colspan="3"><a href="tools"><b>tools/</b></a></td></tr>
<tr><td>&emsp;<a href="tools/calibrate">calibrate/</a></td><td>tessera-calibrate</td><td>Times kernels on this machine and stores what they cost</td></tr>
<tr><td>&emsp;<a href="tools/check">check/</a></td><td>tessera-check</td><td>Runs the style checks, for the terminal or one PR comment</td></tr>
<tr><td>&emsp;<a href="tools/findings">findings/</a></td><td>tessera-findings</td><td>The JSON each check writes, for tessera-check to render</td></tr>
<tr><td>&emsp;<a href="tools/inspect">inspect/</a></td><td>tessera-inspect</td><td>Draws a model's graph, its lowering per rank, and tokens between ranks</td></tr>
<tr><td>&emsp;<a href="tools/lint">lint/</a></td><td>tessera-lint</td><td>The repository's code-shape rules, run by CI and the pre-commit hook</td></tr>
<tr><td>&emsp;<a href="tools/load">load/</a></td><td>tessera-load</td><td>Drives concurrent requests against tessera-serve and reports throughput</td></tr>
<tr><td>&emsp;<a href="tools/parity">parity/</a></td><td>tessera-parity</td><td>Every model against its reference, the interpreter and the device</td></tr>
<tr><td>&emsp;<a href="tools/publish">publish/</a></td><td>tessera-publish</td><td>Merges the workspace crates into the crates that go to crates.io</td></tr>
<tr><td>&emsp;<a href="tools/ranks">ranks/</a></td><td>tessera-ranks</td><td>Runs one rank of a sharded Qwen3 forward pass, joined over TCP</td></tr>
<tr><td>&emsp;<a href="tools/ranks">ranks/</a></td><td>tessera-ranks</td><td>Runs one rank of a sharded Qwen3 forward pass, joined over TCP</td></tr>
<tr><td>&emsp;<a href="tools/ranks">ranks/</a></td><td>tessera-ranks</td><td>Runs one rank of a sharded Qwen3 forward pass, joined over TCP</td></tr>
<tr><td>&emsp;<a href="tools/shape">shape/</a></td><td>tessera-shape</td><td>Line shapes, with names, numbers and strings collapsed, and merge conflicts</td></tr>
<tr><td>&emsp;<a href="tools/simcheck">simcheck/</a></td><td>tessera-simcheck</td><td>Fails on near-copies of code, found with EmbeddingGemma 2 embeddings</td></tr>
<tr><td>&emsp;<a href="tools/style">style/</a></td><td>tessera-style</td><td>Comment and prose style rules, as part-of-speech patterns</td></tr>
</table>
