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
<tr><th>Crate</th><th>Description</th></tr>
<tr><td colspan="2"><a href="engine"><b>engine/</b></a></td></tr>
<tr><td>&emsp;<a href="engine/engine">sensai-engine</a></td><td>Runs graphs: a decode loop over a stepper, or one forward launch</td></tr>
<tr><td colspan="2">&emsp;<a href="engine/gpu"><b>gpu/</b></a></td></tr>
<tr><td>&emsp;&emsp;<a href="engine/gpu/dl">sensai-dl</a></td><td>dlopen for the runtimes that load at run time</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/gpu/gpukit">sensai-gpukit</a></td><td>The GPU runtimes tessera loads at run time: pipelines and buffers</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/gpu/platform">sensai-platform</a></td><td>Plain hardware description: platforms, limits and the per-architecture table</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/gpu/work">sensai-work</a></td><td>A launch's work: ops per trip and the load they make</td></tr>
<tr><td colspan="2">&emsp;<a href="engine/ir"><b>ir/</b></a></td></tr>
<tr><td>&emsp;&emsp;<a href="engine/ir/graph">sensai-graph</a></td><td>The model graph: typed nodes, symbolic shapes, a JSON form</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/ir/interp">sensai-interp</a></td><td>The CPU reference for every plan: runs a graph op by op</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/ir/kv">sensai-kv</a></td><td>Pages of kept rows, shared between runs by refcount</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/ir/mem">sensai-mem</a></td><td>Buffers in zones, typed views onto them, and storage forms</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/ir/testkit">sensai-testkit</a></td><td>Helpers that tests share: noise, comparisons and scratch files</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/ir/trace">sensai-trace</a></td><td>Builds graphs by running model descriptions</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/ir/trace-derive">sensai-trace-derive</a></td><td>#[derive(Weights)] for model descriptions</td></tr>
<tr><td>&emsp;<a href="engine/models">sensai-models</a></td><td>Model descriptions, traced into graphs</td></tr>
<tr><td colspan="2">&emsp;<a href="engine/plan"><b>plan/</b></a></td></tr>
<tr><td>&emsp;&emsp;<a href="engine/plan/codegen">sensai-codegen</a></td><td>Generates kernel text for every dialect, and counts what it costs</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/plan/comm">sensai-engine-comm</a></td><td>Moves bytes between ranks over sockets, shared memory or RDMA</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/plan/host">sensai-host</a></td><td>Runs kernel bodies on the host, each group in lockstep</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/plan/jit">sensai-jit</a></td><td>Compiles kernel bodies for the host with Cranelift as plans are made</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/plan/kernel-macro">sensai-kernel-macro</a></td><td>#[kernel]: a Rust function body as a kernel tree</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/plan/lower">sensai-lower</a></td><td>Lowers a graph to a plan of kernel launches</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/plan/run">sensai-run</a></td><td>Runs lowered plans on the host, Metal, CUDA or ROCm</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/plan/sim">sensai-sim</a></td><td>Analytic models of GPUs: what a launch should cost</td></tr>
<tr><td colspan="2">&emsp;<a href="engine/rules"><b>rules/</b></a></td></tr>
<tr><td>&emsp;&emsp;<a href="engine/rules/attention">sensai-attention</a></td><td>Attention kernels for the CPU and the GPU, registered as lowering rules</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/rules/cpu">sensai-cpu</a></td><td>CPU kernels for matmuls: short bodies over a dot primitive</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/rules/device">sensai-device</a></td><td>Device headers, import scanner and host proxy for whole-model recipe kernels</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/rules/guest">sensai-guest</a></td><td>What a recipe module links: memory hand-off and the table format</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/rules/kernels">sensai-kernels</a></td><td>Kernels and the lowering rules for each</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/rules/linear">sensai-linear</a></td><td>Generated matmul kernels, registered as lowering rules</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/rules/rules">sensai-rules</a></td><td>Links every lowering rule crate</td></tr>
<tr><td>&emsp;&emsp;<a href="engine/rules/wasm">sensai-wasm</a></td><td>Lowering rules a sandboxed wasm module proposes, over JSON</td></tr>
<tr><td>&emsp;<a href="engine/settings">sensai-engine-settings</a></td><td>Knobs the front end sets once and the engine reads</td></tr>
<tr><td colspan="2"><a href="examples"><b>examples/</b></a></td></tr>
<tr><td colspan="2">&emsp;<a href="examples/recipes"><b>recipes/</b></a></td></tr>
<tr><td colspan="2">&emsp;&emsp;<a href="examples/recipes/glm53-moe"><b>glm53-moe/</b></a></td></tr>
<tr><td>&emsp;&emsp;&emsp;<a href="examples/recipes/glm53-moe/model">sensai-glm53-moe-module</a></td><td>GLM-5.3's routed block as one CUDA kernel, as a recipe module</td></tr>
<tr><td colspan="2">&emsp;&emsp;<a href="examples/recipes/tessera-small"><b>tessera-small/</b></a></td></tr>
<tr><td>&emsp;&emsp;&emsp;<a href="examples/recipes/tessera-small/model">sensai-sprite-module</a></td><td>tessera-small's graph and tables, as a recipe module</td></tr>
<tr><td colspan="2"><a href="frontend"><b>frontend/</b></a></td></tr>
<tr><td>&emsp;<a href="frontend/bundle">sensai-bundle</a></td><td>A Hugging Face model directory as graph, weights, tokenizer and template</td></tr>
<tr><td>&emsp;<a href="frontend/checkpoint">sensai-checkpoint</a></td><td>Reads weights from safetensors files</td></tr>
<tr><td>&emsp;<a href="frontend/cli">sensai-cli</a></td><td>The sensai command: opens a recipe, then serves, fetches or runs tasks</td></tr>
<tr><td>&emsp;<a href="frontend/comm">sensai-frontend-comm</a></td><td>Joins ranks over TCP or a shared file for the engine</td></tr>
<tr><td>&emsp;<a href="frontend/eval">sensai-eval</a></td><td>Measures a recipe's model: batch speed, and parity with a reference</td></tr>
<tr><td>&emsp;<a href="frontend/grammar">sensai-grammar</a></td><td>Grammars that limit which tokens a decode step may pick</td></tr>
<tr><td>&emsp;<a href="frontend/recipe">sensai-recipe</a></td><td>A model, weights, and plan and serve settings, from disk or git</td></tr>
<tr><td>&emsp;<a href="frontend/serve">sensai-serve</a></td><td>OpenAI-shaped HTTP routes over generation, chat and embedding</td></tr>
<tr><td>&emsp;<a href="frontend/settings">sensai-frontend-settings</a></td><td>Settings read once from the environment, passed down</td></tr>
<tr><td>&emsp;<a href="frontend/tasks">sensai-tasks</a></td><td>Text in, answers out: generation and tagging over a loaded model</td></tr>
<tr><td colspan="2"><a href="tools"><b>tools/</b></a></td></tr>
<tr><td>&emsp;<a href="tools/calibrate">sensai-calibrate</a></td><td>Times kernels on this machine and stores what they cost</td></tr>
<tr><td>&emsp;<a href="tools/check">tessera-check</a></td><td>Runs the style checks, for the terminal or one PR comment</td></tr>
<tr><td>&emsp;<a href="tools/findings">sensai-findings</a></td><td>The JSON each check writes, for tessera-check to render</td></tr>
<tr><td>&emsp;<a href="tools/inspect">sensai-inspect</a></td><td>Draws a model's graph, its lowering per rank, and tokens between ranks</td></tr>
<tr><td>&emsp;<a href="tools/lint">tessera-lint</a></td><td>The repository's code-shape rules, run by CI and the pre-commit hook</td></tr>
<tr><td>&emsp;<a href="tools/load">sensai-load</a></td><td>Drives concurrent requests against tessera-serve and reports throughput</td></tr>
<tr><td>&emsp;<a href="tools/parity">sensai-parity</a></td><td>Every model against its reference, the interpreter and the device</td></tr>
<tr><td>&emsp;<a href="tools/publish">tessera-publish</a></td><td>Merges the workspace crates into the crates that go to crates.io</td></tr>
<tr><td>&emsp;<a href="tools/ranks">sensai-ranks</a></td><td>Runs one rank of a sharded Qwen3 forward pass, joined over TCP</td></tr>
<tr><td>&emsp;<a href="tools/shape">sensai-shape</a></td><td>Line shapes, with names, numbers and strings collapsed, and merge conflicts</td></tr>
<tr><td>&emsp;<a href="tools/simcheck">tessera-simcheck</a></td><td>Fails on near-copies of code, found with EmbeddingGemma 2 embeddings</td></tr>
<tr><td>&emsp;<a href="tools/style">tessera-style</a></td><td>Comment and prose style rules, as part-of-speech patterns</td></tr>
</table>
