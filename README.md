# tessera

tessera is an engine to build engines.

In tessera, you define the shape of your model in code, and the engine makes an optimized engine for your
hardware, automatically.

## State

tessera is not alpha or beta. It's an experiment right now and no numbers are available.

## Supported Hardware

 - CUDA (NVIDIA)
 - ROCm (AMD)
 - Metal (Apple)
 - CPU (any)

## Supported Models

 - nanogpt (barely)
 - qwen (barely)

<!-- crate table -->
Crate | Description
--- | ---
[engine](engine/engine) | Token ids in, token ids out: a stepper graph on a runner
[gemm](engine/gpu/gemm) | The GEMM family: tiled Metal and CUDA bodies with a shared CPU oracle
[gpukit](engine/gpu/gpukit) | GPU compute infrastructure: pipeline compilation, specialization, caching, buffers
[graph](engine/ir/graph) | The model graph: typed nodes, symbolic shapes, a JSON form
[interp](engine/ir/interp) | Runs a graph op by op on the CPU: the reference for every plan
[mem](engine/ir/mem) | Buffers in zones, and typed views onto them
[trace](engine/ir/trace) | Builds graphs by running model descriptions
[trace-derive](engine/ir/trace-derive) | #[derive(Weights)] for model descriptions
[models](engine/models) | Model descriptions, traced into graphs
[codegen](engine/plan/codegen) | Generates kernel text for every dialect, and counts what it costs
[kernel-macro](engine/plan/kernel-macro) | #[kernel]: a Rust function body as a kernel tree
[lower](engine/plan/lower) | Lowers a graph to a plan of kernel launches
[sim](engine/plan/sim) | Analytic models of GPUs: what a launch should cost
[kernels](engine/rules/kernels) | Kernels and the lowering rules that propose them
[linear](engine/rules/linear) | Generated matmul kernels, registered as lowering rules
[checkpoint](frontend/checkpoint) | Reads weights from safetensors files
[eval](frontend/eval) | Measures a model under a recipe: speed per batch, parity against a reference
[hf](frontend/hf) | A Hugging Face model directory as graph, weights, tokenizer and template
[serve](frontend/serve) | An HTTP front end: text in, streamed text out, over the engine
[settings](frontend/settings) | Settings read once from the environment, passed down
[calibrate](tools/calibrate) | Times kernels on this machine and stores what they cost
[tessera-lint](tools/lint) | The repository's code-shape rules, run by CI and the pre-commit hook
[shape](tools/shape) | Line shapes: a line with names, numbers and strings collapsed
[tessera-simcheck](tools/simcheck) | Fails on near-copies of code, found with EmbeddingGemma 2 embeddings
