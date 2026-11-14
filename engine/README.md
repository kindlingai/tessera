# engine

The engine crates deal with the GPU/CPU, and work entirely in preloaded raw weights
and tokens.

<!-- crate table -->
Crate | Description
--- | ---
[engine](engine) | Token ids in, token ids out: a stepper graph on a runner
[gemm](gpu/gemm) | The GEMM family: tiled Metal and CUDA bodies with a shared CPU oracle
[gpukit](gpu/gpukit) | GPU compute infrastructure: pipeline compilation, specialization, caching, buffers
[graph](ir/graph) | The model graph: typed nodes, symbolic shapes, a JSON form
[interp](ir/interp) | Runs a graph op by op on the CPU: the reference for every plan
[mem](ir/mem) | Buffers in zones, and typed views onto them
[trace](ir/trace) | Builds graphs by running model descriptions
[trace-derive](ir/trace-derive) | #[derive(Weights)] for model descriptions
[models](models) | Model descriptions, traced into graphs
[codegen](plan/codegen) | Generates kernel text for every dialect, and counts what it costs
[kernel-macro](plan/kernel-macro) | #[kernel]: a Rust function body as a kernel tree
[lower](plan/lower) | Lowers a graph to a plan of kernel launches
[sim](plan/sim) | Analytic models of GPUs: what a launch should cost
[kernels](rules/kernels) | Kernels and the lowering rules that propose them
[linear](rules/linear) | Generated matmul kernels, registered as lowering rules
