# engine

The engine crates deal with the GPU/CPU, and work entirely in preloaded raw weights
and tokens.

<!-- crate table -->
<table>
<tr><th>Crate</th><th>Description</th></tr>
<tr><td><a href="engine">sensai-engine</a></td><td>Graphs on runners: a decode loop over a stepper, or one launch of a forward graph</td></tr>
<tr><td colspan="2"><a href="gpu"><b>gpu/</b></a></td></tr>
<tr><td>&emsp;<a href="gpu/dl">sensai-dl</a></td><td>dlopen for the runtimes that load at run time</td></tr>
<tr><td>&emsp;<a href="gpu/gpukit">sensai-gpukit</a></td><td>The GPU runtimes tessera loads at run time: pipelines and buffers</td></tr>
<tr><td>&emsp;<a href="gpu/platform">sensai-platform</a></td><td>Plain hardware description: platforms, limits and the per-architecture table</td></tr>
<tr><td>&emsp;<a href="gpu/work">sensai-work</a></td><td>What a launch does: ops per trip and the load they make</td></tr>
<tr><td colspan="2"><a href="ir"><b>ir/</b></a></td></tr>
<tr><td>&emsp;<a href="ir/graph">sensai-graph</a></td><td>The model graph: typed nodes, symbolic shapes, a JSON form</td></tr>
<tr><td>&emsp;<a href="ir/interp">sensai-interp</a></td><td>Runs a graph op by op on the CPU: the reference for every plan</td></tr>
<tr><td>&emsp;<a href="ir/kv">sensai-kv</a></td><td>Pages of kept rows, shared between runs by refcount</td></tr>
<tr><td>&emsp;<a href="ir/mem">sensai-mem</a></td><td>Buffers in zones, typed views onto them, and storage forms</td></tr>
<tr><td>&emsp;<a href="ir/testkit">sensai-testkit</a></td><td>Helpers that tests share: noise, comparisons and scratch files</td></tr>
<tr><td>&emsp;<a href="ir/trace">sensai-trace</a></td><td>Builds graphs by running model descriptions</td></tr>
<tr><td>&emsp;<a href="ir/trace-derive">sensai-trace-derive</a></td><td>#[derive(Weights)] for model descriptions</td></tr>
<tr><td><a href="models">sensai-models</a></td><td>Model descriptions, traced into graphs</td></tr>
<tr><td colspan="2"><a href="plan"><b>plan/</b></a></td></tr>
<tr><td>&emsp;<a href="plan/codegen">sensai-codegen</a></td><td>Generates kernel text for every dialect, and counts what it costs</td></tr>
<tr><td>&emsp;<a href="plan/comm">sensai-engine-comm</a></td><td>Moves bytes between ranks over sockets, shared memory or RDMA</td></tr>
<tr><td>&emsp;<a href="plan/host">sensai-host</a></td><td>Runs kernel bodies on the host, each group in lockstep</td></tr>
<tr><td>&emsp;<a href="plan/jit">sensai-jit</a></td><td>Kernel bodies compiled for the host with Cranelift when a plan is made</td></tr>
<tr><td>&emsp;<a href="plan/kernel-macro">sensai-kernel-macro</a></td><td>#[kernel]: a Rust function body as a kernel tree</td></tr>
<tr><td>&emsp;<a href="plan/lower">sensai-lower</a></td><td>Lowers a graph to a plan of kernel launches</td></tr>
<tr><td>&emsp;<a href="plan/run">sensai-run</a></td><td>Runs lowered plans on the host, Metal, CUDA or ROCm</td></tr>
<tr><td>&emsp;<a href="plan/sim">sensai-sim</a></td><td>Analytic models of GPUs: what a launch should cost</td></tr>
<tr><td colspan="2"><a href="rules"><b>rules/</b></a></td></tr>
<tr><td>&emsp;<a href="rules/attention">sensai-attention</a></td><td>Attention kernels for the CPU and the GPU, registered as lowering rules</td></tr>
<tr><td>&emsp;<a href="rules/cpu">sensai-cpu</a></td><td>CPU kernels for matmuls: short bodies over a dot primitive</td></tr>
<tr><td>&emsp;<a href="rules/guest">sensai-guest</a></td><td>What a recipe module links: memory hand-off and the table format</td></tr>
<tr><td>&emsp;<a href="rules/kernels">sensai-kernels</a></td><td>Kernels and the lowering rules that propose them</td></tr>
<tr><td>&emsp;<a href="rules/linear">sensai-linear</a></td><td>Generated matmul kernels, registered as lowering rules</td></tr>
<tr><td>&emsp;<a href="rules/rules">sensai-rules</a></td><td>Links every lowering rule crate</td></tr>
<tr><td>&emsp;<a href="rules/wasm">sensai-wasm</a></td><td>Lowering rules a sandboxed wasm module proposes, over JSON</td></tr>
<tr><td><a href="settings">sensai-engine-settings</a></td><td>Knobs the front end sets once and the engine reads</td></tr>
</table>
