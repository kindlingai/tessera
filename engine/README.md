# engine

The engine crates deal with the GPU/CPU, and work entirely in preloaded raw weights
and tokens.

<!-- crate table -->
<table>
<tr><th>Folder</th><th>Crate</th><th>Description</th></tr>
<tr><td><a href="engine">engine/</a></td><td>tessera-engine</td><td>Runs graphs: a decode loop over a stepper, or one forward launch</td></tr>
<tr><td colspan="3"><a href="gpu"><b>gpu/</b></a></td></tr>
<tr><td>&emsp;<a href="gpu/dl">dl/</a></td><td>tessera-dl</td><td>dlopen for the runtimes that load at run time</td></tr>
<tr><td>&emsp;<a href="gpu/gpukit">gpukit/</a></td><td>tessera-gpukit</td><td>The GPU runtimes tessera loads at run time: pipelines and buffers</td></tr>
<tr><td>&emsp;<a href="gpu/platform">platform/</a></td><td>tessera-platform</td><td>Plain hardware description: platforms, limits and the per-architecture table</td></tr>
<tr><td>&emsp;<a href="gpu/work">work/</a></td><td>tessera-work</td><td>A launch's work: ops per trip and the load they make</td></tr>
<tr><td><a href="hooks">hooks/</a></td><td>tessera-engine-hooks</td><td>Engine calls and types that speculative decoders share with the engine</td></tr>
<tr><td colspan="3"><a href="ir"><b>ir/</b></a></td></tr>
<tr><td>&emsp;<a href="ir/graph">graph/</a></td><td>tessera-graph</td><td>The model graph: typed nodes, symbolic shapes, a JSON form</td></tr>
<tr><td>&emsp;<a href="ir/interp">interp/</a></td><td>tessera-interp</td><td>The CPU reference for every plan: runs a graph op by op</td></tr>
<tr><td>&emsp;<a href="ir/kv">kv/</a></td><td>tessera-kv</td><td>Pages of kept rows, shared between runs by refcount</td></tr>
<tr><td>&emsp;<a href="ir/mem">mem/</a></td><td>tessera-mem</td><td>Buffers in zones, typed views onto them, and storage forms</td></tr>
<tr><td>&emsp;<a href="ir/testkit">testkit/</a></td><td>tessera-testkit</td><td>Helpers that tests share: noise, comparisons and scratch files</td></tr>
<tr><td>&emsp;<a href="ir/trace">trace/</a></td><td>tessera-trace</td><td>Builds graphs by running model descriptions</td></tr>
<tr><td>&emsp;<a href="ir/trace-derive">trace-derive/</a></td><td>tessera-trace-derive</td><td>#[derive(Weights)] for model descriptions</td></tr>
<tr><td>&emsp;<a href="ir/turns">turns/</a></td><td>tessera-turns</td><td>Turns between models on one device, by load cost</td></tr>
<tr><td><a href="models">models/</a></td><td>tessera-models</td><td>Model descriptions, traced into graphs</td></tr>
<tr><td colspan="3"><a href="plan"><b>plan/</b></a></td></tr>
<tr><td>&emsp;<a href="plan/codegen">codegen/</a></td><td>tessera-codegen</td><td>Generates kernel text for every dialect, and counts what it costs</td></tr>
<tr><td>&emsp;<a href="plan/comm">comm/</a></td><td>tessera-engine-comm</td><td>Moves bytes between ranks over sockets, shared memory or RDMA</td></tr>
<tr><td>&emsp;<a href="plan/host">host/</a></td><td>tessera-host</td><td>Runs kernel bodies on the host, each group in lockstep</td></tr>
<tr><td>&emsp;<a href="plan/jit">jit/</a></td><td>tessera-jit</td><td>Compiles kernel bodies for the host with Cranelift as plans are made</td></tr>
<tr><td>&emsp;<a href="plan/kernel-macro">kernel-macro/</a></td><td>tessera-kernel-macro</td><td>#[kernel]: a Rust function body as a kernel tree</td></tr>
<tr><td>&emsp;<a href="plan/lower">lower/</a></td><td>tessera-lower</td><td>Lowers a graph to a plan of kernel launches</td></tr>
<tr><td>&emsp;<a href="plan/run">run/</a></td><td>tessera-run</td><td>Runs lowered plans on the host, Metal, CUDA or ROCm</td></tr>
<tr><td>&emsp;<a href="plan/sim">sim/</a></td><td>tessera-sim</td><td>Analytic models of GPUs: what a launch should cost</td></tr>
<tr><td colspan="3"><a href="rules"><b>rules/</b></a></td></tr>
<tr><td>&emsp;<a href="rules/attention">attention/</a></td><td>tessera-attention</td><td>Attention kernels for the CPU and the GPU, registered as lowering rules</td></tr>
<tr><td>&emsp;<a href="rules/cpu">cpu/</a></td><td>tessera-cpu</td><td>CPU kernels for matmuls: short bodies over a dot primitive</td></tr>
<tr><td>&emsp;<a href="rules/device">device/</a></td><td>tessera-device</td><td>Device headers, import scanner, host proxy and launcher for recipe kernels</td></tr>
<tr><td>&emsp;<a href="rules/device-grid">device-grid/</a></td><td>tessera-device-grid</td><td>Persistent-grid collectives and the resident whole-step decoder</td></tr>
<tr><td>&emsp;<a href="rules/device-linear">device-linear/</a></td><td>tessera-device-linear</td><td>Device CUDA linear matmul, trellis MMA, and decode GEMV rules</td></tr>
<tr><td>&emsp;<a href="rules/device-rules">device-rules/</a></td><td>tessera-device-rules</td><td>Device CUDA rules for attention and calibration</td></tr>
<tr><td>&emsp;<a href="rules/guest">guest/</a></td><td>tessera-guest</td><td>The recipe module ABI: memory hand-off and the table format</td></tr>
<tr><td>&emsp;<a href="rules/kernels">kernels/</a></td><td>tessera-kernels</td><td>Kernels and the lowering rules for each</td></tr>
<tr><td>&emsp;<a href="rules/linear">linear/</a></td><td>tessera-linear</td><td>Generated matmul kernels, registered as lowering rules</td></tr>
<tr><td>&emsp;<a href="rules/rules">rules/</a></td><td>tessera-rules</td><td>Links every lowering rule crate</td></tr>
<tr><td>&emsp;<a href="rules/wasm">wasm/</a></td><td>tessera-wasm</td><td>Lowering rules a sandboxed wasm module proposes, over JSON</td></tr>
<tr><td><a href="settings">settings/</a></td><td>tessera-engine-settings</td><td>Knobs the front end sets once and the engine reads</td></tr>
</table>
