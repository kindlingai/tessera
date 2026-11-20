# engine/plan

<!-- crate table -->
<table>
<tr><th>Folder</th><th>Crate</th><th>Description</th></tr>
<tr><td><a href="codegen">codegen/</a></td><td>sensai-codegen</td><td>Generates kernel text for every dialect, and counts what it costs</td></tr>
<tr><td><a href="comm">comm/</a></td><td>sensai-engine-comm</td><td>Moves bytes between ranks over sockets, shared memory or RDMA</td></tr>
<tr><td><a href="host">host/</a></td><td>sensai-host</td><td>Runs kernel bodies on the host, each group in lockstep</td></tr>
<tr><td><a href="jit">jit/</a></td><td>sensai-jit</td><td>Compiles kernel bodies for the host with Cranelift as plans are made</td></tr>
<tr><td><a href="kernel-macro">kernel-macro/</a></td><td>sensai-kernel-macro</td><td>#[kernel]: a Rust function body as a kernel tree</td></tr>
<tr><td><a href="lower">lower/</a></td><td>sensai-lower</td><td>Lowers a graph to a plan of kernel launches</td></tr>
<tr><td><a href="run">run/</a></td><td>sensai-run</td><td>Runs lowered plans on the host, Metal, CUDA or ROCm</td></tr>
<tr><td><a href="sim">sim/</a></td><td>sensai-sim</td><td>Analytic models of GPUs: what a launch should cost</td></tr>
</table>
