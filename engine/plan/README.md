# engine/plan

<!-- crate table -->
<table>
<tr><th>Crate</th><th>Description</th></tr>
<tr><td><a href="codegen">sensai-codegen</a></td><td>Generates kernel text for every dialect, and counts what it costs</td></tr>
<tr><td><a href="comm">sensai-engine-comm</a></td><td>Moves bytes between ranks over sockets, shared memory or RDMA</td></tr>
<tr><td><a href="host">sensai-host</a></td><td>Runs kernel bodies on the host, each group in lockstep</td></tr>
<tr><td><a href="jit">sensai-jit</a></td><td>Compiles kernel bodies for the host with Cranelift as plans are made</td></tr>
<tr><td><a href="kernel-macro">sensai-kernel-macro</a></td><td>#[kernel]: a Rust function body as a kernel tree</td></tr>
<tr><td><a href="lower">sensai-lower</a></td><td>Lowers a graph to a plan of kernel launches</td></tr>
<tr><td><a href="run">sensai-run</a></td><td>Runs lowered plans on the host, Metal, CUDA or ROCm</td></tr>
<tr><td><a href="sim">sensai-sim</a></td><td>Analytic models of GPUs: what a launch should cost</td></tr>
</table>
