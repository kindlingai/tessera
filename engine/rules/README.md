# engine/rules

<!-- crate table -->
<table>
<tr><th>Folder</th><th>Crate</th><th>Description</th></tr>
<tr><td><a href="attention">attention/</a></td><td>sensai-attention</td><td>Attention kernels for the CPU and the GPU, registered as lowering rules</td></tr>
<tr><td><a href="cpu">cpu/</a></td><td>sensai-cpu</td><td>CPU kernels for matmuls: short bodies over a dot primitive</td></tr>
<tr><td><a href="device">device/</a></td><td>sensai-device</td><td>Device headers, import scanner and host proxy for whole-model recipe kernels</td></tr>
<tr><td><a href="guest">guest/</a></td><td>sensai-guest</td><td>The recipe module ABI: memory hand-off and the table format</td></tr>
<tr><td><a href="kernels">kernels/</a></td><td>sensai-kernels</td><td>Kernels and the lowering rules for each</td></tr>
<tr><td><a href="linear">linear/</a></td><td>sensai-linear</td><td>Generated matmul kernels, registered as lowering rules</td></tr>
<tr><td><a href="rules">rules/</a></td><td>sensai-rules</td><td>Links every lowering rule crate</td></tr>
<tr><td><a href="wasm">wasm/</a></td><td>sensai-wasm</td><td>Lowering rules a sandboxed wasm module proposes, over JSON</td></tr>
</table>
