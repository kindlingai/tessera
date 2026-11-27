# engine/rules

<!-- crate table -->
<table>
<tr><th>Folder</th><th>Crate</th><th>Description</th></tr>
<tr><td><a href="attention">attention/</a></td><td>tessera-attention</td><td>Attention kernels for the CPU and the GPU, registered as lowering rules</td></tr>
<tr><td><a href="cpu">cpu/</a></td><td>tessera-cpu</td><td>CPU kernels for matmuls: short bodies over a dot primitive</td></tr>
<tr><td><a href="device">device/</a></td><td>tessera-device</td><td>Device headers, import scanner, host proxy and launcher for recipe kernels</td></tr>
<tr><td><a href="device-grid">device-grid/</a></td><td>tessera-device-grid</td><td>Persistent-grid collectives and the resident whole-step decoder</td></tr>
<tr><td><a href="device-linear">device-linear/</a></td><td>tessera-device-linear</td><td>Device CUDA linear matmul, trellis MMA, and decode GEMV rules</td></tr>
<tr><td><a href="device-rules">device-rules/</a></td><td>tessera-device-rules</td><td>Device CUDA rules for attention, MoE and calibration</td></tr>
<tr><td><a href="guest">guest/</a></td><td>tessera-guest</td><td>The recipe module ABI: memory hand-off and the table format</td></tr>
<tr><td><a href="kernels">kernels/</a></td><td>tessera-kernels</td><td>Kernels and the lowering rules for each</td></tr>
<tr><td><a href="linear">linear/</a></td><td>tessera-linear</td><td>Generated matmul kernels, registered as lowering rules</td></tr>
<tr><td><a href="rules">rules/</a></td><td>tessera-rules</td><td>Links every lowering rule crate</td></tr>
<tr><td><a href="wasm">wasm/</a></td><td>tessera-wasm</td><td>Lowering rules a sandboxed wasm module proposes, over JSON</td></tr>
</table>
