# engine/ir

<!-- crate table -->
<table>
<tr><th>Crate</th><th>Description</th></tr>
<tr><td><a href="graph">sensai-graph</a></td><td>The model graph: typed nodes, symbolic shapes, a JSON form</td></tr>
<tr><td><a href="interp">sensai-interp</a></td><td>The CPU reference for every plan: runs a graph op by op</td></tr>
<tr><td><a href="kv">sensai-kv</a></td><td>Pages of kept rows, shared between runs by refcount</td></tr>
<tr><td><a href="mem">sensai-mem</a></td><td>Buffers in zones, typed views onto them, and storage forms</td></tr>
<tr><td><a href="testkit">sensai-testkit</a></td><td>Helpers that tests share: noise, comparisons and scratch files</td></tr>
<tr><td><a href="trace">sensai-trace</a></td><td>Builds graphs by running model descriptions</td></tr>
<tr><td><a href="trace-derive">sensai-trace-derive</a></td><td>#[derive(Weights)] for model descriptions</td></tr>
</table>
