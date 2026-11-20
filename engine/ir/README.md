# engine/ir

<!-- crate table -->
<table>
<tr><th>Folder</th><th>Crate</th><th>Description</th></tr>
<tr><td><a href="graph">graph/</a></td><td>sensai-graph</td><td>The model graph: typed nodes, symbolic shapes, a JSON form</td></tr>
<tr><td><a href="interp">interp/</a></td><td>sensai-interp</td><td>The CPU reference for every plan: runs a graph op by op</td></tr>
<tr><td><a href="kv">kv/</a></td><td>sensai-kv</td><td>Pages of kept rows, shared between runs by refcount</td></tr>
<tr><td><a href="mem">mem/</a></td><td>sensai-mem</td><td>Buffers in zones, typed views onto them, and storage forms</td></tr>
<tr><td><a href="testkit">testkit/</a></td><td>sensai-testkit</td><td>Helpers that tests share: noise, comparisons and scratch files</td></tr>
<tr><td><a href="trace">trace/</a></td><td>sensai-trace</td><td>Builds graphs by running model descriptions</td></tr>
<tr><td><a href="trace-derive">trace-derive/</a></td><td>sensai-trace-derive</td><td>#[derive(Weights)] for model descriptions</td></tr>
</table>
