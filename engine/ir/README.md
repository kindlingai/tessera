# engine/ir

<!-- crate table -->
Crate | Description
--- | ---
[graph](graph) | The model graph: typed nodes, symbolic shapes, a JSON form
[interp](interp) | Runs a graph op by op on the CPU: the reference for every plan
[mem](mem) | Buffers in zones, and typed views onto them
[testkit](testkit) | Helpers that tests share: noise, comparisons and scratch files
[trace](trace) | Builds graphs by running model descriptions
[trace-derive](trace-derive) | #[derive(Weights)] for model descriptions
