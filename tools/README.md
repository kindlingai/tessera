# tools

Build tools that run as part of CI or pre-commit hooks.

<!-- crate table -->
<table>
<tr><th>Crate</th><th>Description</th></tr>
<tr><td><a href="calibrate">sensai-calibrate</a></td><td>Times kernels on this machine and stores what they cost</td></tr>
<tr><td><a href="check">tessera-check</a></td><td>Runs the style checks, for the terminal or one PR comment</td></tr>
<tr><td><a href="findings">sensai-findings</a></td><td>The JSON each check writes, for tessera-check to render</td></tr>
<tr><td><a href="inspect">sensai-inspect</a></td><td>Draws a model's graph, its lowering per rank, and tokens between ranks</td></tr>
<tr><td><a href="lint">tessera-lint</a></td><td>The repository's code-shape rules, run by CI and the pre-commit hook</td></tr>
<tr><td><a href="load">sensai-load</a></td><td>Drives concurrent requests against tessera-serve and reports throughput</td></tr>
<tr><td><a href="parity">sensai-parity</a></td><td>Every model against its reference, the interpreter and the device</td></tr>
<tr><td><a href="publish">tessera-publish</a></td><td>Merges the workspace crates into the crates that go to crates.io</td></tr>
<tr><td><a href="ranks">sensai-ranks</a></td><td>Runs one rank of a sharded Qwen3 forward pass, joined over TCP</td></tr>
<tr><td><a href="shape">sensai-shape</a></td><td>Line shapes, with names, numbers and strings collapsed, and merge conflicts</td></tr>
<tr><td><a href="simcheck">tessera-simcheck</a></td><td>Fails on near-copies of code, found with EmbeddingGemma 2 embeddings</td></tr>
<tr><td><a href="style">tessera-style</a></td><td>Comment and prose style rules, as part-of-speech patterns</td></tr>
</table>
