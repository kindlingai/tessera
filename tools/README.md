# tools

Build tools that run as part of CI or pre-commit hooks.

<!-- crate table -->
<table>
<tr><th>Folder</th><th>Crate</th><th>Description</th></tr>
<tr><td><a href="calibrate">calibrate/</a></td><td>tessera-calibrate</td><td>Times kernels on this machine and stores what they cost</td></tr>
<tr><td><a href="check">check/</a></td><td>tessera-check</td><td>Runs the style checks, for the terminal or one PR comment</td></tr>
<tr><td><a href="findings">findings/</a></td><td>tessera-findings</td><td>The JSON each check writes, for tessera-check to render</td></tr>
<tr><td><a href="inspect">inspect/</a></td><td>tessera-inspect</td><td>Draws a model's graph, its lowering per rank, and tokens between ranks</td></tr>
<tr><td><a href="lint">lint/</a></td><td>tessera-lint</td><td>The repository's code-shape rules, run by CI and the pre-commit hook</td></tr>
<tr><td><a href="load">load/</a></td><td>tessera-load</td><td>Drives concurrent requests against tessera-serve and reports throughput</td></tr>
<tr><td><a href="maps">maps/</a></td><td>tessera-maps</td><td>Checks task maps: ordered verb steps over code references</td></tr>
<tr><td><a href="parity">parity/</a></td><td>tessera-parity</td><td>Every model against its reference, the interpreter and the device</td></tr>
<tr><td><a href="publish">publish/</a></td><td>tessera-publish</td><td>Merges the workspace crates into the crates that go to crates.io</td></tr>
<tr><td><a href="ranks">ranks/</a></td><td>tessera-ranks</td><td>Runs one rank of a sharded Qwen3 forward pass, joined over TCP</td></tr>
<tr><td><a href="references">references/</a></td><td>tessera-references</td><td>Fetches the reference files the parity tests read</td></tr>
<tr><td><a href="shape">shape/</a></td><td>tessera-shape</td><td>Line shapes, with names, numbers and strings collapsed, and merge conflicts</td></tr>
<tr><td><a href="simcheck">simcheck/</a></td><td>tessera-simcheck</td><td>Fails on near-copies of code, found with EmbeddingGemma 2 embeddings</td></tr>
<tr><td><a href="style">style/</a></td><td>tessera-style</td><td>Comment and prose style rules, as part-of-speech patterns</td></tr>
</table>
