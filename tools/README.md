# tools

Build tools that run as part of CI or pre-commit hooks.

<!-- crate table -->
<table>
<tr><th>Folder</th><th>Crate</th><th>Description</th></tr>
<tr><td><a href="calibrate">calibrate/</a></td><td>tessera-calibrate</td><td>Times kernels on this machine and stores what they cost</td></tr>
<tr><td><a href="hessians">hessians/</a></td><td>tessera-hessians</td><td>Layer-streaming BF16 Hessian calibration generator for exlr8</td></tr>
<tr><td><a href="html">html/</a></td><td>tessera-html</td><td>An HTML element builder that escapes all text</td></tr>
<tr><td><a href="inspect">inspect/</a></td><td>tessera-inspect</td><td>Draws a model's graph, its lowering per rank, and tokens between ranks</td></tr>
<tr><td><a href="load">load/</a></td><td>tessera-load</td><td>Drives concurrent requests against tessera-serve and reports throughput</td></tr>
<tr><td><a href="parity">parity/</a></td><td>tessera-parity</td><td>Every model against its reference, the interpreter and the device</td></tr>
<tr><td><a href="publish">publish/</a></td><td>tessera-publish</td><td>Merges the workspace crates into the crates that go to crates.io</td></tr>
<tr><td><a href="ranks">ranks/</a></td><td>tessera-ranks</td><td>Runs one rank of a sharded Qwen3 forward pass, joined over TCP</td></tr>
<tr><td><a href="rebase">rebase/</a></td><td>tessera-rebase</td><td>Rebases a branch when each conflict only adds list lines</td></tr>
<tr><td><a href="references">references/</a></td><td>tessera-references</td><td>Fetches the reference files the parity tests read</td></tr>
<tr><td><a href="tiers">tiers/</a></td><td>tessera-tiers</td><td>A board that sets pull request tiers as a GitHub App</td></tr>
</table>
