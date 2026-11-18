# frontend

The frontend is responsible for loading weights, mapping KV caches to real memory or storage,
speaking front-end protocols like HTTP, WebSockets or image formats, and translating inputs
like text and images to tokens for the engine to work with.

<!-- crate table -->
<table>
<tr><th>Crate</th><th>Description</th></tr>
<tr><td><a href="bundle">sensai-bundle</a></td><td>A Hugging Face model directory as graph, weights, tokenizer and template</td></tr>
<tr><td><a href="checkpoint">sensai-checkpoint</a></td><td>Reads weights from safetensors files</td></tr>
<tr><td><a href="cli">sensai-cli</a></td><td>The sensai command: opens a recipe, then serves, fetches or runs tasks</td></tr>
<tr><td><a href="comm">sensai-frontend-comm</a></td><td>Joins ranks over TCP or a shared file for the engine</td></tr>
<tr><td><a href="eval">sensai-eval</a></td><td>Measures a recipe's model: batch speed, and parity with a reference</td></tr>
<tr><td><a href="grammar">sensai-grammar</a></td><td>Grammars that limit which tokens a decode step may pick</td></tr>
<tr><td><a href="recipe">sensai-recipe</a></td><td>A model, weights, and plan and serve settings, from disk or git</td></tr>
<tr><td><a href="serve">sensai-serve</a></td><td>OpenAI-shaped HTTP routes over generation, chat and embedding</td></tr>
<tr><td><a href="settings">sensai-frontend-settings</a></td><td>Settings read once from the environment, passed down</td></tr>
<tr><td><a href="tasks">sensai-tasks</a></td><td>Text in, answers out: generation and tagging over a loaded model</td></tr>
</table>
