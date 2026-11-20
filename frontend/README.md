# frontend

The frontend is responsible for loading weights, mapping KV caches to real memory or storage,
speaking front-end protocols like HTTP, WebSockets or image formats, and translating inputs
like text and images to tokens for the engine to work with.

<!-- crate table -->
<table>
<tr><th>Folder</th><th>Crate</th><th>Description</th></tr>
<tr><td><a href="bundle">bundle/</a></td><td>sensai-bundle</td><td>A Hugging Face model directory as graph, weights, tokenizer and template</td></tr>
<tr><td><a href="checkpoint">checkpoint/</a></td><td>sensai-checkpoint</td><td>Reads weights from safetensors files</td></tr>
<tr><td><a href="cli">cli/</a></td><td>sensai-cli</td><td>The sensai command: opens a recipe, then serves, fetches or runs tasks</td></tr>
<tr><td><a href="comm">comm/</a></td><td>sensai-frontend-comm</td><td>Joins ranks over TCP or a shared file for the engine</td></tr>
<tr><td><a href="eval">eval/</a></td><td>sensai-eval</td><td>Measures a recipe's model: batch speed, and parity with a reference</td></tr>
<tr><td><a href="grammar">grammar/</a></td><td>sensai-grammar</td><td>Grammars that limit which tokens a decode step may pick</td></tr>
<tr><td><a href="recipe">recipe/</a></td><td>sensai-recipe</td><td>A model, weights, and plan and serve settings, from disk or git</td></tr>
<tr><td><a href="serve">serve/</a></td><td>sensai-serve</td><td>OpenAI-shaped HTTP routes over generation, chat and embedding</td></tr>
<tr><td><a href="settings">settings/</a></td><td>sensai-frontend-settings</td><td>Settings read once from the environment, passed down</td></tr>
<tr><td><a href="tasks">tasks/</a></td><td>sensai-tasks</td><td>Text in, answers out: generation and tagging over a loaded model</td></tr>
</table>
