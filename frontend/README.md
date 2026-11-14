# frontend

The frontend is responsible for loading weights, mapping KV caches to real memory or storage,
speaking front-end protocols like HTTP, WebSockets or image formats, and translating inputs
like text and images to tokens for the engine to work with.

<!-- crate table -->
Crate | Description
--- | ---
[bundle](bundle) | A Hugging Face model directory as graph, weights, tokenizer and template
[checkpoint](checkpoint) | Reads weights from safetensors files
[eval](eval) | Measures a model under a recipe: speed per batch, parity against a reference
[serve](serve) | An HTTP front end: text in, streamed text out, over the engine
[settings](settings) | Settings read once from the environment, passed down
