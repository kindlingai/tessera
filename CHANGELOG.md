# Changelog

## [0.4.4](https://github.com/kindlingai/tessera/compare/v0.4.3...v0.4.4) (2026-11-25)


### Features

* **bundle:** pick built-in response templates by chat template, GGUF included ([#583](https://github.com/kindlingai/tessera/issues/583)) ([ff792a7](https://github.com/kindlingai/tessera/commit/ff792a7dd0f40640e3725a77f13d1fd33fc993eb))
* **checkpoint:** support manifests on remote HTTP exlr8 releases ([#610](https://github.com/kindlingai/tessera/issues/610)) ([18b69a0](https://github.com/kindlingai/tessera/commit/18b69a0e653ce3a20a18167e46efc171142cb81f))
* **cli:** serve the commands and tasks as MCP tools over stdio ([#617](https://github.com/kindlingai/tessera/issues/617)) ([209bc3c](https://github.com/kindlingai/tessera/commit/209bc3c4f049270d35831514471c3e77611716dd))
* **device:** add grouped SYRK and BF16 routed-expert kernels ([#600](https://github.com/kindlingai/tessera/issues/600)) ([6bc90ba](https://github.com/kindlingai/tessera/commit/6bc90ba0b2c60668d68ed57952ac88b5374c91dc))
* **engine:** interleave chunked prefill with active decode in BatchMtp ([#611](https://github.com/kindlingai/tessera/issues/611)) ([08c93bd](https://github.com/kindlingai/tessera/commit/08c93bd661c75de71f73da182218c734190e1329))
* **graph:** add Role::Calib taps to MLA and Qwen4 MoE layers ([#598](https://github.com/kindlingai/tessera/issues/598)) ([2c909be](https://github.com/kindlingai/tessera/commit/2c909be730d513a329687324f8c6a2a9877d9608))
* **hessians:** add Hessian file writers and the host calibration driver ([#601](https://github.com/kindlingai/tessera/issues/601)) ([47dcb31](https://github.com/kindlingai/tessera/commit/47dcb31698ff94b06b2ffd6f4c15a7b148d25d34))
* **hessians:** add the CUDA runner, kernel bench and parity gate ([#590](https://github.com/kindlingai/tessera/issues/590)) ([7b6d470](https://github.com/kindlingai/tessera/commit/7b6d4700c140d1ef92974bfa19d554dad0d339fe))
* **kv:** support shared paged KV pool capacity via --kv-cache-tokens ([#608](https://github.com/kindlingai/tessera/issues/608)) ([6009d0d](https://github.com/kindlingai/tessera/commit/6009d0df3fd89ec99e538465f61c1b110b523141))
* **serve:** forward clear_thinking chat_template_kwargs ([#613](https://github.com/kindlingai/tessera/issues/613)) ([51ac85f](https://github.com/kindlingai/tessera/commit/51ac85f1a4ba867eac2d642af38208b90a8934b6))


### Bug Fixes

* **device:** keep NaN and one overflow rule for FP16 sum wires ([#616](https://github.com/kindlingai/tessera/issues/616)) ([8a69c2c](https://github.com/kindlingai/tessera/commit/8a69c2c2a07c2238960a1fcb88db32595c3e7087))
* **docker:** install libibverbs runtime and providers in GPU images ([#609](https://github.com/kindlingai/tessera/issues/609)) ([4c679f5](https://github.com/kindlingai/tessera/commit/4c679f500e212158f823200acd963087e3bd058b))
* **graph:** shard residual matmuls read through row gather after trim ([#615](https://github.com/kindlingai/tessera/issues/615)) ([ecd205a](https://github.com/kindlingai/tessera/commit/ecd205a2f8c175775548dc952123814359499781))
* **lower:** keep MoE residual stages in the pinned router form ([#599](https://github.com/kindlingai/tessera/issues/599)) ([c3fbbaa](https://github.com/kindlingai/tessera/commit/c3fbbaa9867b79c9c0f15607dee852bee65ed3cc))
* **recipe:** fetch exlr8 expert widths required by manifest ([#612](https://github.com/kindlingai/tessera/issues/612)) ([224b4fa](https://github.com/kindlingai/tessera/commit/224b4fae5271e7f62d5e1526339306b96b5d4995))


### Performance

* **device:** add cross-CTA split-K for narrow BF16 Tensor-Core matmuls ([#602](https://github.com/kindlingai/tessera/issues/602)) ([da9bea6](https://github.com/kindlingai/tessera/commit/da9bea63af09235238bc80a437430e0a0342efcd))
* **device:** add FP16 wire transport for P2P all-reduce ([#595](https://github.com/kindlingai/tessera/issues/595)) ([fb0756c](https://github.com/kindlingai/tessera/commit/fb0756c92a84a9f1c48e1ab1988b1c08aef1c13a))
* **device:** add opt-in BF16 wire transport for P2P all-reduce ([#592](https://github.com/kindlingai/tessera/issues/592)) ([7ef5bff](https://github.com/kindlingai/tessera/commit/7ef5bfff4711987533f16c186a2028e0559f9944))
* **device:** add Tensor-Core grouped-Q split-K GQA attention for head_dim 128 ([#605](https://github.com/kindlingai/tessera/issues/605)) ([79341eb](https://github.com/kindlingai/tessera/commit/79341eb0d6199ed19c785e761a0cb15343bfd96a))
* **device:** add Tensor-Core grouped-Q split-K MLA attention kernel ([#587](https://github.com/kindlingai/tessera/issues/587)) ([a1c59af](https://github.com/kindlingai/tessera/commit/a1c59afbf050e8328b615efa9a95996b8463150e))
* **device:** use 16x16 BF16 Tensor-Core MMA for stacked_lanes ([#603](https://github.com/kindlingai/tessera/issues/603)) ([dc2113e](https://github.com/kindlingai/tessera/commit/dc2113eebf75f6b8684b483f796fabc9f70d7f1f))
* **device:** use 16x32 BF16 Tensor-Core MMA for stacked_wide ([#594](https://github.com/kindlingai/tessera/issues/594)) ([b6ac50c](https://github.com/kindlingai/tessera/commit/b6ac50c858e5f65daa596b88c572cdeebdb4a32f))
* **device:** vectorize MLA FP8 KV loads, use __expf, and cut barriers ([#597](https://github.com/kindlingai/tessera/issues/597)) ([7f68235](https://github.com/kindlingai/tessera/commit/7f68235dab4c68be34488faae2f3dcfacce12b0f))
* **device:** widen MXFP8 split-K to N&lt;4096 and add r16 split-K ([#591](https://github.com/kindlingai/tessera/issues/591)) ([02a4074](https://github.com/kindlingai/tessera/commit/02a407451551ad4eeb9e698ed092ddf63153e0f3))
* **engine:** wire DFlash selector log-probs into SpecLen confidence buckets ([#588](https://github.com/kindlingai/tessera/issues/588)) ([4416f68](https://github.com/kindlingai/tessera/commit/4416f68fecb917dda068d461a9a747cc60346192))
* **glm53-moe:** use Tensor-Core split-K and register top-k in glm53_route ([#607](https://github.com/kindlingai/tessera/issues/607)) ([de386a0](https://github.com/kindlingai/tessera/commit/de386a0053b759c12c0ef0ca0ffad4f248d81e4d))
* **kernels:** cache pass-1 sums in registers for add_rms_norm ([#604](https://github.com/kindlingai/tessera/issues/604)) ([c1e4ef8](https://github.com/kindlingai/tessera/commit/c1e4ef8f2c4be8df6e867653c05678400995c69d))
* **kernels:** parallelize argmax reduction across 256 threads per row ([#589](https://github.com/kindlingai/tessera/issues/589)) ([3f70bb6](https://github.com/kindlingai/tessera/commit/3f70bb6aca3ac6b63213bb85cee46ca9bdd95407))
* **kernels:** parallelize chunk_top across 128 threads and cache merge_top in shared memory ([#606](https://github.com/kindlingai/tessera/issues/606)) ([61c163a](https://github.com/kindlingai/tessera/commit/61c163ac80c15889246c19f8edb3a5ec1e693819))

## [0.4.3](https://github.com/kindlingai/tessera/compare/v0.4.2...v0.4.3) (2026-11-25)


### Features

* **attention:** fuse Lightning Indexer scoring over kept keys ([#553](https://github.com/kindlingai/tessera/issues/553)) ([3e875ea](https://github.com/kindlingai/tessera/commit/3e875ea1c0e6e82b52ba08d77b51bacc8f3e8ed9))
* **attention:** support narrow bf16 and fp8 KV in listed attention ([#557](https://github.com/kindlingai/tessera/issues/557)) ([8aa647e](https://github.com/kindlingai/tessera/commit/8aa647e0ef28a554379d6b3df7a11a25f7168e39))
* **bundle:** add Qwen3 and Qwen3.5 tool calling ([#572](https://github.com/kindlingai/tessera/issues/572)) ([c982121](https://github.com/kindlingai/tessera/commit/c982121380363b289bb2383984c5d623fe24d050))
* **calibrate:** check device kernels against the host and fallbacks ([#556](https://github.com/kindlingai/tessera/issues/556)) ([115be99](https://github.com/kindlingai/tessera/commit/115be99d5991d93b9f2598ba6001f7af3f8fb844))
* **checkpoint:** add quantization_config tensor-pairing adapter ([#544](https://github.com/kindlingai/tessera/issues/544)) ([a06433b](https://github.com/kindlingai/tessera/commit/a06433bc47a4ab0274ec0ba34fa3cb74cd8683af))
* **cli:** route a bare prompt through Gemma to a tesserax command ([#563](https://github.com/kindlingai/tessera/issues/563)) ([7b51496](https://github.com/kindlingai/tessera/commit/7b51496f9f2d20eb9a8ce235e5ee7f9dd97bbf98))
* **cli:** run routed tasks in-process and answer from their results ([#586](https://github.com/kindlingai/tessera/issues/586)) ([71ecb20](https://github.com/kindlingai/tessera/commit/71ecb20d72f0d87ca179441be0c00f1949f8db0e))
* **kernels:** add radix-select large-k TopK kernel ([#559](https://github.com/kindlingai/tessera/issues/559)) ([31e4a6c](https://github.com/kindlingai/tessera/commit/31e4a6ccacc796839c65a9537ad7a8c38f422231))
* **kernels:** fuse residual add and RMSNorm with multi-output lowering ([#569](https://github.com/kindlingai/tessera/issues/569)) ([85bd1f9](https://github.com/kindlingai/tessera/commit/85bd1f9469b3da46b754299305e417e5e28d146d))
* **lint:** let spec files hold near-copies of each other ([#524](https://github.com/kindlingai/tessera/issues/524)) ([1a37c5e](https://github.com/kindlingai/tessera/commit/1a37c5efd2d24597a1c042ba2eef206346090347))
* **serve:** add --kv-dtype flag and merge recipe float pins ([#564](https://github.com/kindlingai/tessera/issues/564)) ([c69f751](https://github.com/kindlingai/tessera/commit/c69f751dbcd209737a39665d428204eb358e97bc))
* **serve:** add DeepSeek V3 tool calling ([#573](https://github.com/kindlingai/tessera/issues/573)) ([a422cea](https://github.com/kindlingai/tessera/commit/a422cea70a5e931bde4705d1e058935d754dafdf))
* **serve:** add generic response-template tool calling ([#561](https://github.com/kindlingai/tessera/issues/561)) ([b0b58be](https://github.com/kindlingai/tessera/commit/b0b58be5f959a9524e47bf24d6f6119883e29264))
* **serve:** add GLM 5.3 tool calling ([#574](https://github.com/kindlingai/tessera/issues/574)) ([180eb95](https://github.com/kindlingai/tessera/commit/180eb95ec143c182e8deb3702cc691e2c3118769))
* **serve:** add SmolLM2 tool calling through a recipe ([#575](https://github.com/kindlingai/tessera/issues/575)) ([5ec1630](https://github.com/kindlingai/tessera/commit/5ec16305ee392cdd3ee89051252efaca71d5b0cc))
* **serve:** add tool calling for Granite ([#576](https://github.com/kindlingai/tessera/issues/576)) ([6229aee](https://github.com/kindlingai/tessera/commit/6229aeeb83fa894d100eea141364b8e6649f3a03))


### Bug Fixes

* **bundle:** strip regex thinking openers from reasoning ([#566](https://github.com/kindlingai/tessera/issues/566)) ([42c4197](https://github.com/kindlingai/tessera/commit/42c41973b446939b86de410976baec8f805ff3a0))
* **device:** leave bf16 GEMV widths off 256 to lanes ([#546](https://github.com/kindlingai/tessera/issues/546)) ([76fab3a](https://github.com/kindlingai/tessera/commit/76fab3a5abbb90dc4c54fe24aeeebe8549218e7f))
* **lower:** plan guessed GPU devices on the runtime's platform ([#554](https://github.com/kindlingai/tessera/issues/554)) ([64f9f5d](https://github.com/kindlingai/tessera/commit/64f9f5d8e7515279a2e3a828877879679bb4d480))
* **plan:** price fast_gemv, fast_stacked and stacked launches ([#571](https://github.com/kindlingai/tessera/issues/571)) ([0dd1409](https://github.com/kindlingai/tessera/commit/0dd1409888d0303374886aab958ee72c5e4b0649))
* **qwen4_exp:** size the PLE hash from the checkpoint and open the release once ([#547](https://github.com/kindlingai/tessera/issues/547)) ([09377df](https://github.com/kindlingai/tessera/commit/09377df86f7ba1c45c0bdce86a4e4b2dca936cc5))


### Performance

* **device:** add recursive-doubling butterfly P2P all-reduce ([#567](https://github.com/kindlingai/tessera/issues/567)) ([2b03ab4](https://github.com/kindlingai/tessera/commit/2b03ab49a0e41f4152a282119dbf5cfe05baf707))
* **device:** add split-K for narrow MXFP8 small-batch matmuls ([#585](https://github.com/kindlingai/tessera/issues/585)) ([2effe5f](https://github.com/kindlingai/tessera/commit/2effe5fccf8e5ed6a6dcf252d3c45a659d3b6205))
* **device:** implement CUDA/HIP graph-capturable P2P all-gather ([#584](https://github.com/kindlingai/tessera/issues/584)) ([224f78a](https://github.com/kindlingai/tessera/commit/224f78aa6265951fc9a238fcc866bde23fbb0e79))
* **device:** make P2P all-reduce CUDA/HIP graph capturable ([#582](https://github.com/kindlingai/tessera/issues/582)) ([b691203](https://github.com/kindlingai/tessera/commit/b691203b19c85e2e36bace5098e6d8213c3f3beb))
* **device:** sum MCG trellis halves on the FMA pipe ([#555](https://github.com/kindlingai/tessera/issues/555)) ([da5bfd3](https://github.com/kindlingai/tessera/commit/da5bfd3e6e751a0904816d55222f052006c89ef2))
* **glm53-moe:** fuse plan into route and finish into fc2 ([#565](https://github.com/kindlingai/tessera/issues/565)) ([b1ac13a](https://github.com/kindlingai/tessera/commit/b1ac13a42b77dedef02e9ca827cc5038997f2897))
* **glm53-moe:** restore parallel finish and cache route scores in registers ([#580](https://github.com/kindlingai/tessera/issues/580)) ([c5e405b](https://github.com/kindlingai/tessera/commit/c5e405bf102adf99babe980d7070a0fecd1d23f6))

## [0.4.2](https://github.com/kindlingai/tessera/compare/v0.4.1...v0.4.2) (2026-11-24)


### Features

* **cli:** add tesserax --version and a teleport verb ([#529](https://github.com/kindlingai/tessera/issues/529)) ([f8446b9](https://github.com/kindlingai/tessera/commit/f8446b978260b5e4ae759c1d4e5b240d21510e0d))
* **deploy:** build the tesserax image for ROCm as well as CUDA ([#530](https://github.com/kindlingai/tessera/issues/530)) ([96e3665](https://github.com/kindlingai/tessera/commit/96e3665e568f5a8de9a9e1fda5e9b6a37ffa1f5b))
* **dflash:** add Qwen 3.8 melded spine and compressed drafter primitives ([#540](https://github.com/kindlingai/tessera/issues/540)) ([de9616c](https://github.com/kindlingai/tessera/commit/de9616c14f94a4e426ae6925d69810886bd4d518))
* **gpukit:** bind GPU threads and pinned allocations to local NUMA node ([#541](https://github.com/kindlingai/tessera/issues/541)) ([417db6f](https://github.com/kindlingai/tessera/commit/417db6fc0ac3f1b8670779aec3cd8b7402501d37))
* **lower:** add fast-path lowering diagnostics, CLI flag, and boot warning ([#523](https://github.com/kindlingai/tessera/issues/523)) ([8cb9c9a](https://github.com/kindlingai/tessera/commit/8cb9c9a598bbc95b6d2b478d8db4fe5a7bf97ed3))
* **maps:** task maps, a checker and a query command ([#520](https://github.com/kindlingai/tessera/issues/520)) ([ad98139](https://github.com/kindlingai/tessera/commit/ad98139d7ec6b4717aa81b14bb4685f3fb34f173))


### Bug Fixes

* **checkpoint:** decode exlr8 FP8_BLOCK and FP8_ROW in Release::packed ([#545](https://github.com/kindlingai/tessera/issues/545)) ([5f34e59](https://github.com/kindlingai/tessera/commit/5f34e59677b0e7ebb97953b823570e73636d89c2))
* **comm:** fast-fail and bound timeouts in TCP tests ([#537](https://github.com/kindlingai/tessera/issues/537)) ([b186ba4](https://github.com/kindlingai/tessera/commit/b186ba4d698ae6718a78042aeeec52ad0a668c9d))
* **engine:** start PLE context fresh when a slot is reused ([#532](https://github.com/kindlingai/tessera/issues/532)) ([9e33ad6](https://github.com/kindlingai/tessera/commit/9e33ad64439240962a95c34215ac50e202c0ef2d))


### Performance

* **calibrate:** skip known steps, cache source hashes, and optimize dev JIT ([#536](https://github.com/kindlingai/tessera/issues/536)) ([ee4352c](https://github.com/kindlingai/tessera/commit/ee4352cf5ed1e7982a6e6a6f3201e371b99fad29))

## [0.4.1](https://github.com/kindlingai/tessera/compare/v0.4.0...v0.4.1) (2026-11-24)


### Features

* **qwen4_exp:** serve Qwen3.8-Flash-Next EXLR8 with PLE store and CUDA MoE trellis ([#508](https://github.com/kindlingai/tessera/issues/508)) ([ce53018](https://github.com/kindlingai/tessera/commit/ce53018b45a5321371213e2352194aaced73d8a4))
