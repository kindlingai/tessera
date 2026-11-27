# Changelog

## [0.4.5](https://github.com/kindlingai/tessera/compare/v0.4.4...v0.4.5) (2026-11-27)


### Features

* **attend:** add optional per-head attention sinks ([#722](https://github.com/kindlingai/tessera/issues/722)) ([499017e](https://github.com/kindlingai/tessera/commit/499017ecf8bb63fea3331b97897bc85a0030139d))
* **calibrate:** add ragged MLA/GQA and Zipf MoE harnesses ([#778](https://github.com/kindlingai/tessera/issues/778)) ([5726f1d](https://github.com/kindlingai/tessera/commit/5726f1d65b2dbd3129773d753ba5c670e9859d0a))
* **calibrate:** sweep any subject's plan across one dimension ([#672](https://github.com/kindlingai/tessera/issues/672)) ([18bb06f](https://github.com/kindlingai/tessera/commit/18bb06faabfaaeb0b2f43c954e46b038b97f15b4))
* **cli:** add tesserax size ([#643](https://github.com/kindlingai/tessera/issues/643)) ([c3b10c7](https://github.com/kindlingai/tessera/commit/c3b10c7d60b24c846549cbf4728813ffd4b6b3ff))
* **cli:** allow --model to override a recipe's weights path ([#703](https://github.com/kindlingai/tessera/issues/703)) ([1ad8ed3](https://github.com/kindlingai/tessera/commit/1ad8ed35d30587e6aeed321ef170fd3b71af586a))
* **comm:** add dual-root arx RoCE ring all-reduce ([#660](https://github.com/kindlingai/tessera/issues/660)) ([33c17be](https://github.com/kindlingai/tessera/commit/33c17be04da1ac6774fb0dfdf062767edfbbb69e))
* **device:** let fast_mm and fast attention take any row count ([#680](https://github.com/kindlingai/tessera/issues/680)) ([e33fffd](https://github.com/kindlingai/tessera/commit/e33fffd5a9013d07aa6c1bc605676aaa3957c4b5))
* **engine,serve:** add continuous-batching DFlash speculative decoding ([#656](https://github.com/kindlingai/tessera/issues/656)) ([2fa3db0](https://github.com/kindlingai/tessera/commit/2fa3db04d77053796071f987c3234e4b04d1cff6))
* **engine,serve:** wire chunked prefill into speculative decoders ([#618](https://github.com/kindlingai/tessera/issues/618)) ([4fab0cf](https://github.com/kindlingai/tessera/commit/4fab0cf61f80736a07310c05a77022588b8bd0ab))
* **engine:** apply guide logit edits on the device head ([#630](https://github.com/kindlingai/tessera/issues/630)) ([9a90ac3](https://github.com/kindlingai/tessera/commit/9a90ac3539cdfbeb03573702c762a338d339d46e))
* **engine:** apply guides under solo speculative decoding ([#648](https://github.com/kindlingai/tessera/issues/648)) ([e530b97](https://github.com/kindlingai/tessera/commit/e530b97c03573fdd79a6db3cdf77dbc3026e15a9))
* **engine:** clamp default prefill chunks to FAST_MAX_ROWS and time P2P sums ([#655](https://github.com/kindlingai/tessera/issues/655)) ([2704f63](https://github.com/kindlingai/tessera/commit/2704f6332ef26f8e428ee7357df9e7d79b2e261b))
* **engine:** let guides edit logits, stack guides and apply penalties ([#621](https://github.com/kindlingai/tessera/issues/621)) ([8b285d6](https://github.com/kindlingai/tessera/commit/8b285d6af19d48248829bc95a07634a0fd78e4d4))
* **engine:** roll drafter KV window when BatchDFlash target context exceeds drafter ([#715](https://github.com/kindlingai/tessera/issues/715)) ([ea70dd5](https://github.com/kindlingai/tessera/commit/ea70dd5fa139db2ce9ef7fcd31cabcecdda48ca8))
* **engine:** support prefix reuse in BatchDFlash ([#688](https://github.com/kindlingai/tessera/issues/688)) ([0f4a6e6](https://github.com/kindlingai/tessera/commit/0f4a6e6d0c333cc9767ca0bb98fb2fd8cac5dacf))
* **engine:** support prefix reuse in BatchMtp ([#687](https://github.com/kindlingai/tessera/issues/687)) ([0b7f455](https://github.com/kindlingai/tessera/commit/0b7f455bb3cd992ee8e8c2417efce78ddd58d116))
* **engine:** support sampling parameters under speculative decoding ([#689](https://github.com/kindlingai/tessera/issues/689)) ([3e15b6d](https://github.com/kindlingai/tessera/commit/3e15b6d3fc7ea0f7c8ebd286ba085987ca03fcc4))
* **engine:** verify recurrent state from one checkpoint under speculation ([#636](https://github.com/kindlingai/tessera/issues/636)) ([3b41e5c](https://github.com/kindlingai/tessera/commit/3b41e5cea7c5e70a0718e5776c43533bdbf76b12))
* **gpukit,run:** report kernel register, memory, spill, and occupancy attrs ([#768](https://github.com/kindlingai/tessera/issues/768)) ([db95514](https://github.com/kindlingai/tessera/commit/db95514b6b147f5889fdaa122a3f825ae9538f28))
* **gpukit:** borrow GB10 display carveout from spark-attic for KV cache ([#714](https://github.com/kindlingai/tessera/issues/714)) ([4bb3510](https://github.com/kindlingai/tessera/commit/4bb35109ae36508debf546ed41b0e4693be57ebb))
* **graph,lower,engine:** add split_batch two-microbatch comm/compute overlap ([#733](https://github.com/kindlingai/tessera/issues/733)) ([5d9f87a](https://github.com/kindlingai/tessera/commit/5d9f87acf436221dc4d8e713db110ecdc15e9730))
* **graph:** add row ownership to the graph and the CPU oracle ([#668](https://github.com/kindlingai/tessera/issues/668)) ([ee60219](https://github.com/kindlingai/tessera/commit/ee6021919760cf98903c884e755db925a6eca6cf))
* **hy4:** add exlr8 latent-MoE/Engram loading and HY4-flash recipes ([#783](https://github.com/kindlingai/tessera/issues/783)) ([f44d221](https://github.com/kindlingai/tessera/commit/f44d221f0ed783ddc88371ed6e5146c2b6e65b4b))
* **kv,engine,serve:** add gr8-style --kv-store grammar and default host-RAM tier ([#758](https://github.com/kindlingai/tessera/issues/758)) ([ddd90be](https://github.com/kindlingai/tessera/commit/ddd90be7bb80ac585bb265562d514ea4a2e5192c))
* **kv,lower,run,device,engine,serve:** add TIER_HOST pinned host tier for sparse MLA KV ([#769](https://github.com/kindlingai/tessera/issues/769)) ([6abcd44](https://github.com/kindlingai/tessera/commit/6abcd4402baf0a00cd6f7a22e9f0ed63a748a8b6))
* **kv:** add content-addressed SSD KV page cache ([#692](https://github.com/kindlingai/tessera/issues/692)) ([3cc69dd](https://github.com/kindlingai/tessera/commit/3cc69ddb7fd3e073406630cc2192eb57acead6b7))
* **lower,run,device,engine:** add row ownership lowering, GPU collectives, and auto-activation ([#784](https://github.com/kindlingai/tessera/issues/784)) ([be2dad2](https://github.com/kindlingai/tessera/commit/be2dad2b87ac7c35e1f736dacbf169a1f86c5f4b))
* **lower:** add dtype rules by spot for recipes and --dtype ([#634](https://github.com/kindlingai/tessera/issues/634)) ([0b9ddef](https://github.com/kindlingai/tessera/commit/0b9ddef79eefd67a04b373d0ff67f44dc8782789))
* **lower:** add kernel shape rules for recipes and --shape ([#638](https://github.com/kindlingai/tessera/issues/638)) ([04a13e4](https://github.com/kindlingai/tessera/commit/04a13e45b155fbe3c5593aebf8550c03907d1d5a))
* **models,bundle:** add Hy4-flash (hy_v4_flash) model graph and loader ([#724](https://github.com/kindlingai/tessera/issues/724)) ([f3d58f0](https://github.com/kindlingai/tessera/commit/f3d58f091fbaa96dd076c7e8889c76c659fbf070))
* **models:** add qwen2 decoder graph with Q, K, and V projection biases ([#725](https://github.com/kindlingai/tessera/issues/725)) ([d6b711e](https://github.com/kindlingai/tessera/commit/d6b711e58bbe22ce87b6eece75e66df9e9306b8e))
* **models:** add qwen3_moe decoder graph with separate routed experts ([#730](https://github.com/kindlingai/tessera/issues/730)) ([573b726](https://github.com/kindlingai/tessera/commit/573b7265ce5d2b237dae166da641405d1bf1d987))
* **models:** add qwen35 stepper graph for serving hybrid DeltaNet models ([#726](https://github.com/kindlingai/tessera/issues/726)) ([519a2e6](https://github.com/kindlingai/tessera/commit/519a2e6b97fd318934770b2f37a0b5b3856ae04b))
* **models:** support Gemma 4 E2B double-wide MLP, unified 12B, and 31B tied-KV variants ([#700](https://github.com/kindlingai/tessera/issues/700)) ([8d620ff](https://github.com/kindlingai/tessera/commit/8d620ff44a6fffbb776f28cdd0e39152ad32e1b1))
* **p2p:** add block-scaled FP8 wire transport for P2P all-reduce ([#650](https://github.com/kindlingai/tessera/issues/650)) ([7ebbd04](https://github.com/kindlingai/tessera/commit/7ebbd043455769ec861ce76e86aa62dcc60b2f7f))
* **plan,gpukit:** add multi-lane streams and automatic cross-lane event sync ([#721](https://github.com/kindlingai/tessera/issues/721)) ([d8887d1](https://github.com/kindlingai/tessera/commit/d8887d196bdb684020b3b830a63afb44a54430c5))
* **recipes:** add GLM-5.3 exlr8 recipe form for four GB10 ranks ([#706](https://github.com/kindlingai/tessera/issues/706)) ([a75495d](https://github.com/kindlingai/tessera/commit/a75495d19a8159b00572c0a2fc3e997594ccdddd))
* report structured progress from downloads and model loads ([#619](https://github.com/kindlingai/tessera/issues/619)) ([8d8708e](https://github.com/kindlingai/tessera/commit/8d8708eba7d15f8dff24c514ba1c40681009dd58))
* **run,device:** profile inside CUDA graphs and flag-gate collective timing ([#755](https://github.com/kindlingai/tessera/issues/755)) ([7e540a4](https://github.com/kindlingai/tessera/commit/7e540a4483774c4bb8f9715867fa63b10ef84a28))
* **run:** profile rows by kernel sizes, with work and byte rates ([#666](https://github.com/kindlingai/tessera/issues/666)) ([b68e2da](https://github.com/kindlingai/tessera/commit/b68e2daeffefa373c6068e310e1a7b00bb0b36e0))
* **serve:** accept --port 0 and write the bound port to --port-file ([#756](https://github.com/kindlingai/tessera/issues/756)) ([51d83b8](https://github.com/kindlingai/tessera/commit/51d83b8badc99ea771d7f4d9bfc9a1c25985eb5b))
* **serve:** accept formats and penalties under solo speculative decoding ([#652](https://github.com/kindlingai/tessera/issues/652)) ([0e5b5df](https://github.com/kindlingai/tessera/commit/0e5b5dfdfc52e4885499875229ed6de72bb3d34f))
* **serve:** accept neutral OpenAI SDK request fields ([#748](https://github.com/kindlingai/tessera/issues/748)) ([c4e5c6c](https://github.com/kindlingai/tessera/commit/c4e5c6c66fffc257f08fc5fbfa3411865480c1dc))
* **serve:** accept top-level reasoning_effort on chat requests ([#681](https://github.com/kindlingai/tessera/issues/681)) ([3422c92](https://github.com/kindlingai/tessera/commit/3422c92e718ba606dcb629cdbb5275bf32bfdfa7))
* **serve:** add --layers to keep only the first n layers ([#628](https://github.com/kindlingai/tessera/issues/628)) ([3753d52](https://github.com/kindlingai/tessera/commit/3753d52c10d107c9a56ecd381ada10c97aecdce2))
* **serve:** add a sequence penalty against verbatim loops ([#657](https://github.com/kindlingai/tessera/issues/657)) ([d052993](https://github.com/kindlingai/tessera/commit/d052993d48f1d4e90447a5f5167e6ec24c2beb73))
* **serve:** add Anthropic /v1/messages endpoint ([#684](https://github.com/kindlingai/tessera/issues/684)) ([f99f695](https://github.com/kindlingai/tessera/commit/f99f6950054e3bbc11f167167f304662509e0570))
* **serve:** add GET /v1/models/{id} model retrieval route ([#683](https://github.com/kindlingai/tessera/issues/683)) ([e149b8a](https://github.com/kindlingai/tessera/commit/e149b8a90c6b721caf0bbb1b48bdc49e5c170fa2))
* **serve:** add optional --api-key authentication for API routes ([#699](https://github.com/kindlingai/tessera/issues/699)) ([1afcba7](https://github.com/kindlingai/tessera/commit/1afcba7df76f3d90dca665fe6013588e2ae746f8))
* **serve:** add optional --cors-origin headers and OPTIONS preflight ([#711](https://github.com/kindlingai/tessera/issues/711)) ([566f97f](https://github.com/kindlingai/tessera/commit/566f97fa71213a989d68a67f48c74972f34ec00d))
* **serve:** let scope moves match short token sequences ([#654](https://github.com/kindlingai/tessera/issues/654)) ([4cda636](https://github.com/kindlingai/tessera/commit/4cda6368e0239942d55ce260d8871e87d73a58cd))
* **serve:** profile single requests with TESSERA_PROFILE=dynamic ([#627](https://github.com/kindlingai/tessera/issues/627)) ([1d21e9d](https://github.com/kindlingai/tessera/commit/1d21e9df4d1ff4989bb374e3df89e6a91f093a81))
* **serve:** refuse to boot when lowering falls off the fast path ([#728](https://github.com/kindlingai/tessera/issues/728)) ([0cc126e](https://github.com/kindlingai/tessera/commit/0cc126e8be11b39c5f3e969c63a87fe69c28e3e4))
* **serve:** refuse to boot when weights must stream ([#729](https://github.com/kindlingai/tessera/issues/729)) ([7254c84](https://github.com/kindlingai/tessera/commit/7254c84c140dc47cb5752e339017ad0d791cb29b))
* **serve:** report load steps for klein and embed models ([#620](https://github.com/kindlingai/tessera/issues/620)) ([71b94cc](https://github.com/kindlingai/tessera/commit/71b94cc908373e659ca0fc8dd2815340ed1f99fb))
* **serve:** report reasoning_tokens in completion_tokens_details ([#705](https://github.com/kindlingai/tessera/issues/705)) ([200e9fb](https://github.com/kindlingai/tessera/commit/200e9fb05a9ea782487b5634a6395ff9ed7901a9))
* **serve:** scope logit penalties to answer, thinking and tool calls ([#647](https://github.com/kindlingai/tessera/issues/647)) ([9c8b96e](https://github.com/kindlingai/tessera/commit/9c8b96e35506095cabde8318cff146ad031c78a8))
* **serve:** serve several models that take turns on one device ([#642](https://github.com/kindlingai/tessera/issues/642)) ([641b9d7](https://github.com/kindlingai/tessera/commit/641b9d70a319de623dcb9ebc0cd7ef8088de96a1))
* **serve:** support chunked transfer-encoding on HTTP request bodies ([#685](https://github.com/kindlingai/tessera/issues/685)) ([e86ab9a](https://github.com/kindlingai/tessera/commit/e86ab9a60094f56fad609efa3856142c7de3a7b5))
* **serve:** support min_tokens on chat and completion requests ([#682](https://github.com/kindlingai/tessera/issues/682)) ([cdce1ff](https://github.com/kindlingai/tessera/commit/cdce1ff75955f047fb8690fdc19f6388c876c8b3))
* **serve:** support parallel_tool_calls=false on chat completions ([#743](https://github.com/kindlingai/tessera/issues/743)) ([1f69661](https://github.com/kindlingai/tessera/commit/1f6966172031953449c5a99c2d86813b9f1147e9))
* **serve:** support required and named tool_choice on chat completions ([#686](https://github.com/kindlingai/tessera/issues/686)) ([3fe4a2a](https://github.com/kindlingai/tessera/commit/3fe4a2a20d7ab2498cf6ce0ffd3ce35eb73b5f49))
* **serve:** warn at boot when a prefill step leaves the fast path ([#673](https://github.com/kindlingai/tessera/issues/673)) ([a91162b](https://github.com/kindlingai/tessera/commit/a91162b390226c73518435351a5b7b08ada7a8e5))
* **tiers:** add a board that sets PR tiers as a GitHub App ([#738](https://github.com/kindlingai/tessera/issues/738)) ([a6c5687](https://github.com/kindlingai/tessera/commit/a6c5687d1a7baf524500ef472d1ec1674a4001d7))


### Bug Fixes

* **bundle:** bound sliding-window DFlash drafter KV context ([#691](https://github.com/kindlingai/tessera/issues/691)) ([0193c9a](https://github.com/kindlingai/tessera/commit/0193c9afc3f056019dac40885e090fc0a6253fd7))
* **bundle:** forward manifest in load_module ([#694](https://github.com/kindlingai/tessera/issues/694)) ([cdccbac](https://github.com/kindlingai/tessera/commit/cdccbac156bbc2f95034f189adb5796a6b14f366))
* **bundle:** map developer role to system when template lacks developer ([#693](https://github.com/kindlingai/tessera/issues/693)) ([132d980](https://github.com/kindlingai/tessera/commit/132d980249954e70632a795342a91b4f9ea13357))
* **checkpoint:** say what to do when a manifest mixes widths ([#659](https://github.com/kindlingai/tessera/issues/659)) ([29e321d](https://github.com/kindlingai/tessera/commit/29e321d520dfacce3e29cad48e2cdd9117c31b04))
* **ci:** run the build jobs on pushes and pull requests again ([#772](https://github.com/kindlingai/tessera/issues/772)) ([cdeb01c](https://github.com/kindlingai/tessera/commit/cdeb01c79ad0ad215241430e026e332fb6ff6463))
* **comm:** report a peer's reset stream as a link error ([#757](https://github.com/kindlingai/tessera/issues/757)) ([d891228](https://github.com/kindlingai/tessera/commit/d8912288d411d98263204c5378fba0cbe32889b5))
* **device:** honor the mxfp8 splits knob at prefill and explain attention declines ([#667](https://github.com/kindlingai/tessera/issues/667)) ([a031663](https://github.com/kindlingai/tessera/commit/a0316638f4720ffd3c843168b8261369216ecc3b))
* **engine:** cap sliding-window drafter taps to window size ([#690](https://github.com/kindlingai/tessera/issues/690)) ([f265230](https://github.com/kindlingai/tessera/commit/f26523075dbb6703e82e8e61c1d728a604bbdb59))
* **engine:** map slot zero in solo speculative loops under shared paged KV ([#745](https://github.com/kindlingai/tessera/issues/745)) ([241f6bb](https://github.com/kindlingai/tessera/commit/241f6bbc74f7e04250c8cdd73dbc4f0d128833c6))
* **engine:** wire paged KV lanes in BatchMtp and MtpDrafter ([#744](https://github.com/kindlingai/tessera/issues/744)) ([345d42a](https://github.com/kindlingai/tessera/commit/345d42a14859418e677124cf74e1f04b49e20968))
* **glm53-moe:** use z_chunks=0 sentinel for prefill finish ([#782](https://github.com/kindlingai/tessera/issues/782)) ([5016e68](https://github.com/kindlingai/tessera/commit/5016e68a33b06ef98647bdcdaeb734ba76193068))
* **host:** read integer buffers with every bit in the lockstep executor ([#649](https://github.com/kindlingai/tessera/issues/649)) ([bc19bb1](https://github.com/kindlingai/tessera/commit/bc19bb1eec21456559bf0050b1c981975bc37b3b))
* **lower:** decline kernels whose launch grid exceeds device limits ([#676](https://github.com/kindlingai/tessera/issues/676)) ([dde3df7](https://github.com/kindlingai/tessera/commit/dde3df7c070d3e1d85d700293d2faacc0ffe35d2))
* **lower:** keep a swallowed cover as fallback when favoring a rule ([#716](https://github.com/kindlingai/tessera/issues/716)) ([859f11d](https://github.com/kindlingai/tessera/commit/859f11d666be04e1f6840a037f1eebaf0b142758))
* **metal:** drain autorelease pool around Metal command buffers ([#702](https://github.com/kindlingai/tessera/issues/702)) ([4dc397f](https://github.com/kindlingai/tessera/commit/4dc397fd27d7f97ceb2c1130efa3006c4e79659c))
* **models:** apply interleaved RoPE to MLA indexer queries and keys ([#742](https://github.com/kindlingai/tessera/issues/742)) ([65a67ea](https://github.com/kindlingai/tessera/commit/65a67eaaaa8cd9f83fe62925ec6a344cd5fe3256))
* **models:** apply RoPE to MLA Lightning Indexer queries and keys ([#737](https://github.com/kindlingai/tessera/issues/737)) ([ff4350b](https://github.com/kindlingai/tessera/commit/ff4350beead0f6e809f93ca4fce1e19b08cefb65))
* **p2p:** cap block threads at 512 with launch bounds to prevent register overflow ([#633](https://github.com/kindlingai/tessera/issues/633)) ([5fbedf1](https://github.com/kindlingai/tessera/commit/5fbedf160253eaea8f84765f28c03d53718d9992))
* **plan,device:** keep unoverlapped collectives on lane 0 and cap gather grids ([#741](https://github.com/kindlingai/tessera/issues/741)) ([3d1c6cd](https://github.com/kindlingai/tessera/commit/3d1c6cd7e4a538b5e9b1ec899e28926887ad5e25))
* **recipe:** allow PyTorch weight files in HF lookup ([#734](https://github.com/kindlingai/tessera/issues/734)) ([5b19783](https://github.com/kindlingai/tessera/commit/5b19783f0cad001a4d4203172c05e1305cc722f4))
* **recipe:** build modules from the recipe directory ([#658](https://github.com/kindlingai/tessera/issues/658)) ([cf83c1f](https://github.com/kindlingai/tessera/commit/cf83c1f8c65e8fc283900e28d9bf1c5518698c74))
* **run,lower:** reuse loaded weight arenas on subset plans and free unread temp scratch ([#679](https://github.com/kindlingai/tessera/issues/679)) ([c5c8df6](https://github.com/kindlingai/tessera/commit/c5c8df6f7e9f4cd6775fbb0a0181bfb4a4342fdb))
* **serve:** refuse sampling under speculative decoding ([#624](https://github.com/kindlingai/tessera/issues/624)) ([4361586](https://github.com/kindlingai/tessera/commit/43615864d057201ea3a0899aeedef9e3a3bf3f9b))
* **serve:** set min_tokens in the /v1/messages answer params ([ea972a1](https://github.com/kindlingai/tessera/commit/ea972a117af111a983f8ad738959d20fbc8cdfa3))
* **serve:** size ranked layouts offline without opening peer sockets ([#750](https://github.com/kindlingai/tessera/issues/750)) ([ba8924e](https://github.com/kindlingai/tessera/commit/ba8924e8231ec0fd52282a694fa8d452c24e3ef2))
* **serve:** wait for follower ranks to finish loading before serving ([#739](https://github.com/kindlingai/tessera/issues/739)) ([f133d60](https://github.com/kindlingai/tessera/commit/f133d60d1456c7fd9764c386c29632b15e8b06c2))


### Performance

* **arx:** add coalesced FP8 wire format, 48-block grid, and stub support ([#713](https://github.com/kindlingai/tessera/issues/713)) ([d6ee492](https://github.com/kindlingai/tessera/commit/d6ee492576f00e6267f7c6d73e7a6fd1b85c557c))
* **attend:** add 8-row query-tiled fast_indexer with register Q and shared K ([#664](https://github.com/kindlingai/tessera/issues/664)) ([fa4764e](https://github.com/kindlingai/tessera/commit/fa4764e9635f4aeb4064be1296c29ca2583f2c04))
* **attend:** fuse single-split MLA/GQA epilogue, reverse row order, and use native BF16 MMA ([#663](https://github.com/kindlingai/tessera/issues/663)) ([a1094fa](https://github.com/kindlingai/tessera/commit/a1094fae59d37fdf19ec96594fd4e110dbb5d166))
* **attend:** keep at least 4 K-splits for prefill MLA and GQA attention ([#637](https://github.com/kindlingai/tessera/issues/637)) ([56fa148](https://github.com/kindlingai/tessera/commit/56fa1484359ac5832be4636ab08ddd54136614dc))
* **attend:** support paged KV in fast_mla_attend, fast_gqa_attend, and fast_indexer ([#651](https://github.com/kindlingai/tessera/issues/651)) ([aa09da1](https://github.com/kindlingai/tessera/commit/aa09da1ebd49fc64658063cd179ff7f28002f5ea))
* **attend:** widen fast_mla_attend to 8 warps, 32-key tiles, ldmatrix, and FP8 prefetch ([#662](https://github.com/kindlingai/tessera/issues/662)) ([fb30263](https://github.com/kindlingai/tessera/commit/fb30263867233b881e7a01a1748194ffb891a6b4))
* **build:** fix fixture test timeouts for eg2 and klein flow ([#653](https://github.com/kindlingai/tessera/issues/653)) ([9fa34e2](https://github.com/kindlingai/tessera/commit/9fa34e226142758e3cff6eea635f68b9f3d8348a))
* **checkpoint:** read trellis weights directly into header-prefixed buffers ([#696](https://github.com/kindlingai/tessera/issues/696)) ([7581197](https://github.com/kindlingai/tessera/commit/758119798fa81ec46b5e4ef9839443da5aea8fcb))
* **device-linear:** add Blackwell 128x128 native MXFP8 MMA prefill kernel ([#762](https://github.com/kindlingai/tessera/issues/762)) ([413e55a](https://github.com/kindlingai/tessera/commit/413e55ab1e9a62616e46d81c7c861ca3b2d515a8))
* **device-linear:** add pre-rotated Blackwell mma_mx8 dense trellis prefill ([#678](https://github.com/kindlingai/tessera/issues/678)) ([e7960ae](https://github.com/kindlingai/tessera/commit/e7960ae2b9de0adca95d5dc1c68ad7c5943d0a12))
* **device-linear:** add widened 64x64 BF16 MMA stacked matmul prefill kernels ([#774](https://github.com/kindlingai/tessera/issues/774)) ([435da91](https://github.com/kindlingai/tessera/commit/435da917f146008fb1df523792a4281973fe9a50))
* **device:** add 64x64 staged-X prefill kernel for MXFP8 matmul ([#639](https://github.com/kindlingai/tessera/issues/639)) ([cf0f441](https://github.com/kindlingai/tessera/commit/cf0f44129c8adef9da7eb441f56c4af3258b852b))
* **device:** add Tensor-Core BF16 Lightning Indexer rule for CUDA ([#632](https://github.com/kindlingai/tessera/issues/632)) ([a8ad2e6](https://github.com/kindlingai/tessera/commit/a8ad2e69d1364dc2fe2dd6effe85605ac297a032))
* **device:** double P2P slot size to 16 MiB for 1-chunk FP8 all-reduce ([#773](https://github.com/kindlingai/tessera/issues/773)) ([4f43b09](https://github.com/kindlingai/tessera/commit/4f43b0900b9e0abd2a222101029bdaabbeef7e38))
* **device:** double-buffer MLA KV with cp.async and fuse 512-dim slice ([#780](https://github.com/kindlingai/tessera/issues/780)) ([bc08e40](https://github.com/kindlingai/tessera/commit/bc08e40f27b20ac7142ab3f90abfdfc0ceeddfa7))
* **device:** fuse row-batch permute and slice views into stacked matmul ([#764](https://github.com/kindlingai/tessera/issues/764)) ([bd9ec0b](https://github.com/kindlingai/tessera/commit/bd9ec0b4638d7af403f44bef676c7c84b4385320))
* **device:** pipeline and vectorize persistent grid sums over depth 4 ([#695](https://github.com/kindlingai/tessera/issues/695)) ([001333e](https://github.com/kindlingai/tessera/commit/001333e2cf7e281d4a36f8a709d2157665eedad4))
* **device:** support ring-buffer KV caches in CUDA MLA and GQA attention ([#776](https://github.com/kindlingai/tessera/issues/776)) ([ba9fec5](https://github.com/kindlingai/tessera/commit/ba9fec50a6e3259ad0368c5a11df95e089d0aebf))
* **embed:** preserve compact checkpoint pins and fix turned kernel row-count collisions ([#719](https://github.com/kindlingai/tessera/issues/719)) ([31dfa32](https://github.com/kindlingai/tessera/commit/31dfa32a38d31f0cd6977aa09c8ca2e21a5dc03e))
* **engine:** broadcast and upload only dirty page-table ranges ([#708](https://github.com/kindlingai/tessera/issues/708)) ([d7e482e](https://github.com/kindlingai/tessera/commit/d7e482e3cd8e97cb04e8f4c0aaa6e9ef9522b27e))
* **engine:** bucket prefill rows in steps of 64 up to 512 ([#665](https://github.com/kindlingai/tessera/issues/665)) ([2c4a368](https://github.com/kindlingai/tessera/commit/2c4a368497c2bdcafd2ebc79528b960f90e9c1da))
* **engine:** grow paged KV slots once per batch instead of per row ([#697](https://github.com/kindlingai/tessera/issues/697)) ([b345b95](https://github.com/kindlingai/tessera/commit/b345b95459f6cb7b981f081bf385642470d131eb))
* **engine:** skip the output head on prefill chunks whose pick is dropped ([#669](https://github.com/kindlingai/tessera/issues/669)) ([bbe3bb6](https://github.com/kindlingai/tessera/commit/bbe3bb6d6f5cff24b463c1efd5d977dd82781ea9))
* **glm53-moe,gpukit:** use Blackwell mma_mx8 FP8 Tensor Cores for MoE prefill ([#675](https://github.com/kindlingai/tessera/issues/675)) ([5ba9f05](https://github.com/kindlingai/tessera/commit/5ba9f05c64cf6920995033dbc41569568a151066))
* **glm53-moe:** coalesce fc2_pf writes, fuse output turn, and widen route_pf ([#779](https://github.com/kindlingai/tessera/issues/779)) ([c25d411](https://github.com/kindlingai/tessera/commit/c25d411dd92b9a269bd17ded00c1d67cb0ea3ed5))
* **glm53-moe:** coalesce prefill FC2 partial stores through shared memory ([#760](https://github.com/kindlingai/tessera/issues/760)) ([9298388](https://github.com/kindlingai/tessera/commit/929838861f48f08c6f80d79f4915a7b2a8471eae))
* **glm53-moe:** fuse 16x256 BF16 prefill router and k2 hot dispatch ([#674](https://github.com/kindlingai/tessera/issues/674)) ([a99e7f8](https://github.com/kindlingai/tessera/commit/a99e7f84bbf218125a51a4d64c727d29615c1d0d))
* **glm53-moe:** pre-turn MXFP8 activations once per slot for prefill fc1 ([#712](https://github.com/kindlingai/tessera/issues/712)) ([f17ab8d](https://github.com/kindlingai/tessera/commit/f17ab8d87a6c90a996b2bd6119110af377fbe435))
* **glm53-moe:** support 2048-row prefill with 64-token slots and fused fc1_pf ([#661](https://github.com/kindlingai/tessera/issues/661)) ([c337b4b](https://github.com/kindlingai/tessera/commit/c337b4b8bb624777731a88e65f7ba8eac966771a))
* **glm53-moe:** support 512 rows and 32-token NT=4 slots for prefill ([#641](https://github.com/kindlingai/tessera/issues/641)) ([d6ce02e](https://github.com/kindlingai/tessera/commit/d6ce02e71b11d9557b53177bb5f285da266e4407))
* **glm53-moe:** tile fc1 and fc2 over 16-token dual slots for prefill ([#635](https://github.com/kindlingai/tessera/issues/635)) ([4944295](https://github.com/kindlingai/tessera/commit/494429597920f4c5093824be020ac41b8e90dad4))
* **gpukit,run,kv:** batch KV page transfers over pinned async DMA ([#765](https://github.com/kindlingai/tessera/issues/765)) ([60c9d7e](https://github.com/kindlingai/tessera/commit/60c9d7e98d857b880dfed38799a810a8214b8668))
* **kernels:** cache pass-1 loads in registers for lane and block RMS norm ([#751](https://github.com/kindlingai/tessera/issues/751)) ([fed8267](https://github.com/kindlingai/tessera/commit/fed82671a73efe96c588a3385914edaee656f86c))
* **kernels:** enable vectorized x4 strided views for narrow float forms ([#753](https://github.com/kindlingai/tessera/issues/753)) ([e0d64bd](https://github.com/kindlingai/tessera/commit/e0d64bdd20e6ec7ce7a9af5e842c9d967edf0992))
* **kernels:** use 11-bit atomic shared histograms in radix_top_k ([#754](https://github.com/kindlingai/tessera/issues/754)) ([43c2781](https://github.com/kindlingai/tessera/commit/43c27815ae8aa516abee1fce9a08aae55f96508c))
* **kv,engine:** prioritize and promote hot and shared pages in TIER_HOST ([#775](https://github.com/kindlingai/tessera/issues/775)) ([8e8f0d8](https://github.com/kindlingai/tessera/commit/8e8f0d86a1e53b2d37326a78d4c61d4ef61b3fd5))
* **kv:** index retained prefixes in a compressed chunk trie ([#707](https://github.com/kindlingai/tessera/issues/707)) ([518c718](https://github.com/kindlingai/tessera/commit/518c7181b8779c441bf1b7c1d751ceb9aa95100f))
* **kv:** track active and reclaimable page counts in O(1) ([#698](https://github.com/kindlingai/tessera/issues/698)) ([b3b49ad](https://github.com/kindlingai/tessera/commit/b3b49ad0697f5c50f44c2ba79a5cbac450587612))
* **linear:** add Metal tile-streaming trellis decode and MMA prefill kernels ([#704](https://github.com/kindlingai/tessera/issues/704)) ([df0b58a](https://github.com/kindlingai/tessera/commit/df0b58a638d6da1f4896856b29108f79f67089c9))
* **lower:** check the grid only for each node's winning cover ([#677](https://github.com/kindlingai/tessera/issues/677)) ([ad462cc](https://github.com/kindlingai/tessera/commit/ad462ccbd768241050caa5cd6c35ed1e33ce8bd3))
* **lower:** close rotated matmul chains before elementwise fusion ([#763](https://github.com/kindlingai/tessera/issues/763)) ([5b2d1e4](https://github.com/kindlingai/tessera/commit/5b2d1e42ba1394be1af62a3f72416fbd2dee7df2))
* **lower:** use ring-buffer KV caches for sliding-window attention ([#732](https://github.com/kindlingai/tessera/issues/732)) ([f0c5586](https://github.com/kindlingai/tessera/commit/f0c5586f5719a9ab3bba3722f7aa164da3b1843c))
* **p2p:** add 16-byte wire transfers and 4-rank reduce-scatter/all-gather ([#626](https://github.com/kindlingai/tessera/issues/626)) ([4afc868](https://github.com/kindlingai/tessera/commit/4afc868145f1248da4a5226e9e2e3876213cc651))
* **p2p:** pack FP8/BF16 chunks per slot and tune scatter grid ([#671](https://github.com/kindlingai/tessera/issues/671)) ([6c19f09](https://github.com/kindlingai/tessera/commit/6c19f0933f977f49f8eac3cff8e1fd8c20604b4f))
* **prefill:** raise FAST_MAX_ROWS and wide block norm cap to 2048 ([#670](https://github.com/kindlingai/tessera/issues/670)) ([87d29ff](https://github.com/kindlingai/tessera/commit/87d29ff0137530f60bf9a211289c4565df85e55c))
* **rules:** extend fast_mla_attend, block norm, and fast_mm to prefill batches ([#622](https://github.com/kindlingai/tessera/issues/622)) ([75e1e27](https://github.com/kindlingai/tessera/commit/75e1e2729d82a7f1b1f2276a91a23a7ff92c2bf8))
* **serve:** pre-warm prefill and decode plans on all ranks at boot ([#749](https://github.com/kindlingai/tessera/issues/749)) ([08909cb](https://github.com/kindlingai/tessera/commit/08909cbb9f7f9bdaf8396629788175b523848bd2))

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
