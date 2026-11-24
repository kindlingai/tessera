# Changelog

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
