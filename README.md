# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp)](https://github.com/ggml-org/llama.cpp/releases)
[![Server](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3A0cc4m%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev branches](https://github.com/ggml-org/llama.cpp-dev/blob/master/README-features.md) / [compile times](https://github.com/ggml-org/llama.cpp-dev/blob/master/README-compile-times.md) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>
LLM inference in C/C++

<!-- thecodacus/llama.cpp prefix -->
## ⚡ This fork — Fable's MoE-offload prefill optimizations

Two **opt-in** optimizations for large MoE models whose experts are offloaded to system RAM
(`--n-cpu-moe`), found and implemented by Fable. Both are **off by default**, toggled via
environment variables, and produce **token-identical** output to mainline.

| Env var | What it does |
| --- | --- |
| `GGML_CUDA_REGISTER_HOST=1` | Page-locks (pins) the mmap'd CPU expert weights so host→device copies go straight over DMA instead of through the driver's hidden bounce buffer (~6–7 → ~20 GB/s). |
| `GGML_SCHED_PREFETCH_EXPERTS=1` | Prefetches each layer's experts on a second CUDA stream, so the weight uploads overlap compute instead of stalling the GPU. |

### Benchmark

Measured on an **RTX 3060 12GB** with **Qwen3.6-35B-A3B** (`--n-cpu-moe 26`), prompt-processing at 2048 (`MODEL` = path to your `.gguf`):

```bash
# baseline (patches off):
./build/bin/llama-bench -m MODEL -ngl 99 -ncmoe 26 -p 2048 -n 0 -r 5 -b 2048 -ub 2048

# patched (both optimizations on):
GGML_CUDA_REGISTER_HOST=1 GGML_SCHED_PREFETCH_EXPERTS=1 \
./build/bin/llama-bench -m MODEL -ngl 99 -ncmoe 26 -p 2048 -n 0 -r 5 -b 2048 -ub 2048
```

Result: **~1143 → ~1880 t/s** prefill (**+64%**) — same GPU, same settings, token-identical.

Branches: [`fable5/host-register`](https://github.com/thecodacus/llama.cpp/tree/fable5/host-register) (pinning only) · [`fable5/prefetch-experts`](https://github.com/thecodacus/llama.cpp/tree/fable5/prefetch-experts) (both — this branch).
<!-- end of thecodacus/llama.cpp prefix -->

<!-- TheTom/llama-cpp-turboquant prefix -->

# TurboQuant+

> **🚀 TurboQuant KV cache compression is now in [vLLM](https://github.com/vllm-project/vllm)** ([PR #38479](https://github.com/vllm-project/vllm/pull/38479), merged April 2026): `--kv-cache-dtype turboquant_k8v4` and friends, with fused Triton store/decode kernels. The PR discussion drew on the asymmetric K/V findings from this repo. **Upstream llama.cpp has merged the core idea too**: Hadamard KV cache rotation ([#21038](https://github.com/ggml-org/llama.cpp/pull/21038), citing TurboQuant directly) with fast WHT kernels on CPU ([#22631](https://github.com/ggml-org/llama.cpp/pull/22631)), CUDA ([#23615](https://github.com/ggml-org/llama.cpp/pull/23615)), and Vulkan ([#23687](https://github.com/ggml-org/llama.cpp/pull/23687)). Rotation + the stock q4_0 cache is essentially turbo4's rotation stage; the PolarQuant codebook, norm extraction, and asymmetric policies remain here and in the fork.

> ### [Getting Started Guide](docs/getting-started.md) | [Configuration Recommendations](docs/turboquant-recommendations.md) | [Benchmarks](docs/benchmarks.md) | [Commercial Support](https://x.com/no_stp_on_snek)

Implementation of [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (ICLR 2026) with implementation work, experiments, and follow-on findings beyond the base paper. Compresses transformer KV cache **3.8-6.4x** using PolarQuant + Walsh-Hadamard rotation, at near q8_0 prefill speed and ~0.9x decode throughput at long context. Validated end-to-end from 1.5B to **104B at 128K context on a MacBook** (turbo3, PPL 4.024, 74 GB peak memory).

## Base and the fork which this repository is based on

| Engine | Platform | Status | Notes |
|--------|----------|--------|-------|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | All backends | **Upstream, rotation merged** | Hadamard KV cache rotation ([#21038](https://github.com/ggml-org/llama.cpp/pull/21038)) + fast WHT kernels (CPU [#22631](https://github.com/ggml-org/llama.cpp/pull/22631), CUDA [#23615](https://github.com/ggml-org/llama.cpp/pull/23615), Vulkan [#23687](https://github.com/ggml-org/llama.cpp/pull/23687)). Rotation + q4_0 cache approximates turbo4; full PolarQuant codec is in the fork below |
| [llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) | Metal, CUDA, HIP, CPU | **Production fork** | turbo2/3/4 KV cache + TQ3_1S/TQ4_1S weight formats; [prebuilt binaries](https://github.com/TheTom/llama-cpp-turboquant/releases) for Mac (Metal) and Windows (CUDA) |

## Key Findings

Three follow-on findings, independently validated by multiple researchers across different hardware and backends:

1. **V compression is free.** Compressing the value cache (even down to 2 bits) has zero measurable effect on attention quality when key precision is maintained. Confirmed on Metal (M5 Max), CUDA RTX 4090 (@sztlink), and CUDA RTX 3090 (@HyperionMS2040). See [asymmetric K/V paper](docs/papers/asymmetric-kv-compression.md).
2. **All quality degradation comes from K compression.** This is why asymmetric configs (q8_0-K + turbo-V) rescue models where symmetric fails. Validated across Qwen, Llama, Mistral, and Command-R+ families. See [M5 Max stress test](docs/papers/m5-max-stress-test.md).
3. **Boundary layers are disproportionately sensitive.** Protecting the first 2 + last 2 layers at higher precision recovers 37-91% of the quality gap. See [Boundary V paper](docs/papers/layer-aware-v-compression.md).

Upstream llama.cpp rotations PR got some benchmark results:
https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4146397570
https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4154465964

<!-- end of TheTom/llama-cpp-turboquant prefix -->

## Recent API changes

- [Changelog for `libllama` API](https://github.com/ggml-org/llama.cpp/issues/9289)
- [Changelog for `llama-server` REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

## Hot topics

- **Hugging Face cache migration: models downloaded with `-hf` are now stored in the standard Hugging Face cache directory, enabling sharing with other HF tools.**
- **[guide : using the new WebUI of llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/16938)**
- [guide : running gpt-oss with llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/15396)
- [[FEEDBACK] Better packaging for llama.cpp to support downstream consumers 🤗](https://github.com/ggml-org/llama.cpp/discussions/15313)
- Support for the `gpt-oss` model with native MXFP4 format has been added | [PR](https://github.com/ggml-org/llama.cpp/pull/15091) | [Collaboration with NVIDIA](https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss) | [Comment](https://github.com/ggml-org/llama.cpp/discussions/15095)
- Multimodal support arrived in `llama-server`: [#12898](https://github.com/ggml-org/llama.cpp/pull/12898) | [documentation](./docs/multimodal.md)
- VS Code extension for FIM completions: https://github.com/ggml-org/llama.vscode
- Vim/Neovim plugin for FIM completions: https://github.com/ggml-org/llama.vim
- Hugging Face Inference Endpoints now support GGUF out of the box! https://github.com/ggml-org/llama.cpp/discussions/9669
- Hugging Face GGUF editor: [discussion](https://github.com/ggml-org/llama.cpp/discussions/9268) | [tool](https://huggingface.co/spaces/CISCai/gguf-editor)
- WebGPU support is now available in the browser, see a blog/demo introducing it [here](https://reeselevine.github.io/llamas-on-the-web/).

----

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [stb-image](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [miniaudio.h](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
