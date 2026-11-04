# LFM3-1.7B with llama.cpp

Last updated: 2026-11-04

## Stack

- Model: `LFM3-1.7B-Q8_0.gguf` (`1.73B` parameters, `1.71 GiB`)
- Model file: `$HOME/Documents/llama.cpp-tools-GB10/models/LFM3-1.7B-Q8_0.gguf`
- llama.cpp source: `$HOME/llama.cpp`
- Build directory: `$HOME/llama.cpp/build-gb10`
- llama.cpp: `0.8.0-dev`, build `12059`, commit `1b6935b22`
- Build target: Linux `aarch64`, CUDA for GB10 (`sm_121`)
- GPU reported by llama-bench: NVIDIA GB10, compute capability 12.1,
  124544 MiB VRAM
- No Python environment is used by this runtime.

## Build llama-bench

The GB10 helper scripts initially build `llama-cli` and `llama-server`. If
`llama-bench` is not present, build its target from the existing CUDA build:

```bash
cmake --build "$HOME/llama.cpp/build-gb10" --target llama-bench -j 8
```

## Benchmark

```bash
MODEL="$HOME/Documents/llama.cpp-tools-GB10/models/LFM3-1.7B-Q8_0.gguf"

"$HOME/llama.cpp/build-gb10/bin/llama-bench" \
  -m "$MODEL" \
  -ngl 99 \
  -p 1024 \
  -n 512 \
  -r 5 \
  -o md
```

`-p 1024` measures prompt processing for 1024 tokens (`pp1024`); `-n 512`
measures generation for 512 tokens (`tg512`). These are separate microbenchmarks.
`-r 5` runs five repetitions; llama-bench reports the mean and standard
 deviation. To benchmark a prompt followed by generation as a combined test,
use `-pg 1024,512` instead.

## Recorded result

Result provided on 2026-11-04:

| Test | Mean throughput |
|---|---:|
| Prompt processing, 1024 tokens (`pp1024`) | 13,710.22 +/- 770.76 tok/s |
| Generation, 512 tokens (`tg512`) | 120.23 +/- 3.75 tok/s |

Build reported by the benchmark: `1b6935b22` (build `12059`). The benchmark is
for the llama.cpp inference core; it does not include tokenization or sampling,
and it is not an HTTP, chat-template, TTFT, or end-to-end OpenWebUI measurement.
