# Qwen3.6-35B-A3B-NVFP4-Fast

Last updated: 2026-11-04

## Runtime

- Model: `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast`
- Environment: `$HOME/.venvs/qwen36-unsloth` (Python 3.13.18)
- Installed stack: vLLM 0.33.0, PyTorch 2.14.0+cu132, FlashInfer 0.7.2.post1,
  CUTLASS DSL 4.8.0
- GPU: NVIDIA GB10, compute capability 12.1
- This environment is isolated from `/opt/hp/zrt/venv`.

## Launch

This command combines the user-confirmed B12X target backend with MTP using a
separate Triton backend for its unquantized draft. `MAX_JOBS=1` limits parallel
JIT compilation; `CUTE_DSL_ARCH` selects the GB10 target. The MTP configuration
has completed the benchmark recorded below.

```bash
source "$HOME/.venvs/qwen36-unsloth/bin/activate"
export CUTE_DSL_ARCH=sm_121a

MAX_JOBS=1 vllm serve unsloth/Qwen3.6-35B-A3B-NVFP4-Fast \
  --moe-backend flashinfer_b12x \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2,"moe_backend":"triton"}' \
  --max-num-batched-tokens 8192 \
  --max-model-len 16535 --max-num-seqs 2 --gpu-memory-utilization 0.6 \
  --host 172.18.0.1 --port 8000
```

`moe_backend=triton` applies to the unquantized MTP draft; the quantized target
continues to use `flashinfer_b12x`. `--max-num-batched-tokens 8192` raises the
scheduler token budget above the 2048 setting that triggered the MTP performance
warning. A larger budget can increase GPU memory use.

OpenWebUI API base URL: `http://172.18.0.1:8000/v1`.
Health endpoint: `http://172.18.0.1:8000/health`.

## vLLM Benchmark

Run this from another terminal while the vLLM server is active. It targets the
OpenAI-compatible vLLM endpoint directly, bypassing OpenWebUI. Two warmup
requests precede eight measured requests, one at a time, with 128 random input
and output tokens each.

```bash
source "$HOME/.venvs/qwen36-unsloth/bin/activate"

vllm bench serve \
  --backend openai-chat \
  --base-url http://172.18.0.1:8000 \
  --endpoint /v1/chat/completions \
  --model unsloth/Qwen3.6-35B-A3B-NVFP4-Fast \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 128 \
  --num-warmups 2 \
  --num-prompts 8 \
  --max-concurrency 1
```

Recorded result (2026-11-04): 8 successful, 0 failed; 10.06 s benchmark
duration; 1,107 input and 1,024 generated tokens; 0.80 requests/s; 101.80 output
tokens/s and 211.86 total tokens/s. Mean/median/P99 TTFT: 88.08/88.45/97.96 ms.
Mean/median/P99 TPOT: 9.20/9.11/10.79 ms. Mean/median/P99 ITL:
23.55/23.45/27.30 ms. Speculative-token acceptance: 78.97%, with 2.58 accepted
tokens on average; per-position acceptance was 86.15% and 71.79%.

### TTFT / Prefill / Decode breakdown

The prefill value is measured from the delta of vLLM's Prometheus
`vllm:request_prefill_time_seconds_sum` divided by the matching `_count`.
vLLM defines it as first scheduled time to first generated token, excluding
time queued before scheduling. Prefill throughput is prompt tokens divided by
the sum of per-request prefill durations; it is not wall-clock throughput.

| Run | Input / output per request | Measured requests | Max concurrency | Mean TTFT | Mean prefill | Prefill throughput | Mean decode (TPOT) | Output throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 128 / 128 tokens | 8 | 1 | 97.97 ms | 79.40 ms/request | 1,743 tok/s | 10.67 ms/token | 88.06 tok/s |
| Large, sequential | 1,024 / 512 tokens | 20 | 1 | 187.32 ms | 163.76 ms/request | 6,318 tok/s | 10.98 ms/token | 88.27 tok/s |
| Large, concurrent | 1,024 / 512 tokens | 20 | 2 | 242.16 ms | 211.00 ms/request | 4,904 tok/s | 14.70 ms/token | 129.66 tok/s |

The short run had no benchmark warmups: 1,107 prompt tokens / 0.635 s prefill.
Each large run used 3 warmups; its Prometheus delta covers 23 requests, including
those warmups. The sequential run recorded 23,797 prompt tokens / 3.767 s
prefill; the concurrent run recorded 23,797 / 4.853 s. TTFT, TPOT and output
throughput in the table cover only the 20 measured requests. Prefill throughput
is therefore warmup-inclusive for both large runs. All are warmed-process
measurements and exclude vLLM startup. Avoid other traffic when sampling these
cumulative metrics.

Do not derive prefill by subtracting a client-side guess from TTFT: queue,
network and other overhead are not isolated by that difference.

These are warmed-server random-text measurements, not OpenWebUI end-to-end
latencies or a long-context benchmark. Compare results with their input/output
lengths, concurrency and warmup scope in mind.

### Larger concurrent run

Run this in a terminal while the vLLM server is active. Keep other traffic off
the server during the measurement because the Prometheus counters are
cumulative. This snapshots both prefill duration and prompt tokens around the
benchmark, then computes their ratio.

```bash
set -e
source "$HOME/.venvs/qwen36-unsloth/bin/activate"

read_metrics() {
  curl --fail --silent --show-error --max-time 5 http://172.18.0.1:8000/metrics | awk '
    /^vllm:request_prefill_time_seconds_count/ {prefill_count=$NF}
    /^vllm:request_prefill_time_seconds_sum/ {prefill_sum=$NF}
    /^vllm:request_prompt_tokens_count/ {prompt_count=$NF}
    /^vllm:request_prompt_tokens_sum/ {prompt_sum=$NF}
    END {
      if (prefill_count == "" || prefill_sum == "" || prompt_count == "" || prompt_sum == "") exit 2
      print prefill_count, prefill_sum, prompt_count, prompt_sum
    }'
}

read before_prefill_count before_prefill_sum before_prompt_count before_prompt_sum < <(read_metrics)
printf 'Before: prefill_count=%s prefill_sum=%s prompt_count=%s prompt_sum=%s\n' \
  "$before_prefill_count" "$before_prefill_sum" "$before_prompt_count" "$before_prompt_sum"

vllm bench serve \
  --backend openai-chat \
  --base-url http://172.18.0.1:8000 \
  --endpoint /v1/chat/completions \
  --model unsloth/Qwen3.6-35B-A3B-NVFP4-Fast \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 512 \
  --num-warmups 3 \
  --num-prompts 20 \
  --max-concurrency 2

read after_prefill_count after_prefill_sum after_prompt_count after_prompt_sum < <(read_metrics)
printf 'After: prefill_count=%s prefill_sum=%s prompt_count=%s prompt_sum=%s\n' \
  "$after_prefill_count" "$after_prefill_sum" "$after_prompt_count" "$after_prompt_sum"

awk \
  -v bc="$before_prefill_count" -v bs="$before_prefill_sum" \
  -v btc="$before_prompt_count" -v bts="$before_prompt_sum" \
  -v ac="$after_prefill_count" -v as="$after_prefill_sum" \
  -v atc="$after_prompt_count" -v ats="$after_prompt_sum" \
  'BEGIN {
    requests = ac - bc
    seconds = as - bs
    prompt_requests = atc - btc
    prompt_tokens = ats - bts
    if (requests > 0 && seconds > 0 && prompt_requests == requests)
      printf "Prefill: %d requests, %.3f s, %.2f ms/request; %.0f prompt tokens / %.3f s = %.1f prompt tok/s\n", requests, seconds, seconds * 1000 / requests, prompt_tokens, seconds, prompt_tokens / seconds
    else
      print "Could not calculate matching prefill/prompt deltas; check for concurrent traffic."
  }'
```

With 3 warmups, the prefill delta normally covers 23 requests (20 measured and
3 warmups); the `vllm bench serve` TTFT, TPOT and output-throughput summary
covers only the 20 measured requests. The prefill tok/s value is prompt tokens
divided by the sum of per-request prefill times, not by benchmark wall time.
For the large sequential run, use the same command with `--max-concurrency 1`.

Earlier recorded result (2026-11-04): 20 successful, 0 failed; 67.71 s;
20,689 input tokens and 10,240 output tokens; 0.30 requests/s; 151.23 output
tokens/s and 456.77 total tokens/s. Mean/median/P99 TTFT:
224.24/216.22/309.25 ms. Mean/median/P99 TPOT: 12.80/12.88/14.94 ms.
Mean/median/P99 ITL: 30.85/30.25/38.46 ms. Speculative-token acceptance:
70.65%, with 2.41 accepted tokens on average; per-position acceptance was
80.13% and 61.17%. The report lists max concurrency 2 and peak concurrency 4.

Latest detailed result (2026-11-04): 20 successful, 0 failed; 78.98 s;
20,689 input tokens and 10,240 output tokens; 0.25 requests/s; 129.66 output
tokens/s and 391.61 total tokens/s. Mean/median/P99 TTFT:
242.16/237.12/332.48 ms. Mean/median/P99 TPOT: 14.70/14.55/18.28 ms.
Mean/median/P99 ITL: 34.39/33.89/43.67 ms. Speculative-token acceptance:
67.15%, with 2.34 accepted tokens on average; per-position acceptance was
77.72% and 56.57%. Prefill metrics and workload details are summarized above.

Large sequential result (2026-11-04): 20 successful, 0 failed; 116.00 s;
20,689 input tokens and 10,240 output tokens; 0.17 requests/s; 88.27 output
tokens/s and 266.63 total tokens/s. Mean/median/P99 TTFT:
187.32/185.79/202.13 ms. Mean/median/P99 TPOT: 10.98/11.08/11.70 ms.
Mean/median/P99 ITL: 26.53/26.47/31.47 ms. Speculative-token acceptance:
70.98%, with 2.42 accepted tokens on average; per-position acceptance was
79.91% and 62.04%. The benchmark maximum concurrency was 1; peak concurrent
requests reported 2.

The short and larger runs use different prompt/output lengths and concurrency;
their aggregate throughput is not a like-for-like comparison. The larger run
is a concurrent throughput test, not a long-context benchmark.

## Validation

The B12X target with a Triton MTP draft completed all 20 requests in the latest
concurrent benchmark, confirming server startup and text generation for this
configuration. MTP with an inherited B12X draft backend had failed because
B12X does not support unquantized MoE. No image or video input has been tested.
The benchmark proves the server was responding during the recorded run; its
current health has not been checked.