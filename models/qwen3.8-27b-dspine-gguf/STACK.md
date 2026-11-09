# Qwen3.8-27B DSpine GGUF on GB10

Last updated: 2026-11-09

This is a standalone llama.cpp experiment, separate from the Transformers-based
AI4ATEXIS_LAB_FW runtime. DSpine speculative decoding requires the DSpine fork;
the standard `~/llama.cpp` checkout is not used or modified.

## Hardware and artifacts

- GPU: NVIDIA GB10, compute capability 12.1 (SM121)
- CUDA toolkit: 13.0.88; compiler: `/usr/local/cuda/bin/nvcc`
- Target: `unsloth/Qwen3.8-27B-GGUF`,
  `Qwen3.8-27B-UD-Q4_K_M.gguf` (16,464,440,224 bytes)
- DSpine draft: `aj9o9/Qwen3.8-27B-DSpine-b16-GGUF`,
  `Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf` (1,257,892,896 bytes)
- DFlash2 draft: `Akicou/Qwen3.8-27B-DFlash2-GGUF`,
  `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` (1,143,006,592 bytes)
- The draft is not a standalone language model; it must be paired with the
  matching Qwen3.8-27B target.
- Model directory: `$HOME/Documents/llama.cpp-tools-GB10/models/qwen3.8-dspine`
- DSpine fork: `$HOME/src/llama.cpp-dspine`, branch `dspine`, commit `d493f84`
- Separate download utility environment: `$HOME/.venvs/qwen38-dspine-gguf`.
  llama.cpp itself is native and does not use this Python environment.

The target and draft files were downloaded with `curl --continue-at -` so an
interrupted transfer could be resumed. Their local byte sizes match the sizes
reported by Hugging Face. Both were then loaded together for a real inference
smoke test.

## Build

The build is isolated from the standard llama.cpp checkout:

```bash
CUDACXX=/usr/local/cuda/bin/nvcc cmake \
  -S "$HOME/src/llama.cpp-dspine" \
  -B "$HOME/src/llama.cpp-dspine/build-gb10" \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=121a-real \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$HOME/src/llama.cpp-dspine/build-gb10" \
  --parallel 8 --target llama-server
```

The build completed successfully with CUDA enabled for `121a-real`. CMake
reported that NCCL was not found; that only affects multi-GPU performance and
is not required for this single-GB10 run. It also suggested installing ccache
to speed up future builds; this is optional.

## Launch

This smoke-test configuration follows the DSpine model-card settings and binds
only to loopback. The server UI is disabled.

### Why `draft-dflash` for a DSpine draft?

DSpine is the draft method/model family; it extends DFlash with causal block
attention, code bridges, and serial last-write refinement. The model card's
llama.cpp command selects `--spec-type draft-dflash`, and this fork exposes no
separate `draft-dspine` mode. In this command, `--spec-type` selects the
speculative-decoding implementation path; `--spec-draft-model` selects the
DSpine GGUF weights. Therefore the loaded draft is DSpine, even though the
runtime mode is named `draft-dflash`. This is not the separate DFlash2 draft.

```bash
"$HOME/src/llama.cpp-dspine/build-gb10/bin/llama-server" \
  --model "$HOME/Documents/llama.cpp-tools-GB10/models/qwen3.8-dspine/Qwen3.8-27B-UD-Q4_K_M.gguf" \
  --spec-draft-model "$HOME/Documents/llama.cpp-tools-GB10/models/qwen3.8-dspine/Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf" \
  --spec-type draft-dflash \
  --spec-draft-n-max 15 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --gpu-layers 999 --spec-draft-ngl 999 \
  --flash-attn on --jinja \
  -c 32768 -np 1 \
  --host 127.0.0.1 --port 18080 --no-webui
```

In a second terminal, send a short request:

```bash
curl --fail-with-body --silent --show-error \
  http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-dspine","messages":[{"role":"user","content":"Reponds uniquement par le resultat numerique de 2+2, sans explication."}],"max_tokens":128,"temperature":0}'
```

Stop the server with Ctrl+C in its terminal when finished. The recorded smoke
test returned `4`. For that short request the server reported 76.10 generated
tokens/s, with 46 draft tokens accepted out of 75 proposed. This is a brief
warm-process smoke measurement, not a reproducible benchmark. The server used
for the test was stopped afterward.

## DFlash2 comparison

The matching DFlash2 Q4_K_M draft was downloaded from
`Akicou/Qwen3.8-27B-DFlash2-GGUF`. Its local size was 1,143,006,592 bytes and
its SHA256 matched the published value:
`e44b99d7f4bce8ad5b573190c95e20678613fec9c9f455b073b6d9ef99c241e1`.
The fork detected block size 8 and loaded it using `--spec-type draft-dflash`
with `--spec-draft-n-max 7`.

The comparison used the same target, 32K context, `--fit off`, default cache
types, and this Snake request at temperature 0 with a 1,024-token output limit:

```text
Crée un jeu Snake complet en un seul fichier HTML, avec HTML, CSS et JavaScript intégrés. Il doit être jouable au clavier, afficher le score, permettre la pause et le redémarrage. Réponds uniquement par le code, sans explication.
```

After one warmup, the three DFlash2 decode speeds were 27.50, 27.70, and
28.04 tokens/s (median 27.70 tokens/s). Each request generated 1,024 tokens.
The server reported 686 accepted draft tokens out of 2,355 proposed (29.1%).

| Configuration | Decode speed |
| --- | ---: |
| Target only | 11.3 tokens/s |
| DSpine, `n-max=7` | 27.8 tokens/s |
| DFlash2, `n-max=7` | 27.70 tokens/s median |

For this workload DFlash2 and DSpine are effectively tied; DFlash2 did not
show a measurable speedup over the earlier DSpine result. The brief 64-token
smoke request reported 34.48 tokens/s, but is not comparable to the 1,024-token
benchmark. The DFlash2 server was bound to loopback and stopped after testing.

## DFlash memory-fitting log

Startup printed:

```text
E llama_init_from_model: failed to initialize the context: dflash requires ctx_other to be set (this warning is normal during memory fitting)
W operator(): failed to measure the memory of the extra model, fitting without it: failed to create llama_context from model
```

This is an error-level message from a temporary context initialization, not a
failure of the final model load or a CUDA error. During automatic memory
fitting, llama.cpp tries to measure the optional draft model on its own. DFlash
can share token embeddings and the output head with the target through
`ctx_other`; that standalone probe has no target context to share, so DFlash
rejects it. The memory-fitting code catches that exception and continues by
fitting the target without a separate draft-memory estimate. The normal load
then creates the draft with its target context, and DFlash starts successfully.

In this run the log continued with `loading draft model`,
`adding speculative implementation 'draft-dflash'`, and `model loaded`, followed
by a successful chat completion with speculative tokens accepted. Those later
messages and the successful response confirm that the displayed `E` was
non-fatal for this configuration. If the server exits before `model loaded`,
or requests fail, investigate the subsequent error rather than dismissing it
as this expected memory-fitting probe.

Other startup output included unused draft tensors being ignored. That output
also did not prevent this target/draft pair from loading and generating text.

## Initial cleanup

For this separate GGUF experiment, the previously created
`$HOME/.venvs/dspine` environment and `$HOME/src/DSpine` checkout were removed
before the new download environment and DSpine llama.cpp clone were created.
The external tools directory and the standard `$HOME/llama.cpp` checkout were
left untouched.