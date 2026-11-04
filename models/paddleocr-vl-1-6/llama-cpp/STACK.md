# PaddleOCR-VL-1.6 with llama.cpp

Last updated: 2026-11-04

## Stack

- Model: `PaddleOCR-VL-1.6.Q8_0.gguf`
- Multimodal projector: `PaddleOCR-VL-1.6-GGUF-mmproj.gguf`
- Files: `$HOME/Documents/llama.cpp-tools-GB10/models/`
- llama.cpp source: `$HOME/llama.cpp`
- Build directory: `$HOME/llama.cpp/build-gb10`
- llama.cpp: `0.8.0-dev`, build `12059`, commit `1b6935b22`
- Build target: Linux `aarch64`, CUDA for GB10 (`sm_121`)
- The model and matching `mmproj` are both required for image OCR.

## Gradio box viewer

The app offers two real Paddle paths:

- `PaddleOCR-VL · blocs de page` runs layout detection and recognition. Paddle
  handles layout locally and sends recognition requests to `llama-server`.
- `PaddleOCR · lignes + texte` runs Paddle's OCR detector and recognizer on
  CPU without calling the VLM. Its JSON keeps Paddle's native result and adds
  a top-level `detections` list with each line's `text`, `score`, and `polygon`.
  The detector reuses `PP-OCRv6_medium_det`; `PP-OCRv6_medium_rec` is needed
  for recognition and is downloaded by PaddleX on first use if not cached.

The app opens in `PaddleOCR-VL · blocs de page` mode. Keep this mode selected
for layout boxes and VLM recognition; the PaddleOCR line mode is an optional
comparison and does not use llama.cpp.

On this GB10 (`aarch64`, sm121), the currently published PaddleGPU ARM64 wheel
is compiled for sm103 and fails its GPU architecture check. The app therefore
runs Paddle layout and text detection on CPU; llama.cpp still runs VLM
recognition on the GB10 GPU. The requirements pin PaddlePaddle 3.4.0's official
ARM64 CPU wheel for CPython 3.12. Install into a separate environment so the
existing llama.cpp Gradio environment is unchanged:

```bash
cd "$HOME/Documents/AI4ATEXIS_LAB_FW/models/paddleocr-vl-1-6/llama-cpp"
"$HOME/.local/bin/uv" venv --python 3.12 paddleocr-env/.venv
"$HOME/.local/bin/uv" pip install --python paddleocr-env/.venv/bin/python \
  -r requirements-paddleocr.txt
```

Start the model server in one terminal:

```bash
cd "$HOME/Documents/llama.cpp-tools-GB10"
./start-server.sh PaddleOCR-VL-1.6.Q8_0.gguf \
  --mmproj "$PWD/models/PaddleOCR-VL-1.6-GGUF-mmproj.gguf" \
  --temp 0 --parallel 1 --metrics
```

Start the UI in another terminal and open `http://127.0.0.1:7861`:

```bash
cd "$HOME/Documents/AI4ATEXIS_LAB_FW/models/paddleocr-vl-1-6/llama-cpp"
paddleocr-env/.venv/bin/python spotting_gradio.py
```

Check that both services are ready:

```bash
curl -fsS http://127.0.0.1:8080/health
curl -fsS --output /dev/null http://127.0.0.1:7861/config
```

The first request downloads Paddle's layout/text-detection weights if they are
not cached. `LLAMA_SERVER_URL` accepts the llama.cpp `/v1` base URL, and
`LLAMA_MODEL` defaults to the GGUF path reported by `/v1/models`.

The direct GGUF `Spotting:` request previously returned transcription but no
coordinates. The box viewer now uses Paddle's structured detection results and
visualizations instead of inferring boxes from generated text.

Gradio displays cold model loading, image preparation, Paddle inference, JSON
conversion, preview generation, and total callback time. In layout mode, it
also displays PaddleX's instrumented layout detection, crop preparation, VLM
request preparation, llama.cpp recognition, and result assembly timings. The
internal substeps are included in `PaddleOCRVL.predict` and should not be added
to that end-to-end duration. In PaddleOCR mode, the combined detection and
recognition inference is reported as one step.

For a representative timing, run one request first to warm the models and do
not count it. Then use the same image and settings for the measured requests.
On the GB10, a warm, complex page with about 52 layout boxes was observed at
about 5.55 s total: layout 1.69 s, VLM recognition 3.275 s,
`PaddleOCRVL.predict` 5.463 s, and total interface time 5.553 s. These are
observed values for that page, not a throughput guarantee. Layout and VLM
times are included in `PaddleOCRVL.predict`; do not sum them with it.

The server command above keeps `--no-cache-prompt` to reproduce the current
setup. To test prompt-prefix caching for repeated requests, restart the server
with the same command but omit `--no-cache-prompt`, then repeat the same
warm-up and measurement procedure. Compare like-for-like results; caching may
not help when page prompts differ.

Validated on `inputs/page.jpg` (2026-11-04): TextDetection returned 60 text-line
polygons. The full PaddleOCR-VL path returned layout entries with
`block_bbox` and `block_polygon_points`, plus an annotated preview. Paddle
layout ran on CPU; VLM recognition used the llama.cpp server on the GB10 GPU.

## OCR command

Place a representative test image at `inputs/page.jpg`, then run:

```bash
cd "$HOME/Documents/llama.cpp-tools-GB10"

"$HOME/llama.cpp/build-gb10/bin/llama-cli" \
  -m "$PWD/models/PaddleOCR-VL-1.6.Q8_0.gguf" \
  --mmproj "$PWD/models/PaddleOCR-VL-1.6-GGUF-mmproj.gguf" \
  --image "$PWD/inputs/page.jpg" \
  -p "Transcris le texte visible dans l'ordre de lecture. Arrete-toi a la fin du document et ne repete pas le texte." \
  -n 1024 \
  --temp 0 \
  --repeat-last-n 128 \
  --repeat-penalty 1.05 \
  --single-turn \
  --show-timings
```

`-n 1024` is a generation cap, not an infinite-generation setting.
`--single-turn` exits after one response instead of waiting for another turn.
The repetition penalty can also alter legitimate repeated OCR text; compare
with `--repeat-penalty 1.00` if fidelity changes. Increase the token cap only
if a dense page is visibly truncated. The timing output is for this actual
image inference; `llama-bench` cannot benchmark image input.
