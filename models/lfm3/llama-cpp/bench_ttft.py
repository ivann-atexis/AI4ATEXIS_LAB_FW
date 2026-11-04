import json
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen

url = "http://127.0.0.1:8080/v1/chat/completions"
base_url = "http://127.0.0.1:8080"
model = str(Path.home() / "Documents/llama.cpp-tools-GB10/models/LFM3-1.7B-Q8_0.gguf")
target_prompt_tokens = 1024
headers = {"Content-Type": "application/json"}


def post_json(endpoint, body):
    request = Request(
        f"{base_url}{endpoint}",
        data=json.dumps(body).encode(),
        headers=headers,
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def build_prompt(target_tokens):
    low = 0
    high = target_tokens
    while low <= high:
        repeats = (low + high) // 2
        content = " a" * repeats
        messages = [{"role": "user", "content": content}]
        prompt = post_json("/apply-template", {"messages": messages})["prompt"]
        tokens = post_json(
            "/tokenize",
            {"content": prompt, "add_special": False, "parse_special": True},
        )["tokens"]
        token_count = len(tokens)
        if token_count == target_tokens:
            return content
        if token_count < target_tokens:
            low = repeats + 1
        else:
            high = repeats - 1
    raise RuntimeError(f"Impossible de construire un prompt de {target_tokens} tokens")


prompt_content = build_prompt(target_prompt_tokens)
payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt_content}],
    "stream": True,
    "temperature": 0,
    "max_tokens": 512,
}

def measure_ttft():
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first_text = None
    prompt_tokens = None

    with urlopen(request, timeout=300) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            timings = event.get("timings", {})
            if "prompt_n" in timings:
                prompt_tokens = timings["prompt_n"]
            choices = event.get("choices", [])
            delta = choices[0].get("delta", {}) if choices else {}
            generated_text = (
                delta.get("content")
                or delta.get("reasoning_content")
                or delta.get("reasoning")
            )
            if generated_text and first_text is None:
                first_text = time.perf_counter()

    if first_text is None:
        raise RuntimeError("Aucun delta de texte reçu")
    if prompt_tokens != target_prompt_tokens:
        raise RuntimeError(
            f"Prompt mesuré: {prompt_tokens} tokens; attendu: {target_prompt_tokens}"
        )
    return (first_text - start) * 1000, prompt_tokens

measure_ttft()  # chauffe; ne pas inclure cette valeur
values = [measure_ttft()[0] for _ in range(10)]
print(f"Input: {target_prompt_tokens} tokens par requête")
print(f"TTFT médian: {statistics.median(values):.1f} ms")
print(f"Min / max: {min(values):.1f} / {max(values):.1f} ms")