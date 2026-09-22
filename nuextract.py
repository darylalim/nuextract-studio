"""NuExtract3 MLX runtime wrapper.

Loads numind/NuExtract3-mlx-8bits via mlx-vlm and exposes streaming generation
across the three NuExtract3 modes (structured / markdown / template-generation).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

from huggingface_hub import snapshot_download
from mlx_vlm import load as mlx_vlm_load
from mlx_vlm import stream_generate as mlx_vlm_stream_generate

DEFAULT_MODEL_ID = "numind/NuExtract3-mlx-8bits"
# A full commit SHA of DEFAULT_MODEL_ID, never a branch or tag: the snapshot is
# code as much as data. mlx-vlm >=0.6.16 executes any `model_file` config.json
# names, its qwen3_5 processor loads default to trust_remote_code=True, and the
# chat template that does all of NuExtract3's mode routing ships in the repo —
# so tracking `main` would let an upstream push change what runs in-process,
# with every test mocked past it. A bump re-downloads whichever files changed
# (all ~5 GB if the weights did); keep scripts/probe_mlx_vlm.py's copy in step
# (a test checks) and re-run the probe.
DEFAULT_MODEL_REVISION = "bd8048c41019a63cdcbba93aa2dbfde06cbfc490"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.0

MODE_STRUCTURED = "structured"
MODE_CONTENT = "content"
MODE_MARKDOWN = "markdown"
MODE_TEMPLATE_GENERATION = "template-generation"


def load_model(
    model_id: str = DEFAULT_MODEL_ID, *, revision: str | None = DEFAULT_MODEL_REVISION
) -> tuple[Any, Any]:
    """Download and load NuExtract3-MLX. Returns (model, processor).

    `revision` defaults to the pinned commit of DEFAULT_MODEL_ID, so a caller
    overriding `model_id` must pass its own revision (None tracks `main`).

    The snapshot's processor_config.json is loaded as downloaded, on purpose.
    It names "Qwen3VLImageProcessor" where the model author's own
    numind/NuExtract3 names "Qwen2VLImageProcessor", but nothing here reads that
    key: mlx_vlm.load() builds mlx-vlm's own Qwen3VLProcessor for qwen3_5
    checkpoints, which takes only the geometry keys. Only transformers' own
    AutoProcessor resolves the name, and without torchvision that path fails on
    this checkpoint whichever name is there. A shim that rewrote it changed no
    outcome, and it wrote through the HF cache symlink into the shared blob.
    """
    local_dir = snapshot_download(repo_id=model_id, revision=revision)
    return mlx_vlm_load(local_dir)


def build_messages(
    text: str = "",
    image_path: str | None = None,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Build a chat message list with optional system + user image/text parts.

    The Jinja template inserts the vision placeholder for any user-content item
    that has an 'image' or 'image_url' key or type == 'image'; actual pixel
    data flows separately through stream_generate(image=...). System messages
    must be string-only — the Jinja raises if they contain images.
    """
    messages: list[dict[str, Any]] = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})

    parts: list[dict[str, Any]] = []
    if image_path:
        parts.append({"type": "image", "image": image_path})
    text = (text or "").strip()
    if text:
        parts.append({"type": "text", "text": text})
    if not parts:
        parts.append({"type": "text", "text": ""})
    messages.append({"role": "user", "content": parts})
    return messages


def render_prompt(
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    template: str | None = None,
    instructions: str | None = None,
    mode: str | None = None,
    enable_thinking: bool = False,
) -> str:
    """Render NuExtract3's chat template with task kwargs.

    Kwargs pass inline (HF transformers convention), not nested under
    chat_template_kwargs (which is a vLLM-specific convention).
    """
    kwargs: dict[str, Any] = {"enable_thinking": enable_thinking}
    if template is not None:
        kwargs["template"] = template
    if instructions:
        kwargs["instructions"] = instructions
    if mode is not None:
        kwargs["mode"] = mode
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


def stream_extract(
    model: Any,
    processor: Any,
    *,
    text: str = "",
    image_path: str | None = None,
    system_prompt: str | None = None,
    template: str | None = None,
    instructions: str | None = None,
    mode: str | None = None,
    enable_thinking: bool = False,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Iterator[str]:
    """Stream generation. Yields cumulative output text on each chunk."""
    messages = build_messages(
        text=text, image_path=image_path, system_prompt=system_prompt
    )
    prompt = render_prompt(
        processor,
        messages,
        template=template,
        instructions=instructions,
        mode=mode,
        enable_thinking=enable_thinking,
    )
    kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        # A backstop since mlx-vlm 0.7.0, which added the tokenizer's EOS
        # (<|im_end|>) to the stop set (Blaizzy/mlx-vlm#2112), so generation now
        # ends before that token is decoded. Earlier releases stopped only on
        # config.json's <|endoftext|>, and with mlx-vlm's False default the
        # <|im_end|> landed in the text: structured/template modes hid it because
        # extract_answer_block re-parses the JSON, but markdown mode rendered and
        # downloaded the raw string, so the saved .md ended with a literal token.
        # Kept so a future narrowing of that stop set cannot bring it back.
        "skip_special_tokens": True,
    }
    if image_path:
        kwargs["image"] = [image_path]
    accumulated = ""
    for chunk in mlx_vlm_stream_generate(model, processor, prompt, **kwargs):
        # Read .text directly rather than falling back to str(chunk): if a future
        # mlx-vlm renames the field, an AttributeError here is the loud failure we
        # want. The fallback would instead splice the whole GenerationResult repr
        # into the output — and no test would catch it, since they all feed stubs.
        accumulated += chunk.text
        yield accumulated


def split_reasoning_and_output(text: str, reasoning_enabled: bool) -> tuple[str, str]:
    """Split <think>...</think>... into (reasoning, output).

    When reasoning is disabled, all text is treated as output. When enabled but
    </think> hasn't arrived yet, everything is reasoning and output is empty.
    """
    if not text:
        return "", ""
    if not reasoning_enabled:
        return "", text.strip()
    lower = text.lower()
    end_tag = "</think>"
    if end_tag in lower:
        idx = lower.find(end_tag)
        return text[:idx].strip(), text[idx + len(end_tag) :].strip()
    return text.strip(), ""


_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer_block(text: str) -> str:
    """Pull <answer>...</answer> contents, or the longest valid JSON object.

    Tries `json.JSONDecoder.raw_decode` at every `{` position and returns the
    longest successfully-parsed span. Falls back to the stripped text if no
    valid JSON is found. Handles the common "reasoning text + JSON" case
    correctly, unlike a greedy regex.
    """
    if not text:
        return ""
    match = _ANSWER_RE.search(text)
    if match:
        return match.group(1).strip()
    decoder = json.JSONDecoder()
    best: str | None = None
    start = 0
    while True:
        i = text.find("{", start)
        if i == -1:
            break
        try:
            _obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            start = i + 1
            continue
        span = text[i:end]
        if best is None or len(span) > len(best):
            best = span
        start = end
    return best if best is not None else text.strip()


def pretty_json_or_text(text: str) -> str:
    """Try to pretty-print as JSON; fall back to the original string."""
    stripped = (text or "").strip()
    if not stripped:
        return ""
    try:
        return json.dumps(json.loads(stripped), indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return stripped
