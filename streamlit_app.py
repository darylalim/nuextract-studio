"""NuExtract3 Streamlit app — mirrors the official HF Space, runs locally on MLX.

Single image + optional text input, JSON template editor, three modes:
  - Extract JSON (structured extraction)
  - Convert to Markdown (document-to-markdown)
  - Generate template (NL description → JSON template)

Streams output with optional <think>...</think> reasoning parsing.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from nuextract import (
    DEFAULT_MAX_TOKENS,
    MODE_MARKDOWN,
    MODE_TEMPLATE_GENERATION,
    extract_answer_block,
    load_model,
    pretty_json_or_text,
    split_reasoning_and_output,
    stream_extract,
)

DEFAULT_TEMPLATE = json.dumps(
    {
        "title": "string",
        "entities": ["string"],
        "dates": ["YYYY-MM-DD"],
        "amounts": [{"value": "number", "currency": "string"}],
    },
    indent=2,
)

TEMPLATE_GEN_GUIDANCE = (
    "Generate a concise JSON extraction template for this document. "
    "Use descriptive field names and simple type hints like string, number, "
    "verbatim-string, date, boolean, or arrays of objects. Return only the JSON template."
)


@st.cache_resource
def get_model() -> tuple[Any, Any]:
    return load_model()


_IMG_PATH_KEY = "_uploaded_image_path"
_IMG_ID_KEY = "_uploaded_image_id"


def _save_uploaded_image(uploaded_file: Any) -> str | None:
    """Persist the uploaded image to a temp file once per upload.

    Cached in session_state keyed on uploaded_file.file_id so reruns don't
    re-write the image; the previous temp file is cleaned up when a new upload
    arrives.
    """
    if uploaded_file is None:
        # Upload was removed: delete the orphaned temp file and clear stale
        # session_state so nothing leaks on disk or lingers in state.
        cached_path = st.session_state.pop(_IMG_PATH_KEY, None)
        st.session_state.pop(_IMG_ID_KEY, None)
        if cached_path:
            Path(cached_path).unlink(missing_ok=True)
        return None
    file_id = uploaded_file.file_id
    cached_path = st.session_state.get(_IMG_PATH_KEY)
    if (
        st.session_state.get(_IMG_ID_KEY) == file_id
        and cached_path
        and Path(cached_path).exists()
    ):
        return cached_path
    if cached_path:
        Path(cached_path).unlink(missing_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    st.session_state[_IMG_ID_KEY] = file_id
    st.session_state[_IMG_PATH_KEY] = tmp.name
    return tmp.name


def _validate_template(template_str: str) -> tuple[dict | None, str | None]:
    """Validate template is non-empty JSON dict. Returns (parsed, error)."""
    stripped = (template_str or "").strip()
    if not stripped:
        return None, "Template is empty."
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"
    if not isinstance(parsed, dict):
        return None, "Template must be a JSON object."
    if not parsed:
        return None, "Template must not be empty."
    return parsed, None


def _render_output_pane(
    output_placeholder: Any,
    reasoning_placeholder: Any,
    accumulated: str,
    *,
    reasoning_enabled: bool,
    is_structured: bool,
) -> None:
    """Update the reasoning + output panes from a single accumulated stream chunk."""
    think, output = split_reasoning_and_output(accumulated, reasoning_enabled)

    if reasoning_enabled:
        if think:
            reasoning_placeholder.markdown(f"```text\n{think}\n```")
        else:
            reasoning_placeholder.caption("_(no reasoning yet)_")
    else:
        reasoning_placeholder.caption("_(reasoning disabled)_")

    if not output:
        if reasoning_enabled:
            output_placeholder.caption("_(waiting for output after `</think>`)_")
        else:
            output_placeholder.caption("_(generating...)_")
        return

    if is_structured:
        answer = extract_answer_block(output)
        pretty = pretty_json_or_text(answer)
        if pretty.startswith("{") or pretty.startswith("["):
            output_placeholder.code(pretty, language="json")
        else:
            output_placeholder.markdown(pretty)
    else:
        output_placeholder.markdown(output)


_DOWNLOAD_CONFIGS = {
    "extract": ("Download JSON", "extraction.json", "application/json", True),
    "template": ("Download template", "template.json", "application/json", True),
    "markdown": ("Download Markdown", "document.md", "text/markdown", False),
}


def _render_download_button(
    placeholder: Any, accumulated: str, *, download_kind: str, reasoning: bool
) -> None:
    """Render a download button with content cleaned of reasoning trace + wrappers."""
    _, output = split_reasoning_and_output(accumulated, reasoning)
    label, file_name, mime, is_json = _DOWNLOAD_CONFIGS[download_kind]
    data = extract_answer_block(output) if is_json else output
    with placeholder.container():
        st.download_button(
            label,
            data=data,
            file_name=file_name,
            mime=mime,
            icon=":material/download:",
            key=f"download_{download_kind}",
        )


def _run_mode(
    *,
    mode_label: str,
    model: Any,
    processor: Any,
    image_path: str | None,
    text: str,
    system_prompt: str | None = None,
    template: str | None,
    instructions: str | None,
    mode: str | None,
    reasoning: bool,
    temperature: float,
    max_tokens: int,
    download_kind: str,
    reasoning_placeholder: Any,
    output_placeholder: Any,
    download_placeholder: Any,
) -> None:
    """Drive a streamed generation for one mode and update the UI panes live."""
    is_structured = template is not None and mode is None
    with st.spinner(f"{mode_label}..."):
        accumulated = ""
        try:
            for chunk in stream_extract(
                model,
                processor,
                text=text,
                image_path=image_path,
                system_prompt=system_prompt,
                template=template,
                instructions=instructions,
                mode=mode,
                enable_thinking=reasoning,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                accumulated = chunk
                _render_output_pane(
                    output_placeholder,
                    reasoning_placeholder,
                    accumulated,
                    reasoning_enabled=reasoning,
                    is_structured=is_structured or mode == MODE_TEMPLATE_GENERATION,
                )
        except Exception as e:
            output_placeholder.error(f"{type(e).__name__}: {e}")
            return

    if not accumulated.strip():
        output_placeholder.warning("Empty output from model.")
        return

    _render_download_button(
        download_placeholder,
        accumulated,
        download_kind=download_kind,
        reasoning=reasoning,
    )


# --- Streamlit UI ---

st.set_page_config(
    page_title="NuExtract3",
    page_icon=":material/document_scanner:",
    layout="wide",
)
st.title("NuExtract3")

with st.spinner("Loading model (first run downloads ~5 GB)..."):
    model, processor = get_model()

col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.subheader("Input")
    uploaded_image = st.file_uploader(
        "Image",
        type=["jpg", "jpeg", "png", "webp"],
        help="JPG, PNG, or WEBP image of the document.",
        key="image_input",
    )
    if uploaded_image is not None:
        st.image(uploaded_image, width="stretch")

    text_input = st.text_area(
        "Text (optional)",
        height=100,
        placeholder="Paste document text here, or use the image above.",
        key="text_input",
    )

    st.space("medium")
    st.markdown("**Template (JSON)**")
    st.caption(
        "Describe each field with a type hint, e.g. string, number, or YYYY-MM-DD."
    )
    template_input = st.text_area(
        "Template",
        value=DEFAULT_TEMPLATE,
        height=240,
        label_visibility="collapsed",
        key="template_input",
    )

    instructions_input = st.text_area(
        "Instructions (optional)",
        height=80,
        placeholder="Extra guidance for the model, e.g. 'use British date format'.",
        key="instructions_input",
    )

    col_temp, col_reason, col_tokens = st.columns([2, 1, 2])
    with col_temp:
        temperature = st.slider(
            "Temperature",
            0.0,
            1.0,
            0.0,
            0.05,
            help="0 is deterministic; raise for more varied output.",
            key="temperature_slider",
        )
    with col_reason:
        reasoning = st.checkbox(
            "Reasoning",
            value=False,
            help="Show the model's <think> trace in the Reasoning pane.",
            key="reasoning_checkbox",
        )
    with col_tokens:
        max_tokens = st.slider(
            "Max tokens",
            256,
            8192,
            DEFAULT_MAX_TOKENS,
            256,
            help="Upper bound on generated tokens.",
            key="max_tokens_slider",
        )

    st.space("medium")
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        btn_extract = st.button(
            "Extract JSON",
            type="primary",
            icon=":material/data_object:",
            width="stretch",
            key="extract_button",
        )
    with col_b2:
        btn_markdown = st.button(
            "Convert to Markdown",
            icon=":material/article:",
            width="stretch",
            key="markdown_button",
        )
    with col_b3:
        btn_template = st.button(
            "Generate template",
            icon=":material/auto_awesome:",
            width="stretch",
            key="template_button",
        )

with col_right:
    st.subheader("Output")
    st.markdown("**Reasoning**")
    reasoning_placeholder = st.empty()
    st.markdown("**Result**")
    output_placeholder = st.empty()
    download_placeholder = st.empty()

# --- Button handlers ---

image_path = _save_uploaded_image(uploaded_image)

if btn_extract:
    parsed, error = _validate_template(template_input)
    if error:
        output_placeholder.error(f"Template error: {error}")
    elif not image_path and not text_input.strip():
        output_placeholder.warning("Provide an image, text, or both.")
    else:
        _run_mode(
            mode_label="Extracting",
            model=model,
            processor=processor,
            image_path=image_path,
            text=text_input,
            template=template_input,
            instructions=instructions_input or None,
            mode=None,
            reasoning=reasoning,
            temperature=temperature,
            max_tokens=max_tokens,
            download_kind="extract",
            reasoning_placeholder=reasoning_placeholder,
            output_placeholder=output_placeholder,
            download_placeholder=download_placeholder,
        )

elif btn_markdown:
    if not image_path:
        output_placeholder.warning(
            "Markdown conversion requires an image of the document."
        )
    else:
        _run_mode(
            mode_label="Converting to Markdown",
            model=model,
            processor=processor,
            image_path=image_path,
            text="",
            template=None,
            instructions=None,
            mode=MODE_MARKDOWN,
            reasoning=reasoning,
            temperature=temperature,
            max_tokens=max_tokens,
            download_kind="markdown",
            reasoning_placeholder=reasoning_placeholder,
            output_placeholder=output_placeholder,
            download_placeholder=download_placeholder,
        )

elif btn_template:
    if not image_path and not text_input.strip():
        output_placeholder.warning(
            "Template generation needs an image or text to describe the document."
        )
    else:
        _run_mode(
            mode_label="Generating template",
            model=model,
            processor=processor,
            image_path=image_path,
            text=text_input,
            system_prompt=TEMPLATE_GEN_GUIDANCE,
            template=None,
            instructions=None,
            mode=MODE_TEMPLATE_GENERATION,
            reasoning=False,
            temperature=temperature,
            max_tokens=max_tokens,
            download_kind="template",
            reasoning_placeholder=reasoning_placeholder,
            output_placeholder=output_placeholder,
            download_placeholder=download_placeholder,
        )
