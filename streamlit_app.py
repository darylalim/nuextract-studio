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
    DEFAULT_TEMPERATURE,
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
def get_model() -> tuple[Any, Any] | Exception:
    """Load the model + processor pair once per server process.

    Returns a failure rather than raising it, so the failure is cached too:
    st.cache_resource writes an entry only when the function *returns*, so a
    raising load is re-attempted on every rerun. The input widgets now render
    before this runs and stay live through a failed load, which means every
    slider drag or upload would otherwise queue another download attempt.
    """
    try:
        return load_model()
    except Exception as exc:
        return exc


# Megabytes; see the file_uploader call that passes it.
_MAX_IMAGE_UPLOAD_MB = 25

# The last completed run, replayed when a full rerun re-creates the output
# placeholders with no generate button pressed.
_LAST_RUN_KEY = "_last_run"
_LAST_RUN_FIELDS = (
    "accumulated",
    "payload",
    "download_kind",
    "reasoning",
    "render_as_json",
)

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
    reasoning_placeholder: Any | None,
    accumulated: str,
    *,
    reasoning_enabled: bool,
    is_structured: bool,
    final: bool = False,
) -> None:
    """Update the reasoning + output panes from a single accumulated stream chunk.

    `reasoning_placeholder` is None while the Reasoning pane is hidden (the
    toggle is off). The trace is still split off below, so it stays out of
    the Result pane even when it isn't shown.

    `final` marks the one call made after the stream completes. Only then does
    the text satisfy extract_answer_block's whole-document contract — see the
    structured branch below.
    """
    think, output = split_reasoning_and_output(accumulated, reasoning_enabled)

    if reasoning_placeholder is None:
        pass
    elif reasoning_enabled:
        if think:
            # st.code, not a hand-built ```text fence: `think` is untrusted model
            # output, and a fence inside it would close ours and hand the rest to
            # the Markdown renderer. Markdown-mode reasoning quotes fences often.
            # Fixed height on a *container*, not on st.code, and autoscroll=True.
            # The pane has to scroll rather than grow, because the Result pane and
            # its download button sit *below* it and an uncapped trace pushes them
            # off-screen for the whole run. But st.code exposes no autoscroll, and
            # this element is re-created on every chunk, so a bare
            # st.code(height=...) pins the viewport to the *top* of the trace: the
            # newest tokens stream in below the fold and any manual scroll is
            # reset by the next chunk. That is worse than growing, where the newest
            # text was at least always visible. A fixed-height container with
            # autoscroll is the documented way to tail-follow.
            trace_box = reasoning_placeholder.container(height=300, autoscroll=True)
            trace_box.code(think, language=None, wrap_lines=True)
        else:
            reasoning_placeholder.caption("_(no reasoning yet)_")
    else:
        # The pane is shown but the run being painted had reasoning off: template
        # generation, which always forces it off, or a replayed run made before
        # the toggle was switched on.
        reasoning_placeholder.caption("_(reasoning disabled)_")

    if not output:
        if reasoning_enabled:
            output_placeholder.caption("_(waiting for output after `</think>`)_")
        else:
            output_placeholder.caption("_(generating...)_")
        return

    if is_structured and not final:
        # Mid-stream: show the raw partial, and do not inspect it to decide how.
        # extract_answer_block and pretty_json_or_text have whole-document
        # contracts every partial violates — the outer object is still truncated,
        # so the longest span that *parses* is whichever nested object closed
        # first, and the pane would shrink to it and freeze until the last token.
        # Sniffing the prefix instead is no better: a preamble, an <answer>
        # wrapper in any case, or a stray <think> would each route the JSON
        # through the Markdown renderer for the whole run. Structured mode is
        # asking for JSON, so render it as JSON and let the final pass judge.
        output_placeholder.code(output, language="json", wrap_lines=True)
    elif is_structured:
        body = pretty_json_or_text(extract_answer_block(output))
        if body.startswith("{") or body.startswith("["):
            output_placeholder.code(body, language="json", wrap_lines=True)
        else:
            output_placeholder.markdown(body)
    else:
        output_placeholder.markdown(output)


_DOWNLOAD_CONFIGS = {
    "extract": ("Download JSON", "extraction.json", "application/json", True),
    "template": ("Download template", "template.json", "application/json", True),
    "markdown": ("Download Markdown", "document.md", "text/markdown", False),
}


def _download_payload(accumulated: str, *, download_kind: str, reasoning: bool) -> str:
    """The exact text a download would write: reasoning trace and wrappers gone.

    This is also the only honest test for "did the run produce anything?".
    `accumulated` still holds the trace and any `<answer>` wrapper, so it is
    non-empty for runs that produced no answer at all — see the guard in
    `_run_mode`.
    """
    _, output = split_reasoning_and_output(accumulated, reasoning)
    is_json = _DOWNLOAD_CONFIGS[download_kind][3]
    return extract_answer_block(output) if is_json else output


def _render_download_button(
    placeholder: Any, payload: str, *, download_kind: str
) -> None:
    """Render a download button for an already-cleaned `_download_payload`."""
    label, file_name, mime, _ = _DOWNLOAD_CONFIGS[download_kind]
    with placeholder.container():
        # on_click="ignore" keeps the download client-side so clicking it does
        # not rerun the _output_section fragment — a rerun would repaint the idle
        # hint over the result and drop this button (no generate button is active).
        st.download_button(
            label,
            data=payload,
            file_name=file_name,
            mime=mime,
            on_click="ignore",
            icon=":material/download:",
            key=f"download_{download_kind}",
        )


def _render_completed_run(
    state: dict,
    *,
    output_placeholder: Any,
    reasoning_placeholder: Any,
    download_placeholder: Any,
    streamed_live: bool = False,
) -> None:
    """Paint the terminal state of a finished run: its panes, then the download.

    The Reasoning pane is painted only when shown — its placeholder is None
    while the toggle is off (see `_render_output_pane`).

    The single implementation behind both the just-finished path in `_run_mode`
    and the replay in `_output_section`, so the two cannot drift about what a
    completed run looks like — the hazard flagged for `render_as_json`.

    `streamed_live` marks the call made straight after streaming, and is the one
    place the mode asymmetry lives: there the panes already hold the last
    streaming paint, and in markdown mode the final pass reproduces it byte for
    byte, so repainting would re-send the largest document the app produces for
    no visible change. A replay never qualifies — its placeholders were just
    re-created empty, so there is nothing for the paint to be identical to.
    """
    if state["render_as_json"] or not streamed_live:
        _render_output_pane(
            output_placeholder,
            reasoning_placeholder,
            state["accumulated"],
            reasoning_enabled=state["reasoning"],
            is_structured=state["render_as_json"],
            final=True,
        )
    _render_download_button(
        download_placeholder, state["payload"], download_kind=state["download_kind"]
    )


def _stored_run() -> dict | None:
    """The stored completed run, or None if nothing usable is stored.

    Shape-checked rather than trusted: Streamlit re-runs an edited script in the
    *same* session on hot reload, with session_state preserved, so a stored dict
    outlives the code that wrote it. A bare subscript would turn adding or
    renaming a field into a KeyError traceback in a live session — which is this
    app's own development loop.
    """
    stored = st.session_state.get(_LAST_RUN_KEY)
    if not isinstance(stored, dict) or any(f not in stored for f in _LAST_RUN_FIELDS):
        return None
    return stored


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
    render_as_json = is_structured or mode == MODE_TEMPLATE_GENERATION
    with st.spinner(f"{mode_label}...", show_time=True):
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
                    is_structured=render_as_json,
                )
        except Exception as e:
            output_placeholder.error(f"{type(e).__name__}: {e}")
            return

    # Guard on the payload a download would actually write, never on `accumulated`:
    # the raw text still carries the reasoning trace and any <answer> wrapper, so
    # two runs that produced no answer at all are non-empty there — a budget spent
    # entirely inside <think> (</think> never arrives), and a closed but empty
    # <answer></answer>. Both used to slip past this guard and end with a stale
    # mid-stream caption as the terminal state, or a blanked pane, and an
    # empty-payload download button beside it.
    payload = _download_payload(
        accumulated, download_kind=download_kind, reasoning=reasoning
    )
    if not payload.strip():
        think, output = split_reasoning_and_output(accumulated, reasoning)
        if think and not output:
            # Actionable: the run did work, it just never got to an answer.
            output_placeholder.warning(
                "The model spent its whole token budget reasoning and never "
                "reached an answer. Raise Max tokens in the sidebar, or turn "
                "Reasoning off."
            )
        else:
            output_placeholder.warning("Empty output from model.")
        return

    # One dict, stored and rendered through one renderer, so a later replay
    # cannot disagree with what this run just painted. Built below the
    # empty-payload guard, so a run that produced no answer never becomes a
    # replayable "result", and it carries `payload` rather than re-deriving it —
    # deriving it twice is how the guard and the button drift apart.
    state = {
        "accumulated": accumulated,
        "payload": payload,
        "download_kind": download_kind,
        "reasoning": reasoning,
        "render_as_json": render_as_json,
    }
    st.session_state[_LAST_RUN_KEY] = state
    _render_completed_run(
        state,
        output_placeholder=output_placeholder,
        reasoning_placeholder=reasoning_placeholder,
        download_placeholder=download_placeholder,
        streamed_live=True,
    )


@st.fragment
def _output_section() -> None:
    """Fragment: action buttons + the streamed reasoning/result/download panes.

    Isolated in a fragment so clicking a generate button reruns only this
    region — the input widgets in the sidebar and left column keep their state
    and are not re-rendered while a generation runs. The buttons live inside
    the fragment because that is what gives this isolation by default: a
    fragment reruns on its own when the triggering widget is inside it. (Since
    Streamlit 1.63 an outside widget can also target a keyed fragment via
    st.rerun("<key>") from a callback — more machinery for the same result
    here.) Input values are read from session_state, which the keyed widgets
    outside this fragment populate on the full rerun that precedes it.

    Loads the model itself rather than taking it as an argument, so the load
    can sit below this section's own chrome instead of above it — see the
    comment on the load below.
    """
    # The primary action gets its own row and the two secondary modes share the
    # next. All three on one row need ~493px (measured natural widths plus
    # gaps), and with the sidebar open this column only reaches that from a
    # ~1480px viewport up. Below it — including the 13-inch MacBook Air's 1440
    # and 1470px — a single horizontal container wrapped and left "Generate
    # template" stretched full-width on a row of its own, outweighing the
    # primary action. Two deliberate rows look the same at every width; the
    # secondary pair alone needs ~351px, which fits from ~1200px up.
    btn_extract = st.button(
        "Extract JSON",
        help="Needs a valid JSON template, plus an image or text.",
        type="primary",
        icon=":material/data_object:",
        width="stretch",
        key="extract_button",
    )
    with st.container(horizontal=True):
        btn_markdown = st.button(
            "Convert to Markdown",
            help="Needs an image of the document.",
            icon=":material/article:",
            width="stretch",
            key="markdown_button",
        )
        btn_template = st.button(
            "Generate template",
            help="Needs an image or text to describe the document.",
            icon=":material/auto_awesome:",
            width="stretch",
            key="template_button",
        )

    # The Reasoning pane exists only while the toggle is on. Off is the default,
    # and an always-present pane spent a header and a "(reasoning disabled)"
    # caption above the Result on every run. Keyed to the toggle rather than to
    # the run: when a button fires the toggle *is* the run's setting, and on a
    # replay the pane follows the current choice, so switching reasoning off
    # hides a stored trace and switching it back on replays it.
    reasoning = st.session_state.get("reasoning_checkbox", False)
    reasoning_placeholder = None
    if reasoning:
        st.markdown("**Reasoning**")
        reasoning_placeholder = st.empty()
        # Painted at creation, not only on the idle path: every path that ends
        # without a trace leaves this placeholder unwritten — a validation
        # warning, an empty stream — and a bold header over an unwritten
        # st.empty() renders as a void. Streaming, the replay and the
        # reasoning-disabled caption all overwrite it.
        reasoning_placeholder.caption("_(no run yet)_")
    st.markdown("**Result**")
    output_placeholder = st.empty()
    download_placeholder = st.empty()

    # The load runs here, after the buttons and the pane headers (Reasoning only
    # while the toggle is on) have claimed their positions, so the whole page is
    # painted before it blocks: Streamlit emits a UI delta per st.* call, so
    # only what follows this line waits on it. The wait shows inside the Result
    # slot, where the output will land.
    with output_placeholder.container():
        with st.spinner("Loading model (first run downloads ~5 GB)...", show_time=True):
            loaded = get_model()
    if isinstance(loaded, Exception):
        with output_placeholder.container():
            st.error(f"Model failed to load — {type(loaded).__name__}")
            # Message through st.code, not st.error: st.error renders its body as
            # GitHub-flavored Markdown with soft breaks disabled, so the
            # multi-line messages this path actually produces (an offline hub
            # download, or transformers' eight-line "requires the PyTorch
            # library" error when mlx-vlm's own processor load fails and falls
            # through to it) collapse onto one line. st.code also keeps
            # untrusted text out of the Markdown renderer, the same reason the
            # reasoning trace uses it.
            st.code(str(loaded) or repr(loaded), language=None, wrap_lines=True)
            # get_model returns the exception instead of raising it, so Streamlit
            # never reports it and nothing logs it — this expander is the only
            # place the frame that names the cause survives. Collapsed by
            # default: the headline is enough unless you are debugging. Note this
            # lands in AppTest's `at.exception` bucket, which this suite uses as
            # its crash detector — a test for this branch must assert on
            # `at.exception[0].message`, never `assert not at.exception`.
            with st.expander("Traceback", icon=":material/bug_report:"):
                st.exception(loaded)
            # Cached failure (see get_model): retrying is an explicit click,
            # not something every widget interaction re-triggers.
            if st.button(
                "Retry model load",
                icon=":material/refresh:",
                key="retry_model_load_button",
            ):
                get_model.clear()
                st.rerun()
        return
    model, processor = loaded

    # Idle hint, shown only when no generate button fired this run: keeps the
    # Result pane from being blank on first load, without lingering over the
    # spinner during generation or repainting after an output-less rerun.
    if btn_extract or btn_markdown or btn_template:
        # A generate button supersedes any stored run: from here on the panes
        # belong to *this* attempt. Without this, every failure path leaves the
        # previous run stored — a stream exception, an empty payload, or a
        # validation error that never reaches _run_mode — and the next full rerun
        # replays it, with a live download button, over the error the user just
        # got. That presents stale output as current, and across modes: a failed
        # markdown run would resurrect an extract result under "Download JSON".
        st.session_state.pop(_LAST_RUN_KEY, None)
    else:
        last_run = _stored_run()
        if last_run is None:
            # Nothing has run yet, so keep the Result pane from being blank.
            output_placeholder.caption("Choose an action above to generate output.")
        else:
            # Replay rather than paint the idle hint over a result that cost a
            # whole local generation: the sidebar and left-column widgets sit
            # outside this fragment, so touching any of them is a full rerun
            # that re-creates the placeholders empty with no button pressed.
            _render_completed_run(
                last_run,
                output_placeholder=output_placeholder,
                reasoning_placeholder=reasoning_placeholder,
                download_placeholder=download_placeholder,
            )

    # Inputs live in the left column and the sidebar (outside this fragment);
    # read their current values from session_state via their widget keys.
    # `reasoning` was read above, where it decided whether the pane exists.
    image_path = _save_uploaded_image(st.session_state.get("image_input"))
    text = st.session_state.get("text_input", "")
    template_str = st.session_state.get("template_input", "")
    instructions = st.session_state.get("instructions_input", "")
    temperature = st.session_state.get("temperature_slider", DEFAULT_TEMPERATURE)
    max_tokens = st.session_state.get("max_tokens_slider", DEFAULT_MAX_TOKENS)

    if btn_extract:
        _, error = _validate_template(template_str)
        if error:
            output_placeholder.error(f"Template error: {error}")
        elif not image_path and not text.strip():
            output_placeholder.warning("Provide an image, text, or both.")
        else:
            _run_mode(
                mode_label="Extracting",
                model=model,
                processor=processor,
                image_path=image_path,
                text=text,
                template=template_str,
                instructions=instructions or None,
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
        if not image_path and not text.strip():
            output_placeholder.warning(
                "Template generation needs an image or text to describe the document."
            )
        else:
            _run_mode(
                mode_label="Generating template",
                model=model,
                processor=processor,
                image_path=image_path,
                text=text,
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


# --- Streamlit UI ---

st.set_page_config(
    page_title="NuExtract Studio",
    page_icon=":material/document_scanner:",
    layout="wide",
)
st.title("NuExtract Studio", icon=":material/document_scanner:")

# Generation settings live in the sidebar: they apply to every mode and rarely
# change between runs, so they should not compete with the document and the
# template for the main area's height. They sat under the template editor
# before, below the fold. Keyed like every input, and rendered before the main
# area, so the fragment's model load still waits for them (see col_right).
with st.sidebar:
    st.subheader("Settings", icon=":material/tune:")
    st.slider(
        "Temperature",
        0.0,
        1.0,
        DEFAULT_TEMPERATURE,
        0.05,
        help="0 is deterministic; raise for more varied output.",
        key="temperature_slider",
    )
    st.slider(
        "Max tokens",
        256,
        8192,
        DEFAULT_MAX_TOKENS,
        256,
        help="Upper bound on generated tokens.",
        key="max_tokens_slider",
    )
    # st.toggle, not st.checkbox: this is an app setting that changes how a run
    # behaves, and the bundled selection-widgets.md for this pin reserves the
    # checkbox for forms. The key keeps its original name — it is one of the
    # anchors test_model_loads_after_the_page_chrome_renders asserts on, and
    # churning a load-bearing identifier for cosmetics is not worth it.
    st.toggle(
        "Reasoning",
        value=False,
        help=(
            "Show the model's `<think>` trace in a Reasoning pane above the "
            "result. Ignored by **Generate template** — the model's template "
            "only permits reasoning for extraction and Markdown."
        ),
        key="reasoning_checkbox",
    )

col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    # Two tabs rather than one tall column: stacked, the preview, text, template
    # and instructions ran well past the fold. Both tabs' widgets must still run
    # on every full rerun — the default on_change="ignore" computes every tab —
    # because the fragment reads them from session_state by key, and Streamlit
    # drops the state of a widget that skips a run. Gating a tab's body on
    # `.open` (lazy tabs) would make Extract see an empty template whenever the
    # Document tab was showing.
    tab_document, tab_template = st.tabs(
        [":material/description: Document", ":material/schema: Template"]
    )

    with tab_document:
        uploaded_image = st.file_uploader(
            "Image",
            type=["jpg", "jpeg", "png", "webp"],
            help="JPG, PNG, or WEBP image of the document.",
            # A browser-side bound only, in megabytes. Streamlit's upload route
            # enforces server.maxUploadSize and never reads this value, so this
            # rejects an oversized file in the widget before it uploads rather
            # than guaranteeing anything server-side. It does not bound decode
            # memory either: a 1 MB flat-colour 10000x10000 PNG still expands to
            # ~300 MB of pixels. Kept for the widget hint, which is the part
            # users actually see — a real server-side bound would mean adding
            # server.maxUploadSize to .streamlit/config.toml, which currently
            # holds only the dark theme.
            max_upload_size=_MAX_IMAGE_UPLOAD_MB,
            key="image_input",
        )
        if uploaded_image is not None:
            # A fixed height, not a cap — there is no max-height container, and
            # st.image has no height parameter. The input this app is built for
            # is a document page: a portrait scan is ~1.4x taller than this
            # column is wide, so without this it pushes the Text box below the
            # fold the moment an image is attached. 360 rather than the 320 it
            # was when the template editor shared this column: measured in a
            # 839px-tall viewport, it is the tallest box that still keeps the
            # whole Text box on screen beneath it (400 cut it off by 30px).
            # Tall images scroll inside the box; the cost is that anything
            # shorter than the box is padded. border=False is explicit because
            # a fixed-height container draws one by default.
            with st.container(height=360, border=False):
                st.image(uploaded_image, width="stretch")

        # Keyed inputs feed session_state; the _output_section fragment reads
        # their values by key rather than capturing the return values here.
        st.text_area(
            "Text (optional)",
            height=100,
            placeholder="Paste document text here, or use the image above.",
            key="text_input",
        )

    with tab_template:
        st.caption(
            "Describe each field with a type hint, e.g. string, number, or YYYY-MM-DD."
        )
        st.text_area(
            # Collapsed because the tab already names it, but still the widget's
            # accessible name, so it stays close to the tab label a sighted user
            # reads.
            "Template (JSON)",
            value=DEFAULT_TEMPLATE,
            height=320,
            label_visibility="collapsed",
            key="template_input",
        )

        st.text_area(
            # 98 is the floor Streamlit enforces for a visible label, not a
            # chosen size: the 80 that used to sit here was silently clamped up
            # to it, so the number read as intent while doing nothing. Stating
            # the floor keeps the rendering identical and makes the constraint
            # visible. Dropping the parameter instead would take the default —
            # three lines, i.e. *taller* than the Text box — inverting the
            # intent this field was written with.
            "Instructions (optional)",
            height=98,
            placeholder="Extra guidance for the model, e.g. 'use British date format'.",
            key="instructions_input",
        )

with col_right:
    # No model load here: _output_section loads it itself, below its own buttons
    # and pane headers, so nothing on the page waits on the ~5 GB download
    # except the Result slot the output lands in.
    _output_section()
