import colorsys
import importlib.metadata
import re
import sys
import tomllib
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Streamlit primitives that only need to be silenced, not given a return value.
_ST_NOOPS = (
    "set_page_config",
    "title",
    "subheader",
    "markdown",
    "caption",
    "space",
    "image",
    "spinner",
)


@pytest.fixture(scope="module")
def app():
    """Import streamlit_app with Streamlit + model loading mocked.

    Patches are entered through an ExitStack rather than one `with` statement.
    The 21 context managers here sit exactly at CPython's statically-nested-
    block ceiling (21 compiles, a 22nd raises SyntaxError at collection time,
    on 3.12 and 3.13 alike), so the next patch added would break a flat `with`.
    """
    import streamlit as st

    with ExitStack() as stack:
        for name in _ST_NOOPS:
            stack.enter_context(patch.object(st, name))
        stack.enter_context(patch.object(st, "file_uploader", return_value=None))
        stack.enter_context(patch.object(st, "text_area", return_value=""))
        stack.enter_context(patch.object(st, "slider", return_value=0.0))
        stack.enter_context(patch.object(st, "toggle", return_value=False))
        stack.enter_context(patch.object(st, "button", return_value=False))
        # side_effect, not return_value: _output_section calls st.empty() once
        # per pane — output and download, plus reasoning while the toggle is on
        # (off here, since session_state is empty). A single return_value hands
        # them all the same mock, so output routed to the wrong pane would still
        # record its calls on the expected object and assert clean.
        stack.enter_context(
            patch.object(st, "empty", side_effect=lambda *a, **k: MagicMock())
        )
        stack.enter_context(
            patch.object(
                st,
                "columns",
                side_effect=lambda spec, **kw: [
                    MagicMock()
                    for _ in range(spec if isinstance(spec, int) else len(spec))
                ],
            )
        )
        stack.enter_context(patch.object(st, "cache_resource", side_effect=lambda f: f))
        stack.enter_context(patch.object(st, "fragment", side_effect=lambda f: f))
        stack.enter_context(patch.object(st, "session_state", {}))

        # Patch the dependency, not the consumer. `patch("streamlit_app.X")`
        # resolves its target by IMPORTING streamlit_app, whose module body
        # calls get_model() — so a real 4.8 GB model download fires during patch
        # setup, before the mock is ever installed. The pop+import below then
        # re-runs `from nuextract import load_model` and discards the mock too.
        # Patching the nuextract.* namespace instead means that import binds the
        # mock. snapshot_download and mlx_vlm_load are belt-and-braces, matching
        # test_streamlit_app_apptest.py: even a path that bypasses load_model
        # cannot reach the network.
        stack.enter_context(
            patch("nuextract.load_model", return_value=(MagicMock(), MagicMock()))
        )
        stack.enter_context(
            patch("nuextract.snapshot_download", return_value="/fake/dir")
        )
        stack.enter_context(
            patch("nuextract.mlx_vlm_load", return_value=(MagicMock(), MagicMock()))
        )

        sys.modules.pop("streamlit_app", None)
        import streamlit_app

        yield streamlit_app
        sys.modules.pop("streamlit_app", None)


# --- _validate_template ---


def test_validate_template_valid(app):
    parsed, error = app._validate_template('{"name": "string"}')
    assert parsed == {"name": "string"}
    assert error is None


def test_validate_template_empty(app):
    parsed, error = app._validate_template("")
    assert parsed is None
    assert "empty" in error.lower()


def test_validate_template_whitespace_only(app):
    parsed, error = app._validate_template("   \n  ")
    assert parsed is None
    assert "empty" in error.lower()


def test_validate_template_invalid_json(app):
    parsed, error = app._validate_template("not json {{{")
    assert parsed is None
    assert "invalid json" in error.lower()


def test_validate_template_non_dict(app):
    parsed, error = app._validate_template("[1, 2, 3]")
    assert parsed is None
    assert "object" in error.lower()


def test_validate_template_empty_object(app):
    parsed, error = app._validate_template("{}")
    assert parsed is None
    assert "empty" in error.lower()


# --- _save_uploaded_image ---


def test_save_uploaded_image_none_returns_none(app):
    assert app._save_uploaded_image(None) is None


def test_save_uploaded_image_none_cleans_up_orphaned_temp_file(app):
    """Removing the upload (uploader returns None) deletes the previously
    written temp file and clears the cached session_state keys, instead of
    leaking the file on disk and leaving stale state behind."""
    import streamlit as st

    st.session_state.clear()

    # Simulate a prior upload: a real temp file recorded in session_state.
    prior = MagicMock()
    prior.name = "doc.png"
    prior.file_id = "id-1"
    prior.getvalue.return_value = b"\x89PNG"
    path = app._save_uploaded_image(prior)
    try:
        assert Path(path).exists()
        assert st.session_state[app._IMG_PATH_KEY] == path

        # Next rerun after the user removes the upload: uploader yields None.
        result = app._save_uploaded_image(None)

        assert result is None
        assert not Path(path).exists()  # orphaned temp file cleaned up
        assert app._IMG_PATH_KEY not in st.session_state
        assert app._IMG_ID_KEY not in st.session_state
    finally:
        # Don't leak the temp file into the system temp dir if an assertion
        # fails before the code-under-test deletes it (no-op on a passing run).
        Path(path).unlink(missing_ok=True)


def test_save_uploaded_image_persists_bytes_to_temp(app, tmp_path):
    fake_upload = MagicMock()
    fake_upload.name = "test.png"
    fake_upload.getvalue.return_value = b"\x89PNG\r\n\x1a\n"  # PNG header

    path = app._save_uploaded_image(fake_upload)
    assert path is not None
    assert Path(path).exists()
    assert Path(path).suffix == ".png"
    assert Path(path).read_bytes() == b"\x89PNG\r\n\x1a\n"
    Path(path).unlink()


def test_save_uploaded_image_falls_back_when_no_extension(app):
    fake_upload = MagicMock()
    fake_upload.name = "no_extension"
    fake_upload.getvalue.return_value = b"data"

    path = app._save_uploaded_image(fake_upload)
    assert Path(path).suffix == ".png"
    Path(path).unlink()


def test_save_uploaded_image_preserves_jpg_suffix(app):
    """Non-PNG extensions (jpg, webp, jpeg) must be preserved so mlx-vlm's
    image loader can pick the right codec."""
    fake_upload = MagicMock()
    fake_upload.name = "photo.jpg"
    fake_upload.getvalue.return_value = b"\xff\xd8\xff"  # JPEG magic

    path = app._save_uploaded_image(fake_upload)
    assert Path(path).suffix == ".jpg"
    Path(path).unlink()


def test_save_uploaded_image_caches_by_file_id(app):
    """Repeated calls with the same upload (same file_id) reuse the cached
    temp file — no re-write on every Streamlit rerun."""
    import streamlit as st

    st.session_state.clear()

    fake_upload = MagicMock()
    fake_upload.name = "doc.png"
    fake_upload.file_id = "stable-id-123"
    fake_upload.getvalue.return_value = b"\x89PNG"

    path1 = app._save_uploaded_image(fake_upload)
    path2 = app._save_uploaded_image(fake_upload)

    assert path1 == path2
    assert Path(path1).exists()
    # Bytes written exactly once across the two calls
    assert fake_upload.getvalue.call_count == 1
    Path(path1).unlink()


def test_save_uploaded_image_cleans_up_previous_on_new_upload(app):
    """A new upload (different file_id) deletes the previous temp file
    before writing the new one."""
    import streamlit as st

    st.session_state.clear()

    first = MagicMock()
    first.name = "first.png"
    first.file_id = "id-1"
    first.getvalue.return_value = b"AAA"

    second = MagicMock()
    second.name = "second.png"
    second.file_id = "id-2"
    second.getvalue.return_value = b"BBB"

    path1 = app._save_uploaded_image(first)
    assert Path(path1).exists()

    path2 = app._save_uploaded_image(second)
    assert path2 != path1
    assert not Path(path1).exists()  # Previous temp file cleaned up
    assert Path(path2).exists()
    Path(path2).unlink()


# --- Constants ---


def test_default_template_is_valid_json(app):
    import json

    parsed = json.loads(app.DEFAULT_TEMPLATE)
    assert isinstance(parsed, dict)
    assert len(parsed) > 0


def test_template_gen_guidance_mentions_json(app):
    assert "JSON" in app.TEMPLATE_GEN_GUIDANCE


# --- _render_output_pane behavior ---


def test_render_output_pane_no_reasoning_renders_output_as_json(app):
    output_ph = MagicMock()
    reasoning_ph = MagicMock()
    app._render_output_pane(
        output_ph,
        reasoning_ph,
        accumulated='{"k": 1}',
        reasoning_enabled=False,
        is_structured=True,
    )
    # Structured + valid JSON → renders via code(...)
    output_ph.code.assert_called_once()
    args, kwargs = output_ph.code.call_args
    assert "1" in args[0]
    assert kwargs.get("language") == "json"


def test_render_output_pane_waiting_for_think_close(app):
    output_ph = MagicMock()
    reasoning_ph = MagicMock()
    app._render_output_pane(
        output_ph,
        reasoning_ph,
        accumulated="still thinking",
        reasoning_enabled=True,
        is_structured=True,
    )
    # No </think> yet → output pane shows the waiting caption
    output_ph.caption.assert_called_once()
    assert "</think>" in output_ph.caption.call_args[0][0]


def test_render_output_pane_markdown_mode_uses_markdown(app):
    output_ph = MagicMock()
    reasoning_ph = MagicMock()
    app._render_output_pane(
        output_ph,
        reasoning_ph,
        accumulated="# Heading\n\nbody text",
        reasoning_enabled=False,
        is_structured=False,
    )
    output_ph.markdown.assert_called_once()
    assert "Heading" in output_ph.markdown.call_args[0][0]


def test_render_output_pane_reasoning_completed_populates_both_panes(app):
    """Reasoning enabled AND </think> arrived: trace goes to reasoning pane,
    answer goes to output pane."""
    output_ph = MagicMock()
    reasoning_ph = MagicMock()
    app._render_output_pane(
        output_ph,
        reasoning_ph,
        accumulated='reasoning text here</think>{"k": 1}',
        reasoning_enabled=True,
        is_structured=True,
    )
    # Reasoning text in reasoning_placeholder, as a code block rather than a
    # hand-built markdown fence the trace itself could break out of.
    # The trace lands in a fixed-height autoscroll container, so the code element
    # is created on that container rather than on the placeholder itself.
    trace_box = reasoning_ph.container.return_value
    trace_box.code.assert_called_once()
    assert "reasoning text here" in trace_box.code.call_args[0][0]
    # JSON answer in output_placeholder
    output_ph.code.assert_called_once()
    assert "1" in output_ph.code.call_args[0][0]


def test_render_output_pane_structured_mode_non_json_falls_back_to_markdown(app):
    """When structured mode is requested but the model returns plain text
    (extract_answer_block falls back to stripped text), render as markdown
    instead of crashing. This is a *final*-pass contract: mid-stream nothing
    inspects the partial, so prose stays in the JSON code block until the end."""
    output_ph = MagicMock()
    reasoning_ph = MagicMock()
    app._render_output_pane(
        output_ph,
        reasoning_ph,
        accumulated="model could not produce JSON for this document",
        reasoning_enabled=False,
        is_structured=True,
        final=True,
    )
    # Non-JSON output → markdown render, not code block
    output_ph.markdown.assert_called_once()
    output_ph.code.assert_not_called()


def test_render_output_pane_streaming_partial_json_is_not_collapsed(app):
    """Mid-stream the outer object is still truncated, so extract_answer_block
    would return the nested object that closed first — visibly shrinking the
    pane to a sub-object. The streaming path must show the raw partial."""
    output_ph = MagicMock()
    reasoning_ph = MagicMock()
    partial = '{"title": "Q3", "amounts": [{"value": 1200, "currency": "USD"},'
    app._render_output_pane(
        output_ph,
        reasoning_ph,
        accumulated=partial,
        reasoning_enabled=False,
        is_structured=True,
    )
    rendered = output_ph.code.call_args[0][0]
    assert rendered == partial
    assert "Q3" in rendered


def test_render_output_pane_streaming_answer_wrapper_still_renders_as_json(app):
    """The <answer> closing tag arrives last, so mid-stream the partial leads
    with the opening tag. It must still render as a JSON code block, not as
    markdown that mangles the JSON until the final pass strips the wrapper."""
    output_ph = MagicMock()
    reasoning_ph = MagicMock()
    app._render_output_pane(
        output_ph,
        reasoning_ph,
        accumulated='<answer>{"title": "Q3", "amo',
        reasoning_enabled=False,
        is_structured=True,
    )
    output_ph.code.assert_called_once()
    output_ph.markdown.assert_not_called()


def test_render_output_pane_final_extracts_answer_and_pretty_prints(app):
    """The completed text does satisfy extract_answer_block's contract, so the
    final render strips the <answer> wrapper and pretty-prints."""
    output_ph = MagicMock()
    reasoning_ph = MagicMock()
    app._render_output_pane(
        output_ph,
        reasoning_ph,
        accumulated='<answer>{"k": 1}</answer>',
        reasoning_enabled=False,
        is_structured=True,
        final=True,
    )
    rendered = output_ph.code.call_args[0][0]
    assert "<answer>" not in rendered
    assert rendered == '{\n  "k": 1\n}'


def test_reasoning_trace_is_height_capped_and_tail_follows(app):
    """The trace scrolls in place *and* follows its own tail.

    Both halves matter and neither is visible to an output assertion. Without a
    fixed height the pane grows for the whole run and pushes the Result pane and
    its download button off-screen; with a fixed height but no autoscroll the
    viewport pins to the top of the trace instead, and since this element is
    re-created on every chunk the newest tokens stream in below the fold and any
    manual scroll is undone by the next one.
    """
    reasoning_ph = MagicMock()
    app._render_output_pane(
        MagicMock(),
        reasoning_ph,
        accumulated='trace text</think>{"k": 1}',
        reasoning_enabled=True,
        is_structured=True,
    )
    reasoning_ph.container.assert_called_once_with(height=300, autoscroll=True)


def test_render_output_pane_hidden_reasoning_pane_still_splits_the_trace(app):
    """With the toggle off there is no Reasoning pane (placeholder None), but a
    replayed run made with reasoning on still carries its trace, which must be
    split off rather than rendered into the Result pane.

    Markdown mode on purpose: structured mode's final pass runs
    extract_answer_block, which digs the JSON out of the surrounding trace on
    its own and would mask a missing split."""
    output_ph = MagicMock()
    app._render_output_pane(
        output_ph,
        None,
        accumulated="trace text</think># Heading",
        reasoning_enabled=True,
        is_structured=False,
        final=True,
    )
    output_ph.markdown.assert_called_once_with("# Heading")


def test_render_output_pane_reasoning_disabled_caption(app):
    """When the pane is shown but the run being painted had reasoning off —
    template generation, or a replay of a run from before the toggle was
    switched on — it says so rather than sitting empty under its header."""
    output_ph = MagicMock()
    reasoning_ph = MagicMock()
    app._render_output_pane(
        output_ph,
        reasoning_ph,
        accumulated='{"k": 1}',
        reasoning_enabled=False,
        is_structured=True,
    )
    reasoning_ph.caption.assert_called_once()
    assert "disabled" in reasoning_ph.caption.call_args[0][0].lower()


# --- _render_download_button ---


def test_render_download_button_extract_mode_emits_clean_json(app):
    """Extract mode → extract_answer_block is applied, label says 'Download JSON',
    filename is extraction.json, mime is application/json."""
    import streamlit as st

    with patch.object(st, "download_button") as mock_dl:
        # The model's raw output may include <answer>...</answer> wrappers
        app._render_download_button(
            MagicMock(),
            app._download_payload(
                '<answer>{"name": "Alice"}</answer>',
                download_kind="extract",
                reasoning=False,
            ),
            download_kind="extract",
        )
    mock_dl.assert_called_once()
    call = mock_dl.call_args
    assert call.args[0] == "Download JSON"
    assert call.kwargs["data"] == '{"name": "Alice"}'  # answer block extracted
    assert call.kwargs["file_name"] == "extraction.json"
    assert call.kwargs["mime"] == "application/json"
    # Client-side download (no fragment rerun that would clear the result/button)
    assert call.kwargs["on_click"] == "ignore"


def test_render_download_button_markdown_mode_keeps_raw_output(app):
    """Markdown mode → no answer-block extraction, label says 'Download Markdown'."""
    import streamlit as st

    with patch.object(st, "download_button") as mock_dl:
        app._render_download_button(
            MagicMock(),
            app._download_payload(
                "# Heading\n\nbody text", download_kind="markdown", reasoning=False
            ),
            download_kind="markdown",
        )
    call = mock_dl.call_args
    assert call.args[0] == "Download Markdown"
    assert call.kwargs["data"] == "# Heading\n\nbody text"
    assert call.kwargs["file_name"] == "document.md"
    assert call.kwargs["mime"] == "text/markdown"


def test_render_download_button_template_mode_treats_as_json(app):
    """Template mode → applies extract_answer_block, filename is template.json."""
    import streamlit as st

    with patch.object(st, "download_button") as mock_dl:
        app._render_download_button(
            MagicMock(),
            app._download_payload(
                '{"field": "string"}', download_kind="template", reasoning=False
            ),
            download_kind="template",
        )
    call = mock_dl.call_args
    assert call.args[0] == "Download template"
    assert call.kwargs["data"] == '{"field": "string"}'
    assert call.kwargs["file_name"] == "template.json"


def test_render_download_button_strips_reasoning_trace(app):
    """When reasoning=True, the <think>...</think> prefix is stripped before
    the content goes into the download."""
    import streamlit as st

    with patch.object(st, "download_button") as mock_dl:
        app._render_download_button(
            MagicMock(),
            app._download_payload(
                'reasoning text...</think>{"name": "Bob"}',
                download_kind="extract",
                reasoning=True,
            ),
            download_kind="extract",
        )
    assert mock_dl.call_args.kwargs["data"] == '{"name": "Bob"}'


# --- theme ---

_THEME_CONFIG = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"

# The Streamlit release every internal below was read from. The contrast test
# refuses any other, because a release can change those internals while the
# copies here keep passing.
_THEME_VERIFIED_STREAMLIT = "1.64.0"

# Streamlit hard-codes these code-block token colours (Prism classes mapped to
# fixed palette entries in the frontend bundle) rather than deriving them from
# the theme, so codeBackgroundColor is the only lever over their contrast.
_PRISM_TOKEN_COLORS = {
    "JSON key": "#00a4d4",
    "string": "#09ab3b",
    "number": "#29b09d",
    "boolean": "#21c354",
    "null": "#1c83e1",
    "punctuation": "#808495",
    "colon": "#ed6f13",
}

# How Streamlit derives a semantic colour's variants from its base (per
# `streamlit config show`, and measured in a browser): alert and link text is
# the base lightened by 0.15 HSL lightness, and the alert fill is the base at
# 20% alpha over the canvas. Streamlit's own rounding lands within one unit
# per channel of _lighten's, which moves no ratio here across its floor.
_SEMANTIC_TEXT_LIGHTEN = 0.15
_SEMANTIC_FILL_ALPHA = 0.2
# Stock dark's red (red60) for both alert text and fill base while redColor is
# unset — st.error's colours, measured on this config's canvas.
_STOCK_DARK_RED = "#ff6c6c"


def _theme():
    """The [theme] table of the committed .streamlit/config.toml."""
    with _THEME_CONFIG.open("rb") as f:
        return tomllib.load(f)["theme"]


def _rgb(color):
    """A #rrggbb colour's channels as 0-1 floats.

    Fails loudly on the other spellings Streamlit accepts (#rgb, no leading
    #, #rrggbbaa), which slicing would misread or treat as opaque.
    """
    assert re.fullmatch(r"#[0-9a-fA-F]{6}", color), (
        f"{color!r}: the theme tests handle opaque #rrggbb colours only"
    )
    return tuple(int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))


def _hex(channels):
    """The #rrggbb spelling of 0-1 float channels."""
    return "#" + "".join(f"{round(c * 255):02x}" for c in channels)


def _lighten(color, amount):
    """color with its HSL lightness raised by amount, capped at white."""
    hue, lightness, saturation = colorsys.rgb_to_hls(*_rgb(color))
    return _hex(colorsys.hls_to_rgb(hue, min(1, lightness + amount), saturation))


def _over(color, alpha, bg):
    """color at the given alpha, composited over an opaque bg."""
    return _hex(alpha * c + (1 - alpha) * b for c, b in zip(_rgb(color), _rgb(bg)))


def _contrast(fg, bg):
    """WCAG 2.x contrast ratio between two #rrggbb colours."""

    def luminance(color):
        r, g, b = (
            c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
            for c in _rgb(color)
        )
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    lighter, darker = sorted((luminance(fg), luminance(bg)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def test_theme_config_customises_dark_mode_only():
    """Only dark-mode colours may be set, so Light and the toggle stay stock.

    Streamlit builds each mode as root [theme] merged with that mode's section,
    so a root key would restyle Light as well, and a [theme.light] section would
    stop Light matching stock. Either variant section alone keeps Light/Dark/
    System in the Settings menu; root keys with no variant section would lock
    the app to a single mode. Within [theme.dark], colours only: a font,
    radius, border toggle or link underline would change the app's shape
    whenever the user flips the toggle.
    """
    theme = _theme()
    assert set(theme) == {"dark"}, (
        f"[theme] must hold only the dark variant, found {sorted(theme)}"
    )
    dark = theme["dark"]
    non_colour = [
        key
        for section in (dark, dark.get("sidebar", {}))
        for key in section
        if key != "sidebar" and not key.endswith(("Color", "Colors"))
    ]
    assert not non_colour, f"[theme.dark] must set colours only, found {non_colour}"


def test_dark_theme_meets_its_contrast_floors():
    """The pairs .streamlit/config.toml's comments justify each colour by.

    Each rests on a Streamlit internal: the primary button label is hard-coded
    white, the toggle knob is drawn in textColor, code-block tokens use the
    fixed _PRISM_TOKEN_COLORS, and semantic text and fills derive from their
    base colour. A tweak that looks harmless — a lighter primary, a lifted code
    well — fails here instead of shipping unreadable. Sidebar fallbacks mirror
    Streamlit's, which merges [theme.dark.sidebar] over [theme.dark].
    """
    installed = importlib.metadata.version("streamlit")
    assert installed == _THEME_VERIFIED_STREAMLIT, (
        f"streamlit {installed} is installed, but this test copies internals "
        f"read from {_THEME_VERIFIED_STREAMLIT}. Re-verify _PRISM_TOKEN_COLORS, "
        "the white primary-button label, the textColor toggle knob and the "
        "semantic derivations against the new frontend bundle, then bump "
        "_THEME_VERIFIED_STREAMLIT."
    )
    dark = _theme()["dark"]
    sidebar = dark.get("sidebar", {})
    sidebar_primary = sidebar.get("primaryColor", dark["primaryColor"])
    sidebar_bg = sidebar.get("backgroundColor", dark["secondaryBackgroundColor"])
    sidebar_text = sidebar.get("textColor", dark["textColor"])
    canvas, code_bg = dark["backgroundColor"], dark["codeBackgroundColor"]
    floors = {
        "white label on the primary button": ("#ffffff", dark["primaryColor"], 4.5),
        "primary on the canvas": (dark["primaryColor"], canvas, 3),
        "primary focus border on inputs": (
            dark["primaryColor"],
            dark["secondaryBackgroundColor"],
            3,
        ),
        "text on the canvas": (dark["textColor"], canvas, 4.5),
        "text in the Reasoning pane": (dark["textColor"], code_bg, 4.5),
        "sidebar slider readouts": (sidebar_primary, sidebar_bg, 4.5),
        "toggle knob on its ON track": (sidebar_text, sidebar_primary, 3),
    } | {
        f"{name} token in the Result pane": (color, code_bg, 4.5)
        for name, color in _PRISM_TOKEN_COLORS.items()
    }
    # (text, fill base) for each semantic colour the config sets, plus red:
    # st.error is the app's own alert, and it sits on this canvas even while
    # redColor is unset and it keeps stock dark's values.
    semantic = {"red": (_STOCK_DARK_RED, _STOCK_DARK_RED)}
    for name in ("red", "orange", "yellow", "blue", "green"):
        if base := dark.get(f"{name}Color"):
            semantic[name] = (_lighten(base, _SEMANTIC_TEXT_LIGHTEN), base)
    for name, (text, fill_base) in semantic.items():
        text = dark.get(f"{name}TextColor", text)
        fill = dark.get(
            f"{name}BackgroundColor", _over(fill_base, _SEMANTIC_FILL_ALPHA, canvas)
        )
        floors[f"{name} alert text on its fill"] = (text, fill, 4.5)
    # Links default to the resolved blue text colour.
    if "blue" in semantic or "linkColor" in dark:
        link = dark.get("linkColor") or semantic["blue"][0]
        floors["links on the canvas"] = (link, canvas, 4.5)
    failures = [
        f"{label}: {fg} on {bg} is {_contrast(fg, bg):.2f}:1, needs {floor}:1"
        for label, (fg, bg, floor) in floors.items()
        if _contrast(fg, bg) < floor
    ]
    assert not failures, "\n".join(failures)
