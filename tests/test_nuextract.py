import ast
import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nuextract import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    MODE_CONTENT,
    MODE_MARKDOWN,
    MODE_STRUCTURED,
    MODE_TEMPLATE_GENERATION,
    build_messages,
    extract_answer_block,
    load_model,
    pretty_json_or_text,
    render_prompt,
    split_reasoning_and_output,
    stream_extract,
)

# --- mlx-vlm processor substitution ---
#
# The invariant the stack actually depends on. mlx_vlm.load() builds the
# processor through transformers' AutoProcessor.from_pretrained, but importing
# mlx_vlm.models.qwen3_5 (which load() does while resolving the model class,
# before any processor is built) patches that call to return mlx-vlm's own
# numpy Qwen3VLProcessor for any checkpoint whose config.json says model_type
# "qwen3_5". That substitution is why the app needs neither torch nor
# torchvision, and why the image_processor_type in processor_config.json is
# never read. Should it stop, loading falls through to transformers' classes,
# which need torchvision and do read that key — with the upstream value this
# test writes, that path raises "Unrecognized image processor". Every other
# test mocks mlx-vlm, so this is the only one that would notice.

_UPSTREAM_PROCESSOR_CONFIG = {
    "processor_class": "Qwen3VLProcessor",
    "image_processor": {
        # Deliberately the value numind/NuExtract3-mlx-8bits ships, unpatched.
        "image_processor_type": "Qwen3VLImageProcessor",
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "min_pixels": 65536,
        "max_pixels": 16777216,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    },
    "video_processor": {"video_processor_type": "Qwen3VLVideoProcessor"},
}


def _write_weightless_qwen3_5_checkpoint(directory):
    """Write the smallest local checkpoint mlx-vlm builds a qwen3_5 processor from.

    A word-level tokenizer holding only the special tokens Qwen3VLProcessor
    looks up stands in for the real 20 MB one; there are no weights, so
    nothing here downloads or loads a model.
    """
    from tokenizers import Tokenizer, models

    specials = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]
    vocab = {token: index for index, token in enumerate(["[UNK]", *specials])}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.add_special_tokens(specials)
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "TokenizersBackend",
                "eos_token": "<|im_end|>",
                "pad_token": "<|endoftext|>",
            }
        )
    )
    (directory / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    (directory / "processor_config.json").write_text(
        json.dumps(_UPSTREAM_PROCESSOR_CONFIG)
    )


def test_mlx_vlm_substitutes_its_own_processor_for_qwen3_5(tmp_path, monkeypatch):
    """mlx_vlm.load() must build mlx-vlm's torch-free processor, not transformers'.

    Only weight loading is stubbed. load() resolves the model package itself —
    in load_model(), which the stub mirrors, and again in load_image_processor()
    — before it builds the processor, so everything from that import to the
    returned processor is mlx-vlm's real code path, including the order.
    """
    import mlx_vlm.utils
    from PIL import Image

    def _load_model_without_weights(model_path, lazy=False, **kwargs):
        """Resolve the model package as load_model() does, then skip the weights."""
        config = mlx_vlm.utils.load_config(model_path)
        mlx_vlm.utils.get_model_and_args(config=config, model_path=model_path)
        return MagicMock(config=MagicMock(eos_token_id=None))

    _write_weightless_qwen3_5_checkpoint(tmp_path)
    monkeypatch.setattr(mlx_vlm.utils, "load_model", _load_model_without_weights)

    _, processor = mlx_vlm.utils.load(str(tmp_path))

    for component in (
        processor,
        processor.image_processor,
        processor.video_processor,
    ):
        # A prefix, not a class identity: an mlx-vlm rename should not fail
        # this, falling back to transformers.* should.
        assert type(component).__module__.startswith("mlx_vlm."), type(component)
    # The geometry comes from processor_config.json, not the class defaults
    # (min_pixels 3136), which would give a 4x4 grid for this image.
    grid = processor.image_processor(images=[Image.new("RGB", (32, 32))])
    assert grid["image_grid_thw"].tolist() == [[1, 16, 16]]


# --- build_messages ---


def test_build_messages_text_only():
    msgs = build_messages(text="Hello world")
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == [{"type": "text", "text": "Hello world"}]


def test_build_messages_image_only():
    msgs = build_messages(image_path="/tmp/x.png")
    assert msgs[0]["content"] == [{"type": "image", "image": "/tmp/x.png"}]


def test_build_messages_image_and_text():
    msgs = build_messages(text="describe", image_path="/tmp/x.png")
    content = msgs[0]["content"]
    assert content[0] == {"type": "image", "image": "/tmp/x.png"}
    assert content[1] == {"type": "text", "text": "describe"}


def test_build_messages_empty_yields_empty_text_part():
    msgs = build_messages()
    assert msgs[0]["content"] == [{"type": "text", "text": ""}]


def test_build_messages_strips_whitespace_only_text():
    msgs = build_messages(text="   \n  ", image_path="/tmp/x.png")
    # Whitespace-only text is dropped, only image remains
    assert msgs[0]["content"] == [{"type": "image", "image": "/tmp/x.png"}]


def test_build_messages_with_system_prompt():
    """System prompt prepends a system message; user message stays second."""
    msgs = build_messages(text="user text", system_prompt="be a JSON expert")
    assert len(msgs) == 2
    assert msgs[0] == {"role": "system", "content": "be a JSON expert"}
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == [{"type": "text", "text": "user text"}]


def test_build_messages_whitespace_system_prompt_dropped():
    """Empty/whitespace system_prompt is ignored — no system message added."""
    msgs = build_messages(text="hi", system_prompt="   \n  ")
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"


# --- render_prompt ---


def test_render_prompt_passes_template_kwarg_inline():
    processor = MagicMock()
    processor.apply_chat_template.return_value = "rendered"
    messages = [{"role": "user", "content": []}]

    out = render_prompt(
        processor, messages, template='{"name":"string"}', enable_thinking=False
    )
    assert out == "rendered"
    call = processor.apply_chat_template.call_args
    assert call.args[0] == messages
    assert call.kwargs["template"] == '{"name":"string"}'
    assert call.kwargs["enable_thinking"] is False
    assert call.kwargs["tokenize"] is False
    assert call.kwargs["add_generation_prompt"] is True
    # Critical: should NOT nest under chat_template_kwargs (vLLM convention)
    assert "chat_template_kwargs" not in call.kwargs


def test_render_prompt_omits_none_kwargs():
    processor = MagicMock()
    processor.apply_chat_template.return_value = "rendered"

    render_prompt(processor, [], template=None, mode=None, instructions=None)
    call = processor.apply_chat_template.call_args
    assert "template" not in call.kwargs
    assert "mode" not in call.kwargs
    assert "instructions" not in call.kwargs
    assert call.kwargs["enable_thinking"] is False


def test_render_prompt_includes_mode():
    processor = MagicMock()
    processor.apply_chat_template.return_value = ""

    render_prompt(processor, [], mode="markdown")
    assert processor.apply_chat_template.call_args.kwargs["mode"] == "markdown"


def test_render_prompt_includes_instructions_when_set():
    processor = MagicMock()
    processor.apply_chat_template.return_value = ""

    render_prompt(processor, [], instructions="use ISO dates")
    assert (
        processor.apply_chat_template.call_args.kwargs["instructions"]
        == "use ISO dates"
    )


def test_render_prompt_drops_empty_instructions():
    processor = MagicMock()
    processor.apply_chat_template.return_value = ""

    render_prompt(processor, [], instructions="")
    # Empty string is falsy → dropped, model sees no instructions
    assert "instructions" not in processor.apply_chat_template.call_args.kwargs


def test_render_prompt_passes_image_message_through():
    """Image-bearing messages reach apply_chat_template unchanged so the Jinja
    template can insert the vision placeholder."""
    processor = MagicMock()
    processor.apply_chat_template.return_value = ""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/tmp/x.png"},
                {"type": "text", "text": "describe"},
            ],
        }
    ]
    render_prompt(processor, messages, mode="markdown")
    assert processor.apply_chat_template.call_args.args[0] == messages


# --- mode constants ---


def test_mode_constants_have_expected_values():
    """Sanity check the mode strings the Jinja template branches on."""
    assert MODE_STRUCTURED == "structured"
    assert MODE_CONTENT == "content"
    assert MODE_MARKDOWN == "markdown"
    assert MODE_TEMPLATE_GENERATION == "template-generation"


# --- split_reasoning_and_output ---


def test_split_no_reasoning_returns_full_as_output():
    think, output = split_reasoning_and_output("hello", reasoning_enabled=False)
    assert think == ""
    assert output == "hello"


def test_split_with_complete_think_block():
    text = 'thinking about it</think>{"name":"x"}'
    think, output = split_reasoning_and_output(text, reasoning_enabled=True)
    assert think == "thinking about it"
    assert output == '{"name":"x"}'


def test_split_incomplete_think_returns_only_reasoning():
    think, output = split_reasoning_and_output(
        "still thinking...", reasoning_enabled=True
    )
    assert think == "still thinking..."
    assert output == ""


def test_split_case_insensitive_end_tag():
    think, output = split_reasoning_and_output("R</THINK>O", reasoning_enabled=True)
    assert think == "R"
    assert output == "O"


def test_split_empty_text():
    assert split_reasoning_and_output("", reasoning_enabled=True) == ("", "")
    assert split_reasoning_and_output("", reasoning_enabled=False) == ("", "")


# --- extract_answer_block ---


@pytest.mark.parametrize(
    "text,expected",
    [
        pytest.param('<answer>{"k":1}</answer>', '{"k":1}', id="answer_wrapped"),
        pytest.param("<ANSWER>x</ANSWER>", "x", id="answer_case_insensitive"),
        pytest.param(
            'prefix {"a":1} middle {"b":1,"c":2} suffix',
            '{"b":1,"c":2}',
            id="picks_longest_valid_json",
        ),
        pytest.param(
            'Looking at the doc... {"name": "John", "age": 30}',
            '{"name": "John", "age": 30}',
            id="reasoning_prefix_plus_json",
        ),
        pytest.param(
            '{"outer": 2, "inner": {"k": 1}}',
            '{"outer": 2, "inner": {"k": 1}}',
            id="nested_json_as_single_span",
        ),
        pytest.param(
            "{not json {still bad",
            "{not json {still bad",
            id="unparseable_brace_runs",
        ),
        pytest.param("  just text  ", "just text", id="no_match_returns_stripped"),
        pytest.param("", "", id="empty"),
    ],
)
def test_extract_answer_block(text, expected):
    assert extract_answer_block(text) == expected


# --- pretty_json_or_text ---


@pytest.mark.parametrize(
    "text,expected",
    [
        pytest.param(
            '{"a":1,"b":2}', '{\n  "a": 1,\n  "b": 2\n}', id="valid_json_indented"
        ),
        pytest.param(
            '{"name":"élise"}', '{\n  "name": "élise"\n}', id="preserves_unicode"
        ),
        pytest.param("not json {{{", "not json {{{", id="invalid_returns_original"),
        pytest.param("", "", id="empty"),
        pytest.param("   ", "", id="whitespace_only"),
    ],
)
def test_pretty_json_or_text(text, expected):
    assert pretty_json_or_text(text) == expected


# --- load_model integration boundary ---


def test_conftest_guard_blocks_a_forgotten_mock():
    """A loader reached without mocking fails immediately, never downloads.

    Regression test for the guard itself: the bug it replaces made every CI run
    pull 4.8 GB and hung two runs for six hours. If this test starts passing for
    the wrong reason — i.e. the call succeeds — the guard has stopped working.
    """
    import nuextract as _nuextract

    with pytest.raises(AssertionError, match="real model download"):
        _nuextract.snapshot_download(repo_id="numind/NuExtract3-mlx-8bits")
    with pytest.raises(AssertionError, match="real model download"):
        _nuextract.mlx_vlm_load("/some/dir")


def test_load_model_invokes_snapshot_and_load():
    """load_model orchestrates: snapshot_download → mlx_vlm.load."""
    with (
        patch("nuextract.snapshot_download", return_value="/fake/dir") as mock_dl,
        patch("nuextract.mlx_vlm_load", return_value=("M", "P")) as mock_load,
    ):
        model, processor = load_model("test/repo", revision="abc123")

    mock_dl.assert_called_once_with(repo_id="test/repo", revision="abc123")
    mock_load.assert_called_once_with("/fake/dir")
    assert (model, processor) == ("M", "P")


def test_load_model_pins_the_default_revision():
    """With no arguments, load_model fetches the pinned commit, never `main`."""
    with (
        patch("nuextract.snapshot_download", return_value="/fake/dir") as mock_dl,
        patch("nuextract.mlx_vlm_load", return_value=("M", "P")),
    ):
        load_model()

    mock_dl.assert_called_once_with(
        repo_id=DEFAULT_MODEL_ID, revision=DEFAULT_MODEL_REVISION
    )


def test_default_model_revision_is_a_full_commit_sha():
    """A branch or tag ("main", "v1") would satisfy snapshot_download just as
    well and silently reopen the drift the pin exists to close."""
    assert re.fullmatch(r"[0-9a-f]{40}", DEFAULT_MODEL_REVISION)


def test_probe_pins_the_same_model_as_the_app():
    """scripts/probe_mlx_vlm.py keeps its own copies of the model id and
    revision, since it runs as a script and cannot import nuextract. A probe
    loading a different snapshot passes or fails for reasons unrelated to the
    app, so a drifted copy must fail here rather than in a user's hands."""
    probe = Path(__file__).parents[1] / "scripts" / "probe_mlx_vlm.py"
    constants = {}
    for node in ast.parse(probe.read_text()).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target, value = node.targets[0], node.value
        if isinstance(target, ast.Name) and isinstance(value, ast.Constant):
            constants[target.id] = value.value

    assert constants["MODEL_ID"] == DEFAULT_MODEL_ID
    assert constants["MODEL_REVISION"] == DEFAULT_MODEL_REVISION


# --- stream_extract integration boundary ---


def test_stream_extract_yields_accumulated_text():
    """stream_extract should accumulate per-chunk .text deltas."""
    chunks = [MagicMock(text="Hel"), MagicMock(text="lo"), MagicMock(text=" world")]
    processor = MagicMock()
    processor.apply_chat_template.return_value = "PROMPT"

    with patch(
        "nuextract.mlx_vlm_stream_generate", return_value=iter(chunks)
    ) as mock_sg:
        outputs = list(stream_extract(MagicMock(), processor, text="hi"))

    assert outputs == ["Hel", "Hello", "Hello world"]
    # image kwarg is omitted entirely when no image_path is given
    assert "image" not in mock_sg.call_args.kwargs


def test_stream_extract_passes_image_path_as_list():
    chunks = [MagicMock(text="ok")]
    processor = MagicMock()
    processor.apply_chat_template.return_value = "PROMPT"

    with patch(
        "nuextract.mlx_vlm_stream_generate", return_value=iter(chunks)
    ) as mock_sg:
        list(stream_extract(MagicMock(), processor, image_path="/tmp/x.png"))

    assert mock_sg.call_args.kwargs["image"] == ["/tmp/x.png"]


def test_stream_extract_forwards_template_and_generation_kwargs():
    """The UI sliders (temperature, max_tokens) and template editor must reach
    the underlying calls — otherwise the controls become inert."""
    chunks = [MagicMock(text="ok")]
    processor = MagicMock()
    processor.apply_chat_template.return_value = "PROMPT"

    with patch(
        "nuextract.mlx_vlm_stream_generate", return_value=iter(chunks)
    ) as mock_sg:
        list(
            stream_extract(
                MagicMock(),
                processor,
                text="hi",
                template='{"x": "string"}',
                instructions="be brief",
                mode=None,
                enable_thinking=True,
                temperature=0.7,
                max_tokens=512,
            )
        )

    # Template + instructions + thinking flag reach apply_chat_template inline
    tpl_kwargs = processor.apply_chat_template.call_args.kwargs
    assert tpl_kwargs["template"] == '{"x": "string"}'
    assert tpl_kwargs["instructions"] == "be brief"
    assert tpl_kwargs["enable_thinking"] is True

    # Generation kwargs reach mlx_vlm_stream_generate
    gen_kwargs = mock_sg.call_args.kwargs
    assert gen_kwargs["temperature"] == 0.7
    assert gen_kwargs["max_tokens"] == 512


def test_stream_extract_skips_special_tokens():
    """mlx-vlm defaults skip_special_tokens=False. Before 0.7.0 that left
    <|im_end|> in the decoded text — visible in markdown output and the
    downloaded .md; 0.7.0+ stops on it first, so this flag is now a backstop."""
    chunks = [MagicMock(text="ok")]
    processor = MagicMock()
    processor.apply_chat_template.return_value = "PROMPT"

    with patch(
        "nuextract.mlx_vlm_stream_generate", return_value=iter(chunks)
    ) as mock_sg:
        list(stream_extract(MagicMock(), processor, text="hi"))

    assert mock_sg.call_args.kwargs["skip_special_tokens"] is True


def test_stream_extract_forwards_system_prompt():
    """system_prompt is forwarded to build_messages and becomes a system message."""
    chunks = [MagicMock(text="ok")]
    processor = MagicMock()
    processor.apply_chat_template.return_value = "PROMPT"

    with patch("nuextract.mlx_vlm_stream_generate", return_value=iter(chunks)):
        list(
            stream_extract(
                MagicMock(),
                processor,
                text="user content",
                system_prompt="be a JSON expert",
            )
        )

    sent_messages = processor.apply_chat_template.call_args.args[0]
    assert sent_messages[0] == {"role": "system", "content": "be a JSON expert"}
    assert sent_messages[1]["role"] == "user"
