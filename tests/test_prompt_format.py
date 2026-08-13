"""
Golden tests for app.services.llm.render_conversation() -- the Python port
of the training notebook's render() (nb_dump.txt cell 15). These are
static/offline checks; probe_history_parity.py is the live check against
the actual served model that /api/chat's own templating produces the same
tokenization these strings represent.
"""
from app.services.llm import SYSTEM_JOIN, render_conversation


def test_single_turn_matches_modelfile_template_expansion():
    """The shipped Ollama Modelfile TEMPLATE (nb_dump.txt cell 35) for a
    single exchange with a System field expands to:
        <start_of_turn>user\n{System}\n\n{Prompt}<end_of_turn>\n<start_of_turn>model\n
    render_conversation() must produce byte-identical text for the
    equivalent [system, user] messages list.
    """
    messages = [
        {"role": "system", "content": "You are a tutor."},
        {"role": "user", "content": "What is Article 283?"},
    ]
    expected = (
        "<start_of_turn>user\n"
        "You are a tutor.\n\n"
        "What is Article 283?<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    assert render_conversation(messages) == expected


def test_multi_turn_matches_training_notebook_shape():
    """The documented trained shape (docs/architecture: 'the trained format
    is exactly...') for [system, user, assistant, user]:
        <start_of_turn>user\n{SYSTEM}\n\n{USER_1}<end_of_turn>\n
        <start_of_turn>model\n{ASSISTANT_1}<end_of_turn>\n
        <start_of_turn>user\n{USER_2}<end_of_turn>\n
        <start_of_turn>model\n   (generation prompt)
    """
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "U2"},
    ]
    expected = (
        "<start_of_turn>user\nSYS\n\nU1<end_of_turn>\n"
        "<start_of_turn>model\nA1<end_of_turn>\n"
        "<start_of_turn>user\nU2<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    assert render_conversation(messages) == expected


def test_system_joined_into_first_user_turn_only():
    """The system content must appear exactly once, folded into the first
    user turn with SYSTEM_JOIN -- never repeated on later turns (Gemma-2
    has no native system role; this IS the workaround, and re-injecting it
    per turn is not the trained shape)."""
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "U2"},
    ]
    rendered = render_conversation(messages)
    assert rendered.count("SYS") == 1
    assert f"SYS{SYSTEM_JOIN}U1" in rendered


def test_no_literal_bos_token():
    """render_conversation is deliberately WITHOUT a literal <bos> -- the
    GGUF's own tokenizer adds exactly one; a literal <bos> here would
    double it (the training notebook's own documented warning)."""
    messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "U1"}]
    assert "<bos>" not in render_conversation(messages)


def test_add_generation_prompt_false_omits_trailing_model_header():
    messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "U1"}]
    rendered = render_conversation(messages, add_generation_prompt=False)
    assert not rendered.endswith("<start_of_turn>model\n")
    assert rendered.endswith("<end_of_turn>\n")


def test_no_system_message_renders_body_only():
    """A messages list with no system role (e.g. a raw history window
    without its own system turn) renders straight through -- no crash, no
    phantom SYSTEM_JOIN."""
    messages = [{"role": "user", "content": "U1"}, {"role": "assistant", "content": "A1"}]
    rendered = render_conversation(messages, add_generation_prompt=False)
    assert rendered == "<start_of_turn>user\nU1<end_of_turn>\n<start_of_turn>model\nA1<end_of_turn>\n"


def test_content_is_stripped_like_the_notebook():
    messages = [
        {"role": "system", "content": "  SYS  "},
        {"role": "user", "content": "  U1  "},
    ]
    rendered = render_conversation(messages, add_generation_prompt=False)
    assert rendered == "<start_of_turn>user\nSYS\n\nU1<end_of_turn>\n"
