from conversations import add_message, create_conversation, get_conversation_history


def test_conversation_history_is_chronological_and_limited(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    create_conversation("conv-1", title="Test")
    add_message("conv-1", "user", "first")
    add_message("conv-1", "assistant", "second", context="source chunk")
    add_message("conv-1", "user", "third")

    history = get_conversation_history("conv-1", limit=2)

    assert [message["content"] for message in history] == ["second", "third"]
    assert history[0]["context"] == "source chunk"
