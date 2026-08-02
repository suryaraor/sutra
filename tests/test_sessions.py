from sutra.core.sessions import FileSessionStore
from sutra.core.state import HarnessState, HarnessStatus, Message, Role


def _make_state(session_id: str = None) -> HarnessState:
    kwargs = {"system_prompt": "test system prompt"}
    if session_id is not None:
        kwargs["session_id"] = session_id
    state = HarnessState(**kwargs)
    state.status = HarnessStatus.COMPLETED
    state.append_message(Message(role=Role.USER, content="hello there"))
    state.append_message(Message(role=Role.ASSISTANT, content="hi, how can I help?"))
    return state


def test_save_load_roundtrip_preserves_messages_status_and_session_id(tmp_path):
    store = FileSessionStore(root=tmp_path / ".sutra_sessions")
    state = _make_state()

    store.save(state)
    reloaded = store.load(state.session_id)

    assert reloaded is not None
    assert reloaded.session_id == state.session_id
    assert reloaded.status == HarnessStatus.COMPLETED
    assert len(reloaded.messages) == 2
    assert reloaded.messages[0].content == "hello there"
    assert reloaded.messages[0].role == Role.USER
    assert reloaded.messages[1].content == "hi, how can I help?"


def test_save_creates_parent_directories(tmp_path):
    root = tmp_path / "nested" / ".sutra_sessions"
    store = FileSessionStore(root=root)
    state = _make_state()

    store.save(state)

    assert store.session_path(state.session_id).exists()


def test_load_missing_session_returns_none(tmp_path):
    store = FileSessionStore(root=tmp_path / ".sutra_sessions")
    assert store.load("does-not-exist") is None


def test_list_sessions_empty_when_root_missing(tmp_path):
    store = FileSessionStore(root=tmp_path / ".sutra_sessions")
    assert store.list_sessions() == []


def test_list_sessions_returns_newest_first(tmp_path):
    store = FileSessionStore(root=tmp_path / ".sutra_sessions")

    state_a = _make_state(session_id="session-a")
    store.save(state_a)

    state_b = _make_state(session_id="session-b")
    store.save(state_b)

    # Force distinct mtimes: bump session-a's file after session-b's write so
    # ordering is unambiguous regardless of filesystem timestamp resolution.
    path_a = store.session_path("session-a")
    path_b = store.session_path("session-b")
    mtime_b = path_b.stat().st_mtime
    import os

    os.utime(path_a, (mtime_b + 5, mtime_b + 5))

    sessions = store.list_sessions()
    assert sessions[0] == "session-a"
    assert "session-b" in sessions
    assert len(sessions) == 2


def test_save_overwrites_existing_session_with_same_id(tmp_path):
    store = FileSessionStore(root=tmp_path / ".sutra_sessions")
    state = _make_state(session_id="fixed-id")
    store.save(state)

    state.append_message(Message(role=Role.USER, content="one more message"))
    store.save(state)

    reloaded = store.load("fixed-id")
    assert reloaded is not None
    assert len(reloaded.messages) == 3
    assert len(store.list_sessions()) == 1
