from sutra.core.compaction import CompactionConfig, ContextCompactor
from sutra.core.state import Message, Role


async def test_compaction_preserves_last_n_turns_and_shrinks_window():
    compactor = ContextCompactor(CompactionConfig(token_threshold=20, preserve_last_n_turns=2))
    messages = [Message(role=Role.USER, content="x" * 200) for _ in range(10)]
    assert compactor.needs_compaction(messages)

    compacted = await compactor.compact(messages)

    assert compacted[-2:] == messages[-2:]
    assert compacted[0].role == Role.SUMMARY
    assert len(compacted) == 3


async def test_no_compaction_below_threshold():
    compactor = ContextCompactor(CompactionConfig(token_threshold=100_000, preserve_last_n_turns=2))
    messages = [Message(role=Role.USER, content="hello")]
    assert not compactor.needs_compaction(messages)
    result = await compactor.compact(messages)
    assert result == messages


async def test_repeated_compaction_folds_previous_summary():
    compactor = ContextCompactor(CompactionConfig(token_threshold=20, preserve_last_n_turns=1))
    messages = [Message(role=Role.USER, content="x" * 200) for _ in range(6)]
    first_pass = await compactor.compact(messages)
    assert first_pass[0].role == Role.SUMMARY

    more_messages = first_pass + [Message(role=Role.USER, content="y" * 200) for _ in range(4)]
    second_pass = await compactor.compact(more_messages)
    assert second_pass[0].role == Role.SUMMARY
    assert second_pass[0].content.count("Summary of earlier turns:") >= 1
