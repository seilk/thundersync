"""DataParallelThunderSync methods that need no process group.

The final-forest configuration reads only the direct-path flag, and the
event mirror only the backward it forwards to, so the trainer is built
without its constructor, which needs a process group and a CUDA replica.
"""

from __future__ import annotations

import pytest

from thundersync.grpo.data_parallel import DataParallelThunderSync


def _trainer(*, direct_when_group_bound: bool) -> DataParallelThunderSync:
    trainer = DataParallelThunderSync.__new__(DataParallelThunderSync)
    trainer.direct_when_group_bound = direct_when_group_bound
    trainer._final_forest = None
    return trainer


def test_the_final_forest_is_configured_on_the_direct_path() -> None:
    trainer = _trainer(direct_when_group_bound=True)
    assert not trainer.final_forest_enabled
    with pytest.raises(RuntimeError, match="not configured"):
        trainer.final_forest_max_tokens

    trainer.configure_final_forest(max_tokens_per_microbatch=4096, chunk=256)

    assert trainer.final_forest_enabled
    assert trainer.final_forest_max_tokens == 4096


@pytest.mark.parametrize(
    ("max_tokens", "chunk"), [(0, 256), (4096, 0)]
)
def test_a_nonpositive_token_cap_or_chunk_is_refused(max_tokens: int, chunk: int) -> None:
    trainer = _trainer(direct_when_group_bound=True)
    with pytest.raises(ValueError, match="positive"):
        trainer.configure_final_forest(max_tokens_per_microbatch=max_tokens, chunk=chunk)
    assert not trainer.final_forest_enabled


def test_the_final_forest_needs_the_direct_path() -> None:
    trainer = _trainer(direct_when_group_bound=False)
    with pytest.raises(ValueError, match="direct_when_group_bound"):
        trainer.configure_final_forest(max_tokens_per_microbatch=4096, chunk=256)
    assert not trainer.final_forest_enabled


class _FailingBackward:
    def __init__(self, abort_error: BaseException | None) -> None:
        self.abort_error = abort_error
        self.aborted = False

    def close_group(self, group_id: int) -> None:
        raise KeyError(group_id)

    def abort(self) -> None:
        self.aborted = True
        if self.abort_error is not None:
            raise self.abort_error


def test_a_failed_event_aborts_the_backward_and_reraises() -> None:
    trainer = _trainer(direct_when_group_bound=False)
    trainer.bw = _FailingBackward(abort_error=None)
    trainer._local_groups_closed = 0

    with pytest.raises(KeyError) as raised:
        trainer.close_group(3)

    assert trainer.bw.aborted
    assert not getattr(raised.value, "__notes__", [])
    assert trainer._local_groups_closed == 0


def test_a_failed_abort_is_noted_on_the_original_error() -> None:
    trainer = _trainer(direct_when_group_bound=False)
    trainer.bw = _FailingBackward(abort_error=OSError("abort failed"))
    trainer._local_groups_closed = 0

    with pytest.raises(KeyError) as raised:
        trainer.close_group(3)

    assert any("abort failed" in note for note in raised.value.__notes__)
