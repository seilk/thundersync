"""The asynchronous rollout schedule, against a stub environment.

Three barriers are present in ``schedule="lockstep"`` and absent in
``schedule="async"``: the round barrier, the group grading barrier, and
sequential group execution. This file measures what the first two cost on a
controlled workload, checks the third through :meth:`rollout_groups`, and pins
the invariant that makes the change admissible at all -- scheduling decides WHEN
work runs and never WHAT is sampled.

No containers and no GPU. ``BatchedRolloutEngine.__init__`` constructs vLLM, so
:class:`StubEngine` subclasses the engine and replaces ``__init__``, ``encode``,
``decode`` and ``generate`` only. Everything under test -- the turn loops, the
generation broker, the grading points, the callback contract -- is the real
implementation. Sandboxes and the grader are injected through the
``sandbox_factory`` / ``grade_fn`` hooks, whose defaults are the real
DockerSandbox and run_tests.

Timing assertions use latencies in the tens to hundreds of milliseconds and are
stated as separations rather than absolute values wherever a loaded machine
could move them. The separation being asserted is a factor of roughly 3 on the
chosen workload, far outside scheduler noise at this scale.
"""

from __future__ import annotations

import itertools
import shutil
import subprocess
import threading
import time

import pytest


from thundersync.rollout.agent import SYSTEM_PROMPT, TASK_TEMPLATE, Task  # noqa: E402
import thundersync.rollout.engine as rollout_engine  # noqa: E402
from thundersync.rollout.engine import (  # noqa: E402
    BatchedRolloutEngine,
    CommittedBlock,
    GenResult,
    PolicyPatchRejected,
    _GenerationBroker,
    _capture_bug_base,
    _capture_policy_patch,
    _capture_trusted_bug_tree,
    _is_oracle_controlled,
    _parse_junit,
    _pytest_selector_candidates,
    _run_go_test_json,
    _run_pytest_junit,
    _selector_batches,
    _source_diff,
    grade,
    request_seed,
    grade_isolated,
    new_sandbox,
    parse_test_ids,
)

TURN_OPEN = "<|im_start|>assistant\n"


# ----------------------------------------------------------------------
# stub environment
# ----------------------------------------------------------------------


class StubEngine(BatchedRolloutEngine):
    """The real engine with the model and tokenizer removed.

    ``encode``/``decode`` are an exact ASCII codec, so a prompt can be read back
    from its token ids and ``generate`` can be a pure function of
    ``(prompt_token_ids, seed)``. Purity is the point: it is what lets a test
    distinguish "the schedule changed which tokens were produced" from "the
    schedule changed when they were produced".
    """

    def __init__(self, *, max_model_len: int = 20000, gen_s: float = 0.001,
                 n_turns: int = 4):
        # deliberately does not call super().__init__, which builds a vLLM engine
        self.max_model_len = max_model_len
        self.lora_request = None
        self.gen_s = gen_s
        self.n_turns = n_turns
        self.batch_sizes: list[int] = []
        self.n_prompts = 0
        self._lock = threading.Lock()
        self.fail_after: int | None = None

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids) -> str:
        return "".join(chr(i) for i in ids)

    def generate(self, prompts, *, max_tokens=1024, temperature=1.0, top_p=1.0,
                 seeds=None):
        with self._lock:
            self.batch_sizes.append(len(prompts))
            self.n_prompts += len(prompts)
            if self.fail_after is not None and self.n_prompts > self.fail_after:
                raise RuntimeError("stub engine failure")
        if self.gen_s:
            time.sleep(self.gen_s)  # one sleep per call: generation is batched
        out = []
        for k, p in enumerate(prompts):
            seed = 0 if seeds is None else int(seeds[k])
            text = self.decode(p)
            # the opener for the turn being generated is already appended, so
            # the first call of a rollout counts 1
            turn = text.count(TURN_OPEN)
            body = "submit" if turn >= self.n_turns else f"cmd-{seed}-{turn}"
            gen_text = f"```bash\n{body}\n```"
            ids = self.encode(gen_text)
            out.append(
                GenResult(
                    token_ids=ids,
                    logprobs=[round(-0.01 * ((seed + j) % 11), 6)
                              for j in range(len(ids))],
                    finish_reason="stop",
                    text=gen_text,
                )
            )
        return out


class StubSandbox:
    """Per-rollout sandbox with scripted per-call latency."""

    def __init__(self, image: str, name: str, group_id: int, rollout_id: int,
                 latency: dict, grade_s: float):
        self.image, self.name = image, name
        self.group_id, self.rollout_id = group_id, rollout_id
        self.latency = latency
        self.grade_s = grade_s
        self.calls = 0
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def exec(self, command: str) -> str:
        if " git diff " in f" {command} ":
            # the pre-grading diff capture; not part of the agent's turn chain
            return "M src/a.py | 2 +-"
        if " git ls-files --others" in f" {command}":
            return ""
        lat = self.latency.get(self.rollout_id, [])
        d = lat[self.calls] if self.calls < len(lat) else 0.0
        self.calls += 1
        if d:
            time.sleep(d)
        return f"out:{command}"

    def close(self) -> None:
        self.closed = True


def make_env(latency: dict, *, grade_s: float = 0.05):
    """Return (sandbox_factory, grade_fn, registry-of-boxes)."""
    boxes: dict[tuple[int, int], StubSandbox] = {}

    def factory(image: str, name: str) -> StubSandbox:
        parts = name.rsplit("-", 2)
        gid, rid = int(parts[1]), int(parts[2])
        box = StubSandbox(image, name, gid, rid, latency, grade_s)
        boxes[(gid, rid)] = box
        return box

    def grade_fn(box: StubSandbox, task: Task) -> float:
        if box.grade_s:
            time.sleep(box.grade_s)
        return float(box.rollout_id % 2)  # deterministic, and never all-equal

    return factory, grade_fn, boxes


TASK = Task(
    instance_id="stub-task",
    problem_statement="a stub problem statement",
    image_name="stub:image",
)


LINGER_S = 0.005
"""Broker coalescing window used throughout. Small against the 0.02s floor of
the scripted tool latencies, so it never carries a makespan assertion, and large
enough that concurrently started workers land in the same generate call."""


def run(schedule: str, *, latency: dict, group_size: int = 4, grade_s: float = 0.05,
        engine: StubEngine | None = None, linger_s: float = LINGER_S, **kw):
    eng = engine or StubEngine()
    factory, grade_fn, boxes = make_env(latency, grade_s=grade_s)
    kw.setdefault("max_steps", 20)
    kw.setdefault("max_tokens_per_step", 64)
    if schedule == "async":
        kw["gen_linger_s"] = linger_s
    trajs, timings = eng.rollout_group(
        TASK,
        schedule=schedule,
        group_size=group_size,
        repo_context=False,
        verify_broken=False,
        sandbox_factory=factory,
        grade_fn=grade_fn,
        **kw,
    )
    return eng, trajs, timings, boxes


# Heterogeneous by construction: each of the first three rollouts is the slow
# one in a different round, so no round is cheap and no single rollout is the
# whole critical path. This is the shape a round barrier is worst on.
HETERO = {
    0: [0.40, 0.02, 0.02],
    1: [0.02, 0.40, 0.02],
    2: [0.02, 0.02, 0.40],
    3: [0.02, 0.02, 0.02],
}
GRADE_S = 0.05


def chain_makespan(latency: dict) -> float:
    return max(sum(v) for v in latency.values())


def round_barrier_makespan(latency: dict) -> float:
    n = max(len(v) for v in latency.values())
    return sum(
        max(v[k] for v in latency.values() if k < len(v)) for k in range(n)
    )


# ----------------------------------------------------------------------
# (a) makespan
# ----------------------------------------------------------------------


def test_async_makespan_beats_lockstep_under_heterogeneous_tool_latency():
    _, _, lock_t, _ = run("lockstep", latency=HETERO, grade_s=GRADE_S)
    _, _, async_t, _ = run("async", latency=HETERO, grade_s=GRADE_S)

    assert async_t.rollout_wall_s < lock_t.rollout_wall_s
    # the workload's own ratio is chain/round-sum = 0.44/1.20 = 0.37; assert a
    # separation well inside that so the test reports the barrier, not jitter
    assert async_t.rollout_wall_s < 0.65 * lock_t.rollout_wall_s, (
        f"async {async_t.rollout_wall_s:.3f}s vs lockstep "
        f"{lock_t.rollout_wall_s:.3f}s"
    )


def test_async_tracks_max_rollout_chain_and_lockstep_tracks_sum_of_round_maxima():
    """The makespans are not merely ordered; each sits on its own predicted floor."""
    _, _, lock_t, _ = run("lockstep", latency=HETERO, grade_s=GRADE_S)
    _, _, async_t, _ = run("async", latency=HETERO, grade_s=GRADE_S)

    # async: the longest single rollout chain, plus that rollout's own grading
    want_async = chain_makespan(HETERO) + GRADE_S          # 0.44 + 0.05
    # lockstep: every round pays its slowest sandbox, then one grading barrier
    want_lock = round_barrier_makespan(HETERO) + GRADE_S   # 1.20 + 0.05

    assert want_async <= async_t.rollout_wall_s <= want_async * 1.5 + 0.15, (
        f"async makespan {async_t.rollout_wall_s:.3f}s off predicted "
        f"{want_async:.3f}s"
    )
    assert want_lock * 0.9 <= lock_t.rollout_wall_s <= want_lock * 1.4 + 0.15, (
        f"lockstep makespan {lock_t.rollout_wall_s:.3f}s off predicted "
        f"{want_lock:.3f}s"
    )
    # and the async makespan is nowhere near the sum of round maxima
    assert async_t.rollout_wall_s < 0.7 * want_lock


def test_generation_stays_batched_after_the_round_barrier_is_removed():
    """Removing the barrier must not serialize generation one sequence at a time.

    The broker coalesces outstanding requests, so the mean batch size stays
    above one even though rollouts no longer arrive together.
    """
    eng, _, timings, _ = run("async", latency=HETERO, grade_s=0.0)
    assert eng.batch_sizes, "no generate calls were issued"
    assert max(eng.batch_sizes) > 1, (
        f"every generate call held one prompt: {eng.batch_sizes}"
    )
    # total work is conserved: same number of model rows as lockstep
    _, _, lock_t, _ = run("lockstep", latency=HETERO, grade_s=0.0)
    assert timings.model_rows == lock_t.model_rows
    assert timings.logical_rows == lock_t.logical_rows
    assert timings.tool_calls == lock_t.tool_calls


# ----------------------------------------------------------------------
# (b) callback contract
# ----------------------------------------------------------------------


def collect(schedule: str, *, latency=HETERO, grade_s=GRADE_S, group_size=4):
    """Run one group, recording every callback with a global sequence number."""
    seq = itertools.count()
    log: list[tuple] = []
    prompts: list[list[int]] = []
    lock = threading.Lock()

    def on_prompt(ids):
        with lock:
            log.append((next(seq), "prompt", None, None))
            prompts.append(list(ids))

    def on_step(step_idx, events):
        with lock:
            n = next(seq)
            for rid, st in events:
                log.append((n, "step", rid, (step_idx, st)))

    def on_block(events):
        with lock:
            n = next(seq)
            for rid, block in events:
                log.append((n, "block", rid, block))

    def on_trajectory_complete(rid, traj):
        with lock:
            log.append((next(seq), "complete", rid, traj.finish_reason))

    def on_verdict(rid, traj):
        with lock:
            log.append((next(seq), "verdict", rid, traj.reward))

    eng = StubEngine()
    factory, grade_fn, boxes = make_env(latency, grade_s=grade_s)
    kw = dict(
        group_size=group_size, repo_context=False, verify_broken=False,
        max_steps=20, max_tokens_per_step=64,
        sandbox_factory=factory, grade_fn=grade_fn,
        on_prompt=on_prompt, on_block=on_block, on_step=on_step,
        on_trajectory_complete=on_trajectory_complete,
        on_verdict=on_verdict,
    )
    if schedule == "async":
        kw["gen_linger_s"] = LINGER_S
    trajs, timings = eng.rollout_group(TASK, schedule=schedule, **kw)
    return trajs, timings, log, prompts


def per_trajectory(log):
    """Callback stream restricted to one trajectory, in arrival order.

    Timing fields are excluded on purpose. Lockstep divides a round's generation
    and tool wall-clock across the rollouts of that round; async records each
    rollout's own. Those numbers are expected to differ and are not part of the
    contract. Everything the trainer consumes -- token ids, logprobs, prompt
    length, action and observation text, step index -- is compared.
    """
    out: dict[int, list] = {}
    for _, kind, rid, payload in log:
        if kind == "step":
            step_idx, st = payload
            out.setdefault(rid, []).append(
                (
                    "step", step_idx,
                    tuple(st.action_token_ids), tuple(st.observation_token_ids),
                    tuple(st.action_logprobs), st.prompt_len,
                    st.action, st.observation,
                )
            )
        elif kind == "verdict":
            out.setdefault(rid, []).append(("verdict", payload))
        elif kind == "complete":
            out.setdefault(rid, []).append(("complete", payload))
    return out


def test_callback_sequence_per_trajectory_is_identical_under_both_schedules():
    _, _, lock_log, lock_prompts = collect("lockstep")
    _, _, async_log, async_prompts = collect("async")

    assert lock_prompts == async_prompts
    assert len(lock_prompts) == 1, "on_prompt must fire exactly once per group"

    lock_per = per_trajectory(lock_log)
    async_per = per_trajectory(async_log)
    assert set(lock_per) == set(async_per) == {0, 1, 2, 3}
    for rid in sorted(lock_per):
        assert async_per[rid] == lock_per[rid], f"rollout {rid} callback stream differs"


def test_on_step_event_shape_is_round_under_lockstep_and_per_rollout_under_async():
    """The argument shape is unchanged; what changes is how many events a call
    carries, because a step completing is a per-rollout event once the round
    barrier is gone."""
    _, _, lock_log, _ = collect("lockstep")
    _, _, async_log, _ = collect("async")

    lock_calls: dict[int, int] = {}
    for n, kind, rid, _ in lock_log:
        if kind == "step":
            lock_calls[n] = lock_calls.get(n, 0) + 1
    assert max(lock_calls.values()) == 4, (
        "lockstep should emit all four rollouts of a round in one on_step call"
    )

    async_calls: dict[int, int] = {}
    for n, kind, rid, _ in async_log:
        if kind == "step":
            async_calls[n] = async_calls.get(n, 0) + 1
    assert set(async_calls.values()) == {1}, (
        "async should emit one rollout per on_step call"
    )


def test_committed_blocks_are_token_exact_and_schedule_independent():
    lock_trajs, _, lock_log, _ = collect("lockstep")
    async_trajs, _, async_log, _ = collect("async")

    def blocks_by_trajectory(log):
        result = {rid: [] for rid in range(4)}
        for _sequence, kind, rid, payload in log:
            if kind == "block":
                result[rid].append(payload)
        return result

    lock_blocks = blocks_by_trajectory(lock_log)
    async_blocks = blocks_by_trajectory(async_log)
    for lock_traj, async_traj in zip(lock_trajs, async_trajs):
        rid = lock_traj.rollout_id
        semantic = lambda block: (
            block.step_index,
            block.kind,
            block.token_ids,
            block.scored,
            block.terminal_reason_at_commit,
        )
        assert [semantic(block) for block in lock_blocks[rid]] == [
            semantic(block) for block in async_blocks[rid]
        ]
        blocks = lock_blocks[rid]
        assert all(isinstance(block, CommittedBlock) for block in blocks)
        assert lock_traj.completed_at is not None
        assert async_traj.completed_at is not None
        assert all(block.committed_at <= lock_traj.completed_at for block in blocks)
        assert all(
            earlier.committed_at <= later.committed_at
            for earlier, later in zip(blocks, blocks[1:])
        )
        assert [block.kind for block in blocks] == [
            kind
            for step in lock_traj.steps
            for kind in (
                ("action",)
                if not step.observation_token_ids
                else ("action", "observation")
            )
        ]
        assert [token for block in blocks for token in block.token_ids] == (
            lock_traj.token_ids[len(lock_traj.prompt_token_ids) :]
        )

        expected_score_mask = []
        opener_tokens = len(StubEngine().encode(TURN_OPEN))
        for step in lock_traj.steps:
            expected_score_mask.extend([False] * opener_tokens)
            expected_score_mask.extend(
                [True] * (len(step.action_token_ids) - opener_tokens)
            )
            expected_score_mask.extend(
                [False] * len(step.observation_token_ids)
            )
        assert [value for block in blocks for value in block.scored] == (
            expected_score_mask
        )
        terminal_blocks = [
            block
            for block in blocks
            if block.terminal_reason_at_commit is not None
        ]
        assert len(terminal_blocks) == 1
        assert terminal_blocks[0].kind == "action"
        assert terminal_blocks[0].terminal_reason_at_commit == "submit"

        event_order = [
            (sequence, kind, payload)
            for sequence, kind, event_rid, payload in lock_log
            if event_rid == rid and kind in ("block", "step", "complete")
        ]
        for step_index, step in enumerate(lock_traj.steps):
            action_sequence = next(
                sequence
                for sequence, kind, payload in event_order
                if kind == "block"
                and payload.kind == "action"
                and payload.step_index == step_index
            )
            step_sequence = next(
                sequence
                for sequence, kind, payload in event_order
                if kind == "step" and payload[0] == step_index
            )
            assert action_sequence < step_sequence
            if step.observation_token_ids:
                observation_sequence = next(
                    sequence
                    for sequence, kind, payload in event_order
                    if kind == "block"
                    and payload.kind == "observation"
                    and payload.step_index == step_index
                )
                assert action_sequence < observation_sequence < step_sequence

    first_lockstep_observation = {
        rollout_id: next(
            block.committed_at
            for block in blocks
            if block.kind == "observation"
        )
        for rollout_id, blocks in lock_blocks.items()
    }
    assert (
        first_lockstep_observation[3] + 0.25
        < first_lockstep_observation[0]
    ), "observation timestamps must preserve per-tool source readiness"


def test_trainer_callbacks_are_serialized_under_async():
    """A trainer owning CUDA state must never be entered from two threads."""
    inside = [0]
    overlaps = [0]
    lock = threading.Lock()

    def enter():
        with lock:
            inside[0] += 1
            if inside[0] > 1:
                overlaps[0] += 1

    def leave():
        with lock:
            inside[0] -= 1

    def on_step(step_idx, events):
        enter()
        time.sleep(0.002)
        leave()

    def on_verdict(rid, traj):
        enter()
        time.sleep(0.002)
        leave()

    eng = StubEngine()
    factory, grade_fn, _ = make_env(HETERO, grade_s=0.01)
    eng.rollout_group(
        TASK, schedule="async", group_size=4, repo_context=False,
        verify_broken=False, max_steps=20, max_tokens_per_step=64,
        sandbox_factory=factory, grade_fn=grade_fn, gen_linger_s=LINGER_S,
        on_step=on_step, on_verdict=on_verdict,
    )
    assert overlaps[0] == 0, f"{overlaps[0]} concurrent trainer callback entries"


# ----------------------------------------------------------------------
# token exactness
# ----------------------------------------------------------------------


def test_trajectories_are_token_identical_across_schedules():
    """Scheduling changes when work runs; it must not change what was sampled."""
    _, lock_trajs, _, _ = run("lockstep", latency=HETERO, grade_s=0.0)
    _, async_trajs, _, _ = run("async", latency=HETERO, grade_s=0.0)

    assert len(lock_trajs) == len(async_trajs) == 4
    for a, b in zip(lock_trajs, async_trajs):
        assert a.rollout_id == b.rollout_id
        assert a.prompt_token_ids == b.prompt_token_ids
        assert a.token_ids == b.token_ids
        assert a.action_mask == b.action_mask
        assert a.logprobs == b.logprobs
        assert a.finish_reason == b.finish_reason
        assert a.reward == b.reward
        assert [s.action for s in a.steps] == [s.action for s in b.steps]
        assert [s.observation for s in a.steps] == [s.observation for s in b.steps]
        assert [s.prompt_len for s in a.steps] == [s.prompt_len for s in b.steps]


def test_batch_composition_does_not_change_tokens():
    """The property the whole async path rests on.

    A request's completion is a function of ``(prompt_token_ids, seed)``. The
    broker is free to put any set of outstanding requests in one engine call
    precisely because that is true.
    """
    eng = StubEngine(gen_s=0.0)
    prompts = [eng.encode(f"<|im_start|>user\np{i}<|im_end|>\n" + TURN_OPEN)
               for i in range(4)]
    seeds = [11, 22, 33, 44]

    # one call holding all four
    together = eng.generate(prompts, max_tokens=64, temperature=1.0, seeds=seeds)
    # four calls holding one each
    apart = [eng.generate([p], max_tokens=64, temperature=1.0, seeds=[s])[0]
             for p, s in zip(prompts, seeds)]
    assert [g.token_ids for g in together] == [g.token_ids for g in apart]

    # and through the broker, where composition is decided by arrival timing
    with _GenerationBroker(eng, linger_s=0.005) as brk:
        got: dict[int, list[int]] = {}
        ready_at: dict[int, float] = {}
        errs: list[BaseException] = []

        def submit(k):
            try:
                result = brk.submit(
                    prompts[k], seed=seeds[k], max_tokens=64, temperature=1.0
                )
                got[k] = result.token_ids
                assert result.ready_at is not None
                ready_at[k] = result.ready_at
            except BaseException as exc:  # noqa: BLE001
                errs.append(exc)

        threads = [threading.Thread(target=submit, args=(k,)) for k in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert not errs
    assert [got[k] for k in range(4)] == [g.token_ids for g in apart]
    assert all(value > 0.0 for value in ready_at.values())
    assert len(set(ready_at.values())) <= len(brk.batch_sizes)


# ----------------------------------------------------------------------
# (c) the grading barrier, and the lockstep control
# ----------------------------------------------------------------------


def test_verdicts_are_spread_under_async_and_simultaneous_under_lockstep():
    _, _, lock_t, _ = run("lockstep", latency=HETERO, grade_s=GRADE_S)
    _, _, async_t, _ = run("async", latency=HETERO, grade_s=GRADE_S)

    assert len(lock_t.verdict_at) == len(async_t.verdict_at) == 4
    assert len(set(lock_t.verdict_at)) == 1, "lockstep grades at one barrier"
    assert lock_t.graded_at == max(lock_t.verdict_at)
    assert async_t.graded_at == max(async_t.verdict_at)

    spread = max(async_t.verdict_at) - min(async_t.verdict_at)
    # rollout 3's chain is 0.06s, the others' are 0.44s; its verdict must land
    # roughly that difference earlier rather than at the group barrier
    assert spread > 0.25, f"async verdict spread only {spread:.3f}s"


def test_trajectory_completion_precedes_verification_callback():
    """Reward-independent objectives receive the earliest complete event."""
    _, timings, log, _ = collect("async")
    assert len(timings.trajectory_completed_at) == 4
    for rid in range(4):
        kinds = [kind for _seq, kind, got_rid, _payload in log if got_rid == rid]
        assert kinds[-2:] == ["complete", "verdict"]
        assert timings.trajectory_completed_at[rid] <= timings.verdict_at[rid]


def test_first_verdict_fires_while_other_rollouts_are_still_taking_turns():
    """The group grading barrier is what this removes, stated as an ordering."""
    _, _, async_log, _ = collect("async")
    first_verdict = next(n for n, kind, _, _ in async_log if kind == "verdict")
    later_steps = [
        rid for n, kind, rid, _ in async_log if kind == "step" and n > first_verdict
    ]
    assert later_steps, "no rollout was still taking turns at the first verdict"

    _, _, lock_log, _ = collect("lockstep")
    first_lock_verdict = next(n for n, kind, _, _ in lock_log if kind == "verdict")
    assert not [
        1 for n, kind, _, _ in lock_log if kind == "step" and n > first_lock_verdict
    ], "lockstep must not emit steps after its first verdict"


def test_lockstep_path_is_unchanged():
    """The control schedule, checked on the properties that define it."""
    import inspect

    sig = inspect.signature(BatchedRolloutEngine.rollout_group)
    assert sig.parameters["schedule"].default == "lockstep", (
        "existing callers must keep the lockstep behaviour without editing"
    )

    _, trajs, t, boxes = run("lockstep", latency=HETERO, grade_s=GRADE_S)

    assert t.schedule == "lockstep"
    # round barrier intact: n_steps counts rounds, and the wall-clock is the
    # sum of per-round maxima rather than the longest chain
    assert t.n_steps == 4
    assert t.rollout_wall_s > 0.9 * (round_barrier_makespan(HETERO) + GRADE_S)
    # group grading barrier intact
    assert len(set(t.verdict_at)) == 1
    # phases do not overlap on this path, so the parts still add up to the whole
    assert t.generation_s + t.tool_s + t.verify_s <= t.rollout_wall_s + 0.05
    # and every sandbox was closed
    assert len(boxes) == 4 and all(b.closed for b in boxes.values())
    assert [tr.finish_reason for tr in trajs] == ["submit"] * 4


def test_async_reports_overlapping_sums_and_a_separate_makespan():
    """The async timing fields change in kind, and the docstring says so."""
    _, _, t, _ = run("async", latency=HETERO, grade_s=GRADE_S)
    assert t.schedule == "async"
    # tool_s is now a sum over rollouts of intervals that overlapped in time,
    # so it exceeds the makespan it was measured inside
    assert t.tool_s > t.rollout_wall_s
    assert t.rollout_wall_s > 0


# ----------------------------------------------------------------------
# concurrent groups
# ----------------------------------------------------------------------


def test_concurrent_groups_beat_sequential_groups():
    specs = [
        {"task": TASK, "group_id": g}
        for g in range(3)
    ]
    common = dict(
        group_size=4, repo_context=False, verify_broken=False,
        max_steps=20, max_tokens_per_step=64,
    )

    eng_a = StubEngine()
    fa, ga, _ = make_env(HETERO, grade_s=GRADE_S)
    t0 = time.perf_counter()
    res_a = eng_a.rollout_groups(
        specs, schedule="async", gen_linger_s=0.010,
        sandbox_factory=fa, grade_fn=ga, **common
    )
    wall_async = time.perf_counter() - t0

    eng_l = StubEngine()
    fl, gl, _ = make_env(HETERO, grade_s=GRADE_S)
    t0 = time.perf_counter()
    res_l = eng_l.rollout_groups(
        specs, schedule="lockstep", sandbox_factory=fl, grade_fn=gl, **common
    )
    wall_lock = time.perf_counter() - t0

    assert len(res_a) == len(res_l) == 3
    assert [t.group_id for trajs, _ in res_a for t in trajs] == (
        [0] * 4 + [1] * 4 + [2] * 4
    ), "results must come back in specs order"
    assert wall_async < 0.5 * wall_lock, (
        f"concurrent groups {wall_async:.3f}s vs sequential {wall_lock:.3f}s"
    )
    # three concurrent groups feed one broker, so calls hold more than one group
    assert max(eng_a.batch_sizes) > 4, (
        f"cross-group batching did not happen: {eng_a.batch_sizes}"
    )
    for (trajs_a, _), (trajs_l, _) in zip(res_a, res_l):
        assert [t.token_ids for t in trajs_a] == [t.token_ids for t in trajs_l]


def test_concurrent_groups_share_one_verifier_admission_limit():
    active = 0
    maximum_active = 0
    lock = threading.Lock()

    def grade_fn(_box, _task):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return 1.0

    engine = StubEngine(n_turns=1)
    factory, _unused_grade, _boxes = make_env({}, grade_s=0.0)
    engine.rollout_groups(
        [{"task": TASK, "group_id": group_id} for group_id in (0, 1)],
        schedule="async",
        max_concurrent_verifiers=1,
        group_size=2,
        repo_context=False,
        verify_broken=False,
        max_steps=1,
        max_tokens_per_step=64,
        sandbox_factory=factory,
        grade_fn=grade_fn,
    )
    assert maximum_active == 1


def test_rollout_groups_binds_each_groups_callbacks_to_its_own_group():
    seen: dict[int, list[int]] = {0: [], 1: []}

    def make_cb(gid):
        def on_verdict(rid, traj):
            seen[gid].append(traj.group_id)
        return on_verdict

    eng = StubEngine()
    f, g, _ = make_env(HETERO, grade_s=0.0)
    eng.rollout_groups(
        [{"task": TASK, "group_id": gid, "on_verdict": make_cb(gid)} for gid in (0, 1)],
        schedule="async", gen_linger_s=LINGER_S,
        group_size=4, repo_context=False, verify_broken=False,
        max_steps=20, max_tokens_per_step=64,
        sandbox_factory=f, grade_fn=g,
    )
    assert sorted(seen[0]) == [0] * 4
    assert sorted(seen[1]) == [1] * 4


def test_rollout_groups_serializes_trainer_callbacks_across_groups():
    """One generation batch may share one CUDA trainer across all groups."""
    active = 0
    max_active = 0
    state_lock = threading.Lock()

    def callback(*_args):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.002)
        with state_lock:
            active -= 1

    specs = [
        {
            "task": TASK,
            "group_id": gid,
            "on_prompt": callback,
            "on_step": callback,
            "on_verdict": callback,
        }
        for gid in (0, 1, 2)
    ]
    eng = StubEngine()
    factory, grader, _ = make_env(HETERO, grade_s=0.0)
    eng.rollout_groups(
        specs,
        schedule="async",
        gen_linger_s=LINGER_S,
        group_size=4,
        repo_context=False,
        verify_broken=False,
        max_steps=20,
        max_tokens_per_step=64,
        sandbox_factory=factory,
        grade_fn=grader,
    )
    assert max_active == 1


def test_rollout_groups_reports_shared_broker_generation_metrics():
    specs = [{"task": TASK, "group_id": gid} for gid in (0, 1)]
    engine = StubEngine()
    factory, grader, _ = make_env(HETERO, grade_s=0.0)
    outputs = engine.rollout_groups(
        specs,
        schedule="async",
        gen_linger_s=LINGER_S,
        group_size=2,
        repo_context=False,
        verify_broken=False,
        score=False,
        max_steps=1,
        max_tokens_per_step=64,
        sandbox_factory=factory,
        grade_fn=grader,
    )

    first = outputs[0][1]
    assert first.shared_generation_s >= 0.0
    assert sum(first.shared_generation_batch_sizes) == 4
    assert first.shared_generation_prompt_tokens > 0
    assert first.shared_generation_completion_tokens > 0
    for _trajectories, timings in outputs[1:]:
        assert timings.shared_generation_s == first.shared_generation_s
        assert timings.shared_generation_batch_sizes == first.shared_generation_batch_sizes


# ----------------------------------------------------------------------
# termination parity and refusals
# ----------------------------------------------------------------------


@pytest.mark.parametrize("schedule", ["lockstep", "async"])
def test_submit_is_marked_on_the_committed_action(schedule):
    blocks: dict[int, list[CommittedBlock]] = {rid: [] for rid in range(2)}

    def on_block(events):
        for rollout_id, block in events:
            blocks[rollout_id].append(block)

    engine = StubEngine(n_turns=1)
    _, trajectories, _, _ = run(
        schedule,
        latency={0: [], 1: []},
        group_size=2,
        grade_s=0.0,
        engine=engine,
        max_steps=3,
        on_block=on_block,
    )
    assert [trajectory.finish_reason for trajectory in trajectories] == [
        "submit",
        "submit",
    ]
    for rollout_blocks in blocks.values():
        assert [block.kind for block in rollout_blocks] == ["action"]
        assert [
            block.terminal_reason_at_commit for block in rollout_blocks
        ] == ["submit"]


@pytest.mark.parametrize("schedule", ["lockstep", "async"])
def test_max_steps_is_marked_only_on_the_configured_final_action(schedule):
    blocks: dict[int, list[CommittedBlock]] = {rid: [] for rid in range(2)}

    def on_block(events):
        for rollout_id, block in events:
            blocks[rollout_id].append(block)

    engine = StubEngine(n_turns=50)
    _, trajectories, _, _ = run(
        schedule,
        latency={0: [0.0, 0.0], 1: [0.0, 0.0]},
        group_size=2,
        grade_s=0.0,
        engine=engine,
        max_steps=2,
        on_block=on_block,
    )
    assert [trajectory.finish_reason for trajectory in trajectories] == [
        "max_steps",
        "max_steps",
    ]
    for rollout_blocks in blocks.values():
        assert [block.kind for block in rollout_blocks] == [
            "action",
            "observation",
            "action",
            "observation",
        ]
        assert [
            block.terminal_reason_at_commit for block in rollout_blocks
        ] == [None, None, "max_steps", None]


def test_committed_block_rejects_invalid_terminal_provenance():
    with pytest.raises(ValueError, match="terminal reason"):
        CommittedBlock(
            step_index=0,
            kind="action",
            token_ids=(1,),
            scored=(True,),
            committed_at=1.0,
            terminal_reason_at_commit="context_limit",
        )
    with pytest.raises(ValueError, match="only action blocks"):
        CommittedBlock(
            step_index=0,
            kind="observation",
            token_ids=(1,),
            scored=(False,),
            committed_at=1.0,
            terminal_reason_at_commit="max_steps",
        )


def test_finish_reasons_match_across_schedules_at_the_step_limit():
    lat = {i: [0.0] * 10 for i in range(4)}
    _, lock_trajs, _, _ = run("lockstep", latency=lat, grade_s=0.0, max_steps=2)
    _, async_trajs, _, _ = run("async", latency=lat, grade_s=0.0, max_steps=2)
    assert [t.finish_reason for t in lock_trajs] == ["max_steps"] * 4
    assert [t.finish_reason for t in async_trajs] == ["max_steps"] * 4
    assert [len(t.steps) for t in lock_trajs] == [len(t.steps) for t in async_trajs]


def test_finish_reasons_match_across_schedules_at_the_context_limit():
    lat = {i: [0.0] * 10 for i in range(4)}
    # sized so the budget admits the first turn and not the second, computed
    # from the header rather than hardcoded
    header = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{TASK_TEMPLATE.format(problem_statement=TASK.problem_statement)}<|im_end|>\n"
    )
    max_model_len = len(header) + 64 + len(TURN_OPEN) + 40
    eng_l = StubEngine(max_model_len=max_model_len, n_turns=50)
    eng_a = StubEngine(max_model_len=max_model_len, n_turns=50)
    _, lock_trajs, _, _ = run("lockstep", latency=lat, grade_s=0.0, engine=eng_l,
                              max_steps=20, max_tokens_per_step=64)
    _, async_trajs, _, _ = run("async", latency=lat, grade_s=0.0, engine=eng_a,
                               max_steps=20, max_tokens_per_step=64)
    assert all(len(t.steps) == 1 for t in lock_trajs), (
        "the budget should admit exactly one turn for this test to mean anything"
    )
    assert [t.finish_reason for t in lock_trajs] == ["context_limit"] * 4
    assert [t.finish_reason for t in async_trajs] == ["context_limit"] * 4
    assert [t.token_ids for t in lock_trajs] == [t.token_ids for t in async_trajs]


class CappedStubEngine(StubEngine):
    """A stub whose completion length depends on the rollout and obeys ``max_tokens``.

    The rollout is recovered from the request seed (group 0, base seed 0), and
    every request's cap is recorded per rollout, so a test can check the cap
    each turn asked for. A completion cut by its cap loses its closing fence,
    which makes it an unparsable action; the trajectory continues.
    """

    def __init__(self, *, group_size: int = 4, max_turns: int = 50, **kw):
        super().__init__(**kw)
        self.caps: dict[int, list[int]] = {}
        self.list_calls = 0
        self.rollout_of_seed = {
            request_seed(0, 0, rollout, turn): rollout
            for rollout in range(group_size)
            for turn in range(max_turns)
        }

    def generate(self, prompts, *, max_tokens=1024, temperature=1.0, top_p=1.0,
                 seeds=None):
        caps = list(max_tokens) if isinstance(max_tokens, list) else [
            max_tokens
        ] * len(prompts)
        with self._lock:
            self.batch_sizes.append(len(prompts))
            self.list_calls += isinstance(max_tokens, list)
        out = []
        for k in range(len(prompts)):
            rollout = self.rollout_of_seed[int(seeds[k])]
            with self._lock:
                self.caps.setdefault(rollout, []).append(caps[k])
            ids = self.encode(f"```bash\necho {'x' * (5 + 7 * rollout)}\n```")[: caps[k]]
            out.append(
                GenResult(
                    token_ids=ids,
                    logprobs=[-0.01] * len(ids),
                    finish_reason="length" if len(ids) == caps[k] else "stop",
                    text=self.decode(ids),
                )
            )
        return out


@pytest.mark.parametrize("schedule", ["lockstep", "async"])
def test_generated_token_cap_ends_the_trajectory(schedule):
    per_step, cap = 64, 60
    eng = CappedStubEngine(n_turns=50)
    _, trajs, _, _ = run(
        schedule,
        latency={i: [0.0] * 10 for i in range(4)},
        grade_s=0.0,
        engine=eng,
        max_steps=20,
        max_tokens_per_step=per_step,
        max_generated_tokens=cap,
    )
    for t in trajs:
        length = len(f"```bash\necho {'x' * (5 + 7 * t.rollout_id)}\n```")
        expected, generated = [], 0
        while generated < cap:
            expected.append(min(per_step, cap - generated))
            generated += min(length, expected[-1])
        assert eng.caps[t.rollout_id] == expected
        assert t.finish_reason == "generation_limit"
        assert sum(len(s.action_token_ids) - len(TURN_OPEN) for s in t.steps) == cap
    if schedule == "lockstep":
        assert eng.list_calls > 0, "mixed per-rollout caps never reached generate"


def test_generated_token_cap_is_token_identical_across_schedules():
    lat = {i: [0.0] * 10 for i in range(4)}
    kw = dict(grade_s=0.0, max_steps=20, max_tokens_per_step=64,
              max_generated_tokens=60)
    _, lock_trajs, _, _ = run("lockstep", latency=lat,
                              engine=CappedStubEngine(n_turns=50), **kw)
    _, async_trajs, _, _ = run("async", latency=lat,
                               engine=CappedStubEngine(n_turns=50), **kw)
    assert [t.finish_reason for t in lock_trajs] == ["generation_limit"] * 4
    assert [t.token_ids for t in lock_trajs] == [t.token_ids for t in async_trajs]
    assert [t.action_mask for t in lock_trajs] == [t.action_mask for t in async_trajs]


@pytest.mark.parametrize("schedule", ["lockstep", "async"])
def test_tool_observation_keeps_head_and_tail_tokens(schedule):
    output = "H" * 10 + "m" * 50 + "T" * 10

    class LongOutputSandbox(StubSandbox):
        def exec(self, command: str) -> str:
            text = super().exec(command)
            return text if not text.startswith("out:") else output

    def factory(image: str, name: str) -> LongOutputSandbox:
        _, gid, rid = name.rsplit("-", 2)
        return LongOutputSandbox(image, name, int(gid), int(rid), {}, 0.0)

    trajs, _ = StubEngine(n_turns=2).rollout_group(
        TASK,
        schedule=schedule,
        group_size=2,
        repo_context=False,
        verify_broken=False,
        sandbox_factory=factory,
        grade_fn=lambda box, task: 0.0,
        max_observation_tokens=20,
    )
    for t in trajs:
        assert t.steps[0].observation == (
            "H" * 10 + "\n\n... <50 tokens elided> ...\n\n" + "T" * 10
        )


def test_async_refuses_coupling_rather_than_changing_what_is_sampled():
    eng = StubEngine()
    f, g, _ = make_env(HETERO, grade_s=0.0)
    with pytest.raises(NotImplementedError, match="advance together"):
        eng.rollout_group(
            TASK, schedule="async", group_size=4, repo_context=False,
            verify_broken=False, coupled_prefix_steps=2,
            sandbox_factory=f, grade_fn=g,
        )


class SeedRecordingStubEngine(StubEngine):
    """Records the seeds of every ``generate`` call, one list per call."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.seed_calls: list[list[int]] = []

    def generate(self, prompts, **kw):
        with self._lock:
            self.seed_calls.append(list(kw.get("seeds") or []))
        return super().generate(prompts, **kw)


def test_request_seeds_are_stable_and_distinct_per_turn_and_rollout():
    seed = request_seed(7, 3, 1001, 2)
    assert seed == request_seed(7, 3, 1001, 2)
    assert 0 <= seed < 1 << 63
    keys = [
        (base, group, rollout, turn)
        for base in (0, 1)
        for group in range(3)
        for rollout in (0, 1, 999, 1000, 1001)
        for turn in range(4)
    ]
    assert len({request_seed(*key) for key in keys}) == len(keys)
    # the former base + group * 1000 + rollout formula collided here
    assert request_seed(0, 0, 1000, 0) != request_seed(0, 1, 0, 0)
    assert request_seed(-1, 0, 0, 0) != request_seed(0, 0, 0, 0)


@pytest.mark.parametrize("schedule", ["lockstep", "async"])
def test_each_turn_samples_with_its_own_request_seed(schedule):
    eng = SeedRecordingStubEngine(n_turns=3)
    run(schedule, latency={i: [0.0] * 10 for i in range(2)}, group_size=2,
        grade_s=0.0, engine=eng, base_seed=5, group_id=4)
    used = sorted(seed for call in eng.seed_calls for seed in call)
    expected = sorted(
        request_seed(5, 4, rollout, turn) for rollout in range(2) for turn in range(3)
    )
    assert used == expected


def test_a_coupled_prefix_shares_the_slot_seed_and_keeps_that_rollout_unchanged():
    lat = {i: [0.0] * 10 for i in range(3)}
    coupled_eng = SeedRecordingStubEngine(n_turns=4)
    _, coupled, _, _ = run("lockstep", latency=lat, group_size=3, grade_s=0.0,
                           engine=coupled_eng, coupled_prefix_steps=2)
    _, free, _, _ = run("lockstep", latency=lat, group_size=3, grade_s=0.0,
                        engine=StubEngine(n_turns=4))
    for turn in range(2):
        assert coupled_eng.seed_calls[turn] == [
            request_seed(0, 0, rollout_engine.COUPLED_ROLLOUT_SLOT, turn)
        ]
    slot = rollout_engine.COUPLED_ROLLOUT_SLOT
    assert coupled[slot].token_ids == free[slot].token_ids
    assert all(t.steps[1].action == coupled[slot].steps[1].action for t in coupled)


@pytest.mark.parametrize("schedule", ["lockstep", "async"])
def test_an_initial_prompt_over_the_cap_is_refused_before_generation(schedule):
    eng = StubEngine()
    with pytest.raises(ValueError, match="max_prompt_tokens=64"):
        run(schedule, latency={i: [0.0] * 10 for i in range(2)}, group_size=2,
            grade_s=0.0, engine=eng, max_prompt_tokens=64)
    assert eng.batch_sizes == []
    header = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{TASK_TEMPLATE.format(problem_statement=TASK.problem_statement)}"
        "<|im_end|>\n"
    )
    _, trajs, _, _ = run(schedule, latency={i: [0.0] * 10 for i in range(2)},
                         group_size=2, grade_s=0.0, engine=StubEngine(),
                         max_prompt_tokens=len(header))
    assert len(trajs[0].prompt_token_ids) == len(header)


@pytest.mark.parametrize("cap", [0, -1, 1.5, True])
def test_the_prompt_cap_must_be_a_positive_integer(cap):
    with pytest.raises(ValueError, match="positive integer"):
        run("lockstep", latency={}, group_size=1, grade_s=0.0, max_prompt_tokens=cap)


def test_unknown_schedule_is_rejected():
    eng = StubEngine()
    with pytest.raises(ValueError, match="unknown schedule"):
        eng.rollout_group(TASK, schedule="fastest", group_size=2)


def test_generation_failure_reaches_every_worker_instead_of_hanging():
    """A broker that dropped an error would deadlock every rollout waiting on it."""
    eng = StubEngine(gen_s=0.0)
    eng.fail_after = 4
    f, g, _ = make_env({i: [0.0] * 10 for i in range(4)}, grade_s=0.0)
    with pytest.raises(RuntimeError, match="stub engine failure"):
        eng.rollout_group(
            TASK, schedule="async", group_size=4, repo_context=False,
            verify_broken=False, max_steps=20, max_tokens_per_step=64,
            sandbox_factory=f, grade_fn=g, gen_linger_s=LINGER_S,
        )


def test_generation_broker_request_timeout_is_bounded():
    """A live dispatcher with a stalled engine must not block its caller forever."""
    entered = threading.Event()
    release = threading.Event()

    class StalledEngine(StubEngine):
        def generate(self, *args, **kwargs):
            entered.set()
            release.wait()
            return super().generate(*args, **kwargs)

    eng = StalledEngine(gen_s=0.0)
    broker = _GenerationBroker(eng, linger_s=0.0, request_timeout_s=0.02).start()
    try:
        prompt = eng.encode("test")
        with pytest.raises(TimeoutError, match="exceeded"):
            broker.submit(
                prompt, seed=1, max_tokens=4, temperature=0.0
            )
        assert entered.is_set()
    finally:
        release.set()
        broker.close()


def test_source_diff_is_anchored_before_policy_commits():
    bug_base = "a" * 40

    class CommittedFixSandbox:
        def exec(self, command: str) -> str:
            if command == "GIT_NO_REPLACE_OBJECTS=1 git rev-parse HEAD":
                return bug_base
            if command.startswith(
                f"GIT_NO_REPLACE_OBJECTS=1 git diff {bug_base} "
            ):
                return " src/fix.py | 2 +-\n"
            if command.startswith(
                "GIT_NO_REPLACE_OBJECTS=1 git ls-files --others"
            ):
                return ""
            raise AssertionError(command)

    sandbox = CommittedFixSandbox()
    task = Task("task", "problem", "image", bug_patch="diff --git a/x b/x")
    captured = _capture_bug_base(sandbox, task)
    assert captured == bug_base
    assert _source_diff(sandbox, captured) == "src/fix.py | 2 +-"


def test_source_diff_reports_untracked_work():
    bug_base = "d" * 40

    class UntrackedSandbox:
        def exec(self, command: str) -> str:
            if command.startswith("GIT_NO_REPLACE_OBJECTS=1 git diff"):
                return ""
            if command.startswith("GIT_NO_REPLACE_OBJECTS=1 git ls-files"):
                return "src/new_file.py\n"
            raise AssertionError(command)

    assert _source_diff(UntrackedSandbox(), bug_base) == (
        "untracked files:\nsrc/new_file.py"
    )


def test_patch_capture_refuses_the_policy_git_without_a_trusted_baseline(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "test"], cwd=tmp_path, check=True
    )
    source = tmp_path / "source.py"
    source.write_text("before = 1\n")
    subprocess.run(["git", "add", "source.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "bug"], cwd=tmp_path, check=True)
    source.write_text("after = 2\n")
    subprocess.run(["git", "commit", "-am", "fix", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "new_file.py").write_text("new = 3\n")

    sandbox = type("HostSandbox", (), {"host_workdir": tmp_path})()
    # Without the trusted baseline the policy's own .git is all there is, and
    # host Git must not read its config.
    with pytest.raises(RuntimeError, match="trusted baseline snapshot"):
        _capture_policy_patch(sandbox, max_bytes=100_000)


def test_trusted_patch_capture_survives_policy_git_replacement(tmp_path):
    policy_repo = tmp_path / "policy"
    policy_repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=policy_repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=policy_repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "test"], cwd=policy_repo, check=True
    )
    source = policy_repo / "source.py"
    source.write_text("before = 1\n")
    subprocess.run(["git", "add", "source.py"], cwd=policy_repo, check=True)
    subprocess.run(["git", "commit", "-qm", "bug"], cwd=policy_repo, check=True)
    bug_base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=policy_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    class HostSandbox:
        host_workdir = policy_repo

        def exec(self, command: str) -> str:
            assert command == "GIT_NO_REPLACE_OBJECTS=1 git rev-parse HEAD"
            return bug_base

    sandbox = HostSandbox()
    task = Task("task", "problem", "image", bug_patch="sealed")
    assert _capture_bug_base(
        sandbox, task, trusted_host_snapshot=True
    ) == bug_base

    shutil.rmtree(policy_repo / ".git")
    source.write_text("after = 2\n")
    (policy_repo / "new_file.py").write_text("new = 3\n")
    cache = policy_repo / "__pycache__"
    cache.mkdir()
    (cache / "source.cpython-310.pyc").write_bytes(b"generated bytecode")
    egg_info = policy_repo / "example.egg-info"
    egg_info.mkdir()
    (egg_info / "PKG-INFO").write_text("generated package metadata\n")
    (policy_repo / ".git").mkdir()
    (policy_repo / ".git/object").write_text("policy-controlled replacement\n")

    patch = _capture_policy_patch(sandbox, max_bytes=100_000)
    assert b"+after = 2" in patch
    assert b"new_file.py" in patch
    assert b".git/object" not in patch
    assert b"__pycache__" not in patch
    assert b"generated bytecode" not in patch
    assert b"example.egg-info" not in patch
    assert b"generated package metadata" not in patch
    shutil.rmtree(sandbox._thundersync_trusted_root)


def test_trusted_patch_capture_rejects_oversized_policy_work(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    (tmp_path / "source.py").write_text("before = 1\n")
    subprocess.run(["git", "add", "source.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "bug"], cwd=tmp_path, check=True)
    bug_base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    class HostSandbox:
        host_workdir = tmp_path

        def exec(self, _command: str) -> str:
            return bug_base

    sandbox = HostSandbox()
    _capture_bug_base(
        sandbox,
        Task("task", "problem", "image", bug_patch="sealed"),
        trusted_host_snapshot=True,
    )
    (tmp_path / "source.py").write_text("after = " + "x" * 10_000)
    with pytest.raises(PolicyPatchRejected, match="limit"):
        _capture_policy_patch(sandbox, max_bytes=100)
    shutil.rmtree(sandbox._thundersync_trusted_root)


def test_isolated_grader_applies_only_repository_patch_in_fresh_worktree(tmp_path):
    policy_repo = tmp_path / "policy"
    policy_repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=policy_repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=policy_repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "test"], cwd=policy_repo, check=True
    )
    (policy_repo / "tests").mkdir()
    (policy_repo / "tests/test_x.py").write_text("def test_fixed(): pass\n")
    (policy_repo / "source.py").write_text("before = 1\n")
    subprocess.run(["git", "add", "."], cwd=policy_repo, check=True)
    subprocess.run(["git", "commit", "-qm", "bug"], cwd=policy_repo, check=True)
    bug_base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=policy_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    trusted = type("Trusted", (), {"host_workdir": policy_repo})()
    _capture_trusted_bug_tree(trusted, bug_base)
    (policy_repo / "source.py").write_text("after = 2\n")
    subprocess.run(
        ["git", "commit", "-am", "fix", "-q"], cwd=policy_repo, check=True
    )
    (policy_repo / "new_file.py").write_text("new = 3\n")

    class PolicySandbox:
        host_workdir = policy_repo
        _thundersync_bug_base = bug_base
        _thundersync_trusted_git_dir = trusted._thundersync_trusted_git_dir
        _thundersync_trusted_tree = trusted._thundersync_trusted_tree

    verifier_repo = tmp_path / "verifier"

    class VerifierSandbox:
        host_workdir = verifier_repo
        name = "verifier"

        def start(self):
            return None

        def apply_bug_patch(self, _patch):
            return None

        def assert_bug_not_revertible(self):
            return None

        def exec(self, command: str) -> str:
            if command.startswith("cat /tmp/thundersync_report_"):
                return (
                    '<testsuites><testsuite><testcase classname="tests.test_x" '
                    'name="test_fixed"/></testsuite></testsuites>'
                )
            if command.startswith("rm -f /tmp/thundersync_report_"):
                return "THUNDERSYNC_PYTEST_RC=1\n"
            result = subprocess.run(
                ["bash", "--noprofile", "--norc", "-c", command],
                cwd=verifier_repo,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout + result.stderr

        def close(self):
            return None

    def factory(_image, _name):
        subprocess.run(
            ["git", "clone", "-q", str(policy_repo), str(verifier_repo)],
            check=True,
        )
        subprocess.run(
            ["git", "checkout", "-q", bug_base], cwd=verifier_repo, check=True
        )
        return VerifierSandbox()

    task = Task(
        "task",
        "problem",
        "image",
        bug_patch="sealed by the real sandbox",
        fail_to_pass=["tests/test_x.py::test_fixed"],
    )
    verdict = grade_isolated(
        PolicySandbox(),
        task,
        sandbox_factory=factory,
        verifier_name="verifier",
        check_regressions=False,
    )
    assert verdict.reward == 1.0
    assert (verifier_repo / "source.py").read_text() == "after = 2\n"
    assert (verifier_repo / "new_file.py").read_text() == "new = 3\n"


def test_oracle_controlled_paths_cover_helpers_hooks_and_configuration():
    exact = {"custom/check.py"}
    assert _is_oracle_controlled("tests/helpers/data.json", exact)
    assert _is_oracle_controlled("pkg/test/unit.py", exact)
    assert _is_oracle_controlled("plugins/conftest.py", exact)
    assert _is_oracle_controlled("pyproject.toml", exact)
    assert _is_oracle_controlled("custom/check.py", exact)
    assert _is_oracle_controlled("modules/caddyhttp/server_test.go", exact)
    assert not _is_oracle_controlled("src/runtime.py", exact)


def test_go_json_runner_uses_structured_terminal_events_and_fails_closed():
    class GoSandbox:
        def exec(self, command: str) -> str:
            if command.startswith("rm -f /tmp/thundersync_go_report_"):
                return "THUNDERSYNC_GO_TEST_RC=1\n"
            if command.startswith("cat /tmp/thundersync_go_report_"):
                return "\n".join(
                    [
                        '{"Action":"run","Package":"example/a","Test":"TestPass"}',
                        '{"Action":"pass","Package":"example/a","Test":"TestPass"}',
                        '{"Action":"run","Package":"example/a","Test":"TestFail"}',
                        '{"Action":"fail","Package":"example/a","Test":"TestFail"}',
                        '{"Action":"skip","Package":"example/a","Test":"TestSkip"}',
                    ]
                )
            raise AssertionError(command)

    selectors = ["TestPass", "TestFail", "TestSkip", "TestMissing"]
    outcomes, reason = _run_go_test_json(GoSandbox(), selectors)
    assert reason == ""
    assert outcomes == {
        "TestPass": "passed",
        "TestFail": "failed",
        "TestSkip": "failed",
    }


def test_go_json_runner_rejects_missing_machine_report():
    class MissingReportSandbox:
        def exec(self, command: str) -> str:
            if command.startswith("rm -f /tmp/thundersync_go_report_"):
                return "go: command not found\nTHUNDERSYNC_GO_TEST_RC=127\n"
            if command.startswith("cat /tmp/thundersync_go_report_"):
                return ""
            raise AssertionError(command)

    outcomes, reason = _run_go_test_json(MissingReportSandbox(), ["TestOne"])
    assert outcomes == {}
    assert reason == "no_report"


def test_go_grade_runs_one_structured_suite_for_targets_and_regressions(
    monkeypatch,
):
    calls = []

    def fake_go_runner(_sandbox, selectors):
        calls.append(selectors)
        return {name: "passed" for name in selectors}, ""

    monkeypatch.setattr(rollout_engine, "_run_go_test_json", fake_go_runner)
    task = Task(
        "go-task",
        "problem",
        "image",
        fail_to_pass=["TestTarget"],
        pass_to_pass=["TestRegression", "TestRegression/subtest"],
        test_framework="go",
    )
    verdict = grade(object(), task, restore_tests=False)
    assert verdict.reward == 1.0
    assert verdict.n_pass_to_pass == 2
    assert calls == [["TestTarget", "TestRegression", "TestRegression/subtest"]]


def test_grader_restores_tests_from_the_sealed_bug_commit():
    bug_base = "b" * 40

    class CommittedTestEditSandbox:
        _thundersync_bug_base = bug_base

        def __init__(self):
            self.commands = []

        def exec(self, command: str) -> str:
            self.commands.append(command)
            if command.startswith(f"git cat-file -e {bug_base}"):
                return "tests/conftest.py\nTHUNDERSYNC_ORACLE_RESTORE_OK\n"
            if command.startswith("find . -type f -not -path"):
                return (
                    "./tests/conftest.py\n./injected/conftest.py\n"
                    "THUNDERSYNC_ORACLE_RESTORE_OK\n"
                )
            if "THUNDERSYNC_ORACLE_RESTORE_OK" in command:
                return "THUNDERSYNC_ORACLE_RESTORE_OK\n"
            if command.startswith("rm -f /tmp/thundersync_report_"):
                return "THUNDERSYNC_PYTEST_RC=1\n"
            if command.startswith("cat /tmp/thundersync_report_"):
                return (
                    '<testsuites><testsuite><testcase classname="tests.test_x" '
                    'name="test_fixed"/></testsuite></testsuites>'
                )
            return ""

    sandbox = CommittedTestEditSandbox()
    task = Task(
        "task",
        "problem",
        "image",
        fail_to_pass=["tests/test_x.py::test_fixed"],
    )
    verdict = grade(sandbox, task, check_regressions=False)
    assert verdict.reward == 1.0
    assert any(
        command.startswith(f"git checkout {bug_base} -- tests/test_x.py")
        for command in sandbox.commands
    )
    assert any(
        command.startswith(f"git cat-file -e {bug_base}")
        for command in sandbox.commands
    )
    assert any(command.startswith("find . -type f -not -path") for command in sandbox.commands)
    assert any(
        command.startswith("rm -f -- injected/conftest.py")
        for command in sandbox.commands
    )
    assert any(
        command.startswith(f"git checkout {bug_base} -- tests/conftest.py")
        for command in sandbox.commands
    )


def test_grader_fails_closed_when_oracle_restoration_fails():
    class BrokenGitSandbox:
        _thundersync_bug_base = "c" * 40

        def exec(self, _command: str) -> str:
            return "fatal: bad object\n"

    verdict = grade(
        BrokenGitSandbox(),
        Task(
            "task",
            "problem",
            "image",
            fail_to_pass=["tests/test_x.py::test_fixed"],
        ),
        check_regressions=False,
    )
    assert verdict.reward == 0.0
    assert verdict.reason == "oracle_restore_failed"


def test_grader_retries_transient_oracle_restoration_failure():
    bug_base = "d" * 40

    class TransientRestoreSandbox:
        _thundersync_bug_base = bug_base

        def __init__(self):
            self.restore_attempts = 0

        def exec(self, command: str) -> str:
            if command.startswith(f"git checkout {bug_base} -- tests/test_x.py"):
                self.restore_attempts += 1
                if self.restore_attempts == 1:
                    return "FATAL: container creation failed\n"
                return "THUNDERSYNC_ORACLE_RESTORE_OK\n"
            if command.startswith(f"git cat-file -e {bug_base}"):
                return "tests/test_x.py\nTHUNDERSYNC_ORACLE_RESTORE_OK\n"
            if command.startswith("find . -type f -not -path"):
                return "./tests/test_x.py\nTHUNDERSYNC_ORACLE_RESTORE_OK\n"
            if command.startswith("rm -f /tmp/thundersync_report_"):
                return "THUNDERSYNC_PYTEST_RC=1\n"
            if command.startswith("cat /tmp/thundersync_report_"):
                return (
                    '<testsuites><testsuite><testcase classname="tests.test_x" '
                    'name="test_fixed"/></testsuite></testsuites>'
                )
            raise AssertionError(command)

    sandbox = TransientRestoreSandbox()
    verdict = grade(
        sandbox,
        Task(
            "task",
            "problem",
            "image",
            fail_to_pass=["tests/test_x.py::test_fixed"],
        ),
        check_regressions=False,
    )
    assert verdict.reward == 1.0
    # Two calls belong to the initial retry and one to the complete baseline
    # test-surface restoration later in grade().
    assert sandbox.restore_attempts == 3


def test_file_level_selector_uses_individual_pytest_outcome():
    selector = "tests/examplefiles/lagda/example.lagda::"

    class FileSelectorSandbox:
        def __init__(self):
            self.commands = []

        def exec(self, command: str) -> str:
            self.commands.append(command)
            if command.startswith("rm -f /tmp/thundersync_file_report_"):
                return "THUNDERSYNC_PYTEST_RC=0\n"
            if command.startswith("cat /tmp/thundersync_file_report_"):
                return (
                    '<testsuites><testsuite tests="1"><testcase '
                    'classname="custom.collector" name="example.lagda"/>'
                    "</testsuite></testsuites>"
                )
            raise AssertionError(command)

    sandbox = FileSelectorSandbox()
    outcomes, reason = _run_pytest_junit(sandbox, [selector])
    assert reason == ""
    assert outcomes == {selector: "passed"}
    assert any(selector in command for command in sandbox.commands)
    assert any("-p no:python" in command for command in sandbox.commands)


def test_file_level_selector_rejects_skipped_or_empty_collection():
    class FileSelectorSandbox:
        def __init__(self, xml: str):
            self.xml = xml

        def exec(self, command: str) -> str:
            if command.startswith("rm -f /tmp/thundersync_file_report_"):
                return "THUNDERSYNC_PYTEST_RC=0\n"
            if command.startswith("cat /tmp/thundersync_file_report_"):
                return self.xml
            raise AssertionError(command)

    selector = "tests/example.data::"
    skipped_xml = (
        '<testsuites><testsuite tests="1"><testcase name="example">'
        "<skipped/></testcase></testsuite></testsuites>"
    )
    outcomes, reason = _run_pytest_junit(FileSelectorSandbox(skipped_xml), [selector])
    assert reason == ""
    assert outcomes == {selector: "failed"}

    empty_xml = '<testsuites><testsuite tests="0"/></testsuites>'
    outcomes, reason = _run_pytest_junit(FileSelectorSandbox(empty_xml), [selector])
    assert reason == ""
    assert outcomes == {selector: "failed"}


def test_pytest_selector_candidates_preserve_double_colon_in_parameters():
    selector = "tests/test_func.py::test_func_function[dict::udict_set_builder(]"
    candidates = _pytest_selector_candidates(selector)
    assert "tests.test_func::test_func_function[dict::udict_set_builder(]" in candidates
    assert "test_func_function[dict::udict_set_builder(]" in candidates

    class_selector = "tests/test_api.py::TestClient::test_request[param::value]"
    candidates = _pytest_selector_candidates(class_selector)
    assert "tests.test_api.TestClient::test_request[param::value]" in candidates


def test_selector_batches_limit_count_and_quoted_command_length():
    selectors = [f"tests/test_api.py::test_case[{index}]" for index in range(1201)]
    batches = _selector_batches(selectors, max_items=500)
    assert [len(batch) for batch in batches] == [500, 500, 201]
    assert [selector for batch in batches for selector in batch] == selectors

    long_selectors = ["x" * 40_000, "y" * 40_000, "z" * 40_000]
    assert [len(batch) for batch in _selector_batches(long_selectors)] == [2, 1]


def test_async_free_rider_check_fires_at_the_trajectory_not_the_group():
    """A reward with no source diff must never reach on_verdict."""
    eng = StubEngine()

    class NoDiffSandbox(StubSandbox):
        def exec(self, command: str) -> str:
            if " git diff " in f" {command} ":
                return "   \n"
            return super().exec(command)

    def factory(image, name):
        parts = name.rsplit("-", 2)
        return NoDiffSandbox(image, name, int(parts[1]), int(parts[2]),
                             {i: [0.0] * 10 for i in range(4)}, 0.0)

    fired: list[int] = []
    with pytest.raises(RuntimeError, match="empty source diff"):
        eng.rollout_group(
            TASK, schedule="async", group_size=4, repo_context=False,
            verify_broken=False, max_steps=20, max_tokens_per_step=64,
            sandbox_factory=factory, grade_fn=lambda box, task: 1.0,
            gen_linger_s=LINGER_S, on_verdict=lambda rid, traj: fired.append(rid),
        )
    assert fired == [], "a compromised verdict was handed to the trainer"


def test_a_node_selector_batch_never_reads_a_previous_grades_report():
    """A reused verifier may hold a report file from its last grade; a batch
    that times out must not be scored from that file."""
    stale_pass = (
        '<testsuites><testsuite><testcase classname="tests.test_x" '
        'name="test_fixed"/></testsuite></testsuites>'
    )

    class ReusedVerifier:
        def __init__(self, run_output: str):
            self.run_output = run_output
            self.commands = []

        def exec(self, command: str) -> str:
            self.commands.append(command)
            if command.startswith("rm -f /tmp/thundersync_report_"):
                return self.run_output
            if command.startswith("cat /tmp/thundersync_report_"):
                return stale_pass
            raise AssertionError(command)

    selector = "tests/test_x.py::test_fixed"
    timed_out = ReusedVerifier("<command timed out after 900s>")
    assert _run_pytest_junit(timed_out, [selector]) == ({}, "timeout")
    assert not any(command.startswith("cat ") for command in timed_out.commands)
    assert _run_pytest_junit(ReusedVerifier(""), [selector]) == ({}, "no_report")
    completed = ReusedVerifier("THUNDERSYNC_PYTEST_RC=0\n")
    outcomes, reason = _run_pytest_junit(completed, [selector])
    assert reason == "" and outcomes["tests.test_x::test_fixed"] == "passed"



def test_task_rejects_an_unknown_test_framework():
    with pytest.raises(ValueError, match="test_framework"):
        Task("task", "problem", "image", test_framework="jest")


def test_parse_test_ids_rejects_a_string_that_is_not_a_json_list():
    assert parse_test_ids('["a::b", "c"]') == ["a::b", "c"]
    with pytest.raises(ValueError, match="JSON list"):
        parse_test_ids("tests/test_x.py::test_fixed")
    with pytest.raises(ValueError, match="JSON list"):
        parse_test_ids('{"a": 1}')


def test_junit_reports_with_a_doctype_or_entity_are_malformed():
    body = '<testsuites><testsuite><testcase classname="c" name="n"/></testsuite></testsuites>'
    assert _parse_junit(body) is not None
    assert _parse_junit('<!DOCTYPE x [<!ENTITY a "b">]>' + body) is None
    assert _parse_junit('<!ENTITY a "b">' + body) is None
    assert _parse_junit("<testsuite") is None


def test_report_paths_are_fresh_per_grade():
    seen = []

    class Verifier:
        def exec(self, command: str) -> str:
            if command.startswith("rm -f /tmp/thundersync_report_"):
                seen.append(command.split()[2].rstrip(";"))
                return "THUNDERSYNC_PYTEST_RC=0\n"
            if command.startswith("cat /tmp/thundersync_report_"):
                return (
                    '<testsuites><testsuite><testcase classname="tests.test_x" '
                    'name="test_fixed"/></testsuite></testsuites>'
                )
            raise AssertionError(command)

    for _ in range(2):
        _run_pytest_junit(Verifier(), ["tests/test_x.py::test_fixed"])
    assert len(seen) == 2 and seen[0] != seen[1]


def test_new_sandbox_carries_the_task_network_need():
    class Box:
        requires_network = False

    plain = Task("task", "problem", "image")
    networked = Task("task", "problem", "image", requires_network=True)
    assert new_sandbox(lambda image, name: Box(), plain, "n").requires_network is False
    assert new_sandbox(lambda image, name: Box(), networked, "n").requires_network is True
