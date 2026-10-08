"""The trainer's input: a segment forest, not a padded rectangle.

A conventional training step often receives a sample shaped
`(prompt, response, response_mask)` with the mask derived from padding. That
model comes from single-turn RLHF, where

    response == generated == supervised

An agentic trajectory breaks that identity. It alternates

    [prompt] [action] [observation] [action] [observation] ... [action]

and the observations are *injected by the environment* -- pytest output, file
contents, diffs. The model did not generate them and the objective must never
score them merely because they occupy response positions.

This module makes the alternating structure the thing the trainer actually
receives, so that nothing downstream has to reconstruct it. Three properties come
out of that and are used by `plan.py`:

* **provenance** -- whether tokens were emitted by the model or by the
  environment. This is a systems fact (who wrote the bytes), not an ML one; it
  says nothing about rewards, advantages, or task difficulty.
* **sharing** -- trajectories in a group are built by extending a common
  context, so their token streams agree on a prefix. Represented exactly, by
  construction, rather than rediscovered.
* **raggedness** -- trajectories end at different turns. Nothing is padded to a
  common length at any point.

Sharing is computed as a radix (compressed) trie so it is exact at *token*
granularity: two trajectories that agree for the prompt plus the first 40 tokens
of their first action share all of it, and split at token 41. Segment boundaries
do not constrain where a split may happen.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum


class SegmentKind(str, Enum):
    """Where a run of tokens came from.

    Provenance only. The trainer branches on `Segment.scored`, never on the kind
    -- the kind exists so that diagnostics and forest statistics can report in
    terms an agentic RL practitioner recognises.
    """

    PROMPT = "prompt"
    ACTION = "action"
    OBSERVATION = "observation"


@dataclass(frozen=True)
class Segment:
    """A contiguous run of tokens with one provenance and one scoring decision."""

    tokens: tuple[int, ...]
    kind: SegmentKind
    scored: bool

    def __post_init__(self) -> None:
        if not self.tokens:
            raise ValueError("a segment must contain at least one token")

    def __len__(self) -> int:
        return len(self.tokens)

    # Constructors carrying the default provenance->scoring convention. A caller
    # that scores something else (last turn only, say) passes `scored=` and the
    # rest of the system is unaffected: it only ever reads the flag.
    @classmethod
    def prompt(cls, tokens: Iterable[int], *, scored: bool = False) -> Segment:
        return cls(tuple(tokens), SegmentKind.PROMPT, scored)

    @classmethod
    def action(cls, tokens: Iterable[int], *, scored: bool = True) -> Segment:
        return cls(tuple(tokens), SegmentKind.ACTION, scored)

    @classmethod
    def observation(cls, tokens: Iterable[int], *, scored: bool = False) -> Segment:
        return cls(tuple(tokens), SegmentKind.OBSERVATION, scored)


@dataclass
class Trajectory:
    """One agentic rollout: an alternating segment sequence.

    `group_id` marks rollouts of the same task. Only trajectories in a group are
    considered for sharing -- across tasks the prompts differ, so a cross-group
    comparison costs time and finds nothing.
    """

    segments: list[Segment]
    group_id: int = 0
    traj_id: int = 0

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("a trajectory must contain at least one segment")

    @property
    def tokens(self) -> list[int]:
        return [t for s in self.segments for t in s.tokens]

    @property
    def scored_mask(self) -> list[bool]:
        return [s.scored for s in self.segments for _ in s.tokens]

    @property
    def n_tokens(self) -> int:
        return sum(len(s) for s in self.segments)

    @property
    def n_scored(self) -> int:
        return sum(len(s) for s in self.segments if s.scored)

    @property
    def n_turns(self) -> int:
        return sum(1 for s in self.segments if s.kind is SegmentKind.ACTION)

    def keyed(self) -> list[tuple[int, bool]]:
        """Tokens paired with their scoring flag -- the unit of sharing.

        Two trajectories may share a physical hidden state only if they agree on
        both the token *and* whether it is scored. Sharing tokens that disagree
        on scoring would make one physical position carry two different logical
        roles, which is exactly the kind of coefficient folding that silently
        breaks gradients for theta-dependent objectives.
        """
        return [
            (token, segment.scored)
            for segment in self.segments
            for token in segment.tokens
        ]


@dataclass
class ForestNode:
    """A maximal run of tokens shared by an identical set of trajectories.

    Nodes form a forest: one tree per group, rooted at that group's shared
    prompt. A node's tokens are computed once no matter how many trajectories
    pass through it; `multiplicity` records how many that is, which is what turns
    into the saving.
    """

    tokens: list[int]
    scored: list[bool]
    parent: int | None
    depth: int
    trajectories: list[int]
    children: list[int] = field(default_factory=list)
    group_id: int = 0

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    @property
    def multiplicity(self) -> int:
        return len(self.trajectories)

    @property
    def is_leaf(self) -> bool:
        return not self.children


@dataclass
class SegmentForest:
    """The compiled sharing structure for a batch of trajectories."""

    nodes: list[ForestNode]
    roots: list[int]
    trajectories: list[Trajectory]
    leaf_of: dict[int, int]
    """traj_id -> the node where that trajectory ends."""

    # ---- what it cost, and what it would have cost -------------------

    @property
    def physical_tokens(self) -> int:
        return sum(n.n_tokens for n in self.nodes)

    @property
    def logical_tokens(self) -> int:
        return sum(t.n_tokens for t in self.trajectories)

    @property
    def scored_tokens(self) -> int:
        return sum(sum(n.scored) * n.multiplicity for n in self.nodes)

    @property
    def physical_scored_tokens(self) -> int:
        """Scored tokens counted once per shared node (``scored_tokens``
        counts each node once per trajectory through it); a diagnostic."""
        return sum(sum(n.scored) for n in self.nodes)

    @property
    def padded_tokens(self) -> int:
        """What a `[B, T_max]` rectangle would have cost."""
        if not self.trajectories:
            return 0
        return len(self.trajectories) * max(t.n_tokens for t in self.trajectories)

    @property
    def compression(self) -> float:
        """Logical tokens per physical token: the sharing factor."""
        return self.logical_tokens / max(self.physical_tokens, 1)

    @property
    def max_depth(self) -> int:
        return max((n.depth for n in self.nodes), default=0)

    def path_to_root(self, node: int) -> list[int]:
        """Ancestors of `node`, root first, `node` last."""
        path: list[int] = []
        cur: int | None = node
        while cur is not None:
            path.append(cur)
            cur = self.nodes[cur].parent
        path.reverse()
        return path

    def stats(self) -> dict[str, float | int]:
        return {
            "trajectories": len(self.trajectories),
            "nodes": len(self.nodes),
            "logical_tokens": self.logical_tokens,
            "physical_tokens": self.physical_tokens,
            "padded_tokens": self.padded_tokens,
            "compression": round(self.compression, 4),
            "scored_tokens": self.scored_tokens,
            "scored_fraction": round(
                self.scored_tokens / max(self.logical_tokens, 1), 4
            ),
            "max_depth": self.max_depth,
        }


def _longest_common_prefix(seqs: Sequence[list[tuple[int, bool]]]) -> int:
    """Length of the prefix all `seqs` agree on.

    Binary search over slice equality rather than a token loop: "all share a
    prefix of length m" is monotone in m, and slice comparison runs in C. On a
    9k-token shared prompt across 8 rollouts this is ~14 comparisons instead of
    72k.
    """
    if not seqs:
        return 0
    hi = min(len(s) for s in seqs)
    if len(seqs) == 1:
        return hi
    lo = 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        ref = seqs[0][:mid]
        if all(s[:mid] == ref for s in seqs[1:]):
            lo = mid
        else:
            hi = mid - 1
    return lo


def unshared(trajectories: Sequence[Trajectory]) -> list[Trajectory]:
    """The same trajectories with sharing disabled, by making every group unique.

    `build_forest` on the result gives one root per trajectory: a packed,
    block-diagonal, ragged batch with no prefix deduplication -- exactly the
    padding-free path, using the same forest executor without shared prefixes.
    """
    return [
        Trajectory(list(t.segments), group_id=1_000_000 + i, traj_id=t.traj_id)
        for i, t in enumerate(trajectories)
    ]


def build_forest(trajectories: Sequence[Trajectory]) -> SegmentForest:
    """Compress a batch of trajectories into an exact shared forest.

    Exact: concatenating a leaf's path to the root reproduces that trajectory's
    token stream and scoring mask verbatim. Retained forest tests assert this
    property directly.
    """
    nodes: list[ForestNode] = []
    roots: list[int] = []
    leaf_of: dict[int, int] = {}

    by_group: dict[int, list[Trajectory]] = {}
    for t in trajectories:
        by_group.setdefault(t.group_id, []).append(t)

    for group_id, members in sorted(by_group.items()):
        keyed = {t.traj_id: t.keyed() for t in members}

        def emit(traj_ids: list[int], offset: int, parent: int | None, depth: int) -> None:
            """Emit the node shared by `traj_ids` starting at `offset`, then recurse."""
            suffixes = [keyed[i][offset:] for i in traj_ids]
            share = _longest_common_prefix(suffixes)

            # Siblings split at the first divergent token, so a node with more
            # than one trajectory always has share > 0 except at a root where
            # the trajectories disagree immediately -- allow a zero-length root
            # only as a grouping device, never as a real node.
            if share == 0 and parent is not None:
                raise AssertionError("zero-length non-root node: partition is wrong")

            head = suffixes[0][:share] if share else []
            idx = len(nodes)
            nodes.append(
                ForestNode(
                    tokens=[k[0] for k in head],
                    scored=[k[1] for k in head],
                    parent=parent,
                    depth=depth,
                    trajectories=list(traj_ids),
                    group_id=group_id,
                )
            )
            if parent is None:
                roots.append(idx)
            else:
                nodes[parent].children.append(idx)

            nxt = offset + share
            # trajectories that end exactly here terminate at this node
            ongoing: dict[tuple[int, bool], list[int]] = {}
            for i in traj_ids:
                if len(keyed[i]) == nxt:
                    leaf_of[i] = idx
                else:
                    ongoing.setdefault(keyed[i][nxt], []).append(i)

            for _, group in sorted(ongoing.items()):
                emit(group, nxt, idx, depth + 1)

        emit([t.traj_id for t in members], 0, None, 0)

    return SegmentForest(
        nodes=nodes,
        roots=roots,
        trajectories=list(trajectories),
        leaf_of=leaf_of,
    )
