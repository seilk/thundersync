"""Compile a segment forest into everything the training step needs.

The point of this module is that the four things a framework implements as four
independent monkey-patches are, here, four *fields of one object* derived from
one traversal:

    sharing   -> `node_start` / `node_end`  (each node's tokens exist once)
    packing   -> there is no rectangle; the layout is a flat stream
    gating    -> `score_src` lists exactly the positions the objective reads
    attention -> `key_ranges` says what each query may see

They cannot fall out of sync, because there is nothing to keep in sync.

Layout is DFS pre-order, which gives every node a contiguous token range and puts
every ancestor before its descendants. Two consequences are used downstream:

* a node's `position_ids` are determined by its depth-cumulative offset, and are
  identical for every trajectory through it -- so RoPE is well defined on a
  shared node, which is what makes sharing legal at all;
* ancestry is a range containment test, not a pointer chase.

**The scoring index is the part frameworks cannot express.** Each logical scored
token maps to the physical position that produces it, and several logical tokens
may map to the same physical one when they sit in a shared node. Returning
logprobs indexed *logically* means the framework's objective is untouched, and
autograd sums the shared position's gradient over its logical uses on the way
back. No coefficient is ever folded by hand, so this stays exact for objectives
whose coefficients depend on theta (PPO/GRPO importance ratios), which is the
case a hand-folded weight silently breaks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from thundersync.engine.segments import SegmentForest


@dataclass
class ExecutionPlan:
    """A forest, laid out for execution. All indices are into the physical stream."""

    # --- physical token stream ---
    token_ids: torch.Tensor      # [P] the tokens actually computed
    position_ids: torch.Tensor   # [P] each token's position within its trajectory
    node_of: torch.Tensor        # [P] which node each physical token belongs to

    # --- node geometry ---
    node_start: torch.Tensor     # [N]
    node_end: torch.Tensor       # [N]
    node_parent: torch.Tensor    # [N] -1 for roots
    node_depth: torch.Tensor     # [N]
    subtree_end: torch.Tensor    # [P] exclusive DFS end of each token's subtree

    # --- scoring ---
    score_src: torch.Tensor      # [S] physical index of the *predecessor* hidden state
    score_tgt: torch.Tensor      # [S] token id being scored
    score_pos: torch.Tensor      # [S] physical index of the scored token itself
    logical_to_slot: torch.Tensor  # [L] slot in [0,S) that produces each logical token
    logical_traj: torch.Tensor     # [L] which trajectory each logical token belongs to

    # --- attention ---
    key_ranges: list[list[tuple[int, int]]]
    """`key_ranges[n]` = key spans visible to queries in node n, ancestors first,
    node n's own span last. Causal masking applies only within that last span;
    ancestor spans are fully visible because every ancestor token strictly
    precedes every query in n."""

    forest: SegmentForest
    physical_block_mask: object | None = field(default=None, init=False, repr=False)
    """Lazily compiled FlexAttention metadata for the feature-gated backend."""

    @property
    def n_physical(self) -> int:
        return int(self.token_ids.shape[0])

    @property
    def n_scored(self) -> int:
        return int(self.score_src.shape[0])

    @property
    def n_logical_scored(self) -> int:
        return int(self.logical_to_slot.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.node_start.shape[0])

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "ExecutionPlan":
        """Move compiled tensor fields while retaining immutable CPU geometry."""
        target = torch.device(device)
        if self.token_ids.device == target:
            return self
        fields = (
            "token_ids",
            "position_ids",
            "node_of",
            "node_start",
            "node_end",
            "node_parent",
            "node_depth",
            "subtree_end",
            "score_src",
            "score_tgt",
            "score_pos",
            "logical_to_slot",
            "logical_traj",
        )
        moved = {
            name: getattr(self, name).to(target, non_blocking=non_blocking)
            for name in fields
        }
        return ExecutionPlan(
            **moved,
            key_ranges=self.key_ranges,
            forest=self.forest,
        )

    @property
    def is_two_level(self) -> bool:
        """True when the forest is a trunk plus leaf branches.

        This is the common case when trajectories in a group diverge on the
        first action and never re-converge: the tree is the shared prompt plus
        G leaves. It admits a much faster attention path
        (`attention.py`), so it is worth detecting rather than assuming."""
        return int(self.node_depth.max().item()) <= 1 if self.n_nodes else True

    def physical_path(self, traj_id: int) -> list[int]:
        """Physical positions holding one trajectory's tokens, in its own order.

        This is the bridge between the shared layout and the logical view, and
        the thing equivalence tests are written against: reading the plan along
        a trajectory's path must reproduce exactly what an independent,
        unshared, causal forward of that trajectory would have seen.
        """
        out: list[int] = []
        leaf = self.forest.leaf_of[traj_id]
        for n in self.forest.path_to_root(leaf):
            out.extend(range(int(self.node_start[n]), int(self.node_end[n])))
        return out

    def summary(self) -> dict:
        f = self.forest
        return {
            **f.stats(),
            "physical_scored": self.n_scored,
            "logical_scored": self.n_logical_scored,
            "two_level": self.is_two_level,
            # what a padded, mask-after-the-fact trainer would have projected:
            # every non-pad position of the rectangle
            "baseline_projected": f.padded_tokens,
            "projection_reduction": round(
                f.padded_tokens / max(self.n_scored, 1), 3
            ),
        }


def compile_plan(
    forest: SegmentForest, device: torch.device | str = "cpu"
) -> ExecutionPlan:
    """Lay the forest out in DFS pre-order and derive every index from it."""
    dev = torch.device(device)
    nodes = forest.nodes

    order: list[int] = []
    stack = list(reversed(forest.roots))
    while stack:
        n = stack.pop()
        order.append(n)
        stack.extend(reversed(nodes[n].children))

    # --- physical layout -------------------------------------------------
    # Everything below iterates over nodes, never over individual tokens, so
    # plan construction does not add a Python loop per token.
    start: dict[int, int] = {}
    end: dict[int, int] = {}
    pos_offset: dict[int, int] = {}
    tok_parts: list[torch.Tensor] = []
    pos_parts: list[torch.Tensor] = []
    nod_parts: list[torch.Tensor] = []
    cursor = 0

    for n in order:
        node = nodes[n]
        parent = node.parent
        pos_offset[n] = 0 if parent is None else pos_offset[parent] + len(nodes[parent].tokens)
        ln = len(node.tokens)
        start[n] = cursor
        cursor += ln
        end[n] = cursor
        if ln:
            tok_parts.append(torch.tensor(node.tokens, dtype=torch.long, device=dev))
            pos_parts.append(
                torch.arange(pos_offset[n], pos_offset[n] + ln, dtype=torch.long, device=dev)
            )
            nod_parts.append(torch.full((ln,), n, dtype=torch.long, device=dev))

    def _cat(parts, dtype=torch.long):
        return (
            torch.cat(parts)
            if parts
            else torch.zeros(0, dtype=dtype, device=dev)
        )

    tokens_t = _cat(tok_parts)
    positions_t = _cat(pos_parts)
    node_of_t = _cat(nod_parts)

    # DFS pre-order makes every node subtree a contiguous physical interval.
    # A token in node n is an ancestor of every later token in n's subtree,
    # including later tokens in n itself.  The exclusive subtree end therefore
    # gives the complete ancestry-causal mask with two comparisons:
    #
    #   key is visible to query <=> key <= query < subtree_end[key]
    #
    # No [P,P] mask or node-ancestor table is required by the physical-tree
    # attention backend.
    subtree_node_end: dict[int, int] = {}
    for n in reversed(order):
        subtree_node_end[n] = max(
            [end[n], *(subtree_node_end[c] for c in nodes[n].children)]
        )
    subtree_parts = [
        torch.full(
            (end[n] - start[n],),
            subtree_node_end[n],
            dtype=torch.long,
            device=dev,
        )
        for n in order
        if end[n] > start[n]
    ]
    subtree_end_t = _cat(subtree_parts)

    # --- attention visibility -------------------------------------------
    # Pre-order guarantees a parent is laid out before its children, so its own
    # spans are final by the time a child reads them. Empty spans (a zero-length
    # grouping root, when trajectories diverge at token 0) are dropped rather
    # than passed to a kernel.
    key_ranges: list[list[tuple[int, int]]] = [[] for _ in nodes]
    for n in order:
        parent = nodes[n].parent
        if parent is None:
            inherited: list[tuple[int, int]] = []
        else:
            inherited = key_ranges[parent][:-1] + [(start[parent], end[parent])]
        own = (start[n], end[n])
        key_ranges[n] = [(s, e) for s, e in inherited if e > s] + [own]

    # --- scoring ---------------------------------------------------------
    # A physical position is scored if its node marks it so. The hidden state
    # that predicts it is its predecessor's: the previous physical token inside
    # the same node, or the parent's last token at a node boundary. A root's
    # first token has no predecessor and is never scored.
    #
    # Slots are handed out in `order`, so each node's slots form a contiguous
    # range -- which is what lets the logical view below stay per-node too.
    src_parts: list[torch.Tensor] = []
    tgt_parts: list[torch.Tensor] = []
    pos_parts2: list[torch.Tensor] = []
    slot_start: dict[int, int] = {}
    slot_count: dict[int, int] = {}
    n_slots = 0

    for n in order:
        node = nodes[n]
        parent = node.parent
        slot_start[n] = n_slots
        slot_count[n] = 0
        if not node.tokens:
            continue
        sc = torch.tensor(node.scored, dtype=torch.bool, device=dev)
        if parent is None:
            sc[0] = False  # first token of a root: nothing predicts it
        j = sc.nonzero(as_tuple=True)[0]
        if j.numel() == 0:
            continue
        p = j + start[n]
        # predecessor: p-1 inside the node, parent's last token at j == 0
        src = p - 1
        if parent is not None:
            src = torch.where(j == 0, torch.full_like(p, end[parent] - 1), src)
        src_parts.append(src)
        pos_parts2.append(p)
        tgt_parts.append(
            torch.tensor(node.tokens, dtype=torch.long, device=dev).index_select(0, j)
        )
        slot_count[n] = int(j.numel())
        n_slots += slot_count[n]

    # --- logical view ----------------------------------------------------
    # For each trajectory, its scored slots are the concatenation of the slot
    # ranges of the nodes on its path. Trajectories sharing a node share slots;
    # autograd turns that into a gradient sum -- the tree reduction, obtained
    # without touching anyone's coefficients.
    l2s_parts: list[torch.Tensor] = []
    ltraj_parts: list[torch.Tensor] = []
    for traj in forest.trajectories:
        leaf = forest.leaf_of[traj.traj_id]
        total = 0
        for n in forest.path_to_root(leaf):
            c = slot_count.get(n, 0)
            if c:
                l2s_parts.append(
                    torch.arange(slot_start[n], slot_start[n] + c, dtype=torch.long, device=dev)
                )
                total += c
        if total:
            ltraj_parts.append(
                torch.full((total,), traj.traj_id, dtype=torch.long, device=dev)
            )

    def T(x, dtype=torch.long):
        return torch.tensor(x, dtype=dtype, device=dev)

    return ExecutionPlan(
        token_ids=tokens_t,
        position_ids=positions_t,
        node_of=node_of_t,
        node_start=T([start[n] for n in range(len(nodes))]),
        node_end=T([end[n] for n in range(len(nodes))]),
        node_parent=T([-1 if nodes[n].parent is None else nodes[n].parent for n in range(len(nodes))]),
        node_depth=T([nodes[n].depth for n in range(len(nodes))]),
        subtree_end=subtree_end_t,
        score_src=_cat(src_parts),
        score_tgt=_cat(tgt_parts),
        score_pos=_cat(pos_parts2),
        logical_to_slot=_cat(l2s_parts),
        logical_traj=_cat(ltraj_parts),
        key_ranges=key_ranges,
        forest=forest,
    )


def visibility_mask(plan: ExecutionPlan) -> torch.Tensor:
    """`[P, P]` bool mask implied by the plan. Reference only.

    Materialising the quadratic mask defeats the forest representation. This
    helper exists so the fast kernels in `attention.py` can be checked against
    a direct reference and must never appear on a training path.
    """
    p = plan.n_physical
    mask = torch.zeros(p, p, dtype=torch.bool, device=plan.token_ids.device)
    for n in range(plan.n_nodes):
        s, e = int(plan.node_start[n]), int(plan.node_end[n])
        if s == e:
            continue
        spans = plan.key_ranges[n]
        for ks, ke in spans[:-1]:
            mask[s:e, ks:ke] = True  # ancestors: fully visible
        q = torch.arange(s, e, device=mask.device).unsqueeze(1)
        k = torch.arange(s, e, device=mask.device).unsqueeze(0)
        mask[s:e, s:e] = k <= q      # own span: causal
    return mask
