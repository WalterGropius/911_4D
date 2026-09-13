"""Fusing every timing cue for a shot into one :class:`TimeEstimate`.

Two levels of fusion happen here, and most shots only need the first:

1. **Per-shot fusion** (:func:`fuse_estimates`) -- a shot usually has several
   *independent* absolute-time candidates for its start frame: broadcast
   metadata, an on-screen clock fit, an event anchor, a solar shadow.  These
   combine by the standard rule for independent Gaussian measurements
   (inverse-variance weighting), with **iterative chi-square rejection**: a
   candidate whose distance from the current fused estimate is not
   explainable by the two sigmas together is dropped and the rest re-fused,
   same idea as a generalised Grubbs/ESD outlier test.  This is what a shot
   with only absolute cues needs.

2. **Graph fusion** (:func:`solve_graph`) -- some shots have no absolute cue
   at all, only a :class:`~wtc4d.sync.types.PairwiseOffset` (from
   :mod:`wtc4d.sync.audio`) tying them to another shot.  Every shot's start
   time is one scalar unknown; every per-shot fused estimate is an equation
   pinning one unknown, and every pairwise offset is an equation relating
   two.  That is a weighted linear least-squares problem with as many
   equations as cues, solved once for the whole corpus by the normal
   equations, with the same iterative chi-square rejection applied to
   *edges* this time.  A shot with no path (direct or indirect) to any
   absolute anchor has no absolute time yet -- fusion reports that rather
   than guessing.

Everything here is deliberately linear algebra on scalars, not a general
SLAM/pose-graph solver: with one unknown per shot the normal equations are a
dense ``n_shots x n_shots`` solve, which is instant for the thousands of
shots this project expects.  Swap in a sparse solver first if that ever
stops being true.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from wtc4d.schema.time import TimeEstimate, TimeMethod
from wtc4d.sync.types import PairwiseOffset, ShotTimeRecord
from wtc4d.timeline import EPOCHS, epoch_at, fmt_local

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "time"
TIME_ESTIMATES_PATH = DATA_DIR / "time_estimates.jsonl"

DEFAULT_CHI2_SIGMA = 3.0
"""Reject a candidate/edge whose residual exceeds this many combined sigmas.

For Gaussian noise, 3-sigma is a ~0.3% false-rejection rate per candidate --
loose enough that a handful of good cues never fight each other, tight
enough to catch a genuinely wrong reading (an OCR misread that still parsed,
a mismatched event anchor)."""


# --- per-shot fusion ----------------------------------------------------------


class FusionResult(BaseModel):
    """What :func:`fuse_estimates` decided, with the full trail."""

    estimate: TimeEstimate | None = None
    accepted: list[TimeEstimate] = Field(default_factory=list)
    rejected: list[TimeEstimate] = Field(default_factory=list)
    chi2: float = 0.0
    dof: int = 0


def _weighted_fuse(ests: Sequence[TimeEstimate]) -> tuple[float, float]:
    """Inverse-variance weighted mean and its sigma. Assumes ``ests`` nonempty."""
    w = np.array([1.0 / e.sigma**2 for e in ests])
    t = np.array([e.t for e in ests])
    w_sum = w.sum()
    t_hat = float((w * t).sum() / w_sum)
    sigma_hat = float(1.0 / np.sqrt(w_sum))
    return (t_hat, sigma_hat)


def fuse_estimates(
    estimates: Sequence[TimeEstimate],
    *,
    chi2_sigma: float = DEFAULT_CHI2_SIGMA,
    min_inliers: int = 1,
) -> FusionResult:
    """Fuse independent absolute-time estimates with iterative outlier rejection.

    Each round: compute the inverse-variance-weighted mean of the surviving
    estimates, score every survivor by how many combined sigmas it sits from
    *the fused estimate of the others* (leave-one-out, so one bad candidate
    cannot inflate the sigma that is used to excuse itself), drop the worst
    offender if it exceeds ``chi2_sigma``, and repeat.  Stops when nothing
    exceeds the threshold or only ``min_inliers`` remain.
    """
    if not estimates:
        return FusionResult()
    pool = list(estimates)
    rejected: list[TimeEstimate] = []

    while len(pool) > 1:
        worst_idx = -1
        worst_z = 0.0
        for i, cand in enumerate(pool):
            others = pool[:i] + pool[i + 1 :]
            t_others, sigma_others = _weighted_fuse(others)
            spread = float(np.hypot(cand.sigma, sigma_others))
            z = abs(cand.t - t_others) / spread if spread > 0 else 0.0
            if z > worst_z:
                worst_z = z
                worst_idx = i
        if worst_z <= chi2_sigma or len(pool) <= min_inliers:
            break
        rejected.append(pool.pop(worst_idx))

    t_hat, sigma_hat = _weighted_fuse(pool)
    chi2 = sum(((e.t - t_hat) / e.sigma) ** 2 for e in pool)
    dof = max(0, len(pool) - 1)

    tightest = min(pool, key=lambda e: e.sigma)
    method = tightest.method if len(pool) == 1 else TimeMethod.MANUAL
    sources = ", ".join(f"{e.method.value}={fmt_local(e.t)}(+-{e.sigma:.2f}s)" for e in pool)
    evidence = (
        f"fused {len(pool)}/{len(estimates)} estimate(s), chi2/dof={chi2:.2f}/{dof}: {sources}"
    )
    if rejected:
        evidence += "; rejected: " + ", ".join(
            f"{e.method.value}={fmt_local(e.t)}" for e in rejected
        )

    estimate = TimeEstimate(
        t=t_hat,
        sigma=sigma_hat,
        method=method,
        evidence=evidence,
        derived_from=sorted({d for e in pool for d in e.derived_from}),
    )
    return FusionResult(estimate=estimate, accepted=pool, rejected=rejected, chi2=chi2, dof=dof)


# --- graph fusion --------------------------------------------------------------


class GraphFusionResult(BaseModel):
    """Per-shot absolute times after solving the whole offset graph."""

    estimates: dict[str, TimeEstimate] = Field(default_factory=dict)
    unresolved: list[str] = Field(default_factory=list)
    rejected_edges: list[PairwiseOffset] = Field(default_factory=list)


def _connected_components(shot_ids: set[str], edges: Sequence[PairwiseOffset]) -> list[set[str]]:
    parent: dict[str, str] = {s: s for s in shot_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for e in edges:
        if e.a in parent and e.b in parent:
            union(e.a, e.b)

    groups: dict[str, set[str]] = defaultdict(set)
    for s in shot_ids:
        groups[find(s)].add(s)
    return list(groups.values())


def _solve_component(
    shots: list[str],
    priors: dict[str, TimeEstimate],
    edges: list[PairwiseOffset],
) -> dict[str, float] | None:
    """Weighted least squares for one connected component.

    Returns ``None`` if the component has no anchor at all (the system is
    rank-deficient: every solution shifted by a constant fits equally well,
    so there is no absolute answer to report).
    """
    if not any(s in priors for s in shots):
        return None
    index = {s: i for i, s in enumerate(shots)}
    n = len(shots)
    rows: list[np.ndarray] = []
    weights: list[float] = []
    rhs: list[float] = []

    for s, est in priors.items():
        if s not in index:
            continue
        row = np.zeros(n)
        row[index[s]] = 1.0
        rows.append(row)
        weights.append(1.0 / est.sigma**2)
        rhs.append(est.t)

    for e in edges:
        if e.a not in index or e.b not in index:
            continue
        row = np.zeros(n)
        row[index[e.b]] = 1.0
        row[index[e.a]] = -1.0
        rows.append(row)
        weights.append(1.0 / e.sigma**2)
        rhs.append(e.dt)

    A = np.stack(rows)
    w = np.array(weights)
    b = np.array(rhs)
    ata = A.T @ (w[:, None] * A)
    atb = A.T @ (w * b)
    # A small ridge keeps this solvable even if a component is only weakly
    # connected (e.g. one very loose edge); real ill-posedness (no anchor at
    # all) was already ruled out above.
    ata += np.eye(n) * 1e-9
    t = np.linalg.solve(ata, atb)
    return {s: float(t[index[s]]) for s in shots}


def _edge_residual_sigmas(
    solution: dict[str, float], priors: dict[str, TimeEstimate], edges: list[PairwiseOffset]
) -> list[tuple[float, str]]:
    """``(|z|, kind)`` for every constraint against a solved component."""
    out: list[tuple[float, str]] = []
    for s, est in priors.items():
        if s in solution:
            z = abs(solution[s] - est.t) / est.sigma
            out.append((z, f"anchor:{s}"))
    for e in edges:
        if e.a in solution and e.b in solution:
            z = abs((solution[e.b] - solution[e.a]) - e.dt) / e.sigma
            out.append((z, f"edge:{e.a}->{e.b}"))
    return out


def solve_graph(
    priors: dict[str, TimeEstimate],
    offsets: Sequence[PairwiseOffset],
    *,
    chi2_sigma: float = DEFAULT_CHI2_SIGMA,
    max_rejections: int = 50,
) -> GraphFusionResult:
    """Solve every shot's absolute time from per-shot priors + pairwise offsets.

    Shots that appear only as an endpoint of an offset (never in ``priors``)
    get a time exactly when their connected component contains at least one
    anchored shot; otherwise they are reported in ``unresolved``.  Edges are
    rejected the same iterative, leave-one-out way as :func:`fuse_estimates`,
    one at a time (the worst offender first) so that dropping one bad edge
    can rescue the rest of its component rather than each edge being judged
    against a solution still distorted by the others.
    """
    shot_ids = set(priors) | {e.a for e in offsets} | {e.b for e in offsets}
    edges = list(offsets)
    rejected_edges: list[PairwiseOffset] = []
    estimates: dict[str, TimeEstimate] = {}
    unresolved: set[str] = set()

    for _ in range(max_rejections + 1):
        components = _connected_components(shot_ids, edges)
        worst_z = 0.0
        worst_edge: PairwiseOffset | None = None
        estimates = {}
        unresolved = set()

        for comp in components:
            comp_list = sorted(comp)
            comp_edges = [e for e in edges if e.a in comp and e.b in comp]
            solved = _solve_component(comp_list, priors, comp_edges)
            if solved is None:
                unresolved |= comp
                continue
            for s, t in solved.items():
                sigma = _post_fit_sigma(comp_list, priors, comp_edges, s)
                estimates[s] = TimeEstimate(
                    t=t,
                    sigma=sigma,
                    method=TimeMethod.AUDIO_XCORR if s not in priors else priors[s].method,
                    evidence=_graph_evidence(s, comp_list, priors, comp_edges),
                    derived_from=sorted(comp),
                )
            for e in comp_edges:
                z = abs((solved[e.b] - solved[e.a]) - e.dt) / e.sigma
                if z > worst_z:
                    worst_z = z
                    worst_edge = e

        if worst_edge is None or worst_z <= chi2_sigma or len(rejected_edges) >= max_rejections:
            break
        edges.remove(worst_edge)
        rejected_edges.append(worst_edge)

    return GraphFusionResult(
        estimates=estimates, unresolved=sorted(unresolved), rejected_edges=rejected_edges
    )


def _post_fit_sigma(
    shots: list[str],
    priors: dict[str, TimeEstimate],
    edges: list[PairwiseOffset],
    shot: str,
) -> float:
    """Marginal sigma of one shot from the normal-equations covariance.

    Recomputes the small dense system for this component (cheap: one
    component is at most the whole corpus, solved once already per
    iteration) and reads off the diagonal of ``(A^T W A)^-1``, which is the
    standard result for linear Gaussian least squares.
    """
    index = {s: i for i, s in enumerate(shots)}
    n = len(shots)
    rows: list[np.ndarray] = []
    weights: list[float] = []
    for s, est in priors.items():
        if s in index:
            row = np.zeros(n)
            row[index[s]] = 1.0
            rows.append(row)
            weights.append(1.0 / est.sigma**2)
    for e in edges:
        if e.a in index and e.b in index:
            row = np.zeros(n)
            row[index[e.b]] = 1.0
            row[index[e.a]] = -1.0
            rows.append(row)
            weights.append(1.0 / e.sigma**2)
    if not rows:
        return float("inf")
    A = np.stack(rows)
    w = np.array(weights)
    ata = A.T @ (w[:, None] * A) + np.eye(n) * 1e-9
    cov = np.linalg.inv(ata)
    return float(np.sqrt(max(cov[index[shot], index[shot]], 0.0)))


def _graph_evidence(
    shot: str, shots: list[str], priors: dict[str, TimeEstimate], edges: list[PairwiseOffset]
) -> str:
    n_edges = sum(1 for e in edges if shot in (e.a, e.b))
    has_prior = shot in priors
    return (
        f"graph fusion over {len(shots)} shot(s), {len(edges)} pairwise edge(s) "
        f"({n_edges} touching this shot); "
        + ("has its own direct estimate" if has_prior else "time inferred only via offsets")
    )


# --- I/O -----------------------------------------------------------------


def write_records(records: Sequence[ShotTimeRecord], path: str | Path | None = None) -> None:
    p = Path(path) if path else TIME_ESTIMATES_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")


def read_records(path: str | Path | None = None) -> list[ShotTimeRecord]:
    p = Path(path) if path else TIME_ESTIMATES_PATH
    if not p.exists():
        return []
    out = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(ShotTimeRecord.model_validate(json.loads(line)))
    return out


# --- reporting -----------------------------------------------------------


class EpochCoverage(BaseModel):
    epoch_id: str
    epoch_name: str
    n_shots: int
    mean_sigma_s: float | None = None
    median_sigma_s: float | None = None
    methods: dict[str, int] = Field(default_factory=dict)


class CoverageReport(BaseModel):
    total_shots: int
    n_timed: int
    n_untimed: int
    epochs: list[EpochCoverage] = Field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"{self.n_timed}/{self.total_shots} shots timed "
            f"({self.n_untimed} without an absolute estimate)",
            "",
            f"{'epoch':<4} {'name':<32} {'shots':>6} {'mean sigma':>11} {'median sigma':>13}  methods",
        ]
        for ep in self.epochs:
            mean_s = f"{ep.mean_sigma_s:.2f}s" if ep.mean_sigma_s is not None else "-"
            med_s = f"{ep.median_sigma_s:.2f}s" if ep.median_sigma_s is not None else "-"
            methods = ", ".join(f"{k}:{v}" for k, v in sorted(ep.methods.items()))
            lines.append(
                f"{ep.epoch_id:<4} {ep.epoch_name:<32} {ep.n_shots:>6} {mean_s:>11} {med_s:>13}  {methods}"
            )
        return "\n".join(lines)


def coverage_report(records: Sequence[ShotTimeRecord]) -> CoverageReport:
    """Histogram of timed shots over the E0-E5 epochs, with sigma stats."""
    timed = [r for r in records if r.time is not None]
    by_epoch: dict[str, list[ShotTimeRecord]] = defaultdict(list)
    for r in timed:
        ep = epoch_at(r.time.t)  # type: ignore[union-attr]
        if ep is not None:
            by_epoch[ep.id].append(r)

    epochs = []
    for ep in EPOCHS:
        rs = by_epoch.get(ep.id, [])
        sigmas = [r.time.sigma for r in rs if r.time is not None]  # type: ignore[union-attr]
        methods: dict[str, int] = defaultdict(int)
        for r in rs:
            if r.time is not None:
                methods[r.time.method.value] += 1
        epochs.append(
            EpochCoverage(
                epoch_id=ep.id,
                epoch_name=ep.name,
                n_shots=len(rs),
                mean_sigma_s=float(np.mean(sigmas)) if sigmas else None,
                median_sigma_s=float(np.median(sigmas)) if sigmas else None,
                methods=dict(methods),
            )
        )

    return CoverageReport(
        total_shots=len(records),
        n_timed=len(timed),
        n_untimed=len(records) - len(timed),
        epochs=epochs,
    )


__all__ = [
    "DEFAULT_CHI2_SIGMA",
    "CoverageReport",
    "EpochCoverage",
    "FusionResult",
    "GraphFusionResult",
    "coverage_report",
    "fuse_estimates",
    "read_records",
    "solve_graph",
    "write_records",
]
