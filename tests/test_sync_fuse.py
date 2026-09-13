import json

from wtc4d.schema.time import TimeEstimate, TimeMethod
from wtc4d.sync import fuse as F
from wtc4d.sync.types import PairwiseOffset, ShotTimeRecord
from wtc4d.timeline import hms


def _est(t, sigma, method=TimeMethod.BROADCAST_METADATA):
    return TimeEstimate(t=t, sigma=sigma, method=method)


# --- fuse_estimates ------------------------------------------------------


def test_fuse_estimates_weighted_mean_no_outliers():
    ests = [_est(100.0, 1.0), _est(100.5, 0.5), _est(99.7, 2.0)]
    r = F.fuse_estimates(ests)
    w = [1.0, 4.0, 0.25]
    t = [100.0, 100.5, 99.7]
    expected = sum(wi * ti for wi, ti in zip(w, t, strict=True)) / sum(w)
    assert r.estimate is not None
    assert abs(r.estimate.t - expected) < 1e-9
    assert len(r.accepted) == 3
    assert r.rejected == []
    assert r.dof == 2


def test_fuse_estimates_rejects_clear_outlier():
    ests = [_est(100.0, 1.0), _est(100.5, 0.5), _est(99.7, 2.0), _est(250.0, 1.0)]
    r = F.fuse_estimates(ests)
    assert len(r.rejected) == 1
    assert r.rejected[0].t == 250.0
    assert len(r.accepted) == 3


def test_fuse_estimates_single_estimate_passthrough():
    r = F.fuse_estimates([_est(500.0, 3.0, TimeMethod.ONSCREEN_CLOCK)])
    assert r.estimate is not None
    assert r.estimate.t == 500.0
    assert r.estimate.sigma == 3.0
    assert r.estimate.method == TimeMethod.ONSCREEN_CLOCK


def test_fuse_estimates_empty_returns_none():
    r = F.fuse_estimates([])
    assert r.estimate is None
    assert r.accepted == []


def test_fuse_estimates_sigma_shrinks_with_more_agreeing_cues():
    r1 = F.fuse_estimates([_est(100.0, 1.0)])
    r2 = F.fuse_estimates([_est(100.0, 1.0), _est(100.1, 1.0)])
    assert r2.estimate.sigma < r1.estimate.sigma


# --- solve_graph -----------------------------------------------------------


def test_solve_graph_chain_propagates_uncertainty():
    priors = {"A": _est(1000.0, 1.0)}
    offsets = [
        PairwiseOffset(a="A", b="B", dt=10.0, sigma=0.5, method=TimeMethod.AUDIO_XCORR),
        PairwiseOffset(a="B", b="C", dt=5.0, sigma=0.5, method=TimeMethod.AUDIO_XCORR),
    ]
    res = F.solve_graph(priors, offsets)
    # A tiny ridge term (see _solve_component) is added for numerical
    # stability, so exact equality isn't expected -- just agreement to well
    # within any meaningful sigma here (smallest is ~1.0 s).
    assert abs(res.estimates["A"].t - 1000.0) < 1e-4
    assert abs(res.estimates["B"].t - 1010.0) < 1e-4
    assert abs(res.estimates["C"].t - 1015.0) < 1e-4
    assert res.estimates["A"].sigma < res.estimates["B"].sigma < res.estimates["C"].sigma
    assert res.unresolved == []
    assert res.rejected_edges == []


def test_solve_graph_shot_without_anchor_or_edge_is_absent():
    priors = {"A": _est(1000.0, 1.0)}
    offsets = [PairwiseOffset(a="A", b="B", dt=1.0, sigma=1.0, method=TimeMethod.AUDIO_XCORR)]
    res = F.solve_graph(priors, offsets)
    assert "A" in res.estimates and "B" in res.estimates
    assert "C" not in res.estimates  # never mentioned anywhere


def test_solve_graph_disconnected_component_without_anchor_is_unresolved():
    priors = {"A": _est(1000.0, 1.0)}
    offsets = [
        PairwiseOffset(a="A", b="B", dt=1.0, sigma=1.0, method=TimeMethod.AUDIO_XCORR),
        PairwiseOffset(a="D", b="E", dt=3.0, sigma=1.0, method=TimeMethod.AUDIO_XCORR),
    ]
    res = F.solve_graph(priors, offsets)
    assert set(res.unresolved) == {"D", "E"}
    assert "D" not in res.estimates and "E" not in res.estimates


def test_solve_graph_rejects_outlier_edge_without_corrupting_the_rest():
    priors = {"A": _est(1000.0, 1.0)}
    offsets = [
        PairwiseOffset(a="A", b="B", dt=10.0, sigma=0.5, method=TimeMethod.AUDIO_XCORR),
        PairwiseOffset(a="B", b="C", dt=5.0, sigma=0.5, method=TimeMethod.AUDIO_XCORR),
        PairwiseOffset(a="A", b="C", dt=500.0, sigma=0.5, method=TimeMethod.AUDIO_XCORR),  # wrong
    ]
    res = F.solve_graph(priors, offsets)
    assert len(res.rejected_edges) == 1
    assert (res.rejected_edges[0].a, res.rejected_edges[0].b) == ("A", "C")
    assert abs(res.estimates["C"].t - 1015.0) < 1.0


def test_solve_graph_two_anchors_agree_reduces_sigma():
    priors = {"A": _est(1000.0, 1.0), "B": _est(1010.0, 1.0)}
    offsets = [PairwiseOffset(a="A", b="B", dt=10.0, sigma=0.1, method=TimeMethod.AUDIO_XCORR)]
    res = F.solve_graph(priors, offsets)
    # both anchors and the edge agree closely; the fused sigma should still
    # be tighter than either anchor's own sigma alone
    assert res.estimates["A"].sigma < 1.0
    assert res.estimates["B"].sigma < 1.0


def test_solve_graph_two_disagreeing_anchors_rejects_the_edge():
    priors = {"A": _est(1000.0, 0.5), "B": _est(2000.0, 0.5)}
    offsets = [PairwiseOffset(a="A", b="B", dt=10.0, sigma=0.5, method=TimeMethod.AUDIO_XCORR)]
    res = F.solve_graph(priors, offsets)
    assert len(res.rejected_edges) == 1
    assert abs(res.estimates["A"].t - 1000.0) < 0.1
    assert abs(res.estimates["B"].t - 2000.0) < 0.1


# --- I/O and reporting -----------------------------------------------------


def test_write_read_records_roundtrip(tmp_path):
    records = [
        ShotTimeRecord(shot_id="s1", fps=29.97, time=_est(hms(9, 0, 0), 1.0)),
        ShotTimeRecord(shot_id="s2", fps=29.97, time=None),
    ]
    p = tmp_path / "time_estimates.jsonl"
    F.write_records(records, p)
    lines = p.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["shot_id"] == "s1"

    back = F.read_records(p)
    assert len(back) == 2
    assert back[0].shot_id == "s1" and back[0].time is not None
    assert back[1].shot_id == "s2" and back[1].time is None


def test_read_records_missing_file_returns_empty(tmp_path):
    assert F.read_records(tmp_path / "nope.jsonl") == []


def test_coverage_report_buckets_by_epoch():
    records = [
        ShotTimeRecord(
            shot_id="s1", fps=29.97, time=_est(hms(8, 50, 0), 1.0, TimeMethod.BROADCAST_METADATA)
        ),
        ShotTimeRecord(
            shot_id="s2", fps=29.97, time=_est(hms(9, 10, 0), 0.5, TimeMethod.ONSCREEN_CLOCK)
        ),
        ShotTimeRecord(shot_id="s3", fps=29.97, time=None),
    ]
    rep = F.coverage_report(records)
    assert rep.total_shots == 3
    assert rep.n_timed == 2
    assert rep.n_untimed == 1
    by_id = {e.epoch_id: e for e in rep.epochs}
    assert by_id["E1"].n_shots == 1
    assert by_id["E2"].n_shots == 1
    assert by_id["E0"].n_shots == 0
    assert by_id["E1"].methods == {"broadcast_metadata": 1}
    rendered = rep.render()
    assert "2/3 shots timed" in rendered


def test_coverage_report_empty_records():
    rep = F.coverage_report([])
    assert rep.total_shots == 0
    assert rep.n_timed == 0
    assert all(e.n_shots == 0 for e in rep.epochs)
