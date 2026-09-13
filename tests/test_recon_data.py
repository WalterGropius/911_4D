"""Dataset: frame lookup, masks, times, filtering, splits, export."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from wtc4d import timeline
from wtc4d.recon.conventions import look_at_c2w
from wtc4d.recon.data import (
    ReconDataset,
    load_poses_jsonl,
    load_time_estimates,
    save_poses_jsonl,
)
from wtc4d.schema.camera import CameraIntrinsics, CameraPose

pytest.importorskip("PIL")


def _write_image(path, width, height, value):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((height, width, 3), value, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _make_scene(tmp_path, n_shots=2, n_frames=3, width=16, height=12):
    poses = []
    for s in range(n_shots):
        intr = CameraIntrinsics(
            width=width, height=height, fx=20.0, fy=20.0, cx=width / 2, cy=height / 2
        )
        for f in range(n_frames):
            c2w = look_at_c2w([600.0 + 50 * s, 100.0 * f, 220.0], [0.0, 0.0, 200.0])
            poses.append(
                CameraPose.from_matrix(
                    c2w, shot_id=f"shot{s}", frame_idx=f, intrinsics=intr, method="landmark_pnp"
                )
            )
            _write_image(
                tmp_path / "frames" / f"shot{s}" / f"{f:06d}.jpg", width, height, 40 * (f + 1)
            )
            _write_image(
                tmp_path / "masks" / f"shot{s}" / f"{f:06d}.png",
                width,
                height,
                255 if f == 1 else 0,
            )
    save_poses_jsonl(tmp_path / "poses.jsonl", poses)

    with (tmp_path / "times.jsonl").open("w") as fh:
        for s in range(n_shots):
            for f in range(n_frames):
                fh.write(
                    json.dumps(
                        {
                            "shot_id": f"shot{s}",
                            "frame_idx": f,
                            "t": timeline.WTC2_IMPACT.t + 60.0 * f + 5.0 * s,
                            "sigma": 0.8,
                            "method": "onscreen_clock",
                        }
                    )
                    + "\n"
                )
    return poses


def test_poses_jsonl_round_trip(tmp_path):
    poses = _make_scene(tmp_path)
    back = load_poses_jsonl(tmp_path / "poses.jsonl")
    assert len(back) == len(poses)
    assert np.allclose(back[0].matrix(), poses[0].matrix())
    assert back[0].method == "landmark_pnp"


def test_dataset_loads_images_masks_and_times(tmp_path):
    _make_scene(tmp_path)
    ds = ReconDataset.from_paths(
        tmp_path / "poses.jsonl",
        tmp_path / "frames",
        tmp_path / "masks",
        tmp_path / "times.jsonl",
    )
    assert len(ds) == 6
    sample = ds[1]
    assert sample.image.shape == (12, 16, 3)
    assert 0.0 <= float(sample.image.min()) and float(sample.image.max()) <= 1.0
    assert sample.dynamic_mask.shape == (12, 16, 1)
    assert float(sample.dynamic_mask.mean()) == pytest.approx(
        1.0, abs=1e-3
    )  # frame 1 is "all smoke"
    assert torch.allclose(sample.static_weight, torch.zeros_like(sample.static_weight), atol=1e-3)
    assert sample.t == pytest.approx(timeline.WTC2_IMPACT.t + 60.0)
    assert sample.t_sigma == pytest.approx(0.8)
    assert sample.c2w.shape == (4, 4)


def test_mask_polarity_can_be_inverted(tmp_path):
    _make_scene(tmp_path)
    ds = ReconDataset.from_paths(
        tmp_path / "poses.jsonl", tmp_path / "frames", tmp_path / "masks", mask_is_static=True
    )
    assert float(ds[1].dynamic_mask.mean()) == pytest.approx(0.0, abs=1e-3)


def test_downscale_scales_images_and_intrinsics(tmp_path):
    _make_scene(tmp_path, width=32, height=24)
    ds = ReconDataset.from_paths(tmp_path / "poses.jsonl", tmp_path / "frames", downscale=2.0)
    s = ds[0]
    assert (s.width, s.height) == (16, 12)
    assert s.image.shape == (12, 16, 3)
    assert s.intrinsics.fx == pytest.approx(10.0)


def test_missing_frames_directory_yields_poses_only(tmp_path):
    _make_scene(tmp_path)
    ds = ReconDataset.from_paths(tmp_path / "poses.jsonl")
    assert ds[0].image is None and ds[0].dynamic_mask is None


def test_time_filtering_and_epoch_filtering(tmp_path):
    _make_scene(tmp_path)
    ds = ReconDataset.from_paths(
        tmp_path / "poses.jsonl", tmp_path / "frames", times_path=tmp_path / "times.jsonl"
    )
    t0 = timeline.WTC2_IMPACT.t
    assert len(ds.filter_time(t0 - 1, t0 + 1)) == 1  # only shot0 frame0 is exactly at the impact
    assert len(ds.filter_epoch("E2")) == 6
    assert len(ds.filter_epoch("E0")) == 0
    assert len(ds.filter_epoch("E0", keep_untimed=True)) == 0
    assert ds.time_range()[0] == pytest.approx(t0)


def test_untimed_frames_can_be_kept_or_dropped(tmp_path):
    _make_scene(tmp_path)
    ds = ReconDataset.from_paths(tmp_path / "poses.jsonl", tmp_path / "frames")  # no times
    assert len(ds.filter_epoch("E2")) == 0
    assert len(ds.filter_epoch("E2", keep_untimed=True)) == 6


def test_split_holds_out_strided_frames_covering_every_shot(tmp_path):
    _make_scene(tmp_path, n_shots=2, n_frames=4)
    ds = ReconDataset.from_paths(tmp_path / "poses.jsonl", tmp_path / "frames")
    train, val = ds.split(holdout_every=4)
    assert len(train) == 6 and len(val) == 2
    assert set(val.shots()) == {"shot0", "shot1"}
    assert [s.index for s in val] == [0, 1]  # re-indexed for the subset


def test_export_colmap_and_transforms_round_trip_through_dataset(tmp_path):
    _make_scene(tmp_path)
    ds = ReconDataset.from_paths(
        tmp_path / "poses.jsonl", tmp_path / "frames", times_path=tmp_path / "times.jsonl"
    )
    ds.export_colmap(tmp_path / "sparse")
    ds.export_transforms_json(tmp_path / "transforms.json")

    from_colmap = ReconDataset.from_colmap(tmp_path / "sparse", tmp_path / "frames")
    assert len(from_colmap) == len(ds)
    assert np.allclose(from_colmap[0].c2w.numpy(), ds[0].c2w.numpy(), atol=1e-5)
    assert from_colmap[0].image is not None  # names resolve back to the frame store

    from_ns = ReconDataset.from_transforms_json(tmp_path / "transforms.json", tmp_path / "frames")
    assert np.allclose(from_ns[0].c2w.numpy(), ds[0].c2w.numpy(), atol=1e-5)
    assert from_ns.times[("shot0", 0)].t == pytest.approx(timeline.WTC2_IMPACT.t)


def test_load_time_estimates_accepts_nested_payloads(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text(
        json.dumps({"shot_id": "a", "frame_idx": 2, "estimate": {"t": 1.0, "sigma": 0.5}})
        + "\n"
        + json.dumps({"shot": "b", "frame": 0, "t": 2.0, "sigma": 0.25, "method": "audio_xcorr"})
        + "\n"
    )
    times = load_time_estimates(path)
    assert times[("a", 2)].t == pytest.approx(1.0)
    assert times[("b", 0)].method == "audio_xcorr"
