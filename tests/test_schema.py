import json

import numpy as np
import pytest

from wtc4d.schema import (
    CameraIntrinsics,
    CameraPose,
    SceneManifest,
    Shot,
    Source,
    SourceKind,
    TimeEstimate,
)
from wtc4d.world import WORLD_ORIGIN


def test_source_roundtrip():
    s = Source(
        id="ia-test",
        kind=SourceKind.TV_BROADCAST,
        url="https://archive.org/details/test",
        archive="archive.org",
    )
    s2 = Source.model_validate_json(s.model_dump_json())
    assert s2 == s


def test_shot_times():
    sh = Shot(id="s", source_id="x", start_frame=30, end_frame=90, fps=29.97)
    assert abs(sh.start_s - 1.001) < 1e-3
    assert sh.end_s > sh.start_s


def test_camera_pose_matrix():
    K = CameraIntrinsics(width=720, height=480, fx=800, fy=800, cx=360, cy=240)
    m = np.eye(4)
    m[:3, 3] = [1.0, 2.0, 3.0]
    p = CameraPose.from_matrix(m, shot_id="s", frame_idx=0, intrinsics=K)
    assert np.allclose(p.position(), [1, 2, 3])
    assert 40 < K.hfov_deg < 50
    bad = m.copy()
    bad[3, 3] = 2.0
    with pytest.raises(ValueError):
        CameraPose.from_matrix(bad, shot_id="s", frame_idx=0, intrinsics=K)


def test_scene_manifest_json():
    man = SceneManifest(world_origin=WORLD_ORIGIN, t_min=0, t_max=1)
    d = json.loads(man.model_dump_json())
    assert d["schema_version"] == 1
    TimeEstimate(t=31590.0, sigma=5.0)
