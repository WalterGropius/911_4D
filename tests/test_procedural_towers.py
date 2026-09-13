import pytest

from wtc4d import timeline
from wtc4d.procedural import towers


@pytest.mark.parametrize("tower_id", ["WTC1", "WTC2"])
def test_intact_before_impact(tower_id):
    st = towers.tower_state(tower_id, 0.0)
    assert st.phase == "intact"
    assert st.fire_extent == 0.0
    assert st.collapse_progress == 0.0
    assert st.damage is None
    assert not st.remnant_present


def test_wtc1_burning_after_impact_before_wtc2_impact():
    st = towers.tower_state("WTC1", timeline.WTC1_IMPACT.t + 1.0)
    assert st.phase == "burning"
    assert st.damage is not None
    assert st.damage.face == "north"
    assert 0.0 <= st.fire_extent < 0.01

    st2 = towers.tower_state("WTC2", timeline.WTC1_IMPACT.t + 1.0)
    assert st2.phase == "intact"


def test_wtc2_burning_after_its_own_impact():
    st = towers.tower_state("WTC2", timeline.WTC2_IMPACT.t + 1.0)
    assert st.phase == "burning"
    assert st.damage.face == "south"
    # entering near the south-east corner: centred toward the east half of the face
    assert st.damage.center_frac > 0.5


def test_fire_extent_is_monotonic_and_bounded():
    t0 = timeline.WTC1_IMPACT.t
    ts = [t0 + 60, t0 + 600, t0 + 3600, t0 + 6000]
    values = [towers.tower_state("WTC1", t).fire_extent for t in ts]
    assert all(0.0 <= v <= 1.0 for v in values)
    assert values == sorted(values)


def test_wtc2_collapse_progress_ramps_and_tilts():
    t_start = timeline.WTC2_COLLAPSE.t
    st_start = towers.tower_state("WTC2", t_start + 0.01)
    assert st_start.phase == "collapsing"
    assert st_start.collapse_progress < 0.1
    assert st_start.tilt_deg >= 0.0

    st_mid = towers.tower_state("WTC2", t_start + 5.0)
    assert st_mid.tilt_deg > st_start.tilt_deg
    assert st_mid.top_height_m < st_start.top_height_m


def test_wtc1_collapse_no_tilt_and_antenna_drops_first():
    t_start = timeline.WTC1_COLLAPSE.t
    st_pre = towers.tower_state("WTC1", t_start - 1.0)
    assert st_pre.antenna_present

    st_early = towers.tower_state("WTC1", t_start + 0.1)
    assert st_early.tilt_deg == 0.0
    assert st_early.antenna_present  # antenna drop has a short lead time

    st_later = towers.tower_state("WTC1", t_start + 5.0)
    assert not st_later.antenna_present


@pytest.mark.parametrize("tower_id", ["WTC1", "WTC2"])
def test_tower_is_gone_well_after_collapse(tower_id):
    collapse_t = timeline.WTC1_COLLAPSE.t if tower_id == "WTC1" else timeline.WTC2_COLLAPSE.t
    st = towers.tower_state(tower_id, collapse_t + 120.0)
    assert st.phase == "gone"
    assert st.collapse_progress == 1.0
    assert st.top_height_m == 0.0
    assert st.rubble_height_m > 0.0


def test_wtc1_remnant_present_only_after_wtc1_is_gone():
    st_gone_wtc2_only = towers.tower_state("WTC1", timeline.WTC2_COLLAPSE.t + 60.0)
    assert st_gone_wtc2_only.phase != "gone"
    assert not st_gone_wtc2_only.remnant_present

    st_wtc1_gone = towers.tower_state("WTC1", timeline.WTC1_COLLAPSE.t + 60.0)
    assert st_wtc1_gone.remnant_present
    assert st_wtc1_gone.remnant_height_m > 0.0

    st_wtc2_gone = towers.tower_state("WTC2", timeline.WTC2_COLLAPSE.t + 60.0)
    assert not st_wtc2_gone.remnant_present


def test_epoch_boundaries_are_continuous():
    """State just before and just after each anchor time should not jump
    discontinuously in the continuous fractions."""
    for event in (
        timeline.WTC1_IMPACT,
        timeline.WTC2_IMPACT,
        timeline.WTC1_COLLAPSE,
        timeline.WTC2_COLLAPSE,
    ):
        for tower_id in towers.TOWER_IDS:
            before = towers.tower_state(tower_id, event.t - 0.05)
            after = towers.tower_state(tower_id, event.t + 0.05)
            assert abs(after.fire_extent - before.fire_extent) < 0.05
            assert abs(after.collapse_progress - before.collapse_progress) < 0.05


def test_all_tower_states():
    states = towers.all_tower_states(timeline.hms(9, 30, 0))
    assert set(states) == {"WTC1", "WTC2"}
    assert all(s.phase == "burning" for s in states.values())


def test_wtc7_state_machine():
    st_before = towers.wtc7_state(timeline.hms(10, 0, 0))
    assert st_before.phase == "intact"

    st_damaged = towers.wtc7_state(timeline.WTC1_COLLAPSE.t + 60.0)
    assert st_damaged.phase == "damaged"

    st_collapsing = towers.wtc7_state(timeline.WTC7_COLLAPSE.t + 1.0)
    assert st_collapsing.phase == "collapsing"

    st_gone = towers.wtc7_state(timeline.WTC7_COLLAPSE.t + 120.0)
    assert st_gone.phase == "gone"


def test_three_wtc_state_machine():
    st_before = towers.three_wtc_state(timeline.WTC2_COLLAPSE.t - 1.0)
    assert st_before.phase == "intact"

    st_after = towers.three_wtc_state(timeline.WTC2_COLLAPSE.t + 200.0)
    assert st_after.phase == "gone"
