from wtc4d import timeline as tl


def test_anchor_formatting():
    assert tl.fmt_local(tl.WTC1_IMPACT.t) == "08:46:30"
    assert tl.fmt_local(tl.WTC1_COLLAPSE.t) == "10:28:22"


def test_utc_conversion():
    dt = tl.to_utc(tl.WTC2_IMPACT.t)
    assert (dt.hour, dt.minute, dt.second) == (13, 2, 59)
    assert abs(tl.from_datetime(dt) - tl.WTC2_IMPACT.t) < 1e-9


def test_epochs_are_contiguous():
    for a, b in zip(tl.EPOCHS, tl.EPOCHS[1:], strict=False):
        assert a.t_end == b.t_start
    assert tl.epoch_at(tl.hms(9, 30)).id == "E2"
    assert tl.epoch_at(tl.hms(10, 0)).id == "E3"
