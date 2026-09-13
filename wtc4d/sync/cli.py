"""``wtc4d sync`` -- CLI for the temporal-alignment workstream.

Every subcommand is a thin wrapper around the corresponding module and prints
its result as JSON (one object, or one object per line for a list) so it can
be piped into ``jq`` or into another step.  Commands that need heavy optional
deps (``rapidocr-onnxruntime``, ``internetarchive``) import them lazily, so
``wtc4d sync --help`` and the commands that do not need them always work.

See ``wtc4d/sync/README.md`` for the full pipeline and worked examples.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint
from rich.table import Table

app = typer.Typer(no_args_is_help=True, help="Temporal alignment: absolute time for shots/frames.")
broadcast_app = typer.Typer(no_args_is_help=True, help="TV-archive / file-metadata timing.")
clock_app = typer.Typer(no_args_is_help=True, help="On-screen clock detection and fitting.")
events_app = typer.Typer(no_args_is_help=True, help="Impact/collapse event-anchor detection.")
audio_app = typer.Typer(no_args_is_help=True, help="Audio cross-correlation alignment.")
solar_app = typer.Typer(no_args_is_help=True, help="Sun position / shadow timing.")
fuse_app = typer.Typer(no_args_is_help=True, help="Combine cues into fused TimeEstimates.")
app.add_typer(broadcast_app, name="broadcast")
app.add_typer(clock_app, name="clock")
app.add_typer(events_app, name="events")
app.add_typer(audio_app, name="audio")
app.add_typer(solar_app, name="solar")
app.add_typer(fuse_app, name="fuse")


def _print_model(m) -> None:  # noqa: ANN001 - pydantic model or plain dict
    # Plain print, not rich's: Rich word-wraps long lines to the detected
    # console width, which silently splits a JSON string value across lines
    # with a literal newline -- fine to look at, but it breaks any consumer
    # (jq, json.loads, a test) that re-parses the output. This is meant to be
    # machine-readable output, so it must never be reformatted.
    if hasattr(m, "model_dump"):
        print(json.dumps(m.model_dump(mode="json"), indent=2))
    else:
        print(json.dumps(m, indent=2, default=str))


def _parse_bbox(s: str):  # "x,y,w,h" -> BBox
    from wtc4d.sync.types import BBox

    x, y, w, h = (int(v) for v in s.split(","))
    return BBox(x=x, y=y, w=w, h=h)


# --- broadcast -----------------------------------------------------------


@broadcast_app.command("time")
def broadcast_time(
    identifier: Annotated[
        str, typer.Argument(help="archive.org identifier, or a path to cached metadata JSON")
    ],
    offset_s: Annotated[float, typer.Option(help="seconds into the media file")] = 0.0,
    cache_dir: Annotated[str | None, typer.Option(help="cache fetched metadata here")] = None,
    channel_delays: Annotated[str | None, typer.Option(help="path to channel_delays.yaml")] = None,
) -> None:
    """Absolute time at OFFSET_S into an archive.org TV-archive item."""
    from wtc4d.sync import broadcast as B

    if Path(identifier).exists():
        md = B.load_item_metadata(identifier)
    else:
        md = B.fetch_item_metadata(identifier, cache_dir=cache_dir)
    timing = B.ArchiveItemTiming.from_metadata(md)
    ok, problems = timing.consistent()
    if not ok:
        rprint(f"[yellow]warning: metadata consistency checks failed: {problems}[/yellow]")
    delays = B.load_channel_delays(channel_delays) if channel_delays else None
    est = timing.time_at_offset(offset_s, delays=delays)
    _print_model(est)


@broadcast_app.command("file-time")
def broadcast_file_time(
    path: Annotated[str, typer.Argument(help="a photo or video file")],
    source_id: Annotated[str | None, typer.Option()] = None,
    camera: Annotated[str | None, typer.Option(help="camera model, for the offset table")] = None,
) -> None:
    """Absolute time from a file's own EXIF/container metadata."""
    from wtc4d.sync import sources_time as ST

    res = ST.file_time(path, source_id=source_id, camera=camera)
    if res.ok:
        _print_model(res.estimate)
    else:
        rprint(f"[yellow]no usable time: {res.rejected_reason}[/yellow]")
        raise typer.Exit(code=1)


# --- clock -----------------------------------------------------------------


@clock_app.command("fit")
def clock_fit(
    media: Annotated[str, typer.Argument(help="video file or frames directory")],
    start_s: Annotated[float, typer.Option()] = 0.0,
    duration_s: Annotated[float | None, typer.Option()] = None,
    sample_fps: Annotated[
        float, typer.Option(help="OCR sampling rate once a clock region is found")
    ] = 2.0,
    fps: Annotated[
        float | None, typer.Option(help="required when MEDIA is a frames directory")
    ] = None,
    hint: Annotated[
        str | None, typer.Option(help="HH:MM[:SS] hint to resolve am/pm and 12h ambiguity")
    ] = None,
    region: Annotated[str | None, typer.Option(help="skip detection: 'x,y,w,h' clock crop")] = None,
) -> None:
    """Find and fit an on-screen clock in MEDIA."""
    from wtc4d.sync import clock_ocr as C
    from wtc4d.sync.video import FrameSource
    from wtc4d.timeline import hms

    hint_t = None
    if hint:
        h, m, *rest = (int(x) for x in hint.split(":"))
        hint_t = hms(h, m, rest[0] if rest else 0)

    src = FrameSource(media, fps=fps)
    box = _parse_bbox(region) if region else None
    fit, regions, readings = C.estimate_shot_time(
        src,
        start_s=start_s,
        duration_s=duration_s,
        sample_fps=sample_fps,
        hint_t=hint_t,
        region=box,
    )
    if fit is None:
        rprint(
            f"[yellow]no clock fit; found {len(regions)} candidate region(s), {len(readings)} reading(s)[/yellow]"
        )
        for r in regions[:5]:
            rprint(f"  region {r.bbox} score={r.score:.3f} texts={r.sample_texts}")
        raise typer.Exit(code=1)
    rprint(
        f"[green]fit from {fit.n_inliers}/{fit.n_readings} readings, band={fit.band_s:.3f}s[/green]"
    )
    _print_model(fit.estimate)


# --- events ------------------------------------------------------------------


@events_app.command("detect")
def events_detect(
    media: Annotated[str, typer.Argument()],
    roi: Annotated[str, typer.Option(help="'x,y,w,h' bounding box of the tower(s)")],
    start_s: Annotated[float, typer.Option()] = 0.0,
    duration_s: Annotated[float | None, typer.Option()] = None,
    kind: Annotated[str, typer.Option(help="impact | collapse | both")] = "both",
    fps: Annotated[float | None, typer.Option()] = None,
) -> None:
    """Detect impact-flash / collapse-onset candidates in MEDIA."""
    from wtc4d.sync import events as E
    from wtc4d.sync.types import RoiTrack
    from wtc4d.sync.video import FrameSource

    kinds = {
        "impact": (E.EventKind.IMPACT_FLASH,),
        "collapse": (E.EventKind.COLLAPSE_ONSET,),
        "both": (E.EventKind.IMPACT_FLASH, E.EventKind.COLLAPSE_ONSET),
    }[kind]
    src = FrameSource(media, fps=fps)
    track = RoiTrack.static(_parse_bbox(roi))
    cands, _ = E.detect_events(src, roi=track, start_s=start_s, duration_s=duration_s, kinds=kinds)
    if not cands:
        rprint("[yellow]no candidates found[/yellow]")
        raise typer.Exit(code=1)
    for c in cands:
        _print_model(c)


@events_app.command("anchor")
def events_anchor(
    media: Annotated[str, typer.Argument()],
    roi: Annotated[str, typer.Option()],
    prior_json: Annotated[
        str | None, typer.Option(help="TimeEstimate JSON to disambiguate impact 1 vs 2")
    ] = None,
    channel_delay_s: Annotated[float, typer.Option()] = 0.0,
    start_s: Annotated[float, typer.Option()] = 0.0,
    duration_s: Annotated[float | None, typer.Option()] = None,
) -> None:
    """Detect events in MEDIA and match them to NIST timeline anchors."""
    from wtc4d.schema.time import TimeEstimate
    from wtc4d.sync import events as E
    from wtc4d.sync.types import RoiTrack
    from wtc4d.sync.video import FrameSource

    src = FrameSource(media, fps=None)
    track = RoiTrack.static(_parse_bbox(roi))
    cands, _ = E.detect_events(src, roi=track, start_s=start_s, duration_s=duration_s)
    prior = (
        TimeEstimate.model_validate(json.loads(Path(prior_json).read_text()))
        if prior_json
        else None
    )
    ests = E.match_to_anchors(
        cands, prior=prior, media_offset_s=start_s, channel_delay_s=channel_delay_s
    )
    if not ests:
        rprint("[yellow]no candidates matched a timeline anchor[/yellow]")
        raise typer.Exit(code=1)
    for e in ests:
        _print_model(e)


# --- audio -------------------------------------------------------------------


@audio_app.command("align")
def audio_align(
    clip_a: Annotated[str, typer.Argument()],
    clip_b: Annotated[str, typer.Argument()],
    start_s: Annotated[float, typer.Option()] = 0.0,
    duration_s: Annotated[float | None, typer.Option()] = None,
    max_shift_s: Annotated[float | None, typer.Option()] = None,
    sample_rate: Annotated[int, typer.Option()] = 8000,
) -> None:
    """Align CLIP_A to CLIP_B by audio cross-correlation."""
    from wtc4d.sync import audio as A

    off = A.align_clips(
        clip_a,
        clip_b,
        a_id=clip_a,
        b_id=clip_b,
        sample_rate=sample_rate,
        start_s=start_s,
        duration_s=duration_s,
        max_shift_s=max_shift_s,
    )
    if off is None:
        rprint("[yellow]alignment too ambiguous to trust[/yellow]")
        raise typer.Exit(code=1)
    _print_model(off)


@audio_app.command("rumble")
def audio_rumble(
    media: Annotated[str, typer.Argument()],
    start_s: Annotated[float, typer.Option()] = 0.0,
    duration_s: Annotated[float | None, typer.Option()] = None,
    sample_rate: Annotated[int, typer.Option()] = 8000,
) -> None:
    """Detect a sustained low-frequency (collapse) rumble onset in MEDIA's audio."""
    from wtc4d.sync import audio as A
    from wtc4d.sync.video import load_audio

    x = load_audio(media, sample_rate=sample_rate, start_s=start_s, duration_s=duration_s)
    cands = A.detect_rumble_onset(x, sample_rate)
    if not cands:
        rprint("[yellow]no rumble onset found[/yellow]")
        raise typer.Exit(code=1)
    for c in cands:
        print(json.dumps(c.__dict__, indent=2))


# --- solar -------------------------------------------------------------------


@solar_app.command("table")
def solar_table() -> None:
    """Sun position at each E0-E5 epoch boundary."""
    from wtc4d.sync import solar as S

    table = Table(title="Sun position, WTC site, 2001-09-11")
    for col in (
        "epoch",
        "local",
        "azimuth",
        "elevation",
        "shadow az",
        "shadow ratio",
        "az rate/min",
    ):
        table.add_column(col)
    for row in S.epoch_sun_table():
        table.add_row(
            str(row["label"]),
            str(row["local"]),
            f"{row['azimuth_deg']:.2f}",
            f"{row['elevation_deg']:.2f}",
            f"{row['shadow_azimuth_deg']:.2f}",
            f"{row['shadow_len_ratio']:.2f}",
            f"{row['az_rate_deg_per_min']:.3f}",
        )
    rprint(table)


@solar_app.command("time")
def solar_time(
    shadow_azimuth: Annotated[
        float | None, typer.Option(help="ground shadow bearing, deg from true north")
    ] = None,
    sun_azimuth: Annotated[float | None, typer.Option()] = None,
    sigma_deg: Annotated[float, typer.Option()] = 2.0,
    morning: Annotated[
        bool, typer.Option(help="for --elevation only, which branch to search")
    ] = True,
    elevation: Annotated[float | None, typer.Option(help="apparent sun elevation, deg")] = None,
) -> None:
    """Invert a shadow bearing / sun azimuth / sun elevation to a time."""
    from wtc4d.sync import solar as S

    if shadow_azimuth is not None:
        est = S.time_from_shadow_azimuth(shadow_azimuth, sigma_deg=sigma_deg)
    elif sun_azimuth is not None:
        est = S.time_from_sun_azimuth(sun_azimuth, sigma_deg=sigma_deg)
    elif elevation is not None:
        est = S.time_from_sun_elevation(elevation, sigma_deg=sigma_deg, morning=morning)
    else:
        rprint("[red]pass one of --shadow-azimuth / --sun-azimuth / --elevation[/red]")
        raise typer.Exit(code=2)
    if est is None:
        rprint("[yellow]no solution in the daylight search window[/yellow]")
        raise typer.Exit(code=1)
    _print_model(est)


# --- fuse --------------------------------------------------------------------


@fuse_app.command("shot")
def fuse_shot(
    candidates_json: Annotated[
        str, typer.Argument(help="JSON file: a list of TimeEstimate objects")
    ],
) -> None:
    """Fuse a shot's candidate TimeEstimates into one, with outlier rejection."""
    from wtc4d.schema.time import TimeEstimate
    from wtc4d.sync import fuse as F

    raw = json.loads(Path(candidates_json).read_text())
    ests = [TimeEstimate.model_validate(e) for e in raw]
    result = F.fuse_estimates(ests)
    _print_model(result)


@fuse_app.command("graph")
def fuse_graph(
    priors_jsonl: Annotated[str, typer.Argument(help="JSONL: {shot_id, time: TimeEstimate}")],
    offsets_jsonl: Annotated[str, typer.Argument(help="JSONL of PairwiseOffset")],
    out: Annotated[str | None, typer.Option(help="write ShotTimeRecord JSONL here")] = None,
) -> None:
    """Solve the whole offset graph (priors + pairwise offsets) at once."""
    from wtc4d.schema.time import TimeEstimate
    from wtc4d.sync import fuse as F
    from wtc4d.sync.types import PairwiseOffset, ShotTimeRecord

    priors: dict[str, TimeEstimate] = {}
    for line in Path(priors_jsonl).read_text().splitlines():
        if line.strip():
            obj = json.loads(line)
            priors[obj["shot_id"]] = TimeEstimate.model_validate(obj["time"])
    offsets = [
        PairwiseOffset.model_validate(json.loads(line))
        for line in Path(offsets_jsonl).read_text().splitlines()
        if line.strip()
    ]
    result = F.solve_graph(priors, offsets)
    rprint(
        f"[green]{len(result.estimates)} shot(s) resolved, {len(result.unresolved)} unresolved, "
        f"{len(result.rejected_edges)} edge(s) rejected[/green]"
    )
    if out:
        records = [
            ShotTimeRecord(shot_id=s, fps=29.97, time=est) for s, est in result.estimates.items()
        ]
        F.write_records(records, out)
        rprint(f"wrote {len(records)} record(s) to {out}")
    else:
        for s, est in result.estimates.items():
            print(f"{s}: {est.model_dump_json()}")


@fuse_app.command("report")
def fuse_report(
    path: Annotated[
        str | None,
        typer.Option(help="time_estimates.jsonl; default data/time/time_estimates.jsonl"),
    ] = None,
) -> None:
    """Coverage histogram of a time_estimates.jsonl over the E0-E5 epochs."""
    from wtc4d.sync import fuse as F

    records = F.read_records(path)
    if not records:
        rprint(f"[yellow]no records found at {path or F.TIME_ESTIMATES_PATH}[/yellow]")
        raise typer.Exit(code=1)
    rprint(F.coverage_report(records).render())


# --- run: a practical end-to-end pass over a corpus -------------------------


@app.command("run")
def run(
    shots: Annotated[str, typer.Option(help="shots.jsonl (wtc4d.schema.corpus.Shot)")],
    sources: Annotated[str, typer.Option(help="sources.jsonl (wtc4d.schema.corpus.Source)")],
    frames_dir: Annotated[
        str | None, typer.Option(help="root containing <shot_id>/ frame directories, if extracted")
    ] = None,
    out: Annotated[str | None, typer.Option()] = None,
    channel_delays: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Automatic pass: broadcast metadata + on-screen clock, fused, per shot.

    This covers what needs no per-shot annotation.  Event-anchor ROIs, solar
    shadow points, and audio-landmark links are added separately (``events
    anchor``, ``solar time``, ``audio align`` / ``audio rumble``) as
    :class:`~wtc4d.schema.time.TimeEstimate` / :class:`PairwiseOffset` JSON
    and merged with ``fuse graph`` -- seeing this pass's output first is what
    tells you which shots still need one of those.
    """
    from wtc4d.schema.corpus import Shot, Source
    from wtc4d.sync import broadcast as B
    from wtc4d.sync import clock_ocr as C
    from wtc4d.sync import fuse as F
    from wtc4d.sync.types import ShotTimeRecord
    from wtc4d.sync.video import FrameSource

    delays = B.load_channel_delays(channel_delays) if channel_delays else None
    sources_by_id: dict[str, Source] = {}
    for line in Path(sources).read_text().splitlines():
        if line.strip():
            s = Source.model_validate(json.loads(line))
            sources_by_id[s.id] = s

    records: list[ShotTimeRecord] = []
    for line in Path(shots).read_text().splitlines():
        if not line.strip():
            continue
        shot = Shot.model_validate(json.loads(line))
        source = sources_by_id.get(shot.source_id)
        candidates = []

        if source is not None:
            timing = B.timing_for_source(source)
            est = B.estimate_for_shot(source, shot, timing=timing, delays=delays)
            if est is not None:
                candidates.append(est)

        media_path = None
        if frames_dir and (Path(frames_dir) / shot.id).is_dir():
            media_path = Path(frames_dir) / shot.id
        if media_path is not None:
            try:
                src = FrameSource(media_path, fps=shot.fps)
                fit, _, _ = C.estimate_shot_time(
                    src, hint_t=candidates[0].t if candidates else None, i0=shot.start_frame
                )
                if fit is not None:
                    candidates.append(fit.estimate)
            except Exception as exc:  # noqa: BLE001 - keep going on a bad shot
                rprint(f"[yellow]clock OCR failed for {shot.id}: {exc}[/yellow]")

        fused = F.fuse_estimates(candidates) if candidates else F.FusionResult()
        records.append(
            ShotTimeRecord(
                shot_id=shot.id,
                source_id=shot.source_id,
                fps=shot.fps,
                start_frame=shot.start_frame,
                time=fused.estimate,
                candidates=candidates,
                rejected=fused.rejected,
                chi2=fused.chi2 if candidates else None,
                dof=fused.dof if candidates else None,
            )
        )

    F.write_records(records, out)
    rprint(F.coverage_report(records).render())


if __name__ == "__main__":
    app()
