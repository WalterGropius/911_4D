# wtc4d.sync -- temporal alignment

Assigns every shot (and, within a shot, every frame) an **absolute time**
in project seconds (see `wtc4d.timeline`) with a **defensible 1-sigma
uncertainty** and full provenance. Without this there is no 4D: nothing else
in the project can place a frame's content on the timeline without it.

Six cues, one fuser:

| module | cue | typical sigma |
|---|---|---|
| `broadcast.py` | archive.org TV-archive airtime + channel delay | 2-4 s |
| `sources_time.py` | photo EXIF / camcorder container metadata | 2-5 min (unverified camera clock) |
| `clock_ocr.py` | on-screen broadcast clock, OCR'd and fit | **frames** (well under 1 s) when seconds are shown; a few seconds with minute-only displays |
| `events.py` | impact fireball / collapse onset visible in frame, matched to a NIST anchor | 5-8 s (anchor sigma + a real detection lag, see below) |
| `audio.py` | GCC-PHAT cross-correlation to an already-timed clip; low-frequency collapse rumble | sub-frame for a clean correlation; a few seconds for the rumble |
| `solar.py` | sun azimuth/elevation from a shadow, inverted | 30 s at best (0.1 deg angle error), minutes/tens-of-minutes typically |
| `fuse.py` | combines all of the above per shot, and across shots via pairwise offsets | whatever the best surviving cue gives |

`wtc4d sync --help` exposes one CLI subcommand per module; `wtc4d sync run`
chains broadcast metadata + clock OCR (the two fully-automatic cues) over a
whole corpus and fuses the result. See **CLI reference** below.

## Time conventions (recap of `wtc4d.timeline`)

All absolute times are **project seconds since 2001-09-11 00:00:00 EDT**.
Every `TimeEstimate` carries a **1-sigma** uncertainty in seconds, a
`TimeMethod`, a human-readable `evidence` string, and `derived_from` (the ids
of whatever it was built from) -- nothing in this package emits a bare number.

## Methods

### Broadcast metadata (`broadcast.py`)

The archive.org `collection:911` TV-News-Archive items are half-hour (or
hour, or three-hour) off-air recordings, each timestamped to the second.
Verified fields (see `data/time/examples/*.metadata.json`, two real items
fetched via the `internetarchive` package and committed for offline tests):

* `identifier` -- `CHANNEL_YYYYMMDD_HHMMSS_Title`; the timestamp is **UTC**
  and equals `start_time`.
* `start_time` / `stop_time` -- naive, **UTC**. `stop_time - start_time`
  equals `runtime`, so media offset 0 is exactly `start_time`.
* `start_localtime` / `utc_offset` -- the capture station's local zone;
  cross-checked against `start_time` by `ArchiveItemTiming.consistent()`.
* `imagecount` is **not** a frame count -- it is one derivative thumbnail per
  second, so `imagecount ~= runtime` in seconds (misreading this as a frame
  count was an early mistake this workstream caught by cross-checking
  against `frames_per_second * runtime`; `consistent()` guards it now).
* `source` -- the reception chain (`"Antenna > ..."` vs `"DISH Network >
  ..."`), classified into `antenna | satellite | cable` and used to select a
  channel-delay entry when the specific channel has none of its own.

Two corrections sit between an archive airtime and the moment something
happened: an **archive clock sigma** (`ARCHIVE_CLOCK_SIGMA_S = 2.0 s`,
a conservative allowance, not yet independently measured) and a
**channel delay** (`data/time/channel_delays.yaml`, keyed
`channel -> reception path -> default`, each entry with its own sigma and
`evidence`). See "Validation" below for the one channel delay calibrated
against real footage so far.

### File metadata (`sources_time.py`)

EXIF (`DateTimeOriginal` + subsec + offset) for photos, container
`creation_time` for video, optionally DV/SMPTE timecode read as time-of-day.
The hard part is not parsing these fields, it is **not trusting them blindly**:

* A consumer camera clock in 2001 was set by hand -- default sigma is 5
  minutes (`EXIF_SIGMA_S`) unless a per-source or per-camera offset has been
  *measured* (typically by finding an anchor or a clock-bearing TV in that
  camera's own footage) and recorded in `data/time/camera_clock_offsets.yaml`.
* A container `creation_time` on anything re-encoded (every YouTube
  download, every derivative) is the **transcode** date. Every value is
  checked against `PLAUSIBLE_WINDOW` (2001-09-11/12) and rejected -- with a
  stated reason -- otherwise; see `test_photo_time_rejects_implausible_date`.

### On-screen clock (`clock_ocr.py`)

1. **Find the clock bug**: sample ~12 frames, run OCR (`rapidocr-onnxruntime`,
   CPU, PP-OCRv4 models ship in the wheel -- no network needed at runtime),
   keep text boxes that parse as a time, cluster by position, and score each
   cluster by a **temporal-variance** test: high change *inside* the box
   (digits tick), low change in a ring *around* it (static graphic). This is
   what tells a clock bug apart from a caption or a scene timestamp.
2. **Read it** over the shot at ~2 fps.
3. **Fit it**: each reading is an *interval* (`[HH:MM:00, HH:MM:60)` for a
   minute-only display, one second wide with seconds shown), and the fit is
   the **maximum-overlap interval** across all readings -- an exact,
   O(n log n) sweep, not a sampled RANSAC. A shot whose readings span a
   minute *transition* is pinned to about the OCR sampling gap, not to a
   minute, because the last "9:02" and the first "9:03" intersect in a
   narrow band. Rejected readings are reported, not silently dropped.
4. Time zones (`ET`, `CT`, `MT`, `PT`, `UTC`/`GMT`) are parsed from a label
   near the digits and converted to EDT; CNN's ticker on 9/11 rotated through
   all four US zones every ~20s, and the fit above treats each zone
   correctly (see "Validation": 90/90 readings across all four zones agreed).

### Event anchors (`events.py`)

Two image-domain detectors, run over a caller-supplied ROI
(`wtc4d.sync.types.RoiTrack`, a frame-indexed bbox -- the interface the
`camreg` workstream will eventually fill in from a registered camera pose;
until then, one box per shot, by hand):

* **Impact flash**: a sharp rise in fire-coloured (warm hue, saturated,
  bright) pixel fraction inside the ROI, rejected if the rest of the frame
  changed comparably (a cut, not an explosion).
* **Collapse onset**: a sustained rise in changed-pixel fraction whose
  spatial centroid moves *downward*. Rejecting false positives here needed
  two checks beyond the naive threshold, both found by testing against real
  footage (see "Validation"): **persistence** (a hard cut or feed dropout is
  one frame that goes flat; a real collapse churns for seconds) and a
  **luminance floor** (a fade-to-black transition produces the same
  sustained, downward-biased signal as a real collapse, but real debris/dust
  is lit and never reads as near-black).

Detected events are matched to `wtc4d.timeline`'s NIST anchors via
`match_to_anchors`; without a prior time, both plausible anchors are
returned (impact 1 vs 2, collapse 1 vs 2) and the fusion step (or a
broadcast-metadata prior) disambiguates.

**Known bias**: the collapse detector fires when the *visible* debris signal
clears threshold, which lags the true structural onset by
`COLLAPSE_DETECTION_LAG_S = 6.5 s` (one real-footage measurement, see
"Validation"; sigma widened to 3.5 s accordingly). This is corrected for by
default (`correct_detection_lag=True`) and is exactly why an event-anchor
estimate should never be trusted alone for the project's <=1 s target --
it is a cross-check and a coarse cue, not a replacement for clock OCR.

### Audio (`audio.py`)

GCC-PHAT (phase-transform cross-correlation, Knapp & Carter 1976) at 8-16
kHz: it whitens each signal's spectrum before correlating, so re-encoding and
EQ differences between two copies of the same audio do not bias the peak the
way plain cross-correlation would. A small grid of playback-**rate**
candidates (NTSC drop-frame 30000/1001 vs 30 fps, a 25->24 telecine
mismatch, +-0.1%) is searched; the peak's prominence over the noise floor and
over its best rival elsewhere give a confidence and a sigma that degrade
honestly for an ambiguous match rather than reporting a number nobody should
trust.

Two derived cues reuse the same machinery:

* `detect_rumble_onset` -- the collapses produced a very-low-frequency
  (5-40 Hz) rumble; detected the same way as the visual collapse onset
  (robust z-score + persistence, see "Validation" for why persistence is
  essential here too).
* `match_landmark` -- align an untimed clip against a short reference
  snippet of known time (a live sentence heard on several recordings),
  chaining it onto the TV archive without the clip itself needing a clock or
  a tower in frame.

### Solar shadow (`solar.py`)

The NOAA solar-position algorithm (same one behind NOAA's Solar Calculator
spreadsheet), implemented from scratch and cross-checked against the
`astral` library to **1e-4 deg** agreement on every 9/11 anchor time (see
"Validation"). The sun moves at only ~0.25 deg/minute, so

```
sigma_t = sigma_angle / |d(azimuth)/dt|
```

-- a 30 s estimate needs the shadow bearing to about 0.12 deg, and this cue
is honestly the **weak** one: good for "is this really the morning of 9/11"
and gross-error catching, occasionally reaching the tens of seconds. A
`ShadowAnnotation` (JSON: shot/frame ref, either a directly-measured
`world_azimuth_deg` or two image points on a ground shadow edge plus a
`ground_point_of` camera callback from `camreg`) inverts to a
`TimeEstimate`; see `data/time/annotations/` for the format.

A `WindCue` (surface wind direction/speed) is provided as a **very weak**
plume-direction consistency check, not a timing cue -- the values there are
flagged `verified=False` (a working assumption from the plume geometry
visible in the footage, not yet an NOAA-ISD-observation-backed number); see
the docstring in `solar.py` for exactly what to replace and how.

#### Sun position at each epoch boundary (WTC site, 2001-09-11)

| epoch | local time | azimuth (deg) | elevation (deg) | shadow azimuth (deg) | shadow length ratio | az rate (deg/min) |
|---|---|---|---|---|---|---|
| E0 start | 00:00:00 | 341.83 | -43.14 | 161.83 | 99.00 (night) | 0.332 |
| E0 end (WTC1 impact) | 08:46:30 | 106.02 | 24.26 | 286.02 | 2.22 | 0.187 |
| E1 end (WTC2 impact) | 09:02:59 | 109.16 | 27.23 | 289.16 | 1.94 | 0.195 |
| E2 end (WTC2 collapse) | 09:58:59 | 121.16 | 36.82 | 301.16 | 1.34 | 0.237 |
| E3 end (WTC1 collapse) | 10:28:22 | 128.55 | 41.38 | 308.55 | 1.14 | 0.267 |
| E4 end | 12:00:00 | 158.49 | 51.79 | 338.49 | 0.79 | 0.387 |
| E5 end | 00:00:00 +2d | 341.81 | -43.93 | 161.81 | 99.00 (night) | 0.337 |

(Regenerate with `wtc4d sync solar table`; azimuth is clockwise from true
north, elevation is refraction-corrected apparent elevation.)

### Fusion (`fuse.py`)

Two levels:

1. **Per-shot** (`fuse_estimates`): inverse-variance-weighted mean of a
   shot's candidate `TimeEstimate`s, with iterative leave-one-out
   chi-square rejection (default 3-sigma) -- a bad candidate is scored
   against the fused estimate of *the others*, so it cannot inflate the
   sigma used to excuse itself.
2. **Graph** (`solve_graph`): every shot's start time is one scalar unknown;
   every per-shot fused estimate and every `PairwiseOffset` (from
   `audio.align_clips`) is a linear equation. Solved once for the whole
   corpus by weighted normal equations (dense, instant at the shot counts
   this project expects), with the same iterative chi-square rejection
   applied to *edges*. A connected component with no anchor at all is
   correctly reported as unresolved rather than solved to an arbitrary
   offset -- there is no absolute information to give it one.

Output: `data/time/time_estimates.jsonl`, one `ShotTimeRecord` per line
(shot id, fused `TimeEstimate`, every candidate considered including
rejected ones, and -- when a `LinearClock` was fit -- a `frame_time(i)`
helper for any frame of the shot, not just its start). `wtc4d sync fuse
report` prints a coverage histogram over the E0-E5 epochs.

## Data files this workstream owns

| path | contents |
|---|---|
| `data/time/time_estimates.jsonl` | fused output, one `ShotTimeRecord` per line |
| `data/time/channel_delays.yaml` | per-channel / per-reception-path broadcast delay table |
| `data/time/camera_clock_offsets.yaml` | measured consumer-camera clock offsets |
| `data/time/examples/*.metadata.json` | two real archive.org items, cached for offline tests |
| `data/time/annotations/` | hand-made `ShadowAnnotation` / `AudioLandmark` JSON |

## Adding an annotation

* **Shadow**: write a `ShadowAnnotation` JSON (see `solar.py`'s docstring for
  the two `kind`s) into `data/time/annotations/`, then
  `wtc4d sync solar time --shadow-azimuth <deg> --sigma-deg <deg>` (or call
  `solar.estimate_from_annotation` directly once a camera callback exists).
* **Event ROI**: `wtc4d sync events anchor <media> --roi x,y,w,h [--prior-json
  <TimeEstimate.json>]`.
* **Audio landmark**: construct an `AudioLandmark` (id, reference file path,
  its own known `TimeEstimate`) and call `audio.match_landmark`.
* Every one of these produces a `TimeEstimate` (or `PairwiseOffset`) that
  feeds straight into `fuse.fuse_estimates` / `fuse.solve_graph`.

## CLI reference

```
wtc4d sync broadcast time <identifier-or-cached.json> --offset-s <s>
wtc4d sync broadcast file-time <photo-or-video>
wtc4d sync clock fit <video> [--start-s] [--duration-s] [--hint HH:MM[:SS]]
wtc4d sync events detect <video> --roi x,y,w,h [--kind impact|collapse|both]
wtc4d sync events anchor <video> --roi x,y,w,h [--prior-json est.json]
wtc4d sync audio align <clip_a> <clip_b> [--max-shift-s]
wtc4d sync audio rumble <video>
wtc4d sync solar table
wtc4d sync solar time --shadow-azimuth <deg> [--sigma-deg]
wtc4d sync fuse shot <candidates.json>
wtc4d sync fuse graph <priors.jsonl> <offsets.jsonl> [--out out.jsonl]
wtc4d sync fuse report [--path time_estimates.jsonl]
wtc4d sync run --shots shots.jsonl --sources sources.jsonl [--frames-dir DIR] --out out.jsonl
```

`wtc4d sync run` is the fully-automatic pass (broadcast metadata + clock OCR,
fused); event anchors, solar shadows, and audio links need a per-shot
annotation (ROI, shadow points, or a landmark) and are merged in afterwards
with `fuse graph` -- `run`'s coverage report is what tells you which shots
still need one.

## Validation

Everything above was checked against real 2001-09-11 archive.org footage
(never committed to the repo -- see `CONTRIBUTING.md`), downloaded on-demand
via short `ffmpeg` byte-range pulls (a few hundred MB total, kept outside the
repo). This section is the receipt.

**Solar model**: `sun_position` agrees with the independent `astral` library
to **1e-4 deg** in azimuth and elevation at all four NIST anchor times and at
solar noon/16:00; sunrise/solar-noon/sunset are within a minute of the
published NYC 2001-09-11 almanac (06:33 / 12:53 / 19:11-19:12 EDT).

**On-screen clock, CNN_20010911_130000_CNN_Live_This_Morning** (archive.org
identifier; item start 09:00:00 EDT per its metadata): the WABC NewsCopter 7
feed of the second impact, media offset ~150 s, carries a ticker clock that
rotates ET/CT/MT/PT every ~20 s. `estimate_shot_time` found the clock bug,
read it over 90 frames, and fit frame 0 (media offset 0 of the analysed
window, i.e. absolute item offset 150 s) to **09:02:29.75 +/- 0.20 s**
against a ground truth (the archive's own `start_time` + 150 s) of
**09:02:30.00** -- **-0.25 s error**, with 90/90 readings agreeing across all
four rotating time-zone labels and 0 rejected.

**Impact-flash detection, same item**: the fireball is detected at media
offset 184.8 s (frame 1043 of the analysed window), i.e. archive airtime
09:03:04.8 EDT, against the NIST WTC2-impact time of 09:02:59 (+-5 s) -- a
**+5.8 s** difference. Investigated against a full-frame cut check and a
persistence/luminance-floor check: two genuine false positives were found
and fixed this way during development (a whole-picture-in-picture-window
brightness step at ~t=21s that a naive cut-rejection missed because only the
live-video sub-window changed, and a hard fade-to-black-and-back transition
at ~t=31.6s with the same signature) -- both are why the collapse detector
carries the persistence and luminance-floor checks documented above, and why
`_suppress` in both `events.py` and `audio.py` clusters candidates by the gap
between *consecutive* detections rather than by distance from the
highest-confidence one (the original approach re-triggered a "new" onset
every few seconds for any sustained event, which is exactly how a multi-
second collapse behaves). The **+5.8 s** offset became the working CNN
channel-delay calibration in `channel_delays.yaml`, flagged with a wide
sigma (4.0 s) and an explicit caveat that a single-anchor calibration cannot
separate genuine broadcast-chain delay from an instant-replay.

**Collapse-onset detection, WRC_20010911_140000_News_4_at_10** (WNYW
courtesy feed of WTC1, media offset 5280 s into an item starting 09:00:00
EDT): detected at media offset 5308.7 s, i.e. archive airtime 10:28:28,
against the NIST WTC1-collapse time of 10:28:22 (+-2 s) -- a **+6.7 s**
lag, matching visual inspection (the tower's top visibly starts to
tilt/buckle around 10:28:20, but the pixel-change threshold only clears once
violent debris ejection begins a few seconds later). This is the basis for
`events.COLLAPSE_DETECTION_LAG_S = 6.5 s` (sigma 3.5 s, since it rests on one
measurement).

**Audio alignment**: `align_clips`/`align_signals` recovered a known 5.000 s
offset between two trims of the same real broadcast audio (a self-alignment
sanity check) with **0.985 confidence** and a sub-millisecond fit; a
from-scratch synthetic test with a known NTSC-drop-frame rate change
(30000/1001) plus a 3.257 s offset recovered both to better than 1 ms.

**Rumble onset**: only validated synthetically (a clean single detection at
the true onset, no re-triggering over an 8 s sustained event). The one real
collapse audio track tried (CNN's broadcast of the WNYW feed) is dominated by
studio anchor narration rather than the on-scene rumble, so it did not
trigger -- an honest limitation: this cue is intended for amateur/on-scene
audio with real ambient sound, not a heavily produced studio broadcast with
voice-over, and should not be expected to fire on the latter.

**Fusion math**: `fuse_estimates` and `solve_graph` were checked against
hand-computed expected values (weighted means, chain-propagated sigma,
outlier rejection that leaves the rest of a chain intact, a disconnected
component with no anchor correctly reported as unresolved) -- see
`tests/test_sync_fuse.py`.

All of the above (except the network fetch of the source video, which is
never part of a test) is exercised by the automated suite:
`tests/test_sync_*.py`, synthetic-signal only, no network, runs in a few
seconds.

## Known limitations

* Event-anchor timing has a real, one-directional bias (the collapse
  detector fires late) that is corrected from a *single* real-footage
  measurement -- treat `COLLAPSE_DETECTION_LAG_S` as a reasonable prior, not
  a precisely known constant, and tighten it with more real anchors as they
  come in.
* The z-score behind both onset detectors (`events._robust_z`,
  `audio.detect_rumble_onset`'s band-energy z-score) is computed against the
  *whole* analysed window's median/MAD, so it needs the event to be a
  minority of that window -- do not feed it a clip that is mostly collapse
  footage.
* `sources_time.WindCue` defaults are a working assumption from the plume
  geometry visible in footage, not yet backed by an NOAA NCEI ISD
  observation -- replace before quoting a wind-based cue.
* Clock OCR needs a legible clock bug; it has not been tested against
  interlacing artifacts or heavy VHS noise beyond MPEG-2 broadcast
  compression.
* `RoiTrack` is a hand-supplied bounding box until `camreg` can derive one
  from a registered camera pose.
