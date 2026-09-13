# FAQ

## Is this a conspiracy project?

No. 911_4D is neutral, reproducible, and evidence-first: it reconstructs
what the footage and documentary/engineering record (chiefly NIST NCSTAR
1/1A) actually show, with every number tagged with a source and an
uncertainty, and it does not adjudicate motive, blame, or contested
narratives that aren't resolvable from that evidence. See "Contested
claims" in `docs/ethics.md` for the full policy. If anything, the
project's stated purpose includes countering misinformation with
traceable, checkable evidence — see `docs/ethics.md`, "Purpose."

## Why not just use photogrammetry / classical structure-from-motion?

Because most of this footage is a bad fit for it. Classical SfM needs
consistent feature matches across many viewpoints of largely static
content; here, almost every camera is far from the towers (weak parallax
even for the towers themselves), the footage is 2001-era SD video (motion
blur, interlacing, heavy re-encoding — weak features), and smoke/fire/dust
dominate and change every frame, which breaks cross-video feature
matching outright. Instead, `wtc4d/camreg` registers each camera against a
**known 3D prior** — a georeferenced model of Lower Manhattan as it stood
that morning (`wtc4d/geo`, `wtc4d/world.py`) — by PnP on a landmark
registry (roofline corners, spires, bridge towers), then refines with
render-and-match. See `docs/methodology.md`, "Registering your own
footage," and `PLAN.md` §1 for the full rationale.

## Can I use this in court or in journalism?

Not as-is, and not without independently verifying the specific claim you
need. This is a research project in active early development (see
`README.md`: "phase 1 scaffolding"), not a certified forensic product, and
nothing here is legal advice about admissibility. That said, the project
is *designed* to support that kind of use eventually: every claim carries
a citation back to specific source frames and an explicit uncertainty
(`docs/methodology.md`, "Tracing a viewer claim back to source frames").
If you're relying on something from this project for reporting or legal
work, use the checklist in `docs/methodology.md` ("Checklist for a
defensible claim") and verify the underlying `Source`, `TimeEstimate`, and
`CameraPose` yourself rather than citing the reconstruction as an
unexamined black box.

## How accurate is the timing?

It varies by shot, and the project is explicit about exactly how much,
per shot — there is no single answer. Event anchor times (impacts,
collapses) come from NIST NCSTAR 1, each with a stated 1-sigma uncertainty
of a few seconds (`wtc4d.timeline.EVENTS`; e.g. WTC1 impact at 08:46:30 ±
5s). Individual shots/frames are timed by whichever method applies and
carry their own `TimeEstimate.sigma`: broadcast metadata and on-screen
clocks are typically tightest; audio/visual cross-correlation against an
already-timed reference and solar-shadow timing carry larger, and
sometimes correlated, uncertainty. See `docs/methodology.md`,
"Uncertainty: representation and propagation," and always check a given
estimate's `method` and `sigma` rather than assuming precision.

## Where is the footage?

Not in this repository, and not redistributed by this project at all.
`wtc4d.schema.corpus.Source` stores metadata and a pointer (`url`) to the
footage at its original public location (archive.org, Wikimedia Commons,
YouTube, an agency site); bytes are processed transiently on the project's
own compute volume (see `infra/`) and never committed to git or served
publicly as a re-hosted copy. See `docs/licensing.md` ("Footage:
catalogued, never redistributed") for why, and `docs/sources_overview.md`
for where the major public sources actually live.

## How do I submit footage I filmed?

First, read `docs/ethics.md` in full, particularly "Footage of victims and
first responders" and "What the reconstruction will never depict" — this
determines how footage containing people is handled regardless of what
else it shows. Then:

1. Make sure you're comfortable with the project's fair-use-based
   processing rationale and non-redistribution policy
   (`docs/licensing.md`) — your footage stays hosted wherever you already
   have it (or wherever you're willing to host it publicly); the project
   only needs a stable `url` to point at, plus enough metadata to fill in
   a `Source` entry (`wtc4d.schema.corpus.Source`: kind, archive, creator,
   license, any known time/location hints).
2. Open an issue using the "Footage submission"
   (`.github/ISSUE_TEMPLATE/footage_submission.yml`) template with that
   information; the `corpus` workstream reviews and catalogues it.
3. If you know roughly when and from where you filmed (even approximately
   — "Brooklyn Promenade, shortly after the second impact"), include it;
   this is exactly the `time_hint`/`location_hint`/`location_hint_text`
   data that makes a new source immediately useful rather than requiring
   cold-start time/camera solving.
4. You do not need to solve your own camera pose or timing — that's what
   `sync` and `camreg` do — but if you want to attempt it yourself, see
   `docs/methodology.md`, "Registering your own footage," for the
   landmark-PnP workflow at the level of detail currently designed.

## Why does the project use gaussian splats instead of a hand-modeled 3D scene?

Because a hand-modeled scene can only show what someone decided to model,
while a splat trained from many registered cameras is directly
constrained by what the footage actually shows — including things nobody
would think to model by hand (specific smoke behavior, specific window
breakage patterns, lighting at a specific moment). Structural geometry
that *is* well-known independent of footage (the tower footprints, floor
heights) still comes from a modeled prior (`wtc4d/geo`, `wtc4d/world.py`)
used to initialize and constrain training, and the procedural layer
(`wtc4d/procedural`) fills time/viewpoint gaps kinematically — but neither
substitutes for the evidence-trained splats where footage supports them.
See `docs/methodology.md`'s pipeline section for how the two fit together,
and the note there on never presenting the procedural layer as if it were
evidence-based.

## Who decides what counts as a "verified" landmark or a "settled" time?

Nobody unilaterally — it's structural. A landmark's `approx` flag
(`wtc4d.world.Landmark`) is only cleared by the `geo` workstream against a
citable survey source (NIST structural drawings, NYC DoITT planimetrics,
memorial pool footprints); a time's uncertainty only shrinks with genuine
independent corroboration, never by assertion (`docs/methodology.md`,
"Uncertainty: representation and propagation"). Anyone can open a PR
disputing a specific value, but it has to come with a citable source and,
where two sources disagree, both get recorded rather than one silently
overwriting the other (`docs/ethics.md`, "Contested claims").
