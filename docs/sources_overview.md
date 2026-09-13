# Footage and data sources: overview

This is a prose orientation to the major public sources this project draws
on, what each is good for, and (where verified) a link. It is not the
machine-readable registry — that is `data/manifests/registry.yaml` and
`sources.jsonl`, owned by the `corpus` workstream
(`wtc4d/corpus`, `wtc4d.schema.corpus.Source`; see `data/README.md`). This
file exists so a human — a contributor deciding what to catalogue next, or
an investigator deciding where to go verify something themselves — has a
map of the territory. Licensing terms per source are tracked per-item on
`Source.license`; see `docs/licensing.md` for how that's used, not
restated here per source.

## Television broadcast: Internet Archive's *Understanding 9/11*

**https://archive.org/details/911** — "Understanding 9/11: A Television
News Archive," a collection of over 3,000 hours of contemporaneous
domestic and international TV news coverage from the week of September
11, 2001, digitized and hosted by the Internet Archive in partnership with
TV news researchers.

Good for: this is the backbone of the corpus. Broadcast video from dozens
of stations, often with **on-screen network time/date bugs** and known
**broadcast airtimes** — both are strong, checkable time anchors
(`TimeMethod.broadcast_metadata`, `TimeMethod.onscreen_clock` in
`wtc4d.schema.time`), which is rare for amateur footage. Multiple stations
covering the same moment from different network feeds also gives
independent corroboration for timing and for what was visible at a given
instant. Catalogue items from this collection with `Source.archive =
"archive.org"` and a `source_id`-style id capturing the specific item
identifier (e.g. `ia-<identifier>`).

## NIST FOIA release / 911datasets.org

NIST's investigation (NCSTAR, below) collected video and photographic
evidence from the public, media organizations, and its own contractors.
Following a FOIA lawsuit settled in 2011, a large release of this
underlying video/photo material (reported around 2.5 TB across dozens of
DVDs) was made to the International Center for 9/11 Studies and has since
been mirrored publicly, notably as
**https://archive.org/details/911datasets** (a ~305 GB compiled community
mirror covering much of this material and other FOIA-derived records) and
related items such as **https://archive.org/details/nist-r27-missing-vids**.
The original aggregation point, 911datasets.org, has historically indexed
these NIST releases alongside other FOIA material (FBI, FAA, NTSB, FEMA).

Good for: additional camera vantage points beyond the broadcast archive,
often amateur or local-news footage NIST specifically sought out for its
structural/timing analysis — i.e. material NIST itself judged useful for
exactly the kind of reconstruction this project is doing. Treat licensing
per item conservatively (`license="unknown"` or `"fair-use-research"`
unless a specific item's terms are clear) — this is a compiled mirror of
material from many original rights-holders, not a single license grant.

## Wikimedia Commons

**https://commons.wikimedia.org/wiki/Category:September_11_attacks_at_the_World_Trade_Center**
and related categories (e.g.
**Category:World_Trade_Center_on_9/11**,
**Category:Aerial_photographs_of_Ground_Zero_(World_Trade_Center)**,
**Category:Library_of_Congress_images_of_the_September_11_attacks**)
collect several hundred still photographs, including public-domain federal
government imagery (Library of Congress, NOAA) and openly-licensed
contributor photographs.

Good for: still photography with **explicit, machine-checkable licenses**
per file (Commons file pages state license and author directly) — this
makes it the easiest category of source to mark `license` with real
confidence rather than `"unknown"`. Useful for landmark verification
(`wtc4d.geo`) and for single-moment high-resolution reference frames,
though much less useful than video for temporal alignment since a single
photo has no surrounding footage to cross-correlate against.

## Flickr (Creative Commons)

Flickr hosts a long tail of contributor photographs from the day and its
aftermath under various Creative Commons licenses, searchable by CC
license filter. Good for: additional aerial-recovery-period stills (the
NOAA aerial photograph description above, for instance, is also mirrored
via Flickr/pingnews) and eyewitness ground-level photography with
per-photo CC licensing, similar in kind to Wikimedia Commons but with a
larger and less curated pool — expect more `license="unknown"` entries
here pending per-item verification.

## Documentary works: the Naudet brothers and *CameraPlanet*

Jules and Gedeon Naudet, filming a documentary about an FDNY probationary
firefighter, incidentally captured the only known video of the first
plane striking WTC1, plus extensive interior/exterior firehouse footage
that morning (released as the documentary *9/11*, 2002). *CameraPlanet*
(the production entity behind footage aggregation efforts by Peter
Kalikow, Nick Doob and others) similarly aggregated firsthand video from
non-professional cameras present that day for a documentary compilation.

Good for: rare or unique high-value moments (the Naudet first-impact
footage is a critical, essentially irreplaceable time anchor for
`WTC1_IMPACT` corroboration) and street-level, close-vantage footage that
complements the mostly-distant broadcast/aerial material. These are
commercially released documentary works — catalogue with
`license="fair-use-research"` pending specific rights review, and treat
with particular respect given how much first-responder and bystander
content they contain (see `docs/ethics.md`).

## Individual photographers

A number of individual photojournalists and amateur photographers (some
professionally credentialed, some not) produced widely-circulated still
sequences of the attacks and collapses from specific, often well-known
vantage points. Catalogue each as its own `Source` with `creator` set to
the photographer's name and `license` set per that photographer's own
terms/estate where known — do not lump distinct photographers' work under
a single generic source.

## NOAA / USGS aerial and satellite imagery

NOAA's National Geodetic Survey flew emergency-response aerial photography
missions over Lower Manhattan starting **September 23, 2001** (a Cessna
Citation Jet at ~3,300 ft with a Leica/LH Systems RC30 camera), continuing
daily through October 23, 2001, to support mapping, utility location, and
recovery efforts. The widely-reproduced September 23, 2001 orthophoto of
the WTC site (mirrored on Wikimedia Commons, Library of Congress, and
elsewhere) originates from this NOAA program. USGS similarly holds and
has published aerial/orthoimagery covering the same recovery period.

Good for: **epoch E5** (rubble pile / aftermath) — this is close to the
only source of accurate, top-down, metrically useful imagery of the
debris field's actual extent and shape in the days immediately following,
and anchors the `geo` workstream's rubble-surface reconstruction alongside
LiDAR (below). Public-domain as U.S. federal government work in the
general case; verify per specific product/agency release.

## NOAA / USGS LiDAR of Ground Zero

Airborne LiDAR surveys of the WTC site were flown during the recovery
period (again largely a NOAA/USGS effort, part of the broader post-9/11
emergency-response remote sensing program alongside the aerial photography
above) to support recovery operations, giving an elevation model of the
rubble pile's actual 3D shape over time.

Good for: the ground-truth surface `wtc4d.geo` needs for epoch **E5**
(`docs/methodology.md`'s epoch table) — the rubble pile is not something
that can be reasonably reconstructed from oblique news footage alone, and
LiDAR is the project's primary source for it. Treat as a small, derived
elevation product for `data/geo/` (per `data/README.md`'s size guidance),
not raw point-cloud data at full resolution.

## NIST NCSTAR reports and drawings

The National Institute of Standards and Technology's *Federal Building and
Fire Safety Investigation of the World Trade Center Disaster* produced the
canonical structural/timeline record of the collapses:

- **NCSTAR 1** (2005) — the final report on the collapse of the towers;
  source of the project's canonical event-anchor times
  (`wtc4d.timeline.EVENTS`) and structural narrative.
  https://www.nist.gov/publications/final-report-national-construction-safety-team-collapses-world-trade-center-towers
- **NCSTAR 1A** — the companion final report on the collapse of WTC7.
  https://www.nist.gov/publications/final-report-collapse-world-trade-center-building-7-federal-building-and-fire-safety-0
- Numerous **NCSTAR 1-*** sub-reports** covering structural design,
  building codes, fire dynamics, evacuation, and more — indexed from
  https://www.nist.gov/el/final-reports-nist-world-trade-center-disaster-investigation
- NIST's own photo/video/simulation repository for the investigation:
  https://www.nist.gov/world-trade-center-investigation/photos-videos-and-simulations

Good for: authoritative structural drawings and dimensions (feeding
`wtc4d.world.TowerSpec` and the `geo` workstream's verification of
placeholder tower geometry), the canonical timeline this project anchors
to (`wtc4d.timeline`), and the engineering basis for `wtc4d.procedural`'s
collapse kinematics. These are U.S. federal government publications
(public domain in the U.S.) — the drawings and text are freely usable;
any *photographs* NIST itself sourced from third parties within these
reports carry that third party's own rights, not NIST's.

## NYC open data

NYC's open data portal (data.cityofnewyork.us) and the Department of City
Planning's PLUTO/building-footprint datasets provide present-day building
footprints, heights, and planimetric data for Lower Manhattan. These
describe the city **today**, not as it stood in 2001 — every building
added, demolished, or altered since (the most important example being **30
Hudson Street, Jersey City, built 2004**, flagged in
`wtc4d.world.LANDMARKS` with `existed_on_2001_09_11=False` specifically so
it is never rendered pre-collapse) must be filtered out or dated by the
`geo` workstream before use. Good for: present-day surviving buildings'
footprints and heights as a starting point for landmarks that existed
unchanged in 2001 (verify each), and for present-day reference geometry
when reasoning about what's changed.

---

For how a `Source` entry's fields map to all of the above (id scheme,
`archive` enum values, license tracking), see `wtc4d.schema.corpus.Source`
and `docs/licensing.md`. For how a source's footage becomes a timed,
posed, renderable element of the reconstruction, see
`docs/methodology.md`.
