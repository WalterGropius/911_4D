# Licensing

This document covers three separate things with three separate answers:
the license of the *code* in this repository, the legal basis for
*processing* copyrighted footage as part of the pipeline, and the license
proposed for the *outputs* the project produces (poses, times, manifests,
trained splats). None of this is legal advice; where in doubt, consult a
lawyer, particularly before any commercial or public-facing use beyond
research.

## Code

Everything under version control in this repository (`wtc4d/`, `infra/`,
`web/`, `docs/`, tests, configuration) is licensed **Apache License 2.0**
(see `LICENSE` at the repo root). Contributions are accepted under the
same license — see "Contributor licensing" below.

## Footage: catalogued, never redistributed

The project **does not redistribute source footage**. `wtc4d.schema.corpus.Source`
stores a pointer (`url`) and metadata about an item; the bytes themselves
live only:

- at their original public location (archive.org, Wikimedia Commons,
  YouTube, an agency site), or
- transiently, on the project's own compute volume (`infra/`, see
  `data/README.md`) for processing, never committed to git (`.gitignore`
  blocks all common media extensions) and not served publicly.

This is a hard architectural constraint, not just a policy: `wtc4d.schema.corpus.Source`
has no bytes field, only `checksum_sha256`, `bytes` (a size, not content),
and `url`. If a workstream ever needs to serve a frame or clip to the
public web viewer, that has to be a low-resolution, transformed derivative
(a rendered camera-pose overlay, a processed still), never a re-hosted
copy of the original.

## Legal basis for processing

Downloading and processing publicly available, copyrighted footage
(deinterlacing, frame extraction, feature detection, camera-pose
estimation, temporal alignment) to build a research reconstruction relies
on **fair use** (17 U.S.C. §107) in the United States. This is a
fact-specific, four-factor test, not a rule, and this project does not
claim a blanket exemption. The factors, applied honestly to what this
project actually does:

1. **Purpose and character of the use.** Strongly transformative:
   individual shots are not being reproduced or redistributed, they are
   being used as measurement input (feature points, timing cues, pose
   constraints) to build a structured, queryable 3D+time reconstruction.
   The output is not a substitute for watching the original footage — it
   is a different kind of artifact (see `docs/methodology.md`). Purpose is
   nonprofit, educational, and research/forensic. This factor favors fair
   use, but "transformative" is a legal conclusion a court makes, not a
   label we get to self-assign — hence the caution throughout this doc.
2. **Nature of the copyrighted work.** Mixed. Some source material is
   factual news broadcast of a public event (favors fair use more);
   some is creative photography/documentary work (favors it less).
3. **Amount and substantiality used.** The project needs full shots for
   temporal alignment and full frames for camera registration (partial
   frames break PnP), so this factor does not favor us as strongly as a
   short-clip-only use would. What mitigates it: outputs are derived
   measurements (a pose, a timestamp), not clips — no output is "the
   video, slightly cropped."
4. **Effect on the market.** Low. This project is not a substitute for
   licensing footage from a broadcaster or a stock archive, does not
   compete with the original works commercially, and does not redistribute
   them. If a rights-holder's business model is licensing the footage
   itself, our use (measurement, not redistribution) is not a substitute
   for that license.

**Practical rule this implies for every workstream:** process footage for
its geometric/temporal signal, keep the source unredistributed, and make
sure derived outputs (poses, times, splats, manifests) don't function as a
disguised copy of the original artistic content. A trained splat that
merely reproduces a photographer's copyrighted framing/composition of a
scene is a closer call than one that reconstructs the static geometry many
independent cameras corroborate; prefer corroborated, multi-source
reconstruction (this is also better science).

## Per-source license tracking

Every catalogued item's licensing status is tracked with the exact fields
on `wtc4d.schema.corpus.Source` (`wtc4d/schema/corpus.py`):

| field | meaning |
|---|---|
| `archive` | Where it came from: `archive.org \| youtube \| wikimedia \| nist_foia \| flickr \| noaa \| usgs \| other` |
| `license` | An SPDX id (e.g. `CC-BY-4.0`), `public-domain`, `fair-use-research`, or `unknown` |
| `creator` | Photographer / broadcaster / uploader, for attribution |
| `url` | Canonical pointer to the original, at its original location |

`license="fair-use-research"` marks a source that is used under the fair
use analysis above rather than an explicit open license — this is the
expected value for most broadcast TV footage. `license="unknown"` is a
placeholder that must be resolved (or left conservatively unknown forever)
before any output derived predominantly from that single source is
promoted to a "redistributable" asset (see below). Public domain applies
to most U.S. federal government works (NOAA, USGS, NIST photography and
reports) unless a specific notice says otherwise — verify per item, don't
assume.

## What outputs are redistributable, and under what license

The project draws a clear line between *derived measurements* and
*derived renderings*:

| output | redistributable? | proposed license |
|---|---|---|
| `CameraPose` (poses), `TimeEstimate` (times), `SceneManifest` (manifests), the landmark/geo registry | Yes — these are measurements/metadata, not the copyrighted expression of any source | **CC-BY-4.0** (proposed; attribution to this project, which in turn credits per-source `creator`/`archive`) |
| Procedural gaussians (`wtc4d.procedural`) not trained from any single copyrighted source, but from parametric models + public engineering data (NIST geometry, physics) | Yes | CC-BY-4.0 (proposed), same as above |
| Trained gaussian splats (`wtc4d.recon`) for epochs/windows reconstructed from many independent, corroborating public sources | **Open question, discussed honestly below** | Not yet decided |
| Raw footage, extracted frames, or anything that is substantially the original creative content of one source | **No** | N/A — never redistributed |

**The open question on trained splats:** a gaussian splat trained from
video is, in one sense, a new 3D asset synthesized from many inputs
(closer to a derived measurement, like a pose). In another sense, if a
region of the scene is densely supported by essentially one source's
footage (e.g. a single well-placed camera is the only view of some
surface), the trained splat for that region could be considered a
derivative work of that specific footage's creative content, not just a
geometric measurement corroborated across sources. We do not think this
question has a settled answer, and this project does not resolve it by
assertion. Until it is resolved (ideally with legal review before any
public splat release), the working policy is:

- Splats trained predominantly from many independent, mutually
  corroborating sources (the common case for anything visible from
  multiple vantage points, which is most of the exterior geometry) are
  treated as redistributable research output, tentatively CC-BY-4.0,
  same as poses/times.
- Splats or regions substantially dependent on a single source's unique
  vantage or content are flagged (`SplatAsset.notes` in `wtc4d.schema.scene`)
  and held back from public redistribution pending case-by-case review,
  even if they are used internally / in the hosted viewer.
- Any public dataset release (Phase 3, see `PLAN.md`) should have this
  question specifically reviewed before publishing splats, not just poses
  and times.

## Attribution requirements

Any redistributed output derived substantially from an identifiable
source must carry attribution to that source's `creator` and `archive`
fields, and a pointer to the original `url`, alongside the project's own
attribution. This applies whether the output is a pose, a time estimate,
or (per the open question above) a splat. Concretely: a `SceneManifest`
or dataset release should ship a sources/credits file mapping each
asset back to the `Source.id`s that informed it, not just a single
project-wide license notice.

## DMCA / takedown process

Because this project does not host or redistribute source footage, a
standard DMCA takedown target (a hosted copy of a copyrighted work) mostly
does not exist here — the footage stays at its original host. The
relevant takedown surface is instead: (a) a link/pointer to footage a
rights-holder wants delisted from our catalogue, or (b) a redistributed
output (pose/time/manifest/splat) a rights-holder believes is a derivative
work distributed without adequate basis.

Until the project has a dedicated legal contact, handle both the same way:

1. File a request via the channel in `docs/ethics.md` ("Requests from
   victims' families and rights-holders") — currently a GitHub issue
   marked sensitive/private where possible.
2. State clearly which `Source.id` or output asset is at issue and what
   action is requested (delist from catalogue, withhold an output from
   redistribution, correct an attribution).
3. The project acknowledges within 5 business days and, absent a
   substantive factual or legal dispute, honors a delisting/withholding
   request by default — the catalogue is a means to an end, not something
   worth fighting a rights-holder over.
4. If reinstated after review, the requester is notified before the
   catalogue entry or output is restored.

This process applies in addition to, not instead of, each host platform's
own takedown process (archive.org, YouTube, Wikimedia Commons) for the
original material itself — this project has no ability to remove content
from those platforms.

## Contributor licensing

New contributions to code and docs are accepted under a **Developer
Certificate of Origin (DCO)** model rather than a Contributor License
Agreement: contributors certify (e.g. via `git commit -s`, "Signed-off-by")
that they have the right to submit the contribution under the project's
Apache-2.0 license, without assigning copyright to a central entity. This
is recommended over a CLA because the project has no legal entity to hold
assigned copyright, and DCO is the lighter-weight, well-precedented choice
for an all-volunteer open-source project (used by the Linux kernel, Git,
and most CNCF projects). If the project later forms a foundation or legal
entity, revisiting a CLA is reasonable — that decision is out of scope for
this document.

Data contributions (a new `Source` entry, a manually-verified `CameraPose`,
etc.) are contributions of *facts and measurements about* copyrighted
footage, not the footage itself, and are licensed the same way as other
project outputs (see above) — contributing a `Source` entry does not
require the contributor to own rights in the underlying footage, only to
accurately describe its provenance.
