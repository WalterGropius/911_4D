# Ethics

This document is normative for the project: if a design decision conflicts
with it, the design changes. It is owned by the `docs` workstream (see
`CONTRIBUTING.md`) but binds every workstream's output.

## Purpose

911_4D is, in order:

1. **An homage.** A precise, respectful record of what happened to two
   buildings, thousands of people, and a city, on one morning.
2. **An educational resource.** A way to see the timeline of the attacks —
   impacts, burn, collapse — as a single navigable space, correcting the
   fragmented, single-camera way most people have only ever seen it.
3. **A forensic and research tool.** Camera poses, absolute times, and
   uncertainties (`wtc4d.schema.camera.CameraPose`,
   `wtc4d.schema.time.TimeEstimate`) are structured so an investigator or
   journalist can ask "what does the footage actually show, from where, at
   what time, with what error bars" and get a traceable answer.
4. **A tool against misinformation.** Because every claim in the
   reconstruction is backed by cited source frames and an explicit
   uncertainty (see `docs/methodology.md`), it lets people check specific
   factual claims — trajectory, timing, sequence of structural failure —
   against assembled evidence, rather than against a single low-resolution
   clip or a rhetorical claim.

It is not an entertainment product, is not monetized around the events of
the day, and takes no position on motive, blame, or political
interpretation beyond what the physical evidence shows.

## What the reconstruction will depict

- The exterior geometry of the WTC complex and surrounding Lower Manhattan
  as they stood on 2001-09-11 (`wtc4d/world.py`, `wtc4d/geo`).
- Structural states over time: intact, burning (fire and window breakage,
  as visible in footage), and the progression of each collapse
  (`wtc4d/timeline.py` epochs `E0`-`E5`).
- Smoke, dust, and debris clouds as time-tagged volumetric content,
  distinct from the structures (`wtc4d/procedural`, `wtc4d/recon`).
- Camera positions and viewing frusta for every registered piece of
  footage, so a viewer can jump to the source frame (`web/`).
- Aircraft trajectories only insofar as they are visible or reliably
  documented in cited sources (NIST NCSTAR 1, NTSB); we do not animate
  invented flight paths.

## What the reconstruction will never depict

Regardless of what is visible in source footage, the reconstruction does
**not** render:

- Identifiable individuals: no photorealistic or recognizable
  representation of a specific person, inside or outside the buildings.
- People falling or jumping. This is the single most sensitive category of
  footage/imagery from the day and is out of scope for any rendered
  output, full stop — not blurred, not stylized, not included with a
  warning. Source frames that happen to contain this content are still
  usable for *unrelated* purposes (e.g. deriving a camera pose from
  background landmarks), but the frames themselves are never displayed by
  the viewer and no gaussians are trained to represent people.
- Human remains, in any form.
- Gore, or close-up injury of any kind.

This is enforced procedurally, not just by convention: `wtc4d.procedural`
has no human/body model, `wtc4d.recon` trains splats from masked regions
(dynamic-object masks exclude people; see `docs/methodology.md`), and the
`web/` viewer's content-warning gating (see below) is the last line, not
the only line, of defense.

Depicting people (crowds at street level, first responders arriving,
office workers evacuating) as anonymous, non-identifiable low-detail
figures *may* be considered in a later phase for scale and context, but is
out of scope for phase 1 and requires an explicit decision, not an
implicit one from training on footage that happens to contain people.

## Footage of victims and first responders

- Footage is catalogued and processed for its geometric and temporal
  content (camera pose, timing, structural state) regardless of who or
  what else appears in frame — we do not curate the *corpus* to exclude
  such footage, because doing so would bias camera coverage and timing
  data. We curate the *output*.
- No source frame containing a recognizable person is ever surfaced in the
  viewer UI (thumbnails, "jump to source" overlays) without review; the
  `camreg`/`corpus` pipeline should default such thumbnails to blurred or
  withheld pending that review. If you are working on any code path that
  surfaces a raw frame to a user, treat "does this need review" as a
  blocking question, not a follow-up.
- The project does not use footage as an occasion to relitigate individual
  survival stories, rescue accounts, or casualty details. Where a
  structural or timing claim requires citing what a specific first
  responder or agency reported (e.g. FDNY radio traffic timing an event),
  cite the report or transcript, not a personalized narrative.

## Requests from victims' families and rights-holders

- The project maintains a single contact channel for takedown, correction,
  or objection requests, published in `docs/licensing.md` (DMCA / rights)
  and repeated in `SECURITY.md`'s reporting section for anything
  safety-adjacent. Until a maintained email/org contact exists, requests
  should be filed as a GitHub issue using the report path described in
  `SECURITY.md`, marked private/sensitive where the repository's issue
  tooling allows it.
- **Response time target:** acknowledge within 5 business days, resolve or
  give a substantive status update within 30 days. This mirrors common
  practice at archival institutions (see below) and is a commitment we
  intend to keep even before the project has paid staff.
- A family member's request to remove or reconsider the use of footage
  depicting their relative is honored by default — the burden is on the
  project to justify continued use, not on the requester to justify
  removal. This applies even to footage that is otherwise properly
  licensed or public domain; legal permission to use footage is a floor,
  not a defense against a family's objection.
- A rights-holder's takedown request under fair use / DMCA is handled per
  the process in `docs/licensing.md`.

## Alignment with memorial-institution practice

The project does not speak for, and is not affiliated with, the National
September 11 Memorial & Museum or any other institution. But its practice
is deliberately consistent with publicly stated norms at memorial and
archival institutions that steward this material, in particular:

- The 9/11 Memorial & Museum restricts public display of the most
  sensitive material (imagery of human remains, graphic content) to
  curated, opt-in contexts with content notices, rather than open display
  ([Visitor Guidelines](https://www.911memorial.org/visit/about/visitor-guidelines);
  Museum [Collections Management Policy](https://www.911memorial.org/sites/default/files/inline-files/museum_collections_management_policy_full_1.pdf)).
  911_4D goes further and excludes this material from rendered output
  entirely rather than gating it, because the project has no curatorial
  staff to make case-by-case judgment calls at scale.
- Archival stewards of TV coverage from the day, notably the Internet
  Archive's *Understanding 9/11* television news archive
  (https://archive.org/details/911, see `docs/sources_overview.md`),
  preserve broadcast material as historical record without editorializing
  it; 911_4D treats its corpus the same way — catalogue and preserve the
  evidentiary value, without re-cutting it into a narrative.

If we later publish specific content-warning language for the viewer, it
will live alongside the viewer code in `web/` and be cross-referenced from
this file.

## Language guidelines for contributors

When writing commit messages, code comments, docstrings, issue text, or
any prose in this repository that refers to the events of the day:

- Use neutral, precise, factual language. "WTC2 collapse initiation at
  09:58:59" not "the tower's horrifying final moments."
- Do not editorialize about victims, perpetrators, or agencies in code or
  docs. State what a source says and cite it (`wtc4d.schema.time.TimeEstimate.evidence`,
  `Source.notes`).
- Do not use the event as a metaphor, joke, or rhetorical device anywhere
  in the codebase, including test fixture names and example data.
- Refer to the towers as WTC1 (North Tower) / WTC2 (South Tower), matching
  `wtc4d.world.WTC1` / `WTC2`, not colloquial or narrative names.
- When describing uncertainty, state it as uncertainty ("sigma", "95% CI",
  "unverified") rather than false precision or false doubt. Don't round an
  NIST-sourced anchor time into vague language, and don't state an
  estimated camera pose as if it were surveyed.

## Contested claims

The reconstruction takes an evidence-first, reproducible stance:

- **We reconstruct what the footage and physical/documentary record show,
  with quantified uncertainty.** We do not adjudicate competing
  narratives, motives, or interpretations that are not settled by that
  evidence.
- Where a claim is genuinely contested at the level of *physical fact*
  (e.g. a disputed timing, a disputed camera-visible sequence of events),
  the project's answer is to improve the evidence base — more footage,
  tighter time sync, better camera registration, explicit sigmas — not to
  pick a side. If two credible sources disagree (as `wtc4d.timeline`
  already does for impact times: NIST NCSTAR 1 vs. the 9/11 Commission
  Report, both recorded in `Event.source`), record both, cite both, and
  let the sigma reflect the disagreement.
- The project is not a venue for adjudicating claims that are not
  resolvable from the physical/documentary record it works with (e.g.
  claims about intent, foreknowledge, or motive). Issues or PRs framed
  around such claims, rather than around footage, timing, or geometry,
  are out of scope.
- "Neutral" does not mean "false balance." Where NIST's structural
  findings (NCSTAR 1) are the only citable engineering analysis of the
  collapse mechanism, the project treats them as the working reference for
  anything the reconstruction needs to depict mechanically (e.g. the
  procedural collapse animation in `wtc4d.procedural`), while still citing
  it explicitly as a source rather than presenting it as unauthored fact.
  A future, better-sourced analysis would supersede it the same way a
  better camera pose supersedes an approximate one — by citation and
  sigma, not by silent replacement.

## Practical checklist for contributors

Before merging anything that touches rendered/user-facing output:

- [ ] Does this surface a source frame or thumbnail to a user? If yes, has
      it been checked for identifiable people, falling/jumping, or remains?
- [ ] Does this add or change wording (UI copy, docs, commit message) that
      touches the event itself? Re-read it for the language guidelines
      above.
- [ ] Does this add a new hard-coded factual claim about the event? Does
      it cite a source and, where applicable, a sigma?
