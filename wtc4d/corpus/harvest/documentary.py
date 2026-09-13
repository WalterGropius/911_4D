"""Documentary and photographic works: hand-curated pointer records.

These are well-known named works, not API-enumerable collections. Each
entry below was individually checked against archive.org / public sources
on 2026-09-13 (see ``data/manifests/registry.yaml`` -> ``documentary_pointers``
for the reasoning); this module just turns them into ``Source`` records so
they show up in the catalogue like everything else.
"""

from __future__ import annotations

from wtc4d.schema import Source, SourceKind

NAUDET_DOCUMENTARY = Source(
    id="doc-naudet-911-2002",
    kind=SourceKind.VIDEO,
    url="https://en.wikipedia.org/wiki/9/11_(2002_film)",
    archive="other",
    title='"9/11" (2002) -- Jules & Gedeon Naudet / James Hanlon, CBS',
    creator="Jules Naudet, Gedeon Naudet, James Hanlon",
    license="unknown",
    duration_s=6960.0,  # ~116 min broadcast cut
    tags=["documentary", "wtc1_lobby", "unique_vantage"],
    notes=(
        "The only film crew inside the WTC1 lobby that morning (there "
        "filming a probationary firefighter documentary); captured the only "
        "known footage of the first plane's impact and continued filming "
        "through both collapses. Full film is rights-managed (CBS/Naudet "
        "Bros.), not on archive.org. Re-uploads/clips of varying generational "
        "quality circulate (e.g. archive.org identifier 'youtube-lvAILTYegVI', "
        "titled as a 'poor quality VHS tape, largely watermarked' recut) -- "
        "useful for provenance tracking (multiple copy generations of the "
        "same source) even where the master is unavailable."
    ),
)

CAMERAPLANET_ARCHIVE = Source(
    id="doc-cameraplanet-archive",
    kind=SourceKind.VIDEO,
    url="https://archive.org/details/911-raw-footage-from-hoboken",
    archive="archive.org",
    title="CameraPlanet 9/11 citizen-video aggregation (partial mirror)",
    creator="Gary Pollard / CameraPlanet",
    license="unknown",
    tags=["documentary", "citizen_video", "compilation"],
    notes=(
        "CameraPlanet solicited and aggregated citizen-shot 9/11 video "
        "shortly after the event, later used by National Geographic/CBC "
        "productions. The original CameraPlanet archive/site is not "
        "independently reachable; partial mirrors and derivative broadcast "
        "cuts exist on archive.org (e.g. 'youtube-BQmaGCxHxj8', "
        "'youtube-K8emLw0mkA8', '911-raw-footage-from-hoboken'). Treat as "
        "multiple distinct sources, not one clean corpus."
    ),
)

BIGGART_PHOTOGRAPHS = Source(
    id="doc-biggart-photographs",
    kind=SourceKind.PHOTO,
    url="https://archive.org/details/september-11th-2001",
    archive="other",
    title="Bill Biggart -- final photographs, recovered camera",
    creator="Bill Biggart",
    license="unknown",
    tags=["documentary", "photojournalism", "ground_level", "wtc1_collapse"],
    notes=(
        "Photojournalist killed by the WTC1 collapse while shooting from "
        "street level; his camera was recovered from the rubble and the "
        "final frames developed. Archive held by the International Center "
        "of Photography (ICP); a short documentary about the recovery "
        "('September 11th, 2001', Imperial War Museums) is on archive.org. "
        "Individual photographs are not yet catalogued as separate Sources."
    ),
)

ARELLANO_PHOTOGRAPHS = Source(
    id="doc-arellano-photographs",
    kind=SourceKind.PHOTO,
    url="https://en.wikipedia.org/wiki/Bolivar_Arellano",
    archive="other",
    title="Bolivar Arellano -- ground-level photographs",
    creator="Bolivar Arellano",
    license="unknown",
    tags=["documentary", "photojournalism", "ground_level"],
    notes=(
        "NY Post / El Diario-La Prensa photojournalist known for iconic "
        "ground-level photos of people fleeing the collapse. No verified "
        "openly-licensed archive found yet -- his work is held by his "
        "former outlets and syndication agencies (Getty/Bettmann likely "
        "hold some). Listed as a known gap in the corpus README; worth a "
        "direct outreach or ICP/Getty search next session."
    ),
)

ALL = [NAUDET_DOCUMENTARY, CAMERAPLANET_ARCHIVE, BIGGART_PHOTOGRAPHS, ARELLANO_PHOTOGRAPHS]


def harvest(collection: str = "documentary_pointers") -> list[Source]:
    return list(ALL)
