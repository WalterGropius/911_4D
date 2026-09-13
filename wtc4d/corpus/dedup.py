"""Near-duplicate detection across sources via perceptual hashing.

Re-uploads and re-encodes of the same footage are common in this corpus
(a clip mirrored to YouTube, then re-uploaded to archive.org, then clipped
again). :func:`hash_frames` samples frames from a local video/photo file and
perceptual-hashes them (``imagehash.phash``); :func:`find_duplicates` groups
sources whose hashes are within a Hamming-distance threshold and returns
links to record in each ``Source.notes``/``tags`` (never merged away --
provenance for *all* copies matters even when the pixels are the same).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import imagehash
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_NUM_SAMPLES = 5
DEFAULT_HAMMING_THRESHOLD = 6  # phash is a 64-bit hash; <=6 bits differing ~= same shot


def hash_frames(
    video_path: Path, num_samples: int = DEFAULT_NUM_SAMPLES
) -> list[imagehash.ImageHash]:
    """Sample ``num_samples`` frames evenly across ``video_path`` and phash each."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return []
        n = min(num_samples, total)
        indices = sorted({int(i * (total - 1) / max(1, n - 1)) for i in range(n)})
        hashes = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hashes.append(imagehash.phash(Image.fromarray(rgb)))
        return hashes
    finally:
        cap.release()


def hash_image(image_path: Path) -> imagehash.ImageHash:
    with Image.open(image_path) as img:
        return imagehash.phash(img.convert("RGB"))


@dataclass
class DuplicateLink:
    source_id: str
    other_id: str
    distance: int  # min Hamming distance between any hash pair


def _min_distance(hashes_a: list[imagehash.ImageHash], hashes_b: list[imagehash.ImageHash]) -> int:
    return int(min((ha - hb) for ha in hashes_a for hb in hashes_b))


def find_duplicates(
    hashes_by_source: dict[str, list[imagehash.ImageHash]],
    threshold: int = DEFAULT_HAMMING_THRESHOLD,
) -> list[DuplicateLink]:
    """Pairwise-compare every source's hash set; return links under ``threshold``.

    O(n^2) in the number of sources -- fine for the hundreds-to-low-thousands
    of *locally hashed* items expected per run (this never runs over the
    full catalogue at once, only over whatever was just downloaded/hashed).
    """
    ids = list(hashes_by_source)
    links = []
    for i, id_a in enumerate(ids):
        for id_b in ids[i + 1 :]:
            hashes_a, hashes_b = hashes_by_source[id_a], hashes_by_source[id_b]
            if not hashes_a or not hashes_b:
                continue
            dist = _min_distance(hashes_a, hashes_b)
            if dist <= threshold:
                links.append(DuplicateLink(source_id=id_a, other_id=id_b, distance=dist))
    return links


def annotate_duplicates(sources: dict, links: list[DuplicateLink]) -> None:
    """Mutate ``sources`` (id -> Source) in place: add a ``dup:<id>`` tag and
    a note on each side of every link. Never removes/merges records --
    provenance for every copy is kept."""
    for link in links:
        for this_id, other_id in ((link.source_id, link.other_id), (link.other_id, link.source_id)):
            src = sources.get(this_id)
            if src is None:
                continue
            tag = f"dup:{other_id}"
            if tag not in src.tags:
                src.tags.append(tag)
            note = f"near-duplicate of {other_id} (phash distance {link.distance})"
            if note not in src.notes:
                src.notes = (src.notes + "; " if src.notes else "") + note
