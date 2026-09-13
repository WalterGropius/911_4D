"""wtc4d: 4D reconstruction of the World Trade Center attacks (2001-09-11).

Shared contract modules (owned by the orchestrator, change via PR + discussion):

- ``wtc4d.schema``   pydantic models exchanged between workstreams
- ``wtc4d.world``    world coordinate frame, tower geometry, landmarks
- ``wtc4d.timeline`` event anchors and epochs in absolute time

Workstream packages (each owned by one team / session):

- ``wtc4d.corpus``      footage discovery, catalogue, download, dedup, shots
- ``wtc4d.sync``        temporal alignment of footage to absolute time
- ``wtc4d.geo``         static scene prior (Lower Manhattan, 2001-09-11)
- ``wtc4d.camreg``      camera registration in the world frame
- ``wtc4d.procedural``  procedural 4D baseline (towers, smoke, collapse)
- ``wtc4d.recon``       gaussian splat training (static per epoch, then 4D)
"""

__version__ = "0.0.1"
