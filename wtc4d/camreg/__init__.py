"""Camera registration: intrinsics + world-frame pose for footage frames.

See ``wtc4d/camreg/README.md`` for the full picture. Key entry points:

* :mod:`wtc4d.camreg.pnp` -- pose + focal from 2D-3D correspondences.
* :mod:`wtc4d.camreg.annotate` -- click landmarks on a frame.
* :mod:`wtc4d.camreg.render_match` -- annotation-free registration.
* :mod:`wtc4d.camreg.track` -- propagate a keyframe's pose through its shot.
* :mod:`wtc4d.camreg.priors` -- known vantage points (``data/cameras/priors.yaml``).
"""

from __future__ import annotations
