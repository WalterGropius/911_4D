"""Procedural 4D baseline: a parametric, time-dependent model of the WTC
scene (towers, damage, fire, smoke, dust, rubble) sampled into 3D gaussians
for any project time ``t`` -- a kinematic sketch, not a simulation.

See ``README.md`` for the model description and limitations, and
``params.yaml`` for every physical/artistic parameter and its source.
"""

from wtc4d.procedural.gaussians import LAYER_NAMES, GaussianCloud, sample
from wtc4d.procedural.params import DEFAULT_PARAMS, Params, load_params

__all__ = [
    "DEFAULT_PARAMS",
    "GaussianCloud",
    "LAYER_NAMES",
    "Params",
    "load_params",
    "sample",
]
