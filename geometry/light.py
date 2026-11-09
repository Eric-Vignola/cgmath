"""A light: Maya's light attributes, a shape placed by its transform.

A :class:`LightData` is a point, directional or spot light as Maya's light
shapes hold one: colour, intensity, decay rate, and a spot's cone, penumbra
and dropoff. It sits under the transform its ``parent_uuid`` names; a
directional or spot light shines down that transform's -Z, as a Maya light
does. The defaults are a fresh Maya light's (measured in Maya 2025).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from cgmath.data import Array, DataList, ImmutableArray as numpy_array
from cgmath.hierarchy import ShapeData

# the kinds a LightData holds; Maya's ambient, area and volume lights are
# none of them
LIGHT_KINDS = ("point", "directional", "spot")

# Maya's decayRate, in its order: 0 none, 1 linear, 2 quadratic, 3 cubic
DECAY_RATES = ("none", "linear", "quadratic", "cubic")


@dataclass(repr=False, eq=False)
class LightData(ShapeData):
    """
    A light, as Maya's light shapes hold one; a shape placed by the
    transform its ``parent_uuid`` names (:class:`ShapeData`), a directional
    or spot light shining down that transform's -Z.

    Parameters
    ----------
    name : str, optional
        The shape's name, None for none.
    kind : str
        "point", "directional" or "spot".
    color : (3,) float64
        Maya's color, RGB in 0 to 1. Default white.
    intensity : float
        Maya's intensity. Default 1.
    decay_rate : int
        Maya's decayRate: 0 none, 1 linear, 2 quadratic, 3 cubic. Default 0.
    cone_angle : float
        A spot's cone angle in degrees, the whole cone. Default 40.
    penumbra_angle : float
        A spot's penumbra in degrees, added to the cone's edge. Default 0.
    dropoff : float
        A spot's dropoff. Default 0. FBX holds no such value; Maya's own
        export drops it, and so does ``save_fbx``, with a warning when it
        is not 0.
    """

    EQUALITY_TEST_IGNORE = ShapeData.EQUALITY_TEST_IGNORE + ["name"]

    name:           Optional[str] = None
    kind:           str = "point"
    color:          Array(np.float64, 3) = numpy_array([1.0, 1.0, 1.0])
    intensity:      float = 1.0
    decay_rate:     int = 0
    cone_angle:     float = 40.0
    penumbra_angle: float = 0.0
    dropoff:        float = 0.0

    def __setattr__(self, name: str, value) -> None:
        if name == "kind" and value not in LIGHT_KINDS:
            raise ValueError(f"LightData.kind is one of {list(LIGHT_KINDS)}, not {value!r}")
        if name == "decay_rate":
            if isinstance(value, bool) or int(value) != value or not 0 <= int(value) < len(DECAY_RATES):
                raise ValueError(f"LightData.decay_rate is Maya's decayRate, 0 to 3 ({', '.join(DECAY_RATES)}), not {value!r}")
            value = int(value)
        super().__setattr__(name, value)

    @property
    def decay(self) -> str:
        """the decay rate's name: none, linear, quadratic or cubic"""
        return DECAY_RATES[self.decay_rate]


class LightList(DataList[LightData]):
    """A list of :class:`LightData`, as a file holds its lights."""
