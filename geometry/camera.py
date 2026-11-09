"""A camera: Maya's camera attributes, a shape placed by its transform.

A :class:`CameraData` holds what Maya's camera shape holds: the focal length
in millimetres, the film apertures and offsets in inches, the film fit and
the lens squeeze, the camera scale, the clip planes in centimetres,
orthographic and its width. It sits under the transform its ``parent_uuid``
names and looks down that transform's -Z with +Y up, as a Maya camera does;
the angle of view is derived, as Maya derives it. The defaults are a fresh
Maya camera's (``cmds.camera()`` in Maya 2025, measured): the scene's
``persp`` camera differs only in its orthographic width, 30.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from cgmath.data import Array, DataList, ImmutableArray as numpy_array
from cgmath.hierarchy import ShapeData

# Maya's filmFit, in its order: 0 fill, 1 horizontal, 2 vertical, 3 overscan
FILM_FITS = ("fill", "horizontal", "vertical", "overscan")

# millimetres in an inch: the apertures are in inches, the focal length in mm
INCH = 25.4


@dataclass(repr=False, eq=False)
class CameraData(ShapeData):
    """
    A camera, as Maya's camera shape holds one; a shape placed by the
    transform its ``parent_uuid`` names (:class:`ShapeData`), looking down
    that transform's -Z, +Y up.

    Parameters
    ----------
    name : str, optional
        The shape's name, None for none.
    focal_length : float
        Maya's focalLength, in mm. Default 35.
    horizontal_film_aperture, vertical_film_aperture : float
        Maya's film back, in inches. Default 1.41732 by 0.94488 (36 by 24
        mm).
    film_fit : str
        How the film back fits the image: "fill", "horizontal", "vertical"
        or "overscan", Maya's filmFit 0 to 3. Default "horizontal".
    film_offset : (2,) float64
        Maya's horizontal and vertical film offsets, in inches. Default 0.
    lens_squeeze_ratio : float
        Maya's lensSqueezeRatio, an anamorphic lens's. Default 1.
    camera_scale : float
        Maya's cameraScale, which scales the field of view. Default 1. FBX
        holds no such value; Maya's own export drops it, and so does
        ``save_fbx``, with a warning when it is not 1.
    near_clip_plane, far_clip_plane : float
        Maya's clip planes, in cm. Default 0.1 and 10000.
    orthographic : bool
        Parallel rays when True. Default False.
    orthographic_width : float
        The view's width in cm when orthographic. Default 10, a fresh
        camera's; Maya's persp camera holds 30.
    """

    EQUALITY_TEST_IGNORE = ShapeData.EQUALITY_TEST_IGNORE + ["name"]

    name:                     Optional[str] = None
    focal_length:             float = 35.0
    horizontal_film_aperture: float = 1.41732
    vertical_film_aperture:   float = 0.94488
    film_fit:                 str = "horizontal"
    film_offset:              Array(np.float64, 2) = numpy_array([0.0, 0.0])
    lens_squeeze_ratio:       float = 1.0
    camera_scale:             float = 1.0
    near_clip_plane:          float = 0.1
    far_clip_plane:           float = 10000.0
    orthographic:             bool = False
    orthographic_width:       float = 10.0

    def __setattr__(self, name: str, value) -> None:
        if name == "film_fit" and value not in FILM_FITS:
            raise ValueError(f"CameraData.film_fit is one of {list(FILM_FITS)}, not {value!r}")
        super().__setattr__(name, value)

    @property
    def horizontal_angle_of_view(self) -> float:
        """Maya's horizontal angle of view, in degrees: the horizontal film
        aperture, squeezed and scaled, against the focal length"""
        aperture = self.horizontal_film_aperture * INCH * self.lens_squeeze_ratio * self.camera_scale
        return math.degrees(2.0 * math.atan(aperture / (2.0 * self.focal_length)))

    @property
    def vertical_angle_of_view(self) -> float:
        """Maya's vertical angle of view, in degrees: the vertical film
        aperture, scaled, against the focal length"""
        aperture = self.vertical_film_aperture * INCH * self.camera_scale
        return math.degrees(2.0 * math.atan(aperture / (2.0 * self.focal_length)))

    @property
    def angle_of_view(self) -> float:
        """Maya's angleOfView: the horizontal one"""
        return self.horizontal_angle_of_view

    @property
    def film_aspect_ratio(self) -> float:
        """the film back's width over its height"""
        return self.horizontal_film_aperture / self.vertical_film_aperture


class CameraList(DataList[CameraData]):
    """A list of :class:`CameraData`, as a file holds its cameras."""
