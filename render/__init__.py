"""
Software raytracer rendering pipeline for ``MeshData`` / ``UVData`` pairs.

Public API (front door):

    from cgmath.render import render, Scene, Object, Camera, Light, Frame, look_at

Submodules:
  - :mod:`cgmath.render.scene`     scene-graph: Scene, Object, Camera, Light
  - :mod:`cgmath.render.raytracer` algorithm: ``render()`` and shading pipeline
  - :mod:`cgmath.render.camera`    camera math: look_at, default-view, autofit
  - :mod:`cgmath.render.texture`   texture I/O + bilinear sampler
  - :mod:`cgmath.render.frame`     ``Frame`` result wrapper (array + depth + .save())
"""

from __future__ import annotations

from cgmath.render.camera import look_at
from cgmath.render.frame import Frame
from cgmath.render.raytracer import render
from cgmath.render.scene import Camera, Light, Object, Scene
from cgmath.render.texture import _load_texture as load_texture

__all__ = [
    "render",
    "Frame",
    "Scene",
    "Object",
    "Camera",
    "Light",
    "look_at",
    "load_texture",
]