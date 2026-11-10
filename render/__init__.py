"""
Software raytracer rendering pipeline for the scene an fbx holds.

Public API (front door):

    from cgmath.render import render, Scene, Renderable, aim, Frame, look_at

Submodules:
  - :mod:`cgmath.render.scene`     Scene: the fbx fields and the render settings
  - :mod:`cgmath.render.raytracer` algorithm: ``render()`` and shading pipeline
  - :mod:`cgmath.render.camera`    camera math: look_at, default-view, autofit
  - :mod:`cgmath.render.texture`   texture I/O + bilinear sampler
  - :mod:`cgmath.render.frame`     ``Frame`` result wrapper (array + depth + .save())
"""

from __future__ import annotations

from cgmath.render.camera import look_at
from cgmath.render.frame import Frame
from cgmath.render.raytracer import render
from cgmath.render.scene import aim, Renderable, Scene
from cgmath.render.texture import _load_texture as load_texture

__all__ = [
    "render",
    "Frame",
    "Scene",
    "Renderable",
    "aim",
    "look_at",
    "load_texture",
]
