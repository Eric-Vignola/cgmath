"""Readers and writers for external asset formats."""

from cgmath.formats._load import load_fbx
from cgmath.formats._write import save_fbx

__all__ = ["load_fbx", "save_fbx"]
