"""Readers and writers for external asset formats.

``load_fbx`` and ``save_fbx`` read and write a whole fbx file; ``FbxFile``
holds one open, read a kind at a time and saved back with its records'
changes."""

from cgmath.formats._load import FbxContents, load_fbx
from cgmath.formats._write import save_fbx
from cgmath.formats._fbx_file import FbxFile

__all__ = ["FbxContents", "FbxFile", "load_fbx", "save_fbx"]
