"""Unit tests for the :mod:`cgmath` package.

The suite runs on a plain Python with the packages installed, under mayapy
and in the Maya GUI alike, and nothing in it needs Maya: what Maya's own
import, export and evaluation do was recorded once, under mayapy, into
``test_assets/maya2025`` (:mod:`cgmath._tests.maya_fixtures`,
``tools/make_maya_fbx_fixtures.py``), and the tests hold cgmath to the
recording. One thing still tells the interpreters apart: a test that spawns
a child interpreter asks :func:`child_python` for one, since inside Maya
``sys.executable`` is Maya itself and a child would be a second Maya.
"""

import os
import sys


def child_python() -> str:
    """the interpreter a test spawns a child with: this one, unless this one
    is Maya itself (in the GUI and under ``maya -batch`` ``sys.executable`` is
    maya.exe, and a child would be a second Maya): then the mayapy beside it,
    or this one again where none is found"""
    exe  = sys.executable
    name = os.path.basename(exe).lower()
    if name.startswith("maya") and not name.startswith("mayapy"):
        folder = os.path.dirname(exe)
        for mayapy in (os.path.join(folder, "mayapy" + os.path.splitext(exe)[1]), os.path.join(folder, "mayapy")):
            if os.path.isfile(mayapy):
                return mayapy
    return exe


def child_env(**extra) -> dict:
    """``os.environ`` with PYTHONPATH set to this process's sys.path, so a
    child imports what this one does (a mayapy child has no venv until
    maya.standalone puts one on its path), and ``extra`` on top"""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(x for x in sys.path if x))
    env.update(extra)
    return env
