"""Unit tests for the :mod:`cgmath` package.

The suite runs under python, mayapy and the Maya GUI alike. Two things tell
them apart, and the helpers here hold both: a test that spawns a child
interpreter asks :func:`child_python` for one (inside Maya, ``sys.executable``
is Maya itself, and a child would be a second Maya), and a test that works a
live Maya scene asks :func:`fresh_scene` for one (in the GUI it spares a scene
with unsaved changes).
"""

import os
import sys
import unittest


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


def fresh_scene(test: unittest.TestCase):
    """``maya.cmds`` on a new, empty scene, left as a new scene when the test
    ends; the test skips outside Maya, and in the GUI while the open scene
    has unsaved changes, which the new scene would discard. The scene is
    the one open when the test asks for its first fresh scene: a test that
    asks again (one scene per take) is not stopped by its own changes."""
    try:
        import maya.cmds as cmds

        batch = cmds.about(batch=True)
    except Exception:  # not running in Maya
        test.skipTest("needs Maya")
    if not getattr(test, "_fresh_scene_opened", False):
        if not batch and cmds.file(query=True, modified=True):
            test.skipTest("the open scene has unsaved changes; save it or start a new scene to run the live-scene tests")
        test.addCleanup(cmds.file, new=True, force=True)
        test._fresh_scene_opened = True
    cmds.file(new=True, force=True)
    return cmds
