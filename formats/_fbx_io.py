"""
The one place cgmath creates an FBX SDK manager.

Every object the SDK hands out (scene, nodes, meshes, takes) belongs to the
``FbxManager`` that read it, and only ``Destroy()`` on that manager frees it:
the Python wrapper going away frees nothing. A reader that forgets the call
leaks the whole scene on every read, so readers open files through
:func:`open_fbx`, which destroys the manager however the ``with`` block ends.

Nothing obtained inside the block may be used after it. Copy what you need
out to numpy arrays and strings first.

This module imports nothing from cgmath, so any module can use it without an
import cycle. The caller passes its own ``fbx`` module, which lets tests swap
a fake SDK in per module.
"""

import contextlib
import os
import shutil
import tempfile
import warnings

__all__ = ["FbxReadError", "import_scene", "media_folder", "new_manager", "open_fbx", "typed"]

# IOSettings paths, spelled out: Maya's fbx binding has no IMP_* constants.
# Extraction defaults to on, which writes a <name>.fbm folder beside the file.
EXTRACT_MEDIA  = "Import|AdvOptGrp|FileFormat|Fbx|ExtractEmbeddedData"
EXTRACT_FOLDER = "Import|IncludeGrp|EmbedTexture|ExtractFolder"

SDK_MISSING = "Autodesk FBX Python SDK is not installed"


class FbxReadError(RuntimeError):
    """the FBX SDK could not open or import a file"""


def media_folder():
    """
    A new temporary folder for extracted media, or None when its path is not
    plain ASCII: Maya's fbx binding cannot hand any other string to the SDK.
    """
    folder = tempfile.mkdtemp(prefix="cgmath_fbm_")
    if folder.isascii():
        return folder

    os.rmdir(folder)
    warnings.warn(
        f"the FBX SDK cannot use the temp folder {os.path.dirname(folder)!r}, whose "
        "path is not plain ASCII, so embedded media are extracted beside the FBX "
        "file instead; point TEMP at an ASCII folder to avoid it",
        RuntimeWarning,
        stacklevel=3,
    )
    return None


def new_manager(sdk, media=None):
    """
    An ``FbxManager`` with its IO settings. ``media`` says where embedded
    media (textures packed inside the file) are extracted on import: into a
    folder, beside the file as the SDK's own ``<name>.fbm`` when True, or
    nowhere when None.
    """
    manager = sdk.FbxManager.Create()
    try:
        ios = sdk.FbxIOSettings.Create(manager, sdk.IOSROOT)
        ios.SetBoolProp(EXTRACT_MEDIA, media is not None)
        if isinstance(media, str):
            ios.SetStringProp(EXTRACT_FOLDER, media)
        manager.SetIOSettings(ios)
    except BaseException:
        manager.Destroy()
        raise
    return manager


def import_scene(manager, filename, sdk):
    """
    Reads ``filename`` into a new ``FbxScene`` owned by ``manager``. On
    failure nothing is left in the manager and :class:`FbxReadError` names
    the SDK's own error.
    """
    importer = sdk.FbxImporter.Create(manager, "")
    scene    = None
    try:
        if not importer.Initialize(filename, -1, manager.GetIOSettings()):
            raise FbxReadError(
                f"Failed to initialize FBX importer: {importer.GetStatus().GetErrorString()}"
            )

        scene = sdk.FbxScene.Create(manager, "")
        if not importer.Import(scene):
            raise FbxReadError(
                f"Failed to import FBX scene: {importer.GetStatus().GetErrorString()}"
            )
    except BaseException:
        if scene is not None:
            scene.Destroy()
        raise
    finally:
        importer.Destroy()

    return scene


def typed(obj, kind, sdk):
    """
    ``obj`` as a ``kind`` wrapper. The binding can hand back a live wrapper
    of another type for an object that sits where a destroyed one did; going
    through ``FbxObject`` re-types it.
    """
    if obj is None or not isinstance(kind, type) or isinstance(obj, kind):
        return obj
    return sdk.cast(sdk.cast(obj, sdk.FbxObject), kind)


@contextlib.contextmanager
def open_fbx(filename, sdk, media=False):
    """
    Yields the ``FbxScene`` read from ``filename``; the manager, and with it
    everything read, is destroyed when the block ends.

    ``media`` extracts embedded media into a temporary folder that goes with
    the scene. Without it they are not extracted at all, so a read never
    writes next to the file.
    """
    if not sdk:
        raise ImportError(SDK_MISSING)

    filename = os.path.expanduser(filename)
    if not os.path.exists(filename):
        raise FileNotFoundError(f"FBX file not found: {filename}")

    folder = media_folder() if media else None
    try:
        manager = new_manager(sdk, (folder or True) if media else None)
        try:
            yield import_scene(manager, filename, sdk)
        finally:
            manager.Destroy()
    finally:
        if folder is not None:
            shutil.rmtree(folder, ignore_errors=True)
