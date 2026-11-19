"""``cgmath.formats.FbxFile``: an fbx file held open, read a kind at a time

The SDK's own scene stays open behind the handle: each kind of the file is
translated the first time it is asked for, and :meth:`FbxFile.save` writes
that scene back through the SDK's writer, so everything the fork does not
read (bind poses, LOD groups, user properties, embedded media ...) is kept.
A record changed since it was read is written into that scene first
(:mod:`cgmath.formats._fbx_edit`), in the file's own axes, units and pivots.
"""

import contextlib
import hashlib
import os
import shutil
import tempfile
import warnings

import numpy as np

from cgmath.formats import _fbx_edit, _load
from cgmath.formats._fbx_io import export_scene, import_scene, media_folder, new_manager, short_path

__all__ = ["FbxFile"]

# the reads a property tries, widest first: a read of the nodes with their
# takes, cameras, lights and bind pose; one of the meshes with everything a
# mesh's read serves; one of the NURBS shapes
_NODES  = frozenset(("takes", "cameras", "lights", "binds"))
_WIDE   = frozenset(("takes", "meshes", "uvs", "skins", "morphs", "cameras", "lights"))
_SHAPES = frozenset(("curves", "surfaces"))

# a property -> where its list's class lives, the class, and the record's
# class a lone one is wrapped from (as save_fbx wraps it)
_KINDS = {
    "hierarchy": ("cgmath.hierarchy.hierarchy", "HierarchyData", None),
    "clip":      ("cgmath.hierarchy.hierarchy", "ClipList", "ClipData"),
    "meshes":    ("cgmath.geometry.mesh", "MeshList", "MeshData"),
    "skins":     ("cgmath.geometry.skin_weights", "SkinList", "SkinData"),
    "uvs":       ("cgmath.geometry.mesh", "UVList", "UVData"),
    "cameras":   ("cgmath.geometry.camera", "CameraList", "CameraData"),
    "lights":    ("cgmath.geometry.light", "LightList", "LightData"),
    "materials": ("cgmath.geometry.material", "MaterialList", "MaterialData"),
    "morphs":    ("cgmath.geometry.morph_target", "MorphList", "MorphData"),
    "curves":    ("cgmath.geometry.bspline", "NurbsCurveList", "NurbsCurveData"),
    "surfaces":  ("cgmath.geometry.bspline_patch", "NurbsSurfaceList", "NurbsSurfaceData"),
}


def _plans(wide, *needs) -> tuple:
    """(need, read) pairs a property tries in order: its first need off the
    wide read, then each need off a read of its own kinds alone"""
    plans = [(frozenset(needs[0]), wide)]
    plans.extend((frozenset(need), frozenset(need)) for need in needs)
    return tuple(plans)


# a property's needs, in order: the first is load_fbx's own field, each
# later one what is left to read when a kind before it is refused
_PLANS = {
    "hierarchy": _plans(_NODES, ("takes", "binds"), ("binds",), ("takes",), ()),
    "clip":      _plans(_NODES, ("takes",)),
    "cameras":   _plans(_NODES, ("takes", "cameras"), ("cameras",)),
    "lights":    _plans(_NODES, ("takes", "lights"), ("lights",)),
    "meshes":    _plans(_WIDE, ("meshes",)),
    "uvs":       _plans(_WIDE, ("meshes", "uvs")),
    "skins":     _plans(_WIDE, ("meshes", "skins")),
    "morphs":    _plans(_WIDE, ("takes", "meshes", "morphs"), ("meshes", "morphs")),
    "curves":    _plans(_SHAPES, ("curves",)),
    "surfaces":  _plans(_SHAPES, ("surfaces",)),
}

# what a refused read or build raises
_REFUSALS = (ValueError, NotImplementedError)


def _covers(kinds: frozenset, need: frozenset) -> bool:
    """whether a read of ``kinds`` serves ``need``: the bind pose comes
    from a read of the skins alone, or of the meshes with their skins"""
    if not need - {"binds"} <= kinds:
        return False
    return "binds" not in need or "binds" in kinds or {"meshes", "skins"} <= kinds


def _again(error: BaseException) -> BaseException:
    """a fresh copy of a refusal raised before, its traceback not grown"""
    try:
        return type(error)(*error.args)
    except Exception:
        return error.with_traceback(None)


class _Raw:
    """one read of the open scene (:func:`_load._read`), and what every
    build of it starts from: its hierarchy and the nodes' order, its shapes
    sorted into that order (:func:`_load._sorted`), and its bind pose"""

    _UNREAD = object()

    def __init__(self, read: dict):
        self.read    = read
        self._sorted = None
        self._bind   = self._UNREAD

    def sorted(self) -> tuple:
        if self._sorted is None:
            self._sorted = _load._sorted(self.read)
        return self._sorted

    @property
    def hierarchy(self):
        return self.sorted()[0]

    @property
    def order(self):
        return self.sorted()[1]

    def bind(self):
        """the read's hierarchy at its skins' bind pose, None for no skin;
        a refusal is raised again on every call"""
        if self._bind is self._UNREAD:
            self.sorted()
            self._bind = _load._bind(self.read)
        return self._bind


def _feed(digest, value, seen: dict) -> None:
    """``value``, a ``to_dict()`` tree, into ``digest``: keys sorted, an
    array by dtype, shape and bytes, one array object hashed once"""
    if isinstance(value, dict):
        digest.update(b"{")
        for key in sorted(value, key=repr):
            _feed(digest, key, seen)
            _feed(digest, value[key], seen)
        digest.update(b"}")
    elif isinstance(value, (list, tuple)):
        digest.update(b"[")
        for item in value:
            _feed(digest, item, seen)
        digest.update(b"]")
    elif isinstance(value, np.ndarray):
        held = seen.get(id(value))
        if held is None:
            one = hashlib.blake2b(digest_size=16)
            if value.dtype.hasobject:
                _feed(one, value.tolist(), seen)
            else:
                one.update(f"a{value.dtype.str}{value.shape}".encode())
                one.update(np.ascontiguousarray(value).reshape(-1).view(np.uint8))
            # the array kept alive with its mark, so its id is no other's
            held = seen[id(value)] = (value, one.digest())
        digest.update(held[1])
    elif isinstance(value, (bytes, bytearray)):
        digest.update(b"b%d:" % len(value))
        digest.update(value)
    elif hasattr(value, "to_dict"):
        _feed(digest, value.to_dict(), seen)
    else:
        digest.update(f"{value!r};".encode())


def _mark(tree, seen: dict) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    _feed(digest, tree, seen)
    return digest.digest()


def _marks(value):
    """what a property's value held when read, to tell a change by: one
    mark the record, or for a list one its own fields and one a record,
    with the record's name; None for None"""
    if value is None:
        return None
    seen  = {}
    tree  = value.to_dict()
    items = tree.get("__items__") if isinstance(tree, dict) else None
    if items is None:
        return _mark(tree, seen), None
    rest = {key: held for key, held in tree.items() if key != "__items__"}
    return _mark(rest, seen), [(item.get("name") if isinstance(item, dict) else None, _mark(item, seen)) for item in items]


class FbxFile:
    """
    An fbx file held open, each of its kinds read the first time it is
    asked for, its records' changes written back into it.

    ``FbxFile(path)`` opens the file through the FBX SDK and holds it open
    until :meth:`close`, or the end of a ``with`` block. Each of its eleven
    properties, named as :class:`FbxContents`' fields (``hierarchy``,
    ``clip``, ``meshes``, ``skins``, ``uvs``, ``cameras``, ``lights``,
    ``materials``, ``morphs``, ``curves``, ``surfaces``), is translated the
    first time it is read and kept: it equals the same field of
    :func:`load_fbx` (``load_fbx(path, textures=False)`` for a handle opened
    with ``textures`` False), in whatever order the properties are read,
    every node holding the same uuid in all of them, so a shape's
    ``parent_uuid`` and a UV set's or skin's ``mesh_uuid`` name records of
    the same handle. A property reads what it depends on alone:
    ``hierarchy`` builds no mesh, ``meshes`` no material. ``textures``
    False decodes no pixels (each material's ``texture`` None, as
    ``load_fbx`` holds it), the file's textures kept in a save; ``verbose``
    False silences the reads' warnings, each of which a handle otherwise
    shows once.

    :meth:`save` writes the file through the SDK's own writer: in the
    file's own axes, units and pivots, with everything the fork does not
    read (bind poses, LOD groups, user properties ...) and the file's
    embedded media. A record changed since it was read, in place or by
    giving its property another list, is written into the open file first,
    converted back to the file's axes, units and pivots: the values of the
    records the file holds (a node's transform, a take's keys, a mesh's
    points, a skin's weights, a material's colours ...).

    A change of the file's layout is written too, for the nodes, the meshes
    and the records that name a mesh: a node added, taken off, put under
    another parent or made another kind (``set_node_type()``), in the
    hierarchy and the takes; user defined attributes added or taken off; a mesh added (on the node its ``parent_uuid``
    names), taken off or given other faces; UV sets, skins, influences and
    materials added or taken off. A mesh taken off takes its UV sets, skin,
    blend shapes and material assignments with it, and the materials and
    textures no mesh is left with: after the save the handle's lists hold
    what the file holds. Its node stays. For the other kinds (cameras,
    lights, curves, surfaces, blend shapes, takes) a record added or
    removed is refused, naming ``save_fbx``, which writes any records; so
    are another texture, and a change the file would not read back as the
    records hold it. ``force=True`` lets a change drop data of the file
    that no record holds and the change leaves stale, a warning naming each
    item; without it such a change is refused.

    A file ``load_fbx`` refuses still opens and saves whole. The property
    reading the refused data raises the refusal, every time it is read; the
    others read what they can, as ``load_fbx`` with the refused kind off
    would, or more (the cameras keep their keyed values when only the clip
    is refused). A closed handle keeps the values it read; a property it
    had not read, or a save, raises ValueError. An FbxFile can be neither
    pickled nor copied: its properties' records can.
    """

    def __init__(self, filename, textures: bool = True, verbose: bool = True):
        # imported here: the classes' modules import the reader
        from cgmath.hierarchy.hierarchy import FBX

        # what close() lets go, held before anything here can raise
        self._manager  = None
        self._scene    = None
        self._media    = None
        self._mesh_raw = None
        self._reads    = {}  # the kinds of a read -> its _Raw
        self._refused  = {}  # the kinds of a read that raised -> its error
        if FBX is None:
            raise ImportError("fbx module not found")
        path = os.path.expanduser(os.fspath(filename))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

        # embedded media are always extracted, so a save can embed them again
        with _load._quiet(verbose):
            folder = media_folder()
        try:
            manager = new_manager(FBX, folder or True)
            try:
                scene = import_scene(manager, path, FBX)
            except BaseException:
                manager.Destroy()
                raise
        except BaseException:
            if folder is not None:
                shutil.rmtree(folder, ignore_errors=True)
            raise

        self._fbx            = FBX
        self._manager        = manager
        self._scene          = scene
        self._media          = folder
        self._filename       = path
        self._origin         = path  # the file the open scene was last read from or saved to
        self._textures       = bool(textures)
        self._verbose        = bool(verbose)
        self._uuids          = None  # one a node, in the scene's own order
        self._refusals       = {}    # a property -> the error it raised
        self._values         = {}    # a property -> its value, once read
        self._digests        = {}    # a property -> its marks when read
        self._warned         = set()
        self._hearing        = False
        self._issued         = {}    # the uuid of each mesh the handle read -> its node's
        self._settled        = {}    # a property -> the list a save in progress settles it as
        self._material_uuids = None  # the uuid of each material record, in the order the file lists them

    # ---------------------------------------------------------------- state

    @property
    def filename(self) -> str:
        """the file's path, as given, ``~`` expanded"""
        return self._filename

    @property
    def closed(self) -> bool:
        """whether the file is closed (:meth:`close`)"""
        return self._manager is None

    def close(self) -> None:
        """
        Lets the file go: the SDK's scene and the media extracted from it
        are freed. The values read stay usable; a property not read yet, or
        a save, raises ValueError. Closing a closed handle does nothing.
        """
        manager, self._manager = self._manager, None
        self._scene    = None
        self._mesh_raw = None
        self._reads.clear()
        self._refused.clear()
        if manager is not None:
            manager.Destroy()
        media, self._media = self._media, None
        if media is not None:
            shutil.rmtree(media, ignore_errors=True)

    def __enter__(self) -> "FbxFile":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"FbxFile({self._filename!r}{', closed' if self.closed else ''})"

    def __reduce__(self):
        raise TypeError("an FbxFile holds an open fbx file: it can be neither pickled nor copied; its properties' records can")

    def __copy__(self):
        self.__reduce__()

    def __deepcopy__(self, memo):
        self.__reduce__()

    # ----------------------------------------------------------- the kinds

    @property
    def hierarchy(self):
        """``load_fbx``'s ``hierarchy``: every node at its own values, the
        skins' bind pose where a mesh is skinned; None where the file holds
        a clip, whose nodes ``clip`` holds, or no node"""
        return self._get("hierarchy")

    @hierarchy.setter
    def hierarchy(self, value) -> None:
        self._set("hierarchy", value)

    @property
    def clip(self):
        """``load_fbx``'s ``clip``: a ClipList of the file's takes over
        every node, its active take the file's; None for no take"""
        return self._get("clip")

    @clip.setter
    def clip(self, value) -> None:
        self._set("clip", value)

    @property
    def meshes(self):
        """``load_fbx``'s ``meshes``: a MeshList in their nodes' order, each
        on its node by ``parent_uuid``; None for none"""
        return self._get("meshes")

    @meshes.setter
    def meshes(self, value) -> None:
        self._set("meshes", value)

    @property
    def skins(self):
        """``load_fbx``'s ``skins``: a SkinList, one a skinned mesh, each
        naming its mesh by ``mesh_uuid``; None for none"""
        return self._get("skins")

    @skins.setter
    def skins(self, value) -> None:
        self._set("skins", value)

    @property
    def uvs(self):
        """``load_fbx``'s ``uvs``: a UVList of every mesh's UV sets, each
        naming its mesh by ``mesh_uuid``; None for none"""
        return self._get("uvs")

    @uvs.setter
    def uvs(self, value) -> None:
        self._set("uvs", value)

    @property
    def cameras(self):
        """``load_fbx``'s ``cameras``: a CameraList in their nodes' order;
        None for none"""
        return self._get("cameras")

    @cameras.setter
    def cameras(self, value) -> None:
        self._set("cameras", value)

    @property
    def lights(self):
        """``load_fbx``'s ``lights``: a LightList in their nodes' order;
        None for none"""
        return self._get("lights")

    @lights.setter
    def lights(self, value) -> None:
        self._set("lights", value)

    @property
    def materials(self):
        """``load_fbx``'s ``materials``: a MaterialList, mesh by mesh, each
        naming its mesh by ``mesh_uuid``, its pixels decoded unless the
        handle was opened with ``textures`` False; None for none"""
        return self._get("materials")

    @materials.setter
    def materials(self, value) -> None:
        self._set("materials", value)

    @property
    def morphs(self):
        """``load_fbx``'s ``morphs``: a MorphList of the meshes' blend
        shape targets, each naming its mesh by ``mesh_uuid``; None for
        none"""
        return self._get("morphs")

    @morphs.setter
    def morphs(self, value) -> None:
        self._set("morphs", value)

    @property
    def curves(self):
        """``load_fbx``'s ``curves``: a NurbsCurveList in their nodes'
        order; None for none"""
        return self._get("curves")

    @curves.setter
    def curves(self, value) -> None:
        self._set("curves", value)

    @property
    def surfaces(self):
        """``load_fbx``'s ``surfaces``: a NurbsSurfaceList in their nodes'
        order; None for none. A trimmed surface raises ValueError here"""
        return self._get("surfaces")

    @surfaces.setter
    def surfaces(self, value) -> None:
        self._set("surfaces", value)

    # ---------------------------------------------------------------- save

    def save(self, filename, as_ascii: bool = False, force: bool = False) -> str:
        """
        Writes the file to ``filename``, binary or ASCII with ``as_ascii``,
        through the SDK's own writer: in the file's own axes, units and
        pivots, with its embedded media, everything the fork does not read
        kept. A record changed since it was read, in place or by giving
        its property another value, is written into the file first; the
        file is then read again, and a change that would not read back as
        the records hold it is refused. The file is written beside its
        target and renamed over it, so a failed save leaves the target as
        it was; a folder of the target that does not exist is made
        (``save_fbx`` makes none). Returns the path written.

        ``force`` lets a change drop data of the file the records do not
        hold, one warning naming each item dropped; without it such a
        change is refused. A refused save (ValueError, TypeError or
        NotImplementedError) writes nothing and leaves the handle as it was
        before the call, the records as they were given.

        The SDK's writer adds one empty FbxVideo, once, to a file holding a
        material's binding table (Maya's standardSurface) beside a file
        texture: nothing is lost, and a save of the save adds none.
        """
        self._open_or_raise("save")
        changed = [
            field for field in _fbx_edit.FIELDS
            if field in self._digests and _marks(self._values[field]) != self._digests[field]
        ]
        path    = os.path.abspath(os.path.expanduser(os.fspath(filename)))
        folder  = os.path.dirname(path)
        dropped = []
        began   = []
        try:
            if changed:
                dropped = _fbx_edit.apply(self, changed, force, began)
            os.makedirs(folder, exist_ok=True)
            handle, temp = tempfile.mkstemp(prefix=".cgmath_", suffix=".fbx", dir=folder)
            os.close(handle)
            try:
                # Maya's fbx binding hands the SDK plain ASCII paths alone
                target = temp if temp.isascii() else short_path(temp)
                export_scene(self._manager, self._scene, target, self._fbx, as_ascii=as_ascii, embed=self._embeds())
                os.replace(temp, path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.remove(temp)
                raise
        except BaseException:
            # a change part written, or written and not saved: the scene as it was
            self._settled = {}
            if began:
                self._restore()
            raise

        # the lists as the file now holds them: a mesh's records gone with it
        self._values.update(self._settled)
        self._settled = {}
        if changed:
            for mesh in self._values.get("meshes") or ():
                self._issued[mesh.uuid] = mesh.parent_uuid
            if "materials" in self._values:
                self._material_uuids = [material.uuid for material in self._values["materials"] or ()]
            # the open scene is the file just written: what was read of it is read anew
            self._origin   = path
            self._mesh_raw = None
            self._reads.clear()
            self._refused.clear()
            self._digests = {field: _marks(value) for field, value in self._values.items()}
        if dropped:
            warnings.warn(f"FbxFile.save: with force, {path!r} is written without {'; '.join(dropped)}", stacklevel=2)
        return path

    def _embeds(self) -> bool:
        """whether the file held embedded media, extracted when it was
        opened: a save embeds them again"""
        folder = self._media
        if folder is None:
            folder = os.path.splitext(self._filename)[0] + ".fbm"
        return os.path.isdir(folder) and any(files for _, _, files in os.walk(folder))

    # ------------------------------------------------------------- reading

    def _open_or_raise(self, what: str) -> None:
        if self._manager is None:
            raise ValueError(f"FbxFile.{what}: {self._filename!r} is closed; open it again")

    def _set(self, field: str, value) -> None:
        """a property given another value: its list, one record of it
        (wrapped in its list, as ``save_fbx`` wraps it) or None. The
        property is read first, so a save tells what changed of the file"""
        import importlib

        self._get(field)
        module, many, one = _KINDS[field]
        module = importlib.import_module(module)
        many   = getattr(module, many)
        if one is not None and type(value) is getattr(module, one):
            value = many([value])
        if value is not None and type(value) is not many:
            wrapped = f", a {one}" if one is not None else ""
            raise TypeError(f"FbxFile.{field} is a {many.__name__}{wrapped} or None, not a {type(value).__name__}")
        self._values[field] = value

    def _restore(self) -> None:
        """the scene read again off the file it was last read from or saved
        to, after a change was part written into it. The records and the
        nodes' uuids are kept; a file gone since closes the handle"""
        manager, self._manager = self._manager, None
        self._scene    = None
        self._mesh_raw = None
        self._reads.clear()
        self._refused.clear()
        if manager is not None:
            manager.Destroy()
        try:
            manager = new_manager(self._fbx, self._media or True)
            try:
                scene = import_scene(manager, self._origin, self._fbx)
            except BaseException:
                manager.Destroy()
                raise
        except BaseException:
            self.close()
            raise
        self._manager = manager
        self._scene   = scene

    def _get(self, field: str):
        """a property's value: kept once read; a refusal raised again"""
        if field in self._values:
            return self._values[field]
        if field in self._refusals:
            raise _again(self._refusals[field])
        self._open_or_raise(field)
        with self._heard():
            try:
                value = self._materials() if field == "materials" else self._built(field)
            except _REFUSALS as error:
                self._refusals[field] = error
                raise
        self._values[field]  = value
        self._digests[field] = _marks(value)
        return value

    @contextlib.contextmanager
    def _heard(self):
        """the warnings of a property's reads and builds, each (category,
        text) shown once by the handle, none when it is not verbose; a
        property read while another is reading leaves it to the outer one"""
        if self._hearing:
            yield
            return
        self._hearing = True
        heard         = []
        try:
            with warnings.catch_warnings(record=True) as heard:
                warnings.simplefilter("always")
                yield
        finally:
            self._hearing = False
            if self._verbose:
                for held in heard:
                    key = (held.category, str(held.message))
                    if key not in self._warned:
                        self._warned.add(key)
                        warnings.warn_explicit(held.message, held.category, held.filename, held.lineno)

    def _built(self, field: str):
        """the property built off the first of its plans that reads and
        builds; the last refusal when none does"""
        error = None
        tried = set()
        for need, kinds in _PLANS[field]:
            try:
                raw = self._raw(need, kinds)
                if (id(raw), need) in tried:
                    continue
                tried.add((id(raw), need))
                raw.sorted()
                return getattr(self, f"_build_{field}")(raw, need)
            except _REFUSALS as refused:
                error = refused
        raise error

    def _raw(self, need: frozenset, kinds: frozenset) -> _Raw:
        """a read serving ``need``: one already made, else a read of
        ``kinds``; a read holding kinds a refused read held is refused
        again, unread"""
        for held, raw in self._reads.items():
            if _covers(held, need):
                return raw
        for held, error in self._refused.items():
            if held <= kinds:
                raise _again(error)
        if self._uuids is None:
            from cgmath.hierarchy.hierarchy import generate_uuid

            self._uuids = [generate_uuid() for _ in _load._walk(self._scene)]
        parts = dict(_load._PARTS_NODES, **{kind: True for kind in kinds})
        try:
            read = _load._read(self._scene, self._fbx, parts, self._filename, self._uuids)
        except _REFUSALS as error:
            self._refused[kinds] = error
            raise
        raw = self._reads[kinds] = _Raw(read)
        return raw

    # a property's build off one read, its shapes in their nodes' order

    def _build_hierarchy(self, raw: _Raw, need: frozenset):
        if "binds" in need:
            bind = raw.bind()
            if bind is not None:
                return bind
        if "takes" in need and raw.read["takes"] is not None and len(raw.hierarchy) and self.clip is not None:
            return None
        return raw.hierarchy if len(raw.hierarchy) else None

    def _build_clip(self, raw: _Raw, need: frozenset):
        # a hierarchy of the clip's own: the read's stays the nodes'
        hierarchy, order = _load._build(raw.read)
        return _load._clip(raw.read, hierarchy, order)

    def _build_meshes(self, raw: _Raw, need: frozenset):
        meshes         = _load._build_meshes(raw.read)
        self._mesh_raw = raw
        for mesh in meshes or ():
            self._issued[mesh.uuid] = mesh.parent_uuid
        return meshes

    def _build_uvs(self, raw: _Raw, need: frozenset):
        return _load._build_uvs(raw.read, self.meshes)

    def _build_skins(self, raw: _Raw, need: frozenset):
        if raw.bind() is None:
            return None
        return _load._build_skins(raw.read, self.meshes)

    def _build_morphs(self, raw: _Raw, need: frozenset):
        return _load._build_morphs(raw.read, self.meshes)

    def _build_cameras(self, raw: _Raw, need: frozenset):
        return _load._build_cameras(raw.read)

    def _build_lights(self, raw: _Raw, need: frozenset):
        return _load._build_lights(raw.read)

    def _build_curves(self, raw: _Raw, need: frozenset):
        return _load._build_curves(raw.read)

    def _build_surfaces(self, raw: _Raw, need: frozenset):
        return _load._build_surfaces(raw.read)

    def _materials(self):
        """the meshes' materials, read off the scene for the meshes read"""
        meshes = self.meshes
        if meshes is None:
            return None
        if self._mesh_raw is None:
            # the meshes were read before a save: read again, for their materials alone
            self._mesh_raw = self._raw(frozenset(("meshes",)), _WIDE)
            self._mesh_raw.sorted()
        filled               = _load._fill_materials(self._mesh_raw.read, self._scene, self._fbx, self._filename, self._textures)
        materials            = _load._build_materials(filled, meshes)
        self._material_uuids = [material.uuid for material in materials or ()]
        return materials
