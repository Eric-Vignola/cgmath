"""A list of one mesh's records: the UV sets, materials or blend shape targets of a mesh."""

from typing import Optional

from cgmath.hierarchy.hierarchy import _is_uuid


class OneMeshList:
    """
    Mixin of a DataList whose records name their mesh by ``mesh_uuid``
    (UVList, MaterialList, MorphList). The list's own ``mesh_uuid`` is None
    for records of any mesh, as a file holds them, or the one mesh every
    record names, as a Renderable holds them. Set, it is held on every
    append, insert, extend and item assignment, and on setting it: a record
    of another mesh raises ValueError. Saved beside the records; ``==``
    leaves it out, as the records' ``==`` leaves theirs out.
    """

    # saved beside the records: the one mesh every record names, or None
    LIST_FIELDS          = ("mesh_uuid",)
    EQUALITY_TEST_IGNORE = ("mesh_uuid",)

    mesh_uuid: Optional[str] = None

    def __init__(self, iterable=None, mesh_uuid: Optional[str] = None):
        self.mesh_uuid = mesh_uuid  # first: each record is held to it
        super().__init__(iterable)

    def __setattr__(self, name: str, value) -> None:
        if name == "mesh_uuid" and value is not None:
            value = str(value)  # a uuid.UUID as its text
            if not _is_uuid(value):
                raise ValueError(f"{type(self).__name__}.mesh_uuid takes a uuid or None, not {value!r}")
            foreign = [item for item in getattr(self, "list", None) or [] if item.mesh_uuid != value]
            if foreign:
                raise ValueError(
                    f"{type(self).__name__}.mesh_uuid {value!r}: the list holds {len(foreign)} record(s) naming "
                    f"another mesh ({_names(foreign)}); a list of one mesh holds that mesh's records alone"
                )
        super().__setattr__(name, value)

    def _held(self, value) -> None:
        """``value`` may join: it names the list's mesh, when the list has one"""
        mine = self.mesh_uuid
        if mine is not None and isinstance(value, self.DATA_LIST_CLASS) and value.mesh_uuid != mine:
            raise ValueError(
                f"{type(self).__name__} of mesh {mine!r} takes records naming it; {_name(value)} names {value.mesh_uuid!r}"
            )

    def append(self, value):
        self._held(value)
        super().append(value)

    def insert(self, index, value):
        self._held(value)
        super().insert(index, value)

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            value = list(value)
            for item in value:
                self._held(item)
        else:
            self._held(value)
        super().__setitem__(index, value)


def _name(record) -> str:
    """"UVData 'map1'", or "an unnamed UVData" """
    name = getattr(record, "name", None)
    return f"{type(record).__name__} {name!r}" if name else f"an unnamed {type(record).__name__}"


def _names(records) -> str:
    """the first three records by :func:`_name`, "..." past them"""
    shown = [_name(record) for record in records[:3]]
    return ", ".join(shown) + (", ..." if len(records) > 3 else "")
