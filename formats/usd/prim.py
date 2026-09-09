"""
UsdPrim utilities.
"""

from __future__ import annotations

import logging
import os
import re
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable, Iterator, List, Tuple, Union

from pxr import Sdf, Usd

LOGGER = logging.getLogger(__name__)
_PRIM_TYPES = Union[
    str, Usd.Typed, List[str], List[Usd.Typed], Tuple[str], Tuple[Usd.Typed]
]


def is_descendant(
    stage: Usd.Stage, prim: Union[Usd.Prim, str], ancestor: Union[Usd.Prim, str]
) -> bool:
    """
    # (TODO) Not fully tested
    Returns True if a prim is a descendant of another prim or a path.
    Args:
        stage (Usd.Stage): The USD stage containing the prims.
        prim (Union[Usd.Prim, str]): The prim or path to check.
        ancestor (Union[Usd.Prim, str]): The ancestor prim or path to check against.
    Returns:
        bool: True if the prim is a descendant of the ancestor, False otherwise.
    """
    if isinstance(prim, str):
        prim = stage.GetPrimAtPath(prim)
    if isinstance(ancestor, str):
        ancestor = stage.GetPrimAtPath(ancestor)
    ancestor_path = str(ancestor.GetPath()) + "/"
    if str(prim.GetPath()).startswith(ancestor_path):
        return True
    return False


def is_valid_prim(
    file_path: os.PathLike | str,
    prim_path: os.PathLike | str | None,
) -> bool:
    """Checks if a prim path exists in a given usd file.

    Args:
        file_path: A USD stage to search for the prim.
        prim_path: A prim to search for. If None, use the root prim.

    Returns:
        True if prim is vaild, otherwise False.
    """
    prim_path = prim_path or "/"
    if prim_path != "/":
        stage = Usd.Stage.Open(file_path)
        prim  = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            LOGGER.warning(f"Prim path {prim_path} not in {file_path}.")
            return False
    return True


def is_prim_type(prim, prim_types: _PRIM_TYPES) -> bool:
    """Checks if a prim matches the given types."""
    if not isinstance(prim_types, (list, tuple)):
        prim_types = [prim_types]
    if isinstance(prim_types[0], str):
        if prim.GetTypeName() in prim_types:
            return True
    else:
        for t in prim_types:
            if prim.IsA(t):
                return True
    return False


def iter_prims(
    obj:                    Usd.Stage   | Usd.Prim,
    prim_type:              _PRIM_TYPES | None     = None,
    prim_name:              str         | None     = None,
    first_only:             bool                   = False,
    skip_children_callback: Callable    | None     = None,
    match_prim_path:        str         | None     = None,
) -> Iterator[Usd.Prim]:
    """Iterates over prims that meet the given criteria.

    Args:
        obj: A stage or prim to search in.
        prim_type: One or more prim type to match.
        prim_name: A prim name to match.
        first_only: If True, only return the first prim found.
        skip_children_callback: Function that takes a prim as an arg
            and if returns True, will skip traversing children.
    """
    is_stage = isinstance(obj, Usd.Stage)
    it       = iter(Usd.PrimRange.Stage(obj) if is_stage else Usd.PrimRange(obj))
    for prim in it:
        status = True
        if prim_name is not None and prim.GetName() != str(prim_name):
            status = False
        elif prim_type is not None and not is_prim_type(prim, prim_type):
            status = False
        elif match_prim_path and not fnmatch(str(prim.GetPath()), match_prim_path):
            status = False

        if status:
            yield prim
            if first_only:
                break

        if skip_children_callback and skip_children_callback(prim):
            it.PruneChildren()


def find_prims(
    stage:                  Usd.Stage,
    prim_type:              _PRIM_TYPES | None = None,
    prim_name:              str         | None = None,
    first_only:             bool               = False,
    skip_children_callback: Callable    | None = None,
    match_prim_path:        str         | None = None,
) -> list[Usd.Prim] | Usd.Prim | None:
    """Returns a list of prims that meet the given criteria.

    Args:
        stage: A stage to search in.
        prim_type: A prim type to match.
        prim_name: A prim name to match.
        first_only: If True, only return the first prim found.
        skip_children_callback: Function that takes a prim as an arg
            and if returns True, will skip traversing children.
    """
    it = iter_prims(
        stage,
        prim_type=prim_type,
        prim_name=prim_name,
        first_only=first_only,
        skip_children_callback=skip_children_callback,
        match_prim_path=match_prim_path,
    )
    if first_only:
        return next(it, None)

    return list(it)


def duplicate_prim(
    prim:          Usd.Prim,
    stage:         Usd.Stage,
    path:          str,
    prim_type:     str                  | None                  = None,
    extra_prims:   Usd.Prim             | list[Usd.Prim] | None = None,
    exclude_attrs: list[str]            | str | None            = None,
    attr_rename:   Callable[[str], str] | None                  = None,
) -> Usd.Prim:
    """Duplicates a prim. This does a basic copy operation on the prim type
    and all the values of non-authored attributes. It does NOT handle relationships
    or connections.

    Args:
        prim: A prim to duplicate.
        stage: A stage to place the duplicated prim in.
        path: Path of the duplicated prim.
        prim_type: The type of the duplicated prim.
            If None, use the source prim type.
        extra_prims: A list of prims to copy extra attributes from.
        exclude_attrs: One or a list of regex used to exclude attributes by name.
        attr_rename: A custom function to rename attributes on the duplicated prim.

    Returns:
        The duplicated prim.
    """
    prim_type     = prim_type or prim.GetTypeName()
    exclude_attrs = exclude_attrs or []
    exclude_attrs = [exclude_attrs] if isinstance(exclude_attrs, str) else exclude_attrs
    exclude_regex = re.compile("|".join(exclude_attrs))

    # checks if an attr should be excluded
    def do_exclude(attr_name):
        if exclude_attrs and exclude_regex.search(attr_name):
            return True
        return False

    # duplicate the prim
    dup_prim = stage.DefinePrim(path, prim_type)
    # duplicate prim metadata
    for key, val in prim.GetAllAuthoredMetadata().items():
        val = prim_type if key == "typeName" else val
        dup_prim.SetMetadata(key, val)

    # gather all attrs to copy
    attrs = list(prim.GetAttributes())
    if extra_prims:
        if not isinstance(extra_prims, (list, tuple)):
            extra_prims = [extra_prims]
        for extra_prim in extra_prims:
            attrs += list(extra_prim.GetAttributes())

    for attr in attrs:
        # skip non-authored attributes
        if not attr.IsAuthored() or do_exclude(attr.GetName()):
            continue

        # duplicate the attr and value
        attr_name = attr.GetName()
        if attr_rename:
            attr_name = attr_rename(attr_name)
        dup_attr = dup_prim.CreateAttribute(
            attr_name, attr.GetTypeName(), attr.IsCustom(), attr.GetVariability()
        )
        dup_attr.Set(attr.Get())

        # duplicate attr custom data
        if attr.HasCustomData():
            dup_attr.SetCustomData(attr.GetCustomData())

        # duplicate attr metadata
        for key, val in attr.GetAllAuthoredMetadata().items():
            dup_attr.SetMetadata(key, val)

    return dup_prim


def create_scopes(stage: Usd.Stage, prim_path: str, as_scope: bool = True) -> Usd.Prim:
    """Recursively creates scope prims to satisfy a given prim path.

    Args:
        stage: A stage to create the prim in.
        prim_path: A prim path to create.
        as_scope: If True, create each level as Scope. Otherwise, omit the prim type.

    Returns:
        the leaf prim.
    """
    cur = ""
    for each in prim_path[1:].split("/"):
        cur  = f"{cur}/{each}"
        prim = stage.GetPrimAtPath(cur)
        if not prim.IsValid():
            if as_scope:
                prim = stage.DefinePrim(cur, "Scope")
                prim.SetTypeName("Scope")
            else:
                prim = stage.DefinePrim(cur)
    return prim


def get_references(prim: Usd.Prim, abspath: bool = True) -> list[str]:
    """Returns a list of paths to referenced USD files in the given prim.

    Args:
        prim: A prim to query.
        abspath: If True, returns absolute paths.
    """
    if not prim.HasAuthoredReferences():
        return []

    refs        = []
    remove_refs = set()
    for spec in prim.GetPrimStack():
        dir_path = os.path.dirname(spec.layer.identifier)

        for each in (
            spec.referenceList.prependedItems,
            spec.referenceList.appendedItems,
            spec.referenceList.explicitItems,
            spec.referenceList.addedItems,
        ):
            if abspath:
                refs.extend([f"{dir_path}/{x.assetPath}" for x in each])
            else:
                refs.extend([f"{x.assetPath}" for x in each])

        if abspath:
            paths = {
                f"{dir_path}/{x.assetPath}" for x in spec.referenceList.deletedItems
            }
        else:
            paths = {f"{x.assetPath}" for x in spec.referenceList.deletedItems}
        remove_refs = remove_refs | paths

    path_list = [x for x in refs if x not in remove_refs]
    if abspath:
        return [Path(x).absolute().as_posix() for x in path_list]
    return path_list


def clear_references(prim: Usd.Prim) -> None:
    """Clear references in the given prim."""
    prim.GetReferences().ClearReferences()


def add_references(
    prim: Usd.Prim, ref_paths: Any, start_dir: str | None = None, replace: bool = False
) -> None:
    """Adds one or more references to a given prim. If reference paths are
    absolute paths, check and ignored existing paths.

    Args:
        prim: A prim to add references to.
        ref_paths: This can be one of the following:
            - Single or a list of reference path strings.
            - Single or a list of tuple (reference_path, prim_path)
            - A list of mixed strings and tuples.
        start_dir: A start directory to convert abspath to rel path.
            If None, use the stage path that contains this prim.
        replace: If True, clear existing references before adding new ones.
    """
    ref_obj = prim.GetReferences()
    if replace:
        ref_obj.ClearReferences()
        cur = set()
    else:
        cur = {Path(x) for x in get_references(prim, abspath=True)}

    if start_dir:
        start_dir = Path(start_dir).as_posix()
    else:
        start_dir = Path(prim.GetStage().GetRootLayer().realPath).parent.as_posix()

    if not isinstance(ref_paths, (list, tuple)):
        ref_paths = (ref_paths,)

    for each in ref_paths:
        if isinstance(each, (list, tuple)):
            ref_path  = Path(each[0])
            prim_path = Path(each[1]).as_posix()
        else:
            ref_path  = Path(each)
            prim_path = None

        if ref_path.is_absolute():
            if ref_path in cur:
                continue
            if not start_dir:
                raise RuntimeError("Can't convert absolute path to rel path.")
            try:
                ref_path = Path(os.path.relpath(ref_path.as_posix(), start=start_dir))
            except ValueError:
                pass
        ref_path_str = Path(ref_path).as_posix()

        if prim_path:
            ref_obj.AddReference(Sdf.Reference(ref_path_str, prim_path))
        else:
            ref_obj.AddReference(ref_path_str)


def set_references(
    prim: Usd.Prim, ref_paths: Any, start_dir: str | None = None
) -> None:
    """Sets the references of a given prim. Existing references will be replaced."""
    add_references(prim, ref_paths, start_dir=start_dir, replace=True)


def get_inherits(prim: Usd.Prim) -> list[str]:
    """Returns a list of inheritance paths in the given prim."""
    if prim.HasAuthoredInherits():
        return prim.GetInherits().GetAllDirectInherits()
    return []


def clear_inherits(prim: Usd.Prim) -> None:
    """Clear inheritances in the given prim."""
    if prim.HasAuthoredInherits():
        return prim.GetInherits().ClearInherits()


def add_inherits(
    prim: Usd.Prim, inherits: str | Usd.Prim | list[Any], replace: bool = False
) -> None:
    """Adds inherit(s) to a given prim. Existing inherits are ignored.

    Args:
        prim: A prim to add inherit(s) to.
        inherits: One or more prims or prim paths to inherit from.
        replace: If True, clear existing references before adding new ones.
    """
    if not isinstance(inherits, (list, tuple)):
        inherits = [inherits]

    inherit_obj = prim.GetInherits()
    if replace:
        inherit_obj.ClearInherits()
    for each in inherits:
        if isinstance(each, Usd.Prim):
            each = each.GetPath()
        inherit_obj.AddInherit(Path(each).as_posix())


def set_inherits(prim: Usd.Prim, inherits: str | list[str]) -> None:
    """Sets the inherit(s) of a given prim. Existing inherits will be replaced."""
    add_inherits(prim, inherits, replace=True)


def add_api_schema(prim: Usd.Prim, schema: str) -> None:
    """Adds an api schema to a given prim.
    This function works for Usd versions without `AddAppliedSchema()`

    Args:
        prim: A prim to add an api schema to.
        schema: API schema name.
    """
    if hasattr(prim, "AddAppliedSchema"):
        prim.AddAppliedSchema(schema)
    else:
        token_list = prim.GetMetadata("apiSchemas")
        if not token_list:
            token_list = Sdf.TokenListOp.Create()
        token_list.prependedItems += [schema]
        prim.SetMetadata("apiSchemas", token_list)


def get_api_schemas(prim: Usd.Prim) -> list[str]:
    """Returns a list of api schemas attached to a given prim.

    This function operates on metadata directly to ensure consistent behavior across
    USD versions. UsdPrim.GetAppliedSchemas() behavior changed between 21.8 and 21.11.
    """
    token_list_op = prim.GetMetadata("apiSchemas")
    if token_list_op:
        return token_list_op.GetAddedOrExplicitItems()
    return []


def copy_property(
    src_prim: Usd.Prim, dst_prim: Usd.Prim, src_name: str, dst_name: str | None = None
):
    """Copies a property from a source prim to a destination prim."""
    dst_name     = dst_name or src_name
    src_property = src_prim.GetProperty(src_name)
    if isinstance(src_property, Usd.Attribute):
        dst_property = dst_prim.CreateAttribute(dst_name, src_property.GetTypeName())
    else:
        dst_property = dst_prim.CreateRelationship(dst_name)
    src_property.FlattenTo(dst_property)


def copy_properties(
    src_prim:      Usd.Prim,
    dst_prim:      Usd.Prim,
    src_names:     list[str] | dict[str:str] | None = None,
    exclude_names: list[str] | None                 = None,
):
    """Copies properties from a source prim to a destination prim."""
    for src_name in src_prim.GetPropertyNames():
        dst_name = src_name
        if src_names and src_name not in src_names:
            continue
        if exclude_names and src_name in exclude_names:
            continue
        dst_name = src_name
        if isinstance(src_names, dict):
            dst_name = src_names.get(src_name, src_name)
        copy_property(src_prim, dst_prim, src_name, dst_name=dst_name)
