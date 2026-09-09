"""
UsdStage utilities.
"""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path

from pxr import Usd


@contextlib.contextmanager
def no_stage_cache():
    """A context where all stages created/opened within are registered to a
    throw-away cache."""
    cache = Usd.StageCache()
    with Usd.StageCacheContext(cache):
        try:
            yield
        finally:
            cache.Clear()


def open_stage(
    file_path: str, cache: Usd.StageCache | None = None, no_cache: bool = False
) -> Usd.Stage:
    """Opens a stage.

    Args:
        file_path: A usd file to open.
        cache: A stage cache to register the stage to.
            If None, use the default stage cache.
        no_cache: If True, the `cache` arge is ignored and the stage won't be
            registered in any cache.

    Returns:
        The stage created.
    """
    if no_cache:
        with no_stage_cache():
            stage = Usd.Stage.Open(file_path)
        return stage
    elif cache:
        with Usd.StageCacheContext(cache):
            stage = Usd.Stage.Open(file_path)
        return stage
    else:
        return Usd.Stage.Open(file_path)


def new_stage(
    file_path: str, cache: Usd.StageCache | None = None, no_cache: bool = False
) -> Usd.Stage:
    """Creates a new stage at a given file path. If the file already exists, opens it
    and clear its content.

    Args:
        file_path: A usd file path to create the stage at.
        cache: A stage cache to register the stage to.
            If None, use the default stage cache.
        no_cache: If True, the `cache` arge is ignored and the stage won't be
            registered in any cache.

    Returns:
        The stage created.
    """
    file_path = Path(file_path).as_posix()
    if no_cache:
        with no_stage_cache():
            stage = new_stage(file_path)
        return stage
    elif cache:
        with Usd.StageCacheContext(cache):
            stage = new_stage(file_path)
        return stage
    else:
        if os.path.exists(file_path):
            stage = Usd.Stage.Open(file_path)
            stage.GetRootLayer().Clear()
        else:
            stage = Usd.Stage.CreateNew(file_path)
        return stage


def new_stage_in_memory(
    cache: Usd.StageCache | None = None, no_cache: bool = False
) -> Usd.Stage:
    """Creates a new stage in memory.

    Args:
        cache: A stage cache to register the stage to.
            If None, use the default stage cache.
        no_cache: If True, the `cache` arge is ignored and the stage won't be
            registered in any cache.

    Returns:
        The stage created.
    """
    if no_cache:
        with no_stage_cache():
            stage = Usd.Stage.CreateInMemory()
        return stage
    elif cache:
        with Usd.StageCacheContext(cache):
            stage = Usd.Stage.CreateInMemory()
        return stage
    else:
        return Usd.Stage.CreateInMemory()


def write_stage(
    stage:     Usd.Stage,
    file_path: os.PathLike    | str | None = None,
    logger:    logging.Logger | None       = None,
) -> None:
    """Writes a stage to a file.

    If file_path is provide, and the stage is not pointing at any file or is pointing
    at a different file, run stage.Export(). Otherwise run root_layer.Save().
    """
    layer      = stage.GetRootLayer()
    layer_path = layer.realPath

    if not file_path:
        layer.Save()
    elif not layer_path or Path(layer_path) != Path(file_path):
        stage.Export(Path(file_path).as_posix())
    else:
        layer.Save()

    if not file_path and layer_path:
        file_path = layer_path
    if logger and file_path:
        logger.info(f"Wrote stage {Path(file_path).as_posix()}.")


def clear_sublayers(stage: Usd.Stage) -> None:
    """Clear sublayers in the given stage."""
    stage.GetRootLayer().subLayerPaths = []


def clear_missing_sublayers(stage: Usd.Stage) -> None:
    """Clear missing sublayers in the given stage."""
    root_layer = stage.GetRootLayer()
    layers     = get_sublayers(stage)

    new_layers = []
    changed    = False
    for layer in layers:
        path = Path(layer)
        if not path.is_absolute():
            path = Path(root_layer.realPath).parent / path
        if path.exists():
            new_layers.append(layer)
        else:
            changed = True

    if changed:
        root_layer.subLayerPaths = new_layers


def get_sublayers(stage: Usd.Stage, absolute: bool = False) -> list[str]:
    """Returns a list of sublayers in a given stage.

    Args:
        stage: A stage to get sublayers from.
        absolute: If True, return absolute paths.
            Otherwise return sublayer paths as is.
    """
    root_layer = stage.GetRootLayer()
    stage_dir  = root_layer.realPath
    if stage_dir:
        stage_dir = Path(stage_dir).parent
    paths = root_layer.subLayerPaths
    if stage_dir and absolute:
        return [(stage_dir / x).as_posix() for x in paths if not Path(x).is_absolute()]
    return list(paths)  # ensure copy


def add_sublayers(
    stage:          Usd.Stage,
    sublayer_paths: str       | list[str],
    start_dir:      str       | None      = None,
    replace:        bool                  = False,
) -> None:
    """Adds sublayers to a given stage.

    Args:
        stage: A stage to add sublayers to.
        sublayer_paths: A list of sublayer paths to add.
        start_dir: A start directory to convert abspath to rel path.
            If None, use the stage path that contains this prim.
        replace: If True, clear existing references before adding new ones.
    """
    root_layer = stage.GetRootLayer()
    if replace:
        root_layer.subLayerPaths = []

    if start_dir:
        start_dir = Path(start_dir).as_posix()
    else:
        start_dir = Path(root_layer.realPath).parent.as_posix()

    if not isinstance(sublayer_paths, (list, tuple)):
        sublayer_paths = [sublayer_paths]

    for path in sublayer_paths:
        path = Path(path)
        if path.is_absolute():
            if not start_dir:
                raise RuntimeError("Can't convert absolute path to rel path.")
            try:
                path = Path(os.path.relpath(path.as_posix(), start=start_dir))
            except ValueError:
                pass
        path = path.as_posix()
        if path not in root_layer.subLayerPaths:
            root_layer.subLayerPaths.append(path)


def set_sublayers(
    stage:          Usd.Stage,
    sublayer_paths: str       | list[str],
    start_dir:      str       | None      = None,
) -> None:
    """Sets the sublayers in a given stage.

    Args:
        stage: A stage to set sublayers in.
        sublayer_paths: A list of sublayer paths to set.
        start_dir: A start directory to convert abspath to rel path.
            If None, use the stage path that contains this prim.
    """
    add_sublayers(stage, sublayer_paths, start_dir=start_dir, replace=True)
