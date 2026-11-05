"""``cgmath.formats.load_fbx``: what save_fbx writes, read from one fbx file"""

import os


class FbxContents:
    """one fbx file's rig at its bind pose, its takes, meshes, UV sets and
    skins, as :func:`load_fbx` reads them and :func:`save_fbx` writes them"""

    __slots__ = ("rig", "clips", "meshes", "uvs", "skins")

    def __init__(self, rig=None, clips=None, meshes=None, uvs=None, skins=None):
        from cgmath.geometry.mesh import MeshList, UVList
        from cgmath.geometry.skin_weights import SkinList
        from cgmath.hierarchy.hierarchy import ClipList

        self.rig    = rig
        self.clips  = ClipList() if clips is None else clips
        self.meshes = MeshList() if meshes is None else meshes
        self.uvs    = UVList() if uvs is None else uvs
        self.skins  = SkinList() if skins is None else skins

    def objects(self) -> list:
        held = [self.rig] if self.rig is not None else []
        return held + [part for part in (self.clips, self.meshes, self.uvs, self.skins) if len(part)]

    def save_fbx(self, filename, as_ascii: bool = False) -> None:
        from cgmath.formats._write import save_fbx

        save_fbx(filename, self, as_ascii=as_ascii)

    def __repr__(self) -> str:
        nodes = len(self.rig) if self.rig is not None else 0
        return (
            f"FbxContents({nodes} nodes, {len(self.clips)} takes, {len(self.meshes)} meshes, "
            f"{len(self.uvs)} UV sets, {len(self.skins)} skins)"
        )


def load_fbx(filename) -> FbxContents:
    from cgmath.formats._write import _refuse_shear
    from cgmath.geometry import mesh as mesh_module
    from cgmath.geometry.mesh import MeshList, UVList
    from cgmath.geometry.skin_weights import SkinList
    from cgmath.hierarchy.hierarchy import FBX, ClipList, HierarchyData

    if FBX is None:
        raise ImportError("fbx module not found")
    filename = os.path.expanduser(filename)
    if not os.path.isfile(filename):
        raise FileNotFoundError(filename)

    try:
        rig = HierarchyData.load_fbx(filename, pose="bind")
    except ValueError as error:
        if "holds no bind pose" not in str(error):
            raise
        rig = HierarchyData.load_fbx(filename)
    clips = ClipList.load_fbx(filename)
    _refuse_shear(rig, clips.list, "load_fbx")
    _positive_zeros(rig, clips)

    read   = mesh_module.load_fbx(filename)
    meshes = MeshList([mesh for mesh, _ in read])
    uvs    = UVList([uv for _, sets in read for uv in sets])
    skins  = SkinList.load_fbx(filename) if len(meshes) else SkinList()
    return FbxContents(rig if len(rig) else None, clips, meshes, uvs, skins)


CHANNELS = ("translate", "rotate", "scale", "joint_orient", "rotate_axis", "shear")


def _positive_zeros(rig, clips) -> None:
    """every -0.0 the readers' arithmetic leaves in a channel made 0.0, in
    place: a file holds the value, and a load of its save reads 0.0"""
    import numpy as np

    for node in rig:
        for channel in CHANNELS:
            value = np.asarray(getattr(node, channel))
            if np.signbit(value[value == 0.0]).any():
                setattr(node, channel, value + 0.0)
    for clip in clips:
        for channel in CHANNELS:
            block = np.asarray(getattr(clip.frames, channel))
            if np.signbit(block[block == 0.0]).any():
                setattr(clip.frames, channel, block + 0.0)
