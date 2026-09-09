"""
Move all data types into the same namespace
"""

from cgmath.geometry._base import Data, DataList, ImmutableArray
from cgmath.geometry._saddle_surface import RaycastData
from cgmath.geometry.bspline import BSplineData
from cgmath.geometry.bspline_patch import BSplinePatchData, PatchSampleData
from cgmath.geometry.deform import DeltaMushData, PatchRelaxData
from cgmath.geometry.map import GeomSubsetData, MapData
from cgmath.geometry.mesh import (
    MeshData,
    MeshList,
    TriangulateMethod,
    TriangulateRules,
    UVData,
    UVList,
)
from cgmath.geometry.morph_target import MorphData, MorphList
from cgmath.geometry.skin_weights import CompactSkinData, SkinData, SkinList