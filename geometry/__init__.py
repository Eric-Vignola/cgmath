"""
Move all data types into the same namespace
"""

from cgmath.data import Data, DataList, ImmutableArray
from cgmath.geometry._bilinear_surface import RaycastData
from cgmath.geometry.bspline import NurbsCurveData
from cgmath.geometry.bspline_patch import NurbsSurfaceData, PatchSampleData
from cgmath.geometry.camera import CameraData, CameraList
from cgmath.geometry.deform import DeltaMushData, PatchRelaxData
from cgmath.geometry.light import LightData, LightList
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