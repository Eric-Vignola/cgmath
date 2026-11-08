"""TransformData's shear: Maya's shearXY, shearXZ and shearYZ as a channel.

A node holds it, a list reads and writes it for every node, a clip keeps one
(F, N, 3) block of it, files save it under its public name and leave a zero
one out, and every matrix in the package composes S * Sh * RO * R * JO * IS
* T, one node at a time or batched. test_shear_maya.py holds those matrices
to what Maya recorded. A rig holding no shear skips every product.

The matrix setters split as Maya's xform -m does, Gram-Schmidt on the rows, a
mirror on scale z: a node that is no joint takes the shear its target holds,
a joint keeps its own. set_parent, swapaxes, match_rotate and the pose algebra
follow, one node at a time and batched alike.
"""
from __future__ import annotations

import copy
import inspect
import itertools
import os
import pickle
import tempfile
import unittest
import warnings
import zipfile
from unittest import mock

import numpy as np
from cgmath._tests.test_clip import _DeltaRigs
from cgmath._tests.test_skin_deform import branching_clip, pose_at, skinned_mesh
from cgmath.geometry.deform import SkinDeformData
from cgmath.geometry.sdf import SDFBox, SDFCylinder, SDFSphere
from cgmath.hierarchy import ClipData, HierarchyData, TransformData, TransformList, generate_uuid
from cgmath.hierarchy import hierarchy as H
from cgmath.hierarchy.hierarchy import _diagonal, _pose_matrices, _scale_shears, _span_world
from cgmath.transforms import matrix_local, matrix_multiply, matrix_to_euler
from cgmath.render.scene import Camera, Light, Object, Scene

ASSETS = os.path.join(os.path.dirname(__file__), "test_assets")
SHEAR  = (0.5, -0.25, 0.75)

# a gate on shear reads all three entries: neither xy nor their sum says
# what a node holds
PARTIAL_SHEARS = ((0.0, 0.5, 0.0), (0.0, 0.0, 0.75), (0.0, 0.5, 0.75), (0.25, 0.5, -0.75))
READS = (
    "matrix",
    "world_matrix",
    "parent_matrix",
    "inverse_matrix",
    "world_inverse_matrix",
    "parent_inverse_matrix",
)

# three rigs seeded runs found: batch against one at a time, no shear held,
# every condition number under 1e8. Near gimbal the last bit of a rotation
# picks the side of the euler split's snap
GIMBAL_RIGS = [{'seed': 604,
  'rig':    176,
  'pick':   [3, 5, 8, 9],
  'parent': 'n0',
  'nodes': [{'name': 'n0',
             'joint':     True,
             'scale':     [8.150858531978068, 5.046112778925358, 0.10933836017612208],
             'rotate':    [89.9999986630049, 128.92365780602944, 125.53570891901245],
             'translate': [2.1481857484549565, 0.06940350167705311, 1.1192137772058723],
             'order':     4,
             'jo':        [-2.291512987978109, 24.243360234792092, 33.7522901248592],
             'ra':        [6.506613569705699, 18.01080805984749, -19.53217034810899],
             'ssc':       False,
             'up': None},
            {'name': 'n1',
             'joint':     True,
             'scale':     [0.19306112558212477, 0.3401682733233515, 1.2143798830321253],
             'rotate':    [-89.99999396886403, 37.994167674885574, -47.72655414592316],
             'translate': [-0.369980290037649, 0.7370669281978834, 2.9673736735826948],
             'order':     4,
             'jo':        [29.27953209676224, -32.87155200483053, 32.20144852169935],
             'ra':        None,
             'ssc':       False,
             'up': 0},
            {'name': 'n2',
             'joint':     False,
             'scale':     [7.184098336687063, 2.8413093049244003, 0.5228144475357326],
             'rotate':    [76.15736673255438, 90.000025115471, -31.93440119674227],
             'translate': [1.8298644543822808, -0.3386106461343248, 2.3841235658683626],
             'order':     5,
             'jo':        None,
             'ra':        None,
             'ssc':       False,
             'up': 0},
            {'name': 'n3',
             'joint':     False,
             'scale':     [0.12060830831334758, 1.6107293913269458, 0.3871485901622338],
             'rotate':    [-89.99999307772754, 130.80472858997354, 44.070485969882924],
             'translate': [0.2997562429959997, 2.043084024995922, -2.470020065522321],
             'order':     4,
             'jo':        None,
             'ra':        None,
             'ssc':       False,
             'up': 0},
            {'name': 'n4',
             'joint':     False,
             'scale':     [1.18288439779984, 0.24215229500050442, 0.2554422979771878],
             'rotate':    [-90.00001743356758, 119.30452772539462, 32.38508641114004],
             'translate': [-2.7108131126055737, 1.908455707722764, -2.94566905411095],
             'order':     4,
             'jo':        None,
             'ra':        [13.794061601387995, -23.143022327880473, 22.51468465483846],
             'ssc':       False,
             'up': 1},
            {'name': 'n5',
             'joint':     True,
             'scale':     [1.4866116261822324, 0.5978998964588612, 4.302442808669768],
             'rotate':    [-17.709878011334865, -16.61889262689172, 89.99999576017474],
             'translate': [1.4152569705746156, 0.9644524711134972, -0.8792861376026648],
             'order':     3,
             'jo':        [-24.53130124132165, 23.312112193831297, -25.118308130052185],
             'ra':        None,
             'ssc':       True,
             'up': 4},
            {'name': 'n6',
             'joint':     True,
             'scale':     [9.567548322018139, 7.105342279557972, 0.11961198040763549],
             'rotate':    [-92.09314965415554, 56.978258481580156, -90.00000991518722],
             'translate': [2.709625352287639, 0.8991787064538519, 1.275021804597456],
             'order':     3,
             'jo':        [-36.4240443170361, -5.868970834588616, 30.35734788310667],
             'ra':        None,
             'ssc':       True,
             'up': 3},
            {'name': 'n7',
             'joint':     True,
             'scale':     [1.508633853409654, 0.36482219275564076, 2.678520359940692],
             'rotate':    [-92.04864155549903, 89.9999964857599, -66.16843352943954],
             'translate': [-1.0571401944983605, 2.598490330418999, -2.241777279907367],
             'order':     0,
             'jo':        [-6.705399890899876, 39.0899239217144, -21.735199446318152],
             'ra':        [-16.03613554169873, -12.019256220669988, 27.087105151064662],
             'ssc':       False,
             'up': 4},
            {'name': 'n8',
             'joint':     False,
             'scale':     [0.45840051056406045, 0.21540375578159096, 1.279223531948041],
             'rotate':    [-105.51875333197705, -109.62091604794018, 90.00000196954709],
             'translate': [-1.9441294678925805, -1.5493526109490185, 1.4828053466700837],
             'order':     1,
             'jo':        None,
             'ra':        None,
             'ssc':       False,
             'up': 5},
            {'name': 'n9',
             'joint':     True,
             'scale':     [9.967057792894058, 2.9316118412023506, 8.34834947616705],
             'rotate':    [-19.763076246524264, -89.99999931422701, 55.24174247386816],
             'translate': [-1.8521191880373555, 1.9074135875909146, -1.7543014951853555],
             'order':     5,
             'jo':        [4.0095763155506035, 25.51996739788828, 38.50247632974978],
             'ra':        None,
             'ssc':       False,
             'up': 8}]},
 {'seed': 601,
  'rig':    22,
  'pick':   [2, 3, 8],
  'parent': 'n0',
  'nodes': [{'name': 'n0',
             'joint':     False,
             'scale':     [0.6187470569292869, 6.175372107726462, 0.4914574057267019],
             'rotate':    [90.00000109176536, -84.53818142076791, -26.354489714645297],
             'translate': [0.054491389343548136, -0.3390728105663561, 0.9829318793547994],
             'order':     2,
             'jo':        None,
             'ra':        [-18.95910956397295, -4.4817614939571975, -17.248357938421847],
             'ssc':       False,
             'up': None},
            {'name': 'n1',
             'joint':     True,
             'scale':     [2.4371044058499383, 1.1722555403158754, 1.208621856421593],
             'rotate':    [-94.19745345972004, 113.05638320411117, -90.00000106349636],
             'translate': [-2.541019741867217, 2.4497805046239183, 1.985188023102352],
             'order':     3,
             'jo':        [-26.504710735609834, 1.4064299286098887, -7.330866248507604],
             'ra':        None,
             'ssc':       False,
             'up': 0},
            {'name': 'n2',
             'joint':     True,
             'scale':     [2.5332773475650003, 2.029277139625755, 0.44771369698418706],
             'rotate':    [90.00000130820555, -96.47858086762776, -29.90651781328559],
             'translate': [0.41101905109965653, -1.0944297765925577, -0.41951471210557667],
             'order':     2,
             'jo':        [38.05639112018872, -36.216023672680784, 12.904244772341812],
             'ra':        [-18.096559201642197, 27.921573780775304, -13.856805272051119],
             'ssc':       True,
             'up': 1},
            {'name': 'n3',
             'joint':     False,
             'scale':     [1.8618231951289135, 0.2343696171212628, 3.6324489851055564],
             'rotate':    [-81.61172714149536, 90.00000775696766, -138.85737817012858],
             'translate': [0.4304471648191921, -0.24286033227528936, -1.5280256234314569],
             'order':     0,
             'jo':        None,
             'ra':        None,
             'ssc':       False,
             'up': 0},
            {'name': 'n4',
             'joint':     True,
             'scale':     [8.283029438120872, 0.14954362604923005, 1.2558045561018765],
             'rotate':    [90.00000167959297, 119.31825764957625, 129.29707096464358],
             'translate': [1.18256742075563, -2.1518114820281413, -2.291355841465771],
             'order':     4,
             'jo':        [10.70485382454607, -33.8067516497659, -6.685495983643399],
             'ra':        None,
             'ssc':       False,
             'up': 3},
            {'name': 'n5',
             'joint':     True,
             'scale':     [1.41056384110386, 0.2810053260712991, 2.5431079765238427],
             'rotate':    [-89.99999300623914, -63.6481929203821, -46.178462446400005],
             'translate': [-2.32586569626691, -1.425429184536431, 0.05917425511574681],
             'order':     4,
             'jo':        [37.51110016059832, 38.607938059247985, 24.2763954750303],
             'ra':        None,
             'ssc':       False,
             'up': 4},
            {'name': 'n6',
             'joint':     False,
             'scale':     [6.287952787171763, 0.45256723151132966, 0.49562123448218154],
             'rotate':    [90.00001459691465, -111.09316184251807, -126.96660474712581],
             'translate': [-2.232246833784815, 0.3252959172458594, -0.3278490561839922],
             'order':     4,
             'jo':        None,
             'ra':        [-9.396155095617061, 25.708989353638735, -14.334846432873306],
             'ssc':       False,
             'up': 3},
            {'name': 'n7',
             'joint':     False,
             'scale':     [2.187193829825727, 0.4489114955992353, 8.931519755021654],
             'rotate':    [-38.16952025593798, -90.00002333410475, -10.410534783782197],
             'translate': [-1.0138749919829464, -1.5029301491195433, 1.7149715882192087],
             'order':     5,
             'jo':        None,
             'ra':        [4.220432134916095, -7.4322054179944885, 5.761820154015155],
             'ssc':       False,
             'up': 4},
            {'name': 'n8',
             'joint':     False,
             'scale':     [0.3786534195674364, 2.7145385195530736, 5.511183863607658],
             'rotate':    [-120.22323152855407, 89.9999953613772, 20.541069745137435],
             'translate': [-0.5903180702759316, -1.4996619664400632, 2.498334434575918],
             'order':     0,
             'jo':        None,
             'ra':        None,
             'ssc':       False,
             'up': 6},
            {'name': 'n9',
             'joint':     True,
             'scale':     [0.226999427062, 0.10673269330119654, 1.809504878664756],
             'rotate':    [-90.00000696089185, -135.53215801904264, 15.741897040658273],
             'translate': [-2.852027987123752, -2.9884869055069903, 2.4755738786813],
             'order':     2,
             'jo':        [-6.684215238326445, -9.055956418626685, -22.965169159471273],
             'ra':        None,
             'ssc':       False,
             'up': 6}]},
 {'seed': 601,
  'rig':    266,
  'pick':   [1, 2, 5, 7, 8, 9],
  'parent': 'n0',
  'nodes': [{'name': 'n0',
             'joint':     True,
             'scale':     [0.7498365468645641, 0.23908210746390313, 7.999195329097852],
             'rotate':    [-112.23267496179056, 90.00000163761132, 106.97556840580825],
             'translate': [2.102867072564938, -0.5399734129688931, -1.934340775018515],
             'order':     5,
             'jo':        [3.107070271068423, -19.34272911071945, 19.440377709982357],
             'ra':        None,
             'ssc':       False,
             'up': None},
            {'name': 'n1',
             'joint':     True,
             'scale':     [0.4047763455144769, 0.32365368125596905, 0.4607108389531456],
             'rotate':    [-89.99999893326583, 79.71243173388774, -42.49017659534253],
             'translate': [2.008228354980984, 2.0121078091881968, -1.6840437786554132],
             'order':     4,
             'jo':        [17.27601494449732, 32.58785994950277, 10.789434281391692],
             'ra':        None,
             'ssc':       False,
             'up': 0},
            {'name': 'n2',
             'joint':     False,
             'scale':     [9.665152342360098, 5.585336729022677, 0.4500750116002081],
             'rotate':    [106.29761304065028, 142.97943930327142, 89.99998654120004],
             'translate': [0.7114285062352987, -2.2917837491572812, -1.8541679605379635],
             'order':     1,
             'jo':        None,
             'ra':        None,
             'ssc':       False,
             'up': 0},
            {'name': 'n3',
             'joint':     True,
             'scale':     [2.26381031188162, 0.11770492476810776, 2.3516977057599635],
             'rotate':    [-89.9999939320549, -1.4782529120936374, -56.75021012016265],
             'translate': [-1.1760867304911007, -1.9970864582591552, -0.23975904042588114],
             'order':     2,
             'jo':        [-31.72208385558589, 8.777833515928386, -26.60193396669139],
             'ra':        None,
             'ssc':       True,
             'up': 1},
            {'name': 'n4',
             'joint':     True,
             'scale':     [0.7152624916138355, 0.1565978621190996, 0.40006243176646267],
             'rotate':    [-89.9999978323231, -50.470545273020704, 115.25247459006528],
             'translate': [2.6817689118040375, -1.1624759543629883, -0.05281887812350483],
             'order':     4,
             'jo':        [33.99577585966918, -14.241656283247295, 5.932352501063093],
             'ra':        None,
             'ssc':       False,
             'up': 1},
            {'name': 'n5',
             'joint':     True,
             'scale':     [0.9559622773592572, 0.16423397062153416, 9.439129333252918],
             'rotate':    [-29.792559809177575, 90.00000075368274, 51.18153669001276],
             'translate': [1.290553452562878, 1.3585527613647521, 0.03242124565196125],
             'order':     0,
             'jo':        [2.3859328793660524, -21.122881881651505, -9.075121883916495],
             'ra':        [5.559233224584894, -16.772553513538547, -9.076791818833613],
             'ssc':       True,
             'up': 4},
            {'name': 'n6',
             'joint':     False,
             'scale':     [0.15234786169465317, 0.5402925465087357, 0.5881002752505754],
             'rotate':    [142.59886059971944, -89.9999925173272, -55.748847842945054],
             'translate': [2.338515179896673, 2.945499776063473, -0.4640926037624471],
             'order':     0,
             'jo':        None,
             'ra':        None,
             'ssc':       False,
             'up': 5},
            {'name': 'n7',
             'joint':     True,
             'scale':     [4.695805420358844, 8.124871013843052, 0.1338127009742804],
             'rotate':    [-97.93947418180878, 128.22499930646143, 90.00000950240575],
             'translate': [-0.6491763603404568, 1.4742832224521054, 2.180222358030428],
             'order':     3,
             'jo':        [-21.69891302946133, 14.119915849905844, -21.080094676688155],
             'ra':        [-27.257590639998167, 2.437393292916326, -15.450196456195382],
             'ssc':       False,
             'up': 6},
            {'name': 'n8',
             'joint':     False,
             'scale':     [1.4959438795059, 0.16725016934187484, 0.4732590486117238],
             'rotate':    [-89.99999897293856, -47.649793282519184, 54.16776084065856],
             'translate': [-2.171619207441658, -0.8693191284725019, -2.3518805073619733],
             'order':     4,
             'jo':        None,
             'ra':        None,
             'ssc':       False,
             'up': 6},
            {'name': 'n9',
             'joint':     False,
             'scale':     [0.22179281035913084, 0.22370821920880143, 3.6260731653211784],
             'rotate':    [112.86886286036673, -90.00000080713855, 107.81816297351378],
             'translate': [1.186943144504565, -1.6305509514484533, 0.4242295883578424],
             'order':     5,
             'jo':        None,
             'ra':        [-20.717677625493966, 16.20374993262285, -3.719110092865222],
             'ssc':       False,
             'up': 7}]}]


def _rig(sheared=None, seed: int = 11, count: int = 9) -> HierarchyData:
    """a branching rig of joints, transforms and locators, every channel away
    from its default, half the joints compensating their parent's scale.
    ``sheared`` lists the rows holding shear, every row when None"""
    rng   = np.random.default_rng(seed)
    uuids = [generate_uuid() for _ in range(count)]
    nodes = []
    for i in range(count):
        kind  = ("joint", "transform", "joint", "locator")[i % 4]
        joint = kind == "joint"
        shear = rng.uniform(-0.8, 0.8, 3)
        held  = dict(shear=shear) if sheared is None or i in sheared else {}
        nodes.append(
            TransformData(
                f"n{i}",
                node_type=kind,
                uuid=uuids[i],
                parent_uuid=uuids[int(rng.integers(0, i))] if i else None,
                translate=rng.uniform(-3.0, 3.0, 3),
                rotate=rng.uniform(-150.0, 150.0, 3),
                scale=rng.uniform(0.5, 2.0, 3),
                rotate_order=int(rng.integers(0, 6)),
                rotate_axis=rng.uniform(-30.0, 30.0, 3),
                joint_orient=rng.uniform(-40.0, 40.0, 3) if joint else None,
                segment_scale_compensate=bool(joint and i and rng.random() < 0.5),
                **held,
            )
        )
    return HierarchyData(nodes)


def _animated(rig: HierarchyData, frames: int = 5, seed: int = 21) -> ClipData:
    """``rig`` over ``frames`` frames, bound to the first, every framed channel
    but shear moving"""
    rng   = np.random.default_rng(seed)
    clip  = ClipData(rig, frames=frames)
    shape = clip.frames.rotate.shape
    clip.frames.rotate       = clip.frames.rotate + rng.normal(size=shape) * 20.0
    clip.frames.translate    = clip.frames.translate + rng.normal(size=shape) * 0.3
    clip.frames.scale        = clip.frames.scale * rng.uniform(0.8, 1.2, shape)
    clip.frames.rotate_axis  = clip.frames.rotate_axis + rng.normal(size=shape) * 5.0
    clip.frames.joint_orient = clip.frames.joint_orient + rng.normal(size=shape) * 5.0
    return clip


def _sheared_clip(rig: HierarchyData, frames: int = 5, seed: int = 22) -> ClipData:
    """:func:`_animated`, shear moving on every frame too"""
    clip              = _animated(rig, frames)
    rng               = np.random.default_rng(seed)
    clip.frames.shear = rng.uniform(-0.8, 0.8, clip.frames.shear.shape)
    return clip


def _gap(a, b) -> float:
    """the largest entry of ``a - b``, relative to ``b``'s largest"""
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / max(1.0, np.abs(b).max()))


def _sheared_rows(rows) -> np.ndarray:
    """(xy, xz, yz) as the 4x4 Sh this file spells out, the reference"""
    xy, xz, yz = rows
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [xy, 1.0, 0.0, 0.0],
            [xz, yz, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _spelled(node) -> np.ndarray:
    """``node``'s matrix with Sh written out, the reference"""
    return (
        node.scale_matrix
        @ _sheared_rows(node.shear)
        @ node.rotate_axis_matrix
        @ node.rotate_matrix
        @ node.joint_orient_matrix
        @ node.parent_scale_inverse_matrix
        @ node.translate_matrix
    )


def _spelled_worlds(rig: HierarchyData) -> np.ndarray:
    """every world of ``rig`` down from :func:`_spelled`, in ``rig``'s order"""
    worlds = {}
    for row in rig._parent_first():
        node              = rig[row]
        worlds[node.uuid] = _spelled(node) @ worlds.get(node.parent_uuid, np.eye(4))
    return np.array([worlds[node.uuid] for node in rig.list])


class TestShearChannel(unittest.TestCase):
    def test_a_node_starts_at_zero_and_keeps_its_own(self):
        a, b = TransformData("a"), TransformData("b")
        self.assertEqual(a.shear.dtype, np.float64)
        self.assertEqual(a.shear.tolist(), [0.0, 0.0, 0.0])

        a.shear = SHEAR
        self.assertEqual(a.shear.tolist(), list(SHEAR))
        self.assertEqual(b.shear.tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(TransformData("c", shear=SHEAR).shear.tolist(), list(SHEAR))

        given    = np.array(SHEAR)  # the node copies what it is built from
        c        = TransformData("c", shear=given)
        given[0] = 9.0
        self.assertEqual(c.shear.tolist(), list(SHEAR))

        held    = a.shear  # a copy, as every channel's getter hands back
        held[0] = 9.0
        self.assertEqual(a.shear.tolist(), list(SHEAR))

    def test_it_takes_what_a_channel_takes(self):
        node       = TransformData("a")
        node.shear = 2.0
        self.assertEqual(node.shear.tolist(), [2.0, 2.0, 2.0])
        node.shear = [[1.0, 2.0, 3.0]]
        self.assertEqual(node.shear.tolist(), [1.0, 2.0, 3.0])
        node.shear = np.array(SHEAR)
        self.assertEqual(node.shear.tolist(), list(SHEAR))

        with self.assertRaisesRegex(ValueError, r"^TransformData\.shear must have shape \(3,\), got \(2,\)$"):
            node.shear = (1.0, 2.0)
        self.assertEqual(node.shear.tolist(), list(SHEAR))

    def test_its_components_are_xy_xz_yz(self):
        node = TransformData("a", shear=SHEAR)
        self.assertEqual((node.shear_xy, node.shear_xz, node.shear_yz), SHEAR)

        node.shear_xy = 1.0
        node.shear_xz = 2.0
        node.shear_yz = 3.0
        self.assertEqual(node.shear.tolist(), [1.0, 2.0, 3.0])
        self.assertFalse(hasattr(node, "shear_x"))

    def test_a_write_moves_a_childs_cached_world(self):
        parent = TransformData("p", uuid=generate_uuid(), translate=(1.0, 2.0, 3.0), scale=(1.0, 2.0, 3.0))
        child  = TransformData("c", uuid=generate_uuid(), parent_uuid=parent.uuid, translate=(1.0, 1.0, 1.0))
        rig    = HierarchyData([parent, child])
        before = rig["c"].world_matrix

        rig["p"].shear = SHEAR
        want           = child.matrix @ np.diag([1.0, 2.0, 3.0, 1.0]) @ _sheared_rows(SHEAR) @ parent.translate_matrix
        self.assertGreater(np.abs(rig["c"].world_matrix - before).max(), 0.1)
        self.assertLessEqual(_gap(rig["c"].world_matrix, want), 1e-14)

    def test_it_is_the_last_parameter(self):
        # every positional call written before it stays as it was
        self.assertEqual(list(inspect.signature(TransformData.__init__).parameters)[-1], "shear")

    def test_a_list_reads_and_writes_every_node(self):
        rig = _rig(sheared=())
        self.assertEqual(rig.shear.shape, (len(rig), 3))
        self.assertFalse(rig.shear.any())

        rig.shear = SHEAR  # one value, every node
        self.assertTrue(np.array_equal(rig.shear, np.tile(SHEAR, (len(rig), 1))))

        values    = np.arange(len(rig) * 3, dtype=float).reshape(-1, 3)
        rig.shear = values  # one row a node
        self.assertTrue(np.array_equal(rig.shear, values))
        self.assertTrue(np.array_equal(rig[2:5].shear, values[2:5]))
        self.assertTrue(np.array_equal(rig.shear_xz, values[:, 1]))

        rig[1:3].shear_yz = [7.0, 8.0]
        self.assertEqual(rig.shear_yz[1:3].tolist(), [7.0, 8.0])
        self.assertEqual([n.shear_yz for n in rig.list[1:3]], [7.0, 8.0])

    def test_shear_matrix_on_a_node_a_list_and_a_clip(self):
        node = TransformData("a", shear=SHEAR)
        self.assertTrue(np.array_equal(node.shear_matrix, _sheared_rows(SHEAR)))

        rig = _rig()
        self.assertTrue(np.array_equal(rig.shear_matrix, np.array([_sheared_rows(s) for s in rig.shear])))

        clip = _sheared_clip(_rig())
        got  = clip.frames.shear_matrix
        self.assertEqual(got.shape, clip.frames.shear.shape[:2] + (4, 4))
        for frame, row in np.ndindex(got.shape[:2]):
            self.assertTrue(np.array_equal(got[frame, row], _sheared_rows(clip.frames.shear[frame, row])))

    def test_a_clip_frames_it(self):
        rig   = _rig()
        clip  = _sheared_clip(rig, frames=4)
        shape = (4, len(rig))
        self.assertEqual(clip.frames.shear.shape, shape + (3,))
        self.assertEqual(clip.frames.shear_xy.shape, shape)
        for k, component in enumerate(("shear_xy", "shear_xz", "shear_yz")):
            self.assertTrue(np.array_equal(getattr(clip.frames, component), clip.frames.shear[..., k]))

        # the bound frame reads and writes its row, and no other frame's
        before     = np.array(clip.frames.shear)
        clip.frame = 2
        self.assertTrue(np.array_equal(clip.shear, before[2]))
        self.assertTrue(np.array_equal(clip["n3"].shear, before[2, 3]))

        clip["n3"].shear = SHEAR
        after            = clip.frames.shear
        self.assertEqual(after[2, 3].tolist(), list(SHEAR))
        same       = np.ones(shape, dtype=bool)
        same[2, 3] = False
        self.assertTrue(np.array_equal(after[same], before[same]))

        clip.frames.shear_xy = 4.0
        self.assertTrue((clip.frames.shear[..., 0] == 4.0).all())

    def test_maya_short_names_read_and_write_it_and_stay_out_of_files(self):
        clip = _sheared_clip(_rig())
        rig  = _rig()
        node = rig["n1"]
        for short, long in (("sh", "shear"), ("shxy", "shear_xy"), ("shxz", "shear_xz"), ("shyz", "shear_yz")):
            for target in (node, rig, clip, clip.frames):
                with self.subTest(short=short, target=type(target).__name__):
                    self.assertTrue(np.array_equal(getattr(target, short), getattr(target, long)))

        node.sh = SHEAR
        self.assertEqual(node.shear.tolist(), list(SHEAR))
        node.shxz = 9.0
        self.assertEqual(node.shear_xz, 9.0)
        rig.shyz = 8.0
        self.assertTrue((rig.shear_yz == 8.0).all())
        clip.frames.sh = 7.0
        self.assertTrue((clip.frames.shear == 7.0).all())

        for tree in (node.to_dict(), rig.to_dict(), clip.to_dict()):
            stack, keys = [tree], set()
            while stack:
                item = stack.pop()
                if isinstance(item, dict):
                    keys.update(item)
                    stack.extend(item.values())
                elif isinstance(item, (list, tuple)):
                    stack.extend(item)
            self.assertEqual(keys & {"sh", "shxy", "shxz", "shyz", "shear_xy", "shear_matrix", "_shear"}, set())

    def test_to_attributes_gives_it_on_every_node(self):
        # as it gives rotateAxis, zero included: a create path sets it
        self.assertEqual(TransformData("a").to_attributes()["shear"], [0.0, 0.0, 0.0])
        self.assertEqual(TransformData("a", node_type="joint", shear=SHEAR).to_attributes()["shear"], list(SHEAR))

        clip = _sheared_clip(_rig(), frames=2)
        got  = clip.frames.to_attributes()
        self.assertEqual(got[3]["shear"], clip.frames.shear[:, 3].tolist())


class TestNoShear(unittest.TestCase):
    """the fast path: every read below holds on 3.0.3 as well, and is here
    for what a change to the channel can do to a rig that holds none"""

    def test_a_node_is_the_six_factor_product_to_the_bit(self):
        for node in _rig(sheared=()):
            with self.subTest(node=node.name):
                M = np.eye(4)
                for factor in (
                    node.scale_matrix,
                    node.rotate_axis_matrix,
                    node.rotate_matrix,
                    node.joint_orient_matrix,
                    node.parent_scale_inverse_matrix,
                    node.translate_matrix,
                ):
                    M = np.dot(M, factor)
                self.assertTrue(np.array_equal(node.matrix, M))

    def test_a_rig_and_a_clip_are_the_product_of_their_own_factors(self):
        rig  = _rig(sheared=())
        clip = _animated(rig)
        got  = clip.frames.matrix
        want = (
            clip.frames.scale_matrix
            @ clip.frames.rotate_axis_matrix
            @ clip.frames.rotate_matrix
            @ clip.frames.joint_orient_matrix
            @ clip.frames.parent_scale_inverse_matrix
            @ clip.frames.translate_matrix
        )
        self.assertTrue(np.array_equal(got, want))

    def test_no_tree_and_no_block_names_it(self):
        rig  = _rig(sheared=())
        clip = _animated(rig)
        self.assertNotIn("shear", rig["n1"].to_dict())
        self.assertNotIn("shear", rig.to_json())
        self.assertNotIn("shear", clip.to_dict()["blocks"])
        self.assertNotIn("shear", clip.to_json())

    def test_a_node_holding_none_builds_no_sh(self):
        with mock.patch.object(TransformData, "shear_matrix", new_callable=mock.PropertyMock, create=True) as built:
            for node in _rig(sheared=()):
                node.matrix
        built.assert_not_called()


class TestShearBatched(unittest.TestCase, _DeltaRigs):
    """every batched read against one node at a time, on rigs and clips of
    mixed node types"""

    def _rigs(self):
        yield "every node", _rig()
        yield "one node of many", _rig(sheared=(4,))
        yield "the last node of many", _rig(sheared=(8,))

    def _clips(self):
        yield "every frame", _sheared_clip(_rig())
        clip = _animated(_rig(sheared=()))

        # one node on one frame: neither the first node nor the first frame
        # says what the block holds
        block             = np.zeros(clip.frames.shear.shape)
        block[3, 4]       = SHEAR
        clip.frames.shear = block
        yield "one node on one frame", clip

    def test_a_rig_reads_what_its_nodes_read(self):
        for label, rig in self._rigs():
            for read in READS:
                with self.subTest(label, read=read):
                    want = np.array([getattr(node, read) for node in rig.list])
                    self.assertLessEqual(_gap(getattr(rig, read), want), 1e-12)

    def test_a_selection_reads_through_the_pose_evaluators(self):
        for label, rig in self._rigs():
            with self.subTest(label):
                local, world, parent = _pose_matrices(rig.list)
                self.assertLessEqual(_gap(local, np.array([n.matrix for n in rig.list])),         1e-12)
                self.assertLessEqual(_gap(world, np.array([n.world_matrix for n in rig.list])),   1e-12)
                self.assertLessEqual(_gap(parent, np.array([n.parent_matrix for n in rig.list])), 1e-12)

    def test_a_scale_that_is_not_finite_stays_out_of_a_stack_with_no_shear(self):
        # inf * 0 is NaN
        scale = np.array([[[1.0, np.inf, 2.0], [np.inf, 1.0, np.inf]]])
        got   = _scale_shears(scale, np.zeros_like(scale))
        self.assertTrue(np.array_equal(got, _diagonal(scale)))

    def test_a_clip_reads_what_its_nodes_read_on_every_frame(self):
        for label, clip in self._clips():
            for read in READS:
                got = getattr(clip.frames, read)
                for frame in range(clip.frame_count):
                    with self.subTest(label, read=read, frame=frame):
                        clip.frame = frame
                        want       = np.array([getattr(node, read) for node in clip.list])
                        self.assertLessEqual(_gap(got[frame], want), 1e-12)

    def test_a_span_of_frames_evaluates_what_the_frames_read(self):
        # the route a clip's world target takes, its rows built one by one
        for label, clip in self._clips():
            with self.subTest(label):
                self.assertLessEqual(_gap(_span_world(clip, slice(1, 4)), clip.frames.world_matrix[1:4]), 1e-12)

    def test_a_frame_axis_solve_is_each_pose_solved(self):
        # each node solves under its parent's final world, which _solve_frames
        # composes from the shear a joint keeps
        clip   = _sheared_clip(_rig())
        target = _animated(_rig(sheared=(), seed=5)).frames.world_matrix

        want   = []
        for frame in range(clip.frame_count):
            pose              = pose_at(clip.copy(), frame)
            pose.world_matrix = target[frame]
            want.append((pose.world_matrix, pose.shear))

        clip.frames.world_matrix = target
        self.assertLessEqual(_gap(clip.frames.world_matrix, np.array([w for w, _ in want])), 1e-9)
        self.assertLessEqual(_gap(clip.frames.shear, np.array([h for _, h in want])), 1e-9)

    def test_skinning_a_sheared_clip_is_skinning_its_poses(self):
        rig, clip = branching_clip()
        mesh, skin = skinned_mesh(rig)
        plain = clip.copy()

        rng               = np.random.default_rng(31)
        clip.frames.shear = rng.uniform(-0.5, 0.5, clip.frames.shear.shape)
        for method in ("lbs", "dqs"):
            with self.subTest(method):
                d    = SkinDeformData(mesh, skin, bind_rig=rig, method=method)
                want = np.stack([d.apply(pose_at(clip.copy(), f)) for f in range(clip.frame_count)])
                got  = d.apply(clip.frames)
                self.assertLessEqual(np.abs(got - want).max(), 1e-9)
                self.assertGreater(np.abs(got - d.apply(plain.frames)).max(), 0.01)

    def test_a_clip_of_deltas_whose_shear_moves(self):
        names  = [f"j{i}" for i in range(6)]
        ups    = [None, "j0", "j1", "j1", "j3", "j4"]
        rig    = self._rig(names, ups, 7, scale=True)
        clip   = self._animated(rig.copy(), seed=8)
        deltas = self._animated(self._rig(names, ups, 7, scale=True), seed=9)

        rng                 = np.random.default_rng(32)
        deltas.frames.shear = rng.uniform(-0.5, 0.5, deltas.frames.shear.shape)
        got                 = clip.add_delta(deltas)
        self.assertFalse(np.array_equal(got.frames.shear[0], got.frames.shear[1]))
        self._assert_same(got, self._per_frame("add_delta", clip, deltas))

    def test_one_delta_holding_shear(self):
        names       = [f"j{i}" for i in range(6)]
        ups         = [None, "j0", "j1", "j1", "j3", "j4"]
        rig         = self._rig(names, ups, 7, scale=True)
        clip        = self._animated(rig.copy(), seed=8)
        delta       = self._rig(names, ups, 7, scale=True)
        delta.shear = SHEAR
        self._assert_same(clip.add_delta(delta), self._per_frame("add_delta", clip, delta))

    def test_a_node_holding_some_of_the_entries(self):
        for shear in PARTIAL_SHEARS:
            with self.subTest(shear=shear):
                node = TransformData("a", shear=shear, scale=(1.0, 2.0, 3.0), rotate=(10.0, 20.0, 30.0))
                self.assertTrue(np.array_equal(node.shear_matrix, _sheared_rows(shear)))
                self.assertLessEqual(_gap(node.matrix, _spelled(node)), 1e-14)

    def test_a_rig_holding_some_of_the_entries(self):
        for shear in PARTIAL_SHEARS:
            for label, rows in (("one node", (4,)), ("every node", range(9))):
                with self.subTest(label, shear=shear):
                    rig = _rig(sheared=())
                    for row in rows:
                        rig.list[row].shear = shear
                    local  = np.array([_spelled(node) for node in rig.list])
                    worlds = _spelled_worlds(rig)
                    self.assertLessEqual(_gap(rig.matrix, local), 1e-12)
                    self.assertLessEqual(_gap(rig.world_matrix, worlds), 1e-12)
                    self.assertLessEqual(_gap(_pose_matrices(rig.list)[0], local), 1e-12)
                    self.assertLessEqual(_gap(_pose_matrices(rig.list)[1], worlds), 1e-12)

    def test_a_clip_holding_some_of_the_entries(self):
        for shear in PARTIAL_SHEARS:
            for label, at in (("one node on one frame", (3, 4)), ("every node", Ellipsis)):
                with self.subTest(label, shear=shear):
                    clip              = _animated(_rig(sheared=()))
                    block             = np.zeros(clip.frames.shear.shape)
                    block[at]         = shear
                    clip.frames.shear = block

                    poses  = [pose_at(clip.copy(), f) for f in range(clip.frame_count)]
                    local  = np.array([[_spelled(node) for node in pose.list] for pose in poses])
                    worlds = np.array([_spelled_worlds(pose) for pose in poses])
                    self.assertLessEqual(_gap(clip.frames.matrix, local), 1e-12)
                    self.assertLessEqual(_gap(clip.frames.world_matrix, worlds), 1e-12)
                    self.assertLessEqual(_gap(_span_world(clip, slice(1, 4)), worlds[1:4]), 1e-12)


class TestShearSolvesDrop(unittest.TestCase):
    """a solve of a target that holds no shear: a node that is no joint ends
    with none and reads back as written, a joint keeps its own"""

    TARGET = dict(scale=(1.0, 2.0, 3.0), rotate=(10.0, 20.0, 30.0), translate=(4.0, 5.0, 6.0))

    def _target(self) -> np.ndarray:
        return TransformData("t", **self.TARGET).matrix

    def test_a_node_that_is_no_joint(self):
        for kind in ("transform", "locator"):
            for write in ("matrix", "world_matrix"):
                with self.subTest(kind=kind, write=write):
                    node = TransformData("a", node_type=kind, shear=SHEAR, scale=(2.0, 2.0, 2.0))
                    setattr(node, write, self._target())
                    self.assertEqual(node.shear.tolist(), [0.0, 0.0, 0.0])
                    self.assertLessEqual(_gap(node.matrix, self._target()), 1e-12)

    def test_a_node_holding_some_of_the_entries(self):
        for shear in PARTIAL_SHEARS:
            for write in ("matrix", "world_matrix"):
                with self.subTest(shear=shear, write=write):
                    node = TransformData("a", shear=shear, scale=(2.0, 2.0, 2.0))
                    setattr(node, write, self._target())
                    self.assertEqual(node.shear.tolist(), [0.0, 0.0, 0.0])
                    self.assertLessEqual(_gap(node.matrix, self._target()), 1e-12)

    def test_a_joint_keeps_its_own(self):
        for write in ("matrix", "world_matrix"):
            with self.subTest(write=write):
                node = TransformData("a", node_type="joint", shear=SHEAR, scale=(2.0, 2.0, 2.0))
                setattr(node, write, self._target())
                self.assertEqual(node.shear.tolist(), list(SHEAR))
                self.assertTrue(np.allclose(node.scale, self.TARGET["scale"], rtol=0.0, atol=1e-12))

    def test_a_list_and_a_rig(self):
        rig        = _rig()
        joints     = np.array([n.node_type == "joint" for n in rig.list])
        before     = rig.shear
        rig.matrix = np.tile(self._target(), (len(rig), 1, 1))
        self.assertTrue(np.array_equal(rig.shear[joints], before[joints]))
        self.assertFalse(rig.shear[~joints].any())
        self.assertLessEqual(_gap(rig.matrix[~joints], self._target()), 1e-12)

    def test_a_clips_frames(self):
        clip   = _sheared_clip(_rig())
        joints = np.array([n.node_type == "joint" for n in clip.list])
        before = np.array(clip.frames.shear)
        shape  = before.shape[:2] + (4, 4)

        clip.frames.matrix = np.broadcast_to(self._target(), shape).copy()
        self.assertTrue(np.array_equal(clip.frames.shear[:, joints], before[:, joints]))
        self.assertFalse(clip.frames.shear[:, ~joints].any())

    def test_a_clip_keeps_a_frame_already_there(self):
        # a frame whose channels make the target keeps them, shear included
        clip               = _sheared_clip(_rig(), frames=3)
        before             = np.array(clip.frames.shear)
        target             = clip.frames.matrix.copy()
        target[1, 1]       = self._target()
        clip.frames.matrix = target
        after              = clip.frames.shear
        self.assertFalse(after[1, 1].any())
        self.assertTrue(np.array_equal(after[0], before[0]))
        self.assertTrue(np.array_equal(after[2], before[2]))

    def test_a_list_keeps_what_a_node_that_stays_where_it_is_holds(self):
        # a root going to the world solves nothing: the channels it holds
        # make its target, shear among them, as on a clip's kept frame
        rig    = HierarchyData([TransformData("a", shear=SHEAR, scale=(1.0, 2.0, 3.0), rotate=(10.0, 20.0, 30.0))])
        before = rig["a"].world_matrix
        TransformList([rig["a"]]).set_parent(None)
        self.assertEqual(rig["a"].shear.tolist(), list(SHEAR))
        self.assertEqual(rig["a"].world_matrix.tolist(), before.tolist())

    def test_a_child_put_back_by_a_swap_is_one_node_at_a_time(self):
        # the list writes what the child's setter writes
        uuids = [generate_uuid() for _ in range(2)]
        rig = HierarchyData(
            [
                TransformData("a", uuid=uuids[0], scale=(1.2, 0.8, 1.5), rotate=(10.0, 20.0, 30.0)),
                TransformData("b", uuid=uuids[1], parent_uuid=uuids[0], translate=(1.0, 2.0, 3.0), shear=SHEAR),
            ]
        )
        turn = rig.copy()
        turn["a"].swapaxes(0, 1)
        batch = rig.copy()
        TransformList([batch["a"]]).swapaxes(0, 1)
        self.assertLessEqual(_gap(batch["b"].shear, turn["b"].shear), 1e-12)
        self.assertLessEqual(_gap(batch.world_matrix, turn.world_matrix), 1e-12)

    def _picked(self, rig, parent):
        # the list's route, and the same nodes one at a time: each to the
        # world, then each under the parent
        names = [node.name for node in rig.list[2:7]]
        batch = rig.copy()
        TransformList([batch[name] for name in names]).set_parent(parent)
        turn = rig.copy()
        for name in names:
            turn[name].set_parent(None)
        if parent is not None:
            for name in names:
                turn[name].set_parent(parent)
        return batch, turn

    def test_set_parent_runs_on_a_rig_holding_shear(self):
        for label, parent in (("the world", None), ("a node", "n0")):
            with self.subTest(label):
                for seed in (11, 12, 13):
                    rig = _rig(seed=seed)
                    batch, turn = self._picked(rig, parent)
                    self.assertTrue(np.isfinite(batch.world_matrix).all())
                    self.assertLessEqual(_gap(batch.world_matrix, turn.world_matrix), 1e-9)

    def test_set_parent_keeps_a_joints_shear(self):
        rig    = _rig()
        names  = [node.name for node in rig.list[2:7] if node.node_type == "joint"]
        before = [rig[name].shear for name in names]
        batch, turn = self._picked(rig, None)
        for name, held in zip(names, before):
            self.assertEqual(batch[name].shear.tolist(), held.tolist())
            self.assertEqual(turn[name].shear.tolist(), held.tolist())


class TestShearFiles(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def _held(self, obj):
        return obj.frames.shear if isinstance(obj, ClipData) else obj.shear

    def _objects(self):
        rig = _rig()
        yield "node", TransformData("a", node_type="joint", shear=SHEAR), TransformData
        yield "rig", rig, HierarchyData
        yield "clip", _sheared_clip(_rig()), ClipData

    def test_every_format_keeps_it(self):
        for label, obj, cls in self._objects():
            for ext in ("json", "npz", "pkl"):
                with self.subTest(label, ext=ext):
                    path = obj.save(os.path.join(self.tmp, f"{label}.{ext}"))
                    back = cls.load(path)
                    self.assertIsInstance(back, cls)
                    self.assertTrue(np.array_equal(self._held(back), self._held(obj)))

    def test_a_pickle_and_a_copy_keep_it(self):
        for label, obj, _ in self._objects():
            for how, make in (
                ("pickle", lambda x: pickle.loads(pickle.dumps(x))),
                ("deepcopy", copy.deepcopy),
                ("copy", lambda x: x.copy()),
            ):
                with self.subTest(label, how=how):
                    self.assertTrue(np.array_equal(self._held(make(obj)), self._held(obj)))

    def test_its_public_name_is_saved(self):
        node = TransformData("a", shear=SHEAR)
        tree = node.to_dict()
        self.assertNotIn("_shear", tree)
        self.assertEqual(np.asarray(tree["shear"]).tolist(), list(SHEAR))

        clip   = _sheared_clip(_rig())
        blocks = clip.to_dict()["blocks"]
        self.assertNotIn("_shear", blocks)
        self.assertTrue(np.array_equal(np.asarray(blocks["shear"]), clip.frames.shear))

    def test_an_all_zero_block_is_left_out_and_one_non_zero_frame_keeps_it(self):
        clip = _animated(_rig(sheared=()), frames=4)
        self.assertNotIn("shear", clip.to_dict()["blocks"])

        # on a frame the clip is not bound to
        clip.frame        = 0
        block             = np.zeros(clip.frames.shear.shape)
        block[2, 3, 1]    = 0.5
        clip.frames.shear = block
        clip.frame        = 0
        tree              = clip.to_dict()
        self.assertTrue(np.array_equal(np.asarray(tree["blocks"]["shear"]), block))
        self.assertTrue(np.array_equal(ClipData.from_dict(tree).frames.shear, block))

    def test_a_clip_tree_with_no_shear_block_loads_zeros(self):
        clip           = _sheared_clip(_rig())
        tree           = clip.to_dict()
        tree["blocks"] = {key: block for key, block in tree["blocks"].items() if key != "shear"}
        back           = ClipData.from_dict(tree)
        self.assertEqual(back.frames.shear.shape, clip.frames.shear.shape)
        self.assertFalse(back.frames.shear.any())
        self.assertFalse(back.shear.any())

    def test_a_tree_missing_another_block_still_raises_naming_only_it(self):
        tree           = _animated(_rig(sheared=())).to_dict()
        tree["blocks"] = {key: block for key, block in tree["blocks"].items() if key != "rotate_axis"}
        with self.assertRaisesRegex(ValueError, r"missing the blocks \['rotate_axis'\]"):
            ClipData.from_dict(tree)

    def test_files_older_than_it_load_with_shear_zero(self):
        for ext in ("json", "npz", "pkl"):
            with self.subTest(ext=ext):
                rig = HierarchyData.load(os.path.join(ASSETS, f"test_hierarchy.legacy_parent_node.{ext}"))
                self.assertGreater(len(rig), 0)
                self.assertFalse(rig.shear.any())

        clip = ClipData.load(os.path.join(ASSETS, "test_hierarchy.legacy_parent_node_clip.json"))
        self.assertEqual(clip.frames.shear.shape, clip.frames.scale.shape)
        self.assertFalse(clip.frames.shear.any())

    def test_files_saved_by_2_0_0_load_with_shear_zero(self):
        with zipfile.ZipFile(os.path.join(ASSETS, "test_data_module.saved_by_2_0_0.zip")) as z:
            z.extractall(self.tmp)
        seen = 0
        for where, _, names in os.walk(self.tmp):
            for name in names:
                if name.startswith("hierarchy") and name.endswith((".json", ".npz", ".pkl")):
                    with self.subTest(name):
                        loaded = HierarchyData.load(os.path.join(where, name))
                        self.assertFalse(loaded.shear.any())
                        seen += 1
        self.assertGreater(seen, 0)

    def test_a_user_attribute_with_its_name_loads_hidden_with_a_warning(self):
        for name, what in (
            ("shear", "a built-in"),
            ("shear_xy", "a built-in"),
            ("shear_xz", "a built-in"),
            ("shear_yz", "a built-in"),
            ("shear_matrix", "a built-in"),
            ("sh", "Maya's short name for shear"),
            ("shxy", "Maya's short name for shear_xy"),
            ("shxz", "Maya's short name for shear_xz"),
            ("shyz", "Maya's short name for shear_yz"),
        ):
            with self.subTest(name):
                tree = TransformData("b").to_dict()
                tree["user_defined_attributes"] = {
                    name: {"attributeType": "double", "value": 1.5, "keyable": False, "channel_box": False}
                }
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    node = TransformData.from_dict(tree)

                self.assertEqual(node.shear.tolist(), [0.0, 0.0, 0.0])
                self.assertEqual(node.user_defined_attributes[name]["value"], 1.5)
                self.assertEqual(
                    [str(w.message) for w in caught],
                    [
                        f"user attribute {name!r} of 'b' loaded into user_defined_attributes[{name!r}]: "
                        f"node.{name} is {what}, which hides it"
                    ],
                )

    def test_add_user_attribute_refuses_its_names(self):
        node = TransformData("a")
        for name in ("shear", "sh", "shxy", "shear_xz", "shear_matrix"):
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, f"'{name}' is a built-in attribute of TransformData"):
                    node.add_user_attribute(name, 1.5)

    def _primitives(self):
        for cls in (SDFSphere, SDFBox, SDFCylinder):
            plain      = cls(translate=(1.0, 2.0, 3.0), rotate=(10.0, 20.0, 30.0), scale=(1.0, 2.0, 3.0))
            held       = plain.copy()
            held.shear = SHEAR
            yield cls, plain, held

    def test_an_sdf_primitive_saves_it(self):
        for cls, plain, held in self._primitives():
            with self.subTest(cls.__name__):
                self.assertEqual(pickle.loads(pickle.dumps(held)).shear.tolist(), list(SHEAR))
                self.assertEqual(copy.deepcopy(held).shear.tolist(), list(SHEAR))

                back = cls.from_dict(held.to_dict())  # a file is built as the class loading it
                self.assertIsInstance(back, cls)
                self.assertEqual(back.shear.tolist(), list(SHEAR))
                self.assertGreater(np.abs(held.matrix - plain.matrix).max(), 0.1)

    def test_an_sdf_primitive_does_not_evaluate_it(self):
        # as it does not evaluate rotate_axis, joint_orient or a parent
        X, Y, Z = np.meshgrid(*[np.linspace(-2.0, 2.0, 5)] * 3, indexing="ij")
        for cls, plain, held in self._primitives():
            with self.subTest(cls.__name__):
                self.assertTrue(np.array_equal(plain.sample(X, Y, Z), held.sample(X, Y, Z)))
                for a, b in zip(plain.bounding_box(), held.bounding_box()):
                    self.assertTrue(np.array_equal(a, b))


class TestShearInARender(unittest.TestCase):
    """the render caches key on the raw channel: a write past the setter
    leaves the cached world as it was, so only the raw key sees it"""

    def _object(self) -> Object:
        return Object("o", scale=(1.0, 2.0, 3.0), rotate=(10.0, 20.0, 30.0))

    def test_every_render_node_takes_it(self):
        for cls in (Object, Camera, Light):
            with self.subTest(cls.__name__):
                self.assertEqual(cls(name="a", shear=SHEAR).shear.tolist(), list(SHEAR))

    def test_an_objects_snapshot(self):
        node = self._object()
        node.world_matrix  # cached
        before = node._capture_transform_state()

        node._shear[...] = SHEAR
        self.assertNotEqual(node._capture_transform_state(), before)

        node       = self._object()
        before     = node._capture_transform_state()
        node.shear = SHEAR
        self.assertNotEqual(node._capture_transform_state(), before)

    def test_a_scenes_signature(self):
        scene = Scene()
        scene.append(Camera("c"))
        scene.append(self._object())
        node = scene["o"]
        node.world_matrix  # cached
        before = scene._compute_render_signature()

        node._shear[...] = SHEAR
        self.assertNotEqual(scene._compute_render_signature(), before)

        node.shear = (0.0, 0.0, 0.0)
        before     = scene._compute_render_signature()
        node.shear = SHEAR
        self.assertNotEqual(scene._compute_render_signature(), before)


def _sh3(shear) -> np.ndarray:
    """(xy, xz, yz) as the 3x3 Sh"""
    return _sheared_rows(shear)[:3, :3]


def _turn(rotate, order: int = 0) -> np.ndarray:
    """a 3x3 rotation"""
    return TransformData("q", rotate=rotate, rotate_order=order).rotate_matrix[:3, :3]


def _maya_orientation(block) -> np.ndarray:
    """Maya's split of a 3x3, spelled out here: Gram-Schmidt on rows x, y, z,
    row z negated where the block mirrors"""
    block = np.asarray(block, dtype=float)
    u0    = block[0] / np.linalg.norm(block[0])
    v1    = block[1] - block[1] @ u0 * u0
    u1    = v1 / np.linalg.norm(v1)
    v2    = block[2] - block[2] @ u0 * u0 - block[2] @ u1 * u1
    u2    = v2 / np.linalg.norm(v2)
    if np.linalg.det(block) < 0:
        u2 = -u2
    return np.array([u0, u1, u2])


class TestShearSplit(unittest.TestCase):
    """Maya's split of a 3x3 block, a row at a time: S * Sh * Q, the shear
    read off with the signed scale, a mirror on scale z"""

    HELD = np.array([0.11, 0.22, 0.33])

    def setUp(self):
        self.q       = _turn((10.0, 20.0, 30.0))
        self.current = _turn((50.0, -20.0, 70.0))

    def both(self, block, current=None, held=None) -> tuple:
        """the single split, and the stack's on a stack of one"""
        current = self.current if current is None else current
        held    = self.HELD if held is None else held
        one     = H._scale_shear_and_rotation(block, current, held)
        stack   = H._scales_shears_and_rotations(block[None], current[None], held[None])
        return one, tuple(x[0] for x in stack)

    def test_a_sheared_block_comes_apart_on_every_rotate_order(self):
        rng = np.random.default_rng(41)
        for order in range(6):
            for _ in range(60):
                scale  = rng.uniform(0.2,    3.0,   3)
                shear  = rng.uniform(-2.0,   2.0,   3)
                rotate = rng.uniform(-170.0, 170.0, 3)
                Q      = _turn(rotate, order)
                s, r, h = H._scale_shear_and_rotation(np.diag(scale) @ _sh3(shear) @ Q, self.current, self.HELD)
                with self.subTest(order=order):
                    self.assertLessEqual(np.abs(s - scale).max(), 1e-12)
                    self.assertLessEqual(np.abs(h - shear).max(), 1e-12)
                    self.assertLessEqual(np.abs(r - Q).max(),     1e-12)

    # a block's negative scale axes: the scale's signs the split gives, and
    # the signs (xy, xz, yz) take: Maya's rule, each row's dot with the
    # rotation's rows over the row's signed scale
    MIRRORS = (
        ((-1, 1, 1), (1, 1, -1), (-1, 1, -1)),
        ((1, -1, 1), (1, 1, -1), (-1, -1, 1)),
        ((1, 1, -1), (1, 1, -1), (1, 1, 1)),
        ((-1, -1, 1), (1, 1, 1), (1, -1, -1)),
        ((-1, 1, -1), (1, 1, 1), (-1, 1, -1)),
        ((1, -1, -1), (1, 1, 1), (-1, -1, 1)),
        ((-1, -1, -1), (1, 1, -1), (1, -1, -1)),
    )

    def test_a_mirror_goes_on_scale_z_and_signs_the_shear(self):
        rng = np.random.default_rng(42)
        for signs, scales, shears in self.MIRRORS:
            for _ in range(30):
                scale = rng.uniform(0.2, 3.0, 3)
                shear = rng.uniform(-2.0, 2.0, 3)
                block = np.diag(scale * signs) @ _sh3(shear) @ _turn(rng.uniform(-170.0, 170.0, 3))
                s, r, h = H._scale_shear_and_rotation(block, self.current, self.HELD)
                with self.subTest(signs=signs):
                    self.assertLessEqual(np.abs(s - scale * scales).max(),      1e-12)
                    self.assertLessEqual(np.abs(h - shear * shears).max(),      1e-12)
                    self.assertLessEqual(_gap(np.diag(s) @ _sh3(h) @ r, block), 1e-14)
                    self.assertGreater(np.linalg.det(r), 0.0)

    def test_a_plain_block_is_the_row_length_split_to_the_bit(self):
        rng = np.random.default_rng(43)
        for _ in range(300):
            scale = rng.uniform(0.01, 100.0, 3) * rng.choice([1.0, -1.0], 3)
            block = np.diag(scale) @ _turn(rng.uniform(-170.0, 170.0, 3), int(rng.integers(0, 6)))
            s0, r0 = H._scale_and_rotation(block, self.current)
            s1, r1, h1 = H._scale_shear_and_rotation(block, self.current, self.HELD)
            self.assertTrue(np.array_equal(s0, s1))
            self.assertTrue(np.array_equal(r0, r1))
            self.assertEqual(h1.tolist(), [0.0, 0.0, 0.0])

        # a zero scale too: the row a row-length split fills from ``current``
        for zero in itertools.product((False, True), repeat=3):
            for _ in range(20):
                scale  = np.where(zero, 0.0, rng.uniform(0.1, 10.0, 3))
                block  = np.diag(scale) @ _turn(rng.uniform(-170.0, 170.0, 3))
                turned = _turn(rng.uniform(-170.0, 170.0, 3))
                s0, r0 = H._scale_and_rotation(block, turned)
                s1, r1, _ = H._scale_shear_and_rotation(block, turned, self.HELD)
                with self.subTest(zero=zero):
                    self.assertTrue(np.array_equal(s0, s1))
                    self.assertTrue(np.array_equal(r0, r1))

    def test_every_zero_scale_recomposes_and_keeps_a_flat_rows_entries(self):
        for zero in itertools.product((False, True), repeat=3):
            if not any(zero):
                continue
            for shear in ((0.0, 0.0, 0.0), (0.5, 0.25, 0.75)):
                scale = np.where(zero, 0.0, (2.0, 3.0, 4.0))
                block = np.diag(scale) @ _sh3(shear) @ self.q
                one, two = self.both(block)
                with self.subTest(zero=zero, shear=shear):
                    for s, r, h in (one, two):
                        self.assertLessEqual(_gap(np.diag(s) @ _sh3(h) @ r, block), 1e-14)
                        if zero[1]:
                            self.assertEqual(h[0], self.HELD[0])
                        if zero[2]:
                            self.assertEqual(h[1:].tolist(), self.HELD[1:].tolist())
                    for a, b in zip(one, two):
                        self.assertLessEqual(np.abs(a - b).max(), 1e-14)

    def test_a_tiny_row_keeps_its_row_length_and_a_child_stays_finite(self):
        for axis in range(3):
            scale       = np.array([2.0, 3.0, 4.0])
            scale[axis] = 1e-13
            block       = np.diag(scale) @ _sh3((0.5, 0.25, 0.75)) @ self.q
            one, two = self.both(block)
            row_length = H._scale_and_rotation(block, self.current)[0]
            with self.subTest(axis=axis):
                self.assertEqual(one[0][axis], row_length[axis])
                self.assertEqual(two[0][axis], row_length[axis])
                self.assertGreater(one[0][axis], 0.0)

                M         = np.eye(4)
                M[:3, :3] = block
                parent    = TransformData("a", uuid=generate_uuid())
                child = TransformData(
                    "b",
                    uuid=generate_uuid(),
                    parent_uuid=parent.uuid,
                    node_type="joint",
                    translate=(1.0, 2.0, 3.0),
                    segment_scale_compensate=True,
                )
                rig             = HierarchyData([parent, child])
                clip            = ClipData(rig.copy(), frames=2)
                rig["a"].matrix = M
                self.assertTrue(np.isfinite(rig["b"].world_matrix).all())

                held               = clip.frames.matrix.copy()
                held[:, 0]         = M
                clip.frames.matrix = held
                self.assertTrue(np.isfinite(clip.frames.world_matrix).all())

    def test_a_row_parallel_to_the_one_above_is_flat(self):
        block = np.array([self.q[0] * 2.0, self.q[0] * 3.0, self.q[2] * 1.5])
        one, two = self.both(block)
        for s, r, h in (one, two):
            self.assertAlmostEqual(s[1], 3e-12, delta=1e-27)
            self.assertEqual(h[0], self.HELD[0])
            self.assertTrue(np.isfinite(s).all() and np.isfinite(r).all())

    def test_an_exact_zero_row_is_scale_zero(self):
        block = np.array([self.q[0] * 2.0, self.q[0] + self.q[1], 0.0 * self.q[2]])
        one, two = self.both(block)
        for s, _, h in (one, two):
            self.assertEqual(s[2], 0.0)
            self.assertEqual(h[1:].tolist(), self.HELD[1:].tolist())

    def test_a_cosine_past_1e_12_reads_shear(self):
        # the row length split stands below it, shear exactly 0, on each pair
        for entry in range(3):
            for value, want in ((5e-13, 0.0), (2e-12, 2e-12)):
                shear        = np.zeros(3)
                shear[entry] = value
                block        = _sh3(shear) @ self.q
                for _, _, h in self.both(block, held=np.zeros(3)):
                    with self.subTest(entry=entry, value=value):
                        self.assertAlmostEqual(h[entry], want, delta=1e-16)
        _, _, h = H._scale_shear_and_rotation(np.diag((1.0, 2.0, 3.0)) @ self.q, self.current, np.zeros(3))
        self.assertEqual(h.tolist(), [0.0, 0.0, 0.0])

    def test_two_passes_square_near_parallel_rows(self):
        block = np.array([self.q[0], self.q[0] + 1e-8 * self.q[1], self.q[2]])
        _, r, _ = H._scale_shear_and_rotation(block, self.current, self.HELD)
        self.assertLessEqual(np.abs(r @ r.T - np.eye(3)).max(), 1e-14)

    def test_a_block_that_is_not_finite_stays_on_the_row_length_split(self):
        for value in (np.nan, np.inf):
            for i in range(3):
                block       = np.array(self.q)
                block[i, i] = value
                one, two = self.both(block)
                old = H._scale_and_rotation(block, self.current)
                with self.subTest(value=value, row=i):
                    for got in (one, two):
                        self.assertTrue(np.array_equal(got[0], old[0], equal_nan=True))
                        self.assertTrue(np.array_equal(got[1], old[1], equal_nan=True))
                        self.assertEqual(got[2].tolist(), self.HELD.tolist())

    def test_a_current_that_is_not_finite_on_flat_rows_reads_plain(self):
        block = np.array([[0.0, 0.0, 0.0], self.q[1] * 2.0, [0.0, 0.0, 0.0]])
        one, two = self.both(block, current=np.full((3, 3), np.nan))
        self.assertTrue(np.isfinite(one[2]).all())
        self.assertTrue(np.isfinite(two[2]).all())

    def test_the_stack_is_each_element_and_a_sheared_one_to_the_bit(self):
        rng        = np.random.default_rng(44)
        count      = 400
        scale      = rng.uniform(0.1, 5.0, (count, 3)) * rng.choice([1.0, 1.0, -1.0], (count, 3))
        shear      = rng.uniform(-1.5, 1.5, (count, 3))
        shear[::3] = 0.0
        blocks = np.array(
            [np.diag(s) @ _sh3(h) @ _turn(r) for s, h, r in zip(scale, shear, rng.uniform(-170.0, 170.0, (count, 3)))]
        )
        held    = rng.uniform(-1.0, 1.0, (count, 3))
        current = np.broadcast_to(self.current, blocks.shape)
        stack   = H._scales_shears_and_rotations(blocks, current, held)
        for i in range(count):
            single = H._scale_shear_and_rotation(blocks[i], current[i], held[i])
            for a, b in zip(stack, single):
                if i % 3:
                    self.assertTrue((a[i] == b).all(), i)
                else:
                    self.assertLessEqual(np.abs(a[i] - b).max(), 1e-13, i)


class TestShearFlatRowRoutes(unittest.TestCase):
    """what a flat row keeps reaches the split through its callers: each hands
    it the shear the node holds and the orientation it has"""

    HELD = np.array((0.11, 0.22, 0.33))
    FLAT = ((1, [0]), (2, [1, 2]))  # the row made flat, the entries it defines

    def _target(self, zero) -> np.ndarray:
        scale       = np.array([2.0, 3.0, 4.0])
        scale[zero] = 0.0
        M           = np.eye(4)
        M[:3, :3]   = np.diag(scale) @ _sh3((0.5, 0.25, 0.75)) @ _turn((10.0, 20.0, 30.0))
        return M

    def test_the_setter(self):
        for zero, kept in self.FLAT:
            with self.subTest(zero=zero):
                node        = TransformData("a", shear=self.HELD)
                node.matrix = self._target(zero)
                self.assertEqual(node.shear[kept].tolist(), self.HELD[kept].tolist())

    def test_a_clip(self):
        for zero, kept in self.FLAT:
            with self.subTest(zero=zero):
                clip               = ClipData(HierarchyData([TransformData("a", shear=self.HELD)]), frames=2)
                clip.frames.matrix = self._target(zero)
                self.assertEqual(clip.frames.shear[1, 0, kept].tolist(), self.HELD[kept].tolist())

    def test_a_lists_set_parent(self):
        for zero, kept in self.FLAT:
            with self.subTest(zero=zero):
                scale       = np.ones(3)
                scale[zero] = 0.0
                up          = generate_uuid()
                rig = HierarchyData(
                    [
                        TransformData("p", uuid=up, scale=(1.0, 2.0, 3.0), rotate=(30.0, 40.0, 50.0)),
                        TransformData("c", scale=scale, shear=self.HELD, rotate=(10.0, 0.0, 0.0), translate=(1.0, 1.0, 1.0)),
                    ]
                )
                TransformList([rig["c"]]).set_parent("p")
                self.assertEqual(rig["c"].shear[kept].tolist(), self.HELD[kept].tolist())

    def _swapped(self, how, scale, rotate_axis=(5.0, 10.0, 15.0), axes=(1, 2)) -> HierarchyData:
        rig = HierarchyData(
            [TransformData("a", scale=scale, shear=self.HELD, rotate=(10.0, 20.0, 30.0), rotate_axis=rotate_axis)]
        )
        if how == "one node":
            rig["a"].swapaxes(*axes)
        else:
            TransformList([rig["a"]]).swapaxes(*axes)
        return rig

    def test_swapaxes_keeps_the_entries_of_a_flat_row(self):
        for how in ("one node", "a list"):
            with self.subTest(how):
                rig = self._swapped(how, (2.0, 0.0, 4.0))
                self.assertEqual(rig["a"].shear[1:].tolist(), self.HELD[1:].tolist())

    def test_swapaxes_a_mirror_beside_a_flat_row(self):
        # a flat row reads no sign of its own: it takes the one the mirror
        # leaves over, or the rotation would come out a mirror. With every
        # scale negative the split reads no mirror, so all three of its signs
        # part from the permuted scale's, and the two real rows' go back. The
        # shortest row takes it wherever it lands: on z, on x, on y
        rows = (
            ((1, 2), (-2.0, 0.0, 4.0)),
            ((1, 2), (2.0, 0.0, -4.0)),
            ((1, 2), (-2.0, 1e-13, 4.0)),
            ((1, 2), (-2.0, -1e-13, -4.0)),
            ((0, 1), (-2.0, 0.0, 4.0)),
            ((1, 2), (-2.0, 3.0, 0.0)),
        )
        for axes, scale in rows:
            P    = H._axis_permutation(*axes, False)
            want = np.array(scale)

            want[list(axes)] = want[list(axes[::-1])]

            real = np.abs(want) > 1e-12
            for how in ("one node", "a list"):
                with self.subTest(axes=axes, scale=scale, how=how):
                    node   = TransformData("a", scale=scale, shear=self.HELD, rotate=(10.0, 20.0, 30.0))
                    before = node.world_matrix
                    rig    = self._swapped(how, scale, None, axes)
                    self.assertLessEqual(_gap(rig["a"].world_matrix[:3, :3], (P @ before)[:3, :3]), 1e-12)
                    self.assertEqual(np.sign(rig["a"].scale)[real].tolist(), np.sign(want)[real].tolist())

    def test_swapaxes_fills_two_flat_rows_from_the_orientation_it_has(self):
        P    = H._axis_permutation(1, 2, False)[:3, :3]
        turn = _turn((10.0, 20.0, 30.0))
        for how in ("one node", "a list"):
            with self.subTest(how):
                rig = self._swapped(how, (2.0, 0.0, 0.0))
                self.assertLessEqual(_gap(rig["a"].rotate_matrix[:3, :3], P @ turn @ P.T), 1e-12)

    def test_a_plain_block_with_a_tiny_row(self):
        # the shorter of the row and the floor: one route with no floor reads
        # the row as a row and parts from the other
        turn  = _turn((10.0, 20.0, 30.0))
        block = np.diag((2.0, 1e-13, 3.0)) @ turn
        self.assertEqual(H._scale_shear_and_rotation(block, turn, self.HELD)[2][0], self.HELD[0])
        self.assertEqual(H._scales_shears_and_rotations(block[None], turn[None], self.HELD[None])[2][0, 0], self.HELD[0])

        # 1e-11 of the longest is past the floor on both routes: a row, whose
        # entries the block sets
        for axis, kept in self.FLAT:
            scale       = np.array([2.0, 3.0, 4.0])
            scale[axis] = 4e-11
            block       = np.diag(scale) @ turn
            one         = H._scale_shear_and_rotation(block, turn, self.HELD)[2]
            stack       = H._scales_shears_and_rotations(block[None], turn[None], self.HELD[None])[2][0]
            for h in (one, stack):
                with self.subTest(axis=axis):
                    self.assertEqual(h[kept].tolist(), [0.0] * len(kept))


class TestShearRoundTrips(unittest.TestCase):
    """what a node that is no joint is told it holds, sheared targets under
    sheared and uneven chains, rotate axes and compensation included"""

    def _placed(self, rng):
        """the last of a chain under a sheared uneven parent, its target's
        matrix, and a free transform at that world"""
        uuids = [generate_uuid() for _ in range(3)]
        nodes = []
        for i in range(3):
            nodes.append(
                TransformData(
                    f"n{i}",
                    uuid=uuids[i],
                    parent_uuid=uuids[i - 1] if i else None,
                    scale=rng.uniform(0.3, 2.5, 3),
                    shear=rng.uniform(-1.0, 1.0, 3),
                    rotate=rng.uniform(-170.0, 170.0, 3),
                    translate=rng.uniform(-5.0, 5.0, 3),
                    rotate_order=int(rng.integers(0, 6)),
                    rotate_axis=rng.uniform(-40.0, 40.0, 3) if rng.random() < 0.5 else None,
                    segment_scale_compensate=bool(i and rng.random() < 0.5),
                )
            )
        rig = HierarchyData(nodes)
        target = TransformData(
            "t",
            scale     = rng.uniform(0.3, 2.5, 3) * rng.choice([1.0, 1.0, -1.0], 3),
            shear     = rng.uniform(-1.0, 1.0, 3),
            rotate    = rng.uniform(-170.0, 170.0, 3),
            translate = rng.uniform(-5.0, 5.0, 3),
        )
        return rig, rig["n2"], target

    def test_a_sheared_target_reads_back(self):
        target = TransformData(
            "t", scale=(1.0, 2.0, 3.0), shear=(0.5, -0.25, 0.75), rotate=(10.0, 20.0, 30.0), translate=(4.0, 5.0, 6.0)
        )
        node        = TransformData("a")
        node.matrix = target.matrix
        self.assertLessEqual(_gap(node.matrix, target.matrix),        1e-12)
        self.assertLessEqual(np.abs(node.shear - target.shear).max(), 1e-12)
        self.assertLessEqual(np.abs(node.scale - target.scale).max(), 1e-12)

    def test_a_target_holding_some_of_the_entries_reads_back(self):
        for shear in PARTIAL_SHEARS:
            with self.subTest(shear=shear):
                target = TransformData(
                    "t", scale=(1.0, 2.0, 3.0), shear=shear, rotate=(10.0, 20.0, 30.0), translate=(4.0, 5.0, 6.0)
                )
                node        = TransformData("a")
                node.matrix = target.matrix
                self.assertLessEqual(_gap(node.matrix, target.matrix), 1e-12)
                self.assertLessEqual(np.abs(node.shear - target.shear).max(), 1e-12)

    def test_a_target_at_no_turn_or_a_quarter_turn_reads_back(self):
        # the rests of such a block hold exact zeros, which no random turn's do
        for rotate in ((0.0, 0.0, 0.0), (0.0, 0.0, 90.0), (90.0, 0.0, 0.0), (0.0, 90.0, 0.0)):
            target = TransformData(
                "t", scale=(1.0, 2.0, 3.0), shear=SHEAR, rotate=rotate, translate=(4.0, 5.0, 6.0)
            )
            node               = TransformData("a")
            node.matrix        = target.matrix
            clip               = ClipData(HierarchyData([TransformData("a")]), frames=2)
            clip.frames.matrix = target.matrix
            for label, matrix, scale, shear in (
                ("one node", node.matrix, node.scale, node.shear),
                ("a clip", clip.frames.matrix[1, 0], clip.frames.scale[1, 0], clip.frames.shear[1, 0]),
            ):
                with self.subTest(label, rotate=rotate):
                    self.assertLessEqual(_gap(matrix, target.matrix),        1e-12)
                    self.assertLessEqual(np.abs(scale - target.scale).max(), 1e-12)
                    self.assertLessEqual(np.abs(shear - target.shear).max(), 1e-12)

    def test_matrix_reads_back(self):
        rng = np.random.default_rng(51)
        for _ in range(150):
            _, node, target = self._placed(rng)
            node.matrix = target.matrix
            self.assertLessEqual(_gap(node.matrix, target.matrix), 1e-12)

    def test_world_matrix_reads_back(self):
        rng = np.random.default_rng(52)
        for _ in range(150):
            _, node, target = self._placed(rng)
            node.world_matrix = target.matrix
            self.assertLessEqual(_gap(node.world_matrix, target.matrix), 1e-11)

    def test_match_matrix_lands_on_its_target(self):
        rng = np.random.default_rng(53)
        for _ in range(150):
            _, node, target = self._placed(rng)
            node.match_matrix(target)
            self.assertLessEqual(_gap(node.world_matrix, target.world_matrix), 1e-11)

    def test_match_translate_moves_the_position_alone(self):
        rng = np.random.default_rng(54)
        for _ in range(150):
            _, node, target = self._placed(rng)
            before = node.world_matrix
            node.match_translate(target)
            after = node.world_matrix
            self.assertLessEqual(_gap(after[:3, :3], before[:3, :3]), 1e-11)
            self.assertLessEqual(_gap(after[3, :3], target.world_matrix[3, :3]), 1e-11)

    def test_a_rig_with_no_shear_takes_what_its_uneven_parents_need(self):
        # a turned uneven parent's world shears: the child's local holds it
        parent                = TransformData("p", uuid=generate_uuid(), scale=(1.0, 2.0, 3.0), rotate=(30.0, 40.0, 50.0))
        child                 = TransformData("c", uuid=generate_uuid(), parent_uuid=parent.uuid, translate=(1.0, 1.0, 1.0))
        rig                   = HierarchyData([parent, child])
        target                = TransformData("t", rotate=(10.0, 20.0, 30.0), translate=(1.0, 2.0, 3.0)).matrix
        rig["c"].world_matrix = target
        self.assertLessEqual(_gap(rig["c"].world_matrix, target), 1e-12)
        self.assertGreater(np.abs(rig["c"].shear).max(), 0.01)


class TestShearJointMirror(unittest.TestCase):
    """a joint holding shear with a mirror of its own on x or y: the split moves
    the mirror to z and the shear stays, so a solve moves the joint, as
    xform -m does, even to the matrix it has. On z, or with no shear, it holds.
    Moved to a parent of the other handedness the mirror of its block moves
    with none of its own, and the split signs the xz and yz it keeps"""

    SCALE = dict(x=(-1.3, 1.1, 0.9), y=(1.3, -1.1, 0.9), z=(1.3, 1.1, -0.9), none=(1.3, 1.1, 0.9))
    ODD   = ((-1.0, 1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, -1.0), (-1.0, -1.0, -1.0))

    def _rig(self, scale, shear=SHEAR) -> HierarchyData:
        uuids = [generate_uuid() for _ in range(3)]
        return HierarchyData(
            [
                TransformData("p", uuid=uuids[0], scale=(1.5, 1.5, 1.5), rotate=(30.0, 40.0, 50.0)),
                TransformData(
                    "j",
                    uuid        = uuids[1],
                    parent_uuid = uuids[0],
                    node_type   = "joint",
                    scale       = scale,
                    shear       = shear,
                    rotate      = (10.0, 20.0, 30.0),
                    translate   = (1.0, 2.0, 3.0),
                ),
                TransformData("c", uuid=uuids[2], parent_uuid=uuids[1], translate=(3.0, 2.0, 1.0)),
            ]
        )

    def _edits(self):
        yield "match_translate", "j", lambda rig: rig["j"].match_translate(rig["j"].world_matrix[3, :3] + 1.0)
        yield "a parent's swapaxes", "j", lambda rig: rig["p"].swapaxes(0, 1)
        yield "set_parent", "j", lambda rig: rig["j"].set_parent(None)
        yield "a list's swapaxes", "j", lambda rig: TransformList([rig["p"]]).swapaxes(0, 1)
        yield "a list's set_parent", "j", lambda rig: TransformList([rig["j"]]).set_parent(None)

    def test_a_solve_moves_it_on_x_or_y(self):
        for axis in "xy":
            for label, name, edit in self._edits():
                with self.subTest(label, axis=axis):
                    rig    = self._rig(self.SCALE[axis])
                    before = rig[name].world_matrix[:3, :3]
                    edit(rig)
                    self.assertEqual(rig[name].shear.tolist(), list(SHEAR))
                    self.assertGreater(_gap(rig[name].world_matrix[:3, :3], before), 0.1)

    def test_on_z_or_with_no_shear_it_holds(self):
        for scale, shear in (
            (self.SCALE["z"], SHEAR),
            (self.SCALE["none"], SHEAR),
            (self.SCALE["x"], None),
            (self.SCALE["y"], None),
        ):
            for label, name, edit in self._edits():
                with self.subTest(label, scale=scale, shear=shear):
                    rig    = self._rig(scale, shear)
                    before = rig[name].world_matrix[:3, :3]
                    edit(rig)
                    self.assertLessEqual(_gap(rig[name].world_matrix[:3, :3], before), 1e-12)

    def test_a_clip_lands_where_one_node_does(self):
        for axis in "xy":
            with self.subTest(axis=axis):
                rig    = self._rig(self.SCALE[axis])
                clip   = ClipData(rig.copy(), frames=2)

                target = clip.frames.world_matrix.copy()
                target[:, 1, 3, :3] += 1.0
                clip.frames.world_matrix = target
                rig["j"].world_matrix    = target[0, 1]

                self.assertEqual(clip.frames.shear[1, 1].tolist(), list(SHEAR))
                self.assertLessEqual(_gap(clip.frames.world_matrix[1, 1], rig["j"].world_matrix), 1e-12)

    def _between(self, above, below, shear=SHEAR, node_type="joint") -> HierarchyData:
        """j and its child under an even parent p of ``above``'s signs, and an
        even q of ``below``'s beside them"""
        uuids = [generate_uuid() for _ in range(3)]
        return HierarchyData(
            [
                TransformData("q", scale=1.2 * np.array(below), rotate=(20.0, -70.0, 10.0), translate=(0.5, 1.0, 1.5)),
                TransformData("p", uuid=uuids[0], scale=1.5 * np.array(above), rotate=(30.0, 40.0, 50.0)),
                TransformData(
                    "j",
                    uuid        = uuids[1],
                    parent_uuid = uuids[0],
                    node_type   = node_type,
                    scale       = (1.3, 1.1, 0.9),
                    shear       = shear,
                    rotate      = (10.0, 20.0, 30.0),
                    translate   = (1.0, 2.0, 3.0),
                ),
                TransformData("c", uuid=uuids[2], parent_uuid=uuids[1], translate=(3.0, 2.0, 1.0)),
            ]
        )

    def _moves(self):
        yield "set_parent", lambda rig: rig["j"].set_parent("q")
        yield "a list's set_parent", lambda rig: TransformList([rig["j"]]).set_parent("q")

    def test_a_parent_of_the_other_handedness_moves_it(self):
        for above, below in itertools.chain(
            (((1.0, 1.0, 1.0), sign) for sign in self.ODD), ((sign, (1.0, 1.0, 1.0)) for sign in self.ODD)
        ):
            for label, move in self._moves():
                with self.subTest(label, above=above, below=below):
                    rig    = self._between(above, below)
                    before = rig["j"].world_matrix[:3, :3]
                    move(rig)
                    self.assertEqual(rig["j"].shear.tolist(), list(SHEAR))
                    self.assertGreater(_gap(rig["j"].world_matrix[:3, :3], before), 0.1)

                    # what it took is the split of the block it asked for,
                    # which signs the two entries the mirror moved
                    block = before @ np.linalg.inv(rig["q"].world_matrix[:3, :3])
                    s, q, want = H._scale_shear_and_rotation(block, np.eye(3), np.zeros(3))
                    self.assertLessEqual(_gap(rig["j"].scale, s), 1e-12)
                    self.assertLessEqual(_gap(rig["j"].rotate_matrix[:3, :3], q), 1e-12)
                    self.assertLessEqual(np.abs(want - (SHEAR[0], -SHEAR[1], -SHEAR[2])).max(), 1e-12)

    def test_a_parent_of_the_same_handedness_holds_it(self):
        for above, below in (
            ((1.0, 1.0, 1.0), (-1.0, -1.0, 1.0)),
            ((-1.0, -1.0, 1.0), (1.0, 1.0, 1.0)),
            ((-1.0, 1.0, 1.0), (1.0, -1.0, 1.0)),
            ((-1.0, 1.0, 1.0), (-1.0, -1.0, -1.0)),
        ):
            for label, move in self._moves():
                with self.subTest(label, above=above, below=below):
                    rig    = self._between(above, below)
                    before = rig["j"].world_matrix[:3, :3]
                    move(rig)
                    self.assertLessEqual(_gap(rig["j"].world_matrix[:3, :3], before), 1e-12)

    def test_the_other_handedness_holds_what_holds_no_xz_or_yz(self):
        for sign in self.ODD:
            for shear, node_type in (((0.5, 0.0, 0.0), "joint"), (None, "joint"), (SHEAR, "transform")):
                for label, move in self._moves():
                    with self.subTest(label, sign=sign, shear=shear, node_type=node_type):
                        rig    = self._between((1.0, 1.0, 1.0), sign, shear, node_type)
                        before = rig["j"].world_matrix[:3, :3]
                        move(rig)
                        self.assertLessEqual(_gap(rig["j"].world_matrix[:3, :3], before), 1e-12)


class TestShearBatchedSolves(unittest.TestCase):
    """solves over many frames and many nodes against one pose at a time, and
    a guard that sees a loop"""

    def _worlds(self, seed: int) -> np.ndarray:
        # targets of a rig of another shape, sheared
        return _sheared_clip(_rig(seed=seed), seed=seed + 1).frames.world_matrix

    @staticmethod
    def _no_joint(matrices, clip) -> np.ndarray:
        """the columns of (F, N, 4, 4) ``matrices`` that are no joint: those
        hold the target they were given"""
        return matrices[:, [i for i, node in enumerate(clip.list) if node.node_type != "joint"]]

    def _pose_by_pose(self, clip, write: str, target) -> np.ndarray:
        got = []
        for frame in range(clip.frame_count):
            pose = pose_at(clip.copy(), frame)
            for row in pose._parent_first():
                setattr(pose[row], write, target[frame, row])
            got.append(pose.world_matrix)
        return np.array(got)

    def test_a_clips_world_matrix(self):
        clip   = _sheared_clip(_rig())
        target = self._worlds(5)
        want   = self._pose_by_pose(clip, "world_matrix", target)

        clip.frames.world_matrix = target
        self.assertLessEqual(_gap(clip.frames.world_matrix, want), 1e-11)
        self.assertLessEqual(_gap(self._no_joint(clip.frames.world_matrix, clip), self._no_joint(target, clip)), 1e-11)

    def test_a_clips_matrix(self):
        clip   = _sheared_clip(_rig())
        target = _sheared_clip(_rig(seed=5), seed=6).frames.matrix
        want   = self._pose_by_pose(clip, "matrix", target)

        clip.frames.matrix = target
        self.assertLessEqual(_gap(clip.frames.world_matrix, want), 1e-11)
        self.assertLessEqual(_gap(self._no_joint(clip.frames.matrix, clip), self._no_joint(target, clip)), 1e-11)

    def test_a_clips_match_matrix(self):
        clip   = _sheared_clip(_rig())
        other  = _sheared_clip(_rig(seed=5), seed=6)
        want   = self._pose_by_pose(clip, "world_matrix", other.frames.world_matrix)

        target = other.frames.world_matrix.copy()
        clip.frames.match_matrix(other)
        self.assertLessEqual(_gap(clip.frames.world_matrix, want), 1e-11)
        self.assertLessEqual(_gap(self._no_joint(clip.frames.world_matrix, clip), self._no_joint(target, clip)), 1e-11)

    def test_a_list_and_a_rig(self):
        rig    = _rig()
        target = self._worlds(5)[0]
        turn   = rig.copy()
        for row in turn._parent_first():
            turn[row].world_matrix = target[row]

        rig.world_matrix = target
        self.assertLessEqual(_gap(rig.world_matrix, turn.world_matrix), 1e-11)
        none = np.array([node.node_type != "joint" for node in rig.list])
        self.assertLessEqual(_gap(rig.world_matrix[none], target[none]), 1e-11)

        view              = rig[2:6]
        view.world_matrix = target[2:6] * 1.0
        self.assertLessEqual(_gap(rig[2:6].world_matrix, turn[2:6].world_matrix), 1e-11)

    def test_a_compensated_transform(self):
        # cgmath compensates any node: one that is no joint takes the shear
        # its target asks, by the batch as by the single setter
        rng = np.random.default_rng(23)
        for trial in range(30):
            up = generate_uuid()
            rig = HierarchyData(
                [
                    TransformData(
                        "a",
                        uuid      = up,
                        scale     = rng.uniform(0.5, 2.0, 3),
                        rotate    = rng.uniform(-170.0, 170.0, 3),
                        translate = rng.uniform(-2.0, 2.0, 3),
                    ),
                    TransformData(
                        "c",
                        parent_uuid=up,
                        scale=rng.uniform(0.5, 2.0, 3),
                        rotate=rng.uniform(-170.0, 170.0, 3),
                        translate=rng.uniform(-2.0, 2.0, 3),
                        segment_scale_compensate=True,
                    ),
                ]
            )
            M         = np.eye(4)
            M[:3, :3] = np.diag(rng.uniform(0.5, 2.0, 3)) @ _sh3(rng.uniform(-0.8, 0.8, 3)) @ _turn(rng.uniform(-170.0, 170.0, 3))
            M[3, :3]  = rng.uniform(-2.0, 2.0, 3)

            single                        = rig.copy()
            single["c"].world_matrix      = M
            clip                          = ClipData(rig.copy(), frames=2)
            clip.frames[[1]].world_matrix = M

            local                     = rig.copy()
            local["c"].matrix         = M
            framed                    = ClipData(rig.copy(), frames=2)
            framed.frames[[1]].matrix = M
            with self.subTest(trial=trial):
                self.assertLessEqual(_gap(clip.frames.world_matrix[1, 1], M), 1e-11)
                self.assertLessEqual(_gap(clip.frames.world_matrix[1, 1], single["c"].world_matrix), 1e-11)
                self.assertLessEqual(_gap(framed.frames.matrix[1, 1], M), 1e-11)
                self.assertLessEqual(_gap(framed.frames.matrix[1, 1], local["c"].matrix), 1e-11)

    def test_a_clip_square_on_its_first_frame(self):
        # frame 0 splits to no shear, the later frames to one: each keeps
        # its own
        rig    = HierarchyData([TransformData("a"), TransformData("b"), TransformData("c")])
        rng    = np.random.default_rng(23)
        target = np.zeros((3, 3, 4, 4))
        for frame, row in np.ndindex(3, 3):
            target[frame, row] = TransformData(
                "t",
                scale     = rng.uniform(0.5, 2.0, 3),
                rotate    = rng.uniform(-150.0, 150.0, 3),
                translate = rng.uniform(-3.0, 3.0, 3),
                shear     = rng.uniform(-1.0, 1.0, 3) if frame else None,
            ).matrix

        for write in ("world_matrix", "matrix"):
            with self.subTest(write):
                clip = ClipData(rig.copy(), frames=3)
                want = self._pose_by_pose(clip, write, target)
                setattr(clip.frames, write, target)
                self.assertLessEqual(_gap(clip.frames.world_matrix, want), 1e-11)
                self.assertLessEqual(_gap(clip.frames.world_matrix, target), 1e-11)
                self.assertFalse(clip.frames.shear[0].any())
                self.assertTrue(clip.frames.shear[1:].any(axis=-1).all())

    def test_a_joint_keeps_its_shear_on_every_route(self):
        clip   = _sheared_clip(_rig())
        joints = np.array([node.node_type == "joint" for node in clip.list])
        before = np.array(clip.frames.shear)

        clip.frames.world_matrix = self._worlds(5)
        self.assertTrue(np.array_equal(clip.frames.shear[:, joints], before[:, joints]))
        self.assertTrue((clip.frames.shear[:, ~joints] != before[:, ~joints]).all())
        self.assertTrue(clip.frames.shear[:, ~joints].any(axis=-1).all())

    @staticmethod
    def _wide(wide: int) -> HierarchyData:
        """one root, ``wide`` children, ``wide`` grandchildren, transforms all,
        sheared and unevenly scaled"""
        rng   = np.random.default_rng(7)
        uuids = [generate_uuid() for _ in range(1 + 2 * wide)]
        nodes = [TransformData("root", uuid=uuids[0], scale=(1.0, 2.0, 3.0), rotate=(10.0, 20.0, 30.0))]
        for level, up in ((1, 0), (1 + wide, 1)):
            for i in range(wide):
                nodes.append(
                    TransformData(
                        f"{level}_{i}",
                        uuid        = uuids[level + i],
                        parent_uuid = uuids[up + (i if up else 0)],
                        scale       = rng.uniform(0.5, 2.0, 3),
                        rotate      = rng.uniform(-90.0, 90.0, 3),
                        translate   = rng.uniform(-3.0, 3.0, 3),
                        shear       = rng.uniform(-0.5, 0.5, 3),
                    )
                )
        return HierarchyData(nodes)

    def _counted(self, edit) -> dict:
        names  = ("_scale_shear_and_rotation", "_scales_shears_and_rotations", "_gram_schmidt")
        counts = dict.fromkeys(names, 0)
        real   = {name: getattr(H, name) for name in names}

        def counting(name):
            def call(*args, **kwargs):
                counts[name] += 1
                return real[name](*args, **kwargs)

            return call

        with mock.patch.multiple(H, **{name: counting(name) for name in names}):
            edit()
        return counts

    def test_no_loop_over_nodes_or_frames(self):
        # a rig of 21 nodes over 3 frames and one of 201 over 30, one depth:
        # as many splits on both, and none of the single method's
        def edits(wide, frames):
            rig  = self._wide(wide)
            clip = ClipData(rig.copy(), frames=frames)
            pick = list(rig.copy().list)

            def worlds():
                target                     = clip.copy()
                target.frames.world_matrix = target.frames.world_matrix + 1e-3

            def matrices():
                target               = clip.copy()
                target.frames.matrix = target.frames.matrix + 1e-3

            def to_the_world():
                TransformList(list(rig.copy().list)[1:]).set_parent(None)

            def under_a_node():
                held = rig.copy()
                TransformList(list(held.list)[2:]).set_parent(held.list[1].name)

            def swapped():
                TransformList(pick).swapaxes(0, 1)

            return dict(
                worlds=worlds, matrices=matrices, to_the_world=to_the_world, under_a_node=under_a_node, swapped=swapped
            )

        small, large = edits(10, 3), edits(100, 30)
        for name in small:
            with self.subTest(name):
                a, b = self._counted(small[name]), self._counted(large[name])
                self.assertEqual(a["_scale_shear_and_rotation"], 0)
                self.assertEqual(b["_scale_shear_and_rotation"], 0)
                self.assertGreater(a["_scales_shears_and_rotations"], 0)
                self.assertEqual(a, b)


class TestShearSetParent(unittest.TestCase):
    """world space set_parent: Maya's parent on a node that is no joint, a
    joint's shear left alone, many nodes against one at a time"""

    def _parented(self, rng, parent_shear):
        uuids = [generate_uuid() for _ in range(3)]
        rig = HierarchyData(
            [
                TransformData(
                    "p",
                    uuid      = uuids[0],
                    scale     = rng.uniform(0.4, 2.2, 3),
                    rotate    = rng.uniform(-170.0, 170.0, 3),
                    translate = rng.uniform(-3.0, 3.0, 3),
                    shear     = rng.uniform(-1.0, 1.0, 3) if parent_shear else None,
                ),
                TransformData(
                    "c",
                    uuid         = uuids[1],
                    scale        = rng.uniform(0.4, 2.2, 3),
                    rotate       = rng.uniform(-170.0, 170.0, 3),
                    translate    = rng.uniform(-3.0, 3.0, 3),
                    rotate_axis  = rng.uniform(-30.0, 30.0, 3),
                    rotate_order = int(rng.integers(0, 6)),
                ),
                TransformData(
                    "d",
                    uuid        = uuids[2],
                    parent_uuid = uuids[1],
                    scale       = rng.uniform(0.4, 2.2, 3),
                    rotate      = rng.uniform(-170.0, 170.0, 3),
                    translate   = rng.uniform(-3.0, 3.0, 3),
                ),
            ]
        )
        return rig

    def test_every_route_keeps_the_worlds_of_transforms(self):
        rng = np.random.default_rng(61)
        for trial in range(120):
            sheared = bool(trial % 2)
            rig     = self._parented(rng, sheared)
            want    = {node.name: node.world_matrix for node in rig.list}

            single  = rig.copy()
            single["c"].set_parent("p")

            listed = rig.copy()
            TransformList([listed["c"], listed["d"]]).set_parent("p")

            clip = ClipData(rig.copy(), frames=2)
            clip["c"].set_parent("p")

            root = rig.copy()
            root["c"].set_parent("p")
            root["c"].set_parent(None)
            with self.subTest(trial=trial):
                for name, world in want.items():
                    self.assertLessEqual(_gap(single[name].world_matrix, world), 1e-11)
                    self.assertLessEqual(_gap(listed[name].world_matrix, world), 1e-11)
                    self.assertLessEqual(_gap(root[name].world_matrix, world), 1e-11)
                    self.assertLessEqual(_gap(clip.frames[[name]].world_matrix[0, 0], world), 1e-11)

    def test_a_clip_square_on_its_first_frame(self):
        # c's world is square where c does not turn against its uneven
        # parent, frame 0, and shears where it does
        uuids = [generate_uuid() for _ in range(2)]
        rig = HierarchyData(
            [
                TransformData("p", uuid=uuids[0], scale=(1.0, 2.0, 3.0), rotate=(10.0, 20.0, 30.0)),
                TransformData("c", uuid=uuids[1], parent_uuid=uuids[0], translate=(1.0, 2.0, 3.0)),
            ]
        )
        clip               = ClipData(rig.copy(), frames=3)
        rotate             = np.array(clip.frames.rotate)
        rotate[1:, 1]      = ((40.0, 10.0, -20.0), (-30.0, 50.0, 15.0))
        clip.frames.rotate = rotate

        poses = [pose_at(clip.copy(), frame) for frame in range(3)]
        for pose in poses:
            pose["c"].set_parent(None)
        clip["c"].set_parent(None)

        self.assertLessEqual(_gap(clip.frames.world_matrix, np.array([p.world_matrix for p in poses])), 1e-11)
        self.assertLessEqual(_gap(clip.frames.shear, np.array([p.shear for p in poses])), 1e-11)
        self.assertFalse(clip.frames.shear[0].any())
        self.assertTrue(clip.frames.shear[1:, 1].any(axis=-1).all())

    @staticmethod
    def _bush(rng, exponent, count, joints=True, shear=True, mirror=False) -> HierarchyData:
        uuids = [generate_uuid() for _ in range(count)]
        nodes = []
        for i in range(count):
            joint = joints and rng.random() < 0.5
            held  = rng.uniform(-1.0, 1.0, 3)
            scale = 10.0 ** rng.uniform(-exponent, exponent, 3)
            if mirror and rng.random() < 0.3:
                scale[int(rng.integers(0, 3))] *= -1.0
            nodes.append(
                TransformData(
                    f"n{i}",
                    uuid=uuids[i],
                    parent_uuid=uuids[int(rng.integers(max(0, i - 3), i))] if i else None,
                    node_type="joint" if joint else "transform",
                    scale=scale,
                    rotate=rng.uniform(-150.0, 150.0, 3),
                    translate=rng.uniform(-3.0, 3.0, 3),
                    rotate_order=int(rng.integers(0, 6)),
                    joint_orient=rng.uniform(-40.0, 40.0, 3) if joint else None,
                    segment_scale_compensate=bool(joint and i and rng.random() < 0.4),
                    shear=held if shear and rng.random() < 0.6 else None,
                )
            )
        return HierarchyData(nodes)

    @staticmethod
    def _one_at_a_time(rig, pick, parent) -> HierarchyData:
        out  = rig.copy()
        held = [out.list[i] for i in pick]
        for node in held:
            node.set_parent(None)
        if parent is not None:
            for node in held:
                if node.name != parent:
                    node.set_parent(parent)
        return out

    @staticmethod
    def _taken(rig, names) -> bool:
        """whether one of the picked ``names`` that is no joint holds shear"""
        return any(rig[name].node_type != "joint" and rig[name].shear.any() for name in names)

    def _against_one_at_a_time(self, rig, pick, parent) -> tuple:
        """the largest world entry of one at a time, the gap to the batch, the
        condition number of the worst world, and whether the batch put shear
        on a picked node that is no joint"""
        want  = self._one_at_a_time(rig, pick, parent)
        batch = rig.copy()
        names = [batch.list[i].name for i in pick]
        TransformList([batch.list[i] for i in pick]).set_parent(parent)
        worlds = {node.name: node.world_matrix for node in want.list}
        gap    = max(float(np.abs(node.world_matrix - worlds[node.name]).max()) for node in batch.list)
        size   = max(1.0, max(float(np.abs(world).max()) for world in worlds.values()))
        cond   = max(float(np.linalg.cond(world[:3, :3])) for world in worlds.values())
        return size, gap, cond, self._taken(batch, names)

    def test_transforms_do_not_cascade_and_a_joint_dropping_a_shear_does(self):
        rng = np.random.default_rng(62)
        real, answers = H._cascades, []

        def spy(*args):
            answers.append(real(*args))
            return answers[-1]

        with mock.patch.object(H, "_cascades", spy):
            for trial in range(40):
                rig    = self._bush(rng, 0.2, 8, joints=False)
                pick   = sorted(rng.choice(np.arange(1, 8), size=4, replace=False).tolist())
                parent = None if trial % 2 else "n0"
                self.assertLessEqual(self._against_one_at_a_time(rig, pick, parent)[1], 1e-9)
        self.assertEqual(len(answers), 40)
        self.assertFalse(any(answers))

        # a joint a turned uneven parent shears cannot hold its world
        uuids = [generate_uuid() for _ in range(3)]
        rig = HierarchyData(
            [
                TransformData("p", uuid=uuids[0], scale=(1.0, 2.0, 3.0), rotate=(30.0, 40.0, 50.0)),
                TransformData("a", uuid=uuids[1], parent_uuid=uuids[0], node_type="joint", rotate=(10.0, 0.0, 0.0)),
                TransformData("b", uuid=uuids[2], parent_uuid=uuids[1], node_type="joint", translate=(1.0, 2.0, 3.0)),
            ]
        )
        del answers[:]
        batch = rig.copy()
        with mock.patch.object(H, "_cascades", spy):
            TransformList([batch["a"], batch["b"]]).set_parent(None)
        self.assertEqual(answers, [True])

    def test_a_joint_keeps_its_shear(self):
        rng   = np.random.default_rng(63)
        taken = False
        for _ in range(30):
            rig    = self._bush(rng, 0.3, 8)
            before = {n.name: n.shear for n in rig.list if n.node_type == "joint"}
            batch  = rig.copy()
            names  = [node.name for node in batch.list[2:6]]
            TransformList(list(batch.list)[2:6]).set_parent("n1")
            for name, shear in before.items():
                self.assertEqual(batch[name].shear.tolist(), shear.tolist())
            taken = taken or self._taken(batch, names)
        self.assertTrue(taken)

    def test_within_1e_9_up_to_condition_1e8(self):
        # scales 10^(+-1.5), rigs whose worst world has a condition number up
        # to 1e8: within 1e-9 of the largest world entry
        for label, shear in (("shear held", True), ("no shear held", False)):
            rng   = np.random.default_rng(64)
            seen  = 0
            taken = False
            for trial in range(48):
                count  = int(rng.integers(6, 12))
                rig    = self._bush(rng, 1.5, count, shear=shear)
                pick   = sorted(rng.choice(np.arange(1, count), size=int(rng.integers(2, 6)), replace=False).tolist())
                parent = None if trial % 2 else "n0"
                size, gap, cond, took = self._against_one_at_a_time(rig, pick, parent)
                taken = taken or took
                if cond <= 1e8:
                    seen += 1
                    with self.subTest(label, trial=trial):
                        self.assertLessEqual(gap / size, 1e-9)
            self.assertGreater(seen, 12)
            self.assertTrue(taken)

    def test_wide_scales_stay_finite_with_no_scale_of_zero(self):
        # past the promise no number holds, but nothing breaks: 10^(+-2)
        rng   = np.random.default_rng(69)
        taken = False
        for trial in range(48):
            count  = int(rng.integers(6, 12))
            rig    = self._bush(rng, 2.0, count, shear=bool(trial % 3))
            pick   = sorted(rng.choice(np.arange(1, count), size=int(rng.integers(2, 6)), replace=False).tolist())
            parent = None if trial % 2 else "n0"
            batch  = rig.copy()
            names  = [batch.list[i].name for i in pick]
            TransformList([batch.list[i] for i in pick]).set_parent(parent)
            taken = taken or self._taken(batch, names)
            with self.subTest(trial=trial):
                self.assertTrue(np.isfinite(batch.world_matrix).all())
                self.assertFalse((batch.scale == 0.0).any())
        self.assertTrue(taken)

    def test_small_rigs_within_1e_9_absolute(self):
        # a seeded row, not the promise: no absolute number is promised
        rng   = np.random.default_rng(66)
        taken = False
        for trial in range(48):
            count  = int(rng.integers(6, 12))
            rig    = self._bush(rng, 0.5, count, shear=bool(trial % 3))
            pick   = sorted(rng.choice(np.arange(1, count), size=int(rng.integers(2, 6)), replace=False).tolist())
            parent = None if trial % 2 else "n0"
            _, gap, _, took = self._against_one_at_a_time(rig, pick, parent)
            taken = taken or took
            with self.subTest(trial=trial):
                self.assertLessEqual(gap, 1e-9)
        self.assertTrue(taken)

    def test_three_rigs_near_gimbal(self):
        # two spellings of a sheared split land a node on two sides of the
        # euler split's snap: 2.58e-8, 4.17e-9, 2.16e-9 of the largest entry
        taken = False
        for rig_data in GIMBAL_RIGS:
            uuids = [generate_uuid() for _ in rig_data["nodes"]]
            nodes = [
                TransformData(
                    data["name"],
                    uuid=uuids[i],
                    parent_uuid=uuids[data["up"]] if i else None,
                    node_type="joint" if data["joint"] else "transform",
                    scale=data["scale"],
                    rotate=data["rotate"],
                    translate=data["translate"],
                    rotate_order=data["order"],
                    joint_orient=data["jo"],
                    rotate_axis=data["ra"],
                    segment_scale_compensate=data["ssc"],
                )
                for i, data in enumerate(rig_data["nodes"])
            ]
            size, gap, _, took = self._against_one_at_a_time(HierarchyData(nodes), rig_data["pick"], rig_data["parent"])
            taken = taken or took
            with self.subTest(seed=rig_data["seed"], rig=rig_data["rig"]):
                self.assertLessEqual(gap / size, 1e-12)
        self.assertTrue(taken)


class TestShearInTheCascadeCheck(unittest.TestCase):
    def test_branch_bound_holds_a_branch_of_large_shears(self):
        rng = np.random.default_rng(71)
        for trial in range(140):
            count = 6
            reach = 0.3 if trial % 2 else 6.0
            away  = 0.0 if trial % 4 == 1 else 3.0
            uuids = [generate_uuid() for _ in range(count)]
            # scale * (1 + |shear|): past a scale of 1 no sum of the two holds
            scale = (lambda: rng.uniform(0.5, 2.0, 3)) if trial < 100 else (lambda: 10.0 ** rng.uniform(0.5, 2.0, 3))
            rig = HierarchyData(
                [
                    TransformData(
                        f"n{i}",
                        uuid        = uuids[i],
                        parent_uuid = uuids[i - 1] if i else None,
                        scale       = scale(),
                        rotate      = rng.uniform(-150.0, 150.0, 3),
                        translate   = rng.uniform(-away, away, 3),
                        shear       = rng.uniform(-reach, reach, 3),
                    )
                    for i in range(count)
                ]
            )
            frames    = H._RigFrames(rig)
            parents   = frames.parents
            rows      = [1]
            worlds    = frames.world(rows)
            inside    = np.ones(count, dtype=bool)
            inside[0] = False
            walk      = list(range(1, count))
            exact     = H._weigh(H._branch_worlds(frames, rows, parents, worlds, inside, walk))
            bound     = H._branch_bound(frames, rows, parents, worlds, inside, walk)
            self.assertTrue((bound >= exact).all(), (bound, exact))

    def test_split_input_is_square(self):
        # the input of the euler split, on sheared targets, joints or not
        rng    = np.random.default_rng(3)
        count  = 4000
        scale  = rng.uniform(0.2, 3.0, (1, count, 3)) * rng.choice([1.0, 1.0, 1.0, -1.0], (1, count, 3))
        shear  = rng.uniform(-2.0, 2.0, (1, count, 3))
        rotate = rng.uniform(-170.0, 170.0, (1, count, 3))
        orders = rng.integers(0, 6, count)
        channels = {
            "_scale":        np.ones((1, count, 3)),
            "_rotate":       rotate,
            "_rotate_axis":  rng.uniform(-40.0, 40.0, (1, count, 3)),
            "_joint_orient": rng.uniform(-40.0, 40.0, (1, count, 3)),
            "_translate":    np.zeros((1, count, 3)),
            "_shear":        rng.uniform(-1.0, 1.0, (1, count, 3)),
        }
        target = np.zeros((1, count, 4, 4))
        target[..., :3, :3] = (
            H._scale_shears(scale, shear)[..., :3, :3] @ H._euler_matrices(rotate, orders)[..., :3, :3]
        )
        target[..., 3, 3] = 1.0
        inverse           = rng.uniform(0.3, 2.5, (1, count, 3))
        self.assertLessEqual(H._off_square(H._split_input(target, channels, orders, inverse)).max(), 1e-15)


class TestShearRigEdits(unittest.TestCase):
    """swapaxes re-expresses the shear a node holds: its world is P * W, its
    children keep theirs but for a compensated joint child, on one node, a
    list and a clip. A compensated child of a compensated transform child
    moves with that child's scale. A mirror stays where the swap puts it, on
    every frame and every route"""

    AXES  = ((0, 1, False), (0, 2, True), (1, 2, False), (1, 0, True))
    SIGNS = ((-1.0, 1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, 1.0))

    def _chain(self, rng, kinds, scales=None, shears=None, compensated=()) -> HierarchyData:
        uuids = [generate_uuid() for _ in kinds]
        nodes = []
        for i, kind in enumerate(kinds):
            nodes.append(
                TransformData(
                    "abcdef"[i],
                    uuid=uuids[i],
                    parent_uuid=uuids[i - 1] if i else None,
                    node_type=kind,
                    scale=rng.uniform(0.4, 2.2, 3) if scales is None else scales[i],
                    rotate=rng.uniform(-150.0, 150.0, 3),
                    translate=rng.uniform(-3.0, 3.0, 3),
                    rotate_order=int(rng.integers(0, 6)),
                    rotate_axis=rng.uniform(-30.0, 30.0, 3) if rng.random() < 0.5 else None,
                    joint_orient=rng.uniform(-40.0, 40.0, 3) if kind == "joint" else None,
                    segment_scale_compensate=i in compensated,
                    shear=rng.uniform(-1.0, 1.0, 3) if shears is None else shears[i],
                )
            )
        return HierarchyData(nodes)

    def test_a_node_is_p_w_and_its_children_keep_theirs(self):
        rng = np.random.default_rng(81)
        for trial in range(90):
            kinds  = ("transform", "joint", "transform") if trial % 2 else ("joint", "joint", "joint")
            rig    = self._chain(rng, kinds)
            before = rig.world_matrix.copy()
            axis0, axis1, no = self.AXES[trial % 4]
            P = H._axis_permutation(axis0, axis1, no)
            rig.list[1].swapaxes(axis0, axis1, no)
            with self.subTest(trial=trial):
                self.assertLessEqual(_gap(rig.list[1].world_matrix, P @ before[1]), 1e-12)
                self.assertLessEqual(_gap(rig.list[2].world_matrix, before[2]),     1e-12)
                self.assertLessEqual(_gap(rig.list[0].world_matrix, before[0]),     1e-12)

    def _each_frame_is_its_pose(self, clip, trial: int) -> None:
        poses = [pose_at(clip.copy(), frame) for frame in range(clip.frame_count)]
        clip.list[1].swapaxes(0, 1)
        for frame, pose in enumerate(poses):
            pose.list[1].swapaxes(0, 1)
            with self.subTest(trial=trial, frame=frame):
                self.assertLessEqual(_gap(clip.frames.matrix[frame], pose.matrix), 1e-12)
                self.assertLessEqual(_gap(clip.frames.world_matrix[frame], pose.world_matrix), 1e-12)
                self.assertLessEqual(_gap(clip.frames.shear[frame], pose.shear), 1e-12)
                self.assertLessEqual(_gap(clip.frames.scale[frame], pose.scale), 1e-12)
                self.assertLessEqual(_gap(clip.frames.rotate_matrix[frame], pose.rotate_matrix), 1e-12)

    def test_a_clip_is_one_node_at_a_time(self):
        rng = np.random.default_rng(82)
        for trial in range(60):
            kinds = ("transform", "joint", "joint")
            rig   = self._chain(rng, kinds, compensated=(2,) if trial % 2 else ())
            self._each_frame_is_its_pose(_sheared_clip(rig, frames=3, seed=trial), trial)

    def test_a_clip_square_on_its_first_frame(self):
        # the node holds shear from frame 1 on: frame 0 stays the permuted
        # scale to the bit, the later frames split, and a mirror on x or y
        # stays where the swap puts it on all of them
        rng = np.random.default_rng(89)
        for trial in range(30):
            kinds     = ("transform", "joint", "joint")
            scales    = rng.uniform(0.5, 2.0, (3, 3))
            scales[1] = scales[1] * self.SIGNS[trial % 3]
            rig = self._chain(
                rng, kinds, scales=scales, shears=[None, None, None], compensated=(2,) if trial % 2 else ()
            )
            clip  = _animated(rig, frames=3, seed=trial)
            block = np.zeros(clip.frames.shear.shape)

            block[1:, 1]      = rng.uniform(-1.0, 1.0, (2, 3))
            clip.frames.shear = block

            was = clip.frames.scale[:, 1][:, [1, 0, 2]]
            self._each_frame_is_its_pose(clip, trial)
            with self.subTest(trial=trial):
                self.assertEqual(clip.frames.scale[0, 1].tolist(), was[0].tolist())
                self.assertEqual(np.sign(clip.frames.scale[:, 1]).tolist(), np.sign(was).tolist())

    def test_a_list_holding_one_sheared_node(self):
        # the node holding none stays the permuted scale to the bit, its
        # mirror where the permutation puts it, as the sheared node's is
        rng = np.random.default_rng(90)
        P   = H._axis_permutation(0, 1, False)
        for trial in range(30):
            scale = rng.uniform(0.5, 2.0, 3) * self.SIGNS[trial % 3]
            rig = HierarchyData(
                [
                    TransformData(
                        "x", rotate=rng.uniform(-150.0, 150.0, 3), scale=rng.uniform(0.5, 2.0, 3), shear=SHEAR
                    ),
                    TransformData("y", rotate=rng.uniform(-150.0, 150.0, 3), scale=scale),
                ]
            )
            before = rig.world_matrix.copy()
            listed = rig.copy()
            TransformList(list(listed.list)).swapaxes(0, 1)
            alone = rig.copy()
            for node in alone.list:
                node.swapaxes(0, 1)
            with self.subTest(trial=trial):
                self.assertLessEqual(_gap(listed.scale, alone.scale), 1e-12)
                self.assertLessEqual(_gap(listed.world_matrix, P @ before), 1e-12)
                self.assertEqual(listed["y"].scale.tolist(), scale[[1, 0, 2]].tolist())

    def test_a_node_holding_some_of_the_entries(self):
        rng = np.random.default_rng(87)
        P   = H._axis_permutation(0, 1, False)
        for shear in PARTIAL_SHEARS:
            rig    = self._chain(rng, ("transform", "joint", "transform"), shears=[None, shear, None])
            before = rig.world_matrix.copy()
            for label, edit in (
                ("one node", lambda held: held.list[1].swapaxes(0, 1)),
                ("a list", lambda held: TransformList([held.list[1]]).swapaxes(0, 1)),
            ):
                with self.subTest(label, shear=shear):
                    held = rig.copy()
                    edit(held)
                    self.assertLessEqual(_gap(held.list[1].world_matrix, P @ before[1]), 1e-12)
                    self.assertLessEqual(_gap(held.list[2].world_matrix, before[2]), 1e-12)

    def test_a_compensated_joint_child_moves_and_a_transform_child_holds(self):
        # the split's scale is no permutation of the old one, so the child's
        # block asks a shear: a transform takes it, a joint does not
        rng   = np.random.default_rng(88)
        moved = 0.0
        for trial in range(40):
            parent = ("transform", "joint")[trial % 2]
            for kind, shear in (("transform", (0.4, -0.3, 0.6)), ("joint", (0.4, -0.3, 0.6)), ("joint", None)):
                rig    = self._chain(rng, ("joint", parent, kind), shears=[None, shear, None], compensated=(2,))
                before = rig.list[2].world_matrix
                rig.list[1].swapaxes(0, 1)
                gap = np.abs(rig.list[2].world_matrix - before).max()
                if kind == "joint" and shear is not None:
                    moved = max(moved, float(gap))
                    self.assertEqual(rig.list[2].shear.tolist(), [0.0, 0.0, 0.0])
                else:
                    with self.subTest(trial=trial, kind=kind, shear=shear):
                        self.assertLessEqual(gap, 1e-12)
        self.assertGreater(moved, 0.1)

    def test_a_compensated_child_of_a_compensated_transform_child_moves(self):
        # that child takes the shear the split's scale asks and holds its
        # world, but its own scale moves, and its compensated child divides
        # by that
        rng   = np.random.default_rng(91)
        moved = 0.0
        for trial in range(40):
            kind = ("transform", "joint")[trial % 2]
            for shear, compensated in ((SHEAR, (1, 2)), (None, (1, 2)), (SHEAR, (1,)), (SHEAR, (2,))):
                kinds  = ("transform", "transform", kind)
                rig    = self._chain(rng, kinds, shears=[shear, None, None], compensated=compensated)
                before = rig.world_matrix.copy()
                rig.list[0].swapaxes(0, 1)
                gap = np.abs(rig.list[2].world_matrix - before[2]).max()
                with self.subTest(trial=trial, shear=shear, compensated=compensated):
                    self.assertLessEqual(_gap(rig.list[1].world_matrix, before[1]), 1e-12)
                    if shear is None or len(compensated) == 1:
                        self.assertLessEqual(gap, 1e-11)
                    else:
                        moved = max(moved, float(gap))
        self.assertGreater(moved, 0.1)

    def test_a_list_and_a_clip_over_a_compensated_transform_child(self):
        # that child takes the shear the split's scale asks and keeps its
        # world, on the batched route as on the single method
        rng = np.random.default_rng(92)
        for trial in range(40):
            kind   = ("transform", "locator")[trial % 2]
            rig    = self._chain(rng, ("joint", "transform", kind), shears=[None, SHEAR, None], compensated=(2,))
            before = rig.world_matrix.copy()
            turn   = rig.copy()
            turn.list[1].swapaxes(0, 1)

            listed = rig.copy()
            TransformList([listed.list[1]]).swapaxes(0, 1)
            clip = ClipData(rig.copy(), frames=2)
            clip.list[1].swapaxes(0, 1)
            with self.subTest(trial=trial):
                self.assertLessEqual(_gap(listed.list[2].world_matrix, before[2]),         1e-12)
                self.assertLessEqual(_gap(clip.frames.world_matrix[0, 2], before[2]),      1e-12)
                self.assertLessEqual(_gap(listed.world_matrix, turn.world_matrix),         1e-12)
                self.assertLessEqual(_gap(clip.frames.world_matrix[0], turn.world_matrix), 1e-12)

    def test_a_compensated_child_divides_by_the_split_scale(self):
        # the split's scale is not the permuted one on a sheared node
        rng   = np.random.default_rng(83)
        apart = 0.0
        for trial in range(40):
            rig   = self._chain(rng, ("joint", "joint", "joint"), compensated=(2,))
            batch = rig.copy()
            turn  = rig.copy()
            old   = rig.list[1].scale
            TransformList([batch.list[1]]).swapaxes(0, 1)
            turn.list[1].swapaxes(0, 1)
            apart = max(apart, float(np.abs(turn.list[1].scale - old[[1, 0, 2]]).max()))
            with self.subTest(trial=trial):
                self.assertLessEqual(_gap(batch.world_matrix, turn.world_matrix), 1e-12)
        self.assertGreater(apart, 0.01)

    MIRRORS = (
        ("on x", (-1.3, 1.1, 0.9)),
        ("on y", (1.3, -1.1, 0.9)),
        ("on x and z", (-1.3, 1.1, -0.9)),
        ("on z alone", (1.3, 1.1, -0.9)),
    )

    def _sheared_compensated(self, scale, shear=(0.3, -0.2, 0.5)) -> HierarchyData:
        """joints a, b, c, d, b sheared and compensated"""
        uuids = [generate_uuid() for _ in range(4)]
        return HierarchyData(
            [
                TransformData("a", uuid=uuids[0], node_type="joint", scale=(1.2, 0.8, 1.5), rotate=(10.0, 20.0, 30.0)),
                TransformData(
                    "b",
                    uuid=uuids[1],
                    parent_uuid=uuids[0],
                    node_type="joint",
                    scale=scale,
                    rotate=(40.0, 10.0, -20.0),
                    translate=(1.0, 2.0, 3.0),
                    segment_scale_compensate=True,
                    shear=shear,
                ),
                TransformData(
                    "c",
                    uuid=uuids[2],
                    parent_uuid=uuids[1],
                    node_type="joint",
                    scale=(1.1, 0.7, 1.3),
                    rotate=(5.0, 15.0, 25.0),
                    translate=(0.0, 1.0, 2.0),
                    segment_scale_compensate=True,
                ),
                TransformData("d", uuid=uuids[3], parent_uuid=uuids[2], node_type="joint", translate=(1.0, 0.0, 1.0)),
            ]
        )

    def test_a_selection_holding_a_parent_and_its_child(self):
        # the single method parent first, against the list and the clip
        for label, scale in self.MIRRORS:
            for pick in ((0, 1), (0, 1, 2), (1, 2)):
                with self.subTest(label, pick=pick):
                    rig  = self._sheared_compensated(scale)
                    turn = rig.copy()
                    for i in pick:
                        turn.list[i].swapaxes(0, 1)

                    listed = rig.copy()
                    TransformList([listed.list[i] for i in pick]).swapaxes(0, 1)
                    clip = ClipData(rig.copy(), frames=2)
                    TransformList([clip.list[i] for i in pick]).swapaxes(0, 1)

                    self.assertLessEqual(_gap(listed.world_matrix, turn.world_matrix), 1e-12)
                    self.assertLessEqual(_gap(clip.frames.world_matrix[0], turn.world_matrix), 1e-12)

    def test_a_target_put_back_to_a_new_scale(self):
        # a holds shear and b compensates it, so b's put back moves its scale,
        # and c, compensating b, divides by that scale past b's own swap: one
        # pass hands c the scale b had
        rng = np.random.default_rng(95)
        for trial in range(40):
            shear = (SHEAR, *PARTIAL_SHEARS)[trial % 5]
            kinds = ("transform", ("transform", "joint")[trial % 2], ("joint", "transform")[trial // 2 % 2])
            rig   = self._chain(rng, kinds, shears=[shear, None, None], compensated=(1, 2))
            axes  = self.AXES[trial % 4]
            turn  = rig.copy()
            for i in (0, 1):
                turn.list[i].swapaxes(*axes)

            listed = rig.copy()
            TransformList([listed.list[i] for i in (0, 1)]).swapaxes(*axes)
            clip = ClipData(rig.copy(), frames=2)
            TransformList([clip.list[i] for i in (0, 1)]).swapaxes(*axes)
            with self.subTest(trial=trial, kinds=kinds, shear=shear, axes=axes):
                self.assertLessEqual(_gap(listed.world_matrix, turn.world_matrix), 1e-12)
                self.assertLessEqual(_gap(clip.frames.world_matrix[0], turn.world_matrix), 1e-12)

        # the read of shear among the targets sees every frame: the shear on
        # the middle frame of three alone
        rng = np.random.default_rng(96)
        for trial in range(40):
            kinds = ("transform", ("transform", "joint")[trial % 2], ("joint", "transform")[trial // 2 % 2])
            rig   = self._chain(rng, kinds, shears=[None, None, None], compensated=(1, 2))
            clip  = ClipData(rig, frames=3)
            block = np.zeros(clip.frames.shear.shape)

            block[1, 0]       = (SHEAR, *PARTIAL_SHEARS)[trial % 5]
            clip.frames.shear = block

            poses = [pose_at(clip.copy(), frame) for frame in range(3)]
            axes  = self.AXES[trial % 4]
            TransformList([clip.list[0], clip.list[1]]).swapaxes(*axes)
            for frame, pose in enumerate(poses):
                pose.list[0].swapaxes(*axes)
                pose.list[1].swapaxes(*axes)
                with self.subTest(trial=trial, frame=frame, kinds=kinds, axes=axes):
                    self.assertLessEqual(_gap(clip.frames.world_matrix[frame], pose.world_matrix), 1e-12)

        # and every target: a holds none, b does, c is put back to a new
        # scale and d divides by it
        rng = np.random.default_rng(97)
        for trial in range(40):
            kinds = ("transform", "transform", ("transform", "joint")[trial % 2], ("joint", "transform")[trial // 2 % 2])
            rig   = self._chain(rng, kinds, shears=[None, SHEAR, None, None], compensated=(2, 3))
            axes  = self.AXES[trial % 4]
            turn  = rig.copy()
            for i in (0, 1, 2):
                turn.list[i].swapaxes(*axes)

            listed = rig.copy()
            TransformList([listed.list[i] for i in (0, 1, 2)]).swapaxes(*axes)
            with self.subTest(trial=trial, kinds=kinds, axes=axes):
                self.assertLessEqual(_gap(listed.world_matrix, turn.world_matrix), 1e-12)

        # and every entry: above, the trial walks b's kind and the axes
        # together. Here one entry and a pair of them by each axis pair by
        # each kind of b and c
        rng   = np.random.default_rng(98)
        kinds = ("transform", "joint")
        shears = (
            (0.5, 0.0, 0.0),
            (0.0, 0.5, 0.0),
            (0.0, 0.0, 0.75),
            (0.5, -0.25, 0.0),
            (0.5, 0.0, 0.75),
            (0.0, 0.5, 0.75),
        )
        for shear, axes, b, c in itertools.product(shears, self.AXES, kinds, kinds):
            rig  = self._chain(rng, ("transform", b, c), shears=[shear, None, None], compensated=(1, 2))
            turn = rig.copy()
            for i in (0, 1):
                turn.list[i].swapaxes(*axes)

            listed = rig.copy()
            TransformList([listed.list[i] for i in (0, 1)]).swapaxes(*axes)
            with self.subTest(shear=shear, axes=axes, b=b, c=c):
                self.assertLessEqual(_gap(listed.world_matrix, turn.world_matrix), 1e-12)

    # no shear held. a's scales 1e-4 apart leave b, a mirror on x, a shear of
    # about 1e-12 when it is put back, c is compensated under it
    NOISE = (
        ("a", None, "transform", (1.6733772128848268, 1.507230064548501, 8.560713552066452e-05),
         (-88.73778327810174, 14.809483814241872, 142.0208944468439),
         (-0.33441166063341, 1.5617045166975911, 0.4516846026276875), None, False),
        ("b", "a", "transform", (-1.0199354984274633, 0.7281649845055638, 0.9671442840942918),
         (4.040600793174349, 53.43449038793946, 151.88821374917603),
         (2.4918221441718984, 1.3214079780363228, -2.263473139849636), None, False),
        ("c", "b", "joint", (1.8640121514680073, 1.5699230865390317, 1.0299543804653775),
         (-94.73526314246561, 140.7866230652216, 107.3335977271343),
         (-0.9581197219040618, 2.7769085332574104, 1.1027224091271766),
         (-37.088099169206544, -9.208931727279523, -28.633072094988606), True),
        ("d", "c", "transform", (1.0, 1.0, 1.0), (0.0, 0.0, 0.0),
         (-0.8747735843611917, 2.8063139043624226, 0.5735225727882667), None, False),
    )

    def _noise_rows(self, seed: int = 91, count: int = 60):
        """the rig above, and ``count`` of its kind: which of them reach the
        1e-12 test is a matter of last bits, so these hold on any machine"""
        rng = np.random.default_rng(seed)
        yield self.NOISE
        for _ in range(count):
            a = rng.uniform(0.5, 2.0, 3)
            b = rng.uniform(0.5, 2.0, 3)

            a[int(rng.integers(3))] *= 1e-4
            b[int(rng.integers(2))] *= -1.0

            def rotate():
                return rng.uniform(-170.0, 170.0, 3)

            def translate():
                return rng.uniform(-3.0, 3.0, 3)

            yield (
                ("a", None, "transform", a, rotate(), translate(), None, False),
                ("b", "a", "transform", b, rotate(), translate(), None, False),
                ("c", "b", "joint", rng.uniform(0.5, 2.0, 3), rotate(), translate(), rng.uniform(-40.0, 40.0, 3), True),
                ("d", "c", "transform", (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), translate(), None, False),
            )

    def _noise_rig(self, rows) -> HierarchyData:
        uuids = {row[0]: generate_uuid() for row in rows}
        return HierarchyData(
            [
                TransformData(
                    name,
                    uuid=uuids[name],
                    parent_uuid=uuids.get(up),
                    node_type=kind,
                    scale=scale,
                    rotate=rotate,
                    translate=translate,
                    joint_orient=orient,
                    segment_scale_compensate=compensated,
                )
                for name, up, kind, scale, rotate, translate, orient, compensated in rows
            ]
        )

    def _noise_swap(self, rig) -> None:
        TransformList([rig["a"], rig["b"]]).swapaxes(0, 2, True)

    def test_a_put_back_that_leaves_a_target_a_noise_shear(self):
        # the noise sends b's swap through the split and its signs go back, so
        # b ends as one node at a time leaves it, and c, which divided up front
        # by b's permuted scale, takes the channels one node at a time gives
        # it, to the noise. That scale is not the one b ends on: b's put back
        # moved its mirror
        noisy = 0
        for trial, rows in enumerate(self._noise_rows()):
            rig  = self._noise_rig(rows)
            turn = rig.copy()
            turn["a"].swapaxes(0, 2, True)
            turn["b"].swapaxes(0, 2, True)

            listed = rig.copy()
            self._noise_swap(listed)
            clip = ClipData(rig.copy(), frames=2)
            self._noise_swap(clip)

            noisy += bool(listed["b"].shear.any())
            with self.subTest(trial=trial):
                self.assertLessEqual(_gap(listed.world_matrix, turn.world_matrix), 1e-9)
                self.assertLessEqual(_gap(clip.frames.world_matrix[0], turn.world_matrix), 1e-9)
        self.assertGreater(noisy, 0)

    def test_channels_of_a_put_back_that_leaves_a_target_a_noise_shear(self):
        # the list and the clip take the noise on other rigs than one node at
        # a time does: scale and rotate read the same all the same
        noisy = 0
        for trial, rows in enumerate(self._noise_rows()):
            rig  = self._noise_rig(rows)
            turn = rig.copy()
            turn["a"].swapaxes(0, 2, True)
            turn["b"].swapaxes(0, 2, True)

            listed = rig.copy()
            self._noise_swap(listed)
            clip = ClipData(rig.copy(), frames=2)
            self._noise_swap(clip)

            noisy += bool(listed["b"].shear.any())
            with self.subTest(trial=trial):
                self.assertLessEqual(_gap(listed.scale, turn.scale), 1e-8)
                self.assertLessEqual(_gap(listed.rotate_matrix, turn.rotate_matrix), 1e-8)
                self.assertLessEqual(_gap(clip.frames.scale[0], turn.scale), 1e-8)
                self.assertLessEqual(_gap(clip.frames.rotate_matrix[0], turn.rotate_matrix), 1e-8)
        self.assertGreater(noisy, 0)

    def _turning(self, rng, scales, frames: int = 30) -> ClipData:
        """transforms a over b, ``frames`` frames, each turning up to 1.5
        degrees a frame"""
        uuids = [generate_uuid(), generate_uuid()]
        rig = HierarchyData(
            [
                TransformData(
                    name,
                    uuid        = uuids[i],
                    parent_uuid = uuids[0] if i else None,
                    scale       = scales[i],
                    rotate      = rng.uniform(-170.0, 170.0, 3),
                    translate   = rng.uniform(-3.0, 3.0, 3),
                )
                for i, name in enumerate("ab")
            ]
        )
        clip               = ClipData(rig, frames=frames)
        drift              = rng.uniform(-1.5, 1.5, (2, 3))
        clip.frames.rotate = clip.frames.rotate + np.arange(frames)[:, None, None] * drift
        return clip

    def test_a_noise_shear_moves_no_mirror(self):
        # no shear held. a's scales 1e-4 apart leave b a noise shear on some
        # frames and not on others: b's mirror on z goes to y on all of them
        rng   = np.random.default_rng(93)
        noisy = 0
        for trial in range(30):
            first = rng.uniform(0.5, 2.0, 3)
            first[1] *= 1e-4
            second = rng.uniform(0.5, 2.0, 3)
            second[2] *= -1.0
            clip = self._turning(rng, (first, second))
            for how in ("a list", "one then the other", "b in a later call"):
                held = clip.copy()
                if how == "a list":
                    TransformList([held["a"], held["b"]]).swapaxes(1, 2)
                else:
                    held["a"].swapaxes(1, 2)
                    if how == "b in a later call":
                        held = held.copy()
                    held["b"].swapaxes(1, 2)
                noisy += bool(held.frames.shear[:, 1].any())
                scale = held.frames.scale[:, 1]
                with self.subTest(trial=trial, how=how):
                    self.assertEqual(np.sign(scale).tolist(), [[1.0, -1.0, 1.0]] * len(scale))
                    self.assertLessEqual(np.abs(np.diff(scale, axis=0)).max(), 1e-8)
        self.assertGreater(noisy, 0)

    def test_a_shear_keyed_on_some_frames_keeps_the_mirror(self):
        # no small scale: a real shear on frames 10 to 19 alone, the plain
        # frames around them permuted, the sheared ones split
        rng = np.random.default_rng(94)
        P   = H._axis_permutation(1, 2, False)
        for trial in range(30):
            scale = rng.uniform(0.5, 2.0, 3)
            scale[trial % 3] *= -1.0
            clip              = self._turning(rng, (scale, rng.uniform(0.5, 2.0, 3)))
            block             = np.zeros(clip.frames.shear.shape)
            block[10:20, 0]   = rng.uniform(-0.5, 0.5, 3)
            clip.frames.shear = block
            before            = clip.frames.world_matrix
            for how in ("one node", "a list"):
                held = clip.copy()
                if how == "one node":
                    held["a"].swapaxes(1, 2)
                else:
                    TransformList([held["a"]]).swapaxes(1, 2)
                after = held.frames.world_matrix
                with self.subTest(trial=trial, how=how):
                    self.assertEqual(
                        np.sign(held.frames.scale[:, 0]).tolist(), [np.sign(scale[[0, 2, 1]]).tolist()] * len(after)
                    )
                    self.assertLessEqual(_gap(after[:, 0], P @ before[:, 0]), 1e-12)
                    self.assertLessEqual(_gap(after[:, 1], before[:, 1]), 1e-12)

    def test_a_noise_shear_on_one_frame_of_a_clip(self):
        # one frame plain, the other the noise, either way round: each is its
        # own pose one at a time, on its channels too
        noisy = 0
        for trial, rows in enumerate(self._noise_rows()):
            rig   = self._noise_rig(rows)
            plain = rig.copy()
            scale = plain["a"].scale

            scale[np.abs(scale).argmin()] = 0.9
            plain["a"].scale              = scale

            for noise, poses in ((1, (plain, rig)), (0, (rig, plain))):
                clip              = ClipData(rig.copy(), frames=2)
                clip.frames.scale = np.stack([pose.scale for pose in poses])
                self._noise_swap(clip)
                noisy += bool(clip.frames.shear[noise, 1].any())
                for f, pose in enumerate(poses):
                    turn = pose.copy()
                    turn["a"].swapaxes(0, 2, True)
                    turn["b"].swapaxes(0, 2, True)
                    with self.subTest(trial=trial, noise=noise, frame=f):
                        self.assertLessEqual(_gap(clip.frames.world_matrix[f], turn.world_matrix), 1e-9)
                        self.assertLessEqual(_gap(clip.frames.scale[f], turn.scale), 1e-8)
                        self.assertLessEqual(_gap(clip.frames.rotate_matrix[f], turn.rotate_matrix), 1e-8)
        self.assertGreater(noisy, 0)

    def test_a_selection_with_a_compensated_node_between_its_members(self):
        # a's child b is left out, c hangs under it: a child's local against
        # its parent does not depend on the parent's own move
        uuids = [generate_uuid() for _ in range(3)]
        rig = HierarchyData(
            [
                TransformData(
                    "a",
                    uuid      = uuids[0],
                    node_type = "joint",
                    scale     = (1.2, 0.8, 1.5),
                    rotate    = (10.0, 20.0, 30.0),
                    shear     = (0.5, -0.3, 0.4),
                ),
                TransformData(
                    "b",
                    uuid=uuids[1],
                    parent_uuid=uuids[0],
                    node_type="joint",
                    scale=(1.3, 1.1, 0.9),
                    rotate=(40.0, 10.0, -20.0),
                    translate=(1.0, 2.0, 3.0),
                    segment_scale_compensate=True,
                ),
                TransformData(
                    "c",
                    uuid=uuids[2],
                    parent_uuid=uuids[1],
                    node_type="joint",
                    scale=(1.1, 0.7, 1.3),
                    rotate=(5.0, 15.0, 25.0),
                    translate=(0.0, 1.0, 2.0),
                    segment_scale_compensate=True,
                ),
            ]
        )
        turn = rig.copy()
        for i in (0, 2):
            turn.list[i].swapaxes(0, 1)
        listed = rig.copy()
        TransformList([listed.list[0], listed.list[2]]).swapaxes(0, 1)
        clip = ClipData(rig.copy(), frames=2)
        TransformList([clip.list[0], clip.list[2]]).swapaxes(0, 1)
        self.assertLessEqual(_gap(listed.world_matrix, turn.world_matrix), 1e-12)
        self.assertLessEqual(_gap(clip.frames.world_matrix[0], turn.world_matrix), 1e-12)

    def test_no_shear_is_the_permuted_channels_to_the_bit(self):
        rng = np.random.default_rng(84)
        for _ in range(20):
            shears = [None, None, None]
            rig    = self._chain(rng, ("joint", "joint", "joint"), shears=shears)
            before = rig.list[1].scale
            rig.list[1].swapaxes(0, 2)
            self.assertEqual(rig.list[1].scale.tolist(), before[[2, 1, 0]].tolist())
            self.assertEqual(rig.list[1].shear.tolist(), [0.0, 0.0, 0.0])

            listed = rig.copy()
            before = listed.list[1].scale
            TransformList([listed.list[1]]).swapaxes(0, 2)
            self.assertEqual(listed.list[1].scale.tolist(), before[[2, 1, 0]].tolist())

        # a node and its parent both swapped, no shear held: one call, whose
        # numbers stay 3.0.3's. A generation at a time is for shear alone
        for axes in self.AXES:
            rig = self._chain(rng, ("joint", "joint", "joint"), shears=[None, None, None], compensated=(1, 2))
            with mock.patch.object(H, "_solve_local", wraps=H._solve_local) as solve:
                TransformList([rig.list[0], rig.list[1]]).swapaxes(*axes)
            with self.subTest(axes=axes):
                self.assertEqual(solve.call_count, 1)

    def test_the_rig_edits_that_solve_no_shear_leave_it_alone(self):
        rng = np.random.default_rng(86)
        rig = self._chain(rng, ("joint", "joint", "joint"))
        for label, edit in (
            ("set_rotate_order", lambda r: r.list[1].set_rotate_order(4)),
            ("set_joint_orient_to_rotate", lambda r: r.list[1].set_joint_orient_to_rotate()),
            ("set_rotate_to_joint_orient", lambda r: r.list[1].set_rotate_to_joint_orient()),
            ("set_node_type", lambda r: r.list[1].set_node_type("transform")),
        ):
            with self.subTest(label):
                held   = rig.copy()
                before = held.shear
                edit(held)
                self.assertEqual(held.shear.tolist(), before.tolist())


class TestShearOrientation(unittest.TestCase):
    """match_rotate: Maya's split of the node's world lands on Maya's split of
    the target's, and the orientation reads are orthonormal"""

    def _rig(self, rng, kind="transform", parent_scale=None, turned=True, target_scale=None, sheared=False, own=None):
        held = dict(shear=rng.uniform(-0.7, 0.7, 3)) if sheared else {}
        ups  = [generate_uuid() for _ in range(3)]
        return HierarchyData(
            [
                TransformData("g", uuid=ups[0], scale=rng.uniform(0.5, 2.0, 3), rotate=rng.uniform(-170.0, 170.0, 3)),
                TransformData(
                    "p",
                    uuid        = ups[1],
                    parent_uuid = ups[0] if turned else None,
                    scale       = rng.uniform(0.5, 2.0, 3) if parent_scale is None else parent_scale,
                    rotate      = rng.uniform(-170.0, 170.0, 3) if turned else None,
                    **held,
                ),
                TransformData(
                    "k",
                    parent_uuid  = ups[1],
                    node_type    = kind,
                    scale        = rng.uniform(0.5, 2.0, 3) if own is None else own,
                    rotate       = rng.uniform(-170.0, 170.0, 3),
                    translate    = (1.0, 2.0, 3.0),
                    rotate_order = int(rng.integers(0, 6)),
                    rotate_axis  = rng.uniform(-30.0, 30.0, 3),
                    joint_orient = rng.uniform(-40.0, 40.0, 3) if kind == "joint" else None,
                    **held,
                ),
                TransformData("t0", uuid=ups[2], scale=rng.uniform(0.5, 2.0, 3), rotate=rng.uniform(-170.0, 170.0, 3)),
                TransformData(
                    "t",
                    parent_uuid = ups[2],
                    scale       = rng.uniform(0.5, 2.0, 3) if target_scale is None else target_scale,
                    rotate      = rng.uniform(-170.0, 170.0, 3),
                    **held,
                ),
            ]
        )

    def _landed(self, rig) -> float:
        got = _maya_orientation(rig["k"].world_matrix[:3, :3])
        return float(np.abs(got - _maya_orientation(rig["t"].world_matrix[:3, :3])).max())

    def test_it_lands_on_the_target_on_sheared_and_uneven_worlds(self):
        rng = np.random.default_rng(91)
        for trial in range(120):
            kind = ("transform", "joint")[trial % 2]
            rig  = self._rig(rng, kind, sheared=bool(trial % 3))
            rig["k"].match_rotate(rig["t"])
            with self.subTest(trial=trial):
                self.assertLessEqual(self._landed(rig), 1e-12)

    def test_it_lands_under_an_aligned_uneven_parent(self):
        # a gate on square rows alone keeps these on the transpose: 0.82, 1.83
        rng = np.random.default_rng(92)
        for trial in range(120):
            kind = ("transform", "joint")[trial % 2]
            rig  = self._rig(rng, kind, parent_scale=rng.uniform(0.4, 2.2, 3), turned=False)
            rig["k"].match_rotate(rig["t"])
            with self.subTest(trial=trial):
                self.assertLessEqual(self._landed(rig), 1e-12)

    def test_it_lands_under_a_mirror_above_an_uneven_parent_and_on_a_mirrored_target(self):
        rng = np.random.default_rng(93)
        for trial in range(180):
            parent = rng.uniform(0.5, 2.0, 3)
            parent[trial % 3] *= -1.0
            target = rng.uniform(0.5, 2.0, 3)
            target[(trial // 3) % 3] *= -1.0 if trial % 2 else 1.0
            rig = self._rig(rng, parent_scale=parent, target_scale=target, sheared=bool(trial % 4))
            rig["k"].match_rotate(rig["t"])
            with self.subTest(trial=trial):
                self.assertLessEqual(self._landed(rig), 1e-12)

    def test_it_lands_on_a_nodes_own_mirror_on_z(self):
        rng = np.random.default_rng(94)
        for trial in range(60):
            own = rng.uniform(0.5, 2.0, 3)
            own[2] *= -1.0
            rig = self._rig(rng, own=own)
            rig["k"].match_rotate(rig["t"])
            with self.subTest(trial=trial):
                self.assertLessEqual(self._landed(rig), 1e-12)

    def test_it_takes_an_euler_and_a_quaternion(self):
        rng = np.random.default_rng(95)
        for trial in range(60):
            rig    = self._rig(rng)
            order  = rig["k"].rotate_order
            rotate = rng.uniform(-170.0, 170.0, 3)
            want   = TransformData("r", rotate=rotate, rotate_order=order)
            rig["k"].match_rotate(rotate)
            got = _maya_orientation(rig["k"].world_matrix[:3, :3])
            with self.subTest(trial=trial):
                self.assertLessEqual(np.abs(got - want.rotate_matrix[:3, :3]).max(), 1e-12)

            rig["k"].rotate = (0.0, 0.0, 0.0)
            rig["k"].match_rotate(want.quaternion)
            got = _maya_orientation(rig["k"].world_matrix[:3, :3])
            with self.subTest(trial=trial, operand="quaternion"):
                self.assertLessEqual(np.abs(got - want.rotate_matrix[:3, :3]).max(), 1e-12)

    def test_it_writes_rotate_alone(self):
        rng      = np.random.default_rng(96)
        rig      = self._rig(rng, sheared=True)
        node     = rig["k"]
        channels = (node.scale, node.shear, node.translate, node.rotate_axis, node.joint_orient)
        node.match_rotate(rig["t"])
        for before, after in zip(channels, (node.scale, node.shear, node.translate, node.rotate_axis, node.joint_orient)):
            self.assertEqual(before.tolist(), after.tolist())

    def test_a_clip_is_one_pose_at_a_time(self):
        rng = np.random.default_rng(97)
        for trial in range(30):
            rig   = self._rig(rng, ("transform", "joint")[trial % 2], sheared=bool(trial % 3))
            clip  = ClipData(rig, frames=4)
            block = clip._blocks["_scale"]

            # the parent mirrored on some frames
            for frame, axis in ((1, 0), (2, 1), (3, 2)):
                block[frame, 1, axis] *= -1.0
            clip._blocks["_rotate"][:, 4] += rng.uniform(-40.0, 40.0, (4, 3))
            clip._bind(0)

            want = []
            for frame in range(4):
                pose = pose_at(clip.copy(), frame)
                pose["k"].match_rotate(pose["t"])
                want.append(pose["k"].world_matrix)

            target = clip.copy()
            clip.frames[["k"]].match_rotate(target.frames[["t"]])
            got = clip.frames[["k"]].world_matrix[:, 0]
            with self.subTest(trial=trial):
                self.assertLessEqual(_gap(got, np.array(want)), 1e-12)
                for frame in range(4):
                    gap = _maya_orientation(got[frame, :3, :3]) - _maya_orientation(
                        target.frames[["t"]].world_matrix[frame, 0, :3, :3]
                    )
                    self.assertLessEqual(np.abs(gap).max(), 1e-12)

    def test_a_rigid_parent_mirrored_on_one_side_alone_stays_on_the_old_path(self):
        # no bit of a rig of uniform scales moves: rotate is the euler of
        # local_matrix(inv(RO) * M, JO), which leaves that mirror in
        rng = np.random.default_rng(98)
        for axis in range(3):
            for _ in range(20):
                parent       = np.full(3, rng.uniform(0.5, 2.0))
                parent[axis] = -parent[axis]
                rig          = self._rig(rng, parent_scale=parent, turned=False)
                node         = rig["k"]

                M         = rig["t"].world_matrix
                M[:3, :3] = H._unit_rows(M[:3, :3])
                Y = matrix_multiply(
                    np.dot(node.joint_orient_matrix, node.parent_scale_inverse_matrix), node.get_parent_matrix()
                )
                L    = matrix_local(np.dot(np.linalg.inv(node.rotate_axis_matrix), M), Y)
                want = np.degrees(matrix_to_euler(L, node.rotate_order))[0]

                node.match_rotate(rig["t"])
                self.assertEqual(node.rotate.tolist(), want.tolist())

    def test_the_orientation_reads_are_orthonormal(self):
        rng = np.random.default_rng(99)
        for _ in range(100):
            world         = np.eye(4)
            world[:3, :3] = np.diag(rng.uniform(0.4, 2.2, 3)) @ _sh3(rng.uniform(-1.0, 1.0, 3)) @ _turn(rng.uniform(-170.0, 170.0, 3))
            for read in (H._rotation, H._orientations):
                block = read(world)[:3, :3]
                self.assertLessEqual(np.abs(block @ block.T - np.eye(3)).max(), 1e-14)

    def test_the_orientation_read_squares_near_parallel_rows(self):
        # no row of an orientation read is flat: a floor of its own would
        # zero what the rows above leave of these
        q = _turn((10.0, 20.0, 30.0))
        for gap in (1e-2, 1e-4, 1e-6, 1e-8):
            with self.subTest(gap=gap):
                rows = H._unit_rows(np.array([q[0], q[0] + gap * q[1], q[2]]))
                self.assertLessEqual(np.abs(rows @ rows.T - np.eye(3)).max(), 1e-14)

    def test_a_row_that_is_not_finite_stays_in_the_orientation_read(self):
        block    = np.diag((1.0, 2.0, 3.0)) @ _sh3((0.5, 0.25, 0.75)) @ _turn((10.0, 20.0, 30.0))
        block[2] = (np.nan, 1.0, 1.0)
        self.assertTrue(np.isnan(H._unit_rows(block)[2, 0]))

    def test_a_square_world_stays_at_its_unit_rows_to_the_bit(self):
        rng = np.random.default_rng(100)
        for _ in range(100):
            world         = np.eye(4)
            world[:3, :3] = np.diag(rng.uniform(0.4, 2.2, 3)) @ _turn(rng.uniform(-170.0, 170.0, 3))
            block         = world[:3, :3]
            want          = block / (np.einsum("ij,ij->i", block, block) ** 0.5)[:, None]
            self.assertTrue(np.array_equal(H._rotation(world)[:3, :3], want))

    def test_a_parent_scale_of_zero_reads_nan_and_raises_nothing(self):
        # a bare inverse raises LinAlgError on the whole clip
        for axis in range(3):
            scale       = np.ones(3)
            scale[axis] = 0.0
            a           = TransformData("a", uuid=generate_uuid(), scale=scale, rotate=(10.0, 20.0, 30.0), translate=(1.0, 2.0, 3.0))
            b = TransformData(
                "b", parent_uuid=a.uuid, rotate=(5.0, 6.0, 7.0), translate=(0.0, 1.0, 0.0), scale=(1.0, 2.0, 3.0)
            )
            rig = HierarchyData([a, b, TransformData("t", rotate=(40.0, 50.0, 60.0))])
            with self.subTest(axis=axis):
                rig["b"].match_rotate(rig["t"])
                self.assertFalse(np.isfinite(rig["b"].rotate).all())

                rig = HierarchyData(
                    [
                        TransformData("a", uuid=a.uuid, scale=(1.0, 2.0, 3.0), rotate=(10.0, 20.0, 30.0)),
                        TransformData("b", parent_uuid=a.uuid, rotate=(5.0, 6.0, 7.0), translate=(0.0, 1.0, 0.0)),
                    ]
                )
                clip                     = ClipData(rig, frames=3)
                scales                   = np.array(clip.frames[["a"]].scale)
                scales[1, 0, axis]       = 0.0
                clip.frames[["a"]].scale = scales
                clip.frames[["b"]].match_rotate((40.0, 50.0, 60.0))
                rotate = clip.frames[["b"]].rotate
                self.assertTrue(np.isfinite(rotate[[0, 2]]).all())
                self.assertFalse(np.isfinite(rotate[1]).all())


class TestShearRigidGate(unittest.TestCase):
    """match_rotate's old path takes a JO IS P that is one length times a
    rotation or a mirror: rows of one length that are not square do not qualify,
    nor rows a hair apart"""

    def _rig(self, scale, shear=None) -> HierarchyData:
        up = generate_uuid()
        return HierarchyData(
            [
                TransformData("p", uuid=up, scale=scale, shear=shear, rotate=(30.0, 40.0, 50.0)),
                TransformData(
                    "k",
                    parent_uuid = up,
                    rotate      = (5.0, 6.0, 7.0),
                    scale       = (1.0, 2.0, 3.0),
                    translate   = (1.0, 2.0, 3.0),
                ),
                TransformData("t", rotate=(70.0, -20.0, 35.0)),
            ]
        )

    def _landed(self, world, rig) -> float:
        return _gap(_maya_orientation(world), _maya_orientation(rig["t"].world_matrix[:3, :3]))

    def test_rows_of_one_length_that_are_not_square(self):
        xy, xz, yz = SHEAR
        scale   = (1.0, (1.0 + xy * xy) ** -0.5, (1.0 + xz * xz + yz * yz) ** -0.5)
        rig     = self._rig(scale, SHEAR)
        lengths = np.linalg.norm(rig["p"].world_matrix[:3, :3], axis=1)
        self.assertLessEqual(np.ptp(lengths), 1e-15)

        clip = ClipData(rig.copy(), frames=2)
        rig["k"].match_rotate(rig["t"])
        self.assertLessEqual(self._landed(rig["k"].world_matrix[:3, :3], rig), 1e-12)

        clip.frames[["k"]].match_rotate(clip.copy().frames[["t"]])
        self.assertLessEqual(self._landed(clip.frames[["k"]].world_matrix[0, 0, :3, :3], rig), 1e-12)

    def test_rows_a_hair_from_one_length(self):
        rig = self._rig((1.0, 1.0 + 1e-9, 1.0))
        rig["k"].match_rotate(rig["t"])
        self.assertLessEqual(self._landed(rig["k"].world_matrix[:3, :3], rig), 1e-12)


class TestShearPoseAlgebra(unittest.TestCase, _DeltaRigs):
    """shear is additive, every node's, and a world that shears gives Maya's
    orientation: ``b.add_delta(a.get_delta(b))`` lands on ``a``"""

    def _chain3(self, rng, shear=True, uniform=False) -> HierarchyData:
        uuids = [generate_uuid() for _ in range(3)]
        nodes = []
        for i in range(3):
            scale = np.full(3, rng.uniform(0.5, 2.0)) if uniform else rng.uniform(0.3, 2.5, 3)
            nodes.append(
                TransformData(
                    f"n{i}",
                    uuid         = uuids[i],
                    parent_uuid  = uuids[i - 1] if i else None,
                    scale        = scale,
                    rotate       = rng.uniform(-170.0, 170.0, 3),
                    translate    = rng.uniform(-5.0, 5.0, 3),
                    rotate_order = int(rng.integers(0, 6)),
                    rotate_axis  = rng.uniform(-40.0, 40.0, 3) if rng.random() < 0.5 else None,
                    shear        = rng.uniform(-1.0, 1.0, 3) if shear else None,
                )
            )
        return HierarchyData(nodes)

    def test_a_delta_carries_the_shear_of_the_difference(self):
        rng = np.random.default_rng(101)
        a, b = self._chain3(rng), self._chain3(rng)
        self.assertTrue(np.array_equal(a.get_delta(b).shear, a.shear - b.shear))

    def test_the_round_trip_on_sheared_rigs(self):
        rng = np.random.default_rng(102)
        for trial in range(120):
            a, b = self._chain3(rng), self._chain3(rng)
            back = b.add_delta(a.get_delta(b))
            with self.subTest(trial=trial):
                self.assertLessEqual(np.abs(back.shear - a.shear).max(), 1e-14)
                self.assertLessEqual(np.abs(back.world_matrix - a.world_matrix).max(), 1e-10)

    def test_the_round_trip_on_a_rig_holding_none_whose_worlds_shear(self):
        rng = np.random.default_rng(103)
        for trial in range(120):
            a, b = self._chain3(rng, shear=False), self._chain3(rng, shear=False)
            back = b.add_delta(a.get_delta(b))
            with self.subTest(trial=trial):
                self.assertLessEqual(np.abs(back.world_matrix - a.world_matrix).max(), 1e-10)
                self.assertFalse(back.shear.any())

    def test_a_clip_is_one_pose_at_a_time(self):
        names = [f"j{i}" for i in range(6)]
        ups   = [None, "j0", "j1", "j1", "j3", "j4"]
        rng   = np.random.default_rng(104)

        def sheared(seed):
            clip              = self._animated(self._rig(names, ups, 7, scale=True), seed=seed)
            clip.frames.shear = rng.uniform(-0.5, 0.5, clip.frames.shear.shape)
            return clip

        mine, theirs = sheared(8), sheared(9)
        taken = mine.get_delta(theirs)
        self._assert_same(taken, self._per_frame("get_delta", mine, theirs))
        self.assertTrue(np.array_equal(taken.frames.shear, mine.frames.shear - theirs.frames.shear))

        pose       = self._rig(names, ups, 7, scale=True)
        pose.shear = rng.uniform(-0.5, 0.5, pose.shear.shape)
        self._assert_same(mine.get_delta(pose), self._per_frame("get_delta", mine, pose))
        self._assert_same(mine.add_delta(pose), self._per_frame("add_delta", mine, pose))

        deltas = mine.get_delta(theirs)
        got    = theirs.add_delta(deltas)
        self.assertFalse(np.array_equal(got.frames.shear[0], got.frames.shear[1]))
        self._assert_same(got, self._per_frame("add_delta", theirs, deltas))

    def test_a_clip_of_deltas_in_place(self):
        names  = [f"j{i}" for i in range(6)]
        ups    = [None, "j0", "j1", "j1", "j3", "j4"]
        rng    = np.random.default_rng(105)
        clip   = self._animated(self._rig(names, ups, 7, scale=True), seed=8)
        deltas = self._animated(self._rig(names, ups, 7, scale=True), seed=9)

        clip.frames.shear   = rng.uniform(-0.5, 0.5, clip.frames.shear.shape)
        deltas.frames.shear = rng.uniform(-0.5, 0.5, deltas.frames.shear.shape)
        want                = self._per_frame("__iadd__", deltas, clip)
        added               = deltas.frames.shear + clip.frames.shear
        deltas.frames += clip
        self._assert_same(deltas, want)
        self.assertTrue(np.array_equal(deltas.frames.shear, added))

    def test_a_node_the_pairing_leaves_out_keeps_its_deltas_shear(self):
        rng = np.random.default_rng(106)
        a, b = self._chain3(rng), self._chain3(rng)
        delta            = a.get_delta(b)
        delta["n2"].name = "other"
        got              = b[:2].add_delta(delta, by="name")
        self.assertEqual(got["other"].shear.tolist(), delta["other"].shear.tolist())

        clip = ClipData(b.copy(), frames=2).add_delta(delta, by="name")
        self.assertEqual(clip.frames[["other"]].shear[1, 0].tolist(), delta["other"].shear.tolist())


if __name__ == "__main__":
    unittest.main()
