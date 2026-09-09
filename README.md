cgmath
======

DCC agnostic geometry processing, resampling, and rendering tools


About
-----

A toolset for working with polygonal geometry outside of any host application.
Meshes, UVs, skin weights, morph targets and hierarchies are NumPy arrays
wrapped in dataclasses, so the same code runs inside a DCC, in a standalone
script, or on a farm. Heavy kernels are compiled with Numba.

Topology is derived on first use and cached on the object: adjacency maps, edge
tables, normals, borders, shells and symmetry are each computed once
per mesh and reused.


Requirements
------------

Numpy, Scipy, Numba, and the transforms python modules.

Optional, per feature:

    Pillow                  texture I/O and rendered frames
    pygltflib, trimesh      glTF / GLB read and write
    Autodesk FBX SDK        FBX read and write
    pxr (USD)               USD stage read and write
    scikit-image or OpenCV  UV rasterization


Author
------

* **Eric Vignola** (eric.vignola@gmail.com)

If this was useful to you, [buy me a coffee](https://buymeacoffee.com/ericvignola) ☕


Example
-------

```python
import numpy as np
from cgmath.geometry import MeshData

# a unit cube: flat vertex indices plus a vertex count per face
box = MeshData(
    points  = np.array([[-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5],
                        [-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5],
                        [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5],
                        [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5]]),
    indices = np.array([0, 1, 3, 2,  2, 3, 5, 4,  4, 5, 7, 6,
                        6, 7, 1, 0,  1, 7, 5, 3,  6, 0, 2, 4]),
    counts  = np.array([4, 4, 4, 4, 4, 4]),
)

box.point_count           # 8
box.face_count            # 6
box.edge_count            # 12
box.area                  # 6.0
box.closed                # True

box.e2v                   # (12, 2) edge to vertex pairs
box.get_face_normals()    # (6, 3)
box.get_vertex_normals()  # (8, 3)
box.get_edge_lengths()    # (12,)
```

Meshes also come off disk:

```python
mesh = MeshData.load_obj("head.obj")
mesh = MeshData.load_glb("head.glb")
mesh = MeshData.load_fbx("head.fbx")
```


Packages
--------

    cgmath.geometry     meshes, UVs, skin weights, morph targets, b-splines,
                        SDFs, deformers, resampling and correspondence
    cgmath.hierarchy    transform hierarchies, skeletons and clips
    cgmath.constraints  procrustes / rigid alignment
    cgmath.rbf          radial basis function kernels
    cgmath.render       software raytracer, scene graph, cameras, textures
    cgmath.formats      FBX, glTF / GLB and USD interchange


Tests
-----

```python
from cgmath.utils import run_tests
run_tests()
```


License
-------

BSD 3-Clause License: Copyright (c) 2026, Eric Vignola All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of copyright holders nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
