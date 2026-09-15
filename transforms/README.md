cgmath.transforms
=================

A Numeric Transform operations toolset

Part of [cgmath](../README.md), imported as `cgmath.transforms`. The same code is
also published standalone as [`transforms`](https://github.com/Eric-Vignola/transforms)
for anyone who wants this math without the rest of cgmath.


About
-----

This module provides a toolset to do fast matrix/quaternion/euler/vector
operations. Calculations are made element wise over a batch, and follow NumPy's
broadcasting rules: every input must be as long as the longest one, or exactly
one, in which case it is reused for the whole batch at no memory cost.

For copy-paste recipes covering every function, see
[`CHEATSHEET.md`](CHEATSHEET.md).


Requirements
------------

Numpy, Scipy and Numba python modules.


Author
------

* **Eric Vignola** (eric.vignola@gmail.com)

If this was useful to you, [buy me a coffee](https://buymeacoffee.com/ericvignola) ☕


Example
-------

    import numpy as np
    from cgmath.transforms import vector_random, vector_slerp

    # declare two arrays of 10 million vectors
    V0 = vector_random(10**7)
    V1 = vector_random(10**7)

    # slerp 10 million vectors 1:1 element wise, half way
    slerp = vector_slerp(V0, V1, 0.5)

    # slerp 10 million vectors to a common vector, half way
    slerp = vector_slerp(V0, V1[0], 0.5)

    # slerp 10 million vectors 1:1 element wise, with a random ratio for each
    blend = np.random.random(10**7)
    slerp = vector_slerp(V0, V1, blend)


Supported Functions
-------------------

    axis_angle_to_euler       matrix_random                   quaternion_slerp
    axis_angle_to_matrix      matrix_slerp                    quaternion_squad
    axis_angle_to_quaternion  matrix_to_euler                 quaternion_sub
    euler_filter              matrix_to_quaternion            quaternion_to_euler
    euler_random              matrix_transpose                quaternion_to_matrix
    euler_reorder             matrix_weighted_rotational      vector_angle
    euler_slerp               matrix_weighted_transformation  vector_arc_to_euler
    euler_to_matrix           quaternion_add                  vector_arc_to_matrix
    euler_to_quaternion       quaternion_conjugate            vector_arc_to_quaternion
    matrix_decompose          quaternion_dot                  vector_cross
    matrix_delta              quaternion_exp                  vector_dot
    matrix_flatten            quaternion_intermediate         vector_lerp
    matrix_identity           quaternion_inverse              vector_magnitude
    matrix_interpolate        quaternion_log                  vector_normalize
    matrix_inverse            quaternion_multiply             vector_random
    matrix_local              quaternion_negate               vector_slerp
    matrix_multiply           quaternion_nlerp                vector_to_euler
    matrix_normalize          quaternion_normalize            vector_to_matrix
    matrix_point_multiply     quaternion_random               vector_to_quaternion

Rotate orders and axis indices are exported as constants:

    XYZ  YZX  ZXY  XZY  YXZ  ZYX          X  Y  Z


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
