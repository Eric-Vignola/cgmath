"""
Radial basis function kernels

``WrapData`` moved to :mod:`cgmath.geometry.deform`.  This package must not
re-export it: ``wrap`` imports ``_kernels`` at module scope, so re-exporting
from here would import back into a partially initialized module.
"""