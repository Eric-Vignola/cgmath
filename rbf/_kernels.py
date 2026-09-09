"""
Radial Basis Function (RBF) Kernels Module

This module provides a comprehensive collection of RBF kernels optimized with
NumPy and parallelized Numba JIT compilation. These kernels are commonly used
in scattered data interpolation, mesh deformation, and Gaussian processes.

Kernel Categories:
    - Infinitely Smooth: Gaussian, Multiquadric, Inverse Multiquadric, Inverse Quadratic
    - Piecewise Smooth/Polyharmonic: Linear, Cubic, Quintic, Thin Plate Spline
    - Compactly Supported (Wendland): C0, C2, C4, C6, Wu C2, Wu C4
    - Specialized: Matern (1/2, 3/2, 5/2), Cauchy, Log, Bump
"""

from __future__ import annotations

import numpy as np


# =============================================================================
# PUBLIC WRAPPER FUNCTIONS
# =============================================================================


# -----------------------------------------------------------------------------
# Infinitely Smooth Kernels
# -----------------------------------------------------------------------------


def gaussian(X, epsilon=1.0):
    """
    Gaussian (RBF/Squared Exponential) kernel.

    phi(r) = exp(-(epsr)^2)

    The Gaussian kernel is infinitely differentiable and produces very smooth
    interpolations. It is the most commonly used RBF kernel.

    Parameters
    ----------
    X : ndarray of shape (n, m)
        Distance matrix
    epsilon : float, optional
        Shape parameter controlling the width of the kernel. Larger values
        produce narrower peaks. Default is 1.0.

    Returns
    -------
    result : ndarray of shape (n, m)
        Kernel evaluation at each distance
    """
    from cgmath.rbf._numba._kernels import _gaussian

    X       = np.asarray(X, dtype=np.float64)
    epsilon = float(epsilon)

    return _gaussian(X, epsilon)


def multiquadric(X, epsilon=1.0):
    """
    Multiquadric kernel.

    phi(r) = sqrt(1 + (epsr)^2)

    The multiquadric kernel is infinitely smooth and conditionally positive
    definite of order 1. It tends to produce smooth interpolations that can
    handle large gradients well.

    Parameters
    ----------
    X : ndarray of shape (n, m)
        Distance matrix
    epsilon : float, optional
        Shape parameter. Default is 1.0.

    Returns
    -------
    result : ndarray of shape (n, m)
        Kernel evaluation at each distance
    """
    from cgmath.rbf._numba._kernels import _multiquadric

    X       = np.asarray(X, dtype=np.float64)
    epsilon = float(epsilon)

    return _multiquadric(X, epsilon)


def inverse_multiquadric(X, epsilon=1.0):
    """
    Inverse Multiquadric kernel.

    phi(r) = 1 / sqrt(1 + (epsr)^2)

    The inverse multiquadric kernel is strictly positive definite and infinitely
    differentiable. Unlike the multiquadric, it does not require polynomial
    augmentation for unique interpolation.

    Parameters
    ----------
    X : ndarray of shape (n, m)
        Distance matrix
    epsilon : float, optional
        Shape parameter. Default is 1.0.

    Returns
    -------
    result : ndarray of shape (n, m)
        Kernel evaluation at each distance
    """
    from cgmath.rbf._numba._kernels import _inverse_multiquadric

    X       = np.asarray(X, dtype=np.float64)
    epsilon = float(epsilon)

    return _inverse_multiquadric(X, epsilon)


def inverse_quadratic(X, epsilon=1.0):
    """
    Inverse Quadratic kernel.

    phi(r) = 1 / (1 + (epsr)^2)

    The inverse quadratic kernel is strictly positive definite and infinitely
    smooth. It has faster decay than the inverse multiquadric.

    Parameters
    ----------
    X : ndarray of shape (n, m)
        Distance matrix
    epsilon : float, optional
        Shape parameter. Default is 1.0.

    Returns
    -------
    result : ndarray of shape (n, m)
        Kernel evaluation at each distance
    """
    from cgmath.rbf._numba._kernels import _inverse_quadratic

    X       = np.asarray(X, dtype=np.float64)
    epsilon = float(epsilon)

    return _inverse_quadratic(X, epsilon)


# -----------------------------------------------------------------------------
# Piecewise Smooth / Polyharmonic Kernels
# -----------------------------------------------------------------------------


def linear(X, epsilon=1.0):
    """
    Linear kernel.

    phi(r) = r

    The simplest polyharmonic spline. Produces piecewise linear interpolations.

    Parameters
    ----------
    X : ndarray of shape (n, m)
        Distance matrix
    epsilon : float, optional
        Scale parameter. Default is 1.0.

    Returns
    -------
    result : ndarray of shape (n, m)
        Kernel evaluation at each distance
    """
    from cgmath.rbf._numba._kernels import _linear

    X       = np.asarray(X, dtype=np.float64)
    epsilon = float(epsilon)

    return _linear(X, epsilon)


def cubic(X, epsilon=1.0):
    """
    Cubic kernel.

    phi(r) = r^3

    A polyharmonic spline that produces C^1 continuous interpolations.
    Commonly used in 2D and 3D scattered data interpolation.

    Parameters
    ----------
    X : ndarray of shape (n, m)
        Distance matrix
    epsilon : float, optional
        Scale parameter. Default is 1.0.

    Returns
    -------
    result : ndarray of shape (n, m)
        Kernel evaluation at each distance
    """
    from cgmath.rbf._numba._kernels import _cubic

    X       = np.asarray(X, dtype=np.float64)
    epsilon = float(epsilon)

    return _cubic(X, epsilon)


def quintic(X, epsilon=1.0):
    """
    Quintic kernel.

    phi(r) = r^5

    A polyharmonic spline that produces C^2 continuous interpolations.
    Higher smoothness than cubic but may have more oscillation.

    Parameters
    ----------
    X : ndarray of shape (n, m)
        Distance matrix
    epsilon : float, optional
        Scale parameter. Default is 1.0.

    Returns
    -------
    result : ndarray of shape (n, m)
        Kernel evaluation at each distance
    """
    from cgmath.rbf._numba._kernels import _quintic

    X       = np.asarray(X, dtype=np.float64)
    epsilon = float(epsilon)

    return _quintic(X, epsilon)


def thin_plate_spline(X, epsilon=1.0):
    """
    Thin Plate Spline kernel.

    phi(r) = r^2 log(r)   for r > 0
    phi(0) = 0

    Named after the physical behavior of a thin plate under bending forces.
    Minimizes the bending energy of the interpolation surface. Conditionally
    positive definite of order 2.

    Parameters
    ----------
    X : ndarray of shape (n, m)
        Distance matrix
    epsilon : float, optional
        Scale parameter. Default is 1.0.

    Returns
    -------
    result : ndarray of shape (n, m)
        Kernel evaluation at each distance
    """
    from cgmath.rbf._numba._kernels import _thin_plate_spline

    X       = np.asarray(X, dtype=np.float64)
    epsilon = float(epsilon)

    return _thin_plate_spline(X, epsilon)


def polyharmonic(X, k=2, epsilon=1.0):
    """
    Generalized Polyharmonic Spline kernel.

    phi(r) = r^k           for k odd
    phi(r) = r^k log(r)    for k even

    The polyharmonic splines are a family of conditionally positive definite
    kernels. Special cases include:
        - k=1: Linear
        - k=2: Thin plate spline
        - k=3: Cubic
        - k=5: Quintic

    Parameters
    ----------
    X : ndarray of shape (n, m)
        Distance matrix
    k : int, optional
        Order of the polyharmonic spline. Default is 2 (thin plate spline).
    epsilon : float, optional
        Scale parameter. Default is 1.0.

    Returns
    -------
    result : ndarray of shape (n, m)
        Kernel evaluation at each distance
    """
    from cgmath.rbf._numba._kernels import _polyharmonic

    X       = np.asarray(X, dtype=np.float64)
    k       = int(k)
    epsilon = float(epsilon)

    return _polyharmonic(X, k, epsilon)


# -----------------------------------------------------------------------------
# Compactly Supported Kernels (Wendland Functions)
# -----------------------------------------------------------------------------


def wendland_c0(X, r=1.0):
    """
    Wendland C0 (d <= 3) compactly supported kernel.

    phi(r) = (1 - r)_+^2

    The simplest Wendland function. Continuous (C^0) but not differentiable at
    the boundary. Support radius r controls the locality of the kernel.

    Parameters
    ----------
    X : ndarray of shape (n, m)
        Distance matrix
    r : float, optional
        Support radius. Points beyond this distance have zero influence.
        Default is 1.0.

    Returns
    -------
    result : ndarray of shape (n, m)
        Kernel evaluation at each distance
    """
    from cgmath.rbf._numba._kernels import _wendland_c0

    X = np.asarray(X, dtype=np.float64)
    r = float(r)

    return _wendland_c0(X, r)


def wendland_c2(X, r=1.0):
    """
    Wendland C2 (d <= 3) compactly supported kernel.

    phi(r) = (1 - r)_+^4(4r + 1)

    Positive definite for dimensions up to 3. Has continuous second
    derivatives (C^2). This is equivalent to the Beckert-Wendland C2 basis.

    Parameters
    ----------
    X : ndarray of shape (n, m)
        Distance matrix
    r : float, optional
        Support radius. Default is 1.0.

    Returns
    -------
    result : ndarray of shape (n, m)
        Kernel evaluation at each distance
    """
    from cgmath.rbf._numba._kernels import _wendland_c2

    X = np.asarray(X, dtype=np.float64)
    r = float(r)

    return _wendland_c2(X, r)


def wendland_c4(X, r=1.0):
    """
    Wendland C4 (d <= 3) compactly supported kernel.

    phi(r) = (1 - r)_+^6(35r^2 + 18r + 3)

    Positive definite for dimensions up to 3. Has continuous fourth
    derivatives (C^4). Smoother than C2 but slightly more expensive.

    Parameters
    ----------
    X : ndarray of shape (n, m)
        Distance matrix
    r : float, optional
        Support radius. Default is 1.0.

    Returns
    -------
    result : ndarray of shape (n, m)
        Kernel evaluation at each distance
    """
    from cgmath.rbf._numba._kernels import _wendland_c4

    X = np.asarray(X, dtype=np.float64)
    r = float(r)

    return _wendland_c4(X, r)


def wendland_c6(X, r=1.0):
    """
    Wendland C6 (d <= 3) compactly supported kernel.

    phi(r) = (1 - r)_+^8(32r^3 + 25r^2 + 8r + 1)

    Positive definite for dimensions up to 3. Has continuous sixth
    derivatives (C^6). The smoothest standard Wendland function.

    Parameters
    ----------
    X : ndarray of shape (n, m)
        Distance matrix
    r : float, optional
        Support radius. Default is 1.0.

    Returns
    -------
    result : ndarray of shape (n, m)
        Kernel evaluation at each distance
    """
    from cgmath.rbf._numba._kernels import _wendland_c6

    X = np.asarray(X, dtype=np.float64)
    r = float(r)

    return _wendland_c6(X, r)


def wu_c2(X, r=1.0):
    """
    Wu C2 compactly supported kernel.

    phi(r) = (1 - r)_+^5(8r^2 + 5r + 1)

    An alternative to Wendland C2 with similar properties but different
    polynomial degree and coefficients (Wu, 1995).

    Parameters
    ----------
    X : ndarray of shape (n, m)
        Distance matrix
    r : float, optional
        Support radius. Default is 1.0.

    Returns
    -------
    result : ndarray of shape (n, m)
        Kernel evaluation at each distance
    """
    from cgmath.rbf._numba._kernels import _wu_c2

    X = np.asarray(X, dtype=np.float64)
    r = float(r)

    return _wu_c2(X, r)


def wu_c4(X, r=1.0):
    """
    Wu C4 compactly supported kernel.

    phi(r) = (1 - r)_+^7(16r^3 + 12r^2 + 5r + 1)

    An alternative to Wendland C4 with similar properties but different
    polynomial degree and coefficients (Wu, 1995).

    Parameters
    ----------
    X : ndarray of shape (n, m)
        Distance matrix
    r : float, optional
        Support radius. Default is 1.0.

    Returns
    -------
    result : ndarray of shape (n, m)
        Kernel evaluation at each distance
    """
    from cgmath.rbf._numba._kernels import _wu_c4

    X = np.asarray(X, dtype=np.float64)
    r = float(r)

    return _wu_c4(X, r)


# -----------------------------------------------------------------------------
# Specialized Kernels (Matern and Others)
# -----------------------------------------------------------------------------


def matern_12(X, epsilon=1.0):
    """
    Matern kernel with nu = 1/2 (Exponential kernel).

    phi(r) = exp(-r/eps)

    Equivalent to the exponential kernel. Produces rough (non-differentiable)
    sample paths in Gaussian processes. Has finite smoothness.

    Parameters
    ----------
    X : ndarray of shape (n, m)
        Distance matrix
    epsilon : float, optional
        Length scale parameter. Default is 1.0.

    Returns
    -------
    result : ndarray of shape (n, m)
        Kernel evaluation at each distance
    """
    from cgmath.rbf._numba._kernels import _matern_12

    X       = np.asarray(X, dtype=np.float64)
    epsilon = float(epsilon)

    return _matern_12(X, epsilon)


def matern_32(X, epsilon=1.0):
    """
    Matern kernel with nu = 3/2.

    phi(r) = (1 + sqrt3r/eps) exp(-sqrt3r/eps)

    Produces once-differentiable sample paths in Gaussian processes.
    A good balance between smoothness and computational efficiency.

    Parameters
    ----------
    X : ndarray of shape (n, m)
        Distance matrix
    epsilon : float, optional
        Length scale parameter. Default is 1.0.

    Returns
    -------
    result : ndarray of shape (n, m)
        Kernel evaluation at each distance
    """
    from cgmath.rbf._numba._kernels import _matern_32

    X       = np.asarray(X, dtype=np.float64)
    epsilon = float(epsilon)

    return _matern_32(X, epsilon)


def matern_52(X, epsilon=1.0):
    """
    Matern kernel with nu = 5/2.

    phi(r) = (1 + sqrt5r/eps + 5r^2/(3eps^2)) exp(-sqrt5r/eps)

    Produces twice-differentiable sample paths in Gaussian processes.
    Often used when smooth interpolation is desired but the Gaussian
    kernel's infinite smoothness is not necessary.

    Parameters
    ----------
    X : ndarray of shape (n, m)
        Distance matrix
    epsilon : float, optional
        Length scale parameter. Default is 1.0.

    Returns
    -------
    result : ndarray of shape (n, m)
        Kernel evaluation at each distance
    """
    from cgmath.rbf._numba._kernels import _matern_52

    X       = np.asarray(X, dtype=np.float64)
    epsilon = float(epsilon)

    return _matern_52(X, epsilon)


def cauchy(X, epsilon=1.0):
    """
    Cauchy kernel (Rational Quadratic with alpha=1).

    phi(r) = 1 / (1 + (r/eps)^2)

    A long-tailed kernel that can be seen as a scale mixture of Gaussians.
    Useful when data may have outliers or heavy-tailed distributions.

    Parameters
    ----------
    X : ndarray of shape (n, m)
        Distance matrix
    epsilon : float, optional
        Scale parameter. Default is 1.0.

    Returns
    -------
    result : ndarray of shape (n, m)
        Kernel evaluation at each distance
    """
    from cgmath.rbf._numba._kernels import _cauchy

    X       = np.asarray(X, dtype=np.float64)
    epsilon = float(epsilon)

    return _cauchy(X, epsilon)


def log_kernel(X, epsilon=1.0):
    """
    Logarithmic kernel.

    phi(r) = log(1 + r/eps)

    A conditionally positive definite kernel with logarithmic growth.
    Useful for modeling slowly varying functions.

    Parameters
    ----------
    X : ndarray of shape (n, m)
        Distance matrix
    epsilon : float, optional
        Scale parameter. Default is 1.0.

    Returns
    -------
    result : ndarray of shape (n, m)
        Kernel evaluation at each distance
    """
    from cgmath.rbf._numba._kernels import _log_kernel

    X       = np.asarray(X, dtype=np.float64)
    epsilon = float(epsilon)

    return _log_kernel(X, epsilon)


def bump(X, r=1.0):
    """
    Bump function (smooth compactly supported kernel).

    phi(r) = exp(-1/(1-r^2))   for r < 1
    phi(r) = 0                for r >= 1

    The bump function is infinitely differentiable and has compact support.
    It is one of the few functions that is both C^inf smooth and compactly supported.

    Parameters
    ----------
    X : ndarray of shape (n, m)
        Distance matrix
    r : float, optional
        Support radius. Default is 1.0.

    Returns
    -------
    result : ndarray of shape (n, m)
        Kernel evaluation at each distance
    """
    from cgmath.rbf._numba._kernels import _bump

    X = np.asarray(X, dtype=np.float64)
    r = float(r)

    return _bump(X, r)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


KERNEL_REGISTRY = {
    # Infinitely smooth kernels
    "gaussian":             gaussian,              # aka: rbf, squared_exponential
    "multiquadric":         multiquadric,          # aka: mq
    "inverse_multiquadric": inverse_multiquadric,  # aka: imq
    "inverse_quadratic":    inverse_quadratic,     # aka: iq
    # Piecewise smooth / Polyharmonic kernels
    "linear":            linear,
    "cubic":             cubic,
    "quintic":           quintic,
    "thin_plate_spline": thin_plate_spline,  # aka: tps, thin_plate
    "polyharmonic":      polyharmonic,
    # Compactly supported kernels (Wendland)
    "wendland_c0": wendland_c0,
    "wendland_c2": wendland_c2,  # aka: beckert_wendland_c2
    "wendland_c4": wendland_c4,
    "wendland_c6": wendland_c6,
    "wu_c2":       wu_c2,
    "wu_c4":       wu_c4,
    # Specialized kernels
    "matern_12": matern_12,   # aka: matern_1_2, exponential
    "matern_32": matern_32,   # aka: matern_3_2
    "matern_52": matern_52,   # aka: matern_5_2
    "cauchy":    cauchy,
    "log":       log_kernel,  # aka: logarithmic
    "bump":      bump,
}


def get_kernel(name):
    """
    Get a kernel function by name.

    Parameters
    ----------
    name : str
        Name of the kernel. Case-insensitive. Supported names include:
        - Infinitely smooth: 'gaussian', 'multiquadric',
          'inverse_multiquadric', 'inverse_quadratic'
        - Polyharmonic: 'linear', 'cubic', 'quintic', 'thin_plate_spline',
          'polyharmonic'
        - Compactly supported: 'wendland_c0', 'wendland_c2', 'wendland_c4',
          'wendland_c6', 'wu_c2', 'wu_c4'
        - Specialized: 'matern_12', 'matern_32', 'matern_52', 'cauchy',
          'log', 'bump'

    Returns
    -------
    kernel : callable
        The kernel function.

    Raises
    ------
    ValueError
        If the kernel name is not recognized.

    Examples
    --------
    >>> kernel = get_kernel('gaussian')
    >>> distances = np.array([[0, 1, 2], [1, 0, 1]])
    >>> kernel(distances, epsilon=0.5)
    """
    name_lower = name.lower()
    if name_lower not in KERNEL_REGISTRY:
        available = sorted(set(KERNEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown kernel: '{name}'. Available kernels: {', '.join(available)}"
        )
    return KERNEL_REGISTRY[name_lower]


def list_kernels():
    """
    List all available kernel names.

    Returns
    -------
    kernels : list of str
        Sorted list of unique kernel names (excluding aliases).
    """
    unique_funcs = set()
    unique_names = []

    for name, func in sorted(KERNEL_REGISTRY.items()):
        if func not in unique_funcs:
            unique_funcs.add(func)
            unique_names.append(name)

    return unique_names


# =============================================================================
# LU DECOMPOSITION AND LINEAR SOLVE
# =============================================================================


def __getattr__(name):
    # ``lu_solve`` and ``lu_solve_factored`` are dispatchers that call the
    # other jit kernels in :mod:`cgmath.rbf._numba._lu`, so they are handed
    # back as-is instead of being wrapped -- a plain-Python wrapper could not
    # be called from jit-compiled code.
    if name in ("lu_solve", "lu_solve_factored"):
        from cgmath.rbf._numba import _lu

        return getattr(_lu, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")