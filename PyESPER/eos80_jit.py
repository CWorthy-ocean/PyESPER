#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JIT-compiled functions for EOS80, after the Seawater Python package, which
implements the CSIRO Seawater package in Python. These functions are intended
to be faster (after compilation) for the case of repeated or looped use.

Originally produced by D. Sandborn for the PyESPER package.
"""

import numpy as np
from numba import njit, prange

deg2rad = np.pi / 180.0
Kelvin = 273.15


@njit(parallel=True, cache=True)
def pres(depth, lat):
    p_out = np.empty(depth.shape[0], dtype=np.float64)
    for i in prange(depth.shape[0]):
        X = np.sin(np.abs(lat[i] * deg2rad))
        C1 = 5.92e-3 + X**2 * 5.25e-3
        p_out[i] = (
            (1.0 - C1) - (((1.0 - C1) ** 2.0) - (8.84e-6 * depth[i])) ** 0.5
        ) / 4.42e-6
    return p_out


@njit(cache=True)
def _adtg_scalar(s, t, p):
    T68 = t * 1.00024
    a = (3.5803e-5, 8.5258e-6, -6.836e-8, 6.6228e-10)
    b = (1.8932e-6, -4.2393e-8)
    c = (1.8741e-8, -6.7795e-10, 8.733e-12, -5.4481e-14)
    d = (-1.1351e-10, 2.7759e-12)
    e = (-4.6206e-13, 1.8676e-14, -2.1687e-16)
    return (
        a[0]
        + (a[1] + (a[2] + a[3] * T68) * T68) * T68
        + (b[0] + b[1] * T68) * (s - 35.0)
        + (
            (c[0] + (c[1] + (c[2] + c[3] * T68) * T68) * T68)
            + (d[0] + d[1] * T68) * (s - 35.0)
        )
        * p
        + (e[0] + (e[1] + e[2] * T68) * T68) * p * p
    )


@njit(parallel=True, cache=True)
def ptmp(s, t, p, pr=0.0):
    pt = np.empty(s.shape[0], dtype=np.float64)
    for i in prange(s.shape[0]):
        del_P = pr - p[i]
        del_th = del_P * _adtg_scalar(s[i], t[i], p[i])
        th = (t[i] * 1.00024) + 0.5 * del_th
        q = del_th

        del_th = del_P * _adtg_scalar(s[i], th / 1.00024, p[i] + 0.5 * del_P)
        th = th + (1.0 - 1.0 / 2**0.5) * (del_th - q)
        q = (2.0 - 2**0.5) * del_th + (-2.0 + 3.0 / 2**0.5) * q

        del_th = del_P * _adtg_scalar(s[i], th / 1.00024, p[i] + 0.5 * del_P)
        th = th + (1.0 + 1.0 / 2**0.5) * (del_th - q)
        q = (2.0 + 2**0.5) * del_th + (-2.0 - 3.0 / 2**0.5) * q

        del_th = del_P * _adtg_scalar(s[i], th / 1.00024, p[i] + del_P)
        pt[i] = (th + (del_th - 2.0 * q) / 6.0) / 1.00024
    return pt


@njit(parallel=True, cache=True)
def satO2(s, t):
    s_out = np.empty(s.shape[0], dtype=np.float64)
    a = (-173.4292, 249.6339, 143.3483, -21.8492)
    b = (-0.033096, 0.014259, -0.0017000)

    for i in prange(s.shape[0]):
        t_k = Kelvin + (t[i] * 1.00024)
        t_100 = t_k / 100.0

        lnC = (
            a[0]
            + a[1] * (1.0 / t_100)
            + a[2] * np.log(t_100)
            + a[3] * t_100
            + s[i] * (b[0] + b[1] * t_100 + b[2] * (t_100**2.0))
        )
        s_out[i] = np.exp(lnC)
    return s_out


@njit(cache=True)
def _smow_scalar(t):
    a = (
        999.842594,
        6.793952e-2,
        -9.095290e-3,
        1.001685e-4,
        -1.120083e-6,
        6.536332e-9,
    )
    T68 = t * 1.00024
    return (
        a[0]
        + (a[1] + (a[2] + (a[3] + (a[4] + a[5] * T68) * T68) * T68) * T68)
        * T68
    )


@njit(cache=True)
def _seck_scalar(s, t, p):
    p_atm = p / 10.0
    T68 = t * 1.00024

    h = (3.239908, 1.43713e-3, 1.16092e-4, -5.77905e-7)
    AW = h[0] + (h[1] + (h[2] + h[3] * T68) * T68) * T68

    k_const = (8.50935e-5, -6.12293e-6, 5.2787e-8)
    BW = k_const[0] + (k_const[1] + k_const[2] * T68) * T68

    e = (19652.21, 148.4206, -2.327105, 1.360477e-2, -5.155288e-5)
    KW = e[0] + (e[1] + (e[2] + (e[3] + e[4] * T68) * T68) * T68) * T68

    j0 = 1.91075e-4
    i_const = (2.2838e-3, -1.0981e-5, -1.6078e-6)
    A = (
        AW
        + (i_const[0] + (i_const[1] + i_const[2] * T68) * T68 + j0 * s**0.5)
        * s
    )

    m = (-9.9348e-7, 2.0816e-8, 9.1697e-10)
    B = BW + (m[0] + (m[1] + m[2] * T68) * T68) * s

    f = (54.6746, -0.603459, 1.09987e-2, -6.1670e-5)
    g = (7.944e-2, 1.6483e-2, -5.3009e-4)
    K0 = (
        KW
        + (
            f[0]
            + (f[1] + (f[2] + f[3] * T68) * T68) * T68
            + (g[0] + (g[1] + g[2] * T68) * T68) * s**0.5
        )
        * s
    )
    return K0 + (A + B * p_atm) * p_atm


@njit(parallel=True, cache=True)
def dens(s, t, p):
    dens_out = np.empty(s.shape[0], dtype=np.float64)
    b = (8.24493e-1, -4.0899e-3, 7.6438e-5, -8.2467e-7, 5.3875e-9)
    c = (-5.72466e-3, 1.0227e-4, -1.6546e-6)
    d = 4.8314e-4

    for i in prange(s.shape[0]):
        T68 = t[i] * 1.00024

        densP0 = (
            _smow_scalar(t[i])
            + (b[0] + (b[1] + (b[2] + (b[3] + b[4] * T68) * T68) * T68) * T68)
            * s[i]
            + (c[0] + (c[1] + c[2] * T68) * T68) * s[i] * s[i] ** 0.5
            + d * s[i] ** 2.0
        )

        K = _seck_scalar(s[i], t[i], p[i])
        p_atm = p[i] / 10.0
        dens_out[i] = densP0 / (1.0 - p_atm / K)
    return dens_out
