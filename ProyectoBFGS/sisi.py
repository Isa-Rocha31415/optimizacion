#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:43:06 2026

@author: isabel
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import imageio.v2 as imageio

def to_float01(img):
    arr = np.asarray(img, dtype=float)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        r, g, b = arr[...,0], arr[...,1], arr[...,2]
        arr = 0.299*r + 0.587*g + 0.114*b
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    else:
        arr = np.zeros_like(arr, dtype=float)
    return arr.astype(float)

def transformar_matriz(matriz, t1, t2, t3, t4, t5, t6):
    alto, ancho = matriz.shape
    cx, cy = ancho / 2.0, alto / 2.0

    det = t1 * t4 - t2 * t3
    if abs(det) < 1e-10:
        return np.zeros_like(matriz)

    inv_t1 = t4 / det
    inv_t2 = -t2 / det
    inv_t3 = -t3 / det
    inv_t4 = t1 / det

    inv_t5 = -(inv_t1 * t5 + inv_t2 * t6)
    inv_t6 = -(inv_t3 * t5 + inv_t4 * t6)

    y, x = np.mgrid[0:alto, 0:ancho]
    x_c = x - cx
    y_c = y - cy

    xs = inv_t1 * x_c + inv_t2 * y_c + cx + inv_t5
    ys = inv_t3 * x_c + inv_t4 * y_c + cy + inv_t6

    return ndimage.map_coordinates(matriz, [ys, xs], order=3, mode='constant', cval=0)

def error_cuadratico(a, b):
    return np.sum((a - b) ** 2) / (2 * a.size)

def funcion_objetivo(params, img_t, img_obj):
    rec = transformar_matriz(img_t, *params)
    return error_cuadratico(rec, img_obj)
