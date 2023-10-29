from functools import partial
from math import cos, sin, sqrt, pi, ceil

import numpy as np
from numpy.linalg import norm
import scipy.optimize as so

import sys, os, trace, multiprocessing, time
from multiprocessing import Pool, Lock
import numpy as np
import pandas as pd
from numpy.linalg import norm
import pickle, urllib
from functools import partial
import calculate_orbits as co
Z_AXIS = np.array([0.0, 0.0, 1.0])

EARTH_A = 1.00000261
EARTH_E = 0.01671123
EARTH_I = 0.0
EARTH_W = np.radians(114.20783)
EARTH_OM = np.radians(348.73936)

def _orbpoint_flat(t, a, e): 
    r = _get_r(t, a, e)
    x = r * cos(t)
    y = r * sin(t)
    return [x, y, 0]

def _get_r(t, a, e):
    r = a*(1 - e**2)/(1 + e*cos(t))
    return r

def _rotate(point, ax, angle):
    rot = _rotmatrix(ax, angle)
    return np.dot(point, rot)

def _rotmatrix(ax, angle):
    cosa = cos(angle)
    sina = sin(angle)
    x, y, z = ax
    rot = np.array([[cosa + (x**2)*(1 - cosa), x*y*(1 - cosa) - z*sina, x*z*(1 - cosa) + y*sina],
                    [y*x*(1 - cosa) + z*sina, cosa + (y**2)*(1 - cosa), y*z*(1 - cosa) - x*sina],
                    [z*x*(1 - cosa) - y*sina, z*y*(1 - cosa) + x*sina, cosa + (z**2)*(1 - cosa)]])
    return rot

def get_orbpoint_direct(t, a, e, w, i, om):
    r = _get_r(t, a, e)
    x = r * (cos(om) * cos(t + w) - sin(om) * sin(t + w) * cos(i))
    y = r * (sin(om) * cos(t + w) + cos(om) * sin(t + w) * cos(i))
    z = r * (sin(t + w) * sin(i))
    point = np.array([x, y, z])
    return point

def get_orbpoint_rotation(t, a, e, w, i, om):
    point = _orbpoint_flat(t, a, e)            # point in orbital plane:
    axis_w = np.array([cos(-w), sin(-w), 0.0])
    point_inc = _rotate(point, axis_w, i)      # get inclined point:
    wb = om + w
    point_hc = _rotate(point_inc, Z_AXIS, wb)  # point in heliocentric coords:
    return point_hc

def get_orbpoint(t, a, e, w, i, om, method='direct'):
    if method == 'direct':
        return get_orbpoint_direct(t, a, e, w, i, om)
    elif method == 'rotation':
        return get_orbpoint_rotation(t, a, e, w, i, om)
    else:
        raise AttributeError('method "%s" is not specified.' % method)
    
def get_orbpoint_earth(t, method='direct'):
    return get_orbpoint(t, EARTH_A, EARTH_E, EARTH_W, EARTH_I, EARTH_OM, method=method)

def _find_dist(t, a, e, w, i, om):
    asteroid_point = get_orbpoint(t[0], a, e, w, i, om)
    earth_point = get_orbpoint_earth(t[1])
    dist = norm(asteroid_point - earth_point)
    return dist

def get_moid(a, e, w, i, om):
    "Returns Minimal Earth Orbit Intersection Distance"
    ta0 = [(w - pi*0.5), (w - pi*0.5), (w + pi*0.5), (w + pi*0.5)]
    te0 = [(om - pi*0.5), (om + pi*0.5), (om + pi*0.5), (om - pi*0.5)]
    moid_min = 5.45492 # Jupiter aphelion
    for ta, te in zip(ta0, te0):
        ta_te_min = so.fmin(partial(_find_dist, a=a, e=e, w=w, i=i, om=om), [ta, te], disp=False)
        moid = _find_dist(ta_te_min, a, e, w, i, om)
        moid_min = min(moid_min, moid)
    return moid_min

# def append_moid(ir):
#     index, row = ir
#     w_, i_, om_ = np.radians([row.w, row.i, row.om])
#     moid = co.get_moid(row.a, row.e, w_, i_, om_)
#     # data.set_value(index, 'moid', moid)
#     return (index, moid)

# def calc_moid(data):
#     """append column with values of moid"""
#     corenums = multiprocessing.cpu_count()
#     if corenums > 1:
#         corenums -= 1
#     pool = Pool(processes=corenums)
#     joblist = [(index, row) for index, row in data.iterrows()]
#     # pool.map(partial(append_moid, data=data), joblist)
#     mapresult = pool.map_async(append_moid, joblist)
#     pool.close()
#     pool.join()
#     for index, moid in mapresult.get():
#         data.set_value(index, 'moid', moid)
