# -*- coding: utf-8 -*-

"""
File: vecutil.py
Author: Wen Li
Email: spacelis@gmail.com
Github: http://github.com/spacelis
Description: Utility functions
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided

_LEN = np.vectorize(len)
_SHAPES = np.vectorize(lambda x: np.array(x.shape), signature='()->()', otypes=[np.ndarray])


def bbox_ndarray(ndarr):
    """ Assume the ndarr contains all boundaries of boxes and return the minimum boundaries
        that can put in any of those boxes.
    """
    idx = next(np.ndindex(*ndarr.shape))
    if isinstance(ndarr[idx], np.ndarray):
        bbox = list(ndarr[idx])
        bbox_len = len(bbox)
        for idx in np.ndindex(*ndarr.shape):
            for i in range(bbox_len):
                bbox[i] = max(bbox[i], ndarr[idx][i])
    else:
        bbox = [ndarr.max()]
    return tuple(bbox)


def array_augment(ndarr, func,  # pylint: disable=too-many-locals
                  padding='', with_mask=False, dtype='object'):
    """ Apply the function to each element in the tensor and augment the dimensions with the
        returned list/ndarray.
    """
    farr = ndarr.flatten()
    applied = np.empty(ndarr.size, dtype='object')
    itemsize = applied.dtype.itemsize
    for idx in range(farr.size):
        applied[idx] = np.array(func(farr[idx]))
    elem_shapes = _SHAPES(applied)
    bbox = bbox_ndarray(elem_shapes)
    bbox_size = np.prod(bbox) * itemsize
    bbox_strides = tuple((np.cumprod(bbox[:0:-1])*itemsize)[::-1]) + (itemsize,)

    applied_shape = ndarr.shape + bbox
    applied_strides = tuple(x // ndarr.dtype.itemsize * bbox_size
                            for x in ndarr.strides) + bbox_strides
    unpacked = np.empty((farr.size,) + bbox, dtype=dtype)
    if with_mask:  # pylint: disable=no-else-return
        mask = np.zeros((farr.size,) + bbox, dtype=np.bool_)
        for idx_pre in range(farr.size):
            for idx_suf in np.ndindex(*bbox):
                if np.any(idx_suf >= elem_shapes[idx_pre]):
                    unpacked[(idx_pre,) + idx_suf] = padding
                else:
                    unpacked[(idx_pre,) + idx_suf] = applied[idx_pre][idx_suf]
                    mask[(idx_pre,) + idx_suf] = 1
        unpacked = as_strided(unpacked, shape=applied_shape, strides=applied_strides)
        mask = as_strided(mask, shape=applied_shape,
                          strides=tuple(x // itemsize for x in applied_strides))
        return unpacked, mask
    else:
        for idx_pre in np.ndindex(*applied.shape):
            for idx_suf in np.ndindex(*bbox):
                if np.any(idx_suf >= elem_shapes[idx_pre]):
                    unpacked[idx_pre + idx_suf] = padding
                else:
                    unpacked[idx_pre + idx_suf] = applied[idx_pre][idx_suf]
        unpacked = as_strided(unpacked, shape=applied_shape, strides=applied_strides)
        return unpacked
