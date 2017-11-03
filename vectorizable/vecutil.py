# -*- coding: utf-8 -*-

"""
File: vecutil.py
Author: Wen Li
Email: spacelis@gmail.com
Github: http://github.com/spacelis
Description: Utility functions
"""

import numpy as np

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


def array_augment(ndarr, func, padding='', with_mask=False, dtype='object'):
    """ Apply the function to each element in the tensor and augment the dimensions with the
        returned list/ndarray.
    """
    applied = np.empty(ndarr.shape, dtype='object')
    for idx in np.ndindex(*ndarr.shape):
        applied[idx] = np.array(func(ndarr[idx]))
    elem_shapes = _SHAPES(applied)
    bbox = bbox_ndarray(elem_shapes)

    unpacked = np.empty(ndarr.shape + bbox, dtype=dtype)
    if with_mask:
        mask = np.zeros(applied.shape + bbox, dtype=np.bool_)
        for idx_pre in np.ndindex(*applied.shape):
            for idx_suf in np.ndindex(*bbox):
                if np.any(idx_suf >= elem_shapes[idx_pre]):
                    unpacked[idx_pre + idx_suf] = padding
                else:
                    unpacked[idx_pre + idx_suf] = applied[idx_pre][idx_suf]
                    mask[idx_pre + idx_suf] = 1
        return unpacked, mask
    else:
        for idx_pre in np.ndindex(*applied.shape):
            for idx_suf in np.ndindex(*bbox):
                if np.any(idx_suf >= elem_shapes[idx_pre]):
                    unpacked[idx_pre + idx_suf] = padding
                else:
                    unpacked[idx_pre + idx_suf] = applied[idx_pre][idx_suf]
        return unpacked
