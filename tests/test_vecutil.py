# -*- coding: utf-8 -*-
""" Tests for vecutil """

import numpy as np
from numpy.lib.stride_tricks import as_strided
from vectorizable.vecutil import array_augment, bbox_ndarray


def test_bbox_ndarray():
    """ test bounding box searching function """
    dims = np.empty([2, 3], dtype='object')
    for idx in np.ndindex(*dims.shape):
        dims[idx] = np.array([2, 3, 4])
    assert bbox_ndarray(dims) == (2, 3, 4)


def test_array_augment1():
    """ test vectorizable.vecutil.unpack_lists """
    arr = np.array(['a b', 'a b c'])
    uarr = array_augment(arr, lambda x: x.split())
    print(uarr)
    assert uarr.shape == (2, 3)
    assert uarr[0, 0] == 'a'
    assert uarr[0, 2] == ''

    arr = np.array(['Hello World', '你 好 吗!'])
    uarr = array_augment(arr, lambda x: np.array(x.split()))
    print(uarr)
    assert uarr.shape == (2, 3)
    assert uarr[0, 0] == 'Hello'
    assert uarr[0, 2] == ''


def test_array_augment2():
    """ test vectorizable.vecutil.unpack_lists """
    arr = np.array(['a b', 'a b c', 'a b c d'])
    uarr = array_augment(arr, lambda x: np.array([x.split(), x.split()]))
    print(uarr)
    assert uarr.shape == (3, 2, 4)
    assert uarr[0, 0, 0] == 'a'
    assert uarr[0, 1, 0] == 'a'
    assert uarr[0, 0, 3] == ''


def test_array_augment_strided():
    """ test vectorizable.vecutil.unpack_lists """
    arr = np.array(['a b', 'a b c', 'a b c d'], dtype='object')
    sarr = as_strided(arr, shape=(2, 2), strides=(8, 8))
    uarr = array_augment(sarr, lambda x: np.array([x.split(), x.split()]))
    print(uarr)
    assert uarr.shape == (2, 2, 2, 4)
    assert uarr[0, 0, 0, 0] == 'a'
    assert uarr[1, 0, 0, 0] == 'a'
    assert uarr[0, 1, 1, 0] == 'a'
    assert uarr[0, 0, 0, 3] == ''
    assert uarr[1, 0, 0, 3] == ''
    assert uarr[0, 1, 0, 3] == ''


def test_array_augment_mask():
    """ test vectorizable.vecutil.unpack_lists """
    arr = np.array(['a b', 'a b c', 'a b c d'])
    uarr, mask = array_augment(arr, lambda x: np.array([x.split(), x.split()]), with_mask=True)
    print(uarr)
    assert uarr.shape == (3, 2, 4)
    assert uarr[0, 0, 0] == 'a'
    assert uarr[0, 1, 0] == 'a'
    assert uarr[0, 0, 3] == ''
    print(mask)
    assert uarr.shape == (3, 2, 4)
    assert mask[0, 0, 0] == 1
    assert mask[0, 1, 0] == 1
    assert mask[0, 0, 3] == 0


if __name__ == "__main__":
    test_bbox_ndarray()
