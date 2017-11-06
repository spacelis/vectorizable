# -*- coding: utf-8 -*-
"""
File: test_vectorization.py
Author: Wen Li
Email: wen.li@ucl.ac.uk
Github: http://github.com/spacelis
Description: Test vectorization
"""

from collections import defaultdict
from itertools import cycle

import numpy as np
import pandas as pd
import xarray as xr

_A = 'This is not a test.'
_B = 'I am a vector'
_C = 'Test'
_D = 'Vector'

def test_digitizer_basic():
    ''' Test Digitizer initialization
    '''
    from vectorizable.preprocessor import Digitizer
    symbols = 'abcd'
    assert Digitizer(symbols=symbols)('abcd') == [1, 2, 3, 4], \
        'Digitizer should be 1 based'
    assert Digitizer(symbols=symbols)('abcde') == [1, 2, 3, 4, 5], \
        'Digitizer should be able to incorporate new elements'
    assert Digitizer(mapping=Digitizer(symbols=symbols).mapping)('abcde') == [1, 2, 3, 4, 5], \
        'Digitizer copy mapping'
    assert Digitizer()('abcde') == [1, 2, 3, 4, 5], \
        'Digitizer can start empty'
    assert len(Digitizer(symbols=symbols)) == 4, \
        'Digitizer can start empty'


def test_digitizer_add():
    ''' Test combining two digitizer
    '''
    from vectorizable.preprocessor import Digitizer
    dzr1 = Digitizer(symbols='abc')
    dzr2 = Digitizer(symbols='def')
    assert (dzr1 + dzr2)('f') == [6], 'Digitizers should combine'


def test_tokenizer():
    ''' Test TokenizerVectorizer
    '''
    from vectorizable.preprocessor import Tokenizer
    dst = pd.DataFrame({'a': [_A, _B],
                        'b': ['Test Vector 1', _D]}).to_xarray()
    tkr = Tokenizer(['a'])
    assert 'a' in tkr(dst)
    assert 'b' not in tkr(dst)
    assert tkr(dst)['a'].sel(index=0, token=0) == 'This'
    assert tkr(dst)['a'].sel(index=1, token=0) == 'I'
    assert tkr(dst)['a'].sel(index=0, token=4) == 'test'
    assert tkr(dst)['a'].sel(index=1, token=4) == ''


def test_charvectorizer():
    ''' Test TokenizerVectorizer
    '''
    from vectorizable.preprocessor import CharVectorizer
    dst = pd.DataFrame({'a': [_A, _B],
                        'b': [_C, _D]}).to_xarray()
    cvr = CharVectorizer(['a'])
    assert cvr(dst)['a'].sel(index=0, char=0) == cvr.digitizer(['T'])[0]
    assert cvr(dst)['a'].sel(index=1, char=0) == cvr.digitizer(['I'])[0]
    assert cvr(dst)['a'].sel(index=0, char=4) == cvr.digitizer([' '])[0]
    assert cvr(dst)['a'].sel(index=1, char=14) == 0


def test_pipable_vectorizer():
    ''' Test piped preprocessors
    '''
    from vectorizable.preprocessor import Tokenizer
    from vectorizable.preprocessor import CharVectorizer
    dst = pd.DataFrame({'a': [_A, _B],
                        'b': [_C, _D]}).to_xarray()
    chv = CharVectorizer(['a'])
    vtr = Tokenizer(['a']).pipe(chv)
    assert vtr(dst)['a'].isel(index=0, token=0, char=0) == chv.digitizer(['T'])[0]
    assert all(vtr(dst)['a'].isel(index=1, token=4) == 0)


def test_vdf_naive_vectorizer():
    ''' test vectorizable dataframe '''
    from vectorizable.preprocessor import VectorizableDataframe
    dst = pd.DataFrame({'a': [_A, _B],
                        'b': [_C, _D]
                       })
    vdf = VectorizableDataframe(dst, ['a'], ['b'])
    ivs, ovs = vdf.all()
    assert ivs['a'].sel(index=0) == _A
    assert ovs['b'].sel(index=1) == _D


def test_vectorizable_dataframe():
    ''' test vectorizable dataframe '''
    from vectorizable.preprocessor import VectorizableDataframe
    from vectorizable.preprocessor import Tokenizer
    from vectorizable.preprocessor import CharVectorizer
    dst = pd.DataFrame({'a': [_A, _B],
                        'b': [_C, _D]
                       })
    chv = CharVectorizer(['a'])
    vtr = Tokenizer(['a']).pipe(chv)
    vdf = VectorizableDataframe(dst, vtr, CharVectorizer(['b'], digitizer=chv.digitizer))
    ivs, ovs = vdf.all()
    assert ivs['a'].sel(index=0, token=0, char=0) == chv.digitizer(['T'])[0]
    assert ovs['b'].sel(index=1, char=2) == chv.digitizer(['c'])[0]


def test_vdf_shuffled():
    ''' test vectorizable dataframe '''
    from vectorizable.preprocessor import VectorizableDataframe
    dst = pd.DataFrame({'a': [_A, _B],
                        'b': [_C, _D]
                       })
    vdf = VectorizableDataframe(dst, ['a'], ['b'])
    ivs, ovs = vdf.all(shuffled=[1, 0])
    assert ivs['a'].isel(index=0) == _B
    assert ovs['b'].isel(index=1) == _C
    assert ivs['a'].sel(index=0) == _A
    assert ovs['b'].sel(index=1) == _D


def test_vdf_minibatch_on_fly():
    ''' test vectorizable dataframe '''
    from vectorizable.preprocessor import VectorizableDataframe
    dst = pd.DataFrame({'a': [_A, _B],
                        'b': [_C, _D]
                       })
    vdf = VectorizableDataframe(dst, ['a'], ['b'])
    for _, (ivs, ovs) in zip(range(10), vdf.minibatches(shuffled=[1, 0], batch_size=1)):
        assert (ivs['a'].isel(index=0) == _B) ^ (ivs.index[0] == 0)
        assert (ovs['b'].isel(index=0) == _D) ^ (ivs.index[0] == 0)


def test_vdf_minibatch():
    ''' test vectorizable dataframe '''
    from vectorizable.preprocessor import VectorizableDataframe
    dst = pd.DataFrame({'a': [_A, _B],
                        'b': [_C, _D]
                       })
    vdf = VectorizableDataframe(dst, ['a'], ['b'])
    mbs = vdf.minibatches(shuffled=True, batch_size=1, vectorize_on_fly=False, with_output=False)
    counter = defaultdict(lambda: 0)
    draws = 100
    for _, idx, ivs in zip(range(draws), cycle([0, 1]), mbs):
        counter[(idx, str(ivs['a'].isel(index=0).data))] += 1
    assert 0 < counter[(0, _A)] < draws // 2
    assert 0 < counter[(1, _A)] < draws // 2
    assert counter[(0, _A)] + counter[(1, _A)] == draws // 2


def test_sliding_window():
    ''' test_sliding_window '''
    from vectorizable.preprocessor import SlidingWindow
    dst = xr.Dataset({'a': (['x', 'y'], np.array(range(12)).reshape(3, 4))},
                     coords={'x': range(3),
                             'y': range(4)})
    slw = SlidingWindow(['a'], padding=0, win_widths=(2, 2))
    sds = slw(dst)
    print(sds['a'])
    assert np.allclose(sds['a'][0, 0, 0, :], [0, 1])
    assert np.allclose(sds['a'][0, 0, :, 1], [1, 5])
    assert np.allclose(sds['a'][1, 1, :, 0], [5, 9])


def test_sliding_window_2():
    ''' test_sliding_window '''
    from vectorizable.preprocessor import SlidingWindow
    dst = xr.Dataset({'a': (['x', 'y'], np.array(range(12)).reshape(3, 4))},
                     coords={'x': range(3),
                             'y': range(4)})
    slw = SlidingWindow(['a'], padding=0, output_widths=(2, 2))
    sds = slw(dst)
    print(sds['a'])
    assert np.allclose(sds['a'][0, 0, 0, :], [0, 1, 2])
    assert np.allclose(sds['a'][0, 0, :, 1], [1, 5])
    assert np.allclose(sds['a'][1, 1, :, 0], [5, 9])
