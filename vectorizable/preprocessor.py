#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: preprocessor.py
Author: Wen Li
Email: spacelis@gmail.com
Github: http://github.com/spacelis
Description:
    This module provides common used preprocessors.
"""

import string
from itertools import count
from collections import defaultdict
import numpy as np
import pandas as pd
import xarray as xr
from nltk.tokenize import RegexpTokenizer

from .pipe import Pipable
from .vecutil import array_augment

SIMPLE_TOKENIZER = RegexpTokenizer('[a-zA-Z]+|[0-9]+|-|/')

# pylint: disable=too-few-public-methods,invalid-name,too-many-arguments


def selective_apply(ds, variables, func, args, kwargs):
    """ Apply the func on a selected set of variables """
    return xr.Dataset({v: ds[v] if v not in variables else func(ds[v], *args, **kwargs)
                       for v in ds},
                      coords=ds.coords,
                      attrs=ds.attrs
                     )


def get_samples(ds, idx_name='index', perm=None):
    """ Return a shuffled dataset """
    if perm is None:
        return ds
    elif perm is True:
        perm = np.random.permutation(len(ds[idx_name]))
    return ds.reindex(**{idx_name: perm})


class Digitizer(object):
    """ To digitize the sequence """

    ALPHANUM = string.ascii_lowercase + string.ascii_uppercase + string.digits
    ALPHANUMSEP = ALPHANUM + '-/'

    def __init__(self, symbols=None, skip=True, mapping=None, na=''):
        """ __init__ """
        super(Digitizer, self).__init__()
        self.skip = skip
        self.na = na
        if mapping is not None:
            self.mapping = mapping
        elif symbols is not None:
            self.mapping = defaultdict(
                count(len(symbols) + 1).__next__,
                {ch: idx for idx, ch in enumerate(symbols, 1)})
        else:
            self.mapping = defaultdict(count(1).__next__)

    def __add__(self, other):
        new_mapping = defaultdict(count(len(self.mapping)+1).__next__, self.mapping)
        for k in other.mapping.keys():
            _ = new_mapping[k]
        return Digitizer(mapping=new_mapping)

    def __len__(self):
        return len(self.mapping)

    def __call__(self, seq):
        return [0 if c == self.na else self.mapping[c] for c in seq]


class FreezedDigitizer(Digitizer):
    """ Do not expand mapping by symbols that has not been seen before. """

    def __call__(self, seq):
        return [0 if c not in self.mapping[c] else self.mapping[c] for c in seq]



class VectorizableDataframe(object):
    """ An object for storing and handling dataset operations"""
    def __init__(self, ds, input_vectorizer=None, output_vectorizer=None):
        super(VectorizableDataframe, self).__init__()
        if isinstance(ds, pd.DataFrame):
            self.ds = ds.to_xarray()
        else:
            self.ds = ds
        if isinstance(input_vectorizer, (tuple, list)):
            self.input_vectorizer = lambda ds: ds[input_vectorizer]
        else:
            self.input_vectorizer = input_vectorizer
        if isinstance(input_vectorizer, (tuple, list)):
            self.output_vectorizer = lambda ds: ds[output_vectorizer]
        else:
            self.output_vectorizer = output_vectorizer

    def all(self, with_output=True, shuffled=None):
        """ Return all train records processed
        """
        samples = get_samples(self.ds, 'index', shuffled)

        if with_output:
            if self.output_vectorizer is None:
                raise ValueError('No output veictorizer')
            i_vecs = self.input_vectorizer(samples)
            o_vecs = self.output_vectorizer(samples)
            return i_vecs, o_vecs
        return self.input_vectorizer(samples)


    def minibatches(self, batch_size=100, with_output=True, shuffled=False, vectorize_on_fly=True):
        """ Return a mini batch of the data """
        while True:
            samples = get_samples(self.ds, 'index', shuffled)

            if vectorize_on_fly:
                if with_output:
                    for i in range(0, len(samples['index']), batch_size):
                        mb_samples = samples.isel(index=slice(i, i+batch_size))
                        i_vecs = self.input_vectorizer(mb_samples)
                        o_vecs = self.output_vectorizer(mb_samples)
                        yield i_vecs, o_vecs
                else:
                    for i in range(0, len(samples['index']), batch_size):
                        mb_samples = samples.isel(index=slice(i, i+batch_size))
                        yield self.input_vectorizer(mb_samples)
            else:
                input_vecs = self.input_vectorizer(samples)
                if with_output:
                    output_vecs = self.output_vectorizer(samples)
                    for i in range(0, len(samples['index']), batch_size):
                        print(i, i+batch_size, samples, batch_size)
                        i_vecs = input_vecs.isel(index=slice(i, i+batch_size))
                        o_vecs = output_vecs.isel(index=slice(i, i+batch_size))
                        yield i_vecs, o_vecs
                else:
                    for i in range(0, len(samples['index']), batch_size):
                        yield input_vecs.isel(index=slice(i, i+batch_size))


class Preprocessor(object):
    """ Base clase for vectorizer """
    def __init__(self, variables=None):
        super(Preprocessor, self).__init__()
        self.variables = variables if variables is not None else []

    def __call__(self, ds):
        """ Processing a dataframe """
        raise NotImplementedError()

    def pipe(self, other):
        """ Piping this preprocessor's output to anther preprocessor """
        return Pipable(self).pipe(other)


class Tokenizer(Preprocessor):
    """ Tokenizer """
    def __init__(self, variables, tokenizer=SIMPLE_TOKENIZER):
        super(Tokenizer, self).__init__(variables)
        self.tokenize = tokenizer.tokenize

    def _tokenize(self, arr):
        return xr.DataArray(array_augment(arr.data, self.tokenize), dims=[*arr.dims, 'token'])

    def __call__(self, ds):
        """ Tokenize the elements """
        return ds[self.variables].apply(self._tokenize)


class CharVectorizer(Preprocessor):
    """ Make a vector from a word sequence"""
    def __init__(self, variables, digitizer=None):
        super(CharVectorizer, self).__init__(variables)
        if digitizer is None:
            digitizer = Digitizer(Digitizer.ALPHANUM)
        self.digitizer = digitizer

    def __call__(self, ds):
        """ Vectorize a set of token seqs

        :seqs: a list of token seqs (string lists)
        :returns: a tensor of encoded seqs [batch_size, max(seq_size)]

        """
        return ds[self.variables].apply(
            lambda arr: xr.DataArray(array_augment(arr.data, self.digitizer, padding=0, dtype=int),
                                     dims=[*arr.dims, 'char'])
        )

    def __len__(self):
        return len(self.digitizer)
