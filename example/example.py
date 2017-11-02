#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: example.py
Description: Examples of using vectorizable.
"""

import pandas as pd
from vectorizable.preprocessor import CharVectorizer, Tokenizer

def piped_vectorizer():
    """ Preprocessing a dataframe and make it into a 3D structure. """
    dst = pd.DataFrame({'a': ['This is an example of vectorizable.', 'I am a xarray.'],
                        'b': ['This is short.', 'Even shorter.']}).to_xarray()
    chv = CharVectorizer(['a'])
    vtr = Tokenizer(['a']).pipe(chv)
    assert vtr(dst)['a'].isel(index=0, token=0, char=0) == chv.digitizer(['T'])[0]
    assert all(vtr(dst)['a'].isel(index=1, token=4) == 0)
