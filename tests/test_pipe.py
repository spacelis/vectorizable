# -*- coding: utf-8 -*-
"""
Test pipe.py
"""

from vectorizable.pipe import Pipable

def test_pipable():
    ''' Test Pipable '''
    func1 = lambda x: x + 1
    func2 = lambda x: x * 2
    func3 = lambda x: x ** 2
    func = Pipable(func1).pipe(func2).pipe(func3)
    assert func(4) == 100
