# -*- coding: utf-8 -*-
"""
A base class for pipable objects
"""


class Pipable(object):
    """ A class for pipable objects """
    def __init__(self, this, parent=None):
        super(Pipable, self).__init__()
        self.this = this
        self.call = this
        self.parent = parent

    def pipe(self, other):
        ''' Piping the output of this function to the other '''
        return Pipable(lambda x: other(self(x)), self)

    def __call__(self, arg):
        return self.call(arg)

    def __str__(self):  #FIXME not correct displayed
        if self.parent is not None:
            return '{} -> {}'.format(self.parent, self.this)
        return str(self.this)
