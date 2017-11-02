vectorizable
============

An collection of preprocessors for Machine Learning.

Data processing can be boilerplate sometime. This project aims to provide primary tools for simplify the process of making non-numerical data available for feeding to ML models.
The preprocessors are designed around [xarray](http://xarray.pydata.org/en/stable/), which is a multi-dimensional alternatives to [pandas](http://pandas.pydata.org/).
The idea is to expand the dimensions of a tensor via a function that takes a value and spits out a list of values.
For example, given a list of sentences, we want to make it into a 3d tensor in a shape of _sentences_ x _tokens_ x _chars_ with proper padding (usually 0).

The preprocessors can also be connected into a pipeline for easy preprocessing composition.


Usage
-----
See the examples.


Licence
-------

MIT License.


Authors
-------

`vectorizable` was written by `Wen Li <wen.li@ucl.ac.uk>`_.
