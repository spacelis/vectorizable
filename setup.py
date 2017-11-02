import setuptools
from Cython.Build import cythonize

setuptools.setup(
    name="vectorizable",
    version="0.1.0",
    url="https://github.com/spacelis/vectorizable",

    author="Wen Li",
    author_email="wen.li@ucl.ac.uk",

    description="An collection of helpers for ML data preparation",
    long_description=open('README.rst').read(),

    packages=setuptools.find_packages(),

    extentions = [setuptools.Extension(
        'vectorizable._vecutil',
        ['vectorizable/_vecutil.pyx']
    )],

    install_requires=[
        'numpy',
        'pandas',
        'xarray',
        'nltk'
    ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
