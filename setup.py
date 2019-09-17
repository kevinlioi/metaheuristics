from setuptools import setup, Extension
import numpy

try:
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True

cythonised_files = {}
ext_modules = []
include_dirs = []

if use_cython:
    ext_modules += [
        Extension(name="genetic_functions", sources=["genopt/genetic_functions.pyx"]),
    ]
    include_dirs += [
        numpy.get_include()
    ]
    cythonised_files = cythonize(ext_modules)
else:
    cythonised_files += [
        Extension(name="genetic_functions", sources=["genopt/genetic_functions.c"]),
    ]

setup(
    name='genopt',
    version='1.0.0',
    description='Optimizing analytic functions utilizing a genetic algorithm',
    url='git@github.com:darwynhq/dsor-genopt.git',
    author='Kevin Lioi',
    author_email='kevin@darwynhq.com',
    license='unlicense',
    packages=['genopt'],
    ext_modules=cythonised_files,
    include_dirs=include_dirs,
    install_requires=[
        'numpy',
    ],
    zip_safe=False
)