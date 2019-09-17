from setuptools import setup, Extension

from Cython.Build import cythonize
import numpy

setup(
    name='genopt',
    version='1.0.0',
    description='Optimizing analytic functions utilizing a genetic algorithm',
    url='git@github.com:darwynhq/dsor-genopt.git',
    author='Kevin Lioi',
    author_email='kevin@darwynhq.com',
    license='unlicense',
    packages=['genopt'],
    ext_modules=cythonize(Extension(name="genetic_functions", sources=["genopt/*.pyx"])),
    include_dirs=[numpy.get_include()],
    install_requires=[
        'numpy',
        'Cython',
    ],
    zip_safe=False
)