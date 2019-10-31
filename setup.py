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

if use_cython:
    ext_modules += [
        Extension(name="genetic_functions", sources=["darwyn/genopt/genetic_functions.pyx"]),
    ]
    cythonised_files = cythonize(ext_modules)
else:
    cythonised_files = [
        Extension(name="genetic_functions", sources=["darwyn/genopt/genetic_functions.c"]),
    ]

setup(
    name='darwyn_genopt',
    version='1.0.0',
    description='Optimizing analytic functions utilizing a genetic algorithm',
    url='git@github.com:darwynhq/dsor-genopt.git',
    author='Kevin Lioi',
    author_email='kevin@darwynhq.com',
    license='unlicense',
    packages=['darwyn.genopt'],
    ext_modules=cythonised_files,
    include_dirs=[numpy.get_include()],
    setup_requires=[
        'wheel'
    ],
    install_requires=[
        'numpy',
    ],
    zip_safe=False
)
