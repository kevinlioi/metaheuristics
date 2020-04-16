from setuptools import setup, Extension, find_packages
from distutils.command.build_ext import build_ext

class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""
    def run(self):

        # Import numpy here, only when headers are needed
        import numpy

        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())

        # Call original build_ext command
        build_ext.run(self)
        
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
    version='1.1.0',
    description='Optimizing analytic functions utilizing a genetic algorithm',
    url='git@github.com:darwynhq/dsor-genopt.git',
    author='Kevin Lioi',
    author_email='kevin@darwynhq.com',
    license='unlicense',
    setup_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        'setuptools>=18.0',
        'cython',
        'numpy',
        'scipy',
        'numba',
        'pandas',
        'matplotlib',
        'line-profiler',
        'networkx'
    ],
    packages=find_packages(),
    ext_modules=cythonised_files,
    cmdclass = {'build_ext': CustomBuildExtCommand},
    install_requires=[
        'numpy',
    ],
    zip_safe=False
)
