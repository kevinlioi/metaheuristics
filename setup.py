from setuptools import setup

setup(
    name='genopt',
    version='1.0.0',
    description='Optimizing analytic functions utilizing a genetic algorithm',
    url='git@github.com:darwynhq/dsor-genopt.git',
    author='Kevin Lioi',
    author_email='kevin@darwynhq.com',
    license='unlicense',
    packages=['genopt'],
    install_requires=[
        'numpy',
        'Cython',
    ],
    zip_safe=False
)