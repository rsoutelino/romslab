#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

try:
    from setuptools import setup
    from setuptools.command.sdist import sdist
except ImportError:
    from distutils.core import setup
    from distutils.command.sdist import sdist

try:  # Python 3
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:  # Python 2
    from distutils.command.build_py import build_py

classifiers = """\
Development Status :: Beta
Environment :: Console
Intended Audience :: Science/Research
Intended Audience :: Developers
Intended Audience :: Education
License :: Public License
Operating System :: OS Independent
Programming Language :: Python
Topic :: Scientific/Engineering
Topic :: Education
Topic :: Software Development :: Libraries :: Python Modules
"""

config = dict(name='romslab',
              version='0.2',
              packages=[''],
              license=open('LICENSE.txt').read(),
              description='Module for oceanographic data analysis',
              long_description=open('README.txt').read(),
              author='Rafael Soutelino',
              author_email='rsoutelino@gmail.com',
              maintainer='Rafael Soutelino',
              maintainer_email='rsoutelino@gmail.com',
              url='https://github.com/rsoutelino/romslab/',
              download_url='https://github.com/rsoutelino/romslab/',
              classifiers=filter(None, classifiers.split("\n")),
              platforms='any',
              cmdclass={'build_py': build_py},
              # NOTE: python setup.py sdist --dev
              #cmdclass={'sdist': sdist_hg},
              keywords=['oceanography', 'ocean modeling', 'roms'],
              install_requires=['numpy', 'scipy', 'matplotlib', 'netCDF4', 'seawater']
             )

setup(**config)
