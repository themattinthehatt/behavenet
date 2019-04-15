#!/usr/bin/env python
import numpy as np
from distutils.core import setup
from Cython.Build import cythonize

setup(name='datta',
      version='0.0.1',
      description='Modeling neuro+behavioral data',
      author='Scott Linderman',
      author_email='scott.linderman@gmail.com',
      url='http://www.github.com/slinderman/datta',
      install_requires=['numpy', 'scipy', 'matplotlib'],
      packages=['datta'],
      ext_modules=cythonize('**/*.pyx'),
      include_dirs=[np.get_include(),],
      )
