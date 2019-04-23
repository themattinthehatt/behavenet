#!/usr/bin/env python
import numpy as np
from distutils.core import setup
# from Cython.Build import cythonize

setup(name='behavenet',
      version='0.0.1',
      description='Modeling neuro + behavioral data',
      author='Ella Batty',
      author_email='erb2180@columbia.edu',
      url='http://www.github.com/ebatty/behavenet',
      install_requires=['numpy', 'scipy', 'matplotlib'],
      packages=['analyses', 'data', 'behavenet', 'tests'],
      # ext_modules=cythonize('**/*.pyx'),
      # include_dirs=[np.get_include(),],
      )
