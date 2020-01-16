#!/usr/bin/env python

from distutils.core import setup

setup(name='behavenet',
      version='0.0.2',
      description='Modeling neuro + behavioral data',
      author='Ella Batty',
      author_email='erb2180@columbia.edu',
      url='http://www.github.com/ebatty/behavenet',
      install_requires=['numpy', 'scipy', 'matplotlib'],
      packages=['behavenet', 'tests'],
      )
