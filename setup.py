#!/usr/bin/env python3
''' Installation script for fmri_designs package '''
from os.path import join as pjoin
from glob import glob

import setuptools

from distutils.core import setup

setup(name='fmri_designs',
      version='0.1',
      description='Design analysis functions for psych-214 class',
      packages=['fmri_designs'],
      license='BSD license',
      author='Matthew Brett',
      author_email='matthew.brett@gmail.com',
      maintainer='Matthew Brett',
      maintainer_email='matthew.brett@gmail.com',
      url='http://github.com/psych-214-fall-2016/fmri-designs',
      package_data = {'fmri_designs': [pjoin('tests', '*')]},
      # Add all the scripts in the scripts directory
      scripts = glob(pjoin('scripts', '*')),
      requires=['numpy (>=1.5.1)'],
      )
