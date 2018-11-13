#!/usr/bin/env python
__usage__ = "setup.py command [--options]"
__description__ = "standard install script"
__author__ = "Reed Essick (reed.essick@ligo.org)"

#-------------------------------------------------

from setuptools import (setup, find_packages)
import glob

setup(
    name = 'universality',
    version = '0.0',
    url = 'https://github.com/reedessick/universality',
    author = __author__,
    author_email = 'reed.essick@ligo.org',
    description = __description__,
    license = 'MIT',
    scripts = glob.glob('bin/*'),
    packages = ['universality'],
    data_files = [],
    package_data = {'universality.eos':['*.csv']},
    requires = [],
)
