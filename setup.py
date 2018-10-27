#!/usr/bin/env python
import distutils
from distutils.core import setup

description = "BMX raw file support"

setup(name="bmxdaq", 
      version="0.5.0",
      description=description,
      url="https://github.com/slosar/bmxdaq",
      author="Anze Slosar",
      py_modules=['bmxdata'],
      package_dir={'': 'py'})


