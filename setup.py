#!/usr/bin/env python

import os
import re
from setuptools import setup

# parse version from init.py
with open("traversome/__init__.py") as init:
    CUR_VERSION = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]",
        init.read(),
        re.M,
    ).group(1)


# nasty workaround for RTD low memory limits
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    install_requires = []
else:
    install_requires = [
        "numpy",
        "pandas",
        "scipy",
    	"pymc3",
        "sympy",
    ]


# setup installation
setup(
    name="traversome",
    packages=["traversome"],
    version=CUR_VERSION,
    author="Jianjun Jin",
    author_email="...",
    install_requires=install_requires,
    entry_points={
        'console_scripts': ['traversome = traversome.__main__:main']},
    license='GPL',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
    ],
)
