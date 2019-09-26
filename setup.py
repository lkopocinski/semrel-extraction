#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup

setup(
    name='relextr',
    version='0.1.1',
    author='Arkadiusz Janz',
    author_email='arkadiusz.janz@pwr.edu.pl',
    maintainer='Łukasz Kopociński',
    maintainer_email='lukasz.kopocinski@pwr.edu.pl',
    packages=[
        'relextr',
        'relextr.base',
        'relextr.model',
        'relextr.utils'
    ],
    zip_safe=False
)
