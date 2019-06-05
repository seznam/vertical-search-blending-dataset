#!/usr/bin/env python

from setuptools import setup, find_packages

install_requires = []
dependency_links = []

with open('requirements.txt') as f:
    for line in f:
        line = line.rstrip('\n')
        if line.startswith('-e '):
            dependency_links.append(line)
        else:
            install_requires.append(line)

setup(
    name="vsbd",
    version="1.0",
    url="https://github.com/seznam/vertical-search-blending-dataset",
    author="Seznam.cz research team",
    author_email="srch.vyzkum@firma.seznam.cz",
    description="Python package containing methods and evaluations for Seznam.cz vertical search blending dataset",
    long_description="",
    packages=find_packages(),
    install_requires=install_requires,
    dependency_links=dependency_links,
    include_package_data=True,
)
