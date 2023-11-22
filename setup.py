# -*- coding: utf-8 -*-
"""
@author: bugra
"""

import setuptools

# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()


def parse_requirements(filename):
    with open(filename, encoding='utf-8') as fid:
        requires = [line.strip() for line in fid.readlines() if line]
    return requires

def readme():
   with open('README.txt') as f:
       return f.read()

requirements = parse_requirements('requirements.txt')

setuptools.setup(
    name = 'OME_Zarr',
    version = '0.0.1',
    author = 'Bugra Özdemir',
    author_email = 'bugraa.ozdemir@gmail.com',
    description = 'A package for reading, writing and processing OME-Zarr datasets',
    long_description = readme(),
    long_description_content_type = "text/markdown",
    url = 'https://github.com/Euro-BioImaging/OME-Zarr',
    # license = 'MIT',
    packages = setuptools.find_packages(),
    include_package_data=True,
    install_requires = requirements,
    entry_points={'console_scripts':
                      [
                          "apply_projection = src.bin.ome_zarr_run:apply_projection",
                      ]
                  }
    )
