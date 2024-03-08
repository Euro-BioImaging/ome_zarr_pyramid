# -*- coding: utf-8 -*-
"""
@author: bugra
"""

import setuptools
from ome_zarr_pyramid.process import image_filters as imfilt
from ome_zarr_pyramid.process.parameter_control import get_functions_with_params

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

func_location = f"ome_zarr_pyramid.bin.runners"
functions = get_functions_with_params(func_location)
scripts = [f"{func_name} = {func_location}:{func_name}" for func_name in functions]

setuptools.setup(
    name = 'ome_zarr_pyramid',
    version = '0.0.2',
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
    entry_points={'console_scripts': scripts
                  }
    )
