import os
import re
import setuptools

PKG_NAME = "toolbox"

HERE = os.path.abspath(os.path.dirname(__file__))

PATTERN = r'^{target}\s*=\s*([\'"])(.+)\1$'

AUTHOR = re.compile(PATTERN.format(target='__author__'), re.M)
VERSION = re.compile(PATTERN.format(target='__version__'), re.M)
LICENSE = re.compile(PATTERN.format(target='__license__'), re.M)
AUTHOR_EMAIL = re.compile(PATTERN.format(target='__author_email__'), re.M)


def parse_init():
    with open(os.path.join(HERE, PKG_NAME, '__init__.py')) as f:
        file_data = f.read()
    return [regex.search(file_data).group(2) for regex in
            (AUTHOR, VERSION, LICENSE, AUTHOR_EMAIL)]


with open("README.md", "r") as fh:
    long_description = fh.read()

author, version, license, author_email = parse_init()

setuptools.setup(
    name=PKG_NAME,
    author=author,
    author_email=author_email,
    license=license,
    version=version,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AMergy/intro2torchgeo",
    # packages         = ['dMRV_CH4Rice']+setuptools.find_namespace_packages(),
    packages=setuptools.find_packages(include=['toolbox',
                                               'toolbox.*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=[
        "torch>=1.12.1",
        "torchgeo==0.3.1",
    ],
    include_package_data=True,
)
