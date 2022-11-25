#!/usr/bin/env python
# Authors:
#   Helion du Mas des Bourboux <helion.dumasdesbourboux'at'thalesgroup.com>
#
# MIT License
#
# Copyright (c) 2022 THALES
#   All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# 2022 october 21

import setuptools
import glob

DISTNAME = "trustgan"
DESCRIPTION = "TurstGAN"
LONG_DESCRIPTION = open("README.md").read()
MAINTAINER = "Helion du Mas des Bourboux"
MAINTAINER_EMAIL = "helion.dumasdesbourboux'at'thalesgroup.com"
URL = "https://github.com/ThalesGroup/trustGAN"
LICENSE = "MIT License"

with open("py/trustgan/_version.py") as f:
    VERSION = f.read().splitlines()[-1].split("=")[-1].replace('"', "").strip()

with open("requirements.txt") as f:
    REQUIRED = f.read().splitlines()

SCRIPTS = glob.glob("bin/trustgan-*")

setuptools.setup(
    name=DISTNAME,
    author=MAINTAINER,
    author_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    install_requires=REQUIRED,
    packages=[
        "trustgan",
    ],
    package_dir={
        "trustgan": "py/trustgan/",
    },
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    scripts=SCRIPTS,
    include_package_data=True,
    package_data={"": ["*.png", "*.ico", "*.css", "*.jpg"]},
)
