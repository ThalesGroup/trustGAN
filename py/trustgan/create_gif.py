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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
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

import os
import imageio
import glob
import numpy as np

# https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python


def create_gif(root, pattern):

    """ """

    filenames = np.sort(glob.glob(os.path.join(root, "plots", pattern)))
    idxs = np.array(
        [
            el.replace(
                os.path.join(
                    root, "plots", pattern.replace("*", "").replace(".png", "")
                ),
                "",
            ).replace(".png", "")
            for el in filenames
        ]
    ).astype(int)
    w = np.argsort(idxs)
    filenames = filenames[w]

    print(f"Create gifs, {len(filenames)} files found for {pattern}")
    if len(filenames) == 0:
        return

    images = [imageio.imread(filename) for filename in filenames]

    imageio.mimsave(
        "{}/gifs/{}.gif".format(root, pattern.replace("*", "").replace(".png", "")),
        images,
        loop=1,
        duration=0.01,
    )

    return
