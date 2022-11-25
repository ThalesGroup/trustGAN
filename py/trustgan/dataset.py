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

import os
import torch
import numpy as np

from .transforms import min_max_norm


class Dataset(torch.utils.data.Dataset):
    """ """

    def __init__(self, data, label):

        self.data = data
        self.label = label
        self.nb_tot_labels = np.prod(
            [el for i, el in enumerate(self.label.shape) if i != 1]
        )

    def __getitem__(self, i):
        return self.data[i], self.label[i]

    def __len__(self):
        return self.data.shape[0]


class Modifier:
    def __init__(self, nb_channels):

        self.nb_channels = nb_channels

    def one_channel(self, X):

        if (
            (not self.nb_channels is None)
            and (self.nb_channels == 1)
            and (X.shape[1] > 1)
        ):
            idx = torch.randint(0, X.shape[1], (1,))
            X = X[:, idx : idx + 1]
        elif (
            (not self.nb_channels is None)
            and (self.nb_channels > 1)
            and (X.shape[1] == 1)
        ):
            X = torch.cat([X for _ in range(self.nb_channels)], axis=1)

        return X

    def min_max_norm(self, X):

        X = min_max_norm(X)

        return X

    def tanh_centered(self, X):

        X = 2.0 * X - 1.0

        return X

    def __call__(self, x):

        X = x[0]
        y = x[1]

        X = self.one_channel(X)
        X = self.min_max_norm(X)
        X = self.tanh_centered(X)

        return X, y


def get_dataloader(path2dataset, nb_classes, dataset_type, batch_size=64, verbose=True):

    use_cuda = torch.cuda.is_available()
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    X = torch.load(os.path.join(path2dataset, f"X_{dataset_type}.pt"))
    y = torch.load(os.path.join(path2dataset, f"y_{dataset_type}.pt"))

    ###
    X = X.to(torch.float)
    y = y.to(torch.long)

    w = y >= nb_classes
    if w.sum() > 0:
        print("WARNING: more classes than asked for")
        y[w] = 0
    y = torch.nn.functional.one_hot(y, num_classes=nb_classes)
    y = torch.cat([y[..., i][:, None, ...] for i in range(y.shape[-1])], axis=1)

    loader = torch.utils.data.DataLoader(
        Dataset(X, y),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    return loader
