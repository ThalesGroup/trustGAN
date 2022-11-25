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

import torch
import numpy as np

from .transforms import min_max_norm
from .waveunet import WaveUnet


class resNetUnit(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        dim="2d",
        weight_norm=True,
        batch_norm=False,
        isGan=False,
    ):
        super(resNetUnit, self).__init__()

        if dim == "1d":
            conv = torch.nn.Conv1d
            batchnorm = torch.nn.BatchNorm1d
        elif dim == "2d":
            conv = torch.nn.Conv2d
            batchnorm = torch.nn.BatchNorm2d

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv1 = conv(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        )
        torch.nn.init.constant_(self.conv1.bias, 0.0)

        self.conv2 = conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            padding_mode="replicate",
        )  ## padding_mode='replicate' seem very important for GAN
        torch.nn.init.constant_(self.conv2.bias, 0.0)

        self.conv3 = conv(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            padding_mode="replicate",
        )  ## padding_mode='replicate' seem very important for GAN
        torch.nn.init.constant_(self.conv3.bias, 0.0)

        ### Helps a lot the GAN
        if isGan:
            self.relu = torch.nn.LeakyReLU()
        else:
            self.relu = torch.nn.ReLU()

        if weight_norm:
            self.conv2 = torch.nn.utils.weight_norm(self.conv2)
            self.conv3 = torch.nn.utils.weight_norm(self.conv3)

        if batch_norm:
            self.batch_norm = batchnorm(num_features=out_channels)
        else:
            self.batch_norm = None

        self.scale = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):

        res = self.conv2(x)
        res = self.relu(res)

        res = self.conv3(res)
        res = self.scale * res

        x = self.conv1(x)
        x = x + res

        if not self.batch_norm is None:
            x = self.batch_norm(x)

        x = self.relu(x)

        return x


class Net(torch.nn.Module):
    def __init__(
        self,
        nb_classes,
        nb_channels,
        kernel_size=3,
        fcl=64,
        residualUnits=[1, 2, 4, 8, 16, 32, 64],
        weight_norm=True,
        batch_norm=False,
        dilation_coef=2,
        dim="2d",
    ):
        super(Net, self).__init__()

        ###
        self.nb_dims = int(dim[:-1])
        print(f"INFO: nb_dims = {self.nb_dims}")
        if dim == "1d":
            conv = torch.nn.Conv1d
        elif dim == "2d":
            conv = torch.nn.Conv2d

        ###
        chs = np.array(residualUnits)
        chs = chs[chs > nb_channels]
        chs = np.append([nb_channels], chs)
        print(f"INFO: Using these different chanel steps {chs}")

        self.layers = torch.nn.ModuleList(
            [
                resNetUnit(
                    in_channels=chs[i],
                    out_channels=chs[i + 1],
                    kernel_size=kernel_size,
                    dilation=dilation_coef**i,
                    weight_norm=weight_norm,
                    batch_norm=batch_norm,
                    dim=dim,
                )
                for i in range(len(chs) - 1)
            ]
        )

        ### This convolution is very important. Without the net does not learn
        self.conv1 = conv(in_channels=chs[-1], out_channels=chs[-1], kernel_size=1)

        self.lin_00 = torch.nn.Linear(in_features=chs[-1], out_features=fcl)
        self.lin_01 = torch.nn.Linear(in_features=fcl, out_features=nb_classes)

        self.relu = torch.nn.ReLU()
        self.drp = torch.nn.Dropout(0.1)

    def forward(self, x):

        for unit in self.layers:
            x = unit(x)
        x = self.conv1(x)

        for _ in range(self.nb_dims):
            x, _ = torch.max(x, dim=-1)

        x = self.drp(self.relu(self.lin_00(x)))
        x = self.lin_01(x)

        return x


class Gan(torch.nn.Module):
    def __init__(
        self,
        nb_channels,
        kernel_size=3,
        residualUnits=[1, 2, 4, 8, 16],
        weight_norm=True,
        batch_norm=False,
        dilation_coef=2,
        dim="2d",
    ):
        super(Gan, self).__init__()

        ###
        if dim == "1d":
            conv = torch.nn.Conv1d
        elif dim == "2d":
            conv = torch.nn.Conv2d

        ###
        chs = np.array(residualUnits)
        chs = chs[chs > nb_channels]
        chs = np.append([nb_channels], chs)
        print(f"INFO: Using these different channel steps {chs}")

        self.layers = torch.nn.ModuleList(
            [
                resNetUnit(
                    in_channels=chs[i],
                    out_channels=chs[i + 1],
                    kernel_size=kernel_size,
                    dilation=dilation_coef**i,
                    weight_norm=weight_norm,
                    batch_norm=batch_norm,
                    isGan=True,
                    dim=dim,
                )
                for i in range(len(chs) - 1)
            ]
        )

        self.conv1 = conv(in_channels=chs[-1], out_channels=chs[0], kernel_size=1)

    def forward(self, x):

        for unit in self.layers:
            x = unit(x)

        x = self.conv1(x)
        x = torch.tanh(x)

        return x


class LModCNNResNetRelu(torch.nn.Module):
    def __init__(self, nb_classes, kernel_size=7, **kwargs):
        super(LModCNNResNetRelu, self).__init__()

        self.nb_classes = nb_classes

        nbc = 8
        self.conv00 = torch.nn.Conv1d(in_channels=2, out_channels=nbc, kernel_size=1)
        self.conv01 = torch.nn.Conv1d(
            in_channels=nbc,
            out_channels=nbc,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.conv02 = torch.nn.Conv1d(
            in_channels=nbc,
            out_channels=nbc,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        nbc = 16
        self.conv10 = torch.nn.Conv1d(
            in_channels=nbc // 2, out_channels=nbc, kernel_size=1
        )
        self.conv11 = torch.nn.Conv1d(
            in_channels=nbc,
            out_channels=nbc,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.conv12 = torch.nn.Conv1d(
            in_channels=nbc,
            out_channels=nbc,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        nbc = 32
        self.conv20 = torch.nn.Conv1d(
            in_channels=nbc // 2, out_channels=nbc, kernel_size=1
        )
        self.conv21 = torch.nn.Conv1d(
            in_channels=nbc,
            out_channels=nbc,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.conv22 = torch.nn.Conv1d(
            in_channels=nbc,
            out_channels=nbc,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        nbc = 64
        self.conv30 = torch.nn.Conv1d(
            in_channels=nbc // 2, out_channels=nbc, kernel_size=1
        )
        self.conv31 = torch.nn.Conv1d(
            in_channels=nbc,
            out_channels=nbc,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.conv32 = torch.nn.Conv1d(
            in_channels=nbc,
            out_channels=nbc,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.lin_1 = torch.nn.Linear(in_features=nbc, out_features=256)
        self.lin_2 = torch.nn.Linear(in_features=256, out_features=self.nb_classes)

        self.drp1 = torch.nn.Dropout()

        self.relu = torch.nn.ReLU()

    def forward(self, x):

        x = self.relu(self.conv00(x))
        y = x.clone()
        x = self.relu(self.conv01(x))
        x = self.conv02(x)
        x = self.relu(x + y)

        x = self.relu(self.conv10(x))
        y = x.clone()
        x = self.relu(self.conv11(x))
        x = self.conv12(x)
        x = self.relu(x + y)

        x = self.relu(self.conv20(x))
        y = x.clone()
        x = self.relu(self.conv21(x))
        x = self.conv22(x)
        x = self.relu(x + y)

        x = self.relu(self.conv30(x))
        y = x.clone()
        x = self.relu(self.conv31(x))
        x = self.conv32(x)
        x = self.relu(x + y)

        x = x.mean(axis=-1)

        x = self.relu(self.lin_1(x))
        x = self.drp1(x)

        x = self.lin_2(x)

        return x
