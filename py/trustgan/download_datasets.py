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
import sys
import numpy as np
import tempfile
import torch
import torchvision
import torchvision.transforms
import h5py
import inspect


class Dataset:
    def __init__(self, dataset, path2save, splits=[0.7, 0.15, None], seed=42):

        ###
        self.splits = self.get_splits(splits)
        self.dataset = dataset
        self.path2save = path2save

        np.random.seed(seed)

        ###
        data = self.get_data()
        self.save_data(data)

    def get_splits(self, splits):

        if len(splits) != 3:
            print("ERROR: len(splits) != 3")
            sys.exit()
        if splits[-1] is None:
            splits[-1] = 1.0 - np.sum(splits[:-1])
        splits = np.array(splits)

        return splits

    def get_data(self):

        if hasattr(torchvision.datasets, self.dataset):
            data = self.get_torch_dataset()
        elif self.dataset == "AugMod":
            data = self.get_augmod_dataset()
        elif self.dataset == "RML2016.04C":
            data = self.get_rml_dataset()
        else:
            print(f"ERROR: did not find the dataset {self.dataset}")
            raise AttributeError

        for k, v in data.items():
            uniqs = torch.unique(v["y"]).size()
            print(
                f"INFO: {k} has X shape = {v['X'].shape} and y shape = {v['y'].shape}, nb uniques = {uniqs}"
            )

        return data

    def get_torch_dataset(self):

        data = {}

        ### Train and Valid
        trainvalidset = self.get_torch_dataset_type(True)

        splits = self.splits[:2] / self.splits[:2].sum()
        nb_samples = trainvalidset["y"].shape[0]
        w = np.arange(nb_samples)
        np.random.shuffle(w)
        trainvalidset["X"] = trainvalidset["X"][w]
        trainvalidset["y"] = trainvalidset["y"][w]

        nb_split = int(splits[0] * nb_samples)
        data["train"] = {
            "X": trainvalidset["X"][:nb_split],
            "y": trainvalidset["y"][:nb_split],
        }
        data["valid"] = {
            "X": trainvalidset["X"][nb_split:],
            "y": trainvalidset["y"][nb_split:],
        }

        ### Test
        testset = self.get_torch_dataset_type(False)
        data["test"] = testset

        ### New splits
        print("INFO: previous splits:", self.splits)
        self.splits = np.array(
            [data[el]["y"].shape[0] for el in ["train", "valid", "test"]], dtype=float
        )
        self.splits /= self.splits.sum()
        print("INFO: new splits:", self.splits)

        return data

    def get_torch_dataset_type(self, dataset_type):
        with tempfile.TemporaryDirectory() as tmpdirname:

            if self.dataset in ["OxfordIIITPet"]:
                dataset_type = "test" if dataset_type else "trainval"
                kwargs = {"target_types": "segmentation"}
            else:
                kwargs = {}
            data = (
                getattr(torchvision.datasets, self.dataset)(
                    tmpdirname,
                    dataset_type,
                    download=True,
                    transform=torchvision.transforms.Compose(
                        [
                            torchvision.transforms.ToTensor(),
                        ]
                    ),
                    **kwargs,
                ),
            )

            if hasattr(data[0], "data") and hasattr(data[0], "targets"):
                X = data[0].data
                y = data[0].targets
            else:
                X, y = self.get_data_from_loaders(data[0])

            if "numpy" in str(type(X)):
                X = torch.from_numpy(X)
            if "numpy" in str(type(y)):
                y = torch.from_numpy(y)
            if type(y) == type([]):
                y = torch.from_numpy(np.array(y))

            ### Channels manipulations
            X, y = self.channel_manipulations(X, y)

        return {"X": X, "y": y}

    def channel_manipulations(self, X, y):

        if self.dataset in ["MNIST", "FashionMNIST"]:
            X = torch.unsqueeze(X, 1)
        elif self.dataset in ["CIFAR10"]:
            X = torch.cat([X[..., i][:, None, ...] for i in range(X.shape[-1])], axis=1)

        return X, y

    def get_augmod_dataset(self):
        pass

    def get_rml_dataset(self):
        pass

    def save_data(self, data):

        for k, v in data.items():
            os.makedirs(os.path.join(self.path2save, self.dataset), exist_ok=True)
            for n_data, a_data in v.items():
                tmp_path2save = os.path.join(
                    self.path2save, self.dataset, f"{n_data}_{k}.pt"
                )
                torch.save(a_data, tmp_path2save)

    def get_data_from_loaders(self, data):

        X = []
        y = []
        for i, tmp_data in enumerate(data):
            if i >= 10000:
                break

            tmp_X = tmp_data[0][None, ...]
            tmp_y = tmp_data[1]
            if "PIL.PngImagePlugin.PngImageFile" in str(type(tmp_y)):
                tmp_y = torch.from_numpy(np.asarray(tmp_y).copy())[None, ...]

            X += [tmp_X]
            y += [tmp_y]

        if X[0].ndim == 4:
            X = [
                torchvision.transforms.Resize(
                    size=(64, 64),
                    interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                )(el)
                for el in X
            ]
        if y[0].ndim == 3:
            y = [
                torchvision.transforms.Resize(
                    size=(64, 64),
                    interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                )(el)
                for el in y
            ]

        X = torch.cat(X, axis=0)
        y = torch.cat(y, axis=0)

        for i, val in enumerate(torch.sort(torch.unique(y))[0]):
            w = y == val
            y[w] = i

        return X, y


if __name__ == "__main__":

    dataset = "OxfordIIITPet"
    path2save = "./tmp_data/blalbla"
    Dataset(dataset=dataset, path2save=path2save)
