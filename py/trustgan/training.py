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
import matplotlib.pyplot as plt
import copy
import glob
import numpy as np
import torch
import torchsummaryX

from . import networks
from .losses import softXEnt, combined_gan_loss
from .networks import Gan
from .dataset import get_dataloader, Modifier


class Training:
    def __init__(
        self,
        path_to_save,
        path2dataset,
        nb_classes,
        device_name=None,
        batch_size=64,
        path_to_load_net=None,
        path_to_load_gan=None,
        num_epochs=2,
        nb_step_net_gan=1,
        nb_step_gan=1,
        nb_step_net_alone=1,
        verbose=True,
        prop_net_alone=0.0,
        path2net=None,
        network_name="Net",
        nb_channels=None,
    ):

        self.verbose = verbose

        #
        self.path_to_save = path_to_save
        self.path_to_load_net = path_to_load_net
        self.path_to_load_gan = path_to_load_gan

        #
        if self.path_to_save is not None:
            if not os.path.isdir(self.path_to_save):
                os.mkdir(self.path_to_save)
            for folder in ["plots", "nets", "perfs-plots", "gifs"]:
                if not os.path.isdir(os.path.join(self.path_to_save, folder)):
                    os.mkdir(os.path.join(self.path_to_save, folder))
                else:
                    print("\nWARNING\n Files exists")

        #
        self.nb_classes = nb_classes
        self.trainloader = get_dataloader(
            path2dataset=path2dataset,
            nb_classes=nb_classes,
            dataset_type="train",
            batch_size=batch_size,
            verbose=verbose,
        )
        self.validloader = get_dataloader(
            path2dataset=path2dataset,
            nb_classes=nb_classes,
            dataset_type="valid",
            batch_size=batch_size,
            verbose=verbose,
        )
        self.testloader = get_dataloader(
            path2dataset=path2dataset,
            nb_classes=nb_classes,
            dataset_type="test",
            batch_size=batch_size,
            verbose=verbose,
        )
        self.modifier = Modifier(nb_channels=nb_channels)

        #
        nb_dims = self.modifier(next(iter(self.validloader)))[0].ndim - 2
        print(f"INFO: Found {nb_dims} dimensions")
        nb_channels = self.modifier(next(iter(self.validloader)))[0].shape[1]
        print(f"INFO: Found {nb_channels} channels")

        #
        self.device = self.get_device(device_name=device_name)
        self.sequence_dims_onnx = np.arange(2, 2 + nb_dims)

        #
        if path2net is not None:
            self.net = path2net
        else:
            self.net = getattr(networks, network_name)(
                nb_classes=self.nb_classes,
                nb_channels=nb_channels,
                batch_norm=False,
                weight_norm=True,
                dim=f"{nb_dims}d",
            )
        self.net_loss = softXEnt
        self.net_optim = torch.optim.AdamW(self.net.parameters(), weight_decay=0.05)

        #
        self.gan = Gan(
            nb_channels=nb_channels,
            batch_norm=True,
            weight_norm=False,
            dim=f"{nb_dims}d",
        )
        self.gan_loss = combined_gan_loss
        self.gan_optim = torch.optim.AdamW(self.gan.parameters(), weight_decay=0.05)
        self.gan_scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.gan_optim,
            base_lr=1.0e-3,
            max_lr=5.0e-3,
            step_size_up=50,
            cycle_momentum=False,
        )

        #
        for model, path_to_load in [
            (self.net, self.path_to_load_net),
            (self.gan, self.path_to_load_gan),
        ]:

            model = model.to(self.device)

            if path_to_load is not None:
                ld = torch.load(path_to_load, map_location=self.device)
                model.load_state_dict(ld)

            if self.verbose:
                model.eval()
                x_rand = torch.rand(
                    self.modifier(next(iter(self.validloader)))[0].shape,
                    device=self.device,
                )
                torchsummaryX.summary(model, x_rand)
                model.train()

        self.gan2 = copy.deepcopy(self.gan)

        #
        self.num_epochs = num_epochs
        self.nb_step_net_gan = nb_step_net_gan
        self.nb_step_gan = nb_step_gan
        self.nb_step_net_alone = nb_step_net_alone
        self.prop_net_alone = prop_net_alone
        self.recovered_from_nan_net = 0
        self.recovered_from_nan_gan = 0
        self.grad_clipping_coef = 1.0

    def get_device(self, device_name=None):
        """ """

        if device_name is None:
            device_name = "cuda:0"

        if not torch.cuda.is_available():
            device_name = "cpu"

        device = torch.device(device_name)
        if self.verbose:
            print(f"Device = {device}, {device_name}")

        return device

    @torch.inference_mode()
    def get_predictions(self, loader, score_type="MCP"):

        if type(loader) == str:
            loader = getattr(self, loader)

        self.net.eval()

        softmax = torch.nn.Softmax(dim=1)

        truth = []
        preds = []
        score = []

        for data in loader:

            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            inputs, labels = self.modifier((inputs, labels))

            outputs = softmax(self.net(inputs))
            _, tmp_preds = torch.max(outputs, 1)
            outputs, _ = torch.sort(outputs, dim=1)
            if score_type == "MCP":
                tmp_score = outputs[:, -1]
            elif score_type == "Diff2MCP":
                tmp_score = outputs[:, -1] - outputs[:, -2]
            _, tmp_truth = torch.max(labels, 1)

            truth += [tmp_truth.detach().cpu().numpy()]
            preds += [tmp_preds.detach().cpu().numpy()]
            score += [tmp_score.detach().cpu().numpy()]

        truth = np.hstack(truth)
        preds = np.hstack(preds)
        score = np.hstack(score)

        self.net.train()

        return truth, preds, score

    @torch.inference_mode()
    def recover_from_nan_net(self):
        """
        Recover from Nan in net
        """

        self.net.eval()

        data = next(iter(self.trainloader))

        inputs = data[0][:1].to(self.device)
        inputs, _ = self.modifier((inputs, None))
        outputs = self.net(inputs)

        if torch.any(torch.isnan(outputs)):
            print("\nWARNING: The Net gives NaN")
            nets = glob.glob("{}/nets/net-step-*.pth".format(self.path_to_save))
            nets = sorted(sorted(nets), key=len)
            idx = -1
            while torch.any(torch.isnan(outputs)):
                self.net.load_state_dict(torch.load(nets[idx]))
                outputs = self.net(inputs)
                idx -= 1
            print(f"WARNING: Recover a proper state there: {nets[idx+1]}")

            self.recovered_from_nan_net += 1
            print(f"WARNING: Recovered from NaN {self.recovered_from_nan_net} times\n")
            self.net_optim = torch.optim.AdamW(
                self.net.parameters(),
                weight_decay=0.05,
                lr=1.0e-3 / 2**self.recovered_from_nan_net,
            )

        self.net.train()

    @torch.inference_mode()
    def recover_from_nan_gan(self):
        """
        Recover from Nan in GAN
        """

        self.gan.eval()

        data = self.modifier(next(iter(self.trainloader)))

        rand_inputs = torch.rand(data[0][:1].shape, device=self.device)
        gan_outputs = self.gan(rand_inputs)

        if torch.any(torch.isnan(gan_outputs)):
            print("\nWARNING: The GAN gives NaN")
            nets = glob.glob(
                "{}/nets/gan-not-best-step-*.pth".format(self.path_to_save)
            )
            nets = sorted(sorted(nets), key=len)
            idx = -1
            while torch.any(torch.isnan(gan_outputs)):
                self.gan.load_state_dict(torch.load(nets[idx]))
                gan_outputs = self.gan(rand_inputs)
                idx -= 1
            print(f"WARNING: Recover a proper state there: {nets[idx]}")

            self.recovered_from_nan_gan += 1
            print(f"WARNING: Recovered from NaN {self.recovered_from_nan_gan} times\n")
            self.gan_optim = torch.optim.AdamW(
                self.gan.parameters(),
                weight_decay=0.05,
                lr=1.0e-3 / 2**self.recovered_from_nan_gan,
            )
            self.gan_scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.gan_optim,
                base_lr=1.0e-3 / 2**self.recovered_from_nan_gan,
                max_lr=5.0e-3 / 2**self.recovered_from_nan_gan,
                step_size_up=50,
                cycle_momentum=False,
            )

        self.gan.train()

    @torch.inference_mode()
    def get_perfs(self, loader, header_str=""):

        self.net.eval()
        self.gan.eval()

        accs = {"net": 0.0, "net_on_gan": 0.0, "gan": 0.0}
        loss = {"net": 0.0, "net_on_gan": 0.0, "gan": 0.0}

        for data in loader:

            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            inputs, labels = self.modifier((inputs, labels))

            # Net on real data
            outputs = self.net(inputs)
            loss["net"] += (
                self.net_loss(outputs, labels, reduction="sum").detach().cpu().numpy()
            )
            _, hard_predicted = torch.max(outputs, 1)
            _, hard_labels = torch.max(labels, 1)
            accs["net"] += (
                (hard_predicted == hard_labels).float().sum().detach().cpu().numpy()
            )

            # Net on Gan generated images and gan loss
            rand_inputs = torch.rand(inputs.shape, device=self.device)
            rand_labels = (
                1.0 / self.nb_classes * torch.ones(labels.shape, device=self.device)
            )

            gan_outputs = self.gan(rand_inputs)
            net_outputs = self.net(gan_outputs)

            loss["net_on_gan"] += (
                self.net_loss(net_outputs, rand_labels, reduction="sum")
                .detach()
                .cpu()
                .numpy()
            )
            loss["gan"] += (
                self.gan_loss(rand_inputs, gan_outputs, net_outputs, reduction="sum")
                .detach()
                .cpu()
                .numpy()
            )

        for k in accs.keys():
            accs[k] = accs[k] / loader.dataset.nb_tot_labels
        for k in loss.keys():
            loss[k] = loss[k] / loader.dataset.nb_tot_labels

        self.net.train()
        self.gan.train()

        res_str = header_str + ": Losses: "
        for k, v in loss.items():
            res_str += f"{k} = {v:6.3f}, "
        res_str += "Accuracy: "
        for k, v in accs.items():
            res_str += f"{k} = {v:6.3f}, "

        print(res_str)

        return accs, loss

    def plot_one_example(self, x, pred, score_pred, label, path2save):
        """
        Plot and save one example
        """

        if len(x.shape) == 3:
            x = (x + 1.0) / 2.0
            legend = False
            if x.shape[0] == 1:
                x = x[0, :, :]
                plt.imshow(x, cmap="gray")
            elif x.shape[0] == 3:
                x = np.concatenate(
                    [x[0][:, :, None], x[1][:, :, None], x[2][:, :, None]], axis=-1
                )
                plt.imshow(x)
            else:
                raise ValueError(
                    "ERROR: x does not have the proper dimension: {}".format(x.shape)
                )
        elif len(x.shape) == 2:
            legend = True
            for i in range(x.shape[0]):
                plt.plot(x[i, :], label=r"dim={}".format(i))
        else:
            raise ValueError(
                "ERROR: x does not have the proper dimension: {}".format(x.shape)
            )

        if label == -1:
            plt.title(
                "step = {}, pred = {}, score = {}%".format(
                    self.epoch, pred, round(100.0 * np.nan_to_num(score_pred, 0.0))
                )
            )
        else:
            plt.title(
                "step = {}, pred = {}, score = {}%, truth = {}".format(
                    self.epoch,
                    pred,
                    round(100.0 * np.nan_to_num(score_pred, 0.0)),
                    label,
                )
            )

        if legend:
            plt.legend(loc=1)
        plt.savefig(path2save)
        plt.clf()

    @torch.inference_mode()
    def save_epoch(
        self, best, loader=None, gan_outputs=None, net_outputs=None, save_plot=True
    ):
        """ """

        if save_plot:
            if loader is not None:
                self.net.eval()
                self.gan.eval()
                dims = list(self.modifier(next(iter(loader)))[0].shape)
                rand_inputs = torch.rand(dims, device=self.device)

                gan_outputs = self.gan(rand_inputs)
                net_outputs = self.net(gan_outputs)
                self.net.train()
                self.gan.train()

            net_outputs = torch.nn.functional.softmax(net_outputs, dim=1)
            score_pred, predicted = torch.max(net_outputs, 1)

            if score_pred.ndim > 1:
                score_pred = score_pred.mean(axis=tuple(range(1, score_pred.ndim)))
                predicted = predicted.to(torch.float).mean(
                    axis=tuple(range(1, predicted.ndim))
                )

            idx = torch.argmax(score_pred)

            images = gan_outputs[idx].cpu().detach().numpy()
            self.plot_one_example(
                images,
                pred=predicted[idx].item(),
                score_pred=score_pred[idx].item(),
                label=-1,
                path2save="{}/plots/example-image-{}-step-{}.png".format(
                    self.path_to_save, best, self.epoch
                ),
            )

        torch.save(
            self.gan.state_dict(),
            "{}/nets/gan-{}-step-{}.pth".format(self.path_to_save, best, self.epoch),
        )

    @torch.inference_mode()
    def get_example(self, loader):
        """ """

        self.net.eval()

        inputs, labels = next(iter(loader))
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        inputs, labels = self.modifier((inputs, labels))
        _, labels = torch.max(labels, 1)

        net_outputs = self.net(inputs)

        net_outputs = torch.nn.functional.softmax(net_outputs, dim=1)
        score_pred, predicted = torch.max(net_outputs, 1)

        if score_pred.ndim > 1:
            score_pred = score_pred.mean(axis=tuple(range(1, score_pred.ndim)))
            predicted = predicted.to(torch.float).mean(
                axis=tuple(range(1, predicted.ndim))
            )
            labels = labels.to(torch.float).mean(axis=tuple(range(1, labels.ndim)))

        idx_min = torch.argmin(score_pred)
        idx_max = torch.argmax(score_pred)

        for idx, name in [(idx_min, "min"), (idx_max, "max")]:

            images = inputs[idx].cpu().detach().numpy()

            self.plot_one_example(
                images,
                pred=predicted[idx].item(),
                score_pred=score_pred[idx].item(),
                label=labels[idx],
                path2save="{}/plots/example-true-image-{}-step-{}.png".format(
                    self.path_to_save, name, self.epoch
                ),
            )

        self.net.train()

    def log_perfs(self):
        """ """

        for dataset, name in [(self.trainloader, "train"), (self.validloader, "valid")]:
            accs, loss = self.get_perfs(
                loader=dataset, header_str="{} {}".format(self.epoch, name)
            )

            for met, met_name in [(accs, "accs"), (loss, "loss")]:

                for k, v in met.items():
                    final_name = "{}_{}".format(met_name, k)
                    if final_name not in self.perfs[name].keys():
                        self.perfs[name][final_name] = []

                    self.perfs[name][final_name] += [v]

    def net_train(self, inputs, labels):
        """ """

        self.net.train()
        self.net_optim.zero_grad()

        net_outputs = self.net(inputs)

        loss_net = self.net_loss(net_outputs, labels)
        loss_net.backward()

        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clipping_coef)
        self.net_optim.step()

        _, truth = torch.max(labels, 1)
        _, predicted = torch.max(net_outputs, 1)
        acc_net = (predicted == truth).float().mean()

        return loss_net, acc_net

    def net_gan_train(self, inputs_shape, labels_shape):
        """ """

        self.net.train()
        self.net_optim.zero_grad()

        # for dim in self.sequence_dims_onnx:
        #     inputs_shape[dim] = torch.randint(1, inputs_shape[dim] + 1, size=(1,))[0]

        rand_inputs = torch.rand(inputs_shape, device=self.device)
        rand_labels = (
            1.0 / self.nb_classes * torch.ones(labels_shape, device=self.device)
        )

        r = torch.rand(1)
        nets = glob.glob("{}/nets/gan-*-step-*.pth".format(self.path_to_save))

        if (r < 0.1) and (len(nets) > 0):
            r = torch.randint(low=0, high=len(nets), size=[1])
            self.gan2.load_state_dict(torch.load(nets[r]))

            self.gan2.eval()
            gan_outputs = self.gan2(rand_inputs)
            self.gan2.train()
        else:
            self.gan.eval()
            gan_outputs = self.gan(rand_inputs)
            self.gan.train()

        net_outputs = self.net(gan_outputs)

        loss_net_gan = self.net_loss(net_outputs, rand_labels)
        loss_net_gan.backward()
        # loss_net_gan = loss_net_gan.item()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clipping_coef)
        self.net_optim.step()

        return loss_net_gan

    def gan_train(self, inputs_shape):
        """ """

        # for dim in self.sequence_dims_onnx:
        #    inputs_shape[dim] = torch.randint(1, inputs_shape[dim] + 1, size=(1,))

        self.gan.train()
        self.gan_optim.zero_grad()
        rand_inputs = torch.rand(inputs_shape, device=self.device)

        gan_outputs = self.gan(rand_inputs)
        self.net.eval()

        net_outputs = self.net(gan_outputs)
        self.net.train()

        loss_gan = self.gan_loss(rand_inputs, gan_outputs, net_outputs)
        loss_gan.backward()

        loss_gan = loss_gan.item()
        if loss_gan < self.best_loss:
            self.save_epoch(
                best="best",
                gan_outputs=gan_outputs,
                net_outputs=net_outputs,
                save_plot=True,
            )

            self.best_loss = loss_gan

        torch.nn.utils.clip_grad_norm_(self.gan.parameters(), 1.0)
        self.gan_optim.step()

        return loss_gan

    def train(self):
        """ """

        self.perfs = {"train": {}, "valid": {}}
        self.perfs["train"]["best-gan-loss"] = []
        self.perfs["valid"]["best-gan-loss"] = []
        self.best_loss = float("inf")
        loss_gan = -1.0

        #
        for self.epoch in range(self.num_epochs):

            self.recover_from_nan_net()
            self.recover_from_nan_gan()
            self.perfs["train"]["best-gan-loss"] += [self.best_loss]
            self.perfs["valid"]["best-gan-loss"] += [-1.0]
            self.log_perfs()
            self.get_example(loader=self.trainloader)
            self.save_epoch(best="not-best", loader=self.trainloader)
            np.save("{}/performances.npy".format(self.path_to_save), self.perfs)

            if (len(self.perfs["valid"]["loss_net"]) == 1) or (
                self.perfs["valid"]["loss_net"][-1]
                <= np.min(self.perfs["valid"]["loss_net"][:-1])
            ):
                torch.save(
                    self.net.state_dict(),
                    os.path.join(self.path_to_save, "nets/net-best-valid-loss.pth"),
                )

            self.best_loss = float("inf")

            self.net.train()
            self.gan.train()

            # Train the classifier
            for i, data in enumerate(self.trainloader):

                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                inputs, labels = self.modifier((inputs, labels))

                r = torch.rand(1)
                if (self.epoch >= self.nb_step_net_alone) and (r > self.prop_net_alone):

                    for _ in range(self.nb_step_gan):

                        loss_gan = self.gan_train(list(inputs.shape))

                    for _ in range(self.nb_step_net_gan):
                        loss_net_gan = self.net_gan_train(
                            list(inputs.shape), list(labels.shape)
                        )

                else:
                    loss_net_gan = -1.0
                    loss_gan = -1.0

                loss_net, acc_net = self.net_train(inputs, labels)

                if i % 100 == 0:
                    print(
                        f"{i:03d}/{len(self.trainloader):03d}, Loss: net = {loss_net:6.3f}, net_on_gan = {loss_net_gan:6.3f}, gan = {loss_gan:6.3f}, Accs: net = {acc_net:6.3f}"
                    )

            if self.epoch % 100 == 0:
                torch.save(
                    self.net.state_dict(),
                    "{}/nets/net-step-{}.pth".format(self.path_to_save, self.epoch),
                )
            torch.save(self.net.state_dict(), "{}/net.pth".format(self.path_to_save))
            torch.save(self.gan.state_dict(), "{}/gan.pth".format(self.path_to_save))
            self.save_to_torch_full_model()

        if loss_gan != -1.0:
            self.gan_scheduler.step()

        self.epoch = self.num_epochs
        self.perfs["train"]["best-gan-loss"] += [self.best_loss]
        self.perfs["valid"]["best-gan-loss"] += [-1.0]
        self.log_perfs()
        self.get_example(loader=self.trainloader)
        self.save_epoch(best="not-best", loader=self.trainloader)
        np.save("{}/performances.npy".format(self.path_to_save), self.perfs)

    def save_to_torch_full_model(self):
        """
        Save to Torch full model
        """

        checkpoint = {
            "model": self.net,
            "state_dict": self.net.state_dict(),
        }

        torch.save(checkpoint, "{}/net-fullModel.pth".format(self.path_to_save))
