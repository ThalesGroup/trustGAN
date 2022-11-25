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


def softXEnt(inputs, targets, reduction="mean"):
    """
    Soft cross entropy loss
    """

    log_probs = torch.nn.functional.log_softmax(inputs, dim=1)
    r = -targets * log_probs

    if reduction == "sum":
        r = r.sum()
    elif reduction == "mean":
        r = r.sum() / inputs.shape[0]

    r /= np.log(inputs.shape[1])

    return r


def gan_diversity_loss(rand_inputs, net_outputs, idxs=None, reduction="mean", ln=2):
    """
    GAN diversity loss: compare the outputs of the classifier
        to two different GAN generated images

    Arguments:
        gan_inputs (torch.FloatTensor): random inputs at the entry of the GAN
        net_outputs (torch.FloatTensor): Outputs of the classifier for the different GAN produced images
        idxs (list of tuple): indices to compare
        ln (int) : power as in L-<power> Loss

    """

    probs = torch.nn.functional.softmax(net_outputs, dim=1)
    log_probs = torch.nn.functional.log_softmax(net_outputs, dim=1)
    if idxs is None:
        idxs = np.arange(net_outputs.shape[1])
    idxs = idxs[idxs < rand_inputs.shape[0]]

    r = 0.0
    w = 0.0
    for i in idxs:
        for j in idxs[:i]:
            tmp_r = (-probs[i] * log_probs[j]).sum()
            tmp_w = (torch.absolute(rand_inputs[i] - rand_inputs[j]) ** ln).mean()
            r += tmp_w / (1.0 + tmp_r)
            w += tmp_w
    r /= w

    r /= 1.0 + np.log(net_outputs.shape[1])

    return r


def gan_product_diversity_loss(
    rand_inputs, gan_outputs, idxs=np.arange(10), reduction="mean", ln=2
):
    """
    GAN diversity loss: compare the outputs of the GAN

    Arguments:
        rand_inputs (torch.FloatTensor): random inputs at the entry of the GAN
        gan_outputs (torch.FloatTensor): outputs of the GAN, i.e. generated images
        idxs (list of tuple): indices to compare
        reduction
        ln (int) : power as in L-<power> Loss

    """

    idxs = idxs[idxs < rand_inputs.shape[0]]

    r = 0.0
    w = 0.0
    for i in idxs:
        for j in idxs[:i]:
            tmp_r = (torch.absolute(gan_outputs[i] - gan_outputs[j]) ** ln).mean()
            tmp_w = (torch.absolute(rand_inputs[i] - rand_inputs[j]) ** ln).mean()
            r += tmp_w / (1.0 + tmp_r)
            w += tmp_w
    r /= w

    r /= 1.0 + 2.0**ln

    return r


def gan_loss(inputs, reduction="mean"):
    """
    GAN loss: Get the maximum of the logits

    Arguments:
        inputs (torch.FloatTensor): Outputs of the classifier for the different GAN produced images
        reduction (str): Either 'mean' or 'sum'

    Returns:
        loss

    """

    r = -inputs.max(axis=1)[0] + torch.log(torch.exp(inputs).sum(axis=1))

    if reduction == "sum":
        r = r.sum()
    elif reduction == "mean":
        r = r.mean()

    r /= np.log(inputs.shape[1])

    return r


def combined_gan_loss(rand_inputs, gan_outputs, net_outputs, reduction="mean"):
    """ """

    r0 = gan_loss(net_outputs, reduction=reduction)
    r1 = gan_diversity_loss(rand_inputs, net_outputs, reduction=reduction)
    r2 = gan_product_diversity_loss(rand_inputs, gan_outputs, reduction=reduction)
    r = r0 + r1 + r2
    r /= 3.0

    return r
