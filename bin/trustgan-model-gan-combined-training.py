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
import sys
import click

from trustgan.create_gif import create_gif
from trustgan.plot_perfs import plot_perfs
from trustgan.training import Training


@click.command(
    context_settings={"show_default": True, "help_option_names": ["-h", "--help"]}
)
@click.option("--path2save", help="Path where to save results", required=True)
@click.option(
    "--path2dataset", help="Path to the dataset to run training on", required=True
)
@click.option(
    "--nb-classes", help="Number of classes in the dataset", required=True, type=int
)
@click.option("--num-epochs", default=1, help="Number of epochs")
@click.option(
    "--nb-step-net-gan",
    default=1,
    help="Number of steps to train the Model against the GAN per Model step",
)
@click.option(
    "--nb-step-gan", default=1, help="Number of steps to train the GAN per Model step"
)
@click.option(
    "--nb-step-net-alone",
    default=0,
    help="Number of steps for the model to be alone, without GAN attacks",
)
@click.option(
    "--prop-net-alone",
    default=0.0,
    help="Proportion of the epochs where only the model is learning",
)
@click.option("--path-to-load-net", default=None, help="Path to load a model from")
@click.option("--path-to-load-gan", default=None, help="Path to load a GAN from")
@click.option(
    "--produce-plots",
    is_flag=True,
    help="Choose to produce plots and GIFs instead of training the network",
)
@click.option("--batch-size", default=64, help="Batch size")
@click.option("--device", default="cuda:0", help="Device to run training on")
@click.option("--verbose", default=True)
@click.option(
    "--path2net",
    default=None,
    type=str,
    help="Path where to load a network for the given task",
)
@click.option(
    "--network_name",
    default="Net",
    type=str,
    help="Network name",
)
def main(
    path2save,
    path2dataset,
    nb_classes,
    num_epochs,
    nb_step_net_gan,
    nb_step_gan,
    nb_step_net_alone,
    prop_net_alone,
    path_to_load_net,
    path_to_load_gan,
    produce_plots,
    batch_size,
    device,
    verbose,
    path2net,
    network_name,
):

    if produce_plots:
        plot_perfs(path2save)
        create_gif(path2save, "example-image-best-step-*.png")
        create_gif(path2save, "example-image-not-best-step-*.png")
        create_gif(path2save, "example-true-image-min-step-*.png")
        create_gif(path2save, "example-true-image-max-step-*.png")
    else:

        ###
        if not os.path.isdir(path2save):
            os.mkdir(path2save)
        cfgs = open("{}/configs.txt".format(path2save), "w")
        cfgs.write(" ".join(sys.argv))
        cfgs.close()

        ###
        training_pipeline = Training(
            path_to_save=path2save,
            path2dataset=path2dataset,
            nb_classes=nb_classes,
            num_epochs=num_epochs,
            nb_step_net_gan=nb_step_net_gan,
            nb_step_gan=nb_step_gan,
            nb_step_net_alone=nb_step_net_alone,
            prop_net_alone=prop_net_alone,
            path_to_load_net=path_to_load_net,
            path_to_load_gan=path_to_load_gan,
            batch_size=batch_size,
            device_name=device,
            verbose=verbose,
            path2net=path2net,
            network_name=network_name,
        )
        training_pipeline.train()


if __name__ == "__main__":
    main()
