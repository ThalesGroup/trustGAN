# TrustGAN

TrustGAN: Training safe and trustworthy deep learning models through generative adversarial networks

This package provides the code developped for the paper:\
"TrustGAN: Training safe and trustworthy deep learning models through generative adversarial networks"\
presented at the CAID-2022 (Conference on Artificial Intelligence for Defence)

## Install

With python, pip and setuptools installed, you simply need to:

```bash
python -m pip install .
```

## Get the data and train a model

We will work within `./xps/`:

### Get the datasets

You can download in-distribution (ID) sample datasets within `data/`:

```bash
python3 ../bin/trustgan-download-data.py --path2save "data/" --dataset "MNIST"
```

You can download out-of-distribution (OOD) sample datasets within `data/`:

```bash
python3 ../bin/trustgan-download-data.py --path2save "data/" --dataset "FashionMNIST"
python3 ../bin/trustgan-download-data.py --path2save "data/" --dataset "CIFAR10"
```

### Train a model

We will now train two models, one without TrustGAN and another with it,
with a selected device `<device>`:

```bash
python3 ../bin/trustgan-model-gan-combined-training.py \
    --path2save "mnist-wo-gan/" \
    --path2dataset "data/MNIST" \
    --nb-classes 10 \
    --prop-net-alone 1 \
    --num-epochs 100 \
    --batch-size 512 \
    --device "cuda:0"
```

```bash
python3 ../bin/trustgan-model-gan-combined-training.py \
    --path2save "mnist-wi-gan/" \
    --path2dataset "data/MNIST" \
    --nb-classes 10 \
    --num-epochs 100 \
    --batch-size 512 \
    --nb-step-net-alone 1 \
    --device "cuda:1"
```

## Test

You can get summary plots and gifs with:

```bash
python3 ../bin/trustgan-model-gan-combined-training.py \
    --path2save "mnist-wo-gan/" \
    --path2dataset "data/MNIST" \
    --nb-classes 10 \
    --produce-plots
```

```bash
python3 ../bin/trustgan-model-gan-combined-training.py \
    --path2save "mnist-wi-gan/" \
    --path2dataset "data/MNIST" \
    --nb-classes 10 \
    --produce-plots
```

## Contributing

If you are interested in contributing to the project, start by reading the [Contributing guide](/CONTRIBUTING.md).

## License

This repository is licensed under the terms of the MIT License (see the file [LICENSE](/LICENSE)).

## Citing

Please cite the following paper if you are using TrustGAN

```bibtex
@inproceedings{dMdBTrustGAN,
       author = {{du Mas des Bourboux}, H{\'e}lion},
    booktitle = {CAID 2022-Second Conference on Artificial Intelligence for Defence},
        title = {TrustGAN: Training safe and trustworthy deep learning models through generative adversarial networks},
         year = {2022},
}
```
