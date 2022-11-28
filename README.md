# TrustGAN

TrustGAN: Training safe and trustworthy deep learning models through generative adversarial networks

This package provides the code developped for the paper:\
"TrustGAN: Training safe and trustworthy deep learning models through generative adversarial networks"\
presented at the CAID-2022 (Conference on Artificial Intelligence for Defence) <https://arxiv.org/abs/2211.13991>

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
@ARTICLE{2022arXiv221113991D,
       author = {{du Mas des Bourboux}, H{\'e}lion},
        title = "{TrustGAN: Training safe and trustworthy deep learning models through generative adversarial networks}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Machine Learning, Computer Science - Computer Vision and Pattern Recognition},
         year = 2022,
        month = nov,
          eid = {arXiv:2211.13991},
        pages = {arXiv:2211.13991},
archivePrefix = {arXiv},
       eprint = {2211.13991},
 primaryClass = {cs.LG},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022arXiv221113991D},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
