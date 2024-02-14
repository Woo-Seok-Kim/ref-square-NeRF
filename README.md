# [Project  Page](https://woo-seok-kim.github.io/ref_square_nerf/)

# REF^2-NeRF
Github page of REF^2-NeRF.

This repository's code is heavily borrowed from [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch).

# Installation

This repository is built based on [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch). Follow NeRF-Pytorch's installation guide.

# How to use

Dataset and pre-trained models are avaliable [here](https://drive.google.com/drive/folders/11JuIU2H5ATJ2sbh5CVkEpJ_3P0VEl-RT).
Create log folder of experiment and place checkpoint file.
Set N_importance_{vi, vd, glass} of config file for hierarchical sampling.

```
python ref_square_nerf.py --config config_file_path
```
