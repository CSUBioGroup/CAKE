## CAKE

CAKE: clustering scRNA-seq data via combining contrastive learning with knowledge distillation

## Overview

CAKE employs a two-stage self-supervised learning algorithm to learn cell representation and clustering labels. Firstly, CAKE utilizes a enhanced contrastive learning model inspired by the principles of MOCO, and seamlessly integrates a task-specific data augmentation strategy aimed at learning clustering-friendly cell representation. Subsequently, CAKE uses a preliminary clustering approach to derive pseudo clustering labels based on the learned cell representation. In the second-stage, CAKE introduces a self-knowledge distiller model, which is trained on high-density anchor cells from preliminary clusters, and finally assigns the soft and refined clustering labels to cells.

## Installation

First, clone this repository.

```
git clone git clone https://github.com/CSUBioGroup/CAKE.git
cd CAKE/
```

Please make sure PyTorch is installed in your python environment (our test version: PyTorch  == 1.13.1, Python ==  3.8.16). Then install the dependencies:

```
pip install -r requirements.txt
```

## Datasets

Some of data used in our experiments can be found in [`data`](https://github.com/CSUBioGroup/CAKE/tree/main/data). 

## Usage

We provide the scripts for running CAKE. And the hyperparameters can be found in [`config`](https://github.com/CSUBioGroup/CAKE/tree/main/config).

```
python train_KD.py
```



