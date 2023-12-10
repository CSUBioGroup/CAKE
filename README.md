## CAKE

CAKE: a flexible self-supervised framework for enhancing cell visualization, clustering, and rare cell identification

## Overview

Single cell sequencing technology has provided unprecedented opportunities for comprehensively deciphering cell heterogeneity. Nevertheless, the high dimensionality and intricate nature of cell heterogeneity have presented substantial challenges to computational methods. Numerous novel clustering methods have been proposed to address this issue. However, none of these methods achieve the consistently better performance under different biological scenario. In this study, we developed CAKE, a novel and scalable self-supervised clustering method, which consists of a contrastive learning model with a mixture neighborhood augmentation for cell representation learning, and a self-Knowledge Distiller model for refinement of clustering results. These designs provide more condensed and cluster-friendly cell representations and improve the clustering performance in term of accuracy and robustness. Furthermore, in addition to accurately identify the major type cells, CAKE could also find more biologically meaningful cell subgroups and rare cell types. The comprehensive experiments on real scRNA-seq datasets demonstrated the superiority of CAKE on visualization and clustering than other comparison methods and indicated its extensive application in the field of cell heterogeneity analysis.

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

Some of data used in our experiments can be found in [`data`](https://github.com/CSUBioGroup/CAKE/tree/main/data). Complete data can be found in [`zenodo`](https://zenodo.org/record/8315578)

## Usage

We provide the scripts for running CAKE. And the hyperparameters can be found in [`config`](https://github.com/CSUBioGroup/CAKE/tree/main/config).

```
python train_KD.py
```



