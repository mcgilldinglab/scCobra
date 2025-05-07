<p align="center">
  <img src="https://github.com/mcgilldinglab/scCobra/blob/main/Figure/scCobra_logo.png" width="500">
</p>

# scCobra: Contrastive cell embedding learning with domain-adaptation for single-cell data integration and harmonization 
  
The rapid development of single-cell technologies underscores the need for more effective methods to integrate and harmonize single-cell sequencing data. The technical and biological variations across studies demand accurate and reliable solutions for data integration. Conventional tools often face limitations due to reliance on gene expression distribution assumptions and over-correction issues. Here, we introduce scCobra, a deep neural network tool designed to address these challenges. By leveraging a deep generative model that combines a contrastive neural network with domain adaptation, scCobra mitigates batch effects and minimizes over-correction without gene expression distribution assumptions. Additionally, scCobra enables online label transfer across datasets with batch effects, allowing continuous integration of new data without retraining, and offers batch effect simulation and advanced multi-omic batch integration. These capabilities make scCobra a significant advancement for integrating datasets with batch effects, enabling comprehensive biological examination of the integrated data. Please refer to our [manuscript](https://www.biorxiv.org/content/10.1101/2022.10.23.513389v2) for details.

<p align="center">
  <img src="https://github.com/mcgilldinglab/scCobra/blob/main/Figure/Fig1.png" width="800">
</p>

## Installation

**Step 1**: Create a conda environment for scCobra

```bash
# Recommend you to use python above 3.10
conda create -n scCobra conda-forge::python=3.10 conda-forge::ipykernel 

# Install scanpy scib
pip install scanpy scib
# You can install addtional packages: https://scib.readthedocs.io/en/latest/index.html

# Install pytorch
pip3 install torch torchvision torchaudio torch_optimizer
``` 

**Step 2**: Clone This Repo

```bash
git clone https://github.com/mcgilldinglab/scCobra.git
```

## Run

Please refer run_integration.ipynb

## Data resources

You can click the dataset name to download

* [simulated dataset](https://figshare.com/ndownloader/files/33798263) contains 12097 cells, has 7 cell types from 6 batches
* [pancreas dataset](https://figshare.com/ndownloader/files/24539828) contains 16382 cells, has 14 cell types from 9 batches
* [Immune dataset](https://figshare.com/ndownloader/files/25717328) contains 33506 cells, has 16 cell types from 10 batches
* [Lung atlas dataset](https://figshare.com/ndownloader/files/24539942) contains 32472 cells, has 17 cell types from 16 batches


## scCobra's Document:
https://sccobra.readthedocs.io/

## Credits
scCobra is jointly developed by Bowen Zhao and Yi Xiong from Shanghai Jiaotong University and Jun Ding from McGill University.

