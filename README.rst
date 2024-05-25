ScCobra: Contrastive cell embedding learning with domain adaptation for single-cell data integration and harmonization 
===================================================================
  
The rapid development of single-cell technologies has underscored the need for more effective methods in the integration and harmonization of single-cell sequencing data. The prevalent challenge of batch effects, resulting from technical and biological variations  across studies, demands accurate and reliable solutions for data integration. Traditional tools often have limitations, both due to reliance on gene expression distribution assumptions and the common issue of over-correction, particularly in methods based on  anchor alignments. Here we introduce scCobra, a deep neural network tool designed  specifically to address these challenges. By leveraging a deep generative model that  combines a contrastive neural network with domain adaptation, scCobra effectively mitigates batch effects and minimizes over-correction without depending on gene  expression distribution assumptions. Additionally, scCobra enables online label transfer  across datasets with batch effects, facilitating the continuous integration of new data  without retraining, and offers features for batch effect simulation and advanced multi-omic  batch integration. These capabilities make scCobra a versatile data integration and  harmonization tool for achieving accurate and insightful biological interpretations from  complex datasets.

<p align="center">
  <img src="https://raw.githubusercontent.com/GlancerZ/scCobra/main/Figure/singlecell_model.png" width="800">
</p>

## Installation

**Step 1**: Create a conda environment for scCobra

```bash
# Recommend you to use python above 3.9
conda create -n scCobra conda-forge::python=3.9 bioconda::bioconductor-singlecellexperiment=1.20.0 conda-forge::r-seuratobject=4.1.3 conda-forge::r-seurat=4.3.0 bioconda::anndata2ri=1.1 conda-forge::rpy2=3.5.2 bioconda::r-signac bioconda::bioconductor-ensdb.hsapiens.v75 bioconda::bioconductor-biovizbaseconda-forge::r-irkernel conda-forge::ipykernel

# Install scanpy scib episcanpy snapatac2
pip install scanpy scib episcanpy snapatac2
# You can install addtional packages: https://scib.readthedocs.io/en/latest/index.html

# Install pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
``` 

**Step 2**: Clone This Repo

```bash
git clone https://github.com/GlancerZ/scCobra.git
```

## Data resources

You can click the dataset name to download

* [simulated dataset](https://figshare.com/ndownloader/files/33798263) contains 12097 cells with 9979 genes, has 7 cell types from 6 batches
* [pancreas dataset](https://figshare.com/ndownloader/files/24539828) contains 16382 cells with 19093 genes, has 14 cell types from 9 batches
* [Lung atlas dataset](https://figshare.com/ndownloader/files/24539942) contains 32472 cells with 15148 genes, has 17 cell types from 16 batches


## Example pipeline
* [Tutorial](https://github.com/GlancerZ/scCobra/blob/main/pancreas_demo.ipynb)

## Credits
scCobra is jointly developed by Bowen Zhao and Yi Xiong from Shanghai Jiaotong University and Jun Ding from McGill University.
