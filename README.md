# PASTA

PASTA (PAthway-oriented Spatial gene impuTAtaion) is a tool to imputate pathway expression for a spatial transcriptomic dataset referring to a single-cell RNA sequencing dataset. 
The spatial transcriptomics dataset and the corresponding reference scRNA-seq data should be from the same tissue for imputation accuracy. 

# How to install
- Set up environment \\
`conda env create -f environment.yml`

- Install PASTA by typing the following in shell \\
`
conda activate PASTA\_env
pip install PASTA
`

# Simple example
Suppose you have a spatial transcriptomics dataset `sp.csv` and a scRNA-seq dataset `sc.csv`. 
```
# read the datasets into anndata
import pasta as pasta

sp_ann = anndata.read_csv("sp.csv")
sc_ann = anndata.read_csv("sc.csv")
map = pasta.align(sp_ann, sc_ann)
pasta.impute(map, sc_ann)
```
