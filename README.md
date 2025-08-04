# PASTA
PASTA (PAthway-oriented Spatial gene impuTAtaion) is a tool to imputate pathway expression for a spatial transcriptomic dataset referring to a single-cell RNA sequencing dataset. 
The spatial transcriptomics dataset and the corresponding reference scRNA-seq data should be from the same tissue for imputation accuracy. 

# How to install
Git clone the repo and then enter the folder. The dependencies can be found in the `environment.yml`.

- Set up environment 

`conda env create -f environment.yml`

- Activate the enviornment by typing the following in shell

`
conda activate pastaenv
`

# A quick start 
Git clone the repo and we can start using PASTA. 

The example dataset can be downloaded from our github page under the folder `example_data`. The data can be extracted by

```
import pickle
file = open('./example_data/test.pkl', 'rb')
sp_adata = pickle.load(file)
sc_adata = pickle.load(file)
cluster = pickle.load(file)
coords = pickle.load(file)
pthw_genes = pickle.load(file)
file.close()
```

Then we can run the analysis using

```
import os
sys.path.append('./pasta')
import __init__
import _version
import optimizer
import mapper
import utils

mapper.pp_adatas(sc_adata, sp_adata)
ad_map = mapper.mapping(sc_adata, sp_adata, pthw_genes=genes, 
	sp_coords=coords, ncell_thres=10, sp_celltypes=cluster["Cluster"], 
	lambda_1=2, lambda_2=1, lambda_3=1, lambda_4=1, num_epochs=500,
	learning_rate=0.05)
pthw_exp = utils.project_genes(adata_map=ad_map, adata_sc=sc_adata, pthw=genes)
```

The function takes several inputs:
* `sc_adata`: ScRNA-seq dataset
* `sp_adata`: ST dataset 
* `pthw_genes`: A list of genes from a pathway 
* `sp_coords`: Coordinates of ST data
* `ncell_thres`: Cell neighborhood regulation
* `sp_celltypes`: Cell type of ST dataset
* `lambda_1`: Resconstruction loss regulation
* `lambda_2`: Global similarity regulation 
* `lambda_3`: Pathway loss regulation 
* `lambda_4`: Neighborhood loss regulation
* `learning_rate`: Learning rate

# Usage 
A more detailed description can be found here: https://pasta-website.readthedocs.io/en/latest/.
