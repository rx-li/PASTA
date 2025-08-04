import numpy as np
import pandas as pd
from collections import defaultdict
import gzip
import pickle
import scanpy as sc
from tqdm import tqdm
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold

import mapper as mu

import logging
import warnings

from sklearn.metrics import auc

warnings.filterwarnings("ignore")
logger_ann = logging.getLogger("anndata")
logger_ann.disabled = True


def annotate_gene_sparsity(adata):
    """
    Annotates gene sparsity in given Anndatas. 
    Update given Anndata by creating `var` "sparsity" field with gene_sparsity (1 - % non-zero observations).

    Args:
        adata (Anndata): single cell or spatial data.

    Returns:
        None
    """
    mask = adata.X != 0
    gene_sparsity = np.sum(mask, axis=0) / adata.n_obs
    gene_sparsity = np.asarray(gene_sparsity)
    gene_sparsity = 1 - np.reshape(gene_sparsity, (-1,))
    adata.var["sparsity"] = gene_sparsity


def get_matched_genes(prior_genes_names, sn_genes_names, excluded_genes=None):
    """
    Given the list of genes in the spatial data and the list of genes in the single nuclei, identifies the subset of
    genes included in both lists and returns the corresponding matching indices.

    Args:
        prior_genes_names (sequence): List of gene names in the spatial data.
        sn_genes_names (sequence): List of gene names in the single nuclei data.
        excluded_genes (sequence): Optional. List of genes to be excluded. These genes are excluded even if present in both datasets.
        If None, no genes are excluded. Default is None.

    Returns:
        A tuple (mask_prior_indices, mask_sn_indices, selected_genes), with:
        mask_prior_indices (list): List of indices for the selected genes in 'prior_genes_names'.
        mask_sn_indices (list): List of indices for the selected genes in 'sn_genes_names'.
        selected_genes (list): List of names of the selected genes.
        For each i, selected_genes[i] = prior_genes_names[mask_prior_indices[i]] = sn_genes_names[mask_sn_indices[i].
    """
    prior_genes_names = np.array(prior_genes_names)
    sn_genes_names = np.array(sn_genes_names)

    mask_prior_indices = []
    mask_sn_indices = []
    selected_genes = []
    if excluded_genes is None:
        excluded_genes = []
    for index, i in enumerate(sn_genes_names):
        if i in excluded_genes:
            continue
        try:
            mask_prior_indices.append(np.argwhere(prior_genes_names == i)[0][0])
            # if no exceptions above:
            mask_sn_indices.append(index)
            selected_genes.append(i)
        except IndexError:
            pass

    assert len(mask_prior_indices) == len(mask_sn_indices)
    return mask_prior_indices, mask_sn_indices, selected_genes


def one_hot_encoding(l, keep_aggregate=False):
    """
    Given a sequence, returns a DataFrame with a column for each unique value in the sequence and a one-hot-encoding.

    Args:
        l (sequence): List to be transformed.
        keep_aggregate (bool): Optional. If True, the output includes an additional column for the original list. Default is False.

    Returns:
        A DataFrame with a column for each unique value in the sequence and a one-hot-encoding, and an additional
        column with the input list if 'keep_aggregate' is True.
        The number of rows are equal to len(l).
    """
    df_enriched = pd.DataFrame({"cl": l})
    for i in l.unique():
        df_enriched[i] = list(map(int, df_enriched["cl"] == i))
    if not keep_aggregate:
        del df_enriched["cl"]
    return df_enriched

def create_segment_cell_df(adata_sp):
    """
    Produces a Pandas dataframe where each row is a segmentation object, columns reveals its position information.

    Args:
        adata_sp (AnnData): spot-by-gene AnnData structure. Must contain obsm.['image_features']

    Returns:
        None.
        Update spatial AnnData.uns['tangram_cell_segmentation'] with a dataframe: each row represents a segmentation object (single cell/nuclei). Columns are 'spot_idx' (voxel id), and 'y', 'x', 'centroids' to specify the position of the segmentation object.
        Update spatial AnnData.obsm['trangram_spot_centroids'] with a sequence
    """

    if "image_features" not in adata_sp.obsm.keys():
        raise ValueError(
            "Missing parameter for tangram deconvolution. Run `sqidpy.im.calculate_image_features`."
        )

    centroids = adata_sp.obsm["image_features"][["segmentation_centroid"]].copy()
    centroids["centroids_idx"] = [
        np.array([f"{k}_{j}" for j in np.arange(i)], dtype="object")
        for k, i in zip(
            adata_sp.obs.index.values,
            adata_sp.obsm["image_features"]["segmentation_label"],
        )
    ]
    centroids_idx = centroids.explode("centroids_idx")
    centroids_coords = centroids.explode("segmentation_centroid")
    segmentation_df = pd.DataFrame(
        centroids_coords["segmentation_centroid"].to_list(),
        columns=["y", "x"],
        index=centroids_coords.index,
    )
    segmentation_df["centroids"] = centroids_idx["centroids_idx"].values
    segmentation_df.index.set_names("spot_idx", inplace=True)
    segmentation_df.reset_index(
        drop=False, inplace=True,
    )

    adata_sp.uns["tangram_cell_segmentation"] = segmentation_df
    adata_sp.obsm["tangram_spot_centroids"] = centroids["centroids_idx"]
    logging.info(
        f"cell segmentation dataframe is saved in `uns` `tangram_cell_segmentation` of the spatial AnnData."
    )
    logging.info(
        f"spot centroids is saved in `obsm` `tangram_spot_centroids` of the spatial AnnData."
    )



def project_genes(adata_map, adata_sc, pthw, cluster_label=None, scale=True):
    """
    Transfer gene expression from the single cell onto space.

    Args:
        adata_map (AnnData): single cell data
        adata_sp (AnnData): gene spatial data
        cluster_label (AnnData): Optional. Should be consistent with the 'cluster_label' argument passed to `map_cells_to_space` function.
        scale (bool): Optional. Should be consistent with the 'scale' argument passed to `map_cells_to_space` function.

    Returns:
        AnnData: spot-by-gene AnnData containing spatial gene expression from the single cell data.
    """

    # put all var index to lower case to align
    adata_sc.var.index = [g.lower() for g in adata_sc.var.index]

    # make varnames unique for adata_sc
    adata_sc.var_names_make_unique()

    # remove all-zero-valued genes
    sc.pp.filter_genes(adata_sc, min_cells=1)

    if cluster_label:
        adata_sc = mu.adata_to_cluster_expression(adata_sc, cluster_label, scale=scale)

    if not adata_map.obs.index.equals(adata_sc.obs.index):
        raise ValueError("The two AnnDatas need to have same `obs` index.")
    if hasattr(adata_sc.X, "toarray"):
        adata_sc.X = adata_sc.X.toarray()
    X_space = adata_map.X.T @ adata_sc.X
    
    adata_ge = sc.AnnData(
        X=X_space, obs=adata_map.var, var=adata_sc.var, uns=adata_sc.uns
    )
    df = adata_ge.to_df()
    pthw_exp = df.loc[:, df.columns.isin(pthw.str.lower())].sum(axis=1)
    # training_genes = adata_map.uns["train_genes_df"].index.values
    # adata_ge.var["is_training"] = adata_ge.var.index.isin(training_genes)
    
    return pthw_exp


def compare_spatial_geneexp(adata_ge, adata_sp, adata_sc=None, genes=None):
    """ Compares generated spatial data with the true spatial data

    Args:
        adata_ge (AnnData): generated spatial data returned by `project_genes`
        adata_sp (AnnData): gene spatial data
        adata_sc (AnnData): Optional. When passed, sparsity difference between adata_sc and adata_sp will be calculated. Default is None.
        genes (list): Optional. When passed, returned output will be subset on the list of genes. Default is None.

    Returns:
        Pandas Dataframe: a dataframe with columns: 'score', 'is_training', 'sparsity_sp'(spatial data sparsity). 
                          Columns - 'sparsity_sc'(single cell data sparsity), 'sparsity_diff'(spatial sparsity - single cell sparsity) returned only when adata_sc is passed.
    """

    logger_root = logging.getLogger()
    logger_root.disabled = True

    # Check if training_genes/overlap_genes key exist/is valid in adatas.uns
    if not set(["training_genes", "overlap_genes"]).issubset(set(adata_sp.uns.keys())):
        raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

    if not set(["training_genes", "overlap_genes"]).issubset(set(adata_ge.uns.keys())):
        raise ValueError(
            "Missing tangram parameters. Use `project_genes()` to get adata_ge."
        )

    assert list(adata_sp.uns["overlap_genes"]) == list(adata_ge.uns["overlap_genes"])

    if genes is None:
        overlap_genes = adata_ge.uns["overlap_genes"]
    else:
        overlap_genes = genes

    annotate_gene_sparsity(adata_sp)

    # Annotate cosine similarity of each training gene
    cos_sims = []

    if hasattr(adata_ge.X, "toarray"):
        X_1 = adata_ge[:, overlap_genes].X.toarray()
    else:
        X_1 = adata_ge[:, overlap_genes].X
    if hasattr(adata_sp.X, "toarray"):
        X_2 = adata_sp[:, overlap_genes].X.toarray()
    else:
        X_2 = adata_sp[:, overlap_genes].X

    for v1, v2 in zip(X_1.T, X_2.T):
        norm_sq = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_sims.append((v1 @ v2) / norm_sq)

    df_g = pd.DataFrame(cos_sims, overlap_genes, columns=["score"])
    for adata in [adata_ge, adata_sp]:
        if "is_training" in adata.var.keys():
            df_g["is_training"] = adata.var.is_training

    df_g["sparsity_sp"] = adata_sp[:, overlap_genes].var.sparsity

    if adata_sc is not None:
        if not set(["training_genes", "overlap_genes"]).issubset(
            set(adata_sc.uns.keys())
        ):
            raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

        assert list(adata_sc.uns["overlap_genes"]) == list(
            adata_sp.uns["overlap_genes"]
        )
        annotate_gene_sparsity(adata_sc)

        df_g = df_g.merge(
            pd.DataFrame(adata_sc[:, overlap_genes].var["sparsity"]),
            left_index=True,
            right_index=True,
        )
        df_g.rename({"sparsity": "sparsity_sc"}, inplace=True, axis="columns")
        df_g["sparsity_diff"] = df_g["sparsity_sp"] - df_g["sparsity_sc"]

    else:
        logging.info(
            "To create dataframe with column 'sparsity_sc' or 'aprsity_diff', please also pass adata_sc to the function."
        )

    if genes is not None:
        df_g = df_g.loc[genes]

    df_g = df_g.sort_values(by="score", ascending=False)
    return df_g


def run_batch(
    adata_sc,
    adata_sp,
    pathway_data,
    sp_celltypes,
    sp_coords,
    ncell_thres,
    folder,
    lambda_1=1,
    lambda_2=1,
    lambda_3=1,
    lambda_4=1,
    num_epochs=800,
    device="cpu",
    learning_rate=0.05,
):
    """

    Args:
        adata_sc (AnnData): single cell data
        adata_sp (AnnData): gene spatial data
        lambda_1 (float): Global similarity loss. Default is 1.
        lambda_2 (float): Strength of density regularizer. Default is 0.
        lambda_3 (float): Strength of voxel-gene regularizer. Default is 0.
        lambda_4 (float): Strength of entropy regularizer. Default is 0.
        learning_rate (float): Optional. Learning rate for the optimizer. Default is 0.1.
        device (str or torch.device): Optional. Default is 'cuda:0'.
    Returns:
    """
    pathway_names = list(set(pathway_data["pathway"]))
    result = []
    for i in range(len(pathway_names)):
        genes = pathway_data.loc[pathway_data["pathway"].isin([pathway_names[i]]), "gene"]
        my_map = mu.mapping(
                adata_sc=adata_sc,
                adata_sp=adata_sp,
                device=device,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                pathway_genes=genes,
                ncell_thres=ncell_thres,
                sp_coords=sp_coords,
                sp_celltypes=sp_celltypes,
                lambda_1=lambda_1,
                lambda_2=lambda_2,
                lambda_3=lambda_3,
                lambda_4=lambda_4,
                verbose=False
            )

            # project on space
        pred = project_genes(adata_map=my_map, adata_sc=adata_sc, pthw=genes)
        pred.to_csv(folder + pathway_names[i] + ".csv")
        result.append(pred)
    return result
