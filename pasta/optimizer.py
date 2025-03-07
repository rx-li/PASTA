import numpy as np
import logging
import torch
from torch.nn.functional import softmax, cosine_similarity
from scipy.spatial.distance import euclidean


class Mapping:

    def __init__(
        self,
        S,
        G,
        pathway_index, # spatial gene pathway index 
        celltype_index,
        idx_closet_celltype, 
        ncell_thres, 
       # d=None,
       # d_source=None,
        lambda_g1=1.0,
        lambda_g2=1,
        device="cpu",
        adata_map=None,
        random_state=None,
    ):
        """
        S (ndarray): scRNA-seq matrix, shape: cell x gene.
        G (ndarray): ST matrix, shape: cell x gene.
                
        d (ndarray): Spatial density of cells, shape = (number_spots,). If not provided, the density term is ignored.
                This array should satisfy the constraints d.sum() == 1.
        d_source (ndarray): Density of single cells in single cell clusters. To be used when S corresponds to cluster-level expression.
                This array should satisfy the constraint d_source.sum() == 1.
        
        lambda_g1 (float): Optional. gene/cell loss function. Default is 1.
        lambda_g2 (float): Optional. Strength of voxel-gene regularizer. Default is 1.
        device (str or torch.device): Default is 'cpu'.
        #random_state (int): Optional. pass an int to reproduce training. Default is None.
        """

        self.S = torch.tensor(S, device=device, dtype=torch.float32)
        self.G = torch.tensor(G, device=device, dtype=torch.float32)
        self.idx_closet_celltype = idx_closet_celltype
        self.pathway_index = pathway_index
        self.celltype_index = celltype_index
        
        # self.target_density_enabled = d is not None
        # if self.target_density_enabled:
        #     self.d = torch.tensor(d, device=device, dtype=torch.float32)

        # self.source_density_enabled = d_source is not None
        # if self.source_density_enabled:
        #     self.d_source = torch.tensor(d_source, device=device, dtype=torch.float32)

        self.lambda_g1 = lambda_g1
        self.lambda_g2 = lambda_g2
       # self._density_criterion = torch.nn.KLDivLoss(reduction="sum")

        self.random_state = random_state

        if adata_map is None:
            if self.random_state:
                np.random.seed(seed=self.random_state)
            self.M = np.random.normal(0, 1, (S.shape[0], G.shape[0]))
        else:
            raise NotImplemented
            self.M = adata_map.X  # doesn't work. maybe apply inverse softmax

        self.M = torch.tensor(
            self.M, device=device, requires_grad=True, dtype=torch.float32
        )
        
    def _loss_fn(self, verbose=True):
    
        M_probs = softmax(self.M, dim=1) 

        G_pred = torch.matmul(M_probs.t(), self.S) # prediction spatial 
        gv_term = self.lambda_g1 * cosine_similarity(G_pred, self.G, dim=0).mean() # loss in genes
        gv_term2 = self.lambda_g1 * cosine_similarity(G_pred[:, self.pathway_index], self.G[:, self.pathway_index], dim=0).mean() # loss in genes
        
        # A loss term for pathway 
        pathway_pred_f = G_pred[:, self.pathway_index] # get the pathway genes 
        pathway_g = self.G[:, self.pathway_index]        
        # for each cell, get the closest pathway within the celltype 
        
        pathway_pred_celltype = [pathway_pred_f[i.values, ] for i in self.celltype_index]
        pathway_neighbor_celltype = [pathway_g[i, ] for i in self.idx_closet_celltype]
        
        score_celltypes = []
        vg_celltypes = []
        
        for i in range(len(self.idx_closet_celltype)):
            score = [((pathway_neighbor_celltype[i][:, j, :] - pathway_pred_celltype[i])**2).mean() for j in range(pathway_neighbor_celltype[i].shape[1])]
            vg = [cosine_similarity(pathway_neighbor_celltype[i][:, j, :], pathway_pred_celltype[i], dim=1).mean() for j in range(pathway_neighbor_celltype[i].shape[1])]
            score_celltypes.append(torch.stack(score).mean())
            vg_celltypes.append(torch.stack(vg).mean())
            
        pathway_term = torch.stack(score_celltypes).mean()     
        vg_term = torch.stack(vg_celltypes).mean()

        expression_term = gv_term + vg_term + gv_term2

        main_loss = (gv_term / self.lambda_g1).tolist()

        vg_reg = (vg_term / self.lambda_g2).tolist()
        pathway_reg = pathway_term.tolist()
        
        if verbose:

            term_numbers = [main_loss, vg_reg,  pathway_reg]
            term_names = ["Score", "VG reg", "pathway_reg"]

            d = dict(zip(term_names, term_numbers))
            clean_dict = {k: d[k] for k in d if not np.isnan(d[k])}
            msg = []
            for k in clean_dict:
                m = "{}: {:.3f}".format(k, clean_dict[k])
                msg.append(m)

            print(str(msg).replace("[", "").replace("]", "").replace("'", ""))

        total_loss = - expression_term + pathway_term

        return total_loss, main_loss, vg_reg, pathway_reg

    def train(self, num_epochs, learning_rate=0.01, print_each=100):

        if self.random_state:
            torch.manual_seed(seed=self.random_state)
        optimizer = torch.optim.Adam([self.M], lr=learning_rate)

        if print_each:
            logging.info(f"Printing scores every {print_each} epochs.")

        keys = ["total_loss", "main_loss", "vg_reg", "pathway_reg"]
        values = [[] for i in range(len(keys))]
        training_history = {key: value for key, value in zip(keys, values)}
        for t in range(num_epochs):
            if print_each is None or t % print_each != 0:
                run_loss = self._loss_fn(verbose=False)
            else:
                run_loss = self._loss_fn(verbose=True)

            loss = run_loss[0]

            for i in range(len(keys)):
                training_history[keys[i]].append(str(run_loss[i]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # take final softmax w/o computing gradients
        with torch.no_grad():
            output = softmax(self.M, dim=1).cpu().numpy()
            return output, training_history

