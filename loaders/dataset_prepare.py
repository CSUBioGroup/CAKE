import os
import numpy as np
import scanpy as sc
import random
import math
import hnswlib

from utils.tools import *
from read_data.read_data import *

import torch
from torch.utils.data import Dataset

class CellDataset(Dataset):
    def __init__(self, data, target):
        super(CellDataset, self).__init__()
        self.data = torch.FloatTensor(data)
        self.target = torch.LongTensor(target)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        label = self.target[index]

        return data, label
    
class CellDatasetPseudoLabel(Dataset):
    def __init__(self, adata, pseudo_label="kmeans", oversample_flag=True, seed=42):
        super().__init__()
        self.adata = adata
        self.pseudo_label = pseudo_label
        self.oversample_flag = oversample_flag

        if self.oversample_flag:
            self.oversample_adata = oversample_cells(adata=self.adata, 
                                                     pseudo_label=self.pseudo_label, 
                                                     seed=seed)
            x = self.oversample_adata.X
            pseudo_label = self.oversample_adata.obs[self.pseudo_label]
            pseudo_label = list(map(int, pseudo_label))
            label = self.oversample_adata.obs['label']
        else:
            x = self.adata.X
            pseudo_label = self.adata.obs[self.pseudo_label]
            pseudo_label = list(map(int, pseudo_label))
            label = self.adata.obs['label']
        
        self.data = torch.FloatTensor(x)
        self.pseu_label = torch.LongTensor(pseudo_label)
        self.true_label = torch.LongTensor(label)

    def __getitem__(self, index):
        data = self.data[index]
        pseu_label = self.pseu_label[index]
        true_label = self.true_label[index]

        return data, pseu_label, true_label
    
    def __len__(self):
        return len(self.data)

def oversample_cells(adata, pseudo_label='leiden', seed=42):
    sampled_cells = []
    avg_cellnums = math.ceil(adata.shape[0] / len(np.unique(adata.obs[pseudo_label])))
    
    for c in np.unique(adata.obs[pseudo_label]):
        cell_type = adata.obs[adata.obs[pseudo_label] == c]
        random.seed(seed)
        
        if cell_type.shape[0] < avg_cellnums:
            selected_cells = random.choices(list(cell_type.index), k=avg_cellnums)
        else:
            selected_cells = list(cell_type.index)
            
        sampled_cells.extend(selected_cells)
        
    sampled_adata = adata[sampled_cells]
    
    return sampled_adata.copy()
    
        
def prepareAll(root_dir, data_type, dataset_name, num_genes, scale=True):
    file_name = os.path.join(root_dir, data_type+"_dir", dataset_name)
    print(f"Current Processed Dataset is: {dataset_name}")

    if data_type == "npz":
        X, Y, cell_type = prepare_npz(file_name)
    elif data_type == 'h5_nested':
        X, Y, cell_type = prepare_nested_h5(file_name)
    elif data_type == "h5":
        X, Y, cell_type = prepare_h5(file_name)
    elif data_type == 'h5ad':
        X, Y, cell_type = prepare_h5ad(file_name)
    else:
        raise Exception("Please Input Proper Data Type!")

    adata = sc.AnnData(X, dtype=np.float32)
    adata.obs['Group'] = Y
    adata.obs['annotation'] = cell_type

    adata, adata_hvg = normalize(adata, 
                                 copy=True,
                                 flavor=None,
                                 highly_genes=num_genes,
                                 normalize_input=True,
                                 logtrans_input=True,
                                 scale_input=scale)
    
    x_ndarray = adata_hvg.to_df().values.squeeze()
    y_ndarray = adata.obs["Group"].values

    print(f'X Shape: {x_ndarray.shape}, Y Shape: {y_ndarray.shape}')

    return x_ndarray, y_ndarray, adata_hvg, cell_type

def cal_nn(x, k=30, max_element=95536):
    p = hnswlib.Index(space='cosine', dim=x.shape[1])
    p.init_index(max_elements = max_element, ef_construction = 300, M = 100)
    p.set_num_threads(10)
    p.set_ef(600)
    p.add_items(x)

    neighbors, distance = p.knn_query(x, k = k)
    neighbors = neighbors[:, 1:]
    distance = distance[:, 1:]

    return neighbors, distance

def get_anchor(adata, 
               adata_embedding, 
               pseudo_label='leiden', 
               seed=42,
               k=30, 
               percent=0.4, 
               max_element=95536):
    max_element = max(max_element, adata_embedding.shape[0] + 1)
    _, distance = cal_nn(x=adata_embedding.X, k=k, max_element=max_element)
    
    mean_distance = np.mean(distance, axis=1)
    
    dis_col_name = f"{pseudo_label}_distance"
    adata.obs[dis_col_name] = mean_distance
    adata_embedding.obs[dis_col_name] = mean_distance
    
    anchor_cells = []
    for c in np.unique(adata_embedding.obs[pseudo_label]):
        cell_type = adata_embedding.obs[adata_embedding.obs[pseudo_label] == c]
        num_cells = math.ceil(percent * cell_type.shape[0])
        threshold = np.sort(cell_type[dis_col_name].values)[num_cells]
        cells = cell_type.index[np.where(cell_type[dis_col_name] <= threshold)[0]].to_list()
        
        if len(cells) > num_cells:
            random.seed(seed)
            selected_cells = random.sample(cells, num_cells)
        else:
            selected_cells = cells
        
        anchor_cells.extend(selected_cells)
    
    non_anchor_cells = list(set(adata.obs_names) - set(anchor_cells))
    
    adata_embedding.obs.loc[anchor_cells, f"{pseudo_label}_density_status"] = "low"
    adata_embedding.obs.loc[non_anchor_cells, f"{pseudo_label}_density_status"] = "high"
    
    adata.obs.loc[anchor_cells, f"{pseudo_label}_density_status"] = "low"
    adata.obs.loc[non_anchor_cells, f"{pseudo_label}_density_status"] = "high"
    
    return adata, adata_embedding