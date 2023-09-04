import os
import hnswlib
import scanpy as sc
import numpy as np

from utils.tools import *
from read_data import *

def data_process(root_dir, 
                 data_type, 
                 dataset_name, 
                 num_genes, 
                 k=6, 
                 max_element=95536, 
                 scale=True):
    file_name = os.path.join(root_dir, f"{data_type}_dir", dataset_name)
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
    
    x_array = adata_hvg.to_df().values
    y_array = adata_hvg.obs['Group'].values

    print(f"X shape: {x_array.shape}")
    print(f"Y shape: {y_array.shape}")
    
    if k > 0:
        neighbors, _ = cal_nn(x_array, k=k, max_element=max_element)
    else:
        return x_array, y_array, None

    return x_array, y_array, neighbors
    

def cal_nn(x, k=500, max_element=95536):
    p = hnswlib.Index(space='cosine', dim=x.shape[1])
    p.init_index(max_elements=max_element, 
                 ef_construction=600, 
                 random_seed=600,
                 M=100)
    
    p.set_num_threads(20)
    p.set_ef(600)
    p.add_items(x)

    neighbors, distance = p.knn_query(x, k = k)
    neighbors = neighbors[:, 1:]
    distance = distance[:, 1:]

    return neighbors, distance

    