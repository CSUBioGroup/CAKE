import os
import scanpy as sc
import scipy as sp
import numpy as np
import pandas as pd
from utils.tools import *

def prepare_npz(file_name):
    dname = file_name.split("/")[-1]
    
    X = sp.sparse.load_npz(os.path.join(file_name, "filtered_Counts.npz"))
    X = sp.sparse.csr_matrix(X)
    X = X.toarray()
    annoData = pd.read_table(os.path.join(file_name, "annoData.txt"))

    if dname == "HCF-spleen":
        Y = annoData['cellIden'].to_numpy()
        cell_type = list(map(str, Y))
    else:
        if "cellIden3" in annoData.columns.to_list():
            Y = annoData['cellIden3'].to_numpy()
        else:
            Y = annoData['cellIden'].to_numpy()

        if "cellAnno3" in annoData.columns.to_list():
            cell_type = annoData['cellAnno3'].to_list()
        else:
            cell_type = annoData['cellAnno'].to_list()

    return X, Y, cell_type

def prepare_h5ad(file_name):
    adata = sc.read_h5ad(os.path.join(file_name, "data.h5ad"))
    X = sp.sparse.csr_matrix(adata.X)
    X = X.toarray()
    
    print(adata)
    if file_name.split("/")[-1] == "Pancreas":
        cell_name = np.array(adata.obs['clusters'])
    else:
        cell_name = np.array(adata.obs["cell_type"])
        
    cell_type, Y = np.unique(cell_name, return_inverse=True)

    return X, Y, cell_name

def prepare_h5(file_name):
    import h5py
    data_mat = h5py.File(os.path.join(file_name, "data.h5"), "r")

    X = np.array(data_mat['X'])
    cell_name = np.array(data_mat['Y'])

    cell_type, Y = np.unique(cell_name, return_inverse=True)

    return X, Y, cell_name

def prepare_nested_h5(file_name):
    X, Y, cell_type = prepro(file_name)

    X = np.ceil(X).astype(np.int32)


    return X, Y, cell_type