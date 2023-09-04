import os
import h5py
import pandas as pd
import numpy as np
import scanpy as sc
import scipy as sp

# Some of the following codes are adapted according to the scziDesk Model github
# https://github.com/xuebaliang/scziDesk 

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def empty_safe(fn, dtype):
    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)
    return _fn


decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)
encode = empty_safe(np.vectorize(lambda _x: str(_x).encode("utf-8")), "S")
upper = empty_safe(np.vectorize(lambda x: str(x).upper()), str)
lower = empty_safe(np.vectorize(lambda x: str(x).lower()), str)
tostr = empty_safe(np.vectorize(str), str)


def read_clean(data):
    assert isinstance(data, np.ndarray)
    if data.dtype.type is np.bytes_:
        data = decode(data)
    if data.size == 1:
        data = data.flat[0]
    return data


def dict_from_group(group):
    assert isinstance(group, h5py.Group)
    d = dotdict()
    for key in group:
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = read_clean(group[key][...])
        d[key] = value
    return d


def read_data(filename, sparsify = False, skip_exprs = False):
    with h5py.File(filename, "r") as f:
        obs = pd.DataFrame(dict_from_group(f["obs"]), index = decode(f["obs_names"][...]))
        var = pd.DataFrame(dict_from_group(f["var"]), index = decode(f["var_names"][...]))
        uns = dict_from_group(f["uns"])

        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sp.sparse.csr_matrix((exprs_handle["data"][...], 
                                            exprs_handle["indices"][...],
                                            exprs_handle["indptr"][...]), 
                                            shape = exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sp.sparse.csr_matrix(mat)
        else:
            mat = sp.sparse.csr_matrix((obs.shape[0], var.shape[0]))

    return mat, obs, var, uns


def prepro(filename):
    data_path = os.path.join(filename, "data.h5")
    mat, obs, var, uns = read_data(data_path, sparsify=False, skip_exprs=False)

    if isinstance(mat, np.ndarray):
        X = np.array(mat)
    else:
        X = np.array(mat.toarray())

    cell_name = np.array(obs["cell_type1"])
    cell_type, cell_label = np.unique(cell_name, return_inverse=True)

    return X, cell_label, cell_name


def normalize(adata, 
              copy=True,
              flavor=None, 
              highly_genes=None, 
              filter_min_counts=True, 
              normalize_input=True, 
              logtrans_input=True,
              scale_input=False):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    else:
        raise NotImplementedError

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_cells=3)

    if normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if flavor == 'seurat_v3':
        print("seurat_v3")
        sc.pp.highly_variable_genes(adata, flavor=flavor, n_top_genes = highly_genes)

    if normalize_input:
        sc.pp.normalize_total(adata, target_sum=1e4)

    if logtrans_input:
        sc.pp.log1p(adata)

    if flavor is None:
        if highly_genes is not None:
            print("routine hvg")
            sc.pp.highly_variable_genes(adata, n_top_genes=highly_genes)
        else:
            sc.pp.highly_variable_genes(adata)
    
    adata_hvg = adata[:, adata.var.highly_variable].copy()

    if scale_input:
        sc.pp.scale(adata_hvg)

    return adata, adata_hvg