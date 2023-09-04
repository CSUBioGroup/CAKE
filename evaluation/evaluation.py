import os

import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from sklearn import metrics
from munkres import Munkres
from sklearn.cluster import KMeans

sc.settings.verbosity = 0


def evaluate_kmeans(label, pred):
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    f = metrics.fowlkes_mallows_score(label, pred)

    pred_adjusted = get_y_preds(label, pred, len(set(label)))
    acc = metrics.accuracy_score(pred_adjusted, label)

    return nmi, ari, f, acc


def evaluate_leiden(label, pred):
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    f = metrics.fowlkes_mallows_score(label, pred)

    return nmi, ari, f


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))

    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t

    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)

    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]

    return cluster_labels


def get_y_preds(y_true, cluster_assignments, n_clusters):
    """
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)
    cluster_assignments:    array of labels, outputted by cluster
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset
    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    """
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels

    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    pred_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)

    y_pred = pred_to_true_cluster_labels[cluster_assignments]

    return y_pred


def run_leiden(latent_vector, resolution=0.6):
    adata = sc.AnnData(latent_vector, dtype=np.float32)

    sc.pp.neighbors(adata, n_neighbors=20, n_pcs=0, use_rep='X')
    sc.tl.leiden(adata, resolution=resolution)

    leiden_pred = adata.obs['leiden'].astype("int32")
    
    return adata, leiden_pred


def run_kmeans(latent_vector, n_clusters, random_state=0):
    kmeans = KMeans(init="k-means++", 
                    n_init=10,
                    max_iter=100,
                    n_clusters=n_clusters, 
                    random_state=random_state)

    pred = kmeans.fit_predict(latent_vector)

    return pred


def embedding_cluster_visualization_sc(adata, 
                                       dataset_name,
                                       label, 
                                       pred,
                                       nmi_kmeans,
                                       ari_kmeans,
                                       nmi_leiden,
                                       ari_leiden,
                                       epoch,
                                       seed,
                                       dir_path_name="pictures"):
    
    _, axs = plt.subplots(2, 2, figsize=(18, 16))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    for row in axs:
        for ax in row:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    adata.obs['label'] = label
    adata.obs['kmeans'] = pred

    adata.obs["label"] = adata.obs['label'].astype("category")
    adata.obs['kmeans'] = adata.obs['kmeans'].astype('category')
    
    label_title = f'Epoch {epoch} with True Label'

    kmeans_title = f'Epoch {epoch} with KMeans Prediction\n NMI: {nmi_kmeans:.4f} ARI: {ari_kmeans:.4f}'

    leiden_title = f"Epoch {epoch} with Leiden Label\n NMI: {nmi_leiden:.4f} ARI:{ari_leiden:.4f}"

    colors = [['cell_type', 'kmeans'], ['label', 'leiden']]
    titles = [[label_title, kmeans_title], [label_title, leiden_title]]
    
    dir_path = os.path.join(dir_path_name, dataset_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # UMAP Plotting    
    sc.tl.umap(adata)

    for i in range(len(colors)):
        for j in range(len(colors[0])):
            sc.pl.umap(adata, color=colors[i][j], title=titles[i][j], ax=axs[i, j], show=False)
            
            if i == 0 and j == 0:
                axs[i, j].legend(frameon=False,
                                 loc='center',
                                 bbox_to_anchor=(0.50, -0.12),
                                 ncol= 3,
                                 fontsize=8,
                                 markerscale=0.9)
            else:
                axs[i, j].legend(frameon=False,
                                loc='center',
                                bbox_to_anchor=(0.50, -0.12),
                                ncol=len(np.unique(label)) // 2,
                                fontsize=8,
                                markerscale=0.9)
        
    path_2_save = os.path.join(dir_path, f"Epoch_{epoch}_{seed}.png")

    
    plt.savefig(path_2_save, dpi=160.0)

def plot(adata_embedding, 
         Y, 
         dataset_name, 
         epoch, 
         seed = 42, 
         colors=['leiden', 'label'],
         titles=['Leiden Predictions', 'True Label'],
         dir_path_name="pictures"):
    sc.settings.verbosity = 0
    sc.settings.set_figure_params(dpi=160)

    _, axes = plt.subplots(1, 2, figsize=(18, 8))
    plt.subplots_adjust(wspace=0.3)

    dir_path = os.path.join(dir_path_name, dataset_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for ax in axes:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    if 'X_umap' not in adata_embedding.obsm_keys():
        sc.tl.umap(adata_embedding)

    sc.pl.umap(adata_embedding, 
               color=colors[0], 
               title=titles[0], 
               wspace=0.36, 
               show=False,
               ax=axes[0])

    sc.pl.umap(adata_embedding, 
                color=colors[1], 
                title=titles[1], 
                wspace=0.36, 
                show=False,
                ax=axes[1])

    axes[0].legend(frameon=False, 
                    loc='center', 
                    bbox_to_anchor=(0.50, -0.12),
                    ncol=len(np.unique(Y)) // 2, 
                    fontsize=8,
                    markerscale=0.9) 
    
    if "annotation" in colors:
        axes[1].legend(frameon=False, 
                       loc='center', 
                       bbox_to_anchor=(0.50, -0.12),
                       ncol=3, 
                       fontsize=8,
                       markerscale=0.9)
    else:
        axes[1].legend(frameon=False, 
                        loc='center', 
                        bbox_to_anchor=(0.50, -0.12),
                        ncol=len(np.unique(Y)) // 2, 
                        fontsize=8,
                        markerscale=0.9)

    plt.grid(False)

    path_2_save = os.path.join(dir_path, f"{titles[0]}_Epoch_{epoch}_{seed}.png")

    plt.savefig(path_2_save, dpi=160.0)