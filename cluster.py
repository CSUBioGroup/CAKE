import os
import argparse
import time
import prettytable as pt
    
from loaders import CellDataset, prepareAll
from moco import Encoder, MoCo
from utils import yaml_config_hook, set_seed
from evaluation import *

import torch
from torch import nn
from torch.utils.data import DataLoader

 
def inference(args, model, device):
    model.eval()

    x, y, adata, cell_type = prepareAll(root_dir=args.root_dir,
                                        data_type=args.data_type,
                                        dataset_name=args.dataset_name,
                                        num_genes=args.num_genes,
                                        scale=False)

    val_datasets = CellDataset(x, y)
    in_features = val_datasets.data.size(1)

    print(f"Validation Dataset size: {len(val_datasets)}")
    print(f"The in_features is: {in_features}")

    val_loader = DataLoader(val_datasets,
                            batch_size=256,
                            shuffle=False,
                            num_workers=args.workers,
                            drop_last=False)

    labels_vector = []
    latent_vector = []

    for step, (x, y) in enumerate(val_loader):
        x = x.to(device)

        with torch.no_grad():
            latent = model.get_embedding(x)

        latent = latent.detach()

        latent_vector.extend(latent.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 50 == 0:
            print(f"Step [{step}/{len(val_loader)}]\t Computing features...")

    labels_vector = np.array(labels_vector)
    latent_vector = np.array(latent_vector)

    return labels_vector, latent_vector, cell_type, adata, val_loader

def get_pseudo_label(args, model, device):
    Y, latent, cell_type, adata, val_loader = inference(args, model, device)

    print("### Performming Leiden clustering method on latent vector ###")
    adata_embedding, leiden_pred = run_leiden(latent_vector=latent, 
                                              resolution=args.resolution)
    
    print("### Performming KMeans clustering method on latent vector ###")
    kmeans_pred = run_kmeans(latent, args.classnum, random_state=args.seed)

    adata_embedding.obs['label'] = Y
    adata_embedding.obs['label'] = adata_embedding.obs['label'].astype("category")
    adata_embedding.obs['annotation'] = cell_type
    adata_embedding.obs['annotation'] = np.array(list(map(str, adata.obs['annotation'].values)))

    adata.obs['label'] = Y
    adata.obs['label'] = adata.obs['label'].astype("category")
    adata_embedding.obs['kmeans'] = kmeans_pred
    adata_embedding.obs['kmeans'] = adata_embedding.obs['kmeans'].astype("category")

    adata.obs['kmeans'] = kmeans_pred
    adata.obs['kmeans'] = adata.obs['kmeans'].astype("category")

    adata.obs['leiden'] = adata_embedding.obs['leiden'].values

    return adata_embedding, adata, Y, leiden_pred, kmeans_pred, val_loader


def validate(model, 
             data_loader, 
             device, 
             epoch,
             cell_type,
             seed, 
             dataset_name, 
             n_clusters, 
             resolution=0.6, 
             dir_path_name="validation_pics"):
    print("### Creating features from model ###")
    Y, latent_vector = inference(data_loader, model, device)
    n_clusters = len(np.unique(Y))
    
    print("### Performming Kmeans clustering method on latent vector ###")
    kmeans_pred = run_kmeans(latent_vector=latent_vector, 
                             n_clusters=n_clusters,
                             random_state=seed)
    
    print("### Performming Leiden clustering method on latent vector ###")
    adata, leiden_pred = run_leiden(latent_vector=latent_vector, 
                                    resolution=resolution)
    adata.obs['cell_type'] = cell_type
    adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')

    # Evaluation 
    k_nmi, k_ari, k_f, k_acc = evaluate_kmeans(Y, kmeans_pred)
    nmi_leiden, ari_leiden, f_leiden = evaluate_leiden(Y, leiden_pred)

    # Visualization
    dir_path_name = f"{dir_path_name}_seed_{seed}"
    embedding_cluster_visualization_sc(adata, 
                                       dataset_name,
                                       Y,
                                       kmeans_pred,
                                       k_nmi,
                                       k_ari,
                                       nmi_leiden,
                                       ari_leiden, 
                                       epoch,
                                       seed,
                                       dir_path_name=dir_path_name)
    # Print out related results
    print(f"### Epoch {epoch} Results: ###")

    tb = pt.PrettyTable()
    tb.field_names = ["Method Name", "NMI", "ARI", "F", "Accuracy"]
    
    tb.add_row(["KMeans", round(k_nmi, 4), round(k_ari, 4), round(k_f, 4), round(k_acc, 4)])
    tb.add_row(["Leiden", round(nmi_leiden, 4), round(ari_leiden, 4), round(f_leiden, 4), 0.0])

    print(tb)
    
    # Write out the evaluation results
    result_dir = f"validation_{seed}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    file_name = os.path.join(result_dir, f"{dataset_name}.txt")
    f = open(file_name, "a")
    current_time = time.strftime('%Y_%m_%d %H_%M_%S',time.localtime(time.time()))

    f.write(f"Epoch {epoch} -------> {current_time}\n")
    f.write(str(tb) + '\n')
    f.close()

 
def main():
    parser = argparse.ArgumentParser()
    # config = yaml_config_hook("config/config_10x_PBMC.yaml")
    # config = yaml_config_hook("config/config_Adam.yaml")
    # config = yaml_config_hook("config/config_Bach.yaml")
    # config = yaml_config_hook("config/config_Baron_human.yaml")
    # config = yaml_config_hook("config/config_Bone_Marrow.yaml")
    # config = yaml_config_hook("config/config_Fat.yaml")
    # config = yaml_config_hook("config/config_Guo.yaml")
    # config = yaml_config_hook("config/config_HCF-spleen.yaml")
    # config = yaml_config_hook("config/config_Hrvatin.yaml")
    # config = yaml_config_hook("config/config_Macosko.yaml")
    # config = yaml_config_hook("config/config_Mammary_Gland.yaml")
    # config = yaml_config_hook("config/config_Muraro.yaml")
    # config = yaml_config_hook("config/config_Plasschaert.yaml")
    # config = yaml_config_hook("config/config_QS_Heart.yaml")
    # config = yaml_config_hook("config/config_Qx_Spleen.yaml")
    # config = yaml_config_hook("config/config_Qx_Trachea.yaml")
    # config = yaml_config_hook("config/config_Shekhar.yaml")
    config = yaml_config_hook("config/config_Tosches_turtle.yaml")
    
    # config = yaml_config_hook("config/config_BRCA_100K.yaml")

    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    set_seed(args.seed)
    model_path = os.path.join(args.model_path, f"seed_{args.seed}")

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)

    x, y, _, cell_type = prepareAll(root_dir=args.root_dir,
                                    data_type=args.data_type,
                                    dataset_name=args.dataset_name,
                                    num_genes=args.num_genes,
                                    scale=False)

    val_datasets = CellDataset(x, y)
    in_features = val_datasets.data.size(1)

    print(f"Validation Dataset size: {len(val_datasets)}")
    print(f"The in_features is: {in_features}")

    val_loader = DataLoader(val_datasets,
                            batch_size=256,
                            shuffle=False,
                            num_workers=args.workers,
                            drop_last=False)
    
    model = MoCo(Encoder,
                 in_features=in_features,
                 num_cluster=args.classnum,
                 latent_features=args.latent_feature, 
                 device=device,
                 mlp=True,
                 K=args.K,
                 m=args.m,
                 T=args.T,
                 p=args.p)
        
    # validation
    for epoch in range(1, args.epochs + 1, 1):
        model_fp = os.path.join(model_path, "checkpoint_{}.tar".format(epoch))
        model.load_state_dict(torch.load(model_fp, map_location="cuda:1")['net'])
        model.to(device)

        validate(model=model,
                 data_loader=val_loader,
                 device=device,
                 epoch=epoch,
                 cell_type=cell_type,
                 seed=args.seed,
                 dataset_name=args.dataset_name,
                 n_clusters=args.classnum,
                 resolution=args.resolution)

if __name__ == "__main__":
    main()