import os
import argparse
   
from loaders import data_process
from moco import Encoder, MoCo
from utils import yaml_config_hook, set_seed
from save import save_model
from evaluation import *
 
import torch
from torch import nn

   
def pretrain_epoch(model, 
                   data, 
                   neighbors, 
                   batch_size, 
                   criterion, 
                   optimizer, 
                   device, 
                   c=1, 
                   flag='aug_nn'):
    loss_epoch = 0.0
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    count = 0

    # model.train()
    print(f"Model Traning Phase: {model.training}")
    for step, pre_index in enumerate(range(data.shape[0] // batch_size + 1)):
        indices_idx = np.arange(pre_index * batch_size, min(data.shape[0], (pre_index + 1) * batch_size))
        
        if len(indices_idx) < batch_size:
            continue
        
        count += 1

        batch_indices = indices[indices_idx]
        x = data[batch_indices]
        x = torch.FloatTensor(x).to(device)

        # Use Neighbors as positive instances
        if neighbors is not None:
            batch_nei = neighbors[batch_indices]
            batch_nei_idx = np.array([np.random.choice(nei, c) for nei in batch_nei])
            batch_nei_idx = batch_nei_idx.flatten()
            
            x_nei = data[batch_nei_idx]
            x_nei = torch.FloatTensor(x_nei).to(device)
        
        assert x_nei.size(0) // x.size(0) == c

        if flag == 'aug_nn':   # Using its augmentation counterpart and neighbor to form positive pairs
            loss = model(x, x_nei, flag=flag)
        else:   # Only using its augmentation to form positive pairs
            out1, out2 = model(x, x, flag=flag)   
            loss = criterion(out1, out2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step [{step}/{data.shape[0]}]\t loss_instance: {loss.item()}")
        
        loss_epoch += loss.item()
    
    loss_epoch = loss_epoch / count

    return loss_epoch

def pretrain(args, device):
    data, _, neighbors = data_process(args.root_dir, 
                                      args.data_type,
                                      args.dataset_name,
                                      args.num_genes,
                                      k=args.n,
                                      max_element=args.max_element,
                                      scale=False)
    
    print(f"Data Size: {data.shape}")
    print(f"Neighbors Size: {neighbors.shape}")
    in_features = data.shape[1]
    
    model = MoCo(Encoder,
                 in_features=in_features,
                 num_cluster=args.classnum,
                 latent_features=args.latent_feature, 
                 device=device,
                 mlp=True,
                 K=args.K,
                 m=args.m,
                 T=args.T,
                 p=args.p,
                 lam=args.lam,
                 alpha=args.alpha)
        
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=args.learning_rate, 
                                 weight_decay=0.0)

    model_path = os.path.join(args.model_path, f"seed_{args.seed}")
    if args.reload:
        model_fp = os.path.join(model_path, "checkpoint_{}.tar".format(args.start_epoch))
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1

    # pretrain
    for epoch in range(args.start_epoch, args.epochs+1):
        loss_epoch = pretrain_epoch(model, 
                                    data,
                                    neighbors,
                                    args.batch_size, 
                                    criterion, 
                                    optimizer, 
                                    device,
                                    c=args.c,
                                    flag=args.flag)

        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch}")
        print('-' * 60)

        if epoch % 10 == 0:
            model.eval()
        
        # save_model(model_path, model, optimizer, epoch)
    save_model(model_path, model, optimizer, args.epochs)

    return model
 
def main():
    parser = argparse.ArgumentParser()
    # config = yaml_config_hook("config/config_Adam.yaml")
    # config = yaml_config_hook("config/config_Bach.yaml")
    config = yaml_config_hook("config/config_Baron_human.yaml")
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
    # config = yaml_config_hook("config/config_Tosches_turtle.yaml")

    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    print(f"Current Random seed {args.seed}")
    print(f"Current Flag: {args.flag}")

    set_seed(args.seed)

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(device)

    model = pretrain(args, device=device)
    
if __name__ == "__main__":
    main()