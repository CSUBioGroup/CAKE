import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
 
import os
import argparse
import time

import prettytable as pt

from loaders import *
from moco import *
from read_data import *
from utils import *
from evaluation import *
from pretrain import *
from cluster import *

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import torch
from torch import nn
from torch.utils.data import DataLoader

def train_distiller(train_loader, student, teacher, criterion, optimizer, device):
    student = student.to(device)
    teacher = teacher.to(device)
    loss_epoch = 0.0
    count = 0

    for i, (x, pseu, _) in enumerate(train_loader):
        x = x.to(device)
        pseu = pseu.to(device)
        
        count += 1
        optimizer.zero_grad()
        
        with torch.no_grad():
            teacher_logits = teacher(x)
        student_logits = student(x)
        
        loss = criterion(student_logits, teacher_logits, pseu)
        
        loss.backward()
        optimizer.step()
        
        if i % 50 == 0:
            print(f"Step [{i}]\t loss_instance: {loss.item()}")
         
        loss_epoch += loss.item()

    loss_epoch = loss_epoch / count
        
    return loss_epoch

def train_teacher(train_loader, teacher, criterion, optimizer, device, epoch):
    # batch_time = AverageMeter("Time", ":6.3f")
    # data_time = AverageMeter("Data", ":6.3f")
    # losses = AverageMeter("Loss", ":.4e")
    # top1 = AverageMeter("Acc@1", ":6.2f")
    # progress = ProgressMeter(len(train_loader),
    #                          [batch_time, data_time, losses, top1],
    #                          prefix="Epoch: [{}]".format(epoch))


  
    teacher = teacher.to(device)
    loss_epoch = 0.0
    count = 0
    
    teacher.eval()
    end = time.time()
    for i, (x, pseu, y) in enumerate(train_loader):
        # data_time.update(time.time() - end)
        x = x.to(device)
        pseu = pseu.to(device)

        count += 1

        out = teacher(x)
        loss = criterion(out, pseu)
        
        # acc1 = accuracy(out, pseu, topk=(1, ))
        # losses.update(loss.item(), x.size(0))
        # top1.update(acc1[0].item(), x.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # batch_time.update(time.time() - end)
        # end = time.time()

        if i % 50 == 0:
            print(f"Step [{i}]\t loss_instance: {loss.item()}")
        #     progress.display(i)

        loss_epoch += loss.item()

    loss_epoch = loss_epoch / count

    return loss_epoch
    
def validate_teacher(val_loader, model, criterion, device):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    progress = ProgressMeter(len(val_loader), 
                             [batch_time, losses, top1], 
                             prefix="Test: ")
    
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (x, pseu, y) in enumerate(val_loader):
            x = x.to(device)
            pseu = pseu.to(device)
            
            output = model(x)
            loss = criterion(output, pseu)
            acc1 = accuracy(output, pseu, topk=(1, ))
            
            losses.update(loss.item(), x.size(0))
            top1.update(acc1[0].item(), x.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                progress.display(i)

        print(f"* Acc@1 {top1.avg:.3f}")

    return top1.avg

def get_prediction(model, device, val_loader):
    model.eval()
    pred = []
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            _, res = torch.max(output, dim=1)
            res = res.detach().cpu().numpy()
            
            pred.extend(res)
    
    return pred

def main(dname):
    parser = argparse.ArgumentParser()
    config = yaml_config_hook(f"config_test/config_{dname}.yaml")

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
    
    # start_time1 = time.time()
    # 1. pretrain moco 
    print("---------- Step 1: Pretrain MoCo ----------")
    model, start_time1 = pretrain(args, device=device)

    # 2. get pseudo label 
    print('---------- Step 2: Get Pseudo Labels ----------')
    adata_embedding, adata, Y, leiden_pred, _, val_loader = get_pseudo_label(args,
                                                                             model, 
                                                                             device=device)
    end_time1 = time.time()
    plot(adata_embedding, 
         Y, 
         args.dataset_name, 
         epoch=args.epochs, 
         seed=args.seed,
         dir_path_name="pictures")

    # 3. train teacher model
    # 3.1 data prepare
    print('---------- Step 3: Train & Evaluate Teacher Model ----------')
    adata, adata_embedding = get_anchor(adata, 
                                        adata_embedding, 
                                        pseudo_label='leiden',
                                        k=30, 
                                        percent=0.5)
    train_adata = adata[adata.obs.leiden_density_status == 'low', :].copy()
    test_adata = adata[adata.obs.leiden_density_status == 'high', :].copy()

    pseudo_labels = np.array(list(map(int, train_adata.obs['leiden'].values)))
    print(f"extracted_nmi: {normalized_mutual_info_score(train_adata.obs['Group'].values, pseudo_labels):.4f}")
    print(f"extracted_ari : {adjusted_rand_score(train_adata.obs['Group'].values, pseudo_labels):.4f}")

    train_dataset = CellDatasetPseudoLabel(train_adata, 
                                           pseudo_label='leiden', 
                                           oversample_flag=True)
    test_dataset = CellDatasetPseudoLabel(test_adata,
                                          pseudo_label='leiden', 
                                          oversample_flag=False)

    print(f"teacher train dataset: {len(train_dataset)}")
    print(f"teacher test dataset: {len(test_dataset)}")

    # 3.2 build KD model
    in_features = adata.shape[1]
    teacher = Encoder(in_features=in_features,
                      num_cluster=len(np.unique(leiden_pred)),
                      latent_features=args.latent_feature,
                      device=device,
                      p=args.p)
 
    student = Encoder(in_features=in_features,
                      num_cluster=len(np.unique(leiden_pred)),
                      latent_features=args.latent_feature,
                      device=device,
                      p=args.p)

    # 3.3 loader pretrained weight for teacher model
    for name, param in teacher.named_parameters():
        if name not in ["fc.weight", "fc.bias"]:
            param.requires_grad = False
        
    teacher.fc.weight.data.normal_(mean=0.0, std=0.01)
    teacher.fc.bias.data.zero_()

    epoch = args.epochs
    model_path = os.path.join(args.model_path, f"seed_{args.seed}")
    model_fp = os.path.join(model_path, f"checkpoint_{epoch}.tar")
    state_dict = torch.load(model_fp, map_location="cuda:1")['net']

    for k in list(state_dict.keys()):
        if k.startswith("encoder_k.") and not k.startswith("encoder_k.fc"):
            state_dict[k[len("encoder_k."):]] = state_dict[k]
    
        del state_dict[k]
    
    print(list(state_dict.keys()))
    msg = teacher.load_state_dict(state_dict, strict=False)
    print(set(msg.missing_keys))

    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers)
    teacher_criterion = nn.CrossEntropyLoss()
    parameters = list(filter(lambda p : p.requires_grad, teacher.parameters()))

    assert len(parameters) == 2

    teacher_optimizer = torch.optim.Adam(parameters, 
                                         lr=args.learning_rate, 
                                         weight_decay=0.0)
    val_teacher_loader = DataLoader(test_dataset, 
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)
    
    # 4. train teacher model
    start_time2 = time.time()
    epochs = 100
    best_acc1 = 0.0
    for epoch in range(args.start_epoch, epochs+1):
        loss_epoch = train_teacher(train_loader, 
                                   teacher, 
                                   teacher_criterion, 
                                   teacher_optimizer, 
                                   device, 
                                   epochs)
        
        # evaluate on validation set
        # acc1 = validate_teacher(val_teacher_loader, teacher, teacher_criterion, device)
        # remember best acc@1 and save checkpoint
        # best_acc1 = max(acc1, best_acc1)
        
        print(f"Epoch [{epoch}/{epochs}]\t Loss: {loss_epoch}")
        print('-' * 60)
        
    # print(f"Best Accuracy: {best_acc1}")

    # 4.2 evaluation tearcher model performance
    teacher_pred = get_prediction(teacher, device, val_loader)
    teacher_pred = np.array(teacher_pred, dtype=np.int32)
    adata_embedding.obs['teacher_prediction'] = teacher_pred
    adata_embedding.obs['teacher_prediction'] = adata_embedding.obs['teacher_prediction'].astype('category')
    
    teacher_ari = adjusted_rand_score(Y, teacher_pred)
    teacher_nmi = normalized_mutual_info_score(Y, teacher_pred)

    leiden_ari = adjusted_rand_score(Y, leiden_pred)
    leiden_nmi = normalized_mutual_info_score(Y, leiden_pred)

    # 5. train distiller
    print('---------- Step 4: Train & Evaluate Distiller ----------')
    distiller_loss = DistillerLoss(alpha=args.kd_alpha, 
                                   temperature=args.kd_temperature)
    distiller_optimizer = torch.optim.Adam(student.parameters(), 
                                           lr=args.learning_rate, 
                                           weight_decay=0.0)

    # freeze parameters in teacher model
    for name, param in teacher.named_parameters():
        param.requires_grad = False
        
    epochs = 50
    for epoch in range(args.start_epoch, epochs+1):
        loss_epoch = train_distiller(train_loader, 
                                     student, 
                                     teacher, 
                                     distiller_loss, 
                                     distiller_optimizer, 
                                     device)

        print(f"Epoch [{epoch}/{epochs}]\t Loss: {loss_epoch}")
        print('-' * 60)
    end_time2 = time.time()

    # 5.2 evalation student model
    student_pred = get_prediction(student, device, val_loader)
    student_pred = np.array(student_pred, dtype=np.int32)
    adata_embedding.obs['student_prediction'] = student_pred
    adata_embedding.obs['student_prediction'] = adata_embedding.obs['student_prediction'].astype("category")

    student_ari = adjusted_rand_score(Y, student_pred)
    student_nmi = normalized_mutual_info_score(Y, student_pred)

    # save result 
    tb = pt.PrettyTable()
    tb.field_names = ['Method Name', 'ARI', 'NMI', 'time']
    time1 = end_time1 - start_time1
    time2 = end_time2 - start_time2
    tb.add_row(['Leiden', round(leiden_ari, 4), round(leiden_nmi, 4), time1])
    tb.add_row(['Teacher', round(teacher_ari, 4), round(teacher_nmi, 4), time2])
    tb.add_row(['Student', round(student_ari, 4), round(student_nmi, 4), time1 + time2])

    print(tb)

    result_dir = f"validation_{args.seed}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    file_name = os.path.join(result_dir, f"{args.dataset_name}.txt")
    f = open(file_name, "a")
    current_time = time.strftime("%Y_%m_%d %H_%M_%S", time.localtime(time.time()))

    f.write(f"Epoch {args.epochs} -----------> {current_time}\n")
    f.write(str(tb) + '\n')
    f.close()

    # visualize result
    plot(adata_embedding, 
         Y, 
         args.dataset_name, 
         args.epochs, 
         seed=args.seed, 
         colors=['student_prediction', 'annotation'],
         titles=['Student Prediction', 'Cell Type'],
         dir_path_name="pictures")
    
    plot(adata_embedding, 
         Y, 
         args.dataset_name, 
         args.epochs, 
         seed=args.seed, 
         colors=['teacher_prediction', 'student_prediction'],
         titles=['Teacher Prediction', 'Student Prediction'],
         dir_path_name="pictures")

    # save embeddings
    write_path = f"data_embeddings/{dname}_{args.seed}.h5ad"
    adata_embedding.write_h5ad(write_path)


if __name__ == "__main__":
    dname = "Muraro"
    main(dname)
