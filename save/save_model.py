import os
import torch
from collections import OrderedDict

def save_model(model_path, model, optimizer, current_epoch):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    out = os.path.join(model_path, "checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    
    torch.save(state, out)
    

def save_mask(epoch, model, file_dir):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    filename = os.path.join(file_dir, f"mask_{epoch}.pt")
    pruneMask = OrderedDict()

    for name, module in model.named_modules():
        if hasattr(module, "prune_mask"):
            pruneMask[name] = module.prune_mask.cpu().type(torch.bool)

    torch.save({"epoch": epoch, "pruneMask": pruneMask}, filename)

def load_mask(model, state_dict, device):
    for name, module in model.named_modules():
        if hasattr(module, "prune_mask"):
            module.prune_mask.data = state_dict[name].to(device).float()

    return model