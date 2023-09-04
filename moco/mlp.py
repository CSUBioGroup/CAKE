import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,
                 in_features,
                 num_cluster,
                 latent_features = [1024, 512, 128],
                 device="cpu",
                 p=0.0):
        super().__init__()
        self.in_features = in_features
        self.latent_features = latent_features
        self.device = device

        layers = []
        layers.append(nn.Dropout(p=p))
        for i in range(len(latent_features)):
            if i == 0:
                layers.append(nn.Linear(in_features, latent_features[i]))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(latent_features[i-1], latent_features[i]))
                layers.append(nn.ReLU())
        
        layers = layers[:-1]
        self.encoder = nn.Sequential(*layers)

        self.fc = nn.Linear(latent_features[-1], num_cluster)
        
    def forward(self, x):
        h = self.encoder(x)
        out = self.fc(h)

        return out
    
    def get_embedding(self, x):
        latent = self.encoder(x)

        return latent