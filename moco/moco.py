import torch
from torch import nn
import torch.nn.functional as F
 
class DistillerLoss:
    def __init__(self, alpha=0.1, temperature=3):
        self.alpha = alpha
        self.temperature = temperature
        self.student_loss = nn.CrossEntropyLoss()
        self.distillation_loss = nn.KLDivLoss()
        
    def __call__(self, student_logits, teacher_logits, labels):
        distillation_loss = self.distillation_loss(F.log_softmax(student_logits / self.temperature, dim=1),
                                                   F.softmax(teacher_logits / self.temperature, dim=1))
        student_loss = self.student_loss(student_logits, labels)
        
        loss = self.alpha * student_loss + (1 - self.alpha) * self.temperature * self.temperature * distillation_loss
        
        return loss

class MoCo(nn.Module):
    def __init__(self, 
                 encoder,
                 in_features, 
                 num_cluster,
                 latent_features=[1024, 512, 128],
                 device="cpu",
                 mlp=True,
                 K=65536,
                 m=0.999,
                 T=0.9,
                 p=0.0,
                 lam=0.1,
                 alpha=0.1):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T
        self.lam = lam
        self.alpha = alpha
        self.rep_dim = latent_features[-1]
        self.device = device
        
        self.encoder_q = encoder(in_features=in_features,
                                 num_cluster=num_cluster, 
                                 latent_features=latent_features,
                                 device=device,
                                 p=p)
        self.encoder_k = encoder(in_features=in_features, 
                                 num_cluster=num_cluster,
                                 latent_features=latent_features,
                                 device=device,
                                 p=p)
        
        # Projection Head
        if mlp:
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            print(f"dim_mlp: {dim_mlp}")
            
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), 
                nn.BatchNorm1d(dim_mlp),
                nn.ReLU(), 
                nn.Linear(dim_mlp, dim_mlp)
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), 
                nn.BatchNorm1d(dim_mlp),
                nn.ReLU(), 
                nn.Linear(dim_mlp, dim_mlp)
            )

        for param_k, param_q in zip(self.encoder_k.parameters(), self.encoder_q.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        self.register_buffer("queue", 
                             F.normalize(torch.randn(self.K, self.rep_dim, requires_grad=False), dim=1))
        self.ptr = 0
        
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_k, param_q in zip(self.encoder_k.parameters(), self.encoder_q.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)
            param_k.requires_grad = False
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.size(0)
        
        self.queue[self.ptr: self.ptr + batch_size, :] = keys.detach()
        self.ptr = (self.ptr + batch_size) % self.K
        self.queue.requires_grad = False

    def forward_aug_nn(self, x1, x2):
        q = self.encoder_q(x1)
        latent = self.encoder_q.get_embedding(x1)
        q = F.normalize(q, dim=1)

        c = x2.size(0) // x1.size(0)
        qc = q.unsqueeze(1)
        for _ in range(1, c):
            qc = torch.cat([qc, q.unsqueeze(1)], dim=1)
        qc = qc.reshape(-1, q.size(1))

        assert qc.size(0) == x2.size(0)

        with torch.no_grad():
            self._momentum_update_key_encoder()

            k1 = self.encoder_k(x1)
            k2 = self.encoder_k(x2)

            k1 = F.normalize(k1, dim=1)
            k2 = F.normalize(k2, dim=1)

        pos_sim1 = (1 - self.lam) * torch.einsum("ic, ic -> i", [q, k1]).unsqueeze(-1)
        pos_sim2 = (self.lam / c) * torch.einsum("ic, ic -> i", [qc, k2]).unsqueeze(-1)
        pos_sim2 = pos_sim2.reshape(-1, c)

        assert pos_sim2.size(0) == pos_sim1.size(0)

        pos_sim = torch.cat([pos_sim1, pos_sim2], dim=1)
        neg_sim = torch.einsum("ic, jc -> ij", [q, self.queue.clone().detach()])

        loss = -(torch.logsumexp(pos_sim / self.T, dim=1) - torch.logsumexp(neg_sim / self.T, dim=1)).mean()
        penalty = self.alpha * (torch.mean(torch.abs(latent)))
        loss += penalty

        self._dequeue_and_enqueue(k2)

        return loss
    
    def forward(self, x1, x2, flag="aug_nn"):
        if flag == 'aug_nn':
            return self.forward_aug_nn(x1, x2)

        q = self.encoder_q(x1)
        q = F.normalize(q, dim=1)
        
        with torch.no_grad():
            self._momentum_update_key_encoder()

            k = self.encoder_k(x2)
            k = F.normalize(k, dim=1)

        pos_sim = torch.einsum("ic, ic -> i", [q, k]).unsqueeze(-1)
        neg_sim = torch.einsum("ic, jc -> ij", [q, self.queue.clone().detach()])
        
        logits = torch.cat([pos_sim, neg_sim], dim=1) / self.T
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(self.device)

        self._dequeue_and_enqueue(k)

        return logits, labels
     
    def get_embedding(self, x):
        # out = self.encoder_k.get_embedding(x)
        out = self.encoder_q.get_embedding(x)
        
        return out