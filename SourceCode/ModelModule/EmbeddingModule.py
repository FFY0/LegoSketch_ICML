import torch.nn as nn
import torch
from SourceCode.ModelModule.SubModuleUtil import  RandomHash

class EmbeddingModule(nn.Module):
    def __init__(self, input_dim, output_dim, cons_lower,cons_upper):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.basis_num = self.output_dim * 16
        self.cons_lower = cons_lower
        self.cons_upper = cons_upper
        assert self.cons_lower < self.cons_upper
        self.random_hash = RandomHash(hash_seed=0,hash_range=self.basis_num)
        self.V = torch.nn.Parameter(torch.ones(self.basis_num) * ((self.cons_lower+self.cons_upper)/2), requires_grad=True)
        self.add_noise()
        self.normalize()

    def forward(self, x):
        batch_size = x.shape[0]
        indexes_tensor = self.random_hash(x, sample_size = batch_size,hash_num = self.output_dim).view(-1)
        embedding = self.V[indexes_tensor].view(batch_size, self.output_dim)
        embedding_norm = embedding.sum(dim=1,keepdim=True)
        embedding = embedding / embedding_norm.detach()
        return embedding * self.output_dim

    def add_noise(self):
        with torch.no_grad():
            self.V.data = self.V.data + torch.randn_like(self.V.data) / 5

    def normalize(self):
        with torch.no_grad():
            self.V.data.clamp_(self.cons_lower,self.cons_upper)
           

