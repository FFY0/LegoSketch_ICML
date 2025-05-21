
import torch
import torch.nn as nn
from SourceCode.ModelModule.SubModuleUtil import  RandomHash



class RandomAddressModule(nn.Module):
    def __init__(self, dep_dim, slot_dim):
        # set True to produce sparse address tensor
        self.prod_sparse = False
        super().__init__()
        self.random_hash = RandomHash(hash_seed=1,hash_range=slot_dim)
        self.dep_dim = dep_dim
        self.slot_num = slot_dim

    def get_prod_sparse(self):
        if self.prod_sparse:
            print('producing sparse address tensor')
        else:
            print('producing dense address tensor')
        return self.prod_sparse

    def set_prod_sparse(self, prod_sparse):
        self.prod_sparse = prod_sparse
        print('set prod_sparse to', self.prod_sparse)


    # address shape: batch_size, self.dep_dim, self.slot_num
    # generate address tensor by generating random value according to seed
    def forward(self, input_tensor, batch_size=None):
        if batch_size is None:
            batch_size = input_tensor.shape[0]
        with torch.no_grad():
            address_indexes_tensor = self.random_hash(input_tensor, batch_size, self.dep_dim)
            address_indexes_tensor = address_indexes_tensor.t().reshape(-1)
            index_1 = torch.arange(batch_size, requires_grad=False,device=input_tensor.device).view(-1,1).repeat(1,self.dep_dim).view(-1)
            index_2 = torch.arange(self.dep_dim, requires_grad=False,device=input_tensor.device).repeat(batch_size)

            coordinate_index = torch.stack([index_2,index_1, address_indexes_tensor])
            sparse_address_tensor = torch.sparse_coo_tensor(coordinate_index,
                                                            values=torch.ones(size = (batch_size* self.dep_dim,))
                                                            , size=( self.dep_dim, batch_size,self.slot_num),
                                                            requires_grad=False, device=input_tensor.device)
            if self.prod_sparse:

                address_tensor = sparse_address_tensor
            else:
                address_tensor = sparse_address_tensor.to_dense()
            return address_tensor
