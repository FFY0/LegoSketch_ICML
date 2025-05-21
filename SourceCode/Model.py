import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, address_module, embedding_module, decode_module,memory_module,loss_func):
        super(Model, self).__init__()
        self.embedding_module = embedding_module
        self.memory_module = memory_module
        self.decode_module = decode_module
        self.address_module = address_module
        self.loss_func = loss_func
    
    def query(self, x, statis_info):
        print('not implemented')
        pass

    def adjust_net_arch(self,x, statis_info):
        embedding = self.get_embedding(x[0:6])
        address = self.get_address(x[0:6])
        statis_info = statis_info[0:6]
        item_size_list = [3,3]
        cm_readhead,cm_info,batch_basic_read_info = self.memory_module.read(address, embedding,item_size_list)
        set_info = torch.stack((cm_info,embedding),dim=-1)
        dec_pred = self.decode_module.adjust_net_arch(set_info,statis_info, cm_readhead,self.memory_module.memory_matrix.data,item_size_list)


    def batch_write(self, batch_input_x, batch_input_y,item_size_list):
        batch_embedding = self.get_embedding(batch_input_x)
        batch_address = self.get_address(batch_input_x)
        self.memory_module.write(batch_address, batch_embedding, batch_input_y,item_size_list)

    def write(self,input_x, input_y):
        embedding = self.get_embedding(input_x)
        address = self.get_address(input_x)
        self.memory_module.single_write(address, embedding, input_y)

    def dec_query(self, query_x, weight_sum):
        item_size_list = [query_x.shape[0]]
        embedding = self.get_embedding(query_x)
        address = self.get_address(query_x)
        cm_readhead, cm_info, basic_read_info = self.memory_module.single_read(address, embedding)
        set_info = torch.stack((basic_read_info, embedding), dim=-1)
        dec_pred, stream_info1, stream_info2 = self.decode_module(set_info, weight_sum, cm_readhead, self.memory_module.memory_matrix.data, item_size_list)
        return dec_pred, cm_readhead, stream_info1, stream_info2

    def clear(self,batch_size=1):
        self.memory_module.clear(batch_size)

    def normalize_basis_matrix(self):
        self.embedding_module.normalize()

    def get_embedding(self, x):
        embedding = self.embedding_module(x)
        return embedding

    def get_address(self, x):
        address = self.address_module(x)
        return address

    def batch_dec_query(self,batch_query_x,batch_weight_sum_tensor,item_size_list):
        assert sum(item_size_list) == batch_query_x.shape[0]
        batch_embedding = self.get_embedding(batch_query_x)
        batch_address = self.get_address(batch_query_x)
        batch_cm_readhead, batch_cm_info,batch_basic_read_info = self.memory_module.read(batch_address, batch_embedding, item_size_list)
        set_info = torch.stack((batch_basic_read_info, batch_embedding), dim=-1)
        batch_dec_pred,stream_info1,stream_info2 = self.decode_module(set_info, batch_weight_sum_tensor, batch_cm_readhead, self.memory_module.memory_matrix.data, item_size_list)
        return batch_dec_pred,batch_cm_readhead,stream_info1,stream_info2
