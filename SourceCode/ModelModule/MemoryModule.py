from torch import nn
import torch


class MemoryModule(nn.Module):
    def __init__(self, slot_dim, dep_dim):
        super().__init__()
        self.slot_dim = slot_dim
        self.dep_dim = dep_dim
        self.memory_matrix = nn.Parameter(torch.zeros(2, self.dep_dim, self.slot_dim), requires_grad=False)

    def clear(self, batch_size=1):
        with torch.autograd.no_grad():
            self.memory_matrix.data = torch.zeros((batch_size, self.dep_dim, self.slot_dim),
                                                  device=self.memory_matrix.device, requires_grad=False)

    def write(self, batch_address, batch_embedding, batch_frequency, item_size_list):
        assert sum(item_size_list) == batch_address.shape[1]
        batch_f_embedding = batch_embedding * batch_frequency.view(-1, 1)
        memory_matrix_list = []
        start_pos = 0
        for item_size in item_size_list:
            f_embedding_t = batch_f_embedding[start_pos:start_pos + item_size, :].transpose(0, 1)
            address_t = batch_address[:, start_pos:start_pos + item_size, :].transpose(1, 2)
            f_embedding_t = f_embedding_t.unsqueeze(2)
            write_matrix = address_t.bmm(f_embedding_t, ).squeeze(2)
            start_pos += item_size
            memory_matrix_list.append(write_matrix)
        memory_matrix_data = torch.stack(memory_matrix_list, dim=0)
        assert self.memory_matrix.data.shape == memory_matrix_data.shape
        self.memory_matrix.data += memory_matrix_data

    def single_write(self, address, embedding, frequency):
        f_embedding = embedding * frequency.view(-1, 1)
        start_pos = 0
        f_embedding_t = f_embedding.transpose(0, 1)
        address_t = address.transpose(1, 2)
        f_embedding_t = f_embedding_t.unsqueeze(2)
        write_matrix = address_t.bmm(f_embedding_t, ).squeeze(2).unsqueeze(0)
        assert self.memory_matrix.data.shape == write_matrix.shape
        self.memory_matrix.data += write_matrix

    def single_read(self, address, embedding):
        memory_matrix = self.memory_matrix.squeeze(0)
        if address.is_sparse:
            address = address.coalesce()
            indices = address.indices()
            hash_id = indices[0, :]
            slot_id = indices[2, :]
            info1 = memory_matrix[hash_id, slot_id]
            basic_read_info = info1.contiguous().view(embedding.shape[1], embedding.shape[0]).t()
        else:
            basic_read_info = torch.bmm(address, memory_matrix.unsqueeze(2))
            basic_read_info = basic_read_info.squeeze(2).t()
        cm_info = basic_read_info / embedding
        cm_readhead, _ = cm_info.min(dim=-1, keepdim=True)
        return cm_readhead, cm_info, basic_read_info

    def read(self, batch_address, batch_embedding, item_size_list):
        assert sum(item_size_list) == batch_address.shape[1] and sum(item_size_list) == batch_embedding.shape[0]
        start_pos = 0
        cm_readhead_list = []
        cm_info_list = []
        basic_read_list = []
        # embedding_list = []
        for i, item_size in enumerate(item_size_list):
            address = batch_address[:, start_pos:start_pos + item_size, :]
            embedding = batch_embedding[start_pos:start_pos + item_size, :]
            start_pos += item_size
            basic_read_info = self.basic_read(address, i)
            cm_readhead, cm_info = self.cm_read(address, embedding, basic_read_info)
            cm_readhead_list.append(cm_readhead)
            basic_read_list.append(basic_read_info)
            # embedding_list.append(embedding)
            cm_info_list.append(cm_info)
        return torch.cat(cm_readhead_list, dim=0), torch.cat(cm_info_list, dim=0), torch.cat(basic_read_list, dim=0)

    def cm_read(self, address, embedding, basic_read_info):
        cm_info = (basic_read_info) / (embedding)
        cm_readhead, _ = cm_info.min(dim=-1, keepdim=True)
        return cm_readhead, cm_info

    def basic_read(self, address, index):
        basic_read_info = address.bmm(self.memory_matrix[index, :, :].unsqueeze(2))
        basic_read_info.squeeze_(2).t_()
        return basic_read_info
