import torch
import torch.nn as nn

class MemoryDeepsets(nn.Module):
    def __init__(self,memory_depth) -> None:
        super().__init__()
        self.dim_output = 4
        self.memory_depth = memory_depth
        self.filter_ratio = 0.1
        self.enc = nn.Sequential(
                nn.Linear(1, 8),
                nn.ReLU(),
                nn.Linear(8, 16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU())
        self.main_dec = nn.Sequential(
                nn.Linear(32, 16),
                nn.LeakyReLU(),
                nn.Linear(16, 8),
                nn.LeakyReLU(),
                nn.Linear(8, self.dim_output),
                nn.LeakyReLU(),
        )
        self.dec1 = nn.Sequential(
                nn.Linear(32, 16),
                nn.LeakyReLU(),
                nn.Linear(16, 16),
                nn.LeakyReLU(),
                nn.Linear(16, 8),
                nn.LeakyReLU(),
                nn.Linear(8, 1),
        )
        self.dec2 = nn.Sequential(
                nn.Linear(32, 16),
                nn.LeakyReLU(),
                nn.Linear(16, 16),
                nn.LeakyReLU(),
                nn.Linear(16, 8),
                nn.LeakyReLU(),
                nn.Linear(8, 1),
        )

    def forward(self, x):
        filter_size = int(x.shape[-1] * self.filter_ratio)
        filtered_x = x[:,:,0:filter_size].detach().unsqueeze(-1)
        filtered_x = self.enc(filtered_x).mean(-2)
        main_info = self.main_dec(filtered_x).mean(-2)
        stream_info1 = self.dec1(filtered_x).mean(-2)
        stream_info2 = self.dec2(filtered_x).mean(-2)
        return main_info,stream_info1,stream_info2

class SetDecodeModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = None
        self.memorydeepsets = None

    # Automatically adjusts to the dimension of the input element x
    def adjust_net_arch(self, set_info, sum_info, cm_read_head,memory,item_size_list):
        self.memorydeepsets = MemoryDeepsets(memory_depth = memory.shape[1])
        self.enc = nn.Sequential(
                nn.Linear(set_info.shape[-1], 8),
                nn.ReLU(),
                nn.Linear(8, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(),
        )
        self.dec = nn.Sequential(
                nn.Linear(16 + 2+2 + self.memorydeepsets.dim_output, 32),
                # nn.LayerNorm(32),
                nn.LeakyReLU(),
                nn.Linear(32, 16),
                # nn.LayerNorm(8),
                nn.LeakyReLU(),
        )

        self.out1 = nn.Sequential(
            nn.Linear(16+4, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 1)
        )

        return self.forward(set_info, sum_info, cm_read_head,memory,item_size_list)

    def forward(self, set_info, sum_info, cm_read_head,memory,item_size_list):
        memory_info,stream_info1,stream_info2 = self.memorydeepsets(memory)
        repeat_counts = torch.tensor(item_size_list,device = memory.device)
        repeated_memory_info = memory_info.repeat_interleave(repeat_counts, dim=0)
        repeated_stream_info1 = stream_info1.repeat_interleave(repeat_counts, dim=0)
        repeated_stream_info2 = stream_info2.repeat_interleave(repeat_counts, dim=0)
        y = self.enc(set_info).mean(-2)
        y = torch.cat((y,repeated_memory_info,cm_read_head,sum_info,repeated_stream_info1.detach(),repeated_stream_info2.detach()),dim=-1)
        y = self.dec(y)
        y = torch.cat((y,cm_read_head,sum_info,repeated_stream_info1.detach(),repeated_stream_info2.detach()),dim=-1)
        y = self.out1(y)
        return y,stream_info1,stream_info2
