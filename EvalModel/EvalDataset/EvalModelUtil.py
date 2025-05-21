import importlib.util
import os.path
import pickle
from pyexpat import model
import random
import sys
import time
from collections import Counter
import timeit
from tkinter import NO

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import normal
import tqdm
import torch.nn as nn
import xxhash

from EvalModel.EvalDataset.DatasetLoader import dataset_dic,project_path
from EvalModel.EvalDataset.ElasticSketch import ElasticSketch
from EvalModel.EvalDataset.traditional_sketch_src import count_min, count_sketch,learned_count_sketch_without_pred

import scipy.interpolate as interpolate
from random import sample
import cupy as cp
from numpy import int32
import torch
from cupy.cuda import function
from cupy.cuda import device
from pynvrtc.compiler import Program
from collections import namedtuple
import torch.nn as nn


# CUDA Stream
Stream = namedtuple('Stream', ['ptr'])

class cupyKernel:
    def __init__(self, kernel, func_name):
        self.kernel = kernel
        self.title = func_name + ".cu"
        self.func_name = func_name
        self.compiled = False

    def get_compute_arch(self):
        return "compute_{0}".format(device.Device().compute_capability)

    def compile(self):
        # Create program
        program = Program(self.kernel, self.title)

        # Compile program
        arch = "-arch={0}".format(self.get_compute_arch())
        ptx = program.compile([arch])

        # Load Program
        m = function.Module()
        m.load(bytes(ptx.encode()))

        # Get Function Pointer
        self.func = m.get_function(self.func_name)
        self.compiled = True

    def __call__(self, grid, block, args, strm, smem=0):
        if not self.compiled:
            self.compile()

        # Run Function
        self.func(grid,
                  block,
                  args,
                  smem,
                  stream=Stream(ptr=strm))



kernel = '''
extern "C"

__global__
void hash(const long long  *input, long long  *output, const long long  *a_list, const long long  *b_list, int range, int N,int num_pairs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        for (int pair_idx = 0; pair_idx < num_pairs; ++pair_idx){
            long long  a = a_list[pair_idx];
            long long  b = b_list[pair_idx];
            long long  h = input[idx];
            h = h * a + b;
          	h ^= h >> 16;
	        h *= 0x85ebca6b;
	        h ^= h >> 13;
	        h *= 0xc2b2ae35;
	        h ^= h >> 16;
            h = h % range;
            output[idx * num_pairs + pair_idx] = h;
        }
    }
}
'''

"""
            

"""

class RandomHash(nn.Module):
    def __init__(self, hash_seed,hash_range ):
        super().__init__()
        self.hash_seed = hash_seed
        self.hash_range = hash_range

    # hash each sample by hash_num times
    def forward(self, input_tensor, sample_size,hash_num=1):
        with torch.no_grad():
            seed = xxhash.xxh32((str(input_tensor.sum().cpu().item()) + str(input_tensor.shape)),
                                seed=self.hash_seed).intdigest() % 1000
            torch.cuda.manual_seed(seed)
            generator = torch.Generator(input_tensor.device)
            generator.manual_seed(seed)
            hashed_indexes_tensor = torch.randint(low=0, high=self.hash_range, size=(sample_size,hash_num),
                                                  requires_grad=False, generator=generator,device=input_tensor.device)
            return hashed_indexes_tensor

def load_unit_model(model_path, device, use_cuda_hash = False):
    model = torch.load(model_path)
    if use_cuda_hash:
        model = replace_model_hash(model)
    model.to(device)
    return model, calculate_model_memory_size(model)

def replace_model_hash(model):
    # print("using cuda hash")
    model.embedding_module.random_hash = CUDAHash(hash_range=model.embedding_module.random_hash.hash_range,a_list=[57251,84857,100907,80557,74959],b_list=[33757,24851,36931,50767,70793])
    model.address_module.random_hash = CUDAHash(hash_range=model.address_module.random_hash.hash_range,a_list=[69389,58031,39841,28643,19777],b_list=[48193,56383,63473,69473,81761])
    return model

class CUDAHash(nn.Module):
    def __init__(self,hash_range,a_list,b_list):
        super().__init__()
        self.hash_range = hash_range
        self.cuda_kernel = cupyKernel(kernel, "hash")
        self.a_list = torch.tensor(a_list,requires_grad=False,dtype=torch.int64)
        self.b_list = torch.tensor(b_list,requires_grad=False,dtype=torch.int64)

    # hash each sample by hash_num times
    def forward(self, input_tensor, sample_size,hash_num=1):
        assert input_tensor.device.type == 'cuda', "CUDAHash only support cuda device"
        assert hash_num <= self.a_list.shape[0] and sample_size == input_tensor.shape[0]

        assert len(input_tensor.shape) == 1 or input_tensor.shape[1] == 1
        self.a_list = self.a_list.to(input_tensor.device)
        self.b_list = self.b_list.to(input_tensor.device)
        blk_size = 512
        grid_size = input_tensor.shape[0] // blk_size
        if input_tensor.shape[0] % blk_size != 0:
            grid_size += 1
        hashed_indexes_tensor = torch.zeros((input_tensor.shape[0],hash_num),device=input_tensor.device,dtype=torch.int64)-1
        try:
            assert input_tensor.dtype == torch.int64
        except:
            print('error')
        self.cuda_kernel(grid=(grid_size,1,1), block=(blk_size,1,1), args=(input_tensor.data_ptr(), hashed_indexes_tensor.data_ptr(), self.a_list.data_ptr(),self.b_list.data_ptr(),self.hash_range,input_tensor.shape[0],hash_num), strm=torch.cuda.current_stream().cuda_stream)
        assert torch.sum(hashed_indexes_tensor == -1) == 0,"error"
        return hashed_indexes_tensor

def calculate_model_memory_size(model):
    memory_matrix = model.memory_module.memory_matrix.data
    shape = memory_matrix.shape
    space = 1
    for i in shape:
        space *= i
    # KB
    space = (space * 4) / 1024
    return round(space, 4)


class Cluster:
    def __init__(self, unit_model, cluster_size,use_cuda_hash) -> None:
        unit_model.eval()
        unit_model.clear()
        self.model = unit_model
        memory_matrix = unit_model.memory_module.memory_matrix
        self.memory_matrix_list = []
        self.memory_statis_info_list = []
        self.cluster_size = cluster_size
        self.item_size_threshold = 10000
        self.use_cuda_hash = use_cuda_hash
        if use_cuda_hash:
            self.random_hash = CUDAHash(hash_range=cluster_size,a_list=[80489,66553,],b_list=[61297,24851,])
        else:
            self.random_hash = RandomHash(hash_seed = 0,hash_range=cluster_size)

        for i in range(cluster_size):
            self.memory_matrix_list.append(memory_matrix.data.clone())
            self.memory_statis_info_list.append(torch.zeros(1, device=memory_matrix.data.device))

    def clear(self):
        for i in range(self.cluster_size):
            self.memory_matrix_list[i] = torch.zeros_like(self.memory_matrix_list[i])

    def write(self, support_x, support_y):
        batch_size = support_x.shape[0]
        indexes_tensor = self.random_hash(support_x, sample_size=batch_size).view(-1)
        for i in range(self.cluster_size):
            i_cluster_index = torch.where(indexes_tensor == i)[0]
            if i_cluster_index.shape[0] == 0:
                continue
            i_support_x = support_x[i_cluster_index]
            i_support_y = support_y[i_cluster_index]
            self.model.memory_module.memory_matrix.data = self.memory_matrix_list[i]
            self.memory_statis_info_list[i] = self.memory_statis_info_list[i] + i_support_y.sum()
            self.model.write(i_support_x, i_support_y)
            self.memory_matrix_list[i] = self.model.memory_module.memory_matrix.data

    def dec_query(self, query_x):
        batch_size = query_x.shape[0]
        indexes_tensor = self.random_hash(query_x, sample_size=batch_size).view(-1)
        dec_pred = torch.ones((batch_size, 1), device=query_x.device) * -1
        cm_readhead_read = torch.ones((batch_size, 1), device=query_x.device) * -1
        ensemble_pred = torch.ones((batch_size, 1), device=query_x.device) * -1
        flag_tensor = torch.zeros((batch_size, 1), device=query_x.device)
        for i in range(self.cluster_size):
            i_cluster_index = torch.where(indexes_tensor == i)[0]
            i_query_x = query_x[i_cluster_index]
            self.model.memory_module.memory_matrix.data = self.memory_matrix_list[i]
            i_statis_info = self.memory_statis_info_list[i]
            i_reshape_statis_info = i_statis_info.unsqueeze(0).repeat(i_query_x.shape[0], 1)
            i_dec_pred, i_cm_readhead,i_stream_info1,i_stream_info2 = self.model.dec_query(i_query_x, i_reshape_statis_info)
            i_dec_pred = torch.where(i_dec_pred<i_cm_readhead,i_dec_pred,i_cm_readhead)
            dec_pred[i_cluster_index] = i_dec_pred
            cm_readhead_read[i_cluster_index] = i_cm_readhead
            item_size_pred = i_stream_info2.mean().item()
            alpha_pred = i_stream_info1.mean().item()

            if item_size_pred > self.item_size_threshold and alpha_pred < 1.0:
                ensemble_pred[i_cluster_index] = i_dec_pred
            else:
                ensemble_pred[i_cluster_index] = i_cm_readhead
            flag_tensor[i_cluster_index] += 1
        assert all(flag_tensor == 1), "error"
        return ensemble_pred, cm_readhead_read,


def eval_model_with_budget(cluster_budget,dataset_name,device,model_path,use_cuda_hash,eval_elastic,write_batch_size=10000000,seed_index=0):
    # exec traditional baseline only in seed_index_0
    assert dataset_name in dataset_dic.keys(), "can't find dataset, name:"+ dataset_name
    data_loader_generator = dataset_dic[dataset_name]
    with torch.no_grad():
        model,model_budget = load_unit_model(model_path,device,use_cuda_hash)
        if model_budget != 100:
            print('unit neural sketch budget: ',model_budget)
        model.address_module.set_prod_sparse(True)
        # KB
        cluster = Cluster(model,int(cluster_budget//model_budget),use_cuda_hash)
        count = 0
        query_x = []
        query_y = []
        for i_support_x,i_support_y,i_query_x,i_query_y in data_loader_generator(device,write_batch_size):
            count += 1
            cluster.write(i_support_x,i_support_y)
            query_x.append(i_query_x)
            query_y.append(i_query_y)
        assert count == 1,"have not been tested for datasets that need mutiple-write"
        query_x = torch.cat(query_x,dim=0)
        query_y = torch.cat(query_y,dim=0)
        dec_pred,cm_readhead_pred=cluster.dec_query(query_x)

        if eval_elastic and seed_index == 0:
            count_min_sketch_pred = count_min(query_y.view(-1).cpu().numpy(),(cluster_budget*1024)//(3*4),3)
            count_sketch_pred = count_sketch(query_y.view(-1).cpu().numpy(),(cluster_budget*1024)//(3*4),3)
            learned_count_sketch_pred = learned_count_sketch_without_pred(query_y.view(-1).cpu().numpy(),(cluster_budget*1024)//(4))
            count_min_sketch_ARE = ((torch.tensor(count_min_sketch_pred,device = query_y.device).view(-1,1).float()-query_y).abs()/query_y.abs()).mean().cpu().item()
            count_sketch_ARE = ((torch.tensor(count_sketch_pred,device = query_y.device).view(-1,1).float()-query_y).abs()/query_y.abs()).mean().cpu().item()
            learned_count_sketch_ARE = ((torch.tensor(learned_count_sketch_pred,device = query_y.device).view(-1,1).float()-query_y).abs()/query_y.abs()).mean().cpu().item()

            count_min_sketch_AAE = ((torch.tensor(count_min_sketch_pred,device = query_y.device).view(-1,1).float()-query_y).abs()).mean().cpu().item()
            count_sketch_AAE = ((torch.tensor(count_sketch_pred,device = query_y.device).view(-1,1).float()-query_y).abs()).mean().cpu().item()
            learned_count_sketch_AAE = ((torch.tensor(learned_count_sketch_pred,device = query_y.device).view(-1,1).float()-query_y).abs()).mean().cpu().item()
        else:
            count_min_sketch_ARE = None
            count_sketch_ARE = None
            learned_count_sketch_ARE = None
            count_min_sketch_AAE = None
            count_sketch_AAE = None
            learned_count_sketch_AAE = None

        if eval_elastic:
            es = ElasticSketch(mem_kb=cluster_budget,unit_budget = model_budget)
            cluster_in_es = Cluster(model,int(es.light_part_mem//model_budget),use_cuda_hash)
            assert es.light_part_mem == cluster_in_es.cluster_size * model_budget,"error"
            key_tensor = torch.arange(query_y.shape[0],device = query_y.device)
            value_tensor = query_y.view(-1)
            key_list = key_tensor.tolist()
            value_list = value_tensor.tolist()
            repeat_key_tensor = torch.repeat_interleave(key_tensor, value_tensor, dim=0)
            shuffle_repeat_key_tensor = repeat_key_tensor[torch.randperm(repeat_key_tensor.shape[0])]
            repeat_key_list = shuffle_repeat_key_tensor.tolist()

            key_in_light_part = []
            for k in repeat_key_list:
                res = es.insert(k, 1)
                flag = res[0]
                if flag == 3:
                    key_in_light_part.append(k)
                elif flag == 4:
                    evict_key = res[1]
                    evict_value = res[2]
                    for i in range(evict_value):
                        key_in_light_part.append(evict_key)
            counter = Counter(key_in_light_part)
            light_keys = np.array(list(counter.keys()))
            light_values = np.array(list(counter.values()))
            light_keys_x = torch.tensor(light_keys,device = query_y.device).view(-1,1)
            light_values_y = torch.tensor(light_values,device = query_y.device).view(-1,1)
            cluster_in_es.write(light_keys_x,light_values_y)
            dec_light_pred, cm_readhead_light_pred = cluster_in_es.dec_query(light_keys_x)
            light_keys_x = light_keys_x.type(torch.int32).view(-1).cpu().tolist()
            dec_light_pred = dec_light_pred.view(-1).cpu().tolist()
            cm_readhead_light_pred = cm_readhead_light_pred.view(-1).cpu().tolist()
            dec_light_dic = {key: value for key, value in zip(light_keys_x, dec_light_pred)}
            cm_readhead_light_dic = {key: value for key, value in zip(light_keys_x, cm_readhead_light_pred)}


            es_query_list = []
            cluster_in_es_dec_pred_list = []
            cluster_in_es_cm_pred_list = []
            flag_count = 0
            for k, v in zip(key_list, value_list):
                count,heavy_count,Flag = es.query(k)
                assert count is not None
                es_query_list.append(count)
                if Flag is False:
                    cluster_in_es_dec_pred_list.append(count)
                    cluster_in_es_cm_pred_list.append(count)
                else:
                    try:
                        dec_light = dec_light_dic[k]
                        cm_readhead_light = cm_readhead_light_dic[k]
                    except:
                        dec_light = random.choice(dec_light_pred)
                        cm_readhead_light = random.choice(cm_readhead_light_pred)
                    if heavy_count is None:
                        cluster_in_es_dec_pred_list.append(dec_light)
                        cluster_in_es_cm_pred_list.append(cm_readhead_light)
                    else:
                        cluster_in_es_dec_pred_list.append(dec_light + heavy_count)
                        cluster_in_es_cm_pred_list.append(cm_readhead_light + heavy_count)

            es_pred = np.array(es_query_list)
            cluster_in_es_dec_pred = np.array(cluster_in_es_dec_pred_list)
            cluster_in_es_cm_pred = np.array(cluster_in_es_cm_pred_list)
            es_ARE = ((torch.tensor(es_pred,device = query_y.device).view(-1,1).float()-query_y).abs()/query_y.abs()).mean().cpu().item()
            cluster_in_es_dec_ARE = ((torch.tensor(cluster_in_es_dec_pred,device = query_y.device).view(-1,1).float()-query_y).abs()/query_y.abs()).mean().cpu().item()
            cluster_in_es_cm_ARE = ((torch.tensor(cluster_in_es_cm_pred,device = query_y.device).view(-1,1).float()-query_y).abs()/query_y.abs()).mean().cpu().item()
            es_AAE = ((torch.tensor(es_pred,device = query_y.device).view(-1,1).float()-query_y).abs()).mean().cpu().item()
            cluster_in_es_dec_AAE = ((torch.tensor(cluster_in_es_dec_pred,device = query_y.device).view(-1,1).float()-query_y).abs()).mean().cpu().item()
            cluster_in_es_cm_AAE = ((torch.tensor(cluster_in_es_cm_pred,device = query_y.device).view(-1,1).float()-query_y).abs()).mean().cpu().item()
            elastic_mem_size = es.light_part_mem + es.heavy_part_mem
            elastic_with_cluster_mem_size = es.heavy_part_mem + cluster_in_es.cluster_size * model_budget
            assert int(elastic_mem_size) == int(elastic_with_cluster_mem_size),"Error! mem size error elastic_mem_size "+str(elastic_mem_size)+" elastic_with_cluster_mem_size "+str(elastic_with_cluster_mem_size)
            assert int(elastic_with_cluster_mem_size) == int(cluster_budget),"Error! mem size error elastic_with_cluster_mem_size "+str(elastic_with_cluster_mem_size)+" cluster_budget "+str(cluster_budget)


        else:
            es_ARE = None
            cluster_in_es_dec_ARE = None
            cluster_in_es_cm_ARE =None
            es_AAE =None
            cluster_in_es_dec_AAE = None
            cluster_in_es_cm_AAE = None
        #calculate error
        dec_ARE = ((dec_pred-query_y).abs()/query_y.abs()).mean().cpu().item()
        cm_readhead_ARE = ((cm_readhead_pred-query_y).abs()/query_y.abs()).mean().cpu().item()
        dec_AAE = ((dec_pred-query_y).abs()).mean().cpu().item()
        cm_readhead_AAE = ((cm_readhead_pred-query_y).abs()).mean().cpu().item()
        return dec_ARE,count_min_sketch_ARE,count_sketch_ARE,cm_readhead_ARE,es_ARE,cluster_in_es_dec_ARE,cluster_in_es_cm_ARE,learned_count_sketch_ARE,\
            dec_AAE,count_min_sketch_AAE,count_sketch_AAE,cm_readhead_AAE,es_AAE,cluster_in_es_dec_AAE,cluster_in_es_cm_AAE,learned_count_sketch_AAE

def eval_model_with_budgets(cluster_budget_list,dataset_name,device,model_path,use_cuda_hash,eval_elastic,write_batch_size=10000000):
    dec_ARE_list = []
    count_min_ARE_list = []
    count_sketch_ARE_list = []
    cm_read_ARE_list = []
    es_ARE_list = []
    cluster_in_es_dec_ARE_list = []
    cluster_in_es_cm_ARE_list = []
    learned_count_sketch_ARE_list = []

    dec_AAE_list = []
    count_min_AAE_list = []
    count_sketch_AAE_list = []
    cm_read_AAE_list = []
    es_AAE_list = []
    cluster_in_es_dec_AAE_list = []
    cluster_in_es_cm_AAE_list = []
    learned_count_sketch_AAE_list = []
    for budget in tqdm.tqdm(cluster_budget_list):
        dec_ARE,count_min_ARE,count_sketch_ARE,cm_read_ARE,es_ARE,cluster_in_es_dec_ARE,cluster_in_es_cm_ARE,learned_count_sketch_ARE,\
        dec_AAE,count_min_sketch_AAE,count_sketch_AAE,cm_readhead_AAE,es_AAE,cluster_in_es_dec_AAE,cluster_in_es_cm_AAE,learned_count_sketch_AAE\
            = eval_model_with_budget(budget, dataset_name, device, model_path,use_cuda_hash,eval_elastic,write_batch_size)
        dec_ARE_list.append(dec_ARE)
        count_min_ARE_list.append(count_min_ARE)
        count_sketch_ARE_list.append(count_sketch_ARE)
        cm_read_ARE_list.append(cm_read_ARE)
        es_ARE_list.append(es_ARE)
        cluster_in_es_dec_ARE_list.append(cluster_in_es_dec_ARE)
        cluster_in_es_cm_ARE_list.append(cluster_in_es_cm_ARE)
        learned_count_sketch_ARE_list.append(learned_count_sketch_ARE)

        dec_AAE_list.append(dec_AAE)
        count_min_AAE_list.append(count_min_sketch_AAE)
        count_sketch_AAE_list.append(count_sketch_AAE)
        cm_read_AAE_list.append(cm_readhead_AAE)
        es_AAE_list.append(es_AAE)
        cluster_in_es_dec_AAE_list.append(cluster_in_es_dec_AAE)
        cluster_in_es_cm_AAE_list.append(cluster_in_es_cm_AAE)
        learned_count_sketch_AAE_list.append(learned_count_sketch_AAE)

    return dec_ARE_list,count_min_ARE_list,count_sketch_ARE_list,cm_read_ARE_list,es_ARE_list,cluster_in_es_dec_ARE_list,cluster_in_es_cm_ARE_list,learned_count_sketch_ARE_list,\
        dec_AAE_list,count_min_AAE_list,count_sketch_AAE_list,cm_read_AAE_list,es_AAE_list,cluster_in_es_dec_AAE_list,cluster_in_es_cm_AAE_list,learned_count_sketch_AAE_list



def get_pos(zipf_list,dataset_name):
    for i,zipf in enumerate(zipf_list):
        num1 = int(dataset_name[10])
        num2 = int(dataset_name[-1])
        num3 = int(zipf[0])
        num4 = int(zipf[-1])
        if num1 == num3 and num2 == num4:
            return i
    return None



def get_error(model_performance_list):
    model_error_list = []
    for model_performance in model_performance_list:
        model_error_list.append(model_performance[0])
    return model_error_list

def get_std(model_performance_list):
    model_std_list = []
    for model_performance in model_performance_list:
        model_std_list.append(model_performance[1])
    return model_std_list



def eval_dataset_with_seed(model_path_list,model_name_list,device,dataset_name,cluster_budget_list,use_cuda_hash = False,
                           eval_elastic = True,ablation=False,drectlydraw=False):

    assert eval_elastic^ablation,"eval_elastic and ablation can't be true/False at the same time"
    if ablation:
        eval_elastic = False
    else:
        assert len(model_path_list) == 1,"Not ablation only support one model"
    if use_cuda_hash:
        print("using cuda hash")
    print("evaling dataset: " + dataset_name)
    if not drectlydraw:
        if eval_elastic:
            print("evaling elastic sketch at the same time")
        model_dec_ARE_list, model_count_min_ARE_list, model_count_sketch_ARE_list, model_cm_read_ARE_list, model_es_ARE_list, model_cluster_in_es_dec_ARE_list, model_cluster_in_es_cm_ARE_list, model_learned_count_sketch_ARE_list, \
            model_dec_AAE_list, model_count_min_AAE_list, model_count_sketch_AAE_list, model_cm_read_AAE_list, model_es_AAE_list, model_cluster_in_es_dec_AAE_list, model_cluster_in_es_cm_AAE_list, model_learned_count_sketch_AAE_list \
            = eval_models_with_budgets_with_seed(cluster_budget_list, dataset_name, device, model_path_list,use_cuda_hash,eval_elastic)
        data_dic = {
            'dataset_name':dataset_name,
            'ablation':ablation,
            'model_name_list':model_name_list,
            'cluster_budget_list':cluster_budget_list,
            'MS_budget_list':None,
            'eval_elastic':eval_elastic,

            'model_dec_ARE_list':model_dec_ARE_list,
            'model_count_min_ARE_list':model_count_min_ARE_list,
            'model_count_sketch_ARE_list':model_count_sketch_ARE_list,
            'model_cm_read_ARE_list':model_cm_read_ARE_list,
            'model_es_ARE_list':model_es_ARE_list,
            'model_cluster_in_es_dec_ARE_list':model_cluster_in_es_dec_ARE_list,
            'model_cluster_in_es_cm_ARE_list':model_cluster_in_es_cm_ARE_list,
            'model_learned_count_sketch_ARE_list':model_learned_count_sketch_ARE_list,

            'model_dec_AAE_list':model_dec_AAE_list,
            'model_count_min_AAE_list':model_count_min_AAE_list,
            'model_count_sketch_AAE_list':model_count_sketch_AAE_list,
            'model_cm_read_AAE_list':model_cm_read_AAE_list,
            'model_es_AAE_list':model_es_AAE_list,
            'model_cluster_in_es_dec_AAE_list':model_cluster_in_es_dec_AAE_list,
            'model_cluster_in_es_cm_AAE_list':model_cluster_in_es_cm_AAE_list,
            'model_learned_count_sketch_AAE_list':model_learned_count_sketch_AAE_list,

            'MS_ARE_list':None,
            'MS_AAE_list':None,
        }
        with open(os.path.join(project_path,'EvalModel/EvalDataset/Figure/'+dataset_name+'_data_ablation_'+str(ablation)), 'wb') as file:
            pickle.dump(data_dic, file)

    with open(os.path.join(project_path,'EvalModel/EvalDataset/Figure/'+dataset_name+'_data_ablation_'+str(ablation)), 'rb') as file:
        read_data_dic = pickle.load(file)
    draw_graph(read_data_dic['dataset_name'],read_data_dic['ablation'],read_data_dic['model_name_list'],read_data_dic['cluster_budget_list'],read_data_dic['MS_budget_list'],read_data_dic['eval_elastic'],
               read_data_dic['model_dec_ARE_list'],read_data_dic['model_cluster_in_es_dec_ARE_list'],read_data_dic['model_es_ARE_list'],read_data_dic['model_count_min_ARE_list'],read_data_dic['model_count_sketch_ARE_list'],read_data_dic['model_learned_count_sketch_ARE_list'],read_data_dic['MS_ARE_list'],
               read_data_dic['model_dec_AAE_list'],read_data_dic['model_cluster_in_es_dec_AAE_list'],read_data_dic['model_es_AAE_list'],read_data_dic['model_count_min_AAE_list'],read_data_dic['model_count_sketch_AAE_list'],read_data_dic['model_learned_count_sketch_AAE_list'],read_data_dic['MS_AAE_list'])
    # draw_graph(read_data_dic['dataset_name'],read_data_dic['ablation'],model_name_list,read_data_dic['cluster_budget_list'],read_data_dic['MS_budget_list'],read_data_dic['eval_elastic'],
    #            read_data_dic['model_dec_ARE_list'],read_data_dic['model_cluster_in_es_dec_ARE_list'],read_data_dic['model_es_ARE_list'],read_data_dic['model_count_min_ARE_list'],read_data_dic['model_count_sketch_ARE_list'],read_data_dic['model_learned_count_sketch_ARE_list'],read_data_dic['MS_ARE_list'],
    #            read_data_dic['model_dec_AAE_list'],read_data_dic['model_cluster_in_es_dec_AAE_list'],read_data_dic['model_es_AAE_list'],read_data_dic['model_count_min_AAE_list'],read_data_dic['model_count_sketch_AAE_list'],read_data_dic['model_learned_count_sketch_AAE_list'],read_data_dic['MS_AAE_list'])



def draw_graph(dataset_name,ablation,model_name_list,cluster_budget_list,MS_budget_list,eval_elastic,
               model_dec_ARE_list,model_cluster_in_es_dec_ARE_list,model_es_ARE_list,model_count_min_ARE_list,model_count_sketch_ARE_list,model_learned_count_sketch_ARE_list,MS_ARE_list,
               model_dec_AAE_list,model_cluster_in_es_dec_AAE_list,model_es_AAE_list,model_count_min_AAE_list,model_count_sketch_AAE_list,model_learned_count_sketch_AAE_list,MS_AAE_list):
    ms_res_file_path = os.path.join(project_path,'EvalModel/EvalDataset/{}_res.npz'.format(dataset_name))
    ms_res_file = np.load(ms_res_file_path)
    MS_budget_list = ms_res_file['BUDGET']
    MS_ARE_list = ms_res_file['ARE']
    MS_AAE_list = ms_res_file['AAE']
    MS_ARE_std_list = ms_res_file['ARE_STD']
    MS_AAE_std_list = ms_res_file['AAE_STD']
    cluster_budget_list = np.array(cluster_budget_list)/1024
    MS_budget_list = np.array(MS_budget_list)/1024

    assert eval_elastic^ablation,"eval_elastic and ablation can't be true at the same time"
    ARE_y_lim = None
    AAE_y_lim = None
    normal_skew_datasets = ['aol','lkml','synthetic_0_5','synthetic_0_7','synthetic_0_9','wiki']
    high_skew_datasets = ['kosarak','webdocs','synthetic_1_1','synthetic_1_3','synthetic_1_5']
    normal_skew = None
    if ablation:
        ARE_y_lim = 3.0
        AAE_y_lim = 3.0

    elif dataset_name in normal_skew_datasets:
        ARE_y_lim = 2
        AAE_y_lim = 4
        normal_skew = True
    elif dataset_name in high_skew_datasets:
        ARE_y_lim = 4
        AAE_y_lim = 8
        normal_skew = False
    else:
        print("dataset not found!")
        exit()
    color_list = ['red','lightskyblue','orange','brown','fuchsia','gray','orange','mediumseagreen',]
    marker_list = ['o','^','v','<','s','>']
    assert len(model_name_list) <= len(color_list) and len(model_name_list) <= len(marker_list),"too many model to handle"
    plt.figure(figsize=(10, 10))
    if not ablation:
        plt.plot(cluster_budget_list, get_error(model_count_min_ARE_list[0]), color="gray", label="CMS", linestyle=':',linewidth=8,)
        plt.plot(cluster_budget_list, get_error(model_count_sketch_ARE_list[0]), color="gray", label="CS", linestyle='--',linewidth=8,)
        plt.plot(cluster_budget_list, get_error(model_learned_count_sketch_ARE_list[0]), color="brown", label="LCS", linestyle=(0,(3,1,1,1)),linewidth=8,alpha=0.7)
        plt.errorbar(MS_budget_list, MS_ARE_list,yerr=MS_ARE_std_list, color="orange",marker = 'd', markersize = 20, label="MS",capsize=15,linewidth=4,mfc='w',capthick=4)
    for (i,model_name) in enumerate(model_name_list):
        if not ablation:
            model_name = ''
            if normal_skew or True:
                plt.errorbar(cluster_budget_list, get_error(model_dec_ARE_list[i]),  label=model_name+"Lego", yerr=get_std(model_dec_ARE_list[i]),color=color_list[i],capsize=15,linewidth=4, marker=marker_list[i], markersize=20, mfc='w',capthick=4)
            plt.errorbar(cluster_budget_list, get_error(model_es_ARE_list[0]),  color="brown", label="ES(CMS)", linestyle='-',linewidth=8,alpha=0.7)
            plt.errorbar(cluster_budget_list, get_error(model_cluster_in_es_dec_ARE_list[i]),  label=model_name+"ES(Lego)",color=color_list[i+1], yerr=get_std(model_cluster_in_es_dec_ARE_list[i]),capsize=15,linewidth=4, marker=marker_list[i+1], markersize=20, mfc='w',capthick=4)
        else:
            plt.errorbar(cluster_budget_list, get_error(model_dec_ARE_list[i]),  label=model_name, yerr=get_std(model_dec_ARE_list[i]),color=color_list[i],capsize=15,linewidth=4, marker=marker_list[i], markersize=20, mfc='w',capthick=4)
   
    plt.xlabel('Budget(MB)',fontsize='35')
    plt.ylabel('ARE',fontsize='35')
    plt.ylim(0,ARE_y_lim)
    plt.locator_params(axis='x', nbins=6) 
    plt.locator_params(axis='y', nbins=6) 
    plt.xticks(fontsize='35')
    plt.yticks(fontsize='35')
    plt.legend(prop={'size': 33},loc='upper right')
    plt.subplots_adjust(left=0.2,right=0.95)
    if not ablation:
        plt.savefig(os.path.join(project_path,'EvalModel/EvalDataset/Figure/'+dataset_name+'_ARE'))
    else:
        plt.savefig(os.path.join(project_path,'EvalModel/EvalDataset/Figure/'+dataset_name+'_ARE_ablation'))


    plt.figure(figsize=(10, 10))
    if not ablation:
        plt.plot(cluster_budget_list, get_error(model_count_min_AAE_list[0]), color="gray", label="CMS", linestyle=':',linewidth=8,)
        plt.plot(cluster_budget_list, get_error(model_count_sketch_AAE_list[0]), color="gray", label="CS", linestyle='--',linewidth=8,)
        plt.plot(cluster_budget_list, get_error(model_learned_count_sketch_AAE_list[0]), color="brown", label="LCS", linestyle=(0,(3,1,1,1)),linewidth=8,alpha=0.7)
        plt.errorbar(MS_budget_list, MS_AAE_list,yerr=MS_AAE_std_list, color="orange",marker = 'd', markersize = 20, label="MS",capsize=15,linewidth=4,mfc='w', capthick=4)
    for (i,model_name) in enumerate(model_name_list):
        if not ablation:
            model_name = ''
            if normal_skew or True:
                plt.errorbar(cluster_budget_list, get_error(model_dec_AAE_list[i]),  label=model_name+"Lego",color=color_list[i],yerr=get_std(model_dec_AAE_list[i]), capsize=15,linewidth=4,marker=marker_list[i], markersize=20, mfc='w',capthick=4)
            plt.errorbar(cluster_budget_list, get_error(model_es_AAE_list[0]), color="brown", label="ES(CMS)", linestyle='-',linewidth=8,alpha=0.7)
            plt.errorbar(cluster_budget_list, get_error(model_cluster_in_es_dec_AAE_list[i]),  label=model_name+"ES(Lego)",color=color_list[i+1], yerr=get_std(model_cluster_in_es_dec_AAE_list[i]),capsize=15,linewidth=4,marker=marker_list[i+1], markersize=20, mfc='w',capthick=4)
        else:
            plt.errorbar(cluster_budget_list, get_error(model_dec_AAE_list[i]),  label=model_name,color=color_list[i],yerr=get_std(model_dec_AAE_list[i]), capsize=15,linewidth=4,marker=marker_list[i], markersize=20, mfc='w',capthick=4)

   

    plt.xlabel('Budget(MB)',fontsize='35')
    plt.ylabel('AAE',fontsize='35')
    plt.locator_params(axis='x', nbins=6) 
    plt.locator_params(axis='y', nbins=6) 
    plt.xticks(fontsize='35')
    plt.yticks(fontsize='35')
    plt.ylim(0,AAE_y_lim)
    plt.legend(prop={'size': 33},loc='upper right')
    plt.subplots_adjust(left=0.2,right=0.95) 

    # plt.tight_layout()
    if not ablation:
        plt.savefig(os.path.join(project_path,'EvalModel/EvalDataset/Figure/'+dataset_name+'_AAE'))
    else:
        plt.savefig(os.path.join(project_path,'EvalModel/EvalDataset/Figure/'+dataset_name+'_AAE_ablation'))


def eval_models_with_budgets_with_seed(cluster_budget_list,dataset_name,device,model_path_list,use_cuda_hash,eval_elastic,write_batch_size=10000000,):
    model_dec_ARE_list = []
    model_count_min_ARE_list = []
    model_count_sketch_ARE_list = []
    model_cm_read_ARE_list = []
    model_es_ARE_list = []
    model_cluster_in_es_dec_ARE_list = []
    model_cluster_in_es_cm_ARE_list = []
    model_learned_count_sketch_ARE_list = []
    model_dec_AAE_list = []
    model_count_min_AAE_list = []
    model_count_sketch_AAE_list = []
    model_cm_read_AAE_list = []
    model_es_AAE_list = []
    model_cluster_in_es_dec_AAE_list = []
    model_cluster_in_es_cm_AAE_list = []
    model_learned_count_sketch_AAE_list = []
    for model_path in model_path_list:
        dec_ARE_list,count_min_ARE_list,count_sketch_ARE_list,cm_read_ARE_list,es_ARE_list,cluster_in_es_dec_ARE_list,cluster_in_es_cm_ARE_list,learned_count_sketch_ARE_list,\
            dec_AAE_list,count_min_AAE_list,count_sketch_AAE_list,cm_read_AAE_list,es_AAE_list,cluster_in_es_dec_AAE_list,cluster_in_es_cm_AAE_list,learned_count_sketch_AAE_list = \
            eval_model_with_budgets_with_seed(cluster_budget_list,dataset_name,device,model_path,use_cuda_hash,eval_elastic,write_batch_size)
        model_dec_ARE_list.append(dec_ARE_list)
        model_count_min_ARE_list.append(count_min_ARE_list)
        model_count_sketch_ARE_list.append(count_sketch_ARE_list)
        model_cm_read_ARE_list.append(cm_read_ARE_list)
        model_es_ARE_list.append(es_ARE_list)
        model_cluster_in_es_dec_ARE_list.append(cluster_in_es_dec_ARE_list)
        model_cluster_in_es_cm_ARE_list.append(cluster_in_es_cm_ARE_list)
        model_learned_count_sketch_ARE_list.append(learned_count_sketch_ARE_list)
        model_dec_AAE_list.append(dec_AAE_list)
        model_count_min_AAE_list.append(count_min_AAE_list)
        model_count_sketch_AAE_list.append(count_sketch_AAE_list)
        model_cm_read_AAE_list.append(cm_read_AAE_list)
        model_es_AAE_list.append(es_AAE_list)
        model_cluster_in_es_dec_AAE_list.append(cluster_in_es_dec_AAE_list)
        model_cluster_in_es_cm_AAE_list.append(cluster_in_es_cm_AAE_list)
        model_learned_count_sketch_AAE_list.append(learned_count_sketch_AAE_list)
    return model_dec_ARE_list,model_count_min_ARE_list,model_count_sketch_ARE_list,model_cm_read_ARE_list,model_es_ARE_list,model_cluster_in_es_dec_ARE_list,model_cluster_in_es_cm_ARE_list,model_learned_count_sketch_ARE_list,\
        model_dec_AAE_list,model_count_min_AAE_list,model_count_sketch_AAE_list,model_cm_read_AAE_list,model_es_AAE_list,model_cluster_in_es_dec_AAE_list,model_cluster_in_es_cm_AAE_list,model_learned_count_sketch_AAE_list

def eval_model_with_budgets_with_seed(cluster_budget_list,dataset_name,device,model_path,use_cuda_hash,eval_elastic,write_batch_size=10000000):
    dec_ARE_list = []
    count_min_ARE_list = []
    count_sketch_ARE_list = []
    cm_read_ARE_list = []
    es_ARE_list = []
    cluster_in_es_dec_ARE_list = []
    cluster_in_es_cm_ARE_list = []
    learned_count_sketch_ARE_list = []


    dec_AAE_list = []
    count_min_AAE_list = []
    count_sketch_AAE_list = []
    cm_read_AAE_list = []
    es_AAE_list = []
    cluster_in_es_dec_AAE_list = []
    cluster_in_es_cm_AAE_list = []
    learned_count_sketch_AAE_list = []



    for budget in tqdm.tqdm(cluster_budget_list):
        seed_dec_ARE_list = []
        seed_count_min_ARE_list = []
        seed_count_sketch_ARE_list = []
        seed_cm_read_ARE_list = []
        seed_es_ARE_list = []
        seed_cluster_in_es_dec_ARE_list = []
        seed_cluster_in_es_cm_ARE_list = []
        seed_learned_count_sketch_ARE_list = []
        seed_dec_AAE_list = []
        seed_count_min_AAE_list = []
        seed_count_sketch_AAE_list = []
        seed_cm_read_AAE_list = []
        seed_es_AAE_list = []
        seed_cluster_in_es_dec_AAE_list = []
        seed_cluster_in_es_cm_AAE_list = []
        seed_learned_count_sketch_AAE_list = []

        
        for i,seed_model in enumerate(model_path):
            dec_ARE,count_min_ARE,count_sketch_ARE,cm_read_ARE,es_ARE,cluster_in_es_dec_ARE,cluster_in_es_cm_ARE,learned_count_sketch_ARE,\
            dec_AAE,count_min_sketch_AAE,count_sketch_AAE,cm_readhead_AAE,es_AAE,cluster_in_es_dec_AAE,cluster_in_es_cm_AAE,learned_count_sketch_AAE\
                = eval_model_with_budget(budget, dataset_name, device, seed_model,use_cuda_hash,eval_elastic,write_batch_size,seed_index=i)

            if  eval_elastic:
                seed_es_ARE_list.append(es_ARE)
                seed_cluster_in_es_dec_ARE_list.append(cluster_in_es_dec_ARE)
                seed_cluster_in_es_cm_ARE_list.append(cluster_in_es_cm_ARE)

                seed_es_AAE_list.append(es_AAE)
                seed_cluster_in_es_dec_AAE_list.append(cluster_in_es_dec_AAE)
                seed_cluster_in_es_cm_AAE_list.append(cluster_in_es_cm_AAE)        

                if i == 0:
                    seed_count_sketch_ARE_list.append(count_sketch_ARE)
                    seed_learned_count_sketch_ARE_list.append(learned_count_sketch_ARE)
                    seed_count_min_ARE_list.append(count_min_ARE)
                    seed_count_sketch_AAE_list.append(count_sketch_AAE)
                    seed_learned_count_sketch_AAE_list.append(learned_count_sketch_AAE)
                    seed_count_min_AAE_list.append(count_min_sketch_AAE)

            seed_dec_ARE_list.append(dec_ARE)
            seed_cm_read_ARE_list.append(cm_read_ARE)
            seed_dec_AAE_list.append(dec_AAE)
            seed_cm_read_AAE_list.append(cm_readhead_AAE)

        if eval_elastic:
            es_ARE_list.append((np.mean(seed_es_ARE_list),np.std(seed_es_ARE_list)))
            cluster_in_es_dec_ARE_list.append((np.mean(seed_cluster_in_es_dec_ARE_list),np.std(seed_cluster_in_es_dec_ARE_list)))
            cluster_in_es_cm_ARE_list.append((np.mean(seed_cluster_in_es_cm_ARE_list),np.std(seed_cluster_in_es_cm_ARE_list)))
            count_sketch_ARE_list.append((np.mean(seed_count_sketch_ARE_list),np.std(seed_count_sketch_ARE_list)))
            learned_count_sketch_ARE_list.append((np.mean(seed_learned_count_sketch_ARE_list),np.std(seed_learned_count_sketch_ARE_list)))
            count_min_ARE_list.append((np.mean(seed_count_min_ARE_list),np.std(seed_count_min_ARE_list)))
            es_AAE_list.append((np.mean(seed_es_AAE_list),np.std(seed_es_AAE_list)))
            cluster_in_es_dec_AAE_list.append((np.mean(seed_cluster_in_es_dec_AAE_list),np.std(seed_cluster_in_es_dec_AAE_list)))
            cluster_in_es_cm_AAE_list.append((np.mean(seed_cluster_in_es_cm_AAE_list),np.std(seed_cluster_in_es_cm_AAE_list)))
            count_sketch_AAE_list.append((np.mean(seed_count_sketch_AAE_list),np.std(seed_count_sketch_AAE_list)))
            learned_count_sketch_AAE_list.append((np.mean(seed_learned_count_sketch_AAE_list),np.std(seed_learned_count_sketch_AAE_list)))
            count_min_AAE_list.append((np.mean(seed_count_min_AAE_list),np.std(seed_count_min_AAE_list)))

        dec_ARE_list.append((np.mean(seed_dec_ARE_list),np.std(seed_dec_ARE_list)))
        cm_read_ARE_list.append((np.mean(seed_cm_read_ARE_list),np.std(seed_cm_read_ARE_list)))
        dec_AAE_list.append((np.mean(seed_dec_AAE_list),np.std(seed_dec_AAE_list)))
        cm_read_AAE_list.append((np.mean(seed_cm_read_AAE_list),np.std(seed_cm_read_AAE_list)))
    



    return dec_ARE_list,count_min_ARE_list,count_sketch_ARE_list,cm_read_ARE_list,es_ARE_list,cluster_in_es_dec_ARE_list,cluster_in_es_cm_ARE_list,learned_count_sketch_ARE_list,\
        dec_AAE_list,count_min_AAE_list,count_sketch_AAE_list,cm_read_AAE_list,es_AAE_list,cluster_in_es_dec_AAE_list,cluster_in_es_cm_AAE_list,learned_count_sketch_AAE_list