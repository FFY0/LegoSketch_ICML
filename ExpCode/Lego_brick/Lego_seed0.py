import os
import sys
import argparse
import multiprocessing
os.sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from SourceCode.Factory import Factory

def parse_command_line(config):
    parser = argparse.ArgumentParser()
    parser.add_argument('--prod', action='store_true', default=False)
    parser.add_argument('--pass_cuda', action='store_true', default=False)
    parser.add_argument('--pass_cpu', action='store_true', default=False)
    args = parser.parse_args()
    prod = args.prod
    pass_cuda = args.pass_cuda
    pass_cpu = args.pass_cpu
    if prod:
        print('working in prod, set big batch size and eval gap:')
    else:
        print('working in dev, set small hyperparameter for less memory consume')
    print('memory size:',  config['dim_config']['embedding_dim'] * config['dim_config']['slot_dim'] * 4 / 1024, 'KB')
    if pass_cuda:
        pass_cuda_tensor = True
        print('passing gpu tensor,attention: must in linux!!')
    elif pass_cpu:
        pass_cuda_tensor = False
        print('passing cpu tensor')

    else:
        if not prod:
            pass_cuda_tensor = False
            print('working in pc environment, passing cpu tensor')
        else:
            pass_cuda_tensor = True
            print('working in prod environment ,passing gpu tensor,attention: must in linux!!')
    return prod, pass_cuda_tensor

def init_config():
    data_config = {
        "train_comment": "Base_seed0",
        "dataset_name": 'ZipfDatasetByIS',
        'dataset_path': None,
        'skew_lower': None,
        'skew_upper': None,
        'is_upper': 50000,
        'is_lower': 1000,
        "zipf_param_upper": 0.5,
        "zipf_param_lower": 1.0,
        "base_weight_sum":None,
        "cons_lower":0.001,
        "cons_upper":1,
    }
    dim_config = {
        "embedding_dim": 5,
        "slot_dim": 512*5*2,
        "dep_dim": 5,
        "input_dim": 14,
    }
    train_config = {
        "train_step": 4000000,
        "lr": 0.0001,
        "cuda_num": 0,
        'queue_size': 20,
    }
    logger_config = {
        "flush_gap": 1,
        "save_gap": 10000,
        "eval_gap": 10000,
        "test_task_skew_ratio_list": [0.1, 1, 10],
        "test_task_group_size": 2,
        "test_zipf_param_list": [0.5, 0.7, 0.9,1.1, 1.3,1.5,],
        "item_size_list":  [5000,10000,20000,30000]
    }
    factory_config = {
        "loss_class": "LossFunc_Difficulty_SQUARE",
        "MemoryModule_class": "MemoryModule",
        "AddressModule_class": "RandomAddressModule",
        "EmbeddingModule_class": "EmbeddingModule",
        "sample_IS": "linear",
        "DecodeModule_class": "SetDecodeModule",
        "model_class": "Model",
    }
    config = {
        "train_config": train_config,
        "factory_config": factory_config,
        "dim_config": dim_config,
        "data_config": data_config,
        "logger_config": logger_config
    }
    return config

if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    config = init_config()
    train_config = config['train_config']
    prod, pass_cuda_tensor = parse_command_line(config=config)
    factory = Factory(config)
    MGS = factory.init_Lego(prod)
    MGS.train(train_config["train_step"], pass_cuda_tensor, train_config['queue_size'])
