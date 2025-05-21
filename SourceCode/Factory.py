import torch.multiprocessing.queue
from SourceCode.LegoSketch import LegoSketchTraining
from SourceCode.Model import *
from SourceCode.TaskRelatedClasses.QueryGenerator import SimpleQueryGenerator
from SourceCode.TaskRelatedClasses.SupportGenerator import ZipfSupportGeneratorByIS
from SourceCode.TaskRelatedClasses.TaskConsumer import *
from SourceCode.ModelModule.MemoryModule import *
from SourceCode.ModelModule.AddressModule import *
from SourceCode.ModelModule.EmbeddingModule import *
from SourceCode.ModelModule.DecodeModule import *
from SourceCode.ModelModule.LossFunc import *

from SourceCode.Logger import *
from SourceCode.TaskRelatedClasses.TaskProducer import TaskProducer
import os
import random
import numpy as np
from SourceCode.ModelModule.SubModuleUtil import *


class Factory:
    def __init__(self, config):
        self.config = config
        self.seed_everything(0)

    def seed_everything(self, seed=0):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True

    def init_Lego(self, prod_env):
        print('init model....')
        model = self.init_model()
        device = self.init_device()
        task_producer, task_consumer = self.init_producer_consumer(device)
        test_meta_task_group_list, meta_task_group_discribe_list = task_producer.produce_test_task()
        logger = self.init_logger(test_meta_task_group_list, meta_task_group_discribe_list, prod_env)
        logger.adjust_net_arch(model)
        main_optimizer = self.init_optimizer(model)
        Lego = LegoSketchTraining(task_producer, task_consumer, model, device, main_optimizer, logger)
        print('init model done!')
        return Lego

    def init_producer_consumer(self, device):
        dataset_name = self.config["data_config"]["dataset_name"]
        skew_lower = self.config['data_config']['skew_lower']
        skew_upper = self.config['data_config']['skew_upper']
        base_weight_sum = self.config['data_config']['base_weight_sum']
        dataset_path = self.config['data_config']['dataset_path']
        test_task_skew_ratio_list = self.config['logger_config']['test_task_skew_ratio_list']
        test_task_group_size = self.config['logger_config']['test_task_group_size']
        zipf_param_upper = self.config['data_config']['zipf_param_upper']
        zipf_param_lower = self.config['data_config']['zipf_param_lower']
        test_zipf_param_list = self.config['logger_config']['test_zipf_param_list']
        embedding_dim = self.config["dim_config"]["embedding_dim"]
        input_dim = self.config["dim_config"]["input_dim"]
        train_step = self.config['train_config']['train_step']
        task_producer = None
        task_consumer = None

        if dataset_name == "ZipfDatasetByIS":
            is_upper = self.config['data_config']['is_upper']
            is_lower = self.config['data_config']['is_lower']
            item_size_list = self.config['logger_config']['item_size_list']
            sample_IS = self.config['factory_config']['sample_IS']

            zipf_support_generator = ZipfSupportGeneratorByIS(zipf_param_upper=zipf_param_upper,
                                                              zipf_param_lower=zipf_param_lower,
                                                              is_upper=is_upper,
                                                              is_lower=is_lower,
                                                              device=device, total_task_num=train_step,
                                                              base_weight_sum=base_weight_sum,
                                                              sample_IS_scale=sample_IS)
            query_generator = SimpleQueryGenerator(device)
            # set zipf_decorate True due to zipf basic SketchCode
            task_producer = TaskProducer(zipf_support_generator, query_generator, device,
                                         test_task_skew_ratio_list=test_task_skew_ratio_list,
                                         test_task_group_size=test_task_group_size,
                                         test_zipf_param_list=test_zipf_param_list,
                                         item_size_list=item_size_list)
            task_consumer = TaskConsumer(device)

        assert task_producer is not None and task_consumer is not None, 'error! no producer or consumer has been constructed'
        return task_producer, task_consumer

    def init_logger(self, test_meta_task_group_list, meta_task_group_discribe_list, prod_env):
        logger = Logger(test_meta_task_group_list, meta_task_group_discribe_list,
                        self.config['data_config']['dataset_name'],
                        config=self.config,
                        flush_gap=self.config['logger_config']['flush_gap'],
                        train_comment=self.config['data_config']['train_comment'],
                        eval_gap=self.config['logger_config']['eval_gap'],
                        save_gap=self.config['logger_config']['save_gap'], prod_env=prod_env)
        return logger

    def init_device(self):
        cuda_num = self.config['train_config']['cuda_num']
        if cuda_num == -1:
            device = torch.device('cpu')
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_num)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device

    def init_loss_func(self, loss_class):
        return eval(loss_class + "()")

    def init_ensemble_module(self, ensemble_class):
        return eval(ensemble_class + "()")

    def init_optimizer(self, model):
        lr = self.config['train_config']['lr']
        main_module_params_list = []
        embedding_module_params_list = []
        for name, param in model.named_parameters():
            if "embed" in name:
                embedding_module_params_list.append(param)
            else:
                main_module_params_list.append(param)
        main_optimizer = torch.optim.Adam([
            {"params": main_module_params_list, 'lr': lr * 10},
            {"params": embedding_module_params_list, 'lr': lr * 1}
        ], lr=lr)

        return main_optimizer

    def init_model(self):
        slot_dim = self.config["dim_config"]["slot_dim"]
        embedding_dim = self.config["dim_config"]["embedding_dim"]
        dep_dim = self.config["dim_config"]["dep_dim"]
        input_dim = self.config["dim_config"]["input_dim"]
        address_class = self.config["factory_config"]["AddressModule_class"]
        memory_class = self.config["factory_config"]["MemoryModule_class"]
        decode_class = self.config["factory_config"]["DecodeModule_class"]
        embedding_class = self.config["factory_config"]["EmbeddingModule_class"]

        model = None
        if self.config['factory_config']['model_class'] == "Model":
            assert dep_dim == embedding_dim
            embedding_net = self.init_embedding_module(embedding_class, input_dim, embedding_dim)
            attention_module = self.init_address_module(address_class, dep_dim, slot_dim)
            memory_matrix, decode_net = self.init_memory_module_and_decode_module(memory_class, decode_class, slot_dim,
                                                                                  embedding_dim)
            loss_func = self.init_loss_func(self.config['factory_config']['loss_class'])
            model = Model(attention_module, embedding_net, decode_net, memory_matrix, loss_func)
        assert model is not None, 'error! no model has been constructed'
        return model

    def init_error_module(self, error_class):
        return eval(error_class + "()")

    def init_address_module(self, address_class, dep_dim, slot_dim):
        return eval(address_class + "(dep_dim, slot_dim)")

    def init_embedding_module(self, embedding_class, input_dim, embedding_dim):
        # construct embedding module
        cons_lower = self.config['data_config']['cons_lower']
        cons_upper = self.config['data_config']['cons_upper']
        embedding_module = eval(embedding_class + "(input_dim, embedding_dim,cons_lower,cons_upper)")
        return embedding_module

    def init_memory_module_and_decode_module(self, memory_class, decode_class, slot_dim, embedding_dim):
        memory_module = eval(memory_class + "(slot_dim, embedding_dim)")
        assert memory_module is not None, 'error! no memory_matrix has been constructed'
        weight_decode_net = eval(decode_class + "()")
        return memory_module, weight_decode_net
