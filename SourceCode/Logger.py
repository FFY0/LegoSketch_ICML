import csv
import json
import os
import sys
import numpy as np
import torch
import time
import shutil

class Logger():
    def __init__(self, test_meta_task_group_list, meta_task_group_discribe_list, dataset_name, config,
                 flush_gap=10, train_comment=" ", eval_gap=1000, save_gap=10000, prod_env=False):
        self.flush_gap = flush_gap
        self.config = config
        self.model_path = None
        self.save_gap = save_gap
        self.meta_task_discribe_list = meta_task_group_discribe_list
        self.test_meta_task_group_list = test_meta_task_group_list
        self.train_comment = train_comment
        self.project_root = os.getcwd().split('ExpCode')[0]
        path = os.getcwd()
        self.path = path
        self.py_file_path = sys.argv[0]
        print(self.py_file_path)
        if prod_env:
            self.dataset_name = "Prod_" + time.strftime('%m%d_', time.localtime(time.time())) + dataset_name
        else:
            self.dataset_name = "Dev_" + time.strftime('%m%d_', time.localtime(time.time())) + dataset_name
        self.csv_writer_list = None
        self.log_file_list = None
        self.__init_log_file()
        self.file_header = None
        self.eval_gap = eval_gap

    def adjust_net_arch(self, model):
        sample_task = self.test_meta_task_group_list[0][0]
        with torch.no_grad():
            support_set = sample_task.support_set
            query_set = sample_task.query_set
            with torch.no_grad():
                weight_sum = support_set.support_y.sum(dim=0, keepdim=True)
            statis_info = torch.concat((weight_sum,))
            reshape_statis_info = statis_info.unsqueeze(0).repeat(query_set.query_x.shape[0], 1)
            model.adjust_net_arch(query_set.query_x,  reshape_statis_info)


    def logging(self, model, step, lr):
        print(step, ' step logging begin...')
        model.eval()
        for i in range(len(self.log_file_list)):
            if self.file_header is None:
                for group in self.test_meta_task_group_list:
                    for task in group:
                        task.to_device()
            log_file = self.log_file_list[i]
            csv_writer = self.csv_writer_list[i]
            test_meta_task_group = self.test_meta_task_group_list[i]
            group_test_merged_info_dict = self.eval_on_one_group(test_meta_task_group, model)
            if self.file_header is None:
                self.file_header = list(group_test_merged_info_dict.keys())
                self.file_header.insert(0, "task_num")
                self.file_header.insert(1, "lr")
                for writer in self.csv_writer_list:
                    writer.writerow(self.file_header)
            group_test_merged_info_dict['task_num'] = step
            group_test_merged_info_dict['lr'] = lr
            row_content = []
            for key in self.file_header:
                row_content.append(group_test_merged_info_dict[key])
            csv_writer.writerow(row_content)
            if (step // self.eval_gap) % self.flush_gap == 0:
                log_file.flush()
        model.train()
        try:
            torch.save(model, self.model_path)
        except IOError:
            print('save model IOError except')
        print(step, ' step logging done...')


    def eval_on_one_group(self, test_meta_task_group, model):
        # store all test result for a group of meta tasks
        group_dict_list = []
        for test_meta_task in test_meta_task_group:
            mt_test_info_dict = self.eval_on_one_task(test_meta_task, model)
            group_dict_list.append(mt_test_info_dict)

        group_test_merged_info_dict = {}
        for key in group_dict_list[0].keys():
            value_mean = 0
            for dic in group_dict_list:
                value_mean += dic[key]
            value_mean /= len(group_dict_list)
            group_test_merged_info_dict[key] = value_mean
        return group_test_merged_info_dict

    def eval_on_one_task(self, test_meta_task, model):
        basic_info_dict = self.get_basic_eval_info_on_one_task(test_meta_task, model)
        additional_info_dict = self.get_additional_eval_info_on_one_task(test_meta_task, model)
        # Merge two types of information dictionaries
        info_dict = dict(list(basic_info_dict.items()) + list(additional_info_dict.items()))
        return info_dict

    def get_sparsity(self, batch_data, dim=(1,)):
        sparsity_data = torch.where(torch.abs(batch_data - 0.0) < 0.00001, 0.0, 1.0)
        res = sparsity_data.sum(dim=dim, keepdim=False)
        return res

    def get_additional_eval_info_on_one_task(self, test_meta_task, model):
        info_dict = {}
        with torch.no_grad():
            mean_V = model.embedding_module.V.data.mean().cpu().item()
            std_V = model.embedding_module.V.data.std().cpu().item()
            Max_V = model.embedding_module.V.data.max().cpu().item()
            Min_V = model.embedding_module.V.data.min().cpu().item()
            info_dict['mean_V'] = mean_V
            info_dict['std_V'] = std_V
            info_dict['Max_V'] = Max_V
            info_dict['Min_V'] = Min_V
        return info_dict

    #  basic eval info in log
    def get_basic_eval_info_on_one_task(self, test_meta_task, model):
        info_dict = {}
        with torch.no_grad():
            support_set = test_meta_task.support_set
            query_set = test_meta_task.query_set
            with torch.no_grad():
                weight_sum = support_set.support_y.sum(dim=0, keepdim=True)
                weight_sum = weight_sum.unsqueeze(0).repeat(query_set.query_x.shape[0], 1)

            model.clear()
            model.write(support_set.support_x, support_set.support_y)
            dec_pred,cm_readhead,stream_info1,stream_info2= model.dec_query(query_set.query_x,weight_sum)
            query_y = query_set.query_y
            dec_loss = model.loss_func.forward_dec(dec_pred,cm_readhead,query_set.query_y,support_set.info,stream_info1,stream_info2)
            info_dict['Est_zipf'] = stream_info1.item()
            info_dict['Est_Item_size'] = stream_info2.item()
            info_dict['Dec_loss'] = dec_loss.cpu().item()
            info_dict['Item_size'] = dec_pred.shape[0]
            info_dict["Label_var"] = query_y.var().cpu().item()
            info_dict["Dec_var"] = dec_pred.var().cpu().item()
            info_dict["CM_var"] = cm_readhead.var().cpu().item()
            info_dict['Label_sum'] = query_set.query_y.sum().cpu().item()
            info_dict['Dec_sum'] = dec_pred.sum().cpu().item()
            info_dict['CM_sum'] = cm_readhead.sum().cpu().item()
            dec_aae = torch.mean(torch.abs(dec_pred - query_y)).cpu().item()
            dec_are = torch.mean(torch.abs((dec_pred - query_y) / query_y)).cpu().item()
            dec_aoe = torch.mean(torch.abs((dec_pred - query_y)).sum() / query_y.abs().sum()).cpu().item()
            info_dict['Dec_ARE'] = dec_are
            info_dict['Dec_AAE'] = dec_aae
            info_dict['Dec_AOE'] = dec_aoe
            cm_aae = torch.mean(torch.abs(cm_readhead - query_set.query_y)).cpu().item()
            cm_are = torch.mean(torch.abs((cm_readhead - query_set.query_y) / query_set.query_y)).cpu().item()
            cm_aoe = torch.mean(torch.abs((cm_readhead - query_set.query_y)).sum() / query_set.query_y.abs().sum()).cpu().item()
            info_dict['CM_ARE'] = cm_are
            info_dict['CM_AAE'] = cm_aae
            info_dict['CM_AOE'] = cm_aoe
        return info_dict

    # Create a log file for each group
    def __init_log_file(self):
        self.log_file_list = []
        self.csv_writer_list = []
        time_str = time.strftime('_%m_%d_%H_%M_%S', time.localtime(time.time()))
        if not os.path.exists(os.path.join(self.project_root,
                                           'LogDir/{}/{}/'.format(self.dataset_name, self.train_comment + time_str))):
            os.makedirs(os.path.join(self.project_root,
                                     'LogDir/{}/{}/'.format(self.dataset_name, self.train_comment + time_str)))
        self.model_path = os.path.join(self.project_root, 'LogDir/{}/{}/{}_model'.format(self.dataset_name,
                                                                                              self.train_comment + time_str,self.train_comment))
        self.model_dir = os.path.join(self.project_root, 'LogDir/{}/{}/'.format(self.dataset_name,
                                                                                self.train_comment + time_str))

        config_str = json.dumps(self.config)
        config_file = open(os.path.join(self.project_root, 'LogDir/{}/{}/config'.format(self.dataset_name,
                                                                                        self.train_comment + time_str)),
                           'w',
                           newline='', encoding='utf-8')
        config_file.write(config_str)
        config_file.close()
        for meta_task_group_discribe in self.meta_task_discribe_list:
            self.log_file_list.append(open(os.path.join(self.project_root,
                                                        'LogDir/{}/{}/log{}.csv'.format(self.dataset_name,
                                                                                        self.train_comment + time_str,
                                                                                        meta_task_group_discribe)), 'w',
                                           newline='', encoding='utf-8'))
            self.csv_writer_list.append(csv.writer(self.log_file_list[-1]))
        os.makedirs(os.path.join(self.project_root,
                                 'LogDir/{}/{}/test_tasks_{}/'.format(self.dataset_name, self.train_comment + time_str,
                                                                      self.train_comment + time_str)))
        for i in range(len(self.test_meta_task_group_list)):
            os.makedirs(os.path.join(self.project_root,
                                     'LogDir/{}/{}/test_tasks_{}/{}/'.format(self.dataset_name,
                                                                             self.train_comment + time_str,
                                                                             self.train_comment + time_str,
                                                                             self.meta_task_discribe_list[i])))
            for j in range(len(self.test_meta_task_group_list[i])):
                path = os.path.join(self.project_root,
                                    'LogDir/{}/{}/test_tasks_{}/{}/{}.npz'.format(self.dataset_name,
                                                                                  self.train_comment + time_str,
                                                                                  self.train_comment + time_str,
                                                                                  self.meta_task_discribe_list[i],
                                                                                  str(j)))

                self.save_meta_task(self.test_meta_task_group_list[i][j], path)
        print('Persistent test meta task complete...')
        shutil.copytree(os.path.join(self.project_root, 'SourceCode'), os.path.join(self.project_root,
                                                                                    'LogDir/{}/{}/SourceCode/'.format(
                                                                                        self.dataset_name,
                                                                                        self.train_comment + time_str)))
        if '\\' in self.py_file_path:
            self.py_file_path = self.py_file_path.replace('\\', "/")
            self.path = self.path.replace('\\', "/")
        os.makedirs(os.path.join(self.project_root,
                                 'LogDir/{}/{}/ExpCode/{}/'.format(self.dataset_name, self.train_comment + time_str,
                                                                   self.path.split('/')[-1])))
        shutil.copy(self.py_file_path, os.path.join(self.project_root,
                                                    'LogDir/{}/{}/ExpCode/{}/'.format(self.dataset_name,
                                                                                      self.train_comment + time_str,
                                                                                      self.path.split('/')[-1])))
        print('The code backup is complete.')

    def save_meta_task(self, meta_task, path):
        np.savez(path, support_x=meta_task.support_set.support_x.cpu().numpy(),
                 support_y=meta_task.support_set.support_y.cpu().numpy(),
                 query_x=meta_task.query_set.query_x.cpu().numpy(), query_y=meta_task.query_set.query_y.cpu().numpy(),)

    def close_all_file(self):
        for file in self.log_file_list:
            file.close()
