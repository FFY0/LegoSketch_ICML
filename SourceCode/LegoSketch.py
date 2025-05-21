from multiprocessing import Process
import torch
from torch.optim.lr_scheduler import ExponentialLR


class LegoSketchTraining():

    def __init__(self, task_producer, task_consumer, model, device, main_optimizer, logger):
        self.task_consumer = task_consumer
        self.task_producer = task_producer
        self.device = device
        self.model = model
        self.main_optimizer = main_optimizer
        self.logger = logger
        self.dec_batch_size = 8
        # Resize to accommodate GPU memory constraints.
        self.dec_cumulative_size = 8
        self.scheduler_num = 10
        self.minium_lr = 0.0001
        self.dec_inner_step = 1000

    def train(self, train_task_nums, pass_cuda_tensor=False, queue_size=20, ):
        q = torch.multiprocessing.Queue(queue_size)
        p = Process(target=self.task_producer.produce_train_task, args=(q, pass_cuda_tensor))
        p.start()
        size = self.minium_lr / self.main_optimizer.param_groups[0]['lr']
        gamma = size ** (1 / self.scheduler_num)
        main_scheduler = ExponentialLR(self.main_optimizer, gamma=gamma)
        self.model.to(self.device)
        meta_task_num = 0
        self.logger.logging(self.model, 0, self.main_optimizer.param_groups[0]['lr'])
        cumulative_batch = self.dec_batch_size // self.dec_cumulative_size
        for i in range(train_task_nums // self.dec_batch_size):
            self.main_optimizer.zero_grad()
            for j in range(cumulative_batch):
                self.main_optimizer.zero_grad()
                batch_meta_task_list = []
                batch_weight_sum_list = []
                item_size_list = []
                for k in range(self.dec_cumulative_size):
                    meta_task_num += 1
                    meta_task = self.task_consumer.consume_train_task(q, pass_cuda_tensor)
                    # print(q.qsize(),'tasks in queue')
                    with torch.no_grad():
                        weight_sum = meta_task.support_set.support_y.sum(dim=0, keepdim=True)
                    batch_weight_sum_list.append(weight_sum)
                    if meta_task_num % self.logger.eval_gap == 0:
                        self.logger.logging(self.model, meta_task_num, self.main_optimizer.param_groups[0]['lr'])
                    item_size_list.append(meta_task.support_set.support_x.shape[0])
                    batch_meta_task_list.append(meta_task)
                batch_weight_sum_tensor = torch.cat(batch_weight_sum_list, dim=0)
                batch_weight_sum_tensor = torch.repeat_interleave(batch_weight_sum_tensor, torch.tensor(item_size_list,
                                                                                                        device=batch_weight_sum_tensor.device),
                                                                  dim=0).view(-1, 1)
                batch_support_x = torch.cat([meta_task.support_set.support_x for meta_task in batch_meta_task_list],
                                            dim=0)
                batch_supprt_y = torch.cat([meta_task.support_set.support_y for meta_task in batch_meta_task_list],
                                           dim=0)
                batch_query_x = torch.cat([meta_task.query_set.query_x for meta_task in batch_meta_task_list], dim=0)
                batch_query_y = torch.cat([meta_task.query_set.query_y for meta_task in batch_meta_task_list], dim=0)
                zipf_info = torch.cat([meta_task.support_set.info for meta_task in batch_meta_task_list], dim=0).view(
                    -1, 1)
                for meta_task in batch_meta_task_list:
                    self.task_consumer.del_meta_task(meta_task)
                self.model.clear(batch_size=len(item_size_list))
                self.model.batch_write(batch_support_x, batch_supprt_y, item_size_list)
                batch_dec_pred, batch_cm_readhead, stream_info1, stream_info2 = self.model.batch_dec_query(
                    batch_query_x, batch_weight_sum_tensor, item_size_list)
                dec_loss = self.model.loss_func.batch_forward_dec(batch_dec_pred, batch_cm_readhead, batch_query_y,
                                                                  item_size_list, zipf_info, stream_info1,
                                                                  stream_info2) / cumulative_batch
                dec_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
            self.main_optimizer.step()
            if i % ((train_task_nums // self.dec_batch_size) // self.scheduler_num) == 0:
                if i > 50000:
                    main_scheduler.step()
                print(meta_task_num, " meta tasks for training")
                print("docode params", " lr:", self.main_optimizer.param_groups[0]['lr'])
                print("embedding params", " lr:", self.main_optimizer.param_groups[1]['lr'])
            self.model.normalize_basis_matrix()
        p.terminate()
        self.logger.logging(self.model, meta_task_num, self.main_optimizer.param_groups[0]['lr'])
        self.logger.close_all_file()
