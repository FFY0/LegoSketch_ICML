from SourceCode.TaskRelatedClasses.TaskData import MetaTask, SupportSet, QuerySet


def generate_meta_task_once(taskProducer):
    taskProducer.support_generator.flush_tensor()
    support_x, support_y, _ = taskProducer.support_generator.sample_train_support()
    query_x, query_y = taskProducer.query_generator.generate_train_query(support_x, support_y, _)
    return support_x, support_y, query_x, query_y


class TaskProducer():
    def __init__(self, support_generator, query_generator,
                 device, test_task_skew_ratio_list, test_task_group_size, item_size_list,
                 test_zipf_param_list=None, ):
        self.item_size_list = item_size_list
        self.support_generator = support_generator
        self.query_generator = query_generator
        self.device = device
        self.test_task_skew_ratio_list = test_task_skew_ratio_list
        self.test_task_group_size = test_task_group_size
        self.test_zipf_param_list = test_zipf_param_list

    def produce_train_task(self, q, pass_cuda_tensor):
        self.support_generator.transfer_device()
        while True:
            support_x, support_y, support_info = self.support_generator.sample_train_support()
            query_x, query_y, query_info = self.query_generator.generate_train_query(support_x, support_y, support_info)
            if not pass_cuda_tensor:
                support_x = support_x.cpu()
                support_y = support_y.cpu()
                support_info = support_info.cpu()
                query_x = query_x.cpu()
                query_y = query_y.cpu()
            q.put(support_x)
            q.put(support_y)
            q.put(support_info)
            q.put(query_x)
            q.put(query_y)

    def produce_test_task(self, ):
        print('start generating test task...')
        meta_task_group_discribe_list = []
        test_meta_task_group_list = []
        sample_support_func = None
        sample_support_func = self.support_generator.sample_test_support
        # get the most base support generator type
        base_support_generator = self.support_generator
        if base_support_generator.__class__.__name__ == 'ZipfSupportGeneratorByIS':
            for zipf_param in self.test_zipf_param_list:
                print('zip_param' + str(zipf_param))
                for item_size in self.item_size_list:
                    task_list = []
                    stream_length_list = []
                    item_size_list = []
                    for i in range(self.test_task_group_size):
                        test_support_x, test_support_y, support_info = sample_support_func(item_size=item_size,
                                                                                           skew_ratio=1,
                                                                                           zipf_param=zipf_param)
                        test_query_x, test_query_y, query_info = self.query_generator.generate_test_query(
                            test_support_x,
                            test_support_y, support_info)
                        test_support_set = SupportSet(test_support_x.cpu(),
                                                      test_support_y.cpu(), self.device, support_info)
                        test_query_set = QuerySet(test_query_x.cpu(),
                                                  test_query_y.cpu(), self.device, query_info)
                        test_meta_task = MetaTask(test_support_set, test_query_set)
                        task_list.append(test_meta_task)
                        stream_length_list.append(test_support_set.support_y.sum().item())
                        item_size_list.append(test_support_set.support_y.shape[0])
                    meta_task_group_discribe_list.append(
                        str(zipf_param) + '_zipf_' + str(int(sum(item_size_list) / len(item_size_list))) + '_IS_' + str(
                            int(sum(stream_length_list) / len(stream_length_list))) + '_SL_')
                    test_meta_task_group_list.append(task_list)
                self.support_generator.decorate_test = False
        print('end generating test task...')
        return test_meta_task_group_list, meta_task_group_discribe_list
