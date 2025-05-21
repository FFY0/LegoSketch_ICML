from SourceCode.TaskRelatedClasses.TaskData import MetaTask, SupportSet, QuerySet

class TaskConsumer():
    def __init__(self, device):
        self.device = device

    # get a meta task
    def consume_train_task(self, q, pass_cuda_tensor):
        support_x_tensor = q.get()
        support_y_tensor = q.get()
        support_info = q.get()
        query_x_tensor = q.get()
        query_y_tensor = q.get()
        # query_info = q.get()

        meta_task = MetaTask(SupportSet(support_x_tensor, support_y_tensor, self.device,support_info),
                             QuerySet(query_x_tensor, query_y_tensor, self.device,None))
        if not pass_cuda_tensor:
            meta_task.to_device()
        return meta_task

    # release shared memory
    def del_meta_task(self, meta_task):
        del meta_task.support_set.support_x
        del meta_task.support_set.support_y
        del meta_task.support_set.info
        del meta_task.query_set.query_y
        del meta_task.query_set.query_x

