
class SimpleQueryGenerator():
    def __init__(self,device):
        self.device = device

    def generate_train_query(self, support_x, support_y, info):
        support_x = support_x.clone()
        support_y = support_y.clone().reshape(-1, 1)
        return support_x, support_y,info

    def generate_test_query(self, support_x, support_y, info):
        query_x = support_x
        query_y = support_y.reshape(-1, 1)
        return query_x, query_y, info
