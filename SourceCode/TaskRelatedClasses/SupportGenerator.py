import torch
import random
import numpy as np


class ZipfSupportGeneratorByIS():
    def __init__(self, zipf_param_upper=0.8, zipf_param_lower=1.3, skew_lower=0.5,
                 skew_upper=10, is_lower=None, is_upper=None, base_weight_sum=None, device="cpu",
                 total_task_num=1000000, sample_IS_scale="linear"):
        super().__init__()
        assert base_weight_sum is None
        assert is_lower is not None and is_upper is not None
        assert is_lower <= is_upper
        self.is_lower = is_lower
        self.is_upper = is_upper
        self.samples_ndarray = None
        self.samples_tensor = None
        self.num_of_generated_task = 0
        self.total_task_num = total_task_num
        self.sample_index = 0
        self.zipf_param_upper = zipf_param_upper
        self.zipf_param_lower = zipf_param_lower
        self.skew_lower = skew_lower
        self.skew_upper = skew_upper
        self.base_weight_sum = base_weight_sum
        self.device = device
        assert sample_IS_scale in ["linear", "log"]
        self.sample_IS_scale = sample_IS_scale
        self.sample_IS_func = self.sample_by_linear_scale if sample_IS_scale == "linear" else self.sample_by_log_scale
        print("sample_IS_scale:", sample_IS_scale)
        print("sample_IS_func:", self.sample_IS_func.__name__)
        self.prepare()

    def transfer_device(self):
        self.samples_tensor = torch.tensor(self.samples_ndarray, device=self.device).float()
        self.shuffle()

    def prepare(self):
        self.samples_ndarray = np.arange(1000000).reshape(-1, 1)
        self.transfer_device()

    def shuffle(self):
        index = torch.randperm(self.samples_tensor.shape[0], device=self.device)
        self.samples_tensor = self.samples_tensor[index]

    def get_zipf_by_is(self, zipf_param, item_size):
        size = item_size
        with torch.no_grad():
            x = torch.arange(size + 1, 1, step=-1, device=self.device).float()
            x = x ** (-zipf_param)
            x = x / x.sum()
            vector = x * (1 / x[0])
            index = torch.randperm(vector.shape[0], device=self.device)
            vector = vector[index]
            return vector

    def sample_by_log_scale(self, upper_ratio):
        uniform_ratio = random.random() * upper_ratio
        item_size = int(((self.is_upper / self.is_lower) ** uniform_ratio) * self.is_lower)
        return item_size

    def sample_by_linear_scale(self, upper_ratio):
        uniform_ratio = random.random() * upper_ratio
        item_size = int(uniform_ratio * (self.is_upper - self.is_lower) + self.is_lower)
        return item_size

    def sample_train_support(self, item_size=None, skew_ratio=None, zipf_param=None):
        assert item_size is None and skew_ratio is None and zipf_param is None
        upper_ratio = 1
        item_size = self.sample_IS_func(upper_ratio)
        skew_ratio = (random.random() * (self.skew_upper - self.skew_lower) + self.skew_lower)
        if zipf_param is None:
            zipf_param = (random.random() * (self.zipf_param_upper - self.zipf_param_lower) + self.zipf_param_lower)
        weights = self.get_zipf_by_is(zipf_param=zipf_param, item_size=item_size) * skew_ratio
        # try to allocate item_size
        if item_size + self.sample_index >= self.samples_tensor.shape[0]:
            self.sample_index = 0
            self.shuffle()
            while item_size + self.sample_index >= self.samples_tensor.shape[0]:
                extend_tensor = self.samples_tensor + 10
                self.samples_tensor = torch.cat([self.samples_tensor, extend_tensor], dim=0)
                print('extend samples_tensor!', "original size", extend_tensor.shape[0],
                      "new size", self.samples_tensor.shape[0])
        samples = self.samples_tensor[self.sample_index:item_size + self.sample_index]
        self.sample_index += item_size
        self.num_of_generated_task += 1
        return samples, weights, torch.ones(1, device=self.device) * zipf_param

    def sample_test_support(self, item_size=None, skew_ratio=None, zipf_param=None):
        assert item_size is not None and skew_ratio is not None and zipf_param is not None
        weights = self.get_zipf_by_is(zipf_param=zipf_param, item_size=item_size) * skew_ratio
        if item_size + self.sample_index >= self.samples_tensor.shape[0]:
            self.sample_index = 0
            self.shuffle()
            while item_size + self.sample_index >= self.samples_tensor.shape[0]:
                extend_tensor = self.samples_tensor + 10
                self.samples_tensor = torch.cat([self.samples_tensor, extend_tensor], dim=0)
                print('extend samples_tensor!', "original size", extend_tensor.shape[0],
                      "new size", self.samples_tensor.shape[0])
        samples = self.samples_tensor[self.sample_index:item_size + self.sample_index]
        self.sample_index += item_size
        return samples, weights, torch.ones(1, device=self.device) * zipf_param
