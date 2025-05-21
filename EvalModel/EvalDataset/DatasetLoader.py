import os.path

import pandas as pd
import numpy as np
import torch
import scipy.interpolate as interpolate
import os

project_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def convert_np2tensor(device, index_np, weight_np):
    support_x = torch.tensor(index_np, device=device).view(-1, 1)
    support_y = torch.tensor(weight_np, device=device).view(-1)
    if torch.is_floating_point(support_y):
        support_y = support_y.round().type(torch.int32)
    query_x = support_x.clone()
    query_y = support_y.clone().view(-1, 1)
    return support_x, support_y, query_x, query_y


def load_synthetic_0_5_generator(device, chunksize=10000000):
    dataset_path = os.path.join(project_path, "EvalModel/EvalDataset/Datasets/synthetic/0.5_zipf/0.npz")
    npz_file = np.load(dataset_path)
    yield convert_np2tensor(device, np.arange(npz_file['support_y'].shape[0]), npz_file['support_y'])


def load_synthetic_0_7_generator(device, chunksize=10000000):
    dataset_path = os.path.join(project_path, "EvalModel/EvalDataset/Datasets/synthetic/0.7_zipf/0.npz")
    npz_file = np.load(dataset_path)
    yield convert_np2tensor(device, np.arange(npz_file['support_y'].shape[0]), npz_file['support_y'])


def load_synthetic_0_9_generator(device, chunksize=10000000):
    dataset_path = os.path.join(project_path, "EvalModel/EvalDataset/Datasets/synthetic/0.9_zipf/0.npz")
    npz_file = np.load(dataset_path)
    yield convert_np2tensor(device, np.arange(npz_file['support_y'].shape[0]), npz_file['support_y'])


def load_synthetic_1_1_generator(device, chunksize=10000000):
    dataset_path = os.path.join(project_path, "EvalModel/EvalDataset/Datasets/synthetic/1.1_zipf/0.npz")
    npz_file = np.load(dataset_path)
    yield convert_np2tensor(device, np.arange(npz_file['support_y'].shape[0]), npz_file['support_y'])


def load_synthetic_1_3_generator(device, chunksize=10000000):
    dataset_path = os.path.join(project_path, "EvalModel/EvalDataset/Datasets/synthetic/1.3_zipf/0.npz")
    npz_file = np.load(dataset_path)
    yield convert_np2tensor(device, np.arange(npz_file['support_y'].shape[0]), npz_file['support_y'])


def load_synthetic_1_5_generator(device, chunksize=10000000):
    dataset_path = os.path.join(project_path, "EvalModel/EvalDataset/Datasets/synthetic/1.5_zipf/0.npz")
    npz_file = np.load(dataset_path)
    yield convert_np2tensor(device, np.arange(npz_file['support_y'].shape[0]), npz_file['support_y'])


dataset_dic = {
    "synthetic_0_5": load_synthetic_0_5_generator,
    "synthetic_0_7": load_synthetic_0_7_generator,
    "synthetic_0_9": load_synthetic_0_9_generator,
    "synthetic_1_1": load_synthetic_1_1_generator,
    "synthetic_1_3": load_synthetic_1_3_generator,
    "synthetic_1_5": load_synthetic_1_5_generator,
}
