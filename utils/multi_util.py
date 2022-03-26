from torch.utils.data import DataLoader
from dataloader.multiloader import *
from dataloader.implocLoader import ImplocTrainDataset, ImplocTestDataset
from dataloader.resampleLoader import ResampleTrainDataset, ResampleTestDataset
from dataloader.multiResampleLoader import MultiResampleTrainDataset, MultiResampleTestDataset
from dataloader.implocMergeLoader import ImplocMergeTrainDataset, ImplocMergeTestDataset
from dataloader.multiMergeLoader import MultiMergeTrainDataset, MultiMergeTestDataset
from dataloader.multiLoader_attn import PatchTrainDataset, PatchTestDataset
import numpy as np
import pandas as pd
import torch
from config import *


def make_resample_data_loader(args, **kwargs):
    train_loader = DataLoader(ImplocTrainDataset(patch=False), batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(ImplocTestDataset(patch=False), batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(ImplocTestDataset(patch=False), batch_size=args.batch_size, shuffle=False, **kwargs)
    num_class = 6
    return train_loader, val_loader, test_loader, num_class


def make_merge_data_loader(args, **kwargs):
    if args.dataset == "Multi":
        train_files = config.train_data_Multi
        test_files = config.test_data_Multi
        num_class = config.multi_classes
        train_loader = DataLoader(MultiMergeTrainDataset(train_files, num_class), batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(MultiMergeTestDataset(train_files, num_class, mode='val'), batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(MultiMergeTestDataset(test_files, num_class), batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.dataset == "Imploc":
        train_loader = DataLoader(ImplocMergeTrainDataset(), batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(ImplocTestDataset(mode='val'), batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(ImplocMergeTestDataset(), batch_size=args.batch_size, shuffle=False, **kwargs)
        num_class = 6
    else:
        print('choose correct dataset!')
        exit()
    return train_loader, val_loader, test_loader, num_class


def make_patch_data_loader(args, **kwargs):
    if args.dataset == 'Multi':
        train_dir = 'data/train/cropMulti'
        test_dir = 'data/test/cropMulti'
        train_sort_ind_path = '/dataset/train/Multi/sort_index_random.npy'
        test_sort_ind_path = '/dataset/test/Multi/sort_index_random.npy'
        train_loader = DataLoader(PatchTrainDataset(train_dir, patch=args.patch, sort_ind_file_name=train_sort_ind_path), batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(PatchTestDataset(train_dir, patch=args.patch, sort_ind_file_name=train_sort_ind_path, mode='val'), batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(PatchTestDataset(test_dir, patch=args.patch, sort_ind_file_name=test_sort_ind_path), batch_size=args.batch_size, shuffle=False, **kwargs)
        num_class = 7
    elif args.dataset == 'Imploc':
        train_loader = DataLoader(ImplocTrainDataset(patch=args.patch), batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(ImplocTestDataset(patch=args.patch, mode='val'), batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(ImplocTestDataset(patch=args.patch), batch_size=args.batch_size, shuffle=False, **kwargs)
        num_class = 6
    # return train_loader, val_loader, test_loader, num_class
    return train_loader, val_loader, test_loader, num_class


def tensor2hash(tensor):
    hash_num_list = []
    for on_hot in tensor:
        hash_num = 0
        for i in on_hot:
            hash_num = i + hash_num * 2
        hash_num_list.append(hash_num)
    hash_num_list = torch.stack(hash_num_list)
    return hash_num_list


def hash2tensor(hash_num_list, num_class=6):
    tensor_list = []
    for hash_num in hash_num_list:
        tensor = []
        for i in range(num_class):
            tensor.insert(0, hash_num % 2)
            hash_num = hash_num // 2
        tensor = torch.stack(tensor)
        tensor_list.append(tensor)
    tensor_list = torch.stack(tensor_list)
    return tensor_list


def cal_distance_array(X):
    N = X.shape[0]
    temp = torch.matmul(X, X.T)
    norm_self = torch.diagonal(temp).reshape(N, 1)

    element1 = norm_self.expand((N, N))
    element4 = norm_self.T.expand((N, N))

    results = element1 + element4 - temp - temp.T

    return results


def cal_same_label_mask(labels):
    N = labels.shape[0]
    labels = tensor2hash(labels)
    labels = labels.expand((N, N))

    mask = labels - labels.T
    mask = mask == 0
    # mask = torch.where(mask == 0, True, False)
    return mask