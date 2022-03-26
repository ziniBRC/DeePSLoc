from torch.utils.data import DataLoader
from dataloader.resampleLoader import *
from dataloader.patchLoader import *
from dataloader.mergeLoader import *
import numpy as np
import pandas as pd
from config import *


def make_resample_data_loader(args, sample_num=8, **kwargs):
    if args.dataset =='IHC':
        train_data_dir = 'data/train/cropIHC'
        test_data_dir = 'data/test/cropIHC'
        num_class = len(os.listdir(train_data_dir))
        train_loader = DataLoader(ResampleTrainDataset(train_data_dir, sample_num), batch_size=args.batch_size, drop_last=False, shuffle=True, **kwargs)
        val_loader = DataLoader(ResampleTestDataset(train_data_dir, mode='val'), batch_size=args.batch_size, drop_last=False, shuffle=False, **kwargs)
        test_loader = DataLoader(ResampleTestDataset(test_data_dir), batch_size=args.batch_size, shuffle=True, **kwargs)
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset =='Alta':
        train_data_dir = 'data/train/cropAlta'
        test_data_dir = 'data/test/cropAlta'
        num_class = len(os.listdir(train_data_dir))

        train_loader = DataLoader(ResampleTrainDataset(train_data_dir, sample_num), batch_size=args.batch_size, drop_last=False, shuffle=True, **kwargs)
        val_loader = DataLoader(ResampleTestDataset(train_data_dir, mode='val'), batch_size=args.batch_size, drop_last=False, shuffle=False, **kwargs)
        test_loader = DataLoader(ResampleTestDataset(test_data_dir), batch_size=args.batch_size, shuffle=True, **kwargs)
        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError


def make_merge_data_loader(args, **kwargs):
    if args.dataset =='IHC':
        train_data_dir = 'data/train/cropIHC'
        test_data_dir = 'data/test/cropIHC'
        num_class = config.ihc_classes
        train_loader = DataLoader(MergeTrainDataset(train_data_dir), batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
        val_loader = DataLoader(MergeTestDataset(train_data_dir, mode='val'), batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(MergeTestDataset(test_data_dir), batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, test_loader, num_class

    elif args.dataset =='Alta':
        train_data_dir = 'data/train/cropAlta'
        test_data_dir = 'data/test/cropAlta'
        num_class = config.alta_classes

        train_loader = DataLoader(MergeTrainDataset(train_data_dir), batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
        val_loader = DataLoader(MergeTestDataset(train_data_dir, mode='val'), batch_size=args.batch_size, drop_last=False, shuffle=False, **kwargs)
        test_loader = DataLoader(MergeTestDataset(test_data_dir), batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError


def make_patch_data_loader(args, **kwargs):
    if args.dataset =='IHC':
        train_data_dir = 'data/train/cropIHC'
        test_data_dir = 'data/test/cropIHC'
        # train_sort_ind_path = 'dataset/train/IHC/sort_index_random.npy'
        # test_sort_ind_path = 'dataset/test/IHC/sort_index_random.npy'
        # train_feats = 'dataset/train/IHC/resnet_feats_random.npy'
        # test_feats = 'dataset/test/IHC/resnet_feats_random.npy'
        train_sort_ind_path, test_sort_ind_path = None, None
        train_feats, test_feats = None, None
        num_class = len(os.listdir(train_data_dir))
        train_loader = DataLoader(PatchTrainDataset(train_data_dir, train_sort_ind_path, file_feats=train_feats), batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(PatchTestDataset(train_data_dir, train_sort_ind_path, file_feats=train_feats, mode='val'), batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(PatchTestDataset(test_data_dir, test_sort_ind_path, file_feats=test_feats), batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset =='Alta':
        train_data_dir = 'data/train/cropAlta'
        test_data_dir = 'data/test/cropAlta'
        train_sort_ind_path = 'dataset/train/Alta/sort_index.npy'
        test_sort_ind_path = 'dataset/test/Alta/sort_index.npy'
        train_file_name = 'dataset/train/Alta/filename.npy'
        test_file_name = 'dataset/test/Alta/filename.npy'
        num_class = len(os.listdir(train_data_dir))

        # train_loader = DataLoader(PatchTrainDataset(train_data_dir), batch_size=args.batch_size, shuffle=True, **kwargs)
        # test_loader = DataLoader(PatchTestDataset(test_data_dir), batch_size=args.batch_size, shuffle=False, **kwargs)
        train_loader = DataLoader(PatchTrainDataset(train_data_dir, train_sort_ind_path, train_file_name), batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(PatchTestDataset(train_data_dir, train_sort_ind_path, mode='val'), batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(PatchTestDataset(test_data_dir, test_sort_ind_path, test_file_name), batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError


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
    labels = labels.expand((N, N))

    mask = labels - labels.T
    mask = mask == 0
    # mask = torch.where(mask == 0, True, False)
    return mask


def sortPatch(act_map):
    act_map = act_map.reshape(act_map.shape[0], act_map.shape[1] * act_map.shape[2])
    ind_sort = np.argsort(act_map, axis=1)
    return ind_sort


def save_ind(resnet_feats, ):
    # np.save('dataset/train/Alta/resample_pred.npy', np.array(pred_prob))
    resnet_feats = np.mean(np.array(resnet_feats), axis=1)
    np.save('dataset/train/Alta/resnet_feats.npy', resnet_feats)
    sort_ind = sortPatch(resnet_feats)
    np.save('dataset/train/Alta/sort_index.npy', sort_ind)
    size = 256
    for index, filename in enumerate(file_list):
        img = Image.open(filename)
        num = sort_ind[index, -1]
        x = (2200 // 16) * (num % 16)
        y = (2200 // 16) * (num // 16)
        cropImg = img.crop((x - size // 2, y - size // 2, x + size // 2, y + size // 2))
        plt.imshow(np.array(cropImg))
        plt.show()
    pass