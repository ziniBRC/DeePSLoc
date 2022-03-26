from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from itertools import chain
from glob import glob
import os
from config import *
import random
from random import choice
import pandas as pd
import torch

size = (512, 512)
rate = 1
np.random.seed(39)
ind_list = []


class ResampleTrainDataset(Dataset):
    def __init__(self, img_dir, sample_num=16):
        self.img = []
        self.label_img_map = []
        self.sample_num = sample_num
        folders = os.listdir(img_dir)
        folders = sorted(folders)

        for folder in folders:
            file_list = glob(img_dir + '/' + folder + '/*', )
            ind = np.random.permutation(int(len(file_list)))
            ind_list.append(ind)
            file_list = np.array(file_list)
            file_list = file_list[ind[:int(len(file_list) * rate)]]
            img_list = []
            for file in file_list:
                self.img.append((file, int(folder)))
                img_list.append(file)
            self.label_img_map.append(img_list)

        self.transforms = T.Compose([
            T.Resize(size),
            # T.RandomCrop((config.img_height, config.img_weight)),
            T.RandomRotation(15),
            T.RandomAffine(15),
            T.RandomHorizontalFlip(),
            # T.RandomVerticalFlip(),
            # T.GaussianBlur(kernel_size=(1, 1)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    # def __getitem__(self, index):
    #     filename, label = self.img[index]
    #
    #     img = Image.open(filename)
    #     img = self.transforms(img)
    #
    #     return img, label

    def __getitem__(self, index):
        filename, label_anc = self.img[index]
        filename_pos, label_pos = choice(self.label_img_map[label_anc]), label_anc
        while True:
            label_neg = random.randint(0, len(self.label_img_map)-1)
            if label_anc != label_neg:
                filename_neg = choice(self.label_img_map[label_neg])
                break

        img_anc = Image.open(filename)
        img_pos = Image.open(filename_pos)
        img_neg = Image.open(filename_neg)

        img_anc = self.transforms(img_anc)
        img_pos = self.transforms(img_pos)
        img_neg = self.transforms(img_neg)

        img_list = torch.stack([img_anc, img_pos, img_neg])
        label_list = torch.tensor([label_anc, label_pos, label_neg])
        return img_list, label_list
        # return img_anc, label_anc

    def __len__(self):
        return len(self.img)


class ResampleTestDataset(Dataset):
    def __init__(self, img_dir, mode='test'):
        self.img = []
        self.label_img_map = []
        folders = os.listdir(img_dir)
        folders = sorted(folders)
        for folder in folders:
            file_list = glob(img_dir + '/' + folder + '/*', )
            if mode == 'val':
                ind = ind_list[int(folder)]
                file_list = np.array(file_list)
                file_list = file_list[ind[int(len(file_list) * rate):]]
            img_list = []
            for file in file_list:
                self.img.append((file, int(folder)))
                img_list.append(file)
            self.label_img_map.append(img_list)

        self.transforms = T.Compose([
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        filename, label = self.img[index]
        img = Image.open(filename)
        img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.img)


def get_files(root, mode):
    all_data_path, labels = [], []
    image_folders = list(map(lambda x: root + x, os.listdir(root)))
    all_images = list(chain.from_iterable(list(map(lambda x: glob(x + "/*"), image_folders))))
    if mode == "train":
        print("loading train dataset")
    else:
        print("loading test dataset")
    for file in tqdm(all_images):
        all_data_path.append(file)
        labels.append(int(file.split("/")[-2]))
        all_files = pd.DataFrame({"filename": all_data_path, "label": labels})

    return all_files
