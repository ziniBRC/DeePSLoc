from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
import numpy as np
import torch
from tqdm import tqdm
from itertools import chain
from glob import glob
import os
from config import *
import random
from random import choice
import pandas as pd
import cv2
import matplotlib.pyplot as plt

rate = 0.8
np.random.seed(39)
ind_list = []


def array2hash(numpy_array):
    hash_num = 0
    for i in numpy_array:
        hash_num = i + hash_num * 2
    return int(hash_num)


def hash2array(hash_num, num_class=7):
    numpy_array = []
    for i in range(num_class):
        numpy_array.insert(0, hash_num % 2)
        hash_num = hash_num // 2
    return np.array(numpy_array)


class PatchTrainDataset(Dataset):
    def __init__(self, img_dir, num_class=7, sort_ind_file_name=None, file_name=None, patch=True):
        self.img = []
        self.label_img_map = {}
        self.patch = patch
        folders = os.listdir(img_dir)
        folders = sorted(folders)

        self.file_index_map, index = {}, 0
        for folder in folders:
            label = list(map(int, folder.split("_")))
            one_hot = np.zeros(num_class)
            for l in label:
                one_hot[l] = 1
            label_hash = array2hash(one_hot)
                
            file_list = glob(img_dir + '/' + folder + '/*', )
            ind = np.random.permutation(int(len(file_list)))
            ind_list.append(ind)
            file_list = np.array(file_list)
            file_list = file_list[ind[:int(len(file_list) * rate)]]
            img_list = []
            for file in file_list:
                self.img.append((file, one_hot))
                self.file_index_map[file] = index
                index += 1
                img_list.append(file)
            # self.label_img_map.append(img_list)
            self.label_img_map[label_hash] = img_list

        self.sort_ind = np.load(sort_ind_file_name) if sort_ind_file_name is not None else sort_ind_file_name
        if self.patch is True:
            self.transforms = T.Compose([
                T.RandomRotation(15),
                T.RandomAffine(15),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((512, 512)),
                T.RandomRotation(15),
                T.RandomAffine(15),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        size = 256
        filename, label_anc = self.img[index]
        filename_pos, label_pos = choice(self.label_img_map[array2hash(label_anc)]), label_anc
        while True:
            label_neg = choice(list(self.label_img_map.keys()))
            if array2hash(label_anc) != label_neg:
                filename_neg = choice(self.label_img_map[label_neg])
                label_neg = hash2array(label_neg)
                break

        img_anc = Image.open(filename)
        img_pos = Image.open(filename_pos)
        img_neg = Image.open(filename_neg)
        start_loc = 0 if img_anc.size[0] == 2200 else 400
        if self.patch is True:
            if self.sort_ind is None:
                x = np.random.randint(start_loc, high=start_loc + 2200 - size)
                y = np.random.randint(start_loc, high=start_loc + 2200 - size)
                img_anc = img_anc.crop((x, y, x + size, y + size))
            else:
                num = self.sort_ind[index, random.randint(-10, -1)]
                x = (2200 // 16) * (num % 16) + np.random.randint(low=-size // 2, high=size // 2)
                y = (2200 // 16) * (num // 16) + np.random.randint(low=-size // 2, high=size // 2)
                img_anc = img_anc.crop((x - size // 2, y - size // 2, x + size // 2, y + size // 2))
        img_anc = self.transforms(img_anc)

        if self.patch is True:
            if self.sort_ind is None:
                x = np.random.randint(start_loc, high=start_loc + 2200 - size)
                y = np.random.randint(start_loc, high=start_loc + 2200 - size)
                img_pos = img_pos.crop((x, y, x + size, y + size))
            else:
                num = self.sort_ind[index, random.randint(-10, -1)]
                x = (2200 // 16) * (num % 16) + np.random.randint(low=-size // 2, high=size // 2)
                y = (2200 // 16) * (num // 16) + np.random.randint(low=-size // 2, high=size // 2)
                img_pos = img_pos.crop((x - size // 2, y - size // 2, x + size // 2, y + size // 2))
        img_pos = self.transforms(img_pos)

        if self.patch is True:
            if self.sort_ind is None:
                x = np.random.randint(start_loc, high=start_loc + 2200 - size)
                y = np.random.randint(start_loc, high=start_loc + 2200 - size)
                img_neg = img_neg.crop((x, y, x + size, y + size))
            else:
                num = self.sort_ind[index, random.randint(-10, -1)]
                x = (2200 // 16) * (num % 16) + np.random.randint(low=-size // 2, high=size // 2)
                y = (2200 // 16) * (num // 16) + np.random.randint(low=-size // 2, high=size // 2)
                img_neg = img_neg.crop((x - size // 2, y - size // 2, x + size // 2, y + size // 2))
        img_neg = self.transforms(img_neg)

        img_list = torch.stack([img_anc, img_pos, img_neg])
        label_list = torch.tensor([label_anc, label_pos, label_neg])
        return img_list, label_list
        # return img_anc, label_anc

    def __len__(self):
        return len(self.img)


class PatchTestDataset(Dataset):
    def __init__(self, img_dir, num_class=7, sort_ind_file_name=None, file_name=None, patch=True, mode='test'):
        self.img = []
        self.label_img_map = []
        folders = os.listdir(img_dir)
        folders = sorted(folders)

        self.file_index_map, index = {}, 0
        for folder in folders:
            label = list(map(int, folder.split("_")))
            one_hot = np.zeros(num_class)
            for l in label:
                one_hot[l] = 1

            file_list = glob(img_dir + '/' + folder + '/*', )
            if mode == 'val':
                ind = ind_list[int(folder)]
                file_list = np.array(file_list)
                file_list = file_list[ind[int(len(file_list) * rate):]]
            img_list = []
            for file in file_list:
                self.img.append((file, one_hot))
                self.file_index_map[file] = index
                index += 1
                img_list.append(file)
            self.label_img_map.append(img_list)

        self.sort_ind = np.load(sort_ind_file_name) if sort_ind_file_name is not None else sort_ind_file_name
        self.patch = patch
        if self.patch:
            self.transforms = T.Compose([
                # T.Resize((config.img_height, config.img_weight)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((512, 512)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        Img = []
        filename, label = self.img[index]
        img = Image.open(filename)
        start_loc = 0 if img.size[0] == 2200 else 400
        if self.patch:
            size = 256
            for i in range(0, 10):
                if self.sort_ind is None:
                    x = np.random.randint(400, high=2600 - size)
                    y = np.random.randint(400, high=2600 - size)
                    cropImg = img.crop((x, y, x + size, y + size))
                else:
                    num = self.sort_ind[index, -i]
                    x = (2200 // 16) * (num % 16)
                    y = (2200 // 16) * (num // 16)
                    cropImg = img.crop((x - size // 2, y - size // 2, x + size // 2, y + size // 2))
                I = self.transforms(cropImg)
                Img.append(I)
        else:
            I = self.transforms(img)
            Img.append(I)
        return Img, label

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
