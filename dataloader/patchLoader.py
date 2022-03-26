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

rate = 1
np.random.seed(39)
ind_list = []


class PatchTrainDataset(Dataset):
    def __init__(self, img_dir, sort_ind_file_name=None, file_name=None, file_feats=None):
        self.img = []
        self.label_img_map = []
        folders = os.listdir(img_dir)
        folders = sorted(folders)

        self.file_index_map, index = {}, 0
        for folder in folders:
            file_list = glob(img_dir + '/' + folder + '/*', )
            ind = np.random.permutation(int(len(file_list)))
            ind_list.append(ind)
            file_list = np.array(file_list)
            file_list = file_list[ind[:int(len(file_list) * rate)]]
            img_list = []
            for file in file_list:
                self.img.append((file, int(folder)))
                self.file_index_map[file] = index
                index += 1
                img_list.append(file)
            self.label_img_map.append(img_list)

        self.sort_ind = np.load(sort_ind_file_name) if sort_ind_file_name is not None else sort_ind_file_name
        self.file_name = np.load(file_name) if file_name is not None else file_name
        self.transforms = T.Compose([
            T.RandomRotation(15),
            T.RandomAffine(15),
            T.RandomHorizontalFlip(),
            # T.RandomGaussianBlur(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    # def __getitem__(self, index):
    #     size = 256
    #     filename, label = self.img[index]
    #
    #     img = Image.open(filename)
    #     num = self.sort_ind[index, random.randint(-10, -1)]
    #     x = (2200 // 16) * (num % 16) + np.random.randint(low=-size // 2, high=size // 2)
    #     y = (2200 // 16) * (num // 16) + np.random.randint(low=-size // 2, high=size // 2)
    #     img = img.crop((x - size // 2, y - size // 2, x + size // 2, y + size // 2))
    #
    #     img = self.transforms(img)
    #
    #     return img, label

    def __getitem__(self, index):
        size = 256
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
        start_loc = 0
        if self.sort_ind is None:
            x = np.random.randint(start_loc, high=start_loc + 2200 - size)
            y = np.random.randint(start_loc, high=start_loc + 2200 - size)
            img_anc = img_anc.crop((x, y, x + size, y + size))

            x = np.random.randint(start_loc, high=start_loc + 2200 - size)
            y = np.random.randint(start_loc, high=start_loc + 2200 - size)
            img_pos = img_pos.crop((x, y, x + size, y + size))

            x = np.random.randint(start_loc, high=start_loc + 2200 - size)
            y = np.random.randint(start_loc, high=start_loc + 2200 - size)
            img_neg = img_neg.crop((x, y, x + size, y + size))
        else:
            num = self.sort_ind[index, random.randint(-10, -1)]
            x = (2200 // 16) * (num % 16) + np.random.randint(low=-size // 2, high=size // 2)
            y = (2200 // 16) * (num // 16) + np.random.randint(low=-size // 2, high=size // 2)
            img_anc = img_anc.crop((x, y, x + size, y + size))

            index_pos = self.file_index_map[filename_pos]
            num = self.sort_ind[index_pos, random.randint(-10, -1)]
            x = (2200 // 16) * (num % 16) + np.random.randint(low=-size // 2, high=size // 2)
            y = (2200 // 16) * (num // 16) + np.random.randint(low=-size // 2, high=size // 2)
            img_pos = img_pos.crop((x, y, x + size, y + size))

            index_neg = self.file_index_map[filename_neg]
            num = self.sort_ind[index_neg, random.randint(-10, -1)]
            x = (2200 // 16) * (num % 16) + np.random.randint(low=-size // 2, high=size // 2)
            y = (2200 // 16) * (num // 16) + np.random.randint(low=-size // 2, high=size // 2)
            img_neg = img_neg.crop((x, y, x + size, y + size))

        img_anc = self.transforms(img_anc)
        img_pos = self.transforms(img_pos)
        img_neg = self.transforms(img_neg)

        img_list = torch.stack([img_anc, img_pos, img_neg])
        label_list = torch.tensor([label_anc, label_pos, label_neg])
        return img_list, label_list

    def __len__(self):
        return len(self.img)


class PatchTestDataset(Dataset):
    def __init__(self, img_dir, sort_ind_file_name=None, file_name=None, file_feats=None, mode='test'):
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

        self.sort_ind = np.load(sort_ind_file_name) if sort_ind_file_name is not None else sort_ind_file_name
        self.file_name = np.load(file_name) if file_name is not None else file_name
        self.file_feats = np.load(file_feats) if file_feats is not None else file_feats
        self.transforms = T.Compose([
            # T.Resize((config.img_height, config.img_weight)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        Img = []
        filename, label = self.img[index]
        img = Image.open(filename)
        size = 256
        start_loc = 0
        for i in range(0, 1):
            if self.sort_ind is None:
                x = np.random.randint(start_loc, high=start_loc + 2200 - size)
                y = np.random.randint(start_loc, high=start_loc + 2200 - size)
                cropImg = img.crop((x, y, x + size, y + size))
                # img.show()
                # cropImg.show()
            else:
                num = self.sort_ind[index, -(i + 1)]
                x = (2200 // 16) * (num % 16)
                y = (2200 // 16) * (num // 16)
                cropImg = img.crop((x + 2200 // 16 // 2 - size // 2, y + 2200 // 16 // 2 - size // 2, x + 2200 // 16 // 2 + size // 2, y + 2200 // 16 // 2 + size // 2))
                # img.show()
                # cropImg.show()
            I = self.transforms(cropImg)
            Img.append(I)
        return Img, label, index

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
