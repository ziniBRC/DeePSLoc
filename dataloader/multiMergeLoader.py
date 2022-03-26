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
import cv2

size = (512, 512)
rate = 0.8
np.random.seed(39)
ind_list = []


class MultiMergeTrainDataset(Dataset):
    def __init__(self, img_dir, num_class):
        self.img = []
        self.label_img_map = []
        self.label_one_hot_map = []
        folders = os.listdir(img_dir)
        folders = sorted(folders)

        ind = 0
        for folder in folders:
            file_list = glob(img_dir + '/' + folder + '/*', )
            ind = ind_list[int(folder)]
            file_list = np.array(file_list)
            file_list = file_list[ind[:int(len(file_list) * rate)]]
            img_list = []
            label = list(map(int, folder.split("_")))
            one_hot = np.zeros(num_class)
            for l in label:
                one_hot[l] = 1
            for file in file_list:
                self.img.append((file, ind))
                img_list.append(file)
            ind += 1
            self.label_one_hot_map.append(one_hot)
            self.label_img_map.append(img_list)

        self.transforms_patch = T.Compose([
            # T.Resize(size),
            # T.RandomCrop((config.img_height, config.img_weight)),
            T.RandomRotation(15),
            T.RandomAffine(15),
            T.RandomHorizontalFlip(),
            # T.RandomGaussianBlur(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self.transforms_resample = T.Compose([
            T.Resize(size),
            # T.RandomCrop((config.img_height, config.img_weight)),
            T.RandomRotation(15),
            T.RandomAffine(15),
            T.RandomHorizontalFlip(),
            # T.RandomGaussianBlur(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        size = 256
        filename, label = self.img[index]
        label = self.label_one_hot_map[label]

        img = Image.open(filename)
        img_resample = self.transforms_resample(img)

        x = np.random.randint(0, high=2200 - size)
        y = np.random.randint(0, high=2200 - size)
        cropImg = img.crop((x, y, x + size, y + size))
        img_patch = self.transforms_patch(cropImg)

        return img_resample, img_patch, label

    def __len__(self):
        return len(self.img)


class MultiMergeTestDataset(Dataset):
    def __init__(self, img_dir, num_class, mode='test'):
        self.img = []
        self.label_img_map = []
        self.label_one_hot_map = []
        folders = os.listdir(img_dir)
        folders = sorted(folders)

        ind = 0
        for folder in folders:
            file_list = glob(img_dir + '/' + folder + '/*', )
            if mode == 'val':
                ind = ind_list[int(folder)]
                file_list = np.array(file_list)
                file_list = file_list[ind[int(len(file_list) * rate):]]
            img_list = []
            label = list(map(int, folder.split("_")))
            one_hot = np.zeros(num_class)
            for l in label:
                one_hot[l] = 1
            for file in file_list:
                self.img.append((file, ind))
                img_list.append(file)
            ind += 1
            self.label_one_hot_map.append(one_hot)
            self.label_img_map.append(img_list)

        self.transforms_patch = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self.transforms_resample = T.Compose([
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        size = 256
        ImgResample, ImgPatch = [], []
        filename, label = self.img[index]
        label = self.label_one_hot_map[label]

        img = Image.open(filename)
        img_resample = self.transforms_resample(img)
        for i in range(0, 10):
            x = np.random.randint(0, high=2200 - size)
            y = np.random.randint(0, high=2200 - size)
            cropImg = img.crop((x, y, x + size, y + size))
            I = self.transforms_patch(cropImg)
            ImgPatch.append(I)
            ImgResample.append(img_resample)
        return ImgResample, ImgPatch, label

    def __len__(self):
        return len(self.img)