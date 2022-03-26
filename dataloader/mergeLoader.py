import numpy as np
import random
from random import choice
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
from tqdm import tqdm
from itertools import chain
from glob import glob
import os
import pandas as pd
from config import *
import matplotlib.pyplot as plt

rate = 0.8
np.random.seed(39)
ind_list = []


class MergeTrainDataset(Dataset):
    def __init__(self, img_dir=''):
        resample_size = 512
        self.img = []
        self.label_img_map = []
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

        self.transforms_resample = T.Compose([
            T.Resize(resample_size),
            # T.RandomCrop((config.img_height, config.img_weight)),
            T.RandomRotation(15),
            T.RandomAffine(15),
            T.RandomHorizontalFlip(),
            # T.RandomGaussianBlur(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        self.transforms_patch = T.Compose([
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

        img = Image.open(filename)
        x = np.random.randint(low=0, high=2200 - size)
        y = np.random.randint(low=0, high=2200 - size)
        crop_img = img.crop((x, y, x + size, y + size))

        img_resample = self.transforms_resample(img)
        img_patch = self.transforms_patch(crop_img)

        return img_resample, img_patch, label

    def __len__(self):
        return len(self.img)


class MergeTestDataset(Dataset):
    def __init__(self, img_dir='', mode='test'):
        size = (512, 512)
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

        self.transforms_resample = T.Compose([
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        self.transforms_patch = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        ImgResample, ImgPatch = [], []
        filename, label = self.img[index]
        img = Image.open(filename)
        size = 256

        img_resample = self.transforms_resample(img)
        for i in range(0, 10):
            x = np.random.randint(low=0, high=2200 - size)
            y = np.random.randint(low=0, high=2200 - size)
            cropImg = img.crop((x, y, x + size, y + size))

            img_patch = self.transforms_patch(cropImg)
            ImgResample.append(img_resample)
            ImgPatch.append(img_patch)

        return ImgResample, ImgPatch, label, filename

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
        labels.append(str(file.split("/")[-2]))
        all_files = pd.DataFrame({"filename": all_data_path, "label": labels})

    return all_files
