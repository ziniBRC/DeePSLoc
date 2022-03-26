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


class MultiResampleTrainDataset(Dataset):
    def __init__(self, img_dir, num_class):
        self.img = []
        self.label_img_map = []
        self.label_one_hot_map = []
        folders = os.listdir(img_dir)
        folders = sorted(folders)

        ind = 0
        for folder in folders:
            file_list = glob(img_dir + '/' + folder + '/*', )
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

        self.transforms = T.Compose([
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
        filename, label = self.img[index]
        filename_pos, label_pos = choice(self.label_img_map[label]), label
        while True:
            label_neg = random.randint(0, len(self.label_img_map)-1)
            if label != label_neg:
                filename_neg = choice(self.label_img_map[label_neg])
                break
        label = self.label_one_hot_map[label]
        label_pos = self.label_one_hot_map[label_pos]
        label_neg = self.label_one_hot_map[label_neg]

        img = Image.open(filename)
        img_pos = Image.open(filename_pos)
        img_neg = Image.open(filename_neg)

        img = self.transforms(img)
        img_pos = self.transforms(img_pos)
        img_neg = self.transforms(img_neg)

        return img, label, img_pos, label_pos, img_neg, label_neg

    def __len__(self):
        return len(self.img)


class MultiResampleTestDataset(Dataset):
    def __init__(self, img_dir, num_class):
        self.img = []
        self.label_img_map = []
        self.label_one_hot_map = []
        folders = os.listdir(img_dir)
        folders = sorted(folders)

        ind = 0
        for folder in folders:
            file_list = glob(img_dir + '/' + folder + '/*', )
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

        self.transforms = T.Compose([
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        Img = []
        filename, label = self.img[index]
        label = self.label_one_hot_map[label]

        img = Image.open(filename)
        img_resample = self.transforms(img)
        for i in range(0, 10):
            Img.append(img_resample)
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
