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

label_map = {
    "Nuclear membrane": 0,
    "Cytoplasm": 1,
    "Vesicles": 2,
    "Mitochondria": 3,
    "Golgi Apparatus": 4,
    "Nucleoli": 0,
    "Plasma Membrane": 1,
    "Nucleoplasm": 0,
    "Endoplasmic Reticulum": 5
}
num_class = 6
TISSUE_DIR = '/data/users/liuziyi/PyProgram/ImPloc-revision/data/enhanced_4tissue_piclist/'
IMG_DIR = '/data/users/liuziyi/PyProgram/ImPloc-revision/data/enhanced_4tissue_imgs/'


def load_label_from_file(fname):
    d = {}
    pardir = os.path.join(os.path.dirname(__file__), os.pardir)
    label_file = os.path.join(pardir, "label", fname)
    with open(label_file, 'r') as f:
        for line in f.readlines():
            gene, label = line.strip("\n").split("\t")
            labels = [label_map[x] for x in label.split(",") if x]
            one_hot = np.zeros(num_class)
            if labels:
                for i in labels:
                    one_hot[i] = 1
                d[gene] = array2hash(one_hot)
    return d


def get_enhanced_gene_list():
    DATA_DIR = '/data/users/liuziyi/PyProgram/ImPloc-revision/data/enhanced_4tissue_imgs'
    '''some gene marked as enhanced but do not have enhanced label'''
    return [x for x in os.listdir(DATA_DIR)
            if len(os.listdir(os.path.join(DATA_DIR, x)))]


def array2hash(numpy_array):
    hash_num = 0
    for i in numpy_array:
        hash_num = i + hash_num * 2
    return int(hash_num)


def hash2array(hash_num):
    numpy_array = []
    for i in range(num_class):
        numpy_array.insert(0, hash_num % 2)
        hash_num = hash_num // 2
    return np.array(numpy_array)


class ImplocMergeTrainDataset(Dataset):
    def __init__(self, img_dir='', srate=0.9):
        self.img = []

        all_genes = get_enhanced_gene_list()
        train_genes = all_genes[:int(len(all_genes) * srate)]
        gene_label_dict = load_label_from_file('/data/users/liuziyi/PyProgram/ImPloc-revision/label/enhanced_label.txt')

        self.gene_pics_dict, self.label_gene_dict = {}, {}
        max_gene_cnt = 0
        for gene in train_genes:
            if gene not in gene_label_dict:
                continue
            pics = []
            for t in ['liver', 'breast', 'prostate', 'bladder']:
                tp = os.path.join(TISSUE_DIR, t, "%s.txt" % gene)
                if os.path.exists(tp):
                    with open(tp, 'r') as f:
                        pics.extend([l.strip("\n") for l in f.readlines()])
            if len(pics) != 0:
                self.gene_pics_dict[gene] = pics
                label = gene_label_dict[gene]
                if label not in self.label_gene_dict:
                    self.label_gene_dict[label] = []
                self.label_gene_dict[label].append(gene)
                if len(self.label_gene_dict[label]) > max_gene_cnt:
                    max_gene_cnt = len(self.label_gene_dict[label])

        label_cnt = np.zeros(num_class)
        for gene in self.gene_pics_dict:
            for pic in self.gene_pics_dict[gene]:
                self.img.append((pic, gene, gene_label_dict[gene]))
                label_cnt += hash2array(gene_label_dict[gene])

        self.transforms_patch = T.Compose([
            # T.Resize((config.img_height, config.img_weight)),
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
            T.Resize((512, 512)),
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
        filename, gene, label = self.img[index]
        gene_pos, label_pos = choice(self.label_gene_dict[label]), label
        filename_pos = choice(self.gene_pics_dict[gene_pos])
        while True:
            label_neg = choice(list(self.label_gene_dict.keys()))
            if label != label_neg:
                gene_neg = choice(self.label_gene_dict[label_neg])
                filename_neg = choice(self.gene_pics_dict[gene_neg])
                break

        img = Image.open(IMG_DIR + gene + '/' + filename)
        img_resample = self.transforms_resample(img)
        x = np.random.randint(400, high=2600 - size)
        y = np.random.randint(400, high=2600 - size)
        img_crop = img.crop((x, y, x + size, y + size))
        img_patch = self.transforms_patch(img_crop)

        # img_pos = Image.open(IMG_DIR + gene_pos + '/' + filename_pos)
        # img_resample_pos = self.transforms_resample(img_pos)
        # x = np.random.randint(400, high=2600 - size)
        # y = np.random.randint(400, high=2600 - size)
        # img_crop_pos = img_pos.crop((x, y, x + size, y + size))
        # img_patch_pos = self.transforms_patch(img_crop_pos)
        #
        # img_neg = Image.open(IMG_DIR + gene_neg + '/' + filename_neg)
        # img_resample_neg = self.transforms_resample(img_neg)
        # x = np.random.randint(400, high=2600 - size)
        # y = np.random.randint(400, high=2600 - size)
        # img_crop_neg = img_neg.crop((x, y, x + size, y + size))
        # img_patch_neg = self.transforms_patch(img_crop_neg)

        return img_resample, img_patch, hash2array(label)

    def __len__(self):
        return len(self.img)


class ImplocMergeTestDataset(Dataset):
    def __init__(self, img_dir='', srate=0.9):
        self.img = []

        all_genes = get_enhanced_gene_list()
        test_genes = all_genes[int(len(all_genes) * srate):]
        gene_label_dict = load_label_from_file('/data/users/liuziyi/PyProgram/ImPloc-revision/label/enhanced_label.txt')

        self.gene_pics_dict, self.label_gene_dict = {}, {}
        for gene in test_genes:
            if gene not in gene_label_dict:
                continue
            pics = []
            for t in ['liver', 'breast', 'prostate', 'bladder']:
                tp = os.path.join(TISSUE_DIR, t, "%s.txt" % gene)
                if os.path.exists(tp):
                    with open(tp, 'r') as f:
                        pics.extend([l.strip("\n") for l in f.readlines()])
            if len(pics) != 0:
                self.gene_pics_dict[gene] = pics
                label = gene_label_dict[gene]
                if label not in self.label_gene_dict:
                    self.label_gene_dict[label] = []
                self.label_gene_dict[label].append(gene)

        for gene in self.gene_pics_dict:
            for pic in self.gene_pics_dict[gene]:
                self.img.append((pic, gene, gene_label_dict[gene]))

        self.transforms_patch = T.Compose([
            # T.Resize((config.img_height, config.img_weight)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        self.transforms_resample = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        ImgResample, ImgPatch = [], []
        filename, gene, label = self.img[index]
        img = Image.open(IMG_DIR + gene + '/' + filename)
        img_resample = self.transforms_resample(img)

        size = 256
        for i in range(0, 10):
            x = np.random.randint(400, high=2600 - size)
            y = np.random.randint(400, high=2600 - size)
            cropImg = img.crop((x, y, x + size, y + size))
            I = self.transforms_patch(cropImg)
            ImgPatch.append(I)
            ImgResample.append(img_resample)
        return gene, ImgResample, ImgPatch, hash2array(label)

    def __len__(self):
        return len(self.img)


class TestDataset(Dataset):
    def __init__(self, X1_list):
        img = []
        for index, row in X1_list.iterrows():
            img.append((row["filename"], row["label"]))
            self.img = img
        self.transforms = T.Compose([
            T.Resize((config.img_height, config.img_weight)),
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
        labels.append(str(file.split("/")[-2]))
        all_files = pd.DataFrame({"filename": all_data_path, "label": labels})

    return all_files
