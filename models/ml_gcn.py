import torchvision.models as models
from torch.nn import Parameter
from glob import glob
import torch
import math
import torch.nn as nn
import numpy as np
import os
import pickle


def gen_A(num_classes, t, adj_file):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    for i in range(_adj.shape[0]):
        _adj[i, i] = 0
    # _adj = _adj / _nums
    _adj = _adj / np.sum(_adj, axis=1, keepdims=True)
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int)
    return _adj

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNResnet(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, t=0, adj_file=None):
        super(GCNResnet, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_classes, in_channel)
        self.pooling = nn.MaxPool2d(8, 8)

        self.gc1 = GraphConvolution(in_channel, 256)
        self.gc2 = GraphConvolution(256, 512)
        self.relu = nn.LeakyReLU(0.2)

        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature, inp):
        feature = self.features(feature)
        out_feats = feature
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)

        # inp = self.embedding(inp)
        adj = gen_adj(self.A).detach()
        inp_emb = self.embedding(inp)
        x = self.gc1(inp_emb, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        return x, out_feats

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]


def gcn_resnet18(num_classes, t, pretrained=False, adj_file=None, in_channel=300):
    model = models.resnet18(pretrained=pretrained)
    return GCNResnet(model, num_classes, t=t, adj_file=adj_file, in_channel=in_channel)


def gen_adj_pkl(filename='/data/users/liuziyi/PyProgram/deep_PSL/data/adj/Multi.pkl', t=0.1, p=0.2):
    if os.path.exists(filename):
        f = open(filename, 'rb')
        r = pickle.load(f)
        f.close()
        return r
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

    all_genes = get_enhanced_gene_list()
    train_genes = all_genes
    gene_label_dict = load_label_from_file('/data/users/liuziyi/PyProgram/ImPloc-revision/label/enhanced_label.txt')

    img = []
    gene_pics_dict, label_gene_dict = {}, {}
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
            gene_pics_dict[gene] = pics
            label = gene_label_dict[gene]
            if label not in label_gene_dict:
                label_gene_dict[label] = []
            label_gene_dict[label].append(gene)
            if len(label_gene_dict[label]) > max_gene_cnt:
                max_gene_cnt = len(label_gene_dict[label])

    label_cnt = np.zeros(num_class)
    adj = np.zeros((num_class, num_class))
    for gene in gene_pics_dict:
        for pic in gene_pics_dict[gene]:
            img.append((pic, gene, gene_label_dict[gene]))
            one_hot = hash2array(gene_label_dict[gene])
            label_cnt += hash2array(gene_label_dict[gene])
            occur_labels = []
            for i in range(num_class):
                if one_hot[i] == 1:
                    occur_labels.append(i)
            for ind1 in occur_labels:
                for ind2 in occur_labels:
                    adj[ind1, ind2] += 1

    result = {}
    result['adj'] = adj
    result['nums'] = label_cnt
    f = open(filename, 'wb')
    pickle.dump(result, f)
    return adj


def gen_multi_adj_file(img_dir, num_class=7, filename='/data/users/liuziyi/PyProgram/deep_PSL/data/adj/Multi.pkl'):
    img = []
    label_img_map = {}
    folders = os.listdir(img_dir)
    folders = sorted(folders)
    adj = np.zeros((num_class, num_class))
    label_cnt = np.zeros(num_class)

    file_index_map, index = {}, 0
    for folder in folders:
        label = list(map(int, folder.split("_")))
        one_hot = np.zeros(num_class)
        for l in label:
            one_hot[l] = 1
        file_list = glob(img_dir + '/' + folder + '/*', )
        img_list = []
        for file in file_list:
            img.append((file, one_hot))
            file_index_map[file] = index
            index += 1
            img_list.append(file)
            for ind1 in label:
                for ind2 in label:
                    adj[ind1, ind2] += 1
            label_cnt += one_hot

    result = {}
    result['adj'] = adj
    result['nums'] = label_cnt
    f = open(filename, 'wb')
    pickle.dump(result, f)

    return adj


if __name__ == '__main__':
    train_dir = '/data/users/liuziyi/PyProgram/deep_PSL/data/train/cropMulti'
    test_dir = '/data/users/liuziyi/PyProgram/deep_PSL/data/test/cropMulti'
    adj = gen_multi_adj_file('/data/users/liuziyi/PyProgram/deep_PSL/data/train/cropMulti')
