from sklearn.metrics import *
import torch.nn.functional as F
import numpy as np
import utils.npmetrics as npmetrics

class Evaluator(object):
    def __init__(self, output, target):
        self.output = output
        self.target = target

    def getAcc(self):
        # same as precision/recall with micro
        acc = accuracy_score(self.target, self.output)
        return acc

    def getPrecision(self):
        acc = precision_score(self.target, self.output, average='macro')
        return acc

    def getRecall(self):
        acc = recall_score(self.target, self.output, average='macro')
        return acc

    def getF1(self):
        acc = f1_score(self.target, self.output, average='macro')
        return acc

    def getConfusionMatrix(self):
        matrix = confusion_matrix((self.target, self.output))
        return matrix


def csa_loss(x, y, class_eq):
    margin = 1
    dist = F.pairwise_distance(x, y)
    loss = class_eq * dist.pow(2)
    loss += (1-class_eq) * (margin - dist).clamp(min=0).pow(2)
    return loss.mean()


def torch_metrics(gt, predict):
    ex_subset_acc = npmetrics.example_subset_accuracy(gt, predict)
    ex_acc = npmetrics.example_accuracy(gt, predict)
    ex_precision = npmetrics.example_precision(gt, predict)
    ex_recall = npmetrics.example_recall(gt, predict)
    ex_f1 = npmetrics.compute_f1(ex_precision, ex_recall)

    lab_acc_macro = npmetrics.label_accuracy_macro(gt, predict)
    lab_precision_macro = npmetrics.label_precision_macro(gt, predict)
    lab_recall_macro = npmetrics.label_recall_macro(gt, predict)
    lab_f1_macro = npmetrics.compute_f1(lab_precision_macro, lab_recall_macro)

    lab_acc_micro = npmetrics.label_accuracy_micro(gt, predict)
    lab_precision_micro = npmetrics.label_precision_micro(gt, predict)
    lab_recall_micro = npmetrics.label_recall_micro(gt, predict)
    lab_f1_micro = npmetrics.compute_f1(lab_precision_micro, lab_recall_micro)
    return ex_subset_acc


class Multi_eva(object):
    def __init__(self, output, target):
        self.output = output
        self.target = target

    def getHloss(self):
        d = len(self.output)
        c = len(self.output[0])
        sumD = 0
        for i in range(d):
            sh = sum(self.output[i]!=self.target[i])/c
            sumD += sh
        return sumD/d

    def getSubAcc(self):
        #subset accuracy
        d = len(self.output)
        acc = 0
        for i in range(d):
            acc += int((self.output[i]==self.target[i]).all())
        return acc/d

def getTout(y,T=0.5):
    # y size:[bs,classnum]
    zero = [0]*len(y[0])
    for i in range(len(y)):
        temp = np.array(list(map(int,y[i]>T)))
        if temp == zero:
            temp[np.argmax(y[i])] = 1
        y[i] = temp
    return y

def getTopKout(y,k=2,mode="output"):
    cl = len(y[0])
    if mode=="ensemble":
        for i in range(len(y)):
            ind = np.argsort(y[i])[-k:]
            temp = [0]*cl
            if ind[-2:-1]==0:
                temp[ind[-1]]=1
            else:
                for j in range(k):
                    temp[ind[j]] = 1
            y[i] = temp
    else:
        for i in range(len(y)):
            ind = np.argsort(y[i])[-k:]
            temp = [0]*cl
            for j in range(k):
                temp[ind[j]] = 1
            y[i] = temp
    return y


def getKTout(y,k=3,T=0.2):
    cl = len(y[0])
    for i in range(len(y)):
        ind = np.argsort(y[i])[-k:]
        temp = [0]*cl
        if y[i,ind[2]]-y[i,ind[0]]<T:
            temp[ind[0]],temp[ind[1]],temp[ind[2]]=1,1,1
        else:
            temp[ind[1]],temp[ind[2]]=1,1
        y[i] = temp
    return y


if __name__ == "__main__":
    output = []
    output.append(np.array([1,0,0]))
    output.append(np.array([1,1,0]))
    output.append(np.array([0,1,1]))
    target = []
    target.append(np.array([1,1,0]))
    target.append(np.array([0,1,0]))
    target.append(np.array([1,1,0]))
    s = Multi_eva(output,target)
    s.getHloss()
    a =1