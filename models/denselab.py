import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from config import *

BatchNorm2d = SynchronizedBatchNorm2d


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseModel(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0.4, num_classes=7):

        super(DenseModel, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        self.layer1, num_features0 = self._make_layer(0, block_config[0], num_features, bn_size, growth_rate, drop_rate)
        self.layer2, num_features1 = self._make_layer(1, block_config[1], num_features0, bn_size, growth_rate, drop_rate)
        self.layer3, num_features2 = self._make_layer(2, block_config[2], num_features1, bn_size, growth_rate, drop_rate)
        self.layer4, num_features3 = self._make_layer(3, block_config[3], num_features2, bn_size, growth_rate, drop_rate)


        # Final batch norm
        self.layer4.add_module('norm5', nn.BatchNorm2d(num_features3))

        # Linear layer
        self.classifier = nn.Linear(1920, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(self, i, num_layers, num_features, bn_size, growth_rate, drop_rate):
        layers = nn.Sequential()
        block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                            bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        layers.add_module('denseblock%d' % (i + 1), block)
        num_features = num_features + num_layers * growth_rate
        if i != 3:
            trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
            layers.add_module('transition%d' % (i + 1), trans)
            num_features = num_features // 2
        return layers, num_features

    def forward(self, x):
        x = self.features(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out = self.layer4(x)

        return out
        # out = F.relu(out, inplace=True)
        # out = F.avg_pool2d(out, kernel_size=7)
            # .view(out.size(0), -1)
        # out = self.classifier(out)
        # return out, low_feature


def densenet121(num_class, **kwargs):
    model = DenseModel(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), num_classes=num_class, **kwargs)
    return model


def densenet169(num_class, **kwargs):
    model = DenseModel(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), num_classes=num_class, **kwargs)
    return model


def densenet201(num_class, **kwargs):
    model = DenseModel(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32), num_classes=num_class, **kwargs)
    return model


def densenet161(num_class, **kwargs):
    model = DenseModel(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24), num_classes=num_class,  **kwargs)
    return model

class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, dilation):
        super(ASPP_module, self).__init__()
        if dilation == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = dilation
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class denselab(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=8, os=16, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
            # print("Backbone: Resnet-101")
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))
        super(denselab, self).__init__()

        # Atrous Conv
        self.densenet_features = densenet201(n_classes)

        #ASPP
        if os == 16:
            dilations = [1, 6, 12, 18]
        elif os == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(1920, 256, dilation=dilations[0])
        self.aspp2 = ASPP_module(1920, 256, dilation=dilations[1])
        self.aspp3 = ASPP_module(1920, 256, dilation=dilations[2])
        self.aspp4 = ASPP_module(1920, 256, dilation=dilations[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(1920, 256, 1, stride=1, bias=False),
                                             BatchNorm2d(256),
                                             nn.ReLU())
        # self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.last_conv = nn.Sequential(nn.Conv2d(1280, 256, 1, bias=False),
                                       BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.AdaptiveAvgPool2d((1, 1)))

        self.classifier = nn.Linear(256, n_classes)

    def forward(self, input):
        # x, low_feature = self.densenet_features(input)
        x = self.densenet_features(input)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        feature = self.last_conv(x)
        feature = feature.view(feature.size(0), -1)
        #classifer
        out = self.classifier(feature)

        return out, feature

def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = [model.resnet_features]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = [model.aspp1, model.aspp2, model.aspp3, model.aspp4, model.conv1, model.conv2, model.last_conv]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k






