# @Author: zechenghe
# @Date:   2019-01-21T12:17:31-05:00
# @Last modified by:   zechenghe
# @Last modified time: 2019-02-01T14:01:15-05:00

import time
import math
import os
import sys
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import collections

class LeNet(nn.Module):
    def __init__(self, NChannels):
        super(LeNet, self).__init__()
        self.features = []
        self.classifier = []
        self.layerDict = collections.OrderedDict()

        self.conv1 = nn.Conv2d(
            in_channels = NChannels,
            out_channels = 8,
            kernel_size = 5
        )
        self.features.append(self.conv1)
        self.layerDict['conv1'] = self.conv1

        self.ReLU1 = nn.ReLU(True)
        self.features.append(self.ReLU1)
        self.layerDict['ReLU1'] = self.ReLU1

        self.pool1 = nn.MaxPool2d(2,2)
        self.features.append(self.pool1)
        self.layerDict['pool1'] = self.pool1

        self.conv2 = nn.Conv2d(
            in_channels = 8,
            out_channels = 16,
            kernel_size = 5
        )
        self.features.append(self.conv2)
        self.layerDict['conv2'] = self.conv2

        self.ReLU2 = nn.ReLU(True)
        self.features.append(self.ReLU2)
        self.layerDict['ReLU2'] = self.ReLU2

        self.pool2 = nn.MaxPool2d(2,2)
        self.features.append(self.pool2)
        self.layerDict['pool2'] = self.pool2

        self.feature_dims = 16 * 4 * 4
        self.fc1 = nn.Linear(self.feature_dims, 120)
        self.classifier.append(self.fc1)
        self.layerDict['fc1'] = self.fc1

        self.fc1act = nn.ReLU(True)
        self.classifier.append(self.fc1act)
        self.layerDict['fc1act'] = self.fc1act


        self.fc2 = nn.Linear(120, 84)
        self.classifier.append(self.fc2)
        self.layerDict['fc2'] = self.fc2

        self.fc2act = nn.ReLU(True)
        self.classifier.append(self.fc2act)
        self.layerDict['fc2act'] = self.fc2act

        self.fc3 = nn.Linear(84, 10)
        self.classifier.append(self.fc3)
        self.layerDict['fc3'] = self.fc3


    def forward(self, x):
        for layer in self.features:
            x = layer(x)

        #print 'x.size', x.size()
        x = x.view(-1, self.feature_dims)

        for layer in self.classifier:
            x = layer(x)
        return x

    def forward_from(self, x, layer):

        if layer == 'input':
            return x

        if layer in self.layerDict:
            targetLayer = self.layerDict[layer]

            if targetLayer in self.features:
                layeridx = self.features.index(targetLayer)
                for func in self.features[layeridx+1:]:
                    x = func(x)
#                    print "x.size() ", x.size()

#                print "Pass Features "
                x = x.view(-1, self.feature_dims)
                for func in self.classifier:
                    x = func(x)
                return x

            else:
                layeridx = self.classifier.index(targetLayer)
                for func in self.classifier[layeridx+1:]:
                    x = func(x)
                return x
        else:
            print "layer not exists"
            exit(1)


    def getLayerOutput(self, x, targetLayer):
        if targetLayer == 'input':
            return x

        for layer in self.features:
            x = layer(x)
            if layer == targetLayer:
                return x

        x = x.view(-1, self.feature_dims)
        for layer in self.classifier:
            x = layer(x)
            if layer == targetLayer:
                return x

        # Should not go here
        raise Exception("Target layer not found")

    def getLayerOutputFrom(self, x, sourceLayer, targetLayer):

        if targetLayer == 'input':
            return x

        if sourceLayer == 'input':
            return getLayerOutput(self, x, self.layerDict[targetLayer])

        if sourceLayer in self.layerDict and targetLayer in self.layerDict:
            sourceLayer = self.layerDict[sourceLayer]
            targetLayer = self.layerDict[targetLayer]

            if sourceLayer in self.features and targetLayer in self.features:
                sourceLayeridx = self.features.index(sourceLayer)
                targetLayeridx = self.features.index(targetLayer)

                for func in self.features[sourceLayeridx+1:targetLayeridx+1]:
                    x = func(x)
                return x

            elif sourceLayer in self.classifier and targetLayer in self.classifier:
                sourceLayeridx = self.classifier.index(sourceLayer)
                targetLayeridx = self.classifier.index(targetLayer)

                for func in self.classifier[sourceLayeridx+1:targetLayeridx+1]:
                    x = func(x)
                return x

            elif sourceLayer in self.features and targetLayer in self.classifier:
                sourceLayeridx = self.features.index(sourceLayer)
                for func in self.features[sourceLayeridx+1:]:
                    x = func(x)

                x = x.view(-1, self.feature_dims)
                targetLayeridx = self.classifier.index(targetLayer)
                for func in self.classifier[:targetLayeridx+1]:
                    x = func(x)
                return x

            else:
                print "Target layer cannot before source layer"
                exit(1)
        else:
            print "layer not exists"
            exit(1)


class CIFAR10CNN(nn.Module):
    def __init__(self, NChannels):
        super(CIFAR10CNN, self).__init__()
        self.features = []
        self.layerDict = collections.OrderedDict()

        self.conv11 = nn.Conv2d(
            in_channels = NChannels,
            out_channels = 64,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv11)
        self.layerDict['conv11'] = self.conv11

        self.ReLU11 = nn.ReLU()
        self.features.append(self.ReLU11)
        self.layerDict['ReLU11'] = self.ReLU11

        self.conv12 = nn.Conv2d(
            in_channels = 64,
            out_channels = 64,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv12)
        self.layerDict['conv12'] = self.conv12

        self.ReLU12 = nn.ReLU()
        self.features.append(self.ReLU12)
        self.layerDict['ReLU12'] = self.ReLU12

        self.pool1 = nn.MaxPool2d(2,2)
        self.features.append(self.pool1)
        self.layerDict['pool1'] = self.pool1


        self.conv21 = nn.Conv2d(
            in_channels = 64,
            out_channels = 128,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv21)
        self.layerDict['conv21'] = self.conv21

        self.ReLU21 = nn.ReLU()
        self.features.append(self.ReLU21)
        self.layerDict['ReLU21'] = self.ReLU21

        self.conv22 = nn.Conv2d(
            in_channels = 128,
            out_channels = 128,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv22)
        self.layerDict['conv22'] = self.conv22

        self.ReLU22 = nn.ReLU()
        self.features.append(self.ReLU22)
        self.layerDict['ReLU22'] = self.ReLU22

        self.pool2 = nn.MaxPool2d(2,2)
        self.features.append(self.pool2)
        self.layerDict['pool2'] = self.pool2

        self.conv31 = nn.Conv2d(
            in_channels = 128,
            out_channels = 128,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv31)
        self.layerDict['conv31'] = self.conv31

        self.ReLU31 = nn.ReLU()
        self.features.append(self.ReLU31)
        self.layerDict['ReLU31'] = self.ReLU31

        self.conv32 = nn.Conv2d(
            in_channels = 128,
            out_channels = 128,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv32)
        self.layerDict['conv32'] = self.conv32


        self.ReLU32 = nn.ReLU()
        self.features.append(self.ReLU32)
        self.layerDict['ReLU32'] = self.ReLU32

        self.pool3 = nn.MaxPool2d(2,2)
        self.features.append(self.pool3)
        self.layerDict['pool3'] = self.pool3

        self.classifier = []

        self.feature_dims = 4 * 4 * 128
        self.fc1 = nn.Linear(self.feature_dims, 512)
        self.classifier.append(self.fc1)
        self.layerDict['fc1'] = self.fc1

        self.fc1act = nn.Sigmoid()
        self.classifier.append(self.fc1act)
        self.layerDict['fc1act'] = self.fc1act

        self.fc2 = nn.Linear(512, 10)
        self.classifier.append(self.fc2)
        self.layerDict['fc2'] = self.fc2

    def forward(self, x):
        for layer in self.features:
            x = layer(x)

        x = x.view(-1, self.feature_dims)

        for layer in self.classifier:
            x = layer(x)
        return x


    def forward_from(self, x, layer):

        if layer in self.layerDict:
            targetLayer = self.layerDict[layer]

            if targetLayer in self.features:
                layeridx = self.features.index(targetLayer)
                for func in self.features[layeridx+1:]:
                    x = func(x)

                x = x.view(-1, self.feature_dims)
                for func in self.classifier:
                    x = func(x)
                return x

            else:
                layeridx = self.classifier.index(targetLayer)
                for func in self.classifier[layeridx:]:
                    x = func(x)
                return x
        else:
            print "layer not exists"
            exit(1)


    def getLayerOutput(self, x, targetLayer):
        for layer in self.features:
            x = layer(x)
            if layer == targetLayer:
                return x

        x = x.view(-1, self.feature_dims)
        for layer in self.classifier:
            x = layer(x)
            if layer == targetLayer:
                return x

        # Should not go here
        print "Target layer not found"
        exit(1)


class LeNetAlternativeconv1(nn.Module):
    def __init__(self, NChannels):
        super(LeNetAlternativeconv1, self).__init__()
        self.features = []
        self.classifier = []
        self.layerDict = collections.OrderedDict()

        self.conv0 = nn.Conv2d(
            in_channels = NChannels,
            out_channels = 8,
            kernel_size = 3
        )
        self.features.append(self.conv0)
        self.layerDict['conv0'] = self.conv0

        self.conv1 = nn.Conv2d(
            in_channels = 8,
            out_channels = 8,
            kernel_size = 3
        )
        self.features.append(self.conv1)
        self.layerDict['conv1'] = self.conv1

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        return x


class LeNetAlternativeconv1Archi2(nn.Module):
    def __init__(self, NChannels):
        super(LeNetAlternativeconv1Archi2, self).__init__()
        self.features = []
        self.classifier = []
        self.layerDict = collections.OrderedDict()

        self.conv0 = nn.Conv2d(
            in_channels = NChannels,
            out_channels = 8,
            kernel_size = 3
        )
        self.features.append(self.conv0)
        self.layerDict['conv0'] = self.conv0

        self.conv1 = nn.Conv2d(
            in_channels = 8,
            out_channels = 8,
            kernel_size = 3
        )
        self.features.append(self.conv1)
        self.layerDict['conv1'] = self.conv1

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        return x


class LeNetAlternativeReLU1(nn.Module):
    def __init__(self, NChannels):
        super(LeNetAlternativeReLU1, self).__init__()
        self.features = []
        self.classifier = []
        self.layerDict = collections.OrderedDict()

        self.conv1 = nn.Conv2d(
            in_channels = NChannels,
            out_channels = 8,
            kernel_size = 5
        )
        self.features.append(self.conv1)
        self.layerDict['conv1'] = self.conv1

        self.ReLU1 = nn.ReLU(True)
        self.features.append(self.ReLU1)
        self.layerDict['ReLU1'] = self.ReLU1

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        return x


class LeNetAlternativepool1(nn.Module):
    def __init__(self, NChannels):
        super(LeNetAlternativepool1, self).__init__()
        self.features = []
        self.classifier = []
        self.layerDict = collections.OrderedDict()

        self.conv1 = nn.Conv2d(
            in_channels = NChannels,
            out_channels = 8,
            kernel_size = 5
        )
        self.features.append(self.conv1)
        self.layerDict['conv1'] = self.conv1

        self.ReLU1 = nn.ReLU(True)
        self.features.append(self.ReLU1)
        self.layerDict['ReLU1'] = self.ReLU1

        self.pool1 = nn.MaxPool2d(2,2)
        self.features.append(self.pool1)
        self.layerDict['pool1'] = self.pool1

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        return x

class LeNetAlternativeconv2(nn.Module):
    def __init__(self, NChannels):
        super(LeNetAlternativeconv2, self).__init__()
        self.features = []
        self.classifier = []
        self.layerDict = collections.OrderedDict()

        self.conv1 = nn.Conv2d(
            in_channels = NChannels,
            out_channels = 8,
            kernel_size = 5
        )
        self.features.append(self.conv1)
        self.layerDict['conv1'] = self.conv1

        self.ReLU1 = nn.ReLU(True)
        self.features.append(self.ReLU1)
        self.layerDict['ReLU1'] = self.ReLU1

        self.pool1 = nn.MaxPool2d(2,2)
        self.features.append(self.pool1)
        self.layerDict['pool1'] = self.pool1

        self.conv2 = nn.Conv2d(
            in_channels = 8,
            out_channels = 16,
            kernel_size = 5
        )
        self.features.append(self.conv2)
        self.layerDict['conv2'] = self.conv2

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        return x


class LeNetAlternativeReLU2(nn.Module):
    def __init__(self, NChannels):
        super(LeNetAlternativeReLU2, self).__init__()
        self.features = []
        self.classifier = []
        self.layerDict = collections.OrderedDict()

        self.conv1 = nn.Conv2d(
            in_channels = NChannels,
            out_channels = 8,
            kernel_size = 5
        )
        self.features.append(self.conv1)
        self.layerDict['conv1'] = self.conv1

        self.ReLU1 = nn.ReLU(True)
        self.features.append(self.ReLU1)
        self.layerDict['ReLU1'] = self.ReLU1

        self.pool1 = nn.MaxPool2d(2,2)
        self.features.append(self.pool1)
        self.layerDict['pool1'] = self.pool1

        self.conv2 = nn.Conv2d(
            in_channels = 8,
            out_channels = 16,
            kernel_size = 5
        )
        self.features.append(self.conv2)
        self.layerDict['conv2'] = self.conv2

        self.ReLU2 = nn.ReLU(True)
        self.features.append(self.ReLU2)
        self.layerDict['ReLU2'] = self.ReLU2

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        return x


class LeNetAlternativepool2(nn.Module):
    def __init__(self, NChannels):
        super(LeNetAlternativepool2, self).__init__()
        self.features = []
        self.classifier = []
        self.layerDict = collections.OrderedDict()

        self.conv1 = nn.Conv2d(
            in_channels = NChannels,
            out_channels = 8,
            kernel_size = 5
        )
        self.features.append(self.conv1)
        self.layerDict['conv1'] = self.conv1

        self.ReLU1 = nn.ReLU(True)
        self.features.append(self.ReLU1)
        self.layerDict['ReLU1'] = self.ReLU1

        self.pool1 = nn.MaxPool2d(2,2)
        self.features.append(self.pool1)
        self.layerDict['pool1'] = self.pool1

        self.conv2 = nn.Conv2d(
            in_channels = 8,
            out_channels = 16,
            kernel_size = 5
        )
        self.features.append(self.conv2)
        self.layerDict['conv2'] = self.conv2

        self.ReLU2 = nn.ReLU(True)
        self.features.append(self.ReLU2)
        self.layerDict['ReLU2'] = self.ReLU2

        self.pool2 = nn.MaxPool2d(2,2)
        self.features.append(self.pool2)
        self.layerDict['pool2'] = self.pool2

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        return x


class LeNetAlternativefc1(nn.Module):
    def __init__(self, NChannels):
        super(LeNetAlternativefc1, self).__init__()
        self.features = []
        self.classifier = []
        self.layerDict = collections.OrderedDict()

        self.conv1 = nn.Conv2d(
            in_channels = NChannels,
            out_channels = 8,
            kernel_size = 5
        )
        self.features.append(self.conv1)
        self.layerDict['conv1'] = self.conv1

        self.ReLU1 = nn.ReLU(True)
        self.features.append(self.ReLU1)
        self.layerDict['ReLU1'] = self.ReLU1

        self.pool1 = nn.MaxPool2d(2,2)
        self.features.append(self.pool1)
        self.layerDict['pool1'] = self.pool1

        self.conv2 = nn.Conv2d(
            in_channels = 8,
            out_channels = 16,
            kernel_size = 5
        )
        self.features.append(self.conv2)
        self.layerDict['conv2'] = self.conv2

        self.ReLU2 = nn.ReLU(True)
        self.features.append(self.ReLU2)
        self.layerDict['ReLU2'] = self.ReLU2

        self.pool2 = nn.MaxPool2d(2,2)
        self.features.append(self.pool2)
        self.layerDict['pool2'] = self.pool2

        self.feature_dims = 16 * 4 * 4
        self.fc1 = nn.Linear(self.feature_dims, 120)
        self.classifier.append(self.fc1)
        self.layerDict['fc1'] = self.fc1


    def forward(self, x):
        for layer in self.features:
            x = layer(x)

        #print 'x.size', x.size()
        x = x.view(-1, self.feature_dims)

        for layer in self.classifier:
            x = layer(x)
        return x

class LeNetAlternativefc1act(nn.Module):
    def __init__(self, NChannels):
        super(LeNetAlternativefc1act, self).__init__()
        self.features = []
        self.classifier = []
        self.layerDict = collections.OrderedDict()

        self.conv1 = nn.Conv2d(
            in_channels = NChannels,
            out_channels = 8,
            kernel_size = 5
        )
        self.features.append(self.conv1)
        self.layerDict['conv1'] = self.conv1

        self.ReLU1 = nn.ReLU(True)
        self.features.append(self.ReLU1)
        self.layerDict['ReLU1'] = self.ReLU1

        self.pool1 = nn.MaxPool2d(2,2)
        self.features.append(self.pool1)
        self.layerDict['pool1'] = self.pool1

        self.conv2 = nn.Conv2d(
            in_channels = 8,
            out_channels = 16,
            kernel_size = 5
        )
        self.features.append(self.conv2)
        self.layerDict['conv2'] = self.conv2

        self.ReLU2 = nn.ReLU(True)
        self.features.append(self.ReLU2)
        self.layerDict['ReLU2'] = self.ReLU2

        self.pool2 = nn.MaxPool2d(2,2)
        self.features.append(self.pool2)
        self.layerDict['pool2'] = self.pool2

        self.feature_dims = 16 * 4 * 4
        self.fc1 = nn.Linear(self.feature_dims, 120)
        self.classifier.append(self.fc1)
        self.layerDict['fc1'] = self.fc1

        self.fc1act = nn.ReLU(True)
        self.classifier.append(self.fc1act)
        self.layerDict['fc1act'] = self.fc1act


    def forward(self, x):
        for layer in self.features:
            x = layer(x)

        #print 'x.size', x.size()
        x = x.view(-1, self.feature_dims)

        for layer in self.classifier:
            x = layer(x)
        return x


class LeNetAlternativefc2(nn.Module):
    def __init__(self, NChannels):
        super(LeNetAlternativefc2, self).__init__()
        self.features = []
        self.classifier = []
        self.layerDict = collections.OrderedDict()

        self.conv1 = nn.Conv2d(
            in_channels = NChannels,
            out_channels = 8,
            kernel_size = 5
        )
        self.features.append(self.conv1)
        self.layerDict['conv1'] = self.conv1

        self.ReLU1 = nn.ReLU(True)
        self.features.append(self.ReLU1)
        self.layerDict['ReLU1'] = self.ReLU1

        self.pool1 = nn.MaxPool2d(2,2)
        self.features.append(self.pool1)
        self.layerDict['pool1'] = self.pool1

        self.conv2 = nn.Conv2d(
            in_channels = 8,
            out_channels = 16,
            kernel_size = 5
        )
        self.features.append(self.conv2)
        self.layerDict['conv2'] = self.conv2

        self.ReLU2 = nn.ReLU(True)
        self.features.append(self.ReLU2)
        self.layerDict['ReLU2'] = self.ReLU2

        self.pool2 = nn.MaxPool2d(2,2)
        self.features.append(self.pool2)
        self.layerDict['pool2'] = self.pool2

        self.feature_dims = 16 * 4 * 4
        self.fc1 = nn.Linear(self.feature_dims, 120)
        self.classifier.append(self.fc1)
        self.layerDict['fc1'] = self.fc1

        self.fc1act = nn.ReLU(True)
        self.classifier.append(self.fc1act)
        self.layerDict['fc1act'] = self.fc1act


        self.fc2 = nn.Linear(120, 84)
        self.classifier.append(self.fc2)
        self.layerDict['fc2'] = self.fc2

    def forward(self, x):
        for layer in self.features:
            x = layer(x)

        #print 'x.size', x.size()
        x = x.view(-1, self.feature_dims)

        for layer in self.classifier:
            x = layer(x)
        return x

class LeNetAlternativefc2act(nn.Module):
    def __init__(self, NChannels):
        super(LeNetAlternativefc2act, self).__init__()
        self.features = []
        self.classifier = []
        self.layerDict = collections.OrderedDict()

        self.conv1 = nn.Conv2d(
            in_channels = NChannels,
            out_channels = 8,
            kernel_size = 5
        )
        self.features.append(self.conv1)
        self.layerDict['conv1'] = self.conv1

        self.ReLU1 = nn.ReLU(True)
        self.features.append(self.ReLU1)
        self.layerDict['ReLU1'] = self.ReLU1

        self.pool1 = nn.MaxPool2d(2,2)
        self.features.append(self.pool1)
        self.layerDict['pool1'] = self.pool1

        self.conv2 = nn.Conv2d(
            in_channels = 8,
            out_channels = 16,
            kernel_size = 5
        )
        self.features.append(self.conv2)
        self.layerDict['conv2'] = self.conv2

        self.ReLU2 = nn.ReLU(True)
        self.features.append(self.ReLU2)
        self.layerDict['ReLU2'] = self.ReLU2

        self.pool2 = nn.MaxPool2d(2,2)
        self.features.append(self.pool2)
        self.layerDict['pool2'] = self.pool2

        self.feature_dims = 16 * 4 * 4
        self.fc1 = nn.Linear(self.feature_dims, 120)
        self.classifier.append(self.fc1)
        self.layerDict['fc1'] = self.fc1

        self.fc1act = nn.ReLU(True)
        self.classifier.append(self.fc1act)
        self.layerDict['fc1act'] = self.fc1act


        self.fc2 = nn.Linear(120, 84)
        self.classifier.append(self.fc2)
        self.layerDict['fc2'] = self.fc2

        self.fc2act = nn.ReLU(True)
        self.classifier.append(self.fc2act)
        self.layerDict['fc2act'] = self.fc2act

    def forward(self, x):
        for layer in self.features:
            x = layer(x)

        #print 'x.size', x.size()
        x = x.view(-1, self.feature_dims)

        for layer in self.classifier:
            x = layer(x)
        return x

class LeNetAlternativefc3(nn.Module):
    def __init__(self, NChannels):
        super(LeNetAlternativefc3, self).__init__()
        self.features = []
        self.classifier = []
        self.layerDict = collections.OrderedDict()

        self.conv1 = nn.Conv2d(
            in_channels = NChannels,
            out_channels = 8,
            kernel_size = 5
        )
        self.features.append(self.conv1)
        self.layerDict['conv1'] = self.conv1

        self.ReLU1 = nn.ReLU(True)
        self.features.append(self.ReLU1)
        self.layerDict['ReLU1'] = self.ReLU1

        self.pool1 = nn.MaxPool2d(2,2)
        self.features.append(self.pool1)
        self.layerDict['pool1'] = self.pool1

        self.conv2 = nn.Conv2d(
            in_channels = 8,
            out_channels = 16,
            kernel_size = 5
        )
        self.features.append(self.conv2)
        self.layerDict['conv2'] = self.conv2

        self.ReLU2 = nn.ReLU(True)
        self.features.append(self.ReLU2)
        self.layerDict['ReLU2'] = self.ReLU2

        self.pool2 = nn.MaxPool2d(2,2)
        self.features.append(self.pool2)
        self.layerDict['pool2'] = self.pool2

        self.feature_dims = 16 * 4 * 4
        self.fc1 = nn.Linear(self.feature_dims, 120)
        self.classifier.append(self.fc1)
        self.layerDict['fc1'] = self.fc1

        self.fc1act = nn.ReLU(True)
        self.classifier.append(self.fc1act)
        self.layerDict['fc1act'] = self.fc1act


        self.fc2 = nn.Linear(120, 84)
        self.classifier.append(self.fc2)
        self.layerDict['fc2'] = self.fc2

        self.fc2act = nn.ReLU(True)
        self.classifier.append(self.fc2act)
        self.layerDict['fc2act'] = self.fc2act

        self.fc3 = nn.Linear(84, 10)
        self.classifier.append(self.fc3)
        self.layerDict['fc3'] = self.fc3


    def forward(self, x):
        for layer in self.features:
            x = layer(x)

        #print 'x.size', x.size()
        x = x.view(-1, self.feature_dims)

        for layer in self.classifier:
            x = layer(x)
        return x



class LeNetAlternativeReLU2Archi2(nn.Module):
    def __init__(self, NChannels):
        super(LeNetAlternativeReLU2Archi2, self).__init__()
        self.features = []
        self.classifier = []
        self.layerDict = collections.OrderedDict()

        self.conv11 = nn.Conv2d(
            in_channels = NChannels,
            out_channels = 8,
            kernel_size = 3
        )
        self.features.append(self.conv11)
        self.layerDict['conv11'] = self.conv11

        self.conv12 = nn.Conv2d(
            in_channels = 8,
            out_channels = 16,
            kernel_size = 3
        )
        self.features.append(self.conv12)
        self.layerDict['conv12'] = self.conv12

        self.pool1 = nn.MaxPool2d(2,2)
        self.features.append(self.pool1)
        self.layerDict['pool1'] = self.pool1

        self.conv21 = nn.Conv2d(
            in_channels = 16,
            out_channels = 16,
            kernel_size = 3
        )
        self.features.append(self.conv21)
        self.layerDict['conv21'] = self.conv21

        self.conv2 = nn.Conv2d(
            in_channels = 16,
            out_channels = 16,
            kernel_size = 3
        )
        self.features.append(self.conv2)
        self.layerDict['conv2'] = self.conv2

        self.ReLU2 = nn.ReLU(True)
        self.features.append(self.ReLU2)
        self.layerDict['ReLU2'] = self.ReLU2

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        return x



class CIFAR10CNNAlternativeconv11(nn.Module):
    def __init__(self, NChannels):
        super(CIFAR10CNNAlternativeconv11, self).__init__()
        self.features = []
        self.layerDict = collections.OrderedDict()

        self.conv11 = nn.Conv2d(
            in_channels = NChannels,
            out_channels = 64,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv11)
        self.layerDict['conv11'] = self.conv11

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        return x

    def getLayerOutput(self, x, targetLayer):
        for layer in self.features:
            x = layer(x)
            if layer == targetLayer:
                return x

        # Should not go here
        print "Target layer not found"
        exit(1)


class CIFAR10CNNAlternativeconv11Archi2(nn.Module):
    def __init__(self, NChannels):
        super(CIFAR10CNNAlternativeconv11Archi2, self).__init__()
        self.features = []
        self.layerDict = collections.OrderedDict()

        self.conv10 = nn.Conv2d(
            in_channels = NChannels,
            out_channels = 16,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv10)
        self.layerDict['conv10'] = self.conv10

        self.conv11 = nn.Conv2d(
            in_channels = 16,
            out_channels = 64,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv11)
        self.layerDict['conv11'] = self.conv11

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        return x

    def getLayerOutput(self, x, targetLayer):
        for layer in self.features:
            x = layer(x)
            if layer == targetLayer:
                return x

        # Should not go here
        print "Target layer not found"
        exit(1)


class CIFAR10CNNAlternativeReLU22(nn.Module):
    def __init__(self, NChannels):
        super(CIFAR10CNNAlternativeReLU22, self).__init__()
        self.features = []
        self.layerDict = collections.OrderedDict()

        self.conv11 = nn.Conv2d(
            in_channels = NChannels,
            out_channels = 64,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv11)
        self.layerDict['conv11'] = self.conv11

        self.ReLU11 = nn.ReLU()
        self.features.append(self.ReLU11)
        self.layerDict['ReLU11'] = self.ReLU11

        self.conv12 = nn.Conv2d(
            in_channels = 64,
            out_channels = 64,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv12)
        self.layerDict['conv12'] = self.conv12

        self.ReLU12 = nn.ReLU()
        self.features.append(self.ReLU12)
        self.layerDict['ReLU12'] = self.ReLU12

        self.pool1 = nn.MaxPool2d(2,2)
        self.features.append(self.pool1)
        self.layerDict['pool1'] = self.pool1


        self.conv21 = nn.Conv2d(
            in_channels = 64,
            out_channels = 128,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv21)
        self.layerDict['conv21'] = self.conv21

        self.ReLU21 = nn.ReLU()
        self.features.append(self.ReLU21)
        self.layerDict['ReLU21'] = self.ReLU21

        self.conv22 = nn.Conv2d(
            in_channels = 128,
            out_channels = 128,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv22)
        self.layerDict['conv22'] = self.conv22

        self.ReLU22 = nn.ReLU()
        self.features.append(self.ReLU22)
        self.layerDict['ReLU22'] = self.ReLU22

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        return x

    def getLayerOutput(self, x, targetLayer):
        for layer in self.features:
            x = layer(x)
            if layer == targetLayer:
                return x

        # Should not go here
        raise Exception("Target layer not found")



class CIFAR10CNNAlternativeReLU22Archi2(nn.Module):
    def __init__(self, NChannels):
        super(CIFAR10CNNAlternativeReLU22Archi2, self).__init__()
        self.features = []
        self.layerDict = collections.OrderedDict()

        self.conv11 = nn.Conv2d(
            in_channels = NChannels,
            out_channels = 64,
            kernel_size = 5,
            padding = 2
        )
        self.features.append(self.conv11)
        self.layerDict['conv11'] = self.conv11

        self.ReLU11 = nn.ReLU()
        self.features.append(self.ReLU11)
        self.layerDict['ReLU11'] = self.ReLU11

        self.pool1 = nn.MaxPool2d(2,2)
        self.features.append(self.pool1)
        self.layerDict['pool1'] = self.pool1

        self.conv22 = nn.Conv2d(
            in_channels = 64,
            out_channels = 128,
            kernel_size = 5,
            padding = 2
        )
        self.features.append(self.conv22)
        self.layerDict['conv22'] = self.conv22

        self.ReLU22 = nn.ReLU()
        self.features.append(self.ReLU22)
        self.layerDict['ReLU22'] = self.ReLU22

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        return x

    def getLayerOutput(self, x, targetLayer):
        for layer in self.features:
            x = layer(x)
            if layer == targetLayer:
                return x

        # Should not go here
        raise Exception("Target layer not found")


class CIFAR10CNNAlternativeReLU32(nn.Module):
    def __init__(self, NChannels):
        super(CIFAR10CNNAlternativeReLU32, self).__init__()
        self.features = []
        self.layerDict = collections.OrderedDict()

        self.conv11 = nn.Conv2d(
            in_channels = NChannels,
            out_channels = 64,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv11)
        self.layerDict['conv11'] = self.conv11

        self.ReLU11 = nn.ReLU()
        self.features.append(self.ReLU11)
        self.layerDict['ReLU11'] = self.ReLU11

        self.conv12 = nn.Conv2d(
            in_channels = 64,
            out_channels = 64,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv12)
        self.layerDict['conv12'] = self.conv12

        self.ReLU12 = nn.ReLU()
        self.features.append(self.ReLU12)
        self.layerDict['ReLU12'] = self.ReLU12

        self.pool1 = nn.MaxPool2d(2,2)
        self.features.append(self.pool1)
        self.layerDict['pool1'] = self.pool1


        self.conv21 = nn.Conv2d(
            in_channels = 64,
            out_channels = 128,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv21)
        self.layerDict['conv21'] = self.conv21

        self.ReLU21 = nn.ReLU()
        self.features.append(self.ReLU21)
        self.layerDict['ReLU21'] = self.ReLU21

        self.conv22 = nn.Conv2d(
            in_channels = 128,
            out_channels = 128,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv22)
        self.layerDict['conv22'] = self.conv22

        self.ReLU22 = nn.ReLU()
        self.features.append(self.ReLU22)
        self.layerDict['ReLU22'] = self.ReLU22

        self.pool2 = nn.MaxPool2d(2,2)
        self.features.append(self.pool2)
        self.layerDict['pool2'] = self.pool2

        self.conv31 = nn.Conv2d(
            in_channels = 128,
            out_channels = 128,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv31)
        self.layerDict['conv31'] = self.conv31

        self.ReLU31 = nn.ReLU()
        self.features.append(self.ReLU31)
        self.layerDict['ReLU31'] = self.ReLU31

        self.conv32 = nn.Conv2d(
            in_channels = 128,
            out_channels = 128,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv32)
        self.layerDict['conv32'] = self.conv32


        self.ReLU32 = nn.ReLU()
        self.features.append(self.ReLU32)
        self.layerDict['ReLU32'] = self.ReLU32

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        return x

    def getLayerOutput(self, x, targetLayer):
        for layer in self.features:
            x = layer(x)
            if layer == targetLayer:
                return x

        # Should not go here
        raise Exception("Target layer not found")


class CIFAR10CNNAlternativeReLU32Archi2(nn.Module):
    def __init__(self, NChannels):
        super(CIFAR10CNNAlternativeReLU32Archi2, self).__init__()
        self.features = []
        self.layerDict = collections.OrderedDict()

        self.conv11 = nn.Conv2d(
            in_channels = NChannels,
            out_channels = 64,
            kernel_size = 5,
            padding = 2
        )
        self.features.append(self.conv11)
        self.layerDict['conv11'] = self.conv11

        self.ReLU11 = nn.ReLU()
        self.features.append(self.ReLU11)
        self.layerDict['ReLU11'] = self.ReLU11

        self.pool1 = nn.MaxPool2d(2,2)
        self.features.append(self.pool1)
        self.layerDict['pool1'] = self.pool1

        self.conv22 = nn.Conv2d(
            in_channels = 64,
            out_channels = 128,
            kernel_size = 5,
            padding = 2
        )
        self.features.append(self.conv22)
        self.layerDict['conv22'] = self.conv22

        self.ReLU22 = nn.ReLU()
        self.features.append(self.ReLU22)
        self.layerDict['ReLU22'] = self.ReLU22

        self.pool2 = nn.MaxPool2d(2,2)
        self.features.append(self.pool2)
        self.layerDict['pool2'] = self.pool2

        self.conv32 = nn.Conv2d(
            in_channels = 128,
            out_channels = 128,
            kernel_size = 5,
            padding = 2
        )
        self.features.append(self.conv32)
        self.layerDict['conv32'] = self.conv32

        self.ReLU32 = nn.ReLU()
        self.features.append(self.ReLU32)
        self.layerDict['ReLU32'] = self.ReLU32

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        return x

    def getLayerOutput(self, x, targetLayer):
        for layer in self.features:
            x = layer(x)
            if layer == targetLayer:
                return x

        # Should not go here
        raise Exception("Target layer not found")



class MLPAlternative(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(MLPAlternative, self).__init__()
        self.features = []
        self.classifier = []
        self.layerDict = collections.OrderedDict()

        assert len(inputSize) == 3
        assert len(outputSize) == 3

        self.inputSize = inputSize
        self.outputSize = outputSize

        self.input_dims = 1
        for d in inputSize:
            self.input_dims *= d

        self.output_dims = 1
        for d in outputSize:
            self.output_dims *= d

        self.fc1_N = int(0.7*self.input_dims)
        self.fc2_N = int(0.1*self.input_dims)
        self.fc3_N = self.output_dims

        print "fc1_N: ", self.fc1_N
        print "fc2_N: ", self.fc2_N
        print "fc3_N: ", self.fc3_N

        self.fc1 = nn.Linear(self.input_dims, self.fc1_N)
        self.classifier.append(self.fc1)
        self.layerDict['fc1'] = self.fc1
        self.fc1act = nn.ReLU()
        self.classifier.append(self.fc1act)
        self.layerDict['fc1act'] = self.fc1act

        self.fc2 = nn.Linear(self.fc1_N, self.fc2_N)
        self.classifier.append(self.fc2)
        self.layerDict['fc2'] = self.fc2
        self.fc2act = nn.ReLU()
        self.classifier.append(self.fc2act)
        self.layerDict['fc2act'] = self.fc2act


        self.fc3 = nn.Linear(self.fc2_N, self.fc3_N)
        self.classifier.append(self.fc3)
        self.layerDict['fc3'] = self.fc3
        self.fc3act = nn.ReLU()
        self.classifier.append(self.fc3act)
        self.layerDict['fc3act'] = self.fc3act


    def forward(self, x):
        x = x.view(-1, self.input_dims)
        for layer in self.classifier:
            x = layer(x)
        x = x.view(-1, self.outputSize[0], self.outputSize[1], self.outputSize[2])
        return x

    def getLayerOutput(self, x, targetLayer):
        x = x.view(-1, self.input_dims)
        for layer in self.classifier:
            x = layer(x)
            if layer == targetLayer:
                return x

        # Should not go here
        raise Exception("Target layer not found")


class LeNetDecoderconv1(nn.Module):
    def __init__(self, NChannels):
        super(LeNetDecoderconv1, self).__init__()
        self.decoder = []
        self.layerDict = collections.OrderedDict()

        self.conv1 = nn.Conv2d(
            in_channels = NChannels,
            out_channels = 1,
            kernel_size = 5,
            padding = (4,4)
        )
        self.layerDict['conv1'] = self.conv1

    def forward(self, x):
        for layer in self.layerDict:
            x = self.layerDict[layer](x)
        return x



class LeNetDecoderReLU2(nn.Module):
    def __init__(self, NChannels):
        super(LeNetDecoderReLU2, self).__init__()
        self.decoder = []
        self.layerDict = collections.OrderedDict()


        self.deconv1 = nn.ConvTranspose2d(
            in_channels = 16,
            out_channels = 8,
            kernel_size = 5,
        )

        self.layerDict['deconv1'] = self.deconv1

        self.ReLU1 = nn.ReLU()
        self.layerDict['ReLU1'] = self.ReLU1

        self.deconv2 = nn.ConvTranspose2d(
            in_channels = 8,
            out_channels = 1,
            kernel_size = 5,
            stride = 2,
            output_padding = (1,1)
        )
        self.layerDict['deconv2'] = self.deconv2


    def forward(self, x):
        for layer in self.layerDict:
            x = self.layerDict[layer](x)
        return x



class CIFAR10CNNDecoderconv1(nn.Module):
    def __init__(self, NChannels):
        super(CIFAR10CNNDecoderconv1, self).__init__()
        self.decoder = []
        self.layerDict = collections.OrderedDict()
        self.deconv1 = nn.ConvTranspose2d(
            in_channels = 16,
            out_channels = 8,
            kernel_size = 5,
        )

        self.layerDict['deconv1'] = self.deconv1
        self.ReLU1 = nn.ReLU()
        self.layerDict['ReLU1'] = self.ReLU1
        self.deconv2 = nn.ConvTranspose2d(
            in_channels = 8,
            out_channels = 1,
            kernel_size = 5,
            stride = 2,
            output_padding = (1,1)
        )
        self.layerDict['deconv2'] = self.deconv2
    def forward(self, x):
        for layer in self.layerDict:
            x = self.layerDict[layer](x)
        return x



class CIFAR10CNNDecoderconv1(nn.Module):
    def __init__(self, NChannels):
        super(CIFAR10CNNDecoderconv1, self).__init__()
        self.decoder = []
        self.layerDict = collections.OrderedDict()

        self.deconv11 = nn.ConvTranspose2d(
            in_channels = 64,
            out_channels = 3,
            kernel_size = 3,
            padding = 1
        )

        self.layerDict['deconv11'] = self.deconv11


    def forward(self, x):
        for layer in self.layerDict:
            x = self.layerDict[layer](x)
        return x


class CIFAR10CNNDecoderReLU22(nn.Module):
    def __init__(self, NChannels):
        super(CIFAR10CNNDecoderReLU22, self).__init__()

        self.layerDict = collections.OrderedDict()

        self.deconv11 = nn.ConvTranspose2d(
            in_channels = 128,
            out_channels = 64,
            kernel_size = 3,
            stride = 2,
            padding = 1,
            output_padding = 1
        )
        self.layerDict['deconv11'] = self.deconv11

        self.ReLU11 = nn.ReLU()
        self.layerDict['ReLU11'] = self.ReLU11

        self.deconv21 = nn.ConvTranspose2d(
            in_channels = 64,
            out_channels = 3,
            kernel_size = 3,
            padding = 1
        )

        self.layerDict['deconv21'] = self.deconv21

    def forward(self, x):
        for layer in self.layerDict:
            x = self.layerDict[layer](x)
        return x


class CIFAR10CNNDecoderReLU32(nn.Module):
    def __init__(self, NChannels):
        super(CIFAR10CNNDecoderReLU32, self).__init__()

        self.layerDict = collections.OrderedDict()

        self.deconv11 = nn.ConvTranspose2d(
            in_channels = 128,
            out_channels = 128,
            kernel_size = 3,
            stride = 2,
            padding = 1,
            output_padding = 1
        )
        self.layerDict['deconv11'] = self.deconv11

        self.ReLU11 = nn.ReLU()
        self.layerDict['ReLU11'] = self.ReLU11

        self.deconv21 = nn.ConvTranspose2d(
            in_channels = 128,
            out_channels = 64,
            kernel_size = 3,
            stride = 2,
            padding = 1,
            output_padding = 1
        )
        self.layerDict['deconv21'] = self.deconv21

        self.ReLU21 = nn.ReLU()
        self.layerDict['ReLU21'] = self.ReLU21

        self.deconv31 = nn.ConvTranspose2d(
            in_channels = 64,
            out_channels = 3,
            kernel_size = 3,
            padding = 1
        )

        self.layerDict['deconv31'] = self.deconv31

    def forward(self, x):
        for layer in self.layerDict:
            x = self.layerDict[layer](x)
        return x
