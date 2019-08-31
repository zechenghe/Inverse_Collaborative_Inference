# @Author: zechenghe
# @Date:   2019-01-30T14:28:07-05:00
# @Last modified by:   zechenghe
# @Last modified time: 2019-02-01T14:58:22-05:00


import time
import math
import os
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from net import *
from utils import *

#####################
# Note:
# Training:
# python inverse_blackbox_decoder_CIFAR.py --layer conv11 --iter 50 --training --decodername CIFAR10CNNDecoderconv11
#
# Testing:
# python inverse_blackbox_decoder_CIFAR.py --testing --decodername CIFAR10CNNDecoderconv1 --layer conv1
#####################


def trainDecoderDNN(DATASET = 'CIFAR10', network = 'CIFAR10CNNDecoder', NEpochs = 50, imageWidth = 32,
            imageHeight = 32, imageSize = 32*32, NChannels = 3, NClasses = 10, layer = 'ReLU22', BatchSize = 32, learningRate = 1e-3,
            NDecreaseLR = 20, eps = 1e-3, AMSGrad = True, model_dir = "checkpoints/CIFAR10/", model_name = "ckpt.pth", save_decoder_dir = "checkpoints/CIFAR10/",
            decodername_name = 'CIFAR10CNNDecoderReLU22', gpu = True):

    print "DATASET: ", DATASET

    if DATASET == 'CIFAR10':
        mu = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        sigma = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        Normalize = transforms.Normalize(mu.tolist(), sigma.tolist())
        Unnormalize = transforms.Normalize((-mu / sigma).tolist(), (1.0 / sigma).tolist())
        tsf = {
            'train': transforms.Compose(
            [
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomAffine(degrees = 10, translate = [0.1, 0.1], scale = [0.9, 1.1]),
            transforms.ToTensor(),
            Normalize
            ]),
            'test': transforms.Compose(
            [
            transforms.ToTensor(),
            Normalize
            ])
        }
        trainset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train = True,
                                        download=True, transform = tsf['train'])
        testset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train = False,
                                       download=True, transform = tsf['test'])

    else:
        print "Dataset unsupported"
        exit(1)

    print "len(trainset) ", len(trainset)
    print "len(testset) ", len(testset)
    x_train, y_train = trainset.train_data, trainset.train_labels,
    x_test, y_test = testset.test_data, testset.test_labels,

    print "x_train.shape ", x_train.shape
    print "x_test.shape ", x_test.shape

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = BatchSize,
                                      shuffle = False, num_workers = 1)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 1000,
                                      shuffle = False, num_workers = 1)
    trainIter = iter(trainloader)
    testIter = iter(testloader)

    net = torch.load(model_dir + model_name)
    net.eval()
    print "Validate the model accuracy..."
    accTest = evalTest(testloader, net, gpu = gpu)

    # Get dims of input/output, and construct the network
    batchX, batchY = trainIter.next()
    batchX = batchX.cuda()
    originalModelOutput = net.getLayerOutput(batchX, net.layerDict[layer]).clone()

    decoderNetDict = {
        'CIFAR10CNNDecoder':{
            'conv11': CIFAR10CNNDecoderconv1,
            'ReLU22': CIFAR10CNNDecoderReLU22,
            'ReLU32': CIFAR10CNNDecoderReLU32
        }
    }
    decoderNetFunc = decoderNetDict[network][layer]
    decoderNet = decoderNetFunc(originalModelOutput.shape[1]).cuda()

    print decoderNet

    #alternativeNet = MLPAlternative(batchX.size()[1:], originalModelOutput.size()[1:]).cuda()

    NBatch = len(trainset) / BatchSize
    MSELossLayer = torch.nn.MSELoss().cuda()
    # Find the optimal config according to the hardware
    cudnn.benchmark = True
    optimizer = optim.Adam(params = decoderNet.parameters(), lr = learningRate, eps = eps, amsgrad = AMSGrad)

    #exit(0)

    for epoch in range(NEpochs):
        lossTrain = 0.0
        accTrain = 0.0
        for i in range(NBatch):
            try:
                batchX, batchY = trainIter.next()
            except StopIteration:
                trainIter = iter(trainloader)
                batchX, batchY = trainIter.next()

            batchX = batchX.cuda()
            batchY = batchY.cuda()

            optimizer.zero_grad()

            originalModelOutput = net.getLayerOutput(batchX, net.layerDict[layer]).clone()
            decoderNetOutput = decoderNet.forward(originalModelOutput)

            #print "originalModelOutput.size()", originalModelOutput.size()
            #print "decoderNetOutput.size()", decoderNetOutput.size()

            assert batchX.cpu().detach().numpy().shape == decoderNetOutput.cpu().detach().numpy().shape
            #print "batchX.shape ", batchX.cpu().detach().numpy().shape
            #print "decoderNetOutput.shape ", decoderNetOutput.cpu().detach().numpy().shape

            featureLoss = MSELossLayer(batchX, decoderNetOutput)
            totalLoss = featureLoss
            totalLoss.backward()
            optimizer.step()

            lossTrain += totalLoss / NBatch

        valData, valLabel = iter(testloader).next()
        valData = valData.cuda()
        valLabel = valLabel.cuda()
        originalModelOutput = net.getLayerOutput(valData, net.layerDict[layer]).clone()
        decoderNetOutput = decoderNet.forward(originalModelOutput)
        valLoss = MSELossLayer(valData, decoderNetOutput)

        print "Epoch ", epoch, "Train Loss: ", lossTrain.cpu().detach().numpy(), "Test Loss: ", valLoss.cpu().detach().numpy()
        if (epoch + 1) % NDecreaseLR == 0:
            learningRate = learningRate / 2.0
            setLearningRate(optimizer, learningRate)

    accTestEnd = evalTest(testloader, net, gpu = gpu)
    if accTest != accTestEnd:
        print "Something wrong. Original model has been modified!"
        exit(1)

    if not os.path.exists(save_decoder_dir):
        os.makedirs(save_decoder_dir)
    torch.save(decoderNet, save_decoder_dir + decodername_name)
    print "Model saved"

    newNet = torch.load(save_decoder_dir + decodername_name)
    newNet.eval()
    print "Model restore done"


def inverse(DATASET = 'CIFAR10', imageWidth = 32, inverseClass = None, imageHeight = 32,
        imageSize = 32*32, NChannels = 3, NClasses = 10, layer = 'conv11',
        model_dir = "checkpoints/CIFAR10/", model_name = "ckpt.pth", decoder_name = "CIFAR10CNNDecoderconv11.pth",
        save_img_dir = "inverted_blackbox_decoder/CIFAR10/", gpu = True):

    print "DATASET: ", DATASET
    print "inverseClass: ", inverseClass

    assert inverseClass < NClasses

    if DATASET == 'CIFAR10':
        mu = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        sigma = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        Normalize = transforms.Normalize(mu.tolist(), sigma.tolist())
        Unnormalize = transforms.Normalize((-mu / sigma).tolist(), (1.0 / sigma).tolist())
        tsf = {
            'train': transforms.Compose(
            [
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomAffine(degrees = 10, translate = [0.1, 0.1], scale = [0.9, 1.1]),
            transforms.ToTensor(),
            Normalize
            ]),
            'test': transforms.Compose(
            [
            transforms.ToTensor(),
            Normalize
            ])
        }
        trainset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train = True,
                                        download=True, transform = tsf['train'])
        testset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train = False,
                                       download=True, transform = tsf['test'])

    else:
        print "Dataset unsupported"
        exit(1)


    print "len(trainset) ", len(trainset)
    print "len(testset) ", len(testset)
    x_train, y_train = trainset.train_data, trainset.train_labels,
    x_test, y_test = testset.test_data, testset.test_labels,

    print "x_train.shape ", x_train.shape
    print "x_test.shape ", x_test.shape


    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 1,
                                      shuffle = False, num_workers = 1)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 1000,
                                      shuffle = False, num_workers = 1)
    inverseloader = torch.utils.data.DataLoader(testset, batch_size = 1,
                                      shuffle = False, num_workers = 1)
    trainIter = iter(trainloader)
    testIter = iter(testloader)
    inverseIter = iter(inverseloader)


    net = torch.load(model_dir + model_name)
    net.eval()
    print "Validate the model accuracy..."
    accTest = evalTest(testloader, net, gpu = gpu)

    decoderNet = torch.load(model_dir + decoder_name)
    decoderNet.eval()
    print decoderNet
    print "Validate the alternative model..."
    batchX, batchY = iter(testloader).next()
    batchX = batchX.cuda()
    batchY = batchY.cuda()

    print "batchX.shape ", batchX.cpu().detach().numpy().shape

    MSELossLayer = torch.nn.MSELoss().cuda()
    originalModelOutput = net.getLayerOutput(batchX, net.layerDict[layer]).clone()
    decoderNetOutput = decoderNet.forward(originalModelOutput)

    assert batchX.cpu().detach().numpy().shape == decoderNetOutput.cpu().detach().numpy().shape
    print "decoderNetOutput.shape ", decoderNetOutput.cpu().detach().numpy().shape
    print "MSE ", MSELossLayer(batchX, decoderNetOutput).cpu().detach().numpy()

    targetImg, _ = getImgByClass(inverseIter, C = inverseClass)
    print "targetImg.size()", targetImg.size()

    deprocessImg = deprocess(targetImg.clone())

    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    torchvision.utils.save_image(deprocessImg, save_img_dir + str(inverseClass) + '-ref.png')


    targetImg = targetImg.cuda()
    targetLayer = net.layerDict[layer]
    refFeature = net.getLayerOutput(targetImg, targetLayer)

    print "refFeature.size()", refFeature.size()
    #print "refFeature: ", refFeature.cpu().detach().numpy()

    xGen = decoderNet.forward(refFeature)
    print "MSE ", MSELossLayer(targetImg, xGen).cpu().detach().numpy()

    #refFeature = torch.zeros([1,10], device = 'cuda')
    #refFeature[0,0] = 20


#    print "Iter ", i, "Feature loss: ", featureLoss.cpu().detach().numpy(), "TVLoss: ", TVLoss.cpu().detach().numpy(), "l2Loss: ", normLoss.cpu().detach().numpy()
#    if (i+1) % saveIter == 0:
#        if not os.path.exists(save_img_dir):
#            os.makedirs(save_img_dir)
#        imgGen = xGen.clone()
#        imgGen = deprocess(imgGen)
#        torchvision.utils.save_image(imgGen, save_img_dir + 'xGen' + str(i+1) + '.png')
#        print "Max: ", np.max(imgGen.cpu().detach().numpy()), "Min ", np.min(imgGen.cpu().detach().numpy())

    # save the final result
    imgGen = xGen.clone()
    imgGen = deprocess(imgGen)
    torchvision.utils.save_image(imgGen, save_img_dir + str(inverseClass) + '-inv.png')

#    print "targetImg l1 Stat:"
#    getL1Stat(net, targetImg)
#    print "xGen l1 Stat:"
#    getL1Stat(net, xGen)

    print "Done"


if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type = str, default = 'CIFAR10')
        parser.add_argument('--network', type = str, default = 'CIFAR10CNNDecoder')

        parser.add_argument('--training', dest='training', action='store_true')
        parser.add_argument('--testing', dest='training', action='store_false')
        parser.set_defaults(training=False)

        #parser.add_argument('--training', type = bool, default = False)
        parser.add_argument('--iters', type = int, default = 500)
        parser.add_argument('--eps', type = float, default = 1e-3)
        parser.add_argument('--AMSGrad', type = bool, default = True)
        parser.add_argument('--batch_size', type = int, default = 32)
        parser.add_argument('--learning_rate', type = float, default = 1e-3)
        parser.add_argument('--decrease_LR', type = int, default = 10)
        parser.add_argument('--gpu', type = bool, default = True)
        parser.add_argument('--layer', type = str, default = 'ReLU22')
        parser.add_argument('--save_iter', type = int, default = 10)
        parser.add_argument('--inverseClass', type = int, default = 0)
        parser.add_argument('--decodername', type = str, default = "CIFAR10CNNDecoderReLU22")
        args = parser.parse_args()

        model_dir = "checkpoints/" + args.dataset + '/'
        model_name = "ckpt.pth"
        decoder_name = args.decodername + '.pth'

        save_img_dir = "inverted_blackbox_decoder/" + args.dataset + '/'

        if args.dataset == 'CIFAR10':

            imageWidth = 32
            imageHeight = 32
            imageSize = imageWidth * imageHeight
            NChannels = 3
            NClasses = 10

        else:
            print "No Dataset Found"
            exit()

        if args.training:
            trainDecoderDNN(DATASET = args.dataset, network = 'CIFAR10CNNDecoder', NEpochs = args.iters, imageWidth = imageWidth,
            imageHeight = imageHeight, imageSize = imageSize, NChannels = NChannels, NClasses = NClasses, layer = args.layer, BatchSize = args.batch_size, learningRate = args.learning_rate,
            NDecreaseLR = args.decrease_LR, eps = args.eps, AMSGrad = True, model_dir = "checkpoints/CIFAR10/", model_name = "ckpt.pth", save_decoder_dir = "checkpoints/CIFAR10/",
            decodername_name = decoder_name, gpu = True)

        else:
            for c in range(NClasses):
                inverse(DATASET = args.dataset, imageHeight = imageHeight, imageWidth = imageWidth, inverseClass = c,
                imageSize = imageSize, NChannels = NChannels, NClasses = NClasses, layer = args.layer,
                model_dir = model_dir, model_name = model_name, decoder_name = decoder_name,
                save_img_dir = save_img_dir, gpu = args.gpu)

    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
