# @Author: Zecheng He
# @Date:   2019-08-31

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
# Training:
# python inverse_access_free_CIFAR.py --layer ReLU22 --iter 50 --training
#
# Testing:
# python inverse_access_free_CIFAR.py --testing --layer ReLU22 --iter 500 --learning_rate 1e-1 --lambda_TV 2e0 --lambda_l2 0.0
#####################


def trainAlternativeDNN(DATASET = 'CIFAR10', network = 'CIFAR10CNNAlternative', NEpochs = 200, imageWidth = 32,
        imageHeight = 32, imageSize = 32*32, NChannels = 3, NClasses = 10, layer = 'conv', BatchSize = 32, learningRate = 1e-3,
        NDecreaseLR = 5, eps = 1e-3, AMSGrad = True, model_dir = "checkpoints/MNIST/", model_name = "ckpt.pth", save_alternative_model_dir = "checkpoints/MNIST/",
        alternative_model_name = "LeNetAccessFree.pth", gpu = True, validation=False):

    print "DATASET: ", DATASET

    if DATASET == 'CIFAR10':
        mu = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        sigma = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        Normalize = transforms.Normalize(mu.tolist(), sigma.tolist())
        Unnormalize = transforms.Normalize((-mu / sigma).tolist(), (1.0 / sigma).tolist())
        tsf = {
            'train': transforms.Compose(
            [
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
    x_train, y_train = trainset.data, trainset.targets,
    x_test, y_test = testset.data, testset.targets,

    print "x_train.shape ", x_train.shape
    print "x_test.shape ", x_test.shape

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = BatchSize,
                                      shuffle = False, num_workers = 1)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 1000,
                                      shuffle = False, num_workers = 1)
    trainIter = iter(trainloader)
    testIter = iter(testloader)

    # Load the trained model
    net = torch.load(model_dir + model_name)
    if not gpu:
        net = net.cpu()

    net.eval()
    print "Validate the model accuracy..."

    if validation:
        accTest = evalTest(testloader, net, gpu = gpu)

    altnetDict = {
        'CIFAR10CNNAlternative':{
            'conv11': CIFAR10CNNAlternativeconv11,
            'ReLU22': CIFAR10CNNAlternativeReLU22,
            'ReLU32': CIFAR10CNNAlternativeReLU32
        }
    }
    alternativeNetFunc = altnetDict[network][layer]

    if gpu:
        alternativeNet = alternativeNetFunc(NChannels).cuda()
    else:
        alternativeNet = alternativeNetFunc(NChannels)

    print alternativeNet

    # Get dims of input/output, and construct the network
    batchX, batchY = trainIter.next()
    if gpu:
        batchX = batchX.cuda()

    edgeOutput = alternativeNet.forward(batchX)

    if gpu:
        cloudOuput = net.forward_from(edgeOutput, layer).clone()
    else:
        cloudOuput = net.forward_from(edgeOutput, layer)

    print "edgeOutput.size", edgeOutput.size()
    print "cloudOuput.size", cloudOuput.size()

    NBatch = len(trainset) / BatchSize
    if gpu:
        CrossEntropyLossLayer = nn.CrossEntropyLoss().cuda()
    else:
        CrossEntropyLossLayer = nn.CrossEntropyLoss()

    # Find the optimal config according to the hardware
    cudnn.benchmark = True
    optimizer = optim.Adam(params = alternativeNet.parameters(), lr = learningRate, eps = eps, amsgrad = AMSGrad)

    # Sanity check
    valData, valLabel = iter(testloader).next()
    if gpu:
        valData = valData.cuda()
        valLabel = valLabel.cuda()

    edgeOutput = alternativeNet.forward(valData)
    cloudOuput = net.forward_from(edgeOutput, layer).clone()
    valLoss = CrossEntropyLossLayer(cloudOuput, valLabel)

    print "Test Loss without training: ", valLoss.cpu().detach().numpy()

    for epoch in range(NEpochs):
        lossTrain = 0.0
        accTrain = 0.0
        for i in range(NBatch):
            try:
                batchX, batchY = trainIter.next()
            except StopIteration:
                trainIter = iter(trainloader)
                batchX, batchY = trainIter.next()

            if gpu:
                batchX = batchX.cuda()
                batchY = batchY.cuda()

            optimizer.zero_grad()

            edgeOutput = alternativeNet.forward(batchX).clone()
            cloudOuput = net.forward_from(edgeOutput, layer)

            featureLoss = CrossEntropyLossLayer(cloudOuput, batchY)

            totalLoss = featureLoss
            totalLoss.backward()
            optimizer.step()

            lossTrain += totalLoss.cpu().detach().numpy() / NBatch

        valData, valLabel = iter(testloader).next()
        if gpu:
            valData = valData.cuda()
            valLabel = valLabel.cuda()
        edgeOutput = alternativeNet.forward(valData)
        cloudOuput = net.forward_from(edgeOutput, layer).clone()
        valLoss = CrossEntropyLossLayer(cloudOuput, valLabel)

        if validation:
            accTestSplitModel = evalTestSplitModel(testloader, alternativeNet, net, layer, gpu = gpu)
            print "Epoch ", epoch, "Train Loss: ", lossTrain, "Test Loss: ", valLoss.cpu().detach().numpy(), "Test Accuracy", accTestSplitModel

        if (epoch + 1) % NDecreaseLR == 0:
            learningRate = learningRate / 2.0
            setLearningRate(optimizer, learningRate)

    if validation:
        accTestEnd = evalTest(testloader, net, gpu = gpu)
        if accTest != accTestEnd:
            print "Something wrong. Original model has been modified!"
            exit(1)

    if not os.path.exists(save_alternative_model_dir):
        os.makedirs(save_alternative_model_dir)
    torch.save(alternativeNet, save_alternative_model_dir + alternative_model_name)
    print "Model saved"

    newNet = torch.load(save_alternative_model_dir + alternative_model_name)
    newNet.eval()
    print "Model restore done"


def inverse(DATASET = 'CIFAR10', NIters = 500, imageWidth = 32, inverseClass = None,
        imageHeight = 32, imageSize = 32*32, NChannels = 3, NClasses = 10, layer = 'ReLU2',
        learningRate = 1e-3, NDecreaseLR = 20, eps = 1e-3, lambda_TV = 1e3, lambda_l2 = 1.0,
        AMSGrad = True, model_dir = "checkpoints/CIFAR10/", model_name = "ckpt.pth",
        alternative_model_name = "CIFAR10CNNAccessFree.pth", save_img_dir = "inverted_access_free/CIFAR10/MSE_TV/",
        saveIter = 10, gpu = True, validation=False):

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

    # Load the trained model
    net = torch.load(model_dir + model_name)
    if not gpu:
        net = net.cpu()
    net.eval()
    print "Validate the model accuracy..."

    if validation:
        accTest = evalTest(testloader, net, gpu = gpu)

    alternativeNet = torch.load(model_dir + alternative_model_name)
    if not gpu:
        alternativeNet = alternativeNet.cpu()
    alternativeNet.eval()
    print alternativeNet
    #print "Validate the alternative model..."
    batchX, batchY = iter(testloader).next()
    if gpu:
        batchX = batchX.cuda()
        batchY = batchY.cuda()

    if gpu:
        MSELossLayer = torch.nn.MSELoss().cuda()
    else:
        MSELossLayer = torch.nn.MSELoss()

    originalModelOutput = net.getLayerOutput(batchX, net.layerDict[layer]).clone()
    alternativeModelOutput = alternativeNet.forward(batchX)

    print "originalModelOutput.shape: ", originalModelOutput.shape, "alternativeModelOutput.shape: ", alternativeModelOutput.shape
    print "MSE difference on layer " + layer, MSELossLayer(originalModelOutput, alternativeModelOutput)


    targetImg, _ = getImgByClass(inverseIter, C = inverseClass)
    print "targetImg.size()", targetImg.size()

    deprocessImg = deprocess(targetImg.clone())

    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    torchvision.utils.save_image(deprocessImg, save_img_dir + str(inverseClass) + '-ref.png')

    if gpu:
        targetImg = targetImg.cuda()
    targetLayer = net.layerDict[layer]
    refFeature = net.getLayerOutput(targetImg, targetLayer)

    print "refFeature.size()", refFeature.size()

    if gpu:
        xGen = torch.zeros(targetImg.size(), requires_grad = True, device="cuda")
    else:
        xGen = torch.zeros(targetImg.size(), requires_grad = True)

    optimizer = optim.Adam(params = [xGen], lr = learningRate, eps = eps, amsgrad = AMSGrad)

    for i in range(NIters):

        optimizer.zero_grad()

        xFeature = alternativeNet.forward(xGen)
        featureLoss = ((xFeature - refFeature)**2).mean()

        TVLoss = TV(xGen)
        normLoss = l2loss(xGen)

        totalLoss = featureLoss + lambda_TV * TVLoss + lambda_l2 * normLoss

        totalLoss.backward(retain_graph=True)
        optimizer.step()

        print "Iter ", i, "Feature loss: ", featureLoss.cpu().detach().numpy(), "TVLoss: ", TVLoss.cpu().detach().numpy(), "l2Loss: ", normLoss.cpu().detach().numpy()

    # save the final result
    imgGen = xGen.clone()
    imgGen = deprocess(imgGen)
    torchvision.utils.save_image(imgGen, save_img_dir + str(inverseClass) + '-inv.png')

    print "Done"


if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type = str, default = 'CIFAR10')
        parser.add_argument('--network', type = str, default = 'CIFAR10CNNAlternative')

        parser.add_argument('--training', dest='training', action='store_true')
        parser.add_argument('--testing', dest='training', action='store_false')
        parser.set_defaults(training=False)
        parser.add_argument('--iters', type = int, default = 50)
        parser.add_argument('--eps', type = float, default = 1e-3)
        parser.add_argument('--lambda_TV', type = float, default = 1.0)
        parser.add_argument('--lambda_l2', type = float, default = 1.0)
        parser.add_argument('--AMSGrad', type = bool, default = True)
        parser.add_argument('--batch_size', type = int, default = 32)
        parser.add_argument('--learning_rate', type = float, default = 1e-3)
        parser.add_argument('--decrease_LR', type = int, default = 10)
        parser.add_argument('--layer', type = str, default = 'conv11')
        parser.add_argument('--method', type = str, default = "MSE_TV")
        parser.add_argument('--save_iter', type = int, default = 10)
        parser.add_argument('--inverseClass', type = int, default = 0)
        parser.add_argument('--altmodelname', type = str, default = "CIFAR10CNNAccessFree")

        parser.add_argument('--nogpu', dest='gpu', action='store_false')
        parser.set_defaults(gpu=True)

        parser.add_argument('--novalidation', dest='validation', action='store_false')
        parser.set_defaults(validation=True)
        args = parser.parse_args()

        model_dir = "checkpoints/" + args.dataset + '/'
        model_name = "ckpt.pth"
        alternative_model_name = args.altmodelname + args.layer + '.pth'

        save_img_dir = "inverted_access_free/" + args.dataset + '/' + args.layer + '/'

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
            trainAlternativeDNN(DATASET = args.dataset, network = 'CIFAR10CNNAlternative', NEpochs = args.iters, imageWidth = imageWidth,
            imageHeight = imageHeight, imageSize = imageSize, NChannels = NChannels, NClasses = NClasses, layer = args.layer, BatchSize = args.batch_size, learningRate = args.learning_rate,
            NDecreaseLR = args.decrease_LR, eps = args.eps, AMSGrad = True, model_dir = "checkpoints/CIFAR10/", model_name = "ckpt.pth", save_alternative_model_dir = "checkpoints/CIFAR10/",
            alternative_model_name = alternative_model_name, gpu = args.gpu, validation=args.validation)

        else:
            for c in range(NClasses):
                inverse(DATASET = args.dataset, NIters = args.iters, imageWidth = imageWidth, inverseClass = c,
                imageHeight = imageHeight, imageSize = imageSize, NChannels = NChannels, NClasses = NClasses, layer = args.layer,
                learningRate = args.learning_rate, NDecreaseLR = args.decrease_LR, eps = args.eps, lambda_TV = args.lambda_TV, lambda_l2 = args.lambda_l2,
                AMSGrad = args.AMSGrad, model_dir = model_dir, model_name = model_name, alternative_model_name = alternative_model_name,
                save_img_dir = save_img_dir, saveIter = args.save_iter, gpu = args.gpu, validation=args.validation)

    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
