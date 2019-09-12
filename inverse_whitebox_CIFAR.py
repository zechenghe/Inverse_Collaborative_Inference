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
# Useful Hyperparameters:

# CIFAR conv11
# python inverse_whitebox_CIFAR.py --iters 5000 --learning_rate 1e-2 --layer conv11 --lambda_TV 0.0 --lambda_l2 0.0

# CIFAR ReLU22
# python inverse_whitebox_CIFAR.py --iters 5000 --learning_rate 1e-2 --layer ReLU22 --lambda_TV 1e1 --lambda_l2 0.0

# CIFAR ReLU32
# python inverse_whitebox_CIFAR.py --iters 5000 --learning_rate 1e-2 --layer ReLU32 --lambda_TV 1e3 --lambda_l2 0.0

# CIFAR fc1
# python inverse_whitebox_CIFAR.py --iters 5000 --learning_rate 1e-3 --layer fc1 --lambda_TV 1e4 --lambda_l2 0.0
# python inverse_whitebox_CIFAR.py --iters 1000 --learning_rate 5e-2 --layer fc1 --lambda_TV 1e5 --lambda_l2 0.0

# CIFAR fc2
# python inverse_whitebox_CIFAR.py --iters 500 --learning_rate 1e-1 --layer fc2 --lambda_TV 5e2 --lambda_l2 1e2

# CIFAR label
# python inverse_whitebox_CIFAR.py --iters 500 --learning_rate 5e-2 --layer label --lambda_TV 5e-1 --lambda_l2 0.0
#####################

def inverse(DATASET = 'CIFAR10', network = 'CIFAR10CNN', NIters = 500, imageWidth = 32, inverseClass = None,
        imageHeight = 32, imageSize = 32*32, NChannels = 3, NClasses = 10, layer = 'conv22',
        BatchSize = 32, learningRate = 1e-3, NDecreaseLR = 20, eps = 1e-3, lambda_TV = 1e3, lambda_l2 = 1.0,
        AMSGrad = True, model_dir = "checkpoints/CIFAR10/", model_name = "ckpt.pth",
        save_img_dir = "inverted/CIFAR10/MSE_TV/", saveIter = 10, gpu = True):

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

    targetImg, _ = getImgByClass(inverseIter, C = inverseClass)
    print "targetImg.size()", targetImg.size()

    deprocessImg = deprocess(targetImg.clone())

    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    torchvision.utils.save_image(deprocessImg, save_img_dir + str(inverseClass) + '-ref.png')

    targetImg = targetImg.cuda()
    softmaxLayer = nn.Softmax().cuda()
    if layer == 'prob':
        reflogits = net.forward(targetImg)
        refFeature = softmaxLayer(reflogits)
    elif layer == 'label':
        refFeature = torch.zeros(1,NClasses)
        refFeature[0, inverseClass] = 1
    else:
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
        if layer == 'prob':
            xlogits = net.forward(xGen)
            xFeature = softmaxLayer(xlogits)
            featureLoss = ((xFeature - refFeature)**2).mean()
        elif layer == 'label':
            xlogits = net.forward(xGen)
            xFeature = softmaxLayer(xlogits)
            featureLoss = - torch.log(xFeature[0, inverseClass])
        else:
            xFeature = net.getLayerOutput(xGen, targetLayer)
            featureLoss = ((xFeature - refFeature)**2).mean()

        TVLoss = TV(xGen)
        normLoss = l2loss(xGen)

        totalLoss = featureLoss + lambda_TV * TVLoss + lambda_l2 * normLoss #- 1.0 * conv1Loss

        totalLoss.backward(retain_graph=True)
        optimizer.step()

        print "Iter ", i, "Feature loss: ", featureLoss.cpu().detach().numpy(), "TVLoss: ", TVLoss.cpu().detach().numpy(), "l2Loss: ", normLoss.cpu().detach().numpy()

    # save the final result
    imgGen = xGen.clone()
    imgGen = deprocess(imgGen)
    torchvision.utils.save_image(imgGen, save_img_dir + str(inverseClass) + '-inv.png')

    print "targetImg l1 Stat:"
    getL1Stat(net, targetImg)
    print "xGen l1 Stat:"
    getL1Stat(net, xGen)
    print "Done"


if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type = str, default = 'CIFAR10')
        parser.add_argument('--network', type = str, default = 'CIFAR10CNN')
        parser.add_argument('--iters', type = int, default = 500)
        parser.add_argument('--eps', type = float, default = 1e-3)
        parser.add_argument('--lambda_TV', type = float, default = 1.0)
        parser.add_argument('--lambda_l2', type = float, default = 1.0)
        parser.add_argument('--AMSGrad', type = bool, default = True)
        parser.add_argument('--batch_size', type = int, default = 32)
        parser.add_argument('--learning_rate', type = float, default = 1e-3)
        parser.add_argument('--decrease_LR', type = int, default = 20)
        parser.add_argument('--layer', type = str, default = 'conv22')
        parser.add_argument('--save_iter', type = int, default = 10)
        parser.add_argument('--inverseClass', type = int, default = None)

        parser.add_argument('--nogpu', dest='gpu', action='store_false')
        parser.set_defaults(gpu=True)
        args = parser.parse_args()

        model_dir = "checkpoints/" + args.dataset + '/'
        model_name = "ckpt.pth"

        save_img_dir = "inverted_whitebox/" + args.dataset + '/' + args.layer + '/'

        if args.dataset == 'MNIST':

            imageWidth = 28
            imageHeight = 28
            imageSize = imageWidth * imageHeight
            NChannels = 1
            NClasses = 10

        elif args.dataset == 'CIFAR10':

            imageWidth = 32
            imageHeight = 32
            imageSize = imageWidth * imageHeight
            NChannels = 3
            NClasses = 10

        else:
            print "No Dataset Found"
            exit()

        for c in range(NClasses):
            inverse(DATASET = args.dataset, network = args.network, NIters = args.iters, imageWidth = imageWidth, inverseClass = c,
            imageHeight = imageHeight, imageSize = imageSize, NChannels = NChannels, NClasses = NClasses, layer = args.layer,
            BatchSize = args.batch_size, learningRate = args.learning_rate, NDecreaseLR = args.decrease_LR, eps = args.eps, lambda_TV = args.lambda_TV, lambda_l2 = args.lambda_l2,
            AMSGrad = args.AMSGrad, model_dir = model_dir, model_name = model_name, save_img_dir = save_img_dir, saveIter = args.save_iter,
            gpu = args.gpu)

    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
