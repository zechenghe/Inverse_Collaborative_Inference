# @Author: Zecheng He
# @Date:   2020-04-20

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

from skimage.measure import compare_ssim

#####################

# This function is used to generate noise that does not affect model output

#####################

def noise_gen(args, model_dir = "checkpoints/MNIST/", model_name = "ckpt.pth"):

    NClasses = args.NClasses
    inverseClass = args.inverseClass
    assert inverseClass < NClasses

    sourceLayer = args.sourceLayer
    targetLayer = args.targetLayer
    gpu = args.gpu

    if args.dataset == 'MNIST':

        mu = torch.tensor([0.5], dtype=torch.float32)
        sigma = torch.tensor([0.5], dtype=torch.float32)
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

        trainset = torchvision.datasets.MNIST(root='./data/MNIST', train=True,
                                        download=True, transform = tsf['train'])

        testset = torchvision.datasets.MNIST(root='./data/MNIST', train=False,
                                       download=True, transform = tsf['test'])

    x_train, y_train = trainset.data, trainset.targets,
    x_test, y_test = testset.data, testset.targets,

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
    if not gpu:
        net = net.cpu()

    net.eval()

    # Only to get the feature size
    targetImg, _ = getImgByClass(inverseIter, C = inverseClass)
    deprocessImg = deprocess(targetImg.clone())

    if gpu:
        targetImg = targetImg.cuda()
        softmaxLayer = nn.Softmax().cuda()

    layer = net.layerDict[sourceLayer]
    sourceLayerOutput = net.getLayerOutput(targetImg, layer)
    xGen = torch.zeros(sourceLayerOutput.size(), requires_grad = True)
    xGen = xGen.cuda() if gpu else xGen

    refSource = torch.randn(size=xGen.size(), requires_grad = True)

    layer = net.layerDict[targetLayer]
    targetLayerOutput = net.getLayerOutput(targetImg, layer)
    refTarget = torch.zeros(targetLayerOutput.size(), requires_grad = True)

    print "xGen.size", xGen.size()
    print "refSource.size", refSource.size()
    print "refTarget.size", refTarget.size()

    targetLayerOutput = net.getLayerOutputFrom(
        x = xGen,
        sourceLayer = sourceLayer,
        targetLayer = targetLayer
    )

    print "targetLayerOutput.size", targetLayerOutput.size()

    optimizer = optim.Adam(
        params = [xGen],
        lr = args.learning_rate,
        eps = args.eps,
        amsgrad = args.AMSGrad
    )

    for i in range(NIters):

        optimizer.zero_grad()

        targetLayerOutput = net.getLayerOutputFrom(
            x = xGen,
            sourceLayer = sourceLayer,
            targetLayer = targetLayer
        )

        sourceLayerLoss = ((xGen - refSource)**2).mean()
        targetLayerLoss = ((targetLayerOutput - refTarget)**2).mean()

        totalLoss = targetLayerLoss + sourceLayerLoss * args.lambda_sourcelayer

        totalLoss.backward(retain_graph=True)
        optimizer.step()

        print "Iter ", i, "loss: ", totalLoss.cpu().detach().numpy(), \
        "sourceLayerLoss: ", sourceLayerLoss.cpu().detach().numpy(), \
        "targetLayerLoss: ", targetLayerLoss.cpu().detach().numpy()


    exit(0)

        #print "Iter ", i, "Feature loss: ", featureLoss.cpu().detach().numpy(), "TVLoss: ", TVLoss.cpu().detach().numpy(), "l2Loss: ", normLoss.cpu().detach().numpy()

    # save the final result
    imgGen = xGen.clone()
    imgGen = deprocess(imgGen)
    torchvision.utils.save_image(imgGen, save_img_dir + str(inverseClass) + '-inv.png')

    ref_img = deprocessImg.detach().cpu().numpy().squeeze()
    inv_img = imgGen.detach().cpu().numpy().squeeze()

    psnr = get_PSNR(ref_img, inv_img, peak=1.0)
    ssim = compare_ssim(ref_img, inv_img, data_range = inv_img.max() - inv_img.min(), multichannel=False)

    return psnr, ssim


if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type = str, default = 'MNIST')
        parser.add_argument('--network', type = str, default = 'LeNet')
        parser.add_argument('--iters', type = int, default = 500)
        parser.add_argument('--eps', type = float, default = 1e-3)
        parser.add_argument('--AMSGrad', type = bool, default = True)
        parser.add_argument('--learning_rate', type = float, default = 1e-2)
        parser.add_argument('--lambda_sourcelayer', type = float, default = 1e-2)
        parser.add_argument('--decrease_LR', type = int, default = 20)
        parser.add_argument('--sourceLayer', type = str, default = 'ReLU2')
        parser.add_argument('--targetLayer', type = str, default = 'fc3')
        parser.add_argument('--save_iter', type = int, default = 10)
        parser.add_argument('--inverseClass', type = int, default = 0)

        parser.add_argument('--nogpu', dest='gpu', action='store_false')
        parser.set_defaults(gpu=True)

        parser.add_argument('--novalidation', dest='validation', action='store_false')
        parser.set_defaults(validation=True)
        args = parser.parse_args()

        model_dir = "checkpoints/" + args.dataset + '/'
        model_name = "ckpt.pth"

        if args.dataset == 'MNIST':
            args.imageWidth = 28
            args.imageHeight = 28
            args.imageSize = 28*28
            args.NChannels = 1
            args.NClasses = 10

        noise_gen(
            args = args,
            model_dir = "checkpoints/MNIST/",
            model_name = "ckpt.pth"
        )


    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
