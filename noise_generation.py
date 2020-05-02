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

    sourceLayer = args.noise_sourceLayer
    targetLayer = args.noise_targetLayer
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
    targetImg, _ = getImgByClass(inverseIter, C = 0)
    deprocessImg = deprocess(targetImg.clone())

    softmaxLayer = nn.Softmax().cuda() if gpu else nn.Softmax()
    ReLULayer = nn.ReLU(False).cuda() if gpu else nn.ReLU(False)
    if gpu:
        targetImg = targetImg.cuda()

    layer = net.layerDict[sourceLayer]
    sourceLayerOutput = net.getLayerOutput(targetImg, layer)
    xGen = torch.ones(sourceLayerOutput.size(), requires_grad = True, device="cuda" if args.gpu else 'cpu')

    refSource = torch.randn(size=xGen.size(), requires_grad = True) * args.noise_level

    # If noise for relu layer, make all entries non-negtive
    if 'ReLU' in args.noise_sourceLayer:
        refSource = ReLULayer(refSource)


    layer = net.layerDict[targetLayer]
    targetLayerOutput = net.getLayerOutput(targetImg, layer)
    refTarget = torch.zeros(targetLayerOutput.size(), requires_grad = True)

    if args.gpu:
        refTarget = refTarget.cuda()
        refSource = refSource.cuda()

    #print "xGen.size", xGen.size()
    #print "refSource.size", refSource.size()
    #print "refTarget.size", refTarget.size()

    #targetLayerOutput = net.getLayerOutputFrom(
    #    x = xGen,
    #    sourceLayer = sourceLayer,
    #    targetLayer = targetLayer
    #)

    #print "targetLayerOutput.size", targetLayerOutput.size()

    optimizer = optim.Adam(
        params = [xGen],
        lr = args.noise_learning_rate,
        eps = args.noise_eps,
        amsgrad = args.noise_AMSGrad
    )

    for i in range(args.noise_iters):

        optimizer.zero_grad()

        targetLayerOutput = net.getLayerOutputFrom(
            x = ReLULayer(xGen) if 'ReLU' in args.noise_sourceLayer else xGen,
            sourceLayer = sourceLayer,
            targetLayer = targetLayer
        )

        sourceLayerLoss = (((ReLULayer(xGen) if 'ReLU' in args.noise_sourceLayer else xGen) - refSource)**2).mean()
        #sourceLayerLoss = -((ReLULayer(xGen) if 'ReLU' in args.noise_sourceLayer else xGen)**2).mean()
        #sourceLayerLoss = -torch.abs(ReLULayer(xGen) if 'ReLU' in args.noise_sourceLayer else xGen).mean()


        targetLayerLoss = ((targetLayerOutput - refTarget)**2).mean()

        totalLoss = targetLayerLoss + sourceLayerLoss * args.noise_lambda_sourcelayer

        totalLoss.backward(retain_graph=True)
        optimizer.step()

        #print "Iter ", i, "loss: ", totalLoss.cpu().detach().numpy(), \
        #"sourceLayerLoss: ", sourceLayerLoss.cpu().detach().numpy(), \
        #"targetLayerLoss: ", targetLayerLoss.cpu().detach().numpy()

    noise_gen = (ReLULayer(xGen) if 'ReLU' in args.noise_sourceLayer else xGen).detach().cpu().numpy()
    noise_dir = 'noise/' + args.dataset + '/'
    noise_file_name = args.noise_sourceLayer + '-' + args.noise_targetLayer + '-' + str(round(args.noise_level, 2))

    #print noise_gen
    #print sum(noise_gen)

    if not os.path.exists(noise_dir):
        os.makedirs(noise_dir)
    np.save(noise_dir + noise_file_name, noise_gen)

    acc = evalTestSplitModel(
            testloader, net, net,
            layer=args.noise_sourceLayer,
            gpu = args.gpu,
            noise_type = 'noise_gen',
            noise_level = args.noise_level,
            args = args
        )
    print "noise level", args.noise_level, "acc", acc

    return noise_gen


if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type = str, default = 'MNIST')
        parser.add_argument('--network', type = str, default = 'LeNet')
        parser.add_argument('--inverseClass', type = int, default = 0)

        parser.add_argument('--noise_iters', type = int, default = 500)
        parser.add_argument('--noise_eps', type = float, default = 1e-3)
        parser.add_argument('--noise_AMSGrad', type = bool, default = True)
        parser.add_argument('--noise_learning_rate', type = float, default = 1e-1)
        parser.add_argument('--noise_lambda_sourcelayer', type = float, default = 1e-1)
        parser.add_argument('--noise_decrease_LR', type = int, default = 20)
        parser.add_argument('--noise_sourceLayer', type = str, default = 'ReLU2')
        parser.add_argument('--noise_targetLayer', type = str, default = 'fc3')
        parser.add_argument('--noise_level', type = float, default = None)

        parser.add_argument('--nogpu', dest='gpu', action='store_false')
        parser.set_defaults(gpu=True)

        parser.add_argument('--novalidation', dest='validation', action='store_false')
        parser.set_defaults(validation=True)
        args = parser.parse_args()

        args.model_dir = "checkpoints/" + args.dataset + '/'
        args.model_name = "ckpt.pth"

        if args.noise_level == None:
            for nl in np.arange(0, 5, 0.5):
                args.noise_level = nl
                noise_gen(
                    args = args,
                    model_dir = args.model_dir,
                    model_name = args.model_name
                )
        else:
            noise_gen(
                args = args,
                model_dir = args.model_dir,
                model_name = args.model_name
            )

    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
