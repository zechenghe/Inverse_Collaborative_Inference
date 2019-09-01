# Model Inverse Attack against Collaborative Inference

This code implements model inversion attacks against collaborative inference in the following paper:

Zecheng He, Tianwei Zhang and Ruby Lee, "Model Inversion Attacks Against Collaborative Inference", 35th Annual Computer Security Applications Conference (ACSAC'19), San Juan, Dec 2019


#### Pre-requisite:
#### 1.python 2.7
#### 2.numpy
pip install numpy
#### 3.pytorch
pip install torch
#### 4.torchvision version 0.2.1:
pip install torchvision==0.2.1

python training.py --dataset CIFAR10 --epochs 50
python inverse_whitebox_CIFAR.py --iters 5000 --learning_rate 1e-2 --layer conv11 --lambda_TV 0.0 --lambda_l2 0.0
