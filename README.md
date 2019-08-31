# Model Inverse Attack against Collaborative Inference

# Pre-requisite:
1. torchvision version 0.2.1:

pip install torchvision==0.2.1

python training.py --dataset CIFAR10 --epochs 50
python inverse_whitebox_CIFAR.py --iters 5000 --learning_rate 1e-2 --layer conv11 --lambda_TV 0.0 --lambda_l2 0.0
