#! /bin/bash

# Random noise
Random_Noise () {
    python inverse_whitebox_MNIST_defense.py --noise_type Gaussian --layer ReLU2 --add_noise_to_input
    python inverse_whitebox_MNIST_defense.py --noise_type Gaussian --layer ReLU2
    python inverse_whitebox_MNIST_defense.py --noise_type Laplace --layer ReLU2 --add_noise_to_input
    python inverse_whitebox_MNIST_defense.py --noise_type Laplace --layer ReLU2
}

# Dropout
Dropout (){
    python inverse_whitebox_MNIST_defense.py --noise_type dropout --layer ReLU2 --add_noise_to_input
    python inverse_whitebox_MNIST_defense.py --noise_type dropout --layer ReLU2
}

Dropout_CIFAR (){
    python2 inverse_whitebox_CIFAR_defense.py --noise_type dropout --layer ReLU22 --iters 5000 --learning_rate 1e-2 --lambda_TV 1e1 --lambda_l2 0.0
    python2 inverse_whitebox_CIFAR_defense.py --noise_type dropout --layer ReLU22 --add_noise_to_input --iters 5000 --learning_rate 1e-2 --lambda_TV 1e1 --lambda_l2 0.0
    python2 inverse_whitebox_CIFAR_defense.py --noise_type dropout --layer ReLU12 --iters 5000 --learning_rate 1e-2 --lambda_TV 0.0 --lambda_l2 0.0
    python2 inverse_whitebox_CIFAR_defense.py --noise_type dropout --layer pool1 --iters 5000 --learning_rate 1e-2 --lambda_TV 0.0 --lambda_l2 0.0
    python2 inverse_whitebox_CIFAR_defense.py --noise_type dropout --layer conv22 --iters 5000 --learning_rate 1e-2 --lambda_TV 1e1 --lambda_l2 0.0
    python2 inverse_whitebox_CIFAR_defense.py --noise_type dropout --layer pool2 --iters 5000 --learning_rate 1e-2 --lambda_TV 1e1 --lambda_l2 0.0
}

# Opt noise generation
Opt_Noise_Generation (){
    python noise_generation_opt.py --noise_sourceLayer pool1 --noise_targetLayer conv2
    python inverse_whitebox_MNIST_defense.py --noise_type noise_gen_opt --layer pool1 --noise_targetLayer conv2

    python noise_generation_opt.py --noise_sourceLayer conv2 --noise_targetLayer ReLU2
    python inverse_whitebox_MNIST_defense.py --noise_type noise_gen_opt --layer conv2 --noise_targetLayer ReLU2

    python noise_generation_opt.py --noise_sourceLayer ReLU2 --noise_targetLayer pool2
    python inverse_whitebox_MNIST_defense.py --noise_type noise_gen_opt --layer ReLU2 --noise_targetLayer pool2

    python noise_generation_opt.py --noise_sourceLayer pool2 --noise_targetLayer fc1
    python inverse_whitebox_MNIST_defense.py --noise_type noise_gen_opt --layer pool2 --noise_targetLayer fc1
}

#Random_Noise
#Dropout
#Opt_Noise_Generation

#python inverse_whitebox_MNIST_defense.py --noise_type dropout --layer pool1
#python inverse_whitebox_MNIST_defense.py --noise_type dropout --layer conv2
#python inverse_whitebox_MNIST_defense.py --noise_type dropout --layer ReLU2
#python inverse_whitebox_MNIST_defense.py --noise_type dropout --layer pool2
#python inverse_whitebox_MNIST_defense.py --noise_type dropout --layer fc1

Dropout_CIFAR
