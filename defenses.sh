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

Random_Noise
Dropout
#Opt_Noise_Generation
