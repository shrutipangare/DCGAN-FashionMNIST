# DCGAN Fashion-MNIST Generator

A Deep Convolutional Generative Adversarial Network (DCGAN) implementation for generating  clothing images using the Fashion-MNIST dataset.

## DCGAN

DCGAN (Deep Convolutional Generative Adversarial Network) is an extension of the original GAN architecture that uses convolutional layers instead of fully connected layers. Introduced by Radford et al. in 2015, DCGAN has become one of the most successful and widely used GAN architectures for image generation.

### Key DCGAN Principles:
- **Replace pooling layers** with strided convolutions (discriminator) and fractional-strided convolutions (generator)
- **Use batch normalization** in both generator and discriminator
- **Remove fully connected hidden layers** for deeper architectures
- **Use ReLU activation** in generator for all layers except output (which uses Tanh)
- **Use LeakyReLU activation** in the discriminator for all layers

### How GANs Work:
GANs consist of two neural networks competing against each other:
- **Generator**: Creates fake images from random noise, trying to fool the discriminator
- **Discriminator**: Distinguishes between real and generated images, trying to detect fakes

This adversarial training process results in a generator that can create highly realistic synthetic images.

## Overview

This project implements a DCGAN to generate synthetic fashion items including shirts, pants, shoes, bags, and other clothing accessories. The model learns to generate 28×28 grayscale images that resemble items from the Fashion-MNIST dataset through adversarial training between a generator and discriminator network.

## Architecture

Following the DCGAN guidelines, this implementation uses:

### Generator
- **Input**: 100-dimensional random noise vector
- **Architecture**: Dense → Reshape → 3 Transposed Convolutions
- **Output**: 28×28×1 grayscale images
- **DCGAN Features**:
  - Fractional-strided convolutions (Conv2DTranspose) for upsampling
  - Batch normalization after each layer (except output)
  - LeakyReLU activation (α=0.3)
  - Tanh activation for final output (maps to [-1,1] range)
  - No fully connected layers after initial dense layer

### Discriminator
- **Input**: 28×28×1 grayscale images
- **Architecture**: 2 Convolutional layers → Dense
- **Output**: Binary classification (real/fake)
- **DCGAN Features**:
  - Strided convolutions instead of pooling for downsampling
  - LeakyReLU activation (α=0.3) throughout
  - Dropout (0.3) for regularization
  - No batch normalization in discriminator (common DCGAN practice)

## Training Details

The training follows standard GAN adversarial training:

- **Dataset**: Fashion-MNIST (60,000 training images)
- **Batch Size**: 64
- **Epochs**: 50
- **Optimizer**: Adam (learning rate: 1e-4) for both networks
- **Loss Function**: Binary Cross-Entropy
- **Data Normalization**: Scaled to [-1, 1] range (matching generator's tanh output)
- **Adversarial Training**: Generator tries to fool discriminator, discriminator learns to detect fakes

## Requirements

```
tensorflow>=2.0
matplotlib
numpy
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dcgan-fashion-mnist.git
cd dcgan-fashion-mnist
```

2. Install dependencies:
```bash
pip install tensorflow matplotlib numpy
```

## Usage

Run the main script to train the DCGAN:

```python
python dcgan_fashion_mnist.py
```

The script will:
1. Load and preprocess the Fashion-MNIST dataset
2. Build the generator and discriminator models
3. Train the models for 50 epochs
4. Display generated images at epochs 10, 30, and 50
5. Plot training loss curves

## Results

### Training Progress

The model shows clear improvement over training epochs:

- **Epoch 10**: Basic shapes and patterns emerge
- **Epoch 30**: Recognizable clothing items with better structure
- **Epoch 50**: High-quality synthetic fashion items

### Loss Curves

- **Generator Loss**: Starts high (~1.5) and stabilizes around 0.9
- **Discriminator Loss**: Converges to ~1.25 showing balanced training

## Sample Generated Images

The final model generates diverse clothing items including:
- T-shirts and tops
- Pants and trousers
- Shoes and boots
- Bags and accessories
- Coats and jackets



## Model Architecture Details

### Generator Network
```
Dense(7×7×256) → BatchNorm → LeakyReLU
Reshape(7, 7, 256)
Conv2DTranspose(128, 5×5, stride=1) → BatchNorm → LeakyReLU
Conv2DTranspose(64, 5×5, stride=2) → BatchNorm → LeakyReLU
Conv2DTranspose(1, 5×5, stride=2) → Tanh
```

### Discriminator Network
```
Conv2D(64, 5×5, stride=2) → LeakyReLU → Dropout(0.3)
Conv2D(128, 5×5, stride=2) → LeakyReLU → Dropout(0.3)
Flatten → Dense(1)
```

## Training Tips

- Model uses fixed random seed for reproducible results
- Images are displayed every 10 epochs to monitor progress
- Loss curves help identify training stability
- Batch size of 64 provides good balance between speed and stability

## Author

Shruti Tulshidas Pangare (stp8232)

## License

This project is open source and available under the MIT License.
