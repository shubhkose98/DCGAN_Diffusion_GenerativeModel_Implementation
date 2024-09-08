# Implementation of GAN and Diffusion Models for Image Generation using CelebA Dataset
## Overview
This project focuses on generating synthetic images using the CelebA dataset by implementing two advanced generative modeling techniques: Generative Adversarial Networks (GANs) and Diffusion Models.
GANs: Introduced by Ian Goodfellow in 2014, GANs involve a game between two neural networks, the Generator and the Discriminator. The Generator creates fake images from random noise, while the Discriminator distinguishes between real and fake images. Through this adversarial process, GANs can produce realistic-looking images.
Diffusion Models: Introduced in 2020, Diffusion Models generate data by progressively applying noise and denoising the input in a series of iterations. These models have demonstrated exceptional performance in generating high-quality images by gradually improving their understanding of the data distribution.

## Implementation
### Generative Adversarial Networks (GANs)
The DCGAN architecture used in this project consists of two core components:
#### Generator:
- Generates fake images from random noise using transposed convolutional layers, batch normalization, and ReLU activations.
The final output layer uses the Tanh activation function to generate realistic images.
#### Discriminator:
- Distinguishes between real and fake images using convolutional layers, batch normalization, leaky ReLU activations, and a sigmoid output to predict the authenticity of the input.

### Diffusion Models
Diffusion models are used for progressive noise application and denoising to generate realistic images over multiple iterations. They focus on modeling the conditional probability distribution of each step.

### FID Calculation
- The Fréchet Inception Distance (FID) is used to evaluate the quality of the generated images.
- Code is provided to calculate FID by comparing the statistics of real images with generated images.

## Dataset
- The project uses the CelebA dataset, which contains over 200,000 64x64 celebrity face images.
- A custom dataset and dataloader are created to ingest this data into the model.

## Training
- Training alternates between updating the Discriminator (to maximize its accuracy) and the Generator (to fool the Discriminator).
- The Discriminator learns to classify real images with a label of 1 and fake images with a label of 0, while the Generator is trained to produce images that will be classified as real.
- The Binary Cross-Entropy Loss (BCELoss) function is used to measure the performance of both networks.

## Requirements:
- Since manually training the diffusion model would have required a level of compute not available currently, the diffusion implementation has been run using pre-existing scripts that emulate an existing pre-trained diffusion model details. These scripts are available in the [DLStudio Library](https://engineering.purdue.edu/kak/distDLS/#109).
- Please refer to the attached PDF file for further details on the outputs' implementation and visual representation and details of all the libraries and datasets required. Also, please check the imports in the .ipynb or .py file..
- Installing the DLStudio Library is necessary for this implementation; it can be found [here](https://engineering.purdue.edu/kak/distDLS/).

