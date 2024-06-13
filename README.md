# projects-assorted:
## 1. Variational Autoencoder (VAE)
This project implements a Convolutional Variational Autoencoder (VAE) to achieve high-quality reconstruction and effective latent space disentanglement using the MNIST dataset.

## Key Features
- **Dataset:** MNIST (handwritten digits)
- **Architecture:** Convolutional layers in both encoder and decoder
  - 4 convolutional layers in the encoder
  - 4 transposed convolutional layers in the decoder
- **Latent Space:** Fully connected layers to map to and from the latent space
- **Activation Functions:** ReLU in hidden layers, Sigmoid in the output layer
- **Loss Function:** Binary Cross-Entropy Loss combined with KL Divergence

## Technical Specifications
- **Programming Language:** Python
- **Framework:** PyTorch
- **Dependencies:** numpy, torch, torchvision, matplotlib, pandas, altair
- **Hardware:** Utilizes GPU if available

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install numpy torch torchvision matplotlib pandas altair
   ```

3. Download and prepare the MNIST dataset.

## Model Training and Evaluation
- **Training:** Includes a script for training the VAE with varying beta values to balance reconstruction and latent space regularization.
- **Evaluation:** Scripts to visualize reconstruction and generated samples, and to assess latent space quality using T-SNE.

## Visualization
- **Loss Curves:** Plot total loss, reconstruction loss, and KL divergence for training and validation.
- **Sample Quality:** Display original, reconstructed, and generated samples to evaluate model performance.
- **Latent Space:** T-SNE visualization of the latent representations.

## 2. Actor-Critic RL Project
Here is an Actor-Critic agent I implemented using DRL techniques. The agent learns from a single stream of experience, updating its policy parameters after each transition in the environment.

## Key Features
- **Algorithm:** Actor-Critic
- **Environment:** `bsuite` Catch environment
- **Neural Network:** JAX-based with separate value and policy networks
- **Policy:** Softmax and epsilon-greedy
- **Optimization:** Adam optimizer with adaptive gradient rescaling

## Technical Specifications
- **Programming Language:** Python
- **Framework:** JAX
- **Dependencies:** jax, jaxlib, bsuite, matplotlib, numpy
- **Hardware:** Utilizes GPU if available

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install jax jaxlib==0.4.23 bsuite matplotlib numpy
   ```

3. Download and prepare the environment:
   ```bash
   git clone https://github.com/deepmind/bsuite.git
   pip install bsuite/
   ```

## Model Architecture
- **Hidden Layer:** 50 units with ReLU activation
- **Output Layers:**
  - **Value:** Scalar state value
  - **Policy:** Vector of action preferences


