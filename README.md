# MNIST From Scratch

## Overview

MNIST From Scratch is an end-to-end Python implementation of a two-layer, fully connected feedforward neural network built entirely from scratch. The project demonstrates the core mathematical mechanics of deep learning—including forward propagation, analytical backpropagation, activation function derivation, and gradient descent optimization—using only NumPy for vector and matrix operations. By avoiding high-level frameworks like PyTorch or TensorFlow, this repository serves as a transparent educational tool for understanding the underlying algorithms that power neural network training.

---

## Features

- **Custom Activation Layer**: Implementations of rectified linear units (`ReLU`) and normalized exponential (`Softmax`) functions, along with analytical derivatives.
- **Categorical Cross-Entropy Loss**: Multi-class loss computation with an added numerical stability offset ($1e^{-9}$) to prevent log-zero errors.
- **Analytical Backpropagation**: Manual calculation and implementation of gradients ($\frac{\partial L}{\partial W}$ and $\frac{\partial L}{\partial b}$) across layers.
- **Batch Gradient Descent**: Vanilla gradient descent optimizer for parameter updates with parameterized learning rates.
- **Custom IDX Reader**: Binary parser using the Python standard library `struct` to decode raw `ubyte` MNIST dataset files directly into NumPy arrays.
- **Jupyter Notebook Experiments**: A comprehensive notebook (`notebook.ipynb`) for hyperparameter evaluation, performance plotting, confusion matrices, and error analysis.
- **CLI Training Tool**: An interactive CLI application (`train.py`) allowing users to define epochs and learning rates dynamically.

---

## Tech Stack

This project uses a minimal set of core libraries to ensure the model is built from fundamental principles:

- **Programming Language**: Python 3
- **Numerical Computation**: NumPy
- **Data Serialization**: `struct` (Standard Library)
- **Data Visualization**: Matplotlib (Jupyter Notebook only)
- **Model Evaluation**: scikit-learn (Jupyter Notebook `confusion_matrix` only)

---

## Project Structure

```text
mnist-from-scratch/
├── data/
│   ├── train-images.idx3-ubyte   # 60,000 raw training images
│   ├── train-labels.idx1-ubyte   # 60,000 raw training labels
│   ├── t10k-images.idx3-ubyte    # 10,000 raw test images
│   └── t10k-labels.idx1-ubyte    # 10,000 raw test labels
├── activations.py                # Activation functions and derivatives
├── loss.py                       # Cross-entropy loss function
├── model.py                      # NeuralNetwork class definition
├── utils.py                      # Data loader and accuracy calculation
├── train.py                      # Interactive terminal training script
├── notebook.ipynb                # Analysis and experimental validation
└── requirements.txt              # Project package requirements
```

---

## How It Works

The machine learning pipeline in this repository follows the standard lifecycle:

### 1. Input Preprocessing
Raw binary MNIST files are parsed. Pixels representing grayscale levels ($0\text{–}255$) are normalized to $[0.0, 1.0]$ by dividing by $255.0$. The $28 \times 28$ image matrices are flattened into $784$-dimensional vectors to form the input design matrix $X$ of shape $(m, 784)$, where $m$ is the batch size.

### 2. Forward Propagation
The forward pass propagates the normalized inputs through the network layers:
1. **First Hidden Layer (Linear + Activation)**:
   $$Z^{[1]} = XW^{[1]} + b^{[1]}$$
   $$A^{[1]} = \text{ReLU}(Z^{[1]})$$
2. **Output Layer (Linear + Activation)**:
   $$Z^{[2]} = A^{[1]}W^{[2]} + b^{[2]}$$
   $$A^{[2]} = \text{Softmax}(Z^{[2]})$$

### 3. Activation Functions
- **ReLU (Rectified Linear Unit)**:
   $$\text{ReLU}(z) = \max(0, z)$$
- **Softmax**:
   $$\text{Softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_{j} e^{z_j - \max(z)}}$$
   *(The maximum value is subtracted from the exponents for numerical stability)*

### 4. Loss Computation
The network optimizes parameters against the Categorical Cross-Entropy Loss:
$$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \log(A^{[2]}_{i, y_i} + 1e^{-9})$$
where $y_i$ is the target class index for sample $i$, and $1e^{-9}$ prevents numerical underflow.

### 5. Backpropagation
Using the chain rule, analytical gradients are computed backward through the layers:
- **Output Layer Gradients**:
  $$dZ^{[2]} = A^{[2]} - Y$$
  $$dW^{[2]} = \frac{1}{m} (A^{[1]})^T dZ^{[2]}$$
  $$db^{[2]} = \frac{1}{m} \sum_{i=1}^{m} dZ^{[2]}$$
- **Hidden Layer Gradients**:
  $$dA^{[1]} = dZ^{[2]} (W^{[2]})^T$$
  $$dZ^{[1]} = dA^{[1]} \odot \text{ReLU}'(Z^{[1]})$$
  $$dW^{[1]} = \frac{1}{m} X^T dZ^{[1]}$$
  $$db^{[1]} = \frac{1}{m} \sum_{i=1}^{m} dZ^{[1]}$$
where $\odot$ represents element-wise multiplication.

### 6. Parameter Updates
Weights and biases are updated in the opposite direction of the gradients scaled by the learning rate ($\alpha$):
$$W^{[1]} \leftarrow W^{[1]} - \alpha \cdot dW^{[1]}$$
$$b^{[1]} \leftarrow b^{[1]} - \alpha \cdot db^{[1]}$$
$$W^{[2]} \leftarrow W^{[2]} - \alpha \cdot dW^{[2]}$$
$$b^{[2]} \leftarrow b^{[2]} - \alpha \cdot db^{[2]}$$

---

## Model Architecture

The network consists of a simple feedforward architecture:
- **Input Layer**: $784$ features (representing the flattened $28 \times 28$ image pixels).
- **Hidden Layer**: $128$ units using $\text{ReLU}$ activation.
  - Weights $W^{[1]}$ of shape $(784, 128)$ are initialized from a normal distribution scaled by $0.01$.
  - Biases $b^{[1]}$ of shape $(1, 128)$ are initialized to zeros.
- **Output Layer**: $10$ units (representing digit classes $0\text{–}9$) using $\text{Softmax}$ activation.
  - Weights $W^{[2]}$ of shape $(128, 10)$ are initialized from a normal distribution scaled by $0.01$.
  - Biases $b^{[2]}$ of shape $(1, 10)$ are initialized to zeros.

---

## Training Process

The training loop implemented in `train.py` and `notebook.ipynb` runs full-batch gradient descent (all $60,000$ training images per iteration):
1. **Epoch Loop**: Iterates through the specified number of training runs.
2. **Metrics Logging**: Evaluates current training loss and accuracy before updating parameters.
3. **Backpropagation**: Calculates analytical gradients on the design matrix.
4. **Parameter Updates**: Applies gradient descent updates using the learning rate.
5. **Accuracy Calculation**: Computes the percentage of matches between predicted classes (`argmax` of output probability distribution) and the target labels.

---

## Results

A hyperparameter search was conducted by evaluating different combinations of training epochs and learning rates ($\alpha$) on the standard MNIST training ($60,000$ samples) and test ($10,000$ samples) splits:

| Config # | Epochs | Learning Rate ($\alpha$) | Final Loss | Training Accuracy | Test Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 10 | 0.1 | 2.2925 | 42.95% | 43.31% |
| 2 | 10 | 0.01 | 2.3011 | 14.79% | 14.91% |
| 3 | 20 | 0.1 | 2.2683 | 53.59% | 53.13% |
| 4 | 20 | 0.01 | 2.3006 | 15.36% | 15.92% |
| 5 | 50 | 0.1 | 2.0102 | 51.90% | 52.43% |
| 6 | 50 | 0.001 | 2.3021 | 10.12% | 9.71% |
| 7 | 300 | 0.1 | 0.4069 | 88.90% | 89.24% |

### Key Experimental Findings
- **Learning Rate Sensitivity**: A low learning rate (e.g., $0.001$) prevents convergence, leaving the network stuck at around $10\%$ accuracy (random guessing). Increasing epochs does not compensate for an excessively low learning rate.
- **Convergence Behavior**: Raising the learning rate to $0.1$ dramatically speeds up convergence, enabling the model to cross $89\%$ accuracy at 300 epochs.
- **Generalization**: Across all successful runs, the training and test accuracy curves remain extremely close. This indicates that the network is underfitting the training distribution due to its low capacity rather than overfitting.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd mnist-from-scratch
   ```

2. **Set up a Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure NumPy, Matplotlib, and scikit-learn are installed if requirements.txt is empty).*

4. **Verify Dataset Placement**:
   Ensure the `data/` folder contains the extracted `idx` MNIST dataset files:
   - `train-images.idx3-ubyte`
   - `train-labels.idx1-ubyte`
   - `t10k-images.idx3-ubyte`
   - `t10k-labels.idx1-ubyte`

---

## Usage

### Run Terminal Training Script
You can train the model from the CLI by executing `train.py`. The script prompts you for the number of epochs and the learning rate:
```bash
python train.py
```
*Example input:*
```text
Epochs: 100
Learning rate: 0.1
```

### Run Jupyter Notebook Analysis
Open `notebook.ipynb` in your preferred editor or launch Jupyter Lab:
```bash
jupyter lab
```
Run all cells to visualize training history, analyze incorrect predictions, and generate the final confusion matrix.

---

## Key Learnings

- **Mathematical Soundness**: Translating partial derivatives into matrix transpose operations in NumPy reinforces the relationship between vector calculus and matrix algebra in machine learning.
- **Epsilon Offsets**: Numerical stability measures like adding a small epsilon ($1e^{-9}$) to the input of $\log$ calculations are critical to preventing execution failures during numerical training.
- **Gradient Vanishing/Exploding**: Weight scale initialization directly dictates gradient magnitude; initialization parameters (like multiplying random matrices by $0.01$) prevent saturation of activation functions.

---

## Future Improvements

- **Mini-batch/Stochastic Gradient Descent**: Transitioning from full-batch gradient descent to Mini-batch SGD to speed up iteration times and introduce stochastic regularization.
- **Alternative Initialization Schemes**: Implementing Xavier/Glorot or He weight initialization rather than static scaling by $0.01$.
- **Advanced Optimization Algorithms**: Coding momentum-based updates or adaptive learning rate optimizers (e.g., Adam or RMSprop) from scratch.
- **Regularization**: Adding $L_2$ weight regularization or Dropout layers to prevent overfitting on longer training runs.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
