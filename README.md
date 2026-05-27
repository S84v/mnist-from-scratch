
# MNIST From Scratch

## Overview  
This repository shows how a tiny neural network can learn to recognise handwritten digits.  
Everything is written from the ground up in plain Python using only NumPy, so you can see *exactly* how the model works without any hidden magic from libraries like TensorFlow or PyTorch.

---

## What This Project Does  

* **Input** – 28 × 28‑pixel images of handwritten numbers (the classic MNIST dataset).  
* **Output** – The model predicts which digit (0‑9) the image represents.  
* **Goal** – Teach a computer to classify these images correctly by learning from examples.

---

## How It Works (High‑Level)

1. **Read the images** – Binary files are turned into NumPy arrays and each picture is flattened into a single line of 784 numbers (pixel values).  
2. **Feed the numbers into a tiny network** – The data passes through a hidden layer and then an output layer.  
3. **Make a guess** – The network turns the final numbers into probabilities for each digit and picks the highest one.  

All of this happens with simple matrix arithmetic that you can inspect line‑by‑line.

---

## Model Architecture  

| Layer | Size | Activation |
|------|------|------------|
| Input | 784 neurons (one per pixel) | – |
| Hidden | 128 neurons | ReLU (keeps only positive values) |
| Output | 10 neurons (one per digit) | Softmax (turns scores into probabilities) |

The network is just two linear transformations with the activations above. No convolutions, no fancy tricks—just a straightforward fully‑connected design.

---

## Training Process  

1. **Forward pass** – Send the data through the network to get predictions.  
2. **Calculate loss** – Compare predictions with the true labels using a simple “cross‑entropy” measure (think of it as how far off the guess is).  
3. **Back‑propagation** – Work backwards through the network, computing how each weight contributed to the error.  
4. **Update weights** – Move each weight a small step opposite to its error gradient (gradient descent).  

The loop repeats for many epochs, gradually reducing the loss and improving accuracy.

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
| 6 | 50 | 0.001 | 2.3021 | 10.12% |9.71% |
| 7 | 300 | 0.1 | 0.4069 | 88.90% | 89.24% |

Running the code with a learning rate of 0.1 for 300 epochs reaches about **89 %** accuracy on the test set.  
Lower learning rates struggle to learn, and the model stays near random‑guess performance (~10 %).  
These results demonstrate that even a tiny network can achieve solid performance when trained properly.

---

## Project Structure  

```
mnist-from-scratch/
├─ data/                     # Raw MNIST files (train / test)
├─ activations.py            # ReLU and Softmax functions
├─ loss.py                   # Cross‑entropy loss implementation
├─ model.py                  # NeuralNetwork class (forward & back‑prop)
├─ utils.py                  # Data loading and accuracy helper
├─ train.py                  # Command‑line training script
├─ notebook.ipynb            # Jupyter notebook for exploration
└─ requirements.txt          # Python dependencies
```

Each file focuses on a single responsibility, making it easy to follow the flow from data to prediction.

---

## How to Run  

1. **Clone the repo**  

   ```bash
   git clone <repository_url>
   cd mnist-from-scratch
   ```

2. **Create and activate a virtual environment**  

   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   ```

3. **Install dependencies**  

   ```bash
   pip install -r requirements.txt
   ```

4. **Make sure the MNIST files are in `data/`** (download them from the official site if needed).  

5. **Train the model**  

   ```bash
   python train.py
   ```

   You’ll be asked for the number of epochs and the learning rate—try `300` epochs and `0.1` as a starting point.

6. **(Optional) Explore in the notebook**  

   ```bash
   jupyter lab notebook.ipynb
   ```

   The notebook visualises loss curves, accuracy, and shows examples of correct and incorrect predictions.

---

## Key Learnings  

* **Neural networks are just math** – matrices, addition, and a few non‑linear functions.  
* **Gradient descent is the workhorse** – small updates based on error direction gradually improve the model.  
* **Learning rate matters** – too small and the model never learns; too large can cause unstable training.  
* **Even tiny models can be powerful** – with enough data and proper training, a two‑layer network can reach ~90 % accuracy on MNIST.

---