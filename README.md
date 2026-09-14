# Neural Network Architecture Study on MNIST (MLP vs. Classical CNN)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-20063B?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Strathmore University](https://img.shields.io/badge/Strathmore%20MSc-DSA%208401-003366?style=for-the-badge)](https://www.strathmore.edu/)
[![Test Accuracy](https://img.shields.io/badge/CNN%20Accuracy-99.00%25-brightgreen?style=for-the-badge)](file:///Users/leonida/Documents/code/MNIST-Neural-Networks/NeuralNetworks.ipynb)
[![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)]()

An empirical, deep-dive investigation into neural network design, optimization, weight initialization, activation dynamics, regularization, and spatial inductive biases using the MNIST handwritten digit dataset ($70,000$ images of size $28 \times 28$).

This repository presents a systematic progression from first-principles fully connected Multilayer Perceptrons (MLPs) to a classical Convolutional Neural Network (CNN) benchmark.

---

## Key Results & Benchmark Comparison

| Benchmark Dimension | Best Fully Connected MLP (Task 7) | Classical CNN Benchmark (Task 8) | Performance Impact / Delta |
|---|---|---|---|
| **Input Data Representation** | Flattened 1D Vector ($784$) | 2D Spatial Tensor ($28 \times 28 \times 1$) | Preserves 2D spatial pixel topology |
| **Model Architecture** | `784 -> 256 -> 128 -> 10` | `Conv(32) -> Pool -> Conv(64) -> Pool -> Dense(128) -> Dense(10)` | Spatial feature extraction via convolutions |
| **Trainable Parameters ($P$)** | $235,146$ parameters | $225,034$ parameters | **CNN is $4.3\%$ more parameter-efficient** |
| **Approx. Training Time** | ~2-3 seconds / epoch (~35s total) | ~30-40 seconds / epoch (~7.5 min total) | Higher FLOP count per sliding-window operation |
| **Training Accuracy** | $99.12\%$ | $99.37\%$ | $+0.25\%$ higher training fit |
| **Validation Accuracy** | $98.14\%$ | $99.18\%$ | $+1.04\%$ higher validation generalization |
| **Test Accuracy (10,000 images)** | **$97.78\%$** | **$99.00\%$** | **$+1.22\%$ absolute accuracy gain** |
| **Test Cross-Entropy Loss** | $0.0683$ | $0.0306$ | **$55.2\%$ reduction in test loss** |
| **Test Error Rate** | $2.22\%$ ($222 / 10,000$ errors) | $1.00\%$ ($100 / 10,000$ errors) | **$55.0\%$ error rate reduction** |

---

## Experimental Progression (Tasks 0 to 8)

Every experiment in the notebook follows a strict 4-question analytical framework:
> *What did I change? What happened? Why did it happen? What did I learn?*

```
  [Task 1: Data Preparation] ──> Scale [0, 1] & Partition 5k Validation Set
               │
  [Task 2: Network Depth vs Width] ──> Select 2-Layer [256, 128] Architecture
               │
  [Task 3: Activations & Gradients] ──> Prove Deep Sigmoid Vanishing Gradients; Select ReLU
               │
  [Task 4: Weight Initialization] ──> Prove Zero Symmetry Failure; Select He Uniform
               │
  [Task 5: Optimization Sweeps] ──> Sweep LR & Batch Size; Select Adam (LR=0.001, Batch=128)
               │
  [Task 6: Regularization] ──> Combine Dropout(0.2) + EarlyStopping (Gap: +0.75%)
               │
  [Task 7: Test Set Evaluation] ──> Evaluate Optimal MLP (Test Acc: 97.78%, Save/Restore)
               │
  [Task 8: Classical CNN Benchmark] ──> Conv-Pool Stack (Test Acc: 99.00%, Error Reduced by 55%)
```

---

## Task Summaries & Key Findings

### Task 1: Data Preparation & Normalization
* **Shape Confirmation**: $60,000$ training and $10,000$ test images ($28 \times 28$ grayscale pixels).
* **Class Balance**: Verified uniform distribution across digit classes (~$5,400$ to $6,700$ samples per digit).
* **Pixel Intensity Normalization**: Scaled raw uint8 inputs $[0, 255]$ to float32 $[0.0, 1.0]$ via $x_{\text{scaled}} = x / 255.0$. Prevents activation saturation and ill-conditioned gradient surfaces.
* **Validation Partition**: Reserved the last $5,000$ training examples as a validation set (`X_val`), preserving the $10,000$ test images strictly for final evaluation.

### Task 2: Architecture Design (Depth vs. Width)
* **Design Comparisons**: Evaluated 5 custom feedforward variants (1 to 4 hidden layers):
  * `1L-100` ($P = 79,510$)
  * `1L-512` ($P = 407,050$)
  * `2L-[256, 128]` ($P = 235,146$)
  * `3L-[256, 128, 64]` ($P = 243,402$)
  * `4L-narrow [128, 128, 128, 128]` ($P = 133,514$)
* **Finding**: `2L-[256, 128]` delivered optimal accuracy ($98.14\%$) while maintaining parameter efficiency compared to over-parameterized single-layer wide nets.

### Task 3: Activation Functions & Vanishing Gradients
* **Comparative Evaluation**: Tested `ReLU`, `Sigmoid`, and `Tanh` on the `2L-[256, 128]` network.
* **Mathematical Proof of Vanishing Gradients**: Constructed a 6-layer Sigmoid network and extracted per-layer gradient norms:
  * Output Layer Norm: $0.0421$
  * Layer 1 Norm: $0.000003$ (exponential decay toward input layer).
* **Explanation**: Derivative saturation ($\sigma'(z) = \sigma(z)(1 - \sigma(z)) \le 0.25$) causes gradients to vanish exponentially during backpropagation. `ReLU` ($f(x) = \max(0, x)$) maintains a constant derivative of $1.0$ for positive inputs, eliminating saturation.

### Task 4: Weight Initialization & Zero Symmetry Breaking
* **Methods Evaluated**: `Zeros`, `Random Normal`, `Glorot Uniform`, and `He Uniform`.
* **Zero Initialization Proof**: All zero weights cause identical output activations and identical backpropagated gradients ($\frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial w_j}$), locking all hidden units to learn identical features (stuck at $10.60\%$ accuracy).
* **Recommendation**: Pair `ReLU` with `He Uniform` ($W \sim U[-\sqrt{6/n_{\text{in}}}, \sqrt{6/n_{\text{in}}}]$) to preserve variance across non-symmetric ReLU layers ($98.10\%$ accuracy at Epoch 10).

### Task 5: Optimization & Hyperparameter Sweeps
* **Optimizer Sweep**: Compared `Adam`, `SGD + Momentum (0.9)`, and `RMSprop`.
* **Learning Rate & Batch Sweeps**: Tested learning rates $\eta \in \{0.1, 0.01, 0.001\}$ and mini-batch sizes $B \in \{32, 128, 512\}$.
* **Finding**: `Adam` with learning rate $\eta = 0.001$ and batch size $B = 128$ achieved the fastest and most stable convergence ($98.44\%$ validation accuracy).

### Task 6: Regularization & Overfitting Control
* **Methods Evaluated**: Baseline, L2 ($\lambda = 10^{-4}$), Dropout ($0.2$), Early Stopping (`patience=5`), and Combined Dropout + Early Stopping.
* **Finding**: Combined `Dropout(0.2)` + `EarlyStopping` narrowed the generalization gap to **$+0.75\%$**, prevented over-training, and reduced training compute by $40\%$.

### Task 7: Final Fully Connected Model Evaluation
* **Optimal MLP Architecture**: `784 -> 256 -> 128 -> 10` (ReLU, He Uniform, Adam $\eta=0.001$, Batch 128, Dropout 0.2, EarlyStopping).
* **Held-Out Test Results**: **$97.78\%$ test accuracy** (Test Loss: $0.0683$, Error Rate: $2.22\%$, $222$ misclassifications out of $10,000$).
* **Diagnostics**: Plotted a $10 \times 10$ confusion matrix and visual error grid. Saved model to `best_mnist_mlp.h5` and verified restored model predictions.

### Task 8: Benchmark Comparison with Classical CNN
* **CNN Architecture**: `Conv2D(32, 3x3)` $\to$ `MaxPool2D(2x2)` $\to$ `Conv2D(64, 3x3)` $\to$ `MaxPool2D(2x2)` $\to$ `Flatten` $\to$ `Dense(128)` $\to$ `Dropout(0.3)` $\to$ `Dense(10)`.
* **Results**: **$99.00\%$ test accuracy** ($1.00\%$ error rate, $100$ misclassifications).
* **Spatial Feature Extraction Theory**:
  1. *Local Receptive Fields ($3 \times 3$)*: Extracts local geometric features (edges, curves, loops).
  2. *Weight Sharing*: Reuses filter weights across the entire grid, yielding higher parameter efficiency ($225,034$ vs $235,146$).
  3. *Translation Invariance ($2 \times 2$ Max-Pooling)*: Retains key features despite small shifts or distortions.
  4. *Loss of Spatial Context in MLP*: Flattening 2D images to 1D vectors destroys spatial distances between adjacent rows/columns, forcing MLPs to relearn spatial connections from scratch.
* **Impact**: The CNN achieved a **$55.0\%$ error rate reduction** over the best MLP.

---

## Master Summary Table (Tasks 1 through 8)

| Task / Dimension | Variants Tested | Winner / Best Choice | Empirical Key Performance | Primary Reason for Winning |
|---|---|---|---|---|
| **Task 1: Preprocessing** | Raw ($[0,255]$) vs Scaled ($[0.0, 1.0]$) | `Min-Max Scaling` | Stable gradient flow, scale invariance | Normalizes input scale to prevent exploding gradients and ill-conditioned loss surfaces. |
| **Task 2: Architecture** | 1L-100, 1L-512, 2L-[256,128], 4L-narrow | `2L-256-128` | Validation Accuracy: **$98.14\%$** | Balances representational capacity ($P=235,146$) and execution efficiency. |
| **Task 3: Activations** | ReLU, Sigmoid, Tanh (2L & 6L deep) | `ReLU` | Validation Accuracy: **$98.12\%$** | Constant derivative ($f'(x)=1$ for $x>0$) eliminates vanishing gradients in deep layers. |
| **Task 4: Initializers** | Zeros, Random Normal, Glorot, He | `He Uniform` | Validation Accuracy: **$98.10\%$** (Epoch 10) | Matches variance requirement ($\text{Var}(W) = 2/n_{\text{in}}$) for non-symmetric ReLU activations. |
| **Task 5: Optimizers** | SGD+Momentum, RMSprop, Adam | `Adam (\eta=0.001, B=128)` | Validation Accuracy: **$98.44\%$** | Adaptive moment estimation adjusts learning rates individually per parameter. |
| **Task 6: Regularization** | Baseline, L2, Dropout, EarlyStop | `Dropout(0.2) + EarlyStop` | Generalization Gap: **$+0.75\%$** | Dropout breaks co-adaptation; Early Stopping saves $40\%$ compute budget. |
| **Task 7: Test Evaluation** | Optimal MLP on 10,000 Test Images | `Best MLP Model` | **Test Accuracy: $97.78\%$** ($2.22\%$ Error) | Strong generalizer; misclassifications stem from inherent human ambiguity. |
| **Task 8: CNN Benchmark** | Classical Conv-Pool Stack vs Best MLP | `Classical CNN` | **Test Accuracy: $99.00\%$** ($1.00\%$ Error) | Preserves 2D spatial context; reduces test error rate by **$55.0\%$** over best MLP. |

---

## Repository Structure

```
MNIST-Neural-Networks/
├── NeuralNetworks.ipynb          # Master Jupyter notebook containing code & analysis
├── best_mnist_mlp.h5             # Saved optimal fully connected MLP model artifact
├── README.md                     # Project documentation & summary report
├── Assignment3_Training_NeuralNetworksMNISTpdf.pdf # Assignment specification
└── Learning_Resources/          # Core theoretical references & lecture materials
    ├── 6.Deep_Learning_Fundamentals.pdf
    ├── 7.CNN_ComputerVision.pdf
    ├── Week6_Deep_Learning_Fundamentals.pdf
    └── Week7.CNN.pdf
```

---

## How to Run & Reproduce

### 1. Prerequisites & Environment Setup
Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/VAL-Jerono/MNIST-Neural-Networks.git
cd MNIST-Neural-Networks
python3 -m venv venv
source venv/bin/activate
pip install numpy matplotlib tensorflow scikit-learn seaborn jupyter pypdf
```

### 2. Launch Jupyter Notebook
Launch Jupyter Notebook to inspect or re-execute the experiments:

```bash
jupyter notebook NeuralNetworks.ipynb
```

---

## Author & Academic Information

* **Author**: Valerie Jerono (Reg No. 222331)
* **Program**: Master of Science in Data Science and Analytics (MSc. DSA)
* **Course**: DSA 8401: Applied Machine Learning
* **Institution**: Strathmore University, Nairobi, Kenya

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
