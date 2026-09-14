# Assignment 3 — Training Neural Networks on MNIST
### DSA 8401: Applied Machine Learning · Master in Data Science and Analytics

---

## Overview

For this assignment, we use a notebook to execute and show the flow of our work.

This notebook is a complete, self-contained study of **how to build, train, diagnose, and improve fully connected Neural Networks**, and ultimately benchmark them against a classical Convolutional Neural Network, all on the MNIST handwritten-digit dataset.

The work is structured as both an academic assignment and a **personal learning journey**. Every experiment asks four questions:

> *What did I change? What happened? Why did it happen? What did I learn?*

---

## Dataset

| Split | Images | Labels | Shape |
|---|---|---|---|
| Train (raw) | 60,000 | 60,000 | 28 × 28 grayscale |
| Validation (held-out) | 5,000 | 5,000 | 28 × 28 grayscale |
| Train (effective) | 55,000 | 55,000 | 28 × 28 grayscale |
| Test (final evaluation only) | 10,000 | 10,000 | 28 × 28 grayscale |

**Source**: `MNIST_Dataset/` (local `.idx3-ubyte` binary files) — mirroring `keras.datasets.mnist.load_data()`.

> [!IMPORTANT]
> The **test set is never touched during model selection**. It is reserved exclusively for the final evaluation in Task 7 and the CNN comparison in Task 8.

---

## Learning Objectives

By completing this notebook you will be able to:

1. Explain the anatomy of a Multi-Layer Perceptron (MLP) and trace a full forward pass.
2. Understand how the training loop works: loss → gradients → weight update.
3. Diagnose classic failure modes (vanishing gradients, poor initialization, overfitting) and apply the correct fix.
4. Justify the choice of activation function, initializer, optimizer, and regularizer with empirical evidence.
5. Read and interpret training curves, confusion matrices, and per-layer gradient norms.
6. Articulate why CNNs outperform MLPs on image data (local receptive fields, weight sharing, translation invariance).
7. Save, restore, and run inference from a trained model.

---

## Notebook Structure (8 Tasks)

## Task 0 — Setup & Reproducibility *(~5 min)*
- Import all libraries (NumPy, Matplotlib, TensorFlow/Keras, Seaborn, scikit-learn).
- Print library versions for reproducibility.
- Set a global random seed (`tf.random.set_seed`, `np.random.seed`).

---

## Task 1 — Data Preparation *(~20 min)*

**What we do:**
- Load MNIST from local binary files (or `keras.datasets.mnist.load_data()`).
- Assert shapes: `(60000, 28, 28)` for images, `(60000,)` for labels.
- **Visualize**: plot a 5×10 grid of sample digits, one per class.
- **Class distribution**: bar chart confirming roughly balanced classes (~6,000 per digit).
- **Scale**: pixel values `[0, 255]` → `[0.0, 1.0]` (divide by 255).
- **Flatten**: reshape `(N, 28, 28)` → `(N, 784)` for the MLP tasks; keep `(N, 28, 28, 1)` tensors for Task 8.
- **Label encoding**: use integer labels + `sparse_categorical_crossentropy` (cleaner than one-hot for multi-class).
- **Validation split**: take the last 5,000 training examples as validation → `X_train (55000, 784)`, `X_val (5000, 784)`, `X_test (10000, 784)`.

**Why it matters (written explanation in notebook):**
- Input scaling keeps gradients in a numerically stable range — without it, activations saturate and learning stalls.
- Flattening destroys spatial structure, which is the very limitation we will expose in Task 8.

---

## Task 2 — Our Own Network Architecture *(~45 min)*

> [!IMPORTANT]
> We **design our own architecture** here. No pre-built or pre-trained model is copied. This is the core creative deliverable of the assignment.

**Experiments to run (summary table at the end):**

| Variant | Hidden Layers | Neurons/Layer | Params | Val Acc | Train Time |
|---|---|---|---|---|---|
| Baseline (1-layer wide) | 1 | 512 | ~400K | ? | ? |
| Shallow-wide | 1 | 1024 | ~800K | ? | ? |
| Medium | 2 | 256, 128 | ~220K | ? | ? |
| Deep-narrow | 4 | 128, 64, 64, 32 | ~130K | ? | ? |
| Deeper | 5 | 256, 128, 64, 32, 16 | ~225K | ? | ? |

**Architecture skeleton (all variants share):**
- **Input**: 784 units (flattened 28×28)
- **Hidden**: Dense(n, activation=...) — we vary depth & width
- **Output**: Dense(10, activation='softmax')
- **Loss**: `sparse_categorical_crossentropy`
- **Metric**: `accuracy`

**Analysis focus**: depth vs. width trade-off, parameter efficiency, overfitting signatures.

---

## Task 3 — Activation Functions & Gradients *(~40 min)*

**Experiments:**

| Config | Activation | Layers | Val Acc | Gradient Behaviour |
|---|---|---|---|---|
| A | ReLU | 3 | ? | healthy |
| B | Sigmoid | 3 | ? | mild saturation |
| C | Tanh | 3 | ? | moderate |
| D | Sigmoid | 6 | ? | vanishing gradient (proof below) |

**Evidence for vanishing gradients (Config D):**
- Plot **per-layer gradient norms** (L2 norm of `layer.weights[0]` gradients) using a `tf.GradientTape` callback or a custom callback that logs gradient norms each epoch.
- Show that norms near the input approach zero while those near the output remain large.
- Plot the training loss curve that stalls early.

**Written explanation** (required): what vanishing/exploding gradients are, why sigmoid is prone to them in deep nets, and how ReLU mitigates this.

---

## Task 4 — Weight Initialization *(~30 min)*

**Experiments:**

| Initializer | Val Acc | Convergence Speed | Notes |
|---|---|---|---|
| Zeros | ~10% | Never | Symmetry problem — all neurons learn identically |
| Random Normal (σ=0.01) | ? | slow | Small variance → weak signals |
| Xavier / Glorot Uniform | ? | fast | Keeps variance stable across layers |
| He Uniform | ? | fast | Preferred with ReLU — corrects for dead neurons |

**Written explanation**: why symmetry breaking is necessary, why zeros fail (all gradients identical → neurons don't differentiate), and the mathematical intuition behind Xavier and He formulas.

---

## Task 5 — Optimisation *(~45 min)*

**Optimizer comparison** (fixed architecture: 2 hidden layers, 256/128 neurons, ReLU, He init):

| Optimizer | LR | Batch | Val Acc | Epochs to Converge | Stability |
|---|---|---|---|---|---|
| SGD (momentum=0.9) | 0.1 | 128 | ? | ? | ? |
| SGD (momentum=0.9) | 0.01 | 128 | ? | ? | ? |
| RMSprop | 0.001 | 128 | ? | ? | ? |
| Adam | 0.001 | 32 | ? | ? | ? |
| Adam | 0.001 | 128 | ? | ? | ? |
| Adam | 0.001 | 512 | ? | ? | ? |
| Adam | 0.01 | 128 | ? | ? | ? |

**Learning-rate sweep plots**: overlay training/validation accuracy curves for each LR on the same axes.

**Written explanation**: adaptive vs. fixed LR, momentum, batch size effect on gradient noise and generalization.

---

## Task 6 — Regularization *(~30 min)*

**Experiments (same architecture as Task 5 winner):**

| Config | Regularizer | Dropout Rate | Early Stopping | Val Acc | Overfitting Gap |
|---|---|---|---|---|---|
| Baseline (no reg) | None | 0.0 | No | ? | ? |
| L2 only | L2=0.001 | 0.0 | No | ? | ? |
| Dropout only | None | 0.3 | No | ? | ? |
| Dropout + Early Stop | None | 0.3 | Yes (patience=5) | ? | ? |
| L2 + Dropout | L2=0.001 | 0.3 | Yes | ? | ? |

**Overfitting gap** = train accuracy − validation accuracy. Plot training vs. validation loss curves.

**Written discussion**: MNIST is relatively "easy" — expected overfitting gap is small. Comment on when regularization helps more (larger models, noisier data).

---

## Task 7 — Final Fully Connected Model *(~30 min)*

**Deliverables:**
- Chosen architecture documented: layer sizes, activations, initializer, optimizer, LR, batch size, regularization.
- `model.summary()` output.
- Training history plots (loss + accuracy, train vs. val).
- **Test set evaluation**:
  - Final accuracy on 10,000 test images.
  - Confusion matrix (10×10 heatmap).
  - Gallery of misclassified digits (show image, true label, predicted label).
- Save model: `model.save('best_mlp.keras')`.
- Restore model and run predictions to demonstrate it works.

---

## Task 8 — Comparison with a Classical CNN *(~45 min)*

> [!IMPORTANT]
> This task begins **only after Task 7 is completed and documented**.

**CNN Architecture (LeNet-style or simple Conv-Pool stack):**
```
Input: (28, 28, 1)
Conv2D(32, 3×3, ReLU, padding='same')
MaxPooling2D(2×2)
Conv2D(64, 3×3, ReLU, padding='same')
MaxPooling2D(2×2)
Flatten
Dense(128, ReLU)
Dropout(0.3)
Dense(10, Softmax)
```

**Side-by-side comparison table:**

| Model | Params | Epochs | Train Time | Test Acc | Best Val Acc |
|---|---|---|---|---|---|
| Our best MLP (Task 7) | ? | ? | ? | ? | ? |
| Classical CNN (Task 8) | ? | ? | ? | ? | ? |

**Error pattern analysis:**
- Show CNN confusion matrix alongside MLP confusion matrix.
- Identify digit pairs where MLP struggles more than CNN (e.g., 4↔9, 3↔8).

**Written explanation** (the conceptual payoff of the entire assignment):
1. **Local receptive fields**: each conv filter sees a small spatial patch, not the entire image — far more efficient for structured data.
2. **Weight sharing**: the same filter is applied across the whole image — reduces parameters dramatically and enforces translational equivariance.
3. **Translation invariance** (after pooling): a "7" shifted 3 pixels left is still recognized as "7".
4. **What is lost by flattening**: pixel `(i,j)` and pixel `(i,j+1)` become two completely independent inputs — all spatial adjacency is destroyed. The MLP must re-learn these relations from scratch via weights, which requires many more parameters for less robust representations.

---

## Required Deliverable Format

Per the assignment brief, the final notebook must contain:

- [x] All Python code (clean, commented, reproducible).
- [x] `model.summary()` for every model discussed.
- [x] Results, plots, and comparison tables for each experiment.
- [x] Model evaluation on the held-out test set.
- [x] Personal analysis and conclusions for each task.
- [x] Random seed declaration and library version report.

---

## File Structure

```
MNIST-Neural-Networks/
├── NeuralNetworks.ipynb          ← working notebook (our deliverable)
├── Assignment3_Training_NeuralNetworksMNISTpdf.pdf
├── Learning_Resources/
│   ├── 6.Deep_Learning_Fundamentals.pdf
│   ├── 7.CNN_ComputerVision.pdf
│   ├── Week6_Deep_Learning_Fundamentals.pdf
│   └── Week7.CNN.pdf
└── MNIST_Dataset/
    ├── train-images.idx3-ubyte   (60,000 images)
    ├── train-labels.idx1-ubyte   (60,000 labels)
    ├── t10k-images.idx3-ubyte    (10,000 images)
    └── t10k-labels.idx1-ubyte    (10,000 labels)
```

---

## Execution Order

```
Task 0 → Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6 → Task 7 → Task 8
  ↑                                                               ↑
Setup &                                                     STOP HERE.
reproducibility                                         Final model locked.
                                                        Then introduce CNN.
```

---

## Key Concepts Covered (Mapped to Course Sessions 6 & 7)

| Concept | Where it Appears |
|---|---|
| MLP anatomy & forward pass | Task 2 |
| Loss → gradients → update | Tasks 2–6 |
| Vanishing gradients (sigmoid deep nets) | Task 3 |
| ReLU & modern activations | Task 3 |
| Xavier / He initialization | Task 4 |
| Symmetry breaking | Task 4 |
| SGD with momentum | Task 5 |
| Adam & adaptive LR | Task 5 |
| Batch size effect on gradient noise | Task 5 |
| Dropout & L2 regularization | Task 6 |
| Early stopping | Task 6 |
| Confusion matrix & error analysis | Task 7 |
| Model persistence (save/restore) | Task 7 |
| Convolutional layers & pooling | Task 8 |
| Local receptive fields & weight sharing | Task 8 |
| Translation invariance | Task 8 |
| MLP vs. CNN on image data | Task 8 |

---

## Expected Outcomes

By the end of the notebook:

- Our **best MLP** should achieve ≥ **97% test accuracy** on MNIST.
- The **CNN** should achieve ≥ **99% test accuracy**, demonstrating the structural advantage of convolution for image data.
- Every design decision is backed by an experiment and a written "What I learned" paragraph.
- The notebook tells a coherent story: *start with raw pixels → understand the network → fix its failures → squeeze out performance → see what CNNs do differently*.

---

*DSA 8401 Applied Machine Learning — Strathmore University, MSc Data Science & Analytics, 2026*
