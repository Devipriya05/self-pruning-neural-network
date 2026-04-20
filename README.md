# Self-pruning-neural-network
Self-pruning neural network for CIFAR-10 that dynamically learns sparse connections using learnable gates and L1 regularization, demonstrating the trade-off between model accuracy and sparsity.
# Self-Pruning Neural Network

## 📌 Overview

This project implements a neural network that dynamically prunes its own weights during training using **learnable gating mechanisms**. Instead of pruning after training, the model learns which connections are unnecessary and removes them on the fly.

The model is trained on the CIFAR-10 dataset using a hybrid architecture:

* Convolutional layers for feature extraction
* Custom prunable fully connected layers for dynamic sparsification

---

## 🧠 Key Idea

Each weight in the prunable layers is multiplied by a **learnable gate**:

* Gates are computed using a sigmoid function → values between 0 and 1
* If a gate approaches **0**, the corresponding weight is effectively removed
* An **L1 regularization penalty** is applied to all gates

### Loss Function:

Total Loss = CrossEntropyLoss + λ × SparsityLoss

Where:

* SparsityLoss = sum of all gate values
* λ controls the trade-off between accuracy and sparsity

👉 L1 regularization pushes many gates toward zero, encouraging sparsity.

---

## 🏗️ Architecture

### Feature Extractor (CNN)

* Conv → BatchNorm → ReLU → MaxPool
* 3 convolutional blocks

### Pruning Layers

* PrunableLinear (custom layer with gating)
* Dropout for regularization

👉 **Pruning is applied only to fully connected layers** to preserve spatial feature extraction.

---

## ⚙️ Training Details

* Dataset: CIFAR-10
* Optimizer: Adam
* Learning Rate: 0.0005
* Scheduler: StepLR (decay every 5 epochs)
* Epochs: **30**
* Batch Size: 64
* Loss: CrossEntropy + L1 Sparsity Loss
* Label Smoothing: 0.1

---

## 📊 Results (30 Epochs)

| Lambda (λ) | Test Accuracy | Sparsity Level |
| ---------- | ------------- | -------------- |
| 1e-6       | 79.06%        | 0.64%          |
| 5e-6       | 78.87%        | 2.38%          |
| 1e-5       | 78.59%        | 2.88%          |

---

## 📈 Observations

* **Low λ (1e-6)**

  * Highest accuracy
  * Minimal pruning

* **Medium λ (5e-6)**

  * Balanced trade-off
  * Good compression with reasonable accuracy

* **High λ (1e-5)**

  * Significant pruning
  * Noticeable drop in accuracy

👉 This demonstrates the expected **sparsity–accuracy trade-off**.

---

## 📉 Gate Distribution

* Most gate values cluster near **0** → pruned connections
* Remaining gates form a separate cluster → important weights

👉 This confirms that the network successfully learns to prune itself.
**Gate Value Distribution**

The distribution of final gate values shows two clear patterns:

A large spike near 0, representing pruned or inactive connections
A separate cluster away from 0, representing important weights

This bimodal distribution confirms that the model successfully learns to distinguish between necessary and unnecessary connections, resulting in effective self-pruning behavior.

---
Why L1 on Sigmoid Gates Encourages Sparsity

Each weight in the prunable layer is controlled by a gate value obtained using a sigmoid function, which restricts values between 0 and 1. Applying an L1 penalty (sum of gate values) encourages the model to minimize these gate values.

Since L1 regularization is known to push parameters toward exact zeros, many gates shrink close to 0 during training. When a gate approaches 0, the corresponding weight contribution becomes negligible, effectively pruning that connection. This leads to a sparse network where only important weights remain active.
## 🧰 Tech Stack

* Python
* PyTorch
* Torchvision
* Matplotlib

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python main.py
```

---

## 📎 Project Structure

```
self-pruning-neural-network/
│── model.py
│── train.py
│── utils.py
│── main.py
│── requirements.txt
│── README.md
```

---

## 🌐 Colab Notebook


```
[Run on Colab](https://colab.research.google.com/drive/1LkWx6jbkf25SoJsObdhKLmEjXnPXB8nj?usp=sharing)
```

---

## 🏁 Conclusion

This project demonstrates how neural networks can **adaptively compress themselves during training** using learnable gates and L1 regularization. The approach effectively balances model performance and efficiency, making it useful for deployment in resource-constrained environments.

---
