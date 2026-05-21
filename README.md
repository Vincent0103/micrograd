# 🧠 micrograd — From Scratch Neural Network

> A scalar-valued autograd engine and a simple neural network library built from scratch in Python, following Andrej Karpathy's *"The spelled-out intro to neural networks and backpropagation"* tutorial on YouTube.

[![Tutorial](https://img.shields.io/badge/Tutorial-Andrej%20Karpathy-red?logo=youtube)](https://www.youtube.com/watch?v=VMj-3S1tku0)
[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter)](./neural-networks.ipynb)

---

## 📖 About

This project is a personal implementation of [micrograd](https://github.com/karpathy/micrograd), a minimalist deep learning library originally created by Andrej Karpathy. It was built by following his excellent YouTube lecture, which walks through every step of building a neural network from the ground up — starting from basic derivatives all the way to a fully working Multi-Layer Perceptron (MLP).

The goal is purely educational: understand exactly what happens under the hood when a neural network trains — how gradients flow backwards through a computation graph and how weights get updated step by step.

---

## ✨ Features

- **`Value` class** — a scalar wrapper that tracks computation history and supports automatic differentiation
- **Operator overloading** — supports `+`, `-`, `*`, `/`, `**`, and unary negation via Python magic methods
- **Activation functions** — `tanh` and `exp` implemented with correct backward passes
- **Backpropagation** — full reverse-mode autodiff via topological sort of the computation graph
- **Graphviz visualization** — renders the computation graph as an SVG for debugging and understanding
- **`Neuron`, `Layer`, `MLP` classes** — a fully functional feed-forward neural network on top of `Value`
- **Training loop** — forward pass, loss computation (MSE), `.backward()`, and gradient descent

---

## 🗂️ Project Structure

```
micrograd/
├── neural-networks.ipynb   # Main Jupyter notebook — full walkthrough
└── README.md
```

---

## 🔬 Core Concepts Implemented

### `Value` — the autograd engine

Every scalar in the network is wrapped in a `Value` object that records the operation that created it and its children in the computation graph. Calling `.backward()` triggers a topological sort and propagates gradients all the way back to the inputs via the chain rule.

```python
from graphviz import Digraph

a = Value(2.0)
b = Value(4.0)
c = a * b + a**2
c.backward()

print(a.grad)  # dc/da
print(b.grad)  # dc/db
```

Supported operations and their backward passes:

| Operation | Forward | Gradient rule |
|-----------|---------|---------------|
| `+`       | `a + b` | `grad * 1.0` for both |
| `*`       | `a * b` | `grad * b.data` / `grad * a.data` |
| `**`      | `a ** n` | `grad * n * a^(n-1)` |
| `exp`     | `e^a`   | `grad * out.data` |
| `tanh`    | `tanh(a)` | `grad * (1 - t²)` |

### Neural Network layers

Built directly on top of `Value`, the three classes mirror the PyTorch API:

```python
# A 3-input MLP with two hidden layers of 4 neurons and 1 output
m = MLP(3, [4, 4, 1])

xs = [
  [2.0,  3.0, -1.0],
  [3.0, -1.0,  0.5],
  [0.5,  1.0,  1.0],
  [1.0,  1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # target labels

# Forward pass
ypred = [m(x) for x in xs]

# Loss (MSE)
loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])

# Backward pass — computes all gradients
loss.backward()
```

### Computation graph visualization

Every intermediate operation can be rendered as a graph using Graphviz, showing data values and gradients at each node:

```python
draw_dot(loss)  # renders the full computation graph as SVG
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy matplotlib graphviz jupyter
```

> Graphviz also requires the system binary: `sudo apt install graphviz` (Linux) or `brew install graphviz` (macOS).

### Run the notebook

```bash
jupyter notebook neural-networks.ipynb
```

---

## 📚 Reference

- 📺 **Tutorial:** [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Andrej Karpathy
- 📦 **Original micrograd repo:** [github.com/karpathy/micrograd](https://github.com/karpathy/micrograd)

---

## 🙏 Credits

All credit for the original design and pedagogy goes to **Andrej Karpathy**. This repository is a personal learning exercise built by following his tutorial.
