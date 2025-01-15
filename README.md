# SymPT: Symbolic Perturbation Theory Toolbox

Welcome to **SymPT**, a Python package designed for performing symbolic perturbative transformations on quantum systems. SymPT leverages the **Schrieffer-Wolff Transformation (SWT)** and other related techniques to compute effective Hamiltonians in a systematic and efficient manner. The library offers a suite of tools for block-diagonalization, full-diagonalization, and custom transformations, supporting both time-independent and time-dependent systems.

This document provides comprehensive guidance on using SymPT, detailing its key features, algorithms, and implementation. Let’s dive in!

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Installation](#installation)
4. [Usage](#usage)
   - [Basic Workflow](#basic-workflow)
   - [Examples](#examples)
5. [Core Classes](#core-classes)
   - [RDSymbol](#rdsymbol)
   - [RDBasis](#rdbasis)
   - [EffectiveFrame](#effectiveframe)
   - [Block](#block)
6. [Algorithms](#algorithms)
   - [Schrieffer-Wolff Transformation (SWT)](#schrieffer-wolff-transformation-swt)
   - [Full-Diagonalization (FD)](#full-diagonalization-fd)
   - [Arbitrary Coupling Elimination (ACE)](#arbitrary-coupling-elimination-ace)
7. [Advanced Tools](#advanced-tools)
8. [Contributing](#contributing)
9. [License](#license)

---

## Overview

SymPT is a symbolic perturbative transformation tool built to help researchers and engineers study quantum systems experiencing perturbative interactions. It computes effective Hamiltonians and performs other transformations using techniques based on SWT, FD, and ACE. The package also supports multi-block diagonalizations, with special emphasis on least-action methods.

---

## Key Features

- **Multiple Transformation Methods**: Includes routines for SWT, FD, ACE.
- **Symbolic Computation**: Powered by `sympy`, enabling exact symbolic results for quantum systems.
- **Customizable Input**: Define Hamiltonians and operators with ease using symbolic expressions.
- **Flexible Output**: Obtain results in operator, matrix, or dictionary forms.
- **Efficient Algorithms**: Leverages caching and optimized partitioning for nested commutator calculations.

---

## Installation

### From PyPI
Make sure to install SymPT on a new python environment. To do so, you can use the Anaconda distribution platform
```bash
conda create -n sympt python
```

SymPT is available on PyPI for easy installation:

```bash
conda activate sympt
pip install sympt
```

### From Source
To install SymPT from the source code, clone the repository and install dependencies:

```bash
git clone https://github.com/qcode-uni-a/sympt.git
cd SymPT
conda activate sympt
pip install .
```

SymPT depends on Python 3.8+ and the following libraries:
- `sympy`: For symbolic computations.
- `numpy`: For matrix operations.
- (Optional) `matplotlib`: For visualizing results.

---

## Usage

### Basic Workflow

1. **Define Symbols and Basis**: Use `RDSymbol` and `RDBasis` to construct system components.
2. **Set Up EffectiveFrame**: Initialize the transformation frame with the Hamiltonian.
3. **Solve the Transformation**: Specify the perturbation order and method (`SWT`, `FD`, or `ACE`).
4. **Extract Results**: Retrieve the effective Hamiltonian or rotate operators into the new frame.

---

### Examples
Here we include two example cases of how to use SymPT. Find more within the Examples folder of this project.
#### Example 1: Schrieffer-Wolff Transformation (SWT)

```python
from sympt import RDSymbol, RDBasis, EffectiveFrame

# Define symbols
omega = RDSymbol('omega', real=True, positive=True)
g = RDSymbol('g', order=1, real=True)

# Define basis and Hamiltonian
spin = RDBasis(name='sigma', dim=2)
s0, sx, sy, sz = spin.basis

H = omega * sz
V = g * sx

# Setup EffectiveFrame
eff_frame = EffectiveFrame(H, V, subspaces=[spin])
eff_frame.solve(max_order=2, method="SW")
H_eff = eff_frame.get_H(return_form="operator")
display(H_eff)
```

#### Example 2: Arbitrary Coupling Elimination (ACE)

```python
from sympt import *
from sympy import Rational

# Define symbols
omega = RDSymbol('omega', order=0, real=True)
omega_z = RDSymbol('omega_z', order=0, real=True)
g = RDSymbol('g', order=1, real=True)

# Define basis and operators
spin = RDBasis(name='sigma', dim=2)
s0, sx, sy, sz = spin.basis
a = BosonOp('a')
ad = Dagger(a)

# Define Hamiltonian
H = omega * ad * a +  Rational(1,2) * omega_z * sz + g * sx * (a + ad)
mask = Block(fin = sx, inf = a, subspaces=[spin]) # this mask is equivalent to "SW" up to second order

# Solve ACE transformation
eff_frame = EffectiveFrame(H, subspaces=[spin])
eff_frame.solve(max_order=2, method="ACE", mask=mask)
H_eff = eff_frame.get_H(return_form="operator")
display(H_eff)
```

---

## Core Classes

### RDSymbol

- Represents scalar, commutative quantities with perturbative orders.
- Example: `omega = RDSymbol('omega', order=0, real=True)`

### RDBasis

- Encodes finite-dimensional subspaces and generates basis operators (e.g., Pauli or Gell-Mann matrices).
- Example: `spin = RDBasis(name='sigma', dim=2)`

### EffectiveFrame

- Central class for setting up and solving perturbative transformations.
- Methods:
  - `.solve()`: Perform the transformation.
  - `.get_H()`: Retrieve the effective Hamiltonian.
  - `.rotate()`: Rotate operators into the new frame.

### Block

- Used in ACE transformations to specify couplings to eliminate.
- Example: `mask = Block(fin=a)`

---

## Algorithms

### Schrieffer-Wolff Transformation (SWT)

Systematically block-diagonalizes the Hamiltonian, focusing on separating diagonal and off-diagonal terms.

### Full-Diagonalization (FD)

Fully diagonalizes the Hamiltonian, eliminating all off-diagonal elements.

### Arbitrary Coupling Elimination (ACE)

Targets specific off-diagonal couplings for elimination, allowing flexible transformations.

---

## Advanced Tools

- **`display_dict`**: Enhanced dictionary printing for Hamiltonians.
- **`group_by_operators`**: Groups terms by their operator components.
- **`get_block_mask`**: Simplifies block-off diagonal mask creation.

---

## Contributing

We welcome contributions! Please:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request with detailed descriptions.

---

## License

SymPT is licensed under the MIT License. See `LICENSE` for details.
