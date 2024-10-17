
# PySW

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Classes and Functions](#classes-and-functions)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Overview
This package implements a solver for perturbative expansions in quantum systems, specifically utilizing the Schrieffer-Wolff transformation technique. It provides classes and functions to compute effective Hamiltonians, manage quantum states, and perform transformations efficiently.

## Installation
To install the package, clone the repository and install the required dependencies. You can do this via pip:

```bash
git clone https://github.com/qcode-uni-a/PySW.git
cd PySW
pip install .
```

**Dependencies:**SW
- `numpy`
- `sympy`
- `tqdm`

## Usage
To use the solver, import the necessary classes and functions from the module. Below is a simple example to get you started.

### Example
```python
from PySW import RDSymbol, BosonOp, RDBasis, EffectiveFrame, Dagger
import sympy as sp
```

```python
omega = RDSymbol('omega', real=True, positive=True)
a = BosonOp('a')
ad = Dagger(a)

Omega_z = RDSymbol('Omega_z', real=True, positive=True)
spin = RDBasis('sigma', 2)
s0, sx, sy, sz = spin.basis

g = RDSymbol('g', order=1, real=True, positive=True)

H = omega * ad * a + sp.Rational(1,2) *  Omega_z * sz
V = g * sx * (ad + a)
```
```python
H_eff = EffectiveFrame(H, V, subspaces=[spin])
```
```python
H_eff.solve(max_order=2, full_diagonalization=True)
```
```python
H_eff.get_H()
```
```python
E0 = RDSymbol('E0', order=1, real=True)
H_drive = E0 * (ad + a)

H_eff.rotate(H_drive).expand()
```


## Classes and Functions
### EffectiveFrame
- **Constructor**: `EffectiveFrame(H, V, subspaces=None)`
  - Initializes the frame with the zeroth-order Hamiltonian and the perturbative interaction.

- **Method**: `solve(max_order=2, full_diagonalization=False, mask=None)`
  - Solves for the effective Hamiltonian up to the specified order.

- **Method**: `get_H(return_operator_form=True)`
  - Returns the effective Hamiltonian in either operator form or matrix form.

### Utility Functions

```
TO BE ADD
```

## Examples
For more detailed examples, including usage with specific quantum systems and additional functionality, refer to the `examples` directory in the repository.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss improvements or bug fixes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
