
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

### Installing from pypi

It is recommended to create a new virtual environment to install PySW
```bash
python -m venv .env
source .env/bin/activate
pip install pysw
```

### Installing from the repository

To install the package, clone the repository and install the required dependencies. This is recommended for developers that want to modify the source code. You can do this via pip:

```bash
git clone https://github.com/qcode-uni-a/PySW.git
cd PySW
pip install -e .
```

## Usage
To use the solver, import the necessary classes and functions from the module. Below is a simple example to get you started.

### Example: Rabi Model

#### Overview
The Rabi model describes the interaction between a two-level quantum system (such as an atom) and a single mode of a quantized electromagnetic field. It is a cornerstone in the study of light-matter interactions and quantum optics.

#### Hamiltonian of the Rabi Model

The Hamiltonian of the Rabi model is given by:

$$
H = H_0 + V
$$

Where:

- $H_0 = \omega a^\dagger a + \frac{ \Omega_z}{2} \sigma_z$
- $V = g \sigma_x (a + a^\dagger)$

- $\Omega_z$: frequency of the two-level system.
- $\omega$: frequency of the field mode.
- $g$: coupling strength between the two-level system and the field, $g\ll\omega - \Omega_z$.
- $\sigma_z$ and $\sigma_x$: Pauli matrices.
- $a^\dagger$ and $a$: creation and annihilation operators for the electromagnetic field mode.

#### Example Code

```python
from pysw import RDSymbol, BosonOp, Dagger, RDBasis, EffectiveFrame
import sympy as sp

# Define symbols and operators
omega = RDSymbol('omega', real=True, positive=True)         # frequency of the oscillator. Perturbation order 0
a = BosonOp('a')                                            # annihilation operator (boson)
ad = Dagger(a)                                              # creation operator (boson)

Omega_z = RDSymbol('Omega_z', real=True, positive=True)     # Zeeman splitting. Perturbation order 0
spin = RDBasis('sigma', 2)                                  # spin 1/2 basis (finite)
s0, sx, sy, sz = spin.basis                                 # spin operators

g = RDSymbol('g', order=1, real=True, positive=True)        # coupling constant. Perturbation order 1

# Define Hamiltonians
H = omega * ad * a + sp.Rational(1,2) *  Omega_z * sz       # unperturbed Hamiltonian
V = g * sx * (ad + a)                                       # perturbation

display(H)
display(V)
```
$\displaystyle \frac{\Omega_{z} \sigma_{3}}{2} + \omega {{a}^\dagger} {a}$
$\displaystyle g \sigma_{1} \left({{a}^\dagger} + {a}\right)$

```python
# Define the effective Hamiltonian
H_eff = EffectiveFrame(H, V, subspaces=[spin])
```
```plain
    The EffectiveFrame object has been initialized successfully.
    
    Effective Frame
    
    ╭────────┬─────────┬─────────────╮
    │  Name  │  Type   │  Dimension  │
    ├────────┼─────────┼─────────────┤
    │ sigma  │ Finite  │     2x2     │
    ├────────┼─────────┼─────────────┤
    │   a    │ Bosonic │      ∞      │
    ╰────────┴─────────┴─────────────╯
    
    Effective Hamiltonian: 	Not computed yet. To do so, run `solve` method. 
```

#### Schrieffer-Wolff transformation

```python
# Compute the effective Hamiltonian up to second order
H_eff.solve(max_order=2)
```

    Solving for each order: 100%|██████████| 2/2 [00:00<00:00, 26.93it/s]

    The Hamiltonian has been solved successfully. Please use the get_H method to get the result in the desired form.


```python
print('Effective Hamiltonian (Operator Form):')
H_eff.get_H()
```
    Effective Hamiltonian (Operator Form):

    Projecting to operator form: 100%|██████████| 4/4 [00:00<00:00, 78.52it/s]


$\displaystyle \frac{\Omega_{z}^{3} \sigma_{3}}{2 \Omega_{z}^{2} - 2 \omega^{2}} + \frac{2 \Omega_{z} g^{2} \sigma_{3}}{2 \Omega_{z}^{2} - 2 \omega^{2}} + \frac{\Omega_{z} g^{2} \sigma_{3} {{a}^\dagger}^{2}}{\Omega_{z}^{2} - \omega^{2}} + \frac{\Omega_{z} g^{2} \sigma_{3} {a}^{2}}{\Omega_{z}^{2} - \omega^{2}} - \frac{\Omega_{z} \omega^{2} \sigma_{3}}{2 \Omega_{z}^{2} - 2 \omega^{2}} + \frac{g^{2} \omega}{\Omega_{z}^{2} - \omega^{2}} + \left(\frac{2 \Omega_{z} g^{2} \sigma_{3}}{\Omega_{z}^{2} - \omega^{2}} + \omega\right) {{a}^\dagger} {a}$


#### Multiblock Diagonalization


```python
mask = Block(fin=sx.matrix, inf=a) + Block(fin=s0.matrix, inf=a**2)
display(mask)
H_eff.solve(max_order=2, mask=mask)
```

$`\left[\begin{matrix}0 & 1\\1 & 0\end{matrix}\right]  \cdot {a} + \left[\begin{matrix}1 & 0\\0 & 1\end{matrix}\right]  \cdot {a}^{2}`$


    The perturbative interaction will be added to the full Hamiltonian


    Solving for each order: 100%|██████████| 2/2 [00:00<00:00, 13.97it/s]

    The Hamiltonian has been solved successfully. Please use the get_H method to get the result in the desired form.

```python
print('Effective Hamiltonian (Operator Form):')
display(H_eff.get_H())
```

    Effective Hamiltonian (Operator Form):
    Projecting to operator form: 100%|██████████| 2/2 [00:00<00:00, 89.03it/s]

$\displaystyle \frac{\Omega_{z}^{3} \sigma_{3}}{2 \Omega_{z}^{2} - 2 \omega^{2}} + \frac{2 \Omega_{z} g^{2} \sigma_{3}}{2 \Omega_{z}^{2} - 2 \omega^{2}} - \frac{\Omega_{z} \omega^{2} \sigma_{3}}{2 \Omega_{z}^{2} - 2 \omega^{2}} + \frac{g^{2} \omega}{\Omega_{z}^{2} - \omega^{2}} + \left(\frac{2 \Omega_{z} g^{2} \sigma_{3}}{\Omega_{z}^{2} - \omega^{2}} + \omega\right) {{a}^\dagger} {a}$

#### Full Diagonalization


```python
H_eff.solve(max_order=2, full_diagonalization=True)
H_eff.get_H()
```

    The perturbative interaction will be added to the full Hamiltonian

    Solving for each order: 100%|██████████| 2/2 [00:00<00:00, 25.64it/s]

    The Hamiltonian has been solved successfully. Please use the get_H method to get the result in the desired form.

    Projecting to operator form: 100%|██████████| 2/2 [00:00<00:00, 143.39it/s]



$\displaystyle \frac{\Omega_{z}^{3} \sigma_{3}}{2 \Omega_{z}^{2} - 2 \omega^{2}} + \frac{2 \Omega_{z} g^{2} \sigma_{3}}{2 \Omega_{z}^{2} - 2 \omega^{2}} - \frac{\Omega_{z} \omega^{2} \sigma_{3}}{2 \Omega_{z}^{2} - 2 \omega^{2}} + \frac{g^{2} \omega}{\Omega_{z}^{2} - \omega^{2}} + \left(\frac{2 \Omega_{z} g^{2} \sigma_{3}}{\Omega_{z}^{2} - \omega^{2}} + \omega\right) {{a}^\dagger} {a}$


## Rotate a driving term to the new rotated basis

$$
H_{\text{drive}} = E_0 (a^\dagger + a)
$$


```python
E0 = RDSymbol('E0', order=1, real=True)
H_drive = E0 * (ad + a)

H_eff.rotate(H_drive).expand()
```
    Rotating for each order: 100%|██████████| 2/2 [00:00<00:00, 637.29it/s]
    Projecting to operator form: 100%|██████████| 3/3 [00:00<00:00, 258.71it/s]

$\displaystyle \frac{2 E_{0} g \omega \sigma_{1}}{\Omega_{z}^{2} - \omega^{2}} + E_{0} {{a}^\dagger} + E_{0} {a}$


## Classes and Functions
### EffectiveFrame
- **Constructor**: `EffectiveFrame(H, V, subspaces=None)`
  - Initializes the frame with the zeroth-order Hamiltonian and the perturbative interaction.

- **Method**: `solve(max_order=2, full_diagonalization=False, mask=None)`
  - Solves for the effective Hamiltonian up to the specified order.

- **Method**: `get_H(return_operator_form=True)`
  - Returns the effective Hamiltonian in either operator form or matrix form.

### Utility Functions

(To be added)

## Examples
For more detailed examples, including usage with specific quantum systems and additional functionality, refer to the `examples` directory in the repository.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss improvements or bug fixes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
