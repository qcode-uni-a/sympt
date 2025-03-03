System found in **Qubit-flip-induced cavity mode squeezing in the strong dispersive regime of the quantum Rabi model** 10.1038/srep45587


```python
# sympt imports
from sympt import *
# Import sympy
import sympy as sp
```

# Setup


```python
# ---------------- Defining the symbols ------------------
# Order 0
omega = RDSymbol('omega', order=0, positive=True, real=True)
Omega_z = RDSymbol('Omega_z', order=0, positive=True, real=True)
# Order 1
g = RDSymbol('g', order=1, positive=True, real=True)

# ----------------- Defining the basis -------------------
# Spin basis: Finite 2x2 Hilbert space
Spin = RDBasis('sigma', dim=2)
s0, sx, sy, sz = Spin.basis # Pauli Operators
# Boson basis: Infinite bosonic Hilbert space
a = BosonOp('a')
ad = Dagger(a)

# -------------- Defining the Hamiltonian ----------------
# Unperturbed Hamiltonian H0
H0 = omega * ad * a + sp.Rational(1,2) * Omega_z * sz
display(H0)
# Interaction Hamiltonian V
V = g * (ad + a) * sx
display(V)
```


$\displaystyle \frac{\Omega_{z} \sigma_{3}}{2} + \omega {{a}^\dagger} {a}$



$\displaystyle g \left({{a}^\dagger} + {a}\right) \sigma_{1}$



```python
# Deffining Effective Hamiltonian Object
Eff_frame = EffectiveFrame(H0, V, subspaces=[Spin], verbose = False)
```

# Standard Schrieffer-Wolff Transformation


```python
# Calculate the effective model using the Schrieffer-Wolff transformation up to the second order
Eff_frame.solve(max_order=2, method='SW')
# Obtaining the result in the operator form
H_eff_SWT = Eff_frame.get_H(return_form='operator')
H_eff_SWT
```




$\displaystyle \frac{\Omega_{z} g^{2} \sigma_{3}}{\Omega_{z}^{2} - \omega^{2}} + \frac{2 \Omega_{z} g^{2} \sigma_{3} {{a}^\dagger} {a}}{\Omega_{z}^{2} - \omega^{2}} + \frac{\Omega_{z} g^{2} \sigma_{3} {{a}^\dagger}^{2}}{\Omega_{z}^{2} - \omega^{2}} + \frac{\Omega_{z} g^{2} \sigma_{3} {a}^{2}}{\Omega_{z}^{2} - \omega^{2}} + \frac{\Omega_{z} \sigma_{3}}{2} + \frac{g^{2} \omega}{\Omega_{z}^{2} - \omega^{2}} + \omega {{a}^\dagger} {a}$



# Multiblock ACE


```python
# Deffining the mask
mask = Block(fin=sx.matrix, inf=a)
# Calculate the effective model using the Mask routine up to the second order
Eff_frame.solve(max_order=2, method='ACE', mask=mask)
H_eff_Mask = Eff_frame.get_H(return_form='operator')
H_eff_Mask
```




$\displaystyle \frac{\Omega_{z} g^{2} \sigma_{3}}{\Omega_{z}^{2} - \omega^{2}} + \frac{2 \Omega_{z} g^{2} \sigma_{3} {{a}^\dagger} {a}}{\Omega_{z}^{2} - \omega^{2}} + \frac{\Omega_{z} g^{2} \sigma_{3} {{a}^\dagger}^{2}}{\Omega_{z}^{2} - \omega^{2}} + \frac{\Omega_{z} g^{2} \sigma_{3} {a}^{2}}{\Omega_{z}^{2} - \omega^{2}} + \frac{\Omega_{z} \sigma_{3}}{2} + \frac{g^{2} \omega}{\Omega_{z}^{2} - \omega^{2}} + \omega {{a}^\dagger} {a}$



# Full-diagonalization


```python
# Calculate the effective model using the Full Diagonalization routine up to the second order
Eff_frame.solve(max_order=2, method='FD')
H_eff_FD = Eff_frame.get_H(return_form='operator')
H_eff_FD
```




$\displaystyle \frac{\Omega_{z} g^{2} \sigma_{3}}{\Omega_{z}^{2} - \omega^{2}} + \frac{2 \Omega_{z} g^{2} \sigma_{3} {{a}^\dagger} {a}}{\Omega_{z}^{2} - \omega^{2}} + \frac{\Omega_{z} \sigma_{3}}{2} + \frac{g^{2} \omega}{\Omega_{z}^{2} - \omega^{2}} + \omega {{a}^\dagger} {a}$




```python
display_dict(Eff_frame.H_corrections)
```


$\displaystyle 0 : \frac{\Omega_{z} \sigma_{3}}{2} + \omega {{a}^\dagger} {a}$



$\displaystyle 2 : \frac{\Omega_{z} g^{2} \sigma_{3}}{\Omega_{z}^{2} - \omega^{2}} + \frac{2 \Omega_{z} g^{2} \sigma_{3} {{a}^\dagger} {a}}{\Omega_{z}^{2} - \omega^{2}} + \frac{g^{2} \omega}{\Omega_{z}^{2} - \omega^{2}}$


## Get Unitary


```python
U = Eff_frame.get_U()
U
```




$\displaystyle - \frac{\Omega_{z} g^{2} \omega \sigma_{3}}{\Omega_{z}^{4} - 2 \Omega_{z}^{2} \omega^{2} + \omega^{4}} + \frac{\Omega_{z} g^{2} \sigma_{3} {{a}^\dagger}^{2}}{2 \Omega_{z}^{2} \omega - 2 \omega^{3}} - \frac{\Omega_{z} g^{2} \sigma_{3} {a}^{2}}{2 \Omega_{z}^{2} \omega - 2 \omega^{3}} - \frac{g^{2} {{a}^\dagger}^{2}}{2 \Omega_{z}^{2} - 2 \omega^{2}} - \frac{g^{2} {a}^{2}}{2 \Omega_{z}^{2} - 2 \omega^{2}} + \frac{- \Omega_{z}^{2} g^{2} - g^{2} \omega^{2}}{2 \Omega_{z}^{4} - 4 \Omega_{z}^{2} \omega^{2} + 2 \omega^{4}} + \frac{\left(- \Omega_{z}^{2} g^{2} - g^{2} \omega^{2}\right) {{a}^\dagger} {a}}{\Omega_{z}^{4} - 2 \Omega_{z}^{2} \omega^{2} + \omega^{4}} + 1 + \left(\frac{i \Omega_{z} g \sigma_{2}}{\Omega_{z}^{2} - \omega^{2}} - \frac{g \omega \sigma_{1}}{\Omega_{z}^{2} - \omega^{2}}\right) {{a}^\dagger} + \left(\frac{i \Omega_{z} g \sigma_{2}}{\Omega_{z}^{2} - \omega^{2}} + \frac{g \omega \sigma_{1}}{\Omega_{z}^{2} - \omega^{2}}\right) {a}$




```python
display_dict(Eff_frame.U_corrections)
```


$\displaystyle 0 : 1$



$\displaystyle 1 : \left(\frac{i \Omega_{z} g \sigma_{2}}{\Omega_{z}^{2} - \omega^{2}} - \frac{g \omega \sigma_{1}}{\Omega_{z}^{2} - \omega^{2}}\right) {{a}^\dagger} + \left(\frac{i \Omega_{z} g \sigma_{2}}{\Omega_{z}^{2} - \omega^{2}} + \frac{g \omega \sigma_{1}}{\Omega_{z}^{2} - \omega^{2}}\right) {a}$



$\displaystyle 2 : - \frac{\Omega_{z} g^{2} \omega \sigma_{3}}{\Omega_{z}^{4} - 2 \Omega_{z}^{2} \omega^{2} + \omega^{4}} + \frac{\Omega_{z} g^{2} \sigma_{3} {{a}^\dagger}^{2}}{2 \Omega_{z}^{2} \omega - 2 \omega^{3}} - \frac{\Omega_{z} g^{2} \sigma_{3} {a}^{2}}{2 \Omega_{z}^{2} \omega - 2 \omega^{3}} - \frac{g^{2} {{a}^\dagger}^{2}}{2 \Omega_{z}^{2} - 2 \omega^{2}} - \frac{g^{2} {a}^{2}}{2 \Omega_{z}^{2} - 2 \omega^{2}} + \frac{- \Omega_{z}^{2} g^{2} - g^{2} \omega^{2}}{2 \Omega_{z}^{4} - 4 \Omega_{z}^{2} \omega^{2} + 2 \omega^{4}} + \frac{\left(- \Omega_{z}^{2} g^{2} - g^{2} \omega^{2}\right) {{a}^\dagger} {a}}{\Omega_{z}^{4} - 2 \Omega_{z}^{2} \omega^{2} + \omega^{4}}$


# Get the generator transformation


```python
S = Eff_frame.get_S()
S
```




$\displaystyle - \frac{\Omega_{z} g^{2} \sigma_{3} {{a}^\dagger}^{2}}{2 \Omega_{z}^{2} \omega - 2 \omega^{3}} + \frac{\Omega_{z} g^{2} \sigma_{3} {a}^{2}}{2 \Omega_{z}^{2} \omega - 2 \omega^{3}} + \left(- \frac{i \Omega_{z} g \sigma_{2}}{\Omega_{z}^{2} - \omega^{2}} - \frac{g \omega \sigma_{1}}{\Omega_{z}^{2} - \omega^{2}}\right) {a} + \left(- \frac{i \Omega_{z} g \sigma_{2}}{\Omega_{z}^{2} - \omega^{2}} + \frac{g \omega \sigma_{1}}{\Omega_{z}^{2} - \omega^{2}}\right) {{a}^\dagger}$




```python
display_dict(Eff_frame.S_corrections)
```


$\displaystyle 0 : 0.0$



$\displaystyle 1 : \left(- \frac{i \Omega_{z} g \sigma_{2}}{\Omega_{z}^{2} - \omega^{2}} - \frac{g \omega \sigma_{1}}{\Omega_{z}^{2} - \omega^{2}}\right) {a} + \left(- \frac{i \Omega_{z} g \sigma_{2}}{\Omega_{z}^{2} - \omega^{2}} + \frac{g \omega \sigma_{1}}{\Omega_{z}^{2} - \omega^{2}}\right) {{a}^\dagger}$



$\displaystyle 2 : - \frac{\Omega_{z} g^{2} \sigma_{3} {{a}^\dagger}^{2}}{2 \Omega_{z}^{2} \omega - 2 \omega^{3}} + \frac{\Omega_{z} g^{2} \sigma_{3} {a}^{2}}{2 \Omega_{z}^{2} \omega - 2 \omega^{3}}$


# Rotate a new term such as a Driving term into the new basis


```python
# ----- Rotating a Drving term into the new basis ----
# Define the symbol for the driving term
E0 = RDSymbol('E0', order=0, positive=True, real=True)
# Define the driving term
H_drive = E0 * (a + ad)
display(H_drive)

# Rotate the driving term into the new basis
Eff_frame.rotate(H_drive, max_order=1, return_form='operator').cancel()
```


$\displaystyle E_{0} \left({{a}^\dagger} + {a}\right)$


    Converting to operator form: 100%|██████████| 3/3 [00:00<00:00, 207.01it/s]





$\displaystyle E_{0} \left(\frac{2 g \omega \sigma_{1}}{\Omega_{z}^{2} - \omega^{2}} + {{a}^\dagger} + {a}\right)$


