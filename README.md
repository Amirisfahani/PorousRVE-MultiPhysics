# PorousRVE-MultiPhysics

This project demonstrates a modular workflow to simulate **multiphysics behavior in nanoporous materials** using an artificial Representative Volume Element (RVE). The focus is to understand how **damage**, **elasticity**, **plasticity**, **diffusion**, and **swelling** interact.

The implementation is based on **FEniCSx**, **NumPy**, and **PETSc**.

---

## 1️⃣ Porous RVE generation (Python/NumPy)

A structured quadrilateral mesh is created and connectivity between nodes is used to define bonds. Random bond removal introduces porosity.

The **cell-wise damage** variable is defined as:

$$
d = \frac{N_{\text{broken}}}{N_{\text{total}}}, \qquad 0 \le d \le 1
$$

Outputs:
- Mesh (XDMF format)
- Damage field \(d(x)\)

---

## 2️⃣ Elasticity with damage-dependent stiffness

Damage reduces material stiffness:

$$
\mathbb{C}_{\text{eff}}(d) = (1-d)^p \, \mathbb{C}_0
$$

We solve mechanical equilibrium:

$$
\nabla \cdot \boldsymbol{\sigma} = \mathbf{0}
$$

with:

$$
\boldsymbol{\sigma} = \mathbb{C}_{\text{eff}}(d) :
\boldsymbol{\varepsilon}(\mathbf{u}), \qquad
\boldsymbol{\varepsilon}(\mathbf{u}) = \frac{1}{2}
(\nabla\mathbf{u} + \nabla\mathbf{u}^T)
$$

---

## 3️⃣ J2 Plasticity (Isotropic Hardening)


Hardening rule:

$$
\sigma_y (\alpha) = \sigma_{y0} + H_{\text{iso}}\alpha
$$

Solved using:
- Return-mapping algorithm at quadrature points
- Newton-Raphson iterations globally

Outputs: plastic strain and updated stress

---

## 4️⃣ Diffusion with damage-dependent transport

Damage increase diffusion coefficient:



$$
D(d) = D_{\text{matrix}} +
\left(D_{\text{pore}} - D_{\text{matrix}}\right) d^{q_D}
$$

Time stepping:
- Backward Euler (implicit, unconditionally stable)

Output: concentration field \(c(x,t)\)

---

## 5️⃣ Swelling-induced stress coupling

Swelling strain due to concentration:

$$ \varepsilon_{\text{sw}}(c) = \beta \, c \, \mathbf{I} $$

Total strain (additive decomposition):

$$ \varepsilon_{\text{tot}} = \varepsilon(\mathbf{u}) - \varepsilon_{\text{sw}}(c) $$

Constitutive relation with swelling:

$$ \sigma_{\text{tot}} = \mathbb{C}_{\text{eff}}(d) \colon \big( \varepsilon(\mathbf{u}) - \beta \, c \, \mathbf{I} \big) $$


---

## 🔧 Software Stack

| Component | Purpose |
|----------|---------|
| FEniCSx | FEM for mechanics and diffusion |
| PETSc | Linear/Nonlinear solvers |
| NumPy | Microstructure generation |
| XDMF/HDF5 | Data output for visualization |

---

## 📌 Summary

| Physics | Effect of Porosity |
|--------|------------------|
| Elasticity | Stress avoids weak regions |
| Plasticity | Localization near damaged channels |
| Diffusion | Faster transport through pores |
| Swelling | Concentration gradients create stress |

This framework allows controlled testing of **microscale coupling mechanisms** in nanoporous materials.



