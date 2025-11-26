# PorousRVE-MultiPhysics

This project explores how nanoporous materials behave when you combine **mechanics**, **diffusion**, and **swelling** on a single artificial **representative volume element (RVE)**. Everything is built on **FEniCSx** (for FEM) plus some **Python/NumPy** pre-processing.

In short: we generate a porous microstructure, see how damage changes stiffness and stress flow, add plasticity, model diffusion that prefers damaged regions, and then couple concentration to swelling-induced stress.

---

## What you’ll do

- Build an artificial porous RVE (no imaging required).
- Compute a damage field from broken bonds.
- Run linear elasticity to see how stiffness loss reroutes stress.
- Add J2 plasticity with isotropic hardening.
- Solve transient diffusion with damage-dependent transport.
- Couple concentration to swelling and observe induced stresses.

Each step produces fields you can visualize (stress, plastic strain, concentration, swelling).

---

## 1) Make the porous RVE (Python/NumPy)

We start with a simple, structured grid and turn it into a porous microstructure using random bond removal.

- **Mesh:** structured quadrilateral grid (regular lattice).
- **Connectivity:** each node is connected to its nearest neighbors by “bonds”.
- **Porosity:** we randomly remove some of these bonds.
- **Damage per cell:** we count how many bonds are broken around each cell and define

\[
d = \frac{N_{\text{broken}}}{N_{\text{total}}}, \quad 0 \le d \le 1.
\]

Here:
- \( d = 0 \): fully intact,
- \( d \to 1 \): highly damaged / porous.

**Outputs**

- Mesh (nodes + elements).
- Damage field \( d(x) \) defined per cell.

**How it’s solved**

This part is purely **Python/NumPy**, no FEM yet:

1. Create a regular grid of points and quadrilateral cells.
2. Build a list of bonds (pairs of neighboring nodes).
3. Randomly “break” some bonds based on a target porosity \( \phi \).
4. For each cell, count broken vs. total bonds around it and compute \( d \).
5. Export:
   - The mesh (e.g. XDMF),
   - A cell-wise field for damage.

**Figures to add**

- Damage field at \(\phi = 0.3\):  
  `![Damage at φ = 0.3](figures/damage_phi0.3.png)`
- Damage field at \(\phi = 0.5\):  
  `![Damage at φ = 0.5](figures/damage_phi0.5.png)`



---

## 2) Linear elasticity (FEniCSx)

Next, we solve a **linear elastic** problem on the damaged RVE. Damage reduces stiffness by a scalar factor:

\[
E_{\text{eff}}(d) = (1 - d)^{p} \, E_0,
\]

where:
- \( E_0 \) is the base Young’s modulus,
- \( p \ge 1 \) controls how aggressively stiffness drops with damage.

The damaged stiffness tensor becomes:

\[
\mathbb{C}_{\text{eff}}(d) = (1 - d)^{p} \, \mathbb{C}_0.
\]

**Boundary conditions**

- Left edge: fixed in the loading direction (and usually constrained to avoid rigid motions).
- Right edge: prescribed displacement or traction to apply **tension**.

**Outputs**

- Displacement field \( \mathbf{u}(x) \)
- Strain \( \boldsymbol{\varepsilon}(\mathbf{u}) \)
- Stress \( \boldsymbol{\sigma} = \mathbb{C}_{\text{eff}} : \boldsymbol{\varepsilon} \)
- Von Mises stress \( \sigma_{\text{VM}} \)
- Strain energy density

**How it’s solved**

We solve the standard static equilibrium equation:

\[
\nabla \cdot \boldsymbol{\sigma}(\mathbf{u}, d) = \mathbf{0}
\]

with:

\[
\boldsymbol{\sigma}(\mathbf{u}, d) = \mathbb{C}_{\text{eff}}(d) : \boldsymbol{\varepsilon}(\mathbf{u}), \quad
\boldsymbol{\varepsilon}(\mathbf{u}) = \frac{1}{2}(\nabla \mathbf{u} + \nabla \mathbf{u}^T).
\]

In FEniCSx:

1. Define a **vector function space** for displacements.
2. Set up the weak form:

\[
\int_{\Omega} \boldsymbol{\varepsilon}(\mathbf{v}) : \mathbb{C}_{\text{eff}}(d) : \boldsymbol{\varepsilon}(\mathbf{u}) \,\mathrm{d}x
= \int_{\Omega} \mathbf{v} \cdot \mathbf{b} \,\mathrm{d}x 
+ \int_{\Gamma_N} \mathbf{v} \cdot \bar{\mathbf{t}} \,\mathrm{d}s.
\]

3. Apply Dirichlet BCs on the left and Neumann or displacement loading on the right.
4. Use the FEniCSx linear solver to compute \( \mathbf{u} \).
5. Postprocess stress and von Mises stress and write to XDMF.

**What you’ll see**

- Stress avoids highly damaged regions (low stiffness) and prefers **stiffer, continuous paths**, often diagonals across the RVE.

**Figures to add**

- Homogeneous vs porous Von Mises stress:  
  `![Von Mises stress: homogeneous](figures/vonmises_elastic_hom.png)`
  `![Von Mises stress: porous](figures/vonmises_elastic_porous.png)`

---

## 3) J2 plasticity (isotropic hardening)

Now we switch from purely elastic to **elasto-plastic** behavior with **J2 plasticity** and **isotropic hardening**.

The yield stress depends on accumulated plastic strain \( \alpha \):

\[
\sigma_y(\alpha) = \sigma_{y0} + H_{\text{iso}} \, \alpha,
\]

where:
- \( \sigma_{y0} \) is the initial yield stress,
- \( H_{\text{iso}} \) is the isotropic hardening modulus.

**Outputs**

- Plastic strain \( \boldsymbol{\varepsilon}^p \)
- Updated stress field \( \boldsymbol{\sigma} \)
- Energy density (elastic + plastic)

**How it’s solved**

We use a **local return-mapping algorithm** at each integration point, embedded in the FEM loop:

1. Compute a **trial elastic stress**:

\[
\boldsymbol{\sigma}^{\text{trial}} = \mathbb{C}_{\text{eff}}(d) : \left(\boldsymbol{\varepsilon}(\mathbf{u}) - \boldsymbol{\varepsilon}^p_{\text{old}}\right).
\]

2. Compute the **von Mises equivalent stress**:

\[
\sigma_{\text{eq}} = \sqrt{\frac{3}{2} \, \boldsymbol{s}^{\text{trial}} : \boldsymbol{s}^{\text{trial}}},
\]
where \( \boldsymbol{s}^{\text{trial}} \) is the deviatoric part.

3. Check the yield function:

\[
f = \sigma_{\text{eq}} - \sigma_y(\alpha_{\text{old}}).
\]

- If \( f \le 0 \): purely elastic step → accept trial state.
- If \( f > 0 \): plastic step → perform **radial return**:
  - Solve for plastic multiplier increment \( \Delta \gamma \).
  - Update plastic strain \( \boldsymbol{\varepsilon}^p \).
  - Update stress \( \boldsymbol{\sigma} \) on the yield surface.
  - Update hardening variable \( \alpha \).

4. Assemble the global residual and tangent operator and solve the nonlinear system (e.g. with Newton iterations in FEniCSx).

**What you’ll see**

- Plastic strain **localizes along paths influenced by damage**.
- Regions around pores/damaged channels yield earlier and form plastic “bands”.

**Figures to add**

- Plastic strain field and corresponding Von Mises stress:  
  `![Plastic strain and Von Mises stress](figures/plastic_strain_vonmises.png)`

---

## 4) Transient diffusion (hydrogen/ions)

Next, we simulate **transient diffusion** (e.g. hydrogen or ions) through the same microstructure. Diffusivity is higher in damaged regions:

\[
D(d) = D_{\text{matrix}} + \left(D_{\text{pore}} - D_{\text{matrix}}\right) d^{q_D},
\]

with:
- \( D_{\text{matrix}} \): diffusivity of the intact material,
- \( D_{\text{pore}} \): effective diffusivity in fully damaged/porous zones,
- \( q_D \): exponent controlling how quickly diffusivity increases with damage.

The governing equation (no reactions) is:

\[
\frac{\partial c}{\partial t} = \nabla \cdot \left( D(d) \, \nabla c \right),
\]

where \( c = c(x,t) \) is concentration.

**Time stepping**

We use **Backward Euler** (fully implicit):

\[
\frac{c^{n+1} - c^{n}}{\Delta t} 
= \nabla \cdot \left( D(d) \, \nabla c^{n+1} \right).
\]

This leads to a linear system for \( c^{n+1} \) at each time step.

**How it’s solved (FEniCSx)**

1. Define a **scalar function space** for \( c \).
2. For each time step:
   - Set up the weak form:

\[
\int_\Omega \frac{c^{n+1} - c^{n}}{\Delta t} \, w \,\mathrm{d}x
+ \int_\Omega D(d) \, \nabla c^{n+1} \cdot \nabla w \,\mathrm{d}x = 0
\]

     for all test functions \( w \).
   - Apply Dirichlet or Neumann boundary conditions for concentration or flux.
   - Solve the resulting linear system for \( c^{n+1} \).
   - Write the solution to XDMF so you can view it as a time series.

**Outputs**

- Time-evolving concentration field \( c(x, t) \) (XDMF time steps).

**What you’ll see**

- Damaged micro-pores act as **fast diffusion channels**.
- Concentration fronts advance more quickly along highly damaged zones.

**Figures to add**

- Final-time concentration field (or a snapshot at an interesting time):  
  `![Final-time concentration field](figures/concentration_final.png)`

---

## 5) Chemomechanical swelling

Finally, we **couple concentration back to mechanics** through **swelling**.

Concentration causes **isotropic volumetric swelling**:

\[
\boldsymbol{\varepsilon}_{\text{sw}}(c) = \beta \, c \, \mathbf{I},
\]

where:
- \( \beta \) is the swelling coefficient,
- \( \mathbf{I} \) is the identity tensor.

The total strain is decomposed as:

\[
\boldsymbol{\varepsilon}_{\text{tot}} = \boldsymbol{\varepsilon}(\mathbf{u}) - \boldsymbol{\varepsilon}_{\text{sw}}(c).
\]

We put a **minus sign** here because swelling is treated as an **eigenstrain**: if the material wants to expand due to \( \boldsymbol{\varepsilon}_{\text{sw}} \), the elastic part \( \boldsymbol{\varepsilon}(\mathbf{u}) \) must “compensate” it in constrained regions, creating internal stress.

The stress is then:

\[
\boldsymbol{\sigma}_{\text{tot}} = \mathbb{C}_{\text{eff}}(d) : \left( \boldsymbol{\varepsilon}(\mathbf{u}) - \boldsymbol{\varepsilon}_{\text{sw}}(c) \right).
\]

**How it’s solved (FEniCSx)**

We typically do this as a **one-way coupling**:

1. Take a concentration field \( c(x) \) from the diffusion step (e.g. at a final time).
2. Compute \( \boldsymbol{\varepsilon}_{\text{sw}}(c) \) as a known field.
3. Solve the mechanical equilibrium again with the modified strain:

\[
\nabla \cdot \boldsymbol{\sigma}_{\text{tot}} = \mathbf{0}.
\]

In the weak form, this just means that the stiffness term now uses
\( \boldsymbol{\varepsilon}(\mathbf{u}) - \boldsymbol{\varepsilon}_{\text{sw}}(c) \).

**Outputs**

- Swelling-induced displacement field \( \mathbf{u}(x) \)
- Stress field due to swelling \( \boldsymbol{\sigma}_{\text{tot}}(x) \)
- Von Mises stress from swelling

**What you’ll see**

- Spatial variations in concentration create **gradients of swelling strain**, which translate into internal **tensile stresses** and additional deformation, even without external loads.

**Figures to add**

- Von Mises stress due to swelling:  
  `![Swelling-induced Von Mises stress](figures/swelling_vonmises.png)`

---

## Key takeaways

- **Elasticity:** Stress follows **stiffer, connected pathways**, avoiding highly damaged regions.
- **Plasticity:** Damage encourages **permanent deformation channels** and earlier yielding along porous paths.
- **Diffusion:** Porosity creates **heterogeneous transport**, with fast channels in damaged regions.
- **Swelling:** Concentration gradients convert into **mechanical stresses** via swelling eigenstrains.

You can plug in your figures at the links above and explore how the response changes with porosity, loading, and parameter choices.
