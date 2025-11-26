from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
import ufl
import meshio
import os

# ---------------------------------------------------------
# 0. User input: base name of your RVE files (without extension)
# ---------------------------------------------------------
base = "rve_output/rve_quad_mesh_Lx1.00_phi50"  # <-- CHANGE to match your files

mesh_file_xdmf = base + ".xdmf"
mesh_file_vtk  = base + ".vtk"

if not os.path.exists(mesh_file_xdmf):
    raise FileNotFoundError(f"Cannot find mesh XDMF file: {mesh_file_xdmf}")
if not os.path.exists(mesh_file_vtk):
    raise FileNotFoundError(f"Cannot find mesh VTK file: {mesh_file_vtk}")

print(f"[INFO] Using XDMF mesh: {mesh_file_xdmf}")
print(f"[INFO] Using VTK damage: {mesh_file_vtk}")

# ---------------------------------------------------------
# 1. Read mesh in dolfinx from XDMF
# ---------------------------------------------------------
with io.XDMFFile(MPI.COMM_WORLD, mesh_file_xdmf, "r") as xdmf:
    msh = xdmf.read_mesh(name="Grid")
    msh.name = "RVE_mesh"

tdim = msh.topology.dim
print(f"[INFO] Mesh: {msh.geometry.x.shape[0]} nodes, "
      f"{msh.topology.index_map(tdim).size_local} cells, dim={tdim}")

msh.topology.create_connectivity(tdim, 0)

# ---------------------------------------------------------
# 2. Read cell-wise damage from VTK via meshio
# ---------------------------------------------------------
mio = meshio.read(mesh_file_vtk)

if "damage" not in mio.cell_data:
    raise RuntimeError("Could not find 'damage' in cell_data of the VTK file.")

damage_list = mio.cell_data["damage"]
if len(damage_list) != 1:
    print(f"[WARNING] Expected one cell block for damage, found {len(damage_list)}.")

damage_cells = np.asarray(damage_list[0]).ravel()

num_cells = msh.topology.index_map(tdim).size_local
if damage_cells.size != num_cells:
    print(f"[WARNING] damage_cells.size = {damage_cells.size}, num_cells = {num_cells}")
    print("          Assuming ordering matches (same generator).")

print(f"[INFO] Damage stats: min={damage_cells.min():.3f}, "
      f"max={damage_cells.max():.3f}, mean={damage_cells.mean():.3f}")

# ---------------------------------------------------------
# 3. DG0 damage field and material parameters (material-based porosity)
# ---------------------------------------------------------
V0 = fem.functionspace(msh, ("DG", 0))   # one value per cell
damage_fn = fem.Function(V0)
damage_fn.x.array[:] = damage_cells

# Base elastic material
E0 = 1.0e9   # Pa
nu = 0.3
p  = 2.0     # exponent for damage -> E mapping
Emin_ratio = 1e-3

E_eff_cells = (1.0 - damage_cells) ** p * E0
E_eff_cells = np.maximum(E_eff_cells, Emin_ratio * E0)

lambda_cells = (E_eff_cells * nu) / ((1 + nu) * (1 - 2 * nu))
mu_cells     = E_eff_cells / (2 * (1 + nu))

lambda_fn = fem.Function(V0)
mu_fn     = fem.Function(V0)
lambda_fn.x.array[:] = lambda_cells
mu_fn.x.array[:]     = mu_cells

print(f"[INFO] Effective E range: {E_eff_cells.min():.3e} .. {E_eff_cells.max():.3e}")

# ---------------------------------------------------------
# 4. Displacement FE space (2D, small strain)
# ---------------------------------------------------------
Vu = fem.functionspace(msh, ("CG", 1, (tdim,)))  # tdim = 2
u = fem.Function(Vu)
u_trial = ufl.TrialFunction(Vu)
v = ufl.TestFunction(Vu)

# ---------------------------------------------------------
# 5. Small-strain tensor (total strain)
# ---------------------------------------------------------
def eps(u_):
    grad_u = ufl.grad(u_)
    return ufl.as_matrix((
        (grad_u[0, 0], 0.5*(grad_u[0, 1] + grad_u[1, 0])),
        (0.5*(grad_u[1, 0] + grad_u[0, 1]), grad_u[1, 1])
    ))

def sigma_elastic(u_):
    lam = lambda_fn
    mu  = mu_fn
    eps_u = eps(u_)
    return 2.0 * mu * eps_u + lam * ufl.tr(eps_u) * ufl.Identity(2)

# ---------------------------------------------------------
# 6. BCs: left fixed, right prescribed displacement
# ---------------------------------------------------------
coords = msh.geometry.x
x_min = np.min(coords[:, 0])
x_max = np.max(coords[:, 0])
y_min = np.min(coords[:, 1])
y_max = np.max(coords[:, 1])

Lx = x_max - x_min
Ly = y_max - y_min

u0 = 0.001  # applied displacement in x

def left_boundary(x):
    return np.isclose(x[0], x_min)

def right_boundary(x):
    return np.isclose(x[0], x_max)

facets_left  = mesh.locate_entities_boundary(msh, tdim-1, left_boundary)
facets_right = mesh.locate_entities_boundary(msh, tdim-1, right_boundary)

dofs_left  = fem.locate_dofs_topological(Vu, tdim-1, facets_left)
dofs_right = fem.locate_dofs_topological(Vu, tdim-1, facets_right)

u_left_val = PETSc.ScalarType((0.0, 0.0))
bc_left = fem.dirichletbc(u_left_val, dofs_left, Vu)

u_right_val = PETSc.ScalarType((u0, 0.0))
bc_right = fem.dirichletbc(u_right_val, dofs_right, Vu)

bcs = [bc_left, bc_right]

# ---------------------------------------------------------
# 7. Solve linear elastic problem: ∫ σ_elastic(u):ε(v) dx = 0
# ---------------------------------------------------------
dx = ufl.Measure("cell", domain=msh)

a = ufl.inner(sigma_elastic(u_trial), eps(v)) * dx
zero_vec = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))
L = ufl.inner(zero_vec, v) * dx

problem = LinearProblem(
    a, L, bcs=bcs, u=u,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="rve_el_"
)
uh = problem.solve()
print("[INFO] Linear elastic displacement solved (will be used for plastic post-processing).")

# ---------------------------------------------------------
# 8. Project TOTAL strain ε(uh) to DG0 (per element)
# ---------------------------------------------------------
W = fem.functionspace(msh, ("DG", 0, (tdim, tdim)))
tau = ufl.TrialFunction(W)
w   = ufl.TestFunction(W)

a_eps = ufl.inner(tau, w) * dx
L_eps = ufl.inner(eps(uh), w) * dx
eps_proj = fem.Function(W)
problem_eps = LinearProblem(
    a_eps, L_eps, u=eps_proj,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="rve_eps_"
)
problem_eps.solve()
print("[INFO] Total strain projected to DG0.")

# ---------------------------------------------------------
# 9. J2 isotropic hardening (single increment) – local return mapping
# ---------------------------------------------------------
# Plasticity parameters
sigma_y0 = 1.0e6    # 1 MPa
H_iso    = 5.0e7 

# Extract total strain per cell from eps_proj
eps_arr = eps_proj.x.array.reshape(num_cells, 4)
eps_xx_tot = eps_arr[:, 0]
eps_xy_tot = 0.5 * (eps_arr[:, 1] + eps_arr[:, 2])  # symmetrize
eps_yy_tot = eps_arr[:, 3]

# Initialize plastic variables (starting from virgin state)
eps_p_xx = np.zeros_like(eps_xx_tot)
eps_p_yy = np.zeros_like(eps_xx_tot)
eps_p_xy = np.zeros_like(eps_xx_tot)
alpha    = np.zeros_like(eps_xx_tot)

# Arrays for output: elastic strain, stress, von Mises, energy
eps_e_xx = np.zeros_like(eps_xx_tot)
eps_e_yy = np.zeros_like(eps_xx_tot)
eps_e_xy = np.zeros_like(eps_xx_tot)

sig_xx = np.zeros_like(eps_xx_tot)
sig_yy = np.zeros_like(eps_xx_tot)
sig_xy = np.zeros_like(eps_xx_tot)
vm     = np.zeros_like(eps_xx_tot)
energy = np.zeros_like(eps_xx_tot)

for i in range(num_cells):
    lam_i = lambda_cells[i]
    mu_i  = mu_cells[i]

    # total strain at this cell
    exx = eps_xx_tot[i]
    eyy = eps_yy_tot[i]
    exy = eps_xy_tot[i]

    # elastic trial stress (plane stress assumption: szz included via deviatoric part)
    sx_tr  = (2*mu_i + lam_i)*exx + lam_i*eyy
    sy_tr  = lam_i*exx + (2*mu_i + lam_i)*eyy
    txy_tr = 2*mu_i*exy
    szz_tr = 0.0  # plane stress assumption

    # deviatoric trial stress (3D)
    p_tr = (sx_tr + sy_tr + szz_tr) / 3.0
    sxx_tr = sx_tr - p_tr
    syy_tr = sy_tr - p_tr
    szz_dev_tr = szz_tr - p_tr
    sxy_tr = txy_tr

    # J2 equivalent stress
    s_sq = (sxx_tr**2 + syy_tr**2 + szz_dev_tr**2 +
            2.0 * (sxy_tr**2))  # other shear components zero
    sigma_eq_tr = np.sqrt(1.5 * s_sq)

    # yield stress with isotropic hardening
    sigma_y = sigma_y0 + H_iso * alpha[i]
    f_tr = sigma_eq_tr - sigma_y

    if f_tr <= 0.0:
        # elastic step
        eps_p_xx[i] = eps_p_xx[i]
        eps_p_yy[i] = eps_p_yy[i]
        eps_p_xy[i] = eps_p_xy[i]
        alpha[i]    = alpha[i]

        sig_xx[i] = sx_tr
        sig_yy[i] = sy_tr
        sig_xy[i] = txy_tr
    else:
        # plastic step: radial return
        dgamma = f_tr / (3.0 * mu_i + H_iso)

        # flow direction n_ij = 3/(2*sigma_eq) * s_ij
        factor = 1.5 / sigma_eq_tr
        nxx = factor * sxx_tr
        nyy = factor * syy_tr
        nzz = factor * szz_dev_tr
        nxy = factor * sxy_tr

        # update plastic strain
        eps_p_xx[i] += dgamma * nxx
        eps_p_yy[i] += dgamma * nyy
        eps_p_xy[i] += dgamma * nxy
        alpha[i]    += np.sqrt(2.0/3.0) * dgamma

        # update deviatoric stresses
        sxx_new = sxx_tr - 2.0 * mu_i * dgamma * nxx
        syy_new = syy_tr - 2.0 * mu_i * dgamma * nyy
        szz_new = szz_dev_tr - 2.0 * mu_i * dgamma * nzz
        sxy_new = sxy_tr - 2.0 * mu_i * dgamma * nxy

        # reconstruct full stresses (same hydrostatic part p_tr)
        sig_xx[i] = sxx_new + p_tr
        sig_yy[i] = syy_new + p_tr
        sig_xy[i] = sxy_new

    # elastic strain = total - plastic
    eps_e_xx[i] = exx - eps_p_xx[i]
    eps_e_yy[i] = eyy - eps_p_yy[i]
    eps_e_xy[i] = exy - eps_p_xy[i]

    # von Mises stress from updated stress (plane stress J2)
    vm[i] = np.sqrt(sig_xx[i]**2 + sig_yy[i]**2 - sig_xx[i]*sig_yy[i] + 3.0*sig_xy[i]**2)

    # strain energy density ψ = 1/2 * σ : ε_e (2D version)
    energy[i] = 0.5 * (sig_xx[i]*eps_e_xx[i] +
                       sig_yy[i]*eps_e_yy[i] +
                       2.0*sig_xy[i]*eps_e_xy[i])

print("[INFO] Local J2 isotropic hardening update done (post-processing).")

# ---------------------------------------------------------
# 10. Export VTU with displacement + cell-wise elastic/plastic fields
# ---------------------------------------------------------
out_vtu = base + "_elastoplastic_J2.vtu"

if MPI.COMM_WORLD.rank == 0:
    points_out = msh.geometry.x
    uh_arr = uh.x.array.reshape(-1, tdim)

    msh.topology.create_connectivity(tdim, 0)
    cells_connectivity = msh.topology.connectivity(tdim, 0)
    num_cells_local = msh.topology.index_map(tdim).size_local

    first_cell_nodes = cells_connectivity.links(0)
    num_nodes_per_cell = len(first_cell_nodes)

    cells_array = np.zeros((num_cells_local, num_nodes_per_cell), dtype=np.int32)
    for i in range(num_cells_local):
        cell_nodes = cells_connectivity.links(i)
        cells_array[i, :] = cell_nodes

    cell_type = "quad" if num_nodes_per_cell == 4 else "triangle"

    m_out = meshio.Mesh(
        points=points_out,
        cells=[(cell_type, cells_array)],
        point_data={"u": uh_arr},
        cell_data={
            "damage":        [damage_cells],
            "eps_xx_tot":    [eps_xx_tot],
            "eps_yy_tot":    [eps_yy_tot],
            "eps_xy_tot":    [eps_xy_tot],
            "eps_p_xx":      [eps_p_xx],
            "eps_p_yy":      [eps_p_yy],
            "eps_p_xy":      [eps_p_xy],
            "eps_e_xx":      [eps_e_xx],
            "eps_e_yy":      [eps_e_yy],
            "eps_e_xy":      [eps_e_xy],
            "sigma_xx":      [sig_xx],
            "sigma_yy":      [sig_yy],
            "sigma_xy":      [sig_xy],
            "von_mises":     [vm],
            "strain_energy": [energy],
            "alpha":         [alpha],
        },
    )
    meshio.write(out_vtu, m_out)
    print(f"[INFO] Wrote elastoplastic VTU: {out_vtu}")

print("[DONE] Elastoplastic J2 post-processing on damaged RVE complete.")
