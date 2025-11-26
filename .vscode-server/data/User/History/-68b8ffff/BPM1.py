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

base = "rve_output/rve_quad_mesh_Lx1.00_phi30"  # <-- CHANGE to match your files

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

# ensure connectivity for later export
msh.topology.create_connectivity(tdim, 0)

# ---------------------------------------------------------
# 2. Read cell-wise damage from VTK via meshio
# ---------------------------------------------------------
mio = meshio.read(mesh_file_vtk)

if "damage" not in mio.cell_data:
    raise RuntimeError("Could not find 'damage' in cell_data of the VTK file.")

damage_list = mio.cell_data["damage"]   # list (one entry per cell block)
if len(damage_list) != 1:
    print(f"[WARNING] Expected one cell block for damage, found {len(damage_list)}.")

damage_cells = np.asarray(damage_list[0]).ravel()

num_cells_dolfinx = msh.topology.index_map(tdim).size_local
if damage_cells.size != num_cells_dolfinx:
    print(f"[WARNING] damage_cells.size = {damage_cells.size}, "
          f"num_cells_dolfinx = {num_cells_dolfinx}")
    print("          Assuming ordering matches (same generator).")

print(f"[INFO] Damage stats: min={damage_cells.min():.3f}, "
      f"max={damage_cells.max():.3f}, mean={damage_cells.mean():.3f}")

# ---------------------------------------------------------
# 3. DG0 damage field and material parameters (material-based porosity)
# ---------------------------------------------------------
V0 = fem.functionspace(msh, ("DG", 0))   # one value per cell
damage_fn = fem.Function(V0)
damage_fn.x.array[:] = damage_cells

# Base material
E0 = 1.0e9   # Pa
nu = 0.3
p  = 2.0
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
# 4. Displacement FE space (2D vector field, small strain)
# ---------------------------------------------------------
Vu = fem.functionspace(msh, ("CG", 1, (tdim,)))  # tdim = 2
u = fem.Function(Vu)
u_trial = ufl.TrialFunction(Vu)
v = ufl.TestFunction(Vu)

# ---------------------------------------------------------
# 5. Small-strain tensor and stress σ(u)
# ---------------------------------------------------------
def eps(u_):
    grad_u = ufl.grad(u_)
    # 2D small-strain tensor
    return ufl.as_matrix((
        (grad_u[0, 0], 0.5*(grad_u[0, 1] + grad_u[1, 0])),
        (0.5*(grad_u[1, 0] + grad_u[0, 1]), grad_u[1, 1])
    ))

def sigma(u_):
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
# 7. Variational formulation and solve: ∫ σ(u):ε(v) dx = 0
# ---------------------------------------------------------
dx = ufl.Measure("cell", domain=msh)

a = ufl.inner(sigma(u_trial), eps(v)) * dx
zero_vec = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))
L = ufl.inner(zero_vec, v) * dx

problem = LinearProblem(
    a, L, bcs=bcs, u=u,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="rve_main_"
)
uh = problem.solve()
print("[INFO] Linear elasticity problem solved.")

# ---------------------------------------------------------
# 8. PROJECT local strain, stress, von Mises, strain energy to DG0 (per element)
# ---------------------------------------------------------
# Tensor DG0 space for 2x2 tensors
W = fem.functionspace(msh, ("DG", 0, (tdim, tdim)))
tau = ufl.TrialFunction(W)
w   = ufl.TestFunction(W)

# 8.1 Strain tensor eps(uh)
a_eps = ufl.inner(tau, w) * dx
L_eps = ufl.inner(eps(uh), w) * dx
eps_proj = fem.Function(W)
problem_eps = LinearProblem(a_eps, L_eps, u=eps_proj,
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                            petsc_options_prefix="rve_eps_")
problem_eps.solve()

# 8.2 Stress tensor sigma(uh)
a_sig = ufl.inner(tau, w) * dx
L_sig = ufl.inner(sigma(uh), w) * dx
sig_proj = fem.Function(W)
problem_sig = LinearProblem(a_sig, L_sig, u=sig_proj,
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                            petsc_options_prefix="rve_sig_")
problem_sig.solve()

# 8.3 Von Mises stress (plane strain / 2D invariant)
# use UFL expression then project to DG0 scalar
sigma_u = sigma(uh)
sx = sigma_u[0, 0]
sy = sigma_u[1, 1]
txy = sigma_u[0, 1]

sigma_vm_expr = ufl.sqrt(sx**2 + sy**2 - sx*sy + 3.0*txy**2)

vm_trial = ufl.TrialFunction(V0)
vm_test  = ufl.TestFunction(V0)
a_vm = ufl.inner(vm_trial, vm_test) * dx
L_vm = ufl.inner(sigma_vm_expr, vm_test) * dx
vm_fn = fem.Function(V0)
problem_vm = LinearProblem(a_vm, L_vm, u=vm_fn,
                           petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                           petsc_options_prefix="rve_vm_")
problem_vm.solve()

# 8.4 Strain energy density ψ = 1/2 ε:σ
energy_expr = 0.5 * ufl.inner(eps(uh), sigma(uh))
en_trial = ufl.TrialFunction(V0)
en_test  = ufl.TestFunction(V0)
a_en = ufl.inner(en_trial, en_test) * dx
L_en = ufl.inner(energy_expr, en_test) * dx
energy_fn = fem.Function(V0)
problem_en = LinearProblem(a_en, L_en, u=energy_fn,
                           petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                           petsc_options_prefix="rve_en_")
problem_en.solve()

print("[INFO] Projected strain, stress, von Mises, energy to DG0.")

# ---------------------------------------------------------
# 9. Reaction force and effective modulus (macro E_eff)
# ---------------------------------------------------------
# Tag right boundary facets
tag_vals = np.full(len(facets_right), 1, dtype=np.int32)
mt_right = mesh.meshtags(msh, tdim-1, facets_right, tag_vals)
ds = ufl.Measure("ds", domain=msh, subdomain_data=mt_right)

n = ufl.FacetNormal(msh)
# Traction = σ·n (matrix-vector product)
sigma_u = sigma(uh)
traction = ufl.as_vector([sigma_u[0, 0] * n[0] + sigma_u[0, 1] * n[1],
                          sigma_u[1, 0] * n[0] + sigma_u[1, 1] * n[1]])

# Total reaction force in x-direction on right boundary
F_reac = fem.assemble_scalar(fem.form(traction[0] * ds(1)))
F_reac = MPI.COMM_WORLD.allreduce(F_reac, op=MPI.SUM)

# Macroscopic strain and stress
eps_macro = u0 / Lx
area = Ly * 1.0  # unit thickness
sigma_macro = F_reac / area
E_eff_macro = sigma_macro / eps_macro

if MPI.COMM_WORLD.rank == 0:
    print(f"[INFO] Reaction force Fx on right boundary: {F_reac:.6e} N")
    print(f"[INFO] Macroscopic strain: {eps_macro:.6e}")
    print(f"[INFO] Macroscopic stress: {sigma_macro:.6e} Pa")
    print(f"[INFO] Effective E_macro: {E_eff_macro:.6e} Pa")

# ---------------------------------------------------------
# 10. Export VTU with displacement + cell-wise fields
# ---------------------------------------------------------
out_vtu = base + "_solution_elasticity_damage.vtu"

if MPI.COMM_WORLD.rank == 0:
    points_out = msh.geometry.x
    uh_arr = uh.x.array.reshape(-1, tdim)

    # cell connectivity
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

    # Extract tensor components from DG0 tensor fields
    eps_arr = eps_proj.x.array.reshape(num_cells_local, 4)
    sig_arr = sig_proj.x.array.reshape(num_cells_local, 4)

    eps_xx = eps_arr[:, 0]
    eps_xy = eps_arr[:, 1]  # ≈ eps_yx
    eps_yy = eps_arr[:, 3]

    sig_xx = sig_arr[:, 0]
    sig_xy = sig_arr[:, 1]
    sig_yy = sig_arr[:, 3]

    vm_arr = vm_fn.x.array
    energy_arr = energy_fn.x.array

    m_out = meshio.Mesh(
        points=points_out,
        cells=[(cell_type, cells_array)],
        point_data={"u": uh_arr},
        cell_data={
            "damage":        [damage_cells],
            "eps_xx":        [eps_xx],
            "eps_yy":        [eps_yy],
            "eps_xy":        [eps_xy],
            "sigma_xx":      [sig_xx],
            "sigma_yy":      [sig_yy],
            "sigma_xy":      [sig_xy],
            "von_mises":     [vm_arr],
            "strain_energy": [energy_arr],
        },
    )
    meshio.write(out_vtu, m_out)
    print(f"[INFO] Wrote solution VTU: {out_vtu}")

print("[DONE] RVE elasticity with material-based porosity + postprocessing complete.")
