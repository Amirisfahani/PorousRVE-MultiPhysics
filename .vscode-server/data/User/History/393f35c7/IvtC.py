from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
import ufl
import meshio
import os

comm = MPI.COMM_WORLD
rank = comm.rank

# ---------------------------------------------------------
# 0. User input
# ---------------------------------------------------------
base = "rve_output/rve_quad_mesh_Lx1.00_phi30"  # <-- CHANGE if needed

mesh_file_xdmf   = base + ".xdmf"
mesh_file_vtk    = base + ".vtk"
diffusion_xdmf   = base + "_hydrogen_diffusion_damage_channels.xdmf"

if not os.path.exists(mesh_file_xdmf):
    raise FileNotFoundError(mesh_file_xdmf)
if not os.path.exists(mesh_file_vtk):
    raise FileNotFoundError(mesh_file_vtk)
if not os.path.exists(diffusion_xdmf):
    raise FileNotFoundError(diffusion_xdmf)

if rank == 0:
    print(f"[INFO] Mesh XDMF:    {mesh_file_xdmf}")
    print(f"[INFO] Damage VTK:   {mesh_file_vtk}")
    print(f"[INFO] Diffusion XDMF (c): {diffusion_xdmf}")


# ---------------------------------------------------------
# 1. Read mesh (same RVE as diffusion)
# ---------------------------------------------------------
with io.XDMFFile(comm, mesh_file_xdmf, "r") as xdmf:
    msh = xdmf.read_mesh(name="Grid")
    msh.name = "RVE_mesh"

tdim = msh.topology.dim
msh.topology.create_connectivity(tdim, 0)

if rank == 0:
    print(f"[INFO] Mesh: {msh.geometry.x.shape[0]} nodes, "
          f"{msh.topology.index_map(tdim).size_local} cells, dim={tdim}")

coords = msh.geometry.x
x_min = np.min(coords[:, 0])
x_max = np.max(coords[:, 0])
y_min = np.min(coords[:, 1])
y_max = np.max(coords[:, 1])


# ---------------------------------------------------------
# 2. Read cell-wise damage (for stiffness)
# ---------------------------------------------------------
mio = meshio.read(mesh_file_vtk)

if "damage" not in mio.cell_data:
    raise RuntimeError("Could not find 'damage' in VTK cell_data.")

damage_list = mio.cell_data["damage"]
if len(damage_list) != 1 and rank == 0:
    print(f"[WARNING] Expected one cell block for damage, found {len(damage_list)}.")

damage_cells = np.asarray(damage_list[0]).ravel()
num_cells = msh.topology.index_map(tdim).size_local

if damage_cells.size != num_cells and rank == 0:
    print(f"[WARNING] damage_cells.size = {damage_cells.size}, num_cells = {num_cells}")
    print("          Assuming ordering matches.")

if rank == 0:
    print(f"[INFO] Damage stats: min={damage_cells.min():.3f}, "
          f"max={damage_cells.max():.3f}, mean={damage_cells.mean():.3f}")


# ---------------------------------------------------------
# 3. Elastic material with damage-dependent stiffness (as before)
# ---------------------------------------------------------
V0 = fem.functionspace(msh, ("DG", 0))   # one value per cell
damage_fn = fem.Function(V0)
damage_fn.x.array[:] = damage_cells

# Base elastic material
E0 = 1.0e9   # Pa
nu = 0.3
p  = 2.0
Emin_ratio = 1e-3

E_eff_cells = (1.0 - damage_cells)**p * E0
E_eff_cells = np.maximum(E_eff_cells, Emin_ratio * E0)

lambda_cells = (E_eff_cells * nu) / ((1 + nu) * (1 - 2 * nu))
mu_cells     = E_eff_cells / (2 * (1 + nu))

lambda_fn = fem.Function(V0)
mu_fn     = fem.Function(V0)
lambda_fn.x.array[:] = lambda_cells
mu_fn.x.array[:]     = mu_cells

if rank == 0:
    print(f"[INFO] E_eff range: {E_eff_cells.min():.3e} .. {E_eff_cells.max():.3e}")


# ---------------------------------------------------------
# 4. Read concentration field c(x) from diffusion XDMF
#    (take the last available time step)
# ---------------------------------------------------------
Vc = fem.functionspace(msh, ("CG", 1))
c = fem.Function(Vc, name="c")

with io.XDMFFile(comm, diffusion_xdmf, "r") as xdmf_c:
    # Read all time steps from the temporal collection named "c"
    # The last one will remain in the function after the loop
    try:
        while True:
            xdmf_c.read_function(c)
    except RuntimeError:
        # dolfinx raises RuntimeError when no more time steps
        pass

# Check c min/max
c_values = c.x.array
c_min = np.min(c_values)
c_max = np.max(c_values)
if rank == 0:
    print(f"[INFO] Loaded concentration c(x) from diffusion file.")
    print(f"[INFO] c range: min={c_min:.3e}, max={c_max:.3e}")


# ---------------------------------------------------------
# 5. Swelling elastostatics: ε_total = ε(u) - ε_sw(c)
# ---------------------------------------------------------
tdim = msh.topology.dim
Vu = fem.functionspace(msh, ("CG", 1, (tdim,)))
u = fem.Function(Vu, name="u")
u_trial = ufl.TrialFunction(Vu)
v       = ufl.TestFunction(Vu)

def eps(u_):
    grad_u = ufl.grad(u_)
    return ufl.as_matrix((
        (grad_u[0, 0], 0.5*(grad_u[0, 1] + grad_u[1, 0])),
        (0.5*(grad_u[1, 0] + grad_u[0, 1]), grad_u[1, 1])
    ))

# Swelling coefficient (tune this!)
beta = 0.01  # strain per unit concentration

def eps_sw(c_):
    # isotropic swelling: beta * c * I
    return beta * c_ * ufl.Identity(2)

def sigma_elastic(u_):
    lam = lambda_fn
    mu  = mu_fn
    e = eps(u_)
    return 2.0 * mu * e + lam * ufl.tr(e) * ufl.Identity(2)

def sigma_swelling(c_):
    lam = lambda_fn
    mu  = mu_fn
    esw = eps_sw(c_)
    return 2.0 * mu * esw + lam * ufl.tr(esw) * ufl.Identity(2)

# Total stress (for postprocessing): σ = C:(ε(u) - ε_sw) = σ_elastic(u) - σ_swelling(c)
def sigma_total(u_, c_):
    return sigma_elastic(u_) - sigma_swelling(c_)


# ---------------------------------------------------------
# 6. Boundary conditions for swelling problem
# ---------------------------------------------------------
# We'll fix the left edge (u=0) to avoid rigid body motion,
# and leave other sides free. Swelling will push against the left constraint.

def left_boundary(x):
    return np.isclose(x[0], x_min)

facets_left = mesh.locate_entities_boundary(msh, tdim-1, left_boundary)
dofs_left   = fem.locate_dofs_topological(Vu, tdim-1, facets_left)

u_left_val = PETSc.ScalarType((0.0, 0.0))
bc_left = fem.dirichletbc(u_left_val, dofs_left, Vu)
bcs = [bc_left]


# ---------------------------------------------------------
# 7. Variational problem with swelling
# ---------------------------------------------------------
dx = ufl.Measure("cell", domain=msh)

# LHS: usual stiffness
a = ufl.inner(sigma_elastic(u_trial), eps(v)) * dx

# RHS: swelling "load"
L = ufl.inner(sigma_swelling(c), eps(v)) * dx

problem = LinearProblem(
    a, L, bcs=bcs, u=u,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="swelling_"
)

uh = problem.solve()
if rank == 0:
    print("[INFO] Swelling mechanics problem solved.")


# ---------------------------------------------------------
# 8. Postprocessing: total stress, von Mises, strain energy
# ---------------------------------------------------------
# Project σ_total to DG0 tensor space
W = fem.functionspace(msh, ("DG", 0, (tdim, tdim)))
tau = ufl.TrialFunction(W)
w_t = ufl.TestFunction(W)

sigma_tot_expr = sigma_total(uh, c)

a_sig = ufl.inner(tau, w_t) * dx
L_sig = ufl.inner(sigma_tot_expr, w_t) * dx
sig_proj = fem.Function(W, name="sigma")
problem_sig = LinearProblem(
    a_sig, L_sig, u=sig_proj,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="swelling_sig_"
)
problem_sig.solve()

# Von Mises from σ_total
sx = sigma_tot_expr[0, 0]
sy = sigma_tot_expr[1, 1]
txy = sigma_tot_expr[0, 1]
sigma_vm_expr = ufl.sqrt(sx**2 + sy**2 - sx*sy + 3.0*txy**2)

vm_trial = ufl.TrialFunction(V0)
vm_test  = ufl.TestFunction(V0)
a_vm = ufl.inner(vm_trial, vm_test) * dx
L_vm = ufl.inner(sigma_vm_expr, vm_test) * dx
vm_fn = fem.Function(V0, name="von_mises")
problem_vm = LinearProblem(
    a_vm, L_vm, u=vm_fn,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="swelling_vm_"
)
problem_vm.solve()

# Strain energy density ψ = 1/2 σ : ε_total
eps_total_expr = eps(uh) - eps_sw(c)
energy_expr = 0.5 * ufl.inner(sigma_tot_expr, eps_total_expr)

en_trial = ufl.TrialFunction(V0)
en_test  = ufl.TestFunction(V0)
a_en = ufl.inner(en_trial, en_test) * dx
L_en = ufl.inner(energy_expr, en_test) * dx
energy_fn = fem.Function(V0, name="strain_energy")
problem_en = LinearProblem(
    a_en, L_en, u=energy_fn,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="swelling_en_"
)
problem_en.solve()

if rank == 0:
    print("[INFO] Postprocessing (σ_total, von Mises, energy) done.")


# ---------------------------------------------------------
# 9. Export VTU for Paraview
# ---------------------------------------------------------
out_vtu = base + "_swelling_mechanics.vtu"

if rank == 0:
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

    # Extract σ components from DG0 tensor field
    sig_arr = sig_proj.x.array.reshape(num_cells_local, 4)
    sig_xx = sig_arr[:, 0]
    sig_xy = sig_arr[:, 1]
    sig_yy = sig_arr[:, 3]

    vm_arr = vm_fn.x.array
    energy_arr = energy_fn.x.array
    c_arr = c.x.array

    m_out = meshio.Mesh(
        points=points_out,
        cells=[(cell_type, cells_array)],
        point_data={"u": uh_arr,
                    "c": c_arr},
        cell_data={
            "damage":        [damage_cells],
            "sigma_xx":      [sig_xx],
            "sigma_yy":      [sig_yy],
            "sigma_xy":      [sig_xy],
            "von_mises":     [vm_arr],
            "strain_energy": [energy_arr],
        },
    )
    meshio.write(out_vtu, m_out)
    print(f"[INFO] Wrote swelling mechanics VTU: {out_vtu}")

if rank == 0:
    print("[DONE] Swelling mechanics on RVE using diffusion result complete.")
