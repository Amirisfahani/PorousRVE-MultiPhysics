from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

from dolfinx import mesh, fem, io
import ufl
import meshio
import os

# ---------------------------------------------------------
# 0. User input: base name of your RVE files (without extension)
#    Make sure these files exist:
#      <base>.xdmf  (mesh)
#      <base>.vtk   (mesh + cell_data["damage"])
# ---------------------------------------------------------

# EXAMPLE: if you have
#   rve_output/rve_quad_mesh_Lx1.00_phi30.vtk
#   rve_output/rve_quad_mesh_Lx1.00_phi30.xdmf
# then set:
base = "rve_output/rve_quad_mesh_Lx1.00_phi30"  # <-- CHANGE THIS TO YOUR FILENAME BASE

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
    # meshio normally writes <Grid Name="Grid">
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

# When we wrote the mesh, we used: cell_data={"damage": [cell_damage]}
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
    print("          Assuming ordering matches (both files from same generator).")

print(f"[INFO] Damage stats: min={damage_cells.min():.3f}, "
      f"max={damage_cells.max():.3f}, mean={damage_cells.mean():.3f}")

# ---------------------------------------------------------
# 3. Create DG0 function for damage, and material parameters
# ---------------------------------------------------------
# New dolfinx API: use fem.functionspace, not FunctionSpace
V0 = fem.functionspace(msh, ("DG", 0))   # piecewise-constant per cell
damage_fn = fem.Function(V0)
damage_fn.x.array[:] = damage_cells

# Base material parameters
E0 = 1.0e9   # Young's modulus of undamaged solid (Pa) - arbitrary
nu = 0.3     # Poisson's ratio
p  = 2.0     # exponent for damage -> E mapping
Emin_ratio = 1e-3   # minimum stiffness fraction for "void-ish" elements

# Map damage -> effective Young's modulus
E_eff_cells = (1.0 - damage_cells) ** p * E0
E_eff_cells = np.maximum(E_eff_cells, Emin_ratio * E0)

# Convert to Lamé parameters per cell
lambda_cells = (E_eff_cells * nu) / ((1 + nu) * (1 - 2 * nu))
mu_cells     = E_eff_cells / (2 * (1 + nu))

lambda_fn = fem.Function(V0)
mu_fn     = fem.Function(V0)
lambda_fn.x.array[:] = lambda_cells
mu_fn.x.array[:]     = mu_cells

print(f"[INFO] Effective E range: {E_eff_cells.min()::.3e} .. {E_eff_cells.max():.3e}")

# ---------------------------------------------------------
# 4. Displacement FE space (2D vector field)
# ---------------------------------------------------------
# New API: fem.functionspace for CG1 vector-valued
Vu = fem.functionspace(msh, ("CG", 1, (tdim,)))  # CG1, vector-valued
u = fem.Function(Vu)             # unknown displacement u(x)
v = ufl.TestFunction(Vu)

# ---------------------------------------------------------
# 5. Small-strain tensor and stress with spatially varying lambda, mu
# ---------------------------------------------------------
def eps(u_):
    grad_u = ufl.grad(u_)
    # For 2D plane strain, extract 2x2 submatrix from the gradient
    eps_tensor = 0.5 * (grad_u + ufl.transpose(grad_u))
    return eps_tensor[:2, :2]

def sigma(u_):
    # lambda_fn and mu_fn are DG0 (piecewise constant) functions
    lam = lambda_fn
    mu  = mu_fn
    eps_u = eps(u_)
    return 2.0 * mu * eps_u + lam * ufl.tr(eps_u) * ufl.Identity(2)

# ---------------------------------------------------------
# 6. Boundary conditions:
#    - Left boundary: x = x_min --> u = (0, 0)
#    - Right boundary: x = x_max -> u = (u0, 0)
# ---------------------------------------------------------
coords = msh.geometry.x
x_min = np.min(coords[:, 0])
x_max = np.max(coords[:, 0])

u0 = 0.001  # applied displacement in x-direction on right side

def left_boundary(x):
    return np.isclose(x[0], x_min)

def right_boundary(x):
    return np.isclose(x[0], x_max)

facets_left  = mesh.locate_entities_boundary(msh, tdim-1, left_boundary)
facets_right = mesh.locate_entities_boundary(msh, tdim-1, right_boundary)

dofs_left  = fem.locate_dofs_topological(Vu, tdim-1, facets_left)
dofs_right = fem.locate_dofs_topological(Vu, tdim-1, facets_right)

# Left BC: u = (0, 0)
u_left_val = PETSc.ScalarType((0.0, 0.0))
bc_left = fem.dirichletbc(u_left_val, dofs_left, Vu)

# Right BC: u = (u0, 0)
u_right_val = PETSc.ScalarType((u0, 0.0))
bc_right = fem.dirichletbc(u_right_val, dofs_right, Vu)

bcs = [bc_left, bc_right]

# ---------------------------------------------------------
# 7. Variational formulation and solve:
#    ∫_Ω σ(u) : ε(v) dx = 0     (no body forces)
# ---------------------------------------------------------
a = ufl.inner(sigma(u), eps(v)) * ufl.dx
L = ufl.inner(ufl.as_vector((0.0, 0.0)), v) * ufl.dx   # RHS = 0

problem = fem.petsc.LinearProblem(
    a, L, bcs=bcs, u=u,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)

uh = problem.solve()
print("[INFO] Linear elasticity problem solved.")

# ---------------------------------------------------------
# 8. Export VTU with displacement and damage for ParaView
# ---------------------------------------------------------
out_vtu = base + "_solution_elasticity_damage.vtu"

if MPI.COMM_WORLD.rank == 0:
    # Build connectivity (cells -> nodes)
    msh.topology.create_connectivity(tdim, 0)
    ct = msh.topology.connectivity(tdim, 0)
    cells_dolfinx = ct.array.reshape(-1, ct.num_nodes)
    points_out = msh.geometry.x
    uh_arr = uh.x.array.reshape(-1, tdim)

    # Guess cell type: quad if 4 nodes per cell, else polygon
    cell_type = "quad" if cells_dolfinx.shape[1] == 4 else "polygon"

    m_out = meshio.Mesh(
        points=points_out,
        cells=[(cell_type, cells_dolfinx)],
        point_data={"u": uh_arr},
        cell_data={"damage": [damage_cells]},
    )
    meshio.write(out_vtu, m_out)
    print(f"[INFO] Wrote solution VTU: {out_vtu}")

print("[DONE] RVE elasticity with material-based porosity complete.")
