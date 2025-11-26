from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

from dolfinx import mesh, fem, io
import ufl
import meshio
import os

# ---------------------------------------------------------
# 0. User inputs: paths to your generated mesh files
# ---------------------------------------------------------
# Base name (adapt to your files!)
base = "rve_output/rve_quad_mesh_Lx1.00_phi30"  # <-- change this!
mesh_file_xdmf = base + ".xdmf"
mesh_file_vtk  = base + ".vtk"

if not os.path.exists(mesh_file_xdmf):
    raise FileNotFoundError(f"Cannot find mesh XDMF file: {mesh_file_xdmf}")
if not os.path.exists(mesh_file_vtk):
    raise FileNotFoundError(f"Cannot find mesh VTK file: {mesh_file_vtk}")

# ---------------------------------------------------------
# 1. Read mesh in dolfinx from XDMF
# ---------------------------------------------------------
with io.XDMFFile(MPI.COMM_WORLD, mesh_file_xdmf, "r") as xdmf:
    msh = xdmf.read_mesh()   # default name="Grid"
    msh.name = "RVE_mesh"

tdim = msh.topology.dim

print(f"[INFO] Mesh: {msh.geometry.x.shape[0]} nodes, "
      f"{msh.topology.index_map(tdim).size_local} cells, dim={tdim}")

# ---------------------------------------------------------
# 2. Read cell-wise damage from VTK via meshio
# ---------------------------------------------------------
mio = meshio.read(mesh_file_vtk)

# Find the 'quad' cell block (or the first one)
cells = None
damage_cells = None

for (ctype, cdata) in zip(mio.cells, mio.cell_data):
    cell_type = ctype.type
    if "damage" in cdata:
        cells = ctype.data
        damage_cells = np.asarray(cdata["damage"]).ravel()
        print(f"[INFO] Found damage for cell type '{cell_type}' with shape {damage_cells.shape}")
        break

if damage_cells is None:
    raise RuntimeError("Could not find 'damage' cell_data in the VTK file.")

num_cells_dolfinx = msh.topology.index_map(tdim).size_local
if damage_cells.size != num_cells_dolfinx:
    print(f"[WARNING] damage_cells.size = {damage_cells.size}, "
          f"num_cells_dolfinx = {num_cells_dolfinx}")
    print("          Assuming ordering matches (both files from same generator).")

print(f"[INFO] Damage stats: min={damage_cells.min():.3f}, "
      f"max={damage_cells.max():.3f}, mean={damage_cells.mean():.3f}")

# ---------------------------------------------------------
# 3. Create a DG0 function for damage and material parameters
# ---------------------------------------------------------
V0 = fem.FunctionSpace(msh, ("DG", 0))   # one value per cell
damage_fn = fem.Function(V0)
damage_fn.x.array[:] = damage_cells

# Material parameters
E0 = 1.0e9   # base Young's modulus (Pa)
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
# 4. Displacement FE space
# ---------------------------------------------------------
Vu = fem.FunctionSpace(msh, ("CG", 1, (tdim,)))
u = fem.Function(Vu)             # unknown displacement
v = ufl.TestFunction(Vu)

# ---------------------------------------------------------
# 5. Small-strain tensor and stress
# ---------------------------------------------------------
def eps(u_):
    return ufl.sym(ufl.grad(u_))

def sigma(u_):
    lam = lambda_fn
    mu  = mu_fn
    return 2.0 * mu * eps(u_) + lam * ufl.tr(eps(u_)) * ufl.Identity(tdim)

# ---------------------------------------------------------
# 6. Boundary conditions: left fixed, right displaced
# ---------------------------------------------------------
coords = msh.geometry.x
x_min = np.min(coords[:, 0])
x_max = np.max(coords[:, 0])
u0 = 0.001  # applied displacement in x

def left_boundary(x):
    return np.isclose(x[0], x_min)

def right_boundary(x):
    return np.isclose(x[0], x_max)

facets_left  = mesh.locate_entities_boundary(msh, tdim-1, left_boundary)
facets_right = mesh.locate_entities_boundary(msh, tdim-1, right_boundary)

dofs_left  = fem.locate_dofs_topological(Vu, tdim-1, facets_left)
dofs_right = fem.locate_dofs_topological(Vu, tdim-1, facets_right)

# Left BC: u = (0,0)
bc_left = fem.dirichletbc(PETSc.ScalarType((0.0, 0.0)), dofs_left, Vu)

# Right BC: u = (u0, 0)
u_right = fem.Function(Vu)
with u_right.vector.localForm() as loc:
    loc.set(0.0)
# fill x-components
u_right.x.array[0::tdim] = u0
# y-components already 0
bc_right = fem.dirichletbc(u_right, dofs_right)

bcs = [bc_left, bc_right]

# ---------------------------------------------------------
# 7. Variational formulation and solve
# ---------------------------------------------------------
a = ufl.inner(sigma(u), eps(v)) * ufl.dx
L = ufl.inner(ufl.as_vector((0.0, 0.0)), v) * ufl.dx

problem = fem.petsc.LinearProblem(a, L, bcs=bcs, u=u,
                                  petsc_options={"ksp_type": "preonly",
                                                 "pc_type": "lu"})
uh = problem.solve()
print("[INFO] Linear elasticity problem solved.")

# ---------------------------------------------------------
# 8. Export VTU with displacement and damage
# ---------------------------------------------------------
out_vtu = base + "_solution_elasticity_damage.vtu"

if MPI.COMM_WORLD.rank == 0:
    # cell connectivity
    ct = msh.topology.connectivity(tdim, 0)
    cells_dolfinx = ct.array.reshape(-1, ct.num_nodes)
    points_out = msh.geometry.x
    uh_arr = uh.x.array.reshape(-1, tdim)

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
