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
#    Needs:
#       base + ".xdmf" : mesh of artificial porous material
#       base + ".vtk"  : same mesh, with cell_data["damage"]
# ---------------------------------------------------------

base = "rve_output/rve_quad_mesh_Lx1.00_phi30"  # <-- CHANGE THIS

mesh_file_xdmf = base + ".xdmf"
mesh_file_vtk  = base + ".vtk"

if not os.path.exists(mesh_file_xdmf):
    raise FileNotFoundError(f"Cannot find mesh XDMF file: {mesh_file_xdmf}")
if not os.path.exists(mesh_file_vtk):
    raise FileNotFoundError(f"Cannot find mesh VTK file: {mesh_file_vtk}")

if MPI.COMM_WORLD.rank == 0:
    print(f"[INFO] Using XDMF mesh:   {mesh_file_xdmf}")
    print(f"[INFO] Using VTK damage: {mesh_file_vtk}")


# ---------------------------------------------------------
# 1. Read mesh (artificial porous material) from XDMF
# ---------------------------------------------------------
with io.XDMFFile(MPI.COMM_WORLD, mesh_file_xdmf, "r") as xdmf:
    msh = xdmf.read_mesh(name="Grid")
    msh.name = "RVE_mesh"

tdim = msh.topology.dim
msh.topology.create_connectivity(tdim, 0)

if MPI.COMM_WORLD.rank == 0:
    print(f"[INFO] Mesh: {msh.geometry.x.shape[0]} nodes, "
          f"{msh.topology.index_map(tdim).size_local} cells, dim={tdim}")

# For boundary detection (RVE outer box)
coords = msh.geometry.x
x_min = np.min(coords[:, 0])
x_max = np.max(coords[:, 0])
y_min = np.min(coords[:, 1])
y_max = np.max(coords[:, 1])


# ---------------------------------------------------------
# 2. Read cell-wise damage from VTK (artificial porosity)
# ---------------------------------------------------------
mio = meshio.read(mesh_file_vtk)

if "damage" not in mio.cell_data:
    raise RuntimeError("Could not find 'damage' in cell_data of the VTK file.")

damage_list = mio.cell_data["damage"]   # list (one entry per cell block)
if len(damage_list) != 1 and MPI.COMM_WORLD.rank == 0:
    print(f"[WARNING] Expected one cell block for damage, found {len(damage_list)}.")

damage_cells = np.asarray(damage_list[0]).ravel()

num_cells_dolfinx = msh.topology.index_map(tdim).size_local
if damage_cells.size != num_cells_dolfinx and MPI.COMM_WORLD.rank == 0:
    print(f"[WARNING] damage_cells.size = {damage_cells.size}, "
          f"num_cells_dolfinx = {num_cells_dolfinx}")
    print("          Assuming ordering matches (same generator).")

if MPI.COMM_WORLD.rank == 0:
    print(f"[INFO] Damage stats: min={damage_cells.min():.3f}, "
          f"max={damage_cells.max():.3f}, mean={damage_cells.mean():.3f}")


# ---------------------------------------------------------
# 3. DG0 damage field + diffusion coefficient D(damage)
# ---------------------------------------------------------
# DG0 space: one value per cell
V0 = fem.functionspace(msh, ("DG", 0))
damage_fn = fem.Function(V0)
damage_fn.x.array[:] = damage_cells

# Strongly damage-controlled diffusion:
#   - almost intact matrix: very low diffusion
#   - highly damaged/porous: very high diffusion
#
# D(d) = D_matrix + (D_pore - D_matrix) * d^q_D
#
# so:
#   d = 0   -> D = D_matrix (blocking)
#   d = 1   -> D = D_pore   (fast path)

D_matrix = 1.0e-8   # very low diffusion in intact solid
D_pore   = 1.0e0    # much larger diffusion in damaged pores
q_D      = 6.0      # strong nonlinearity (only high damage really matters)

D_eff_cells = D_matrix + (D_pore - D_matrix) * damage_cells**q_D
D_eff_cells = np.clip(D_eff_cells, D_matrix, D_pore)

D_fn = fem.Function(V0)
D_fn.x.array[:] = D_eff_cells

if MPI.COMM_WORLD.rank == 0:
    print(f"[DIFF] D_matrix = {D_matrix:.3e}, D_pore = {D_pore:.3e}")
    print(f"[DIFF] D_eff range: {D_eff_cells.min():.3e} .. {D_eff_cells.max():.3e}")


# ---------------------------------------------------------
# 4. Scalar FE space for hydrogen concentration c(x, t)
# ---------------------------------------------------------
Vc = fem.functionspace(msh, ("CG", 1))    # continuous scalar field
c   = fem.Function(Vc, name="c")          # current time step
c_n = fem.Function(Vc, name="c_n")        # previous time step

# Initial condition: no hydrogen in solid at t=0
c_n.x.array[:] = 0.0
c.x.array[:]   = 0.0


# ---------------------------------------------------------
# 5. Boundary conditions (solid fully in water)
# ---------------------------------------------------------
# RVE is fully surrounded by water with fixed hydrogen concentration c_water
c_water = 1.0  # Dirichlet value at boundary

def water_boundary(x):
    return np.logical_or.reduce((
        np.isclose(x[0], x_min),
        np.isclose(x[0], x_max),
        np.isclose(x[1], y_min),
        np.isclose(x[1], y_max),
    ))

facets_water = mesh.locate_entities_boundary(msh, tdim - 1, water_boundary)
dofs_c_water = fem.locate_dofs_topological(Vc, tdim - 1, facets_water)

c_water_val = fem.Constant(msh, PETSc.ScalarType(c_water))
bc_c_water = fem.dirichletbc(c_water_val, dofs_c_water, Vc)

bcs_c = [bc_c_water]


# ---------------------------------------------------------
# 6. Time-stepping setup
# ---------------------------------------------------------
# With D_pore ~ 1 and domain size ~1, t_end ~ 0.1 gives visible penetration.
dt    = 1.0e-3   # time step
t_end = 0.1      # final time
num_steps = int(t_end / dt)

if MPI.COMM_WORLD.rank == 0:
    print(f"[DIFF] Time stepping: dt={dt:g}, steps={num_steps}, "
          f"c_water={c_water}, D_matrix={D_matrix:g}, D_pore={D_pore:g}")


# ---------------------------------------------------------
# 7. Weak form of diffusion equation (backward Euler)
# ---------------------------------------------------------
# PDE: ∂c/∂t = ∇·(D(d) ∇c)
#
# Weak form (for each step n -> n+1):
#
#   ∫ (c^{n+1} - c^n)/dt * w dΩ + ∫ D(d) ∇c^{n+1} · ∇w dΩ = 0
#
# → ∫ c^{n+1}/dt * w dΩ + ∫ D(d) ∇c^{n+1} · ∇w dΩ = ∫ c^n/dt * w dΩ

c_trial = ufl.TrialFunction(Vc)
w       = ufl.TestFunction(Vc)
dx      = ufl.Measure("cell", domain=msh)

a_c = (c_trial * w / dt
       + D_fn * ufl.dot(ufl.grad(c_trial), ufl.grad(w))) * dx
L_c = (c_n * w / dt) * dx

problem_c = LinearProblem(
    a_c, L_c, bcs=bcs_c, u=c,
    petsc_options={"ksp_type": "cg", "pc_type": "hypre"},
    petsc_options_prefix="diff_"
)


# ---------------------------------------------------------
# 8. Time loop + XDMF output
# ---------------------------------------------------------
diff_outfile = base + "_hydrogen_diffusion_damage_channels.xdmf"

with io.XDMFFile(MPI.COMM_WORLD, diff_outfile, "w") as xdmf_c:
    xdmf_c.write_mesh(msh)

    t = 0.0
    xdmf_c.write_function(c_n, t)  # t = 0 initial state
    if MPI.COMM_WORLD.rank == 0:
        c_min = c_n.x.array.min()
        c_max = c_n.x.array.max()
        print(f"[DIFF] t = {t:.4e}  init, c_min={c_min:.3e}, c_max={c_max:.3e}")

    for n in range(1, num_steps + 1):
        t += dt

        # Solve for c^{n+1}
        c_sol = problem_c.solve()

        # Update previous solution
        c_n.x.array[:] = c_sol.x.array

        # Diagnostics: min/max concentration so you SEE the evolution
        if MPI.COMM_WORLD.rank == 0 and (n % 20 == 0 or n == num_steps):
            c_min = c_sol.x.array.min()
            c_max = c_sol.x.array.max()
            print(f"[DIFF] step {n}/{num_steps}, t={t:.4e}, "
                  f"c_min={c_min:.3e}, c_max={c_max:.3e}")

        # Write every 20 steps (or last step)
        if (n % 20 == 0) or (n == num_steps):
            xdmf_c.write_function(c_sol, t)

if MPI.COMM_WORLD.rank == 0:
    print(f"[DIFF] Diffusion simulation complete. Results in: {diff_outfile}")
