import numpy as np
import meshio
import os

# -----------------------------
# 1. User parameters
# -----------------------------
print("\n===== 2D Nanoporous RVE Generator (bond-based damage) =====")

Lx = float(input("Enter domain length in x (Lx): "))
Ly = float(input("Enter domain length in y (Ly): "))
dx = float(input("Enter grid spacing (dx): "))
phi = float(input("Enter porosity parameter phi (0..1, probability of bond break): "))
m   = float(input("Enter horizon factor m (delta = m*dx): "))

if dx <= 0.0 or Lx <= 0.0 or Ly <= 0.0 or phi < 0.0 or phi > 1.0 or m <= 0.0:
    raise ValueError("Invalid input parameters.")

# -----------------------------
# 2. Create output folder
# -----------------------------
base_dir = os.getcwd()  # directory where you run the script
out_dir = os.path.join(base_dir, "rve_output")
os.makedirs(out_dir, exist_ok=True)

print(f"\n[INFO] Output directory: {out_dir}")

# -----------------------------
# 3. Build regular grid of points
# -----------------------------
Nx = int(np.floor(Lx / dx)) + 1
Ny = int(np.floor(Ly / dx)) + 1
N  = Nx * Ny

print(f"[INFO] Grid: Nx = {Nx}, Ny = {Ny}, total points N = {N}")

xs = np.linspace(0.0, Lx, Nx)
ys = np.linspace(0.0, Ly, Ny)
X, Y = np.meshgrid(xs, ys, indexing="xy")
points = np.column_stack([X.ravel(), Y.ravel(), np.zeros(N)])  # z = 0 for 2D

def ij_to_id(i, j):
    return j * Nx + i

# -----------------------------
# 4. Compute neighbors and bonds within horizon
# -----------------------------
delta  = m * dx
delta2 = delta * delta

N_total   = np.zeros(N, dtype=int)   # total bonds for each point
N_broken  = np.zeros(N, dtype=int)   # broken bonds for each point

neighbor_pairs = []

print("[INFO] Finding neighbors (this may take a bit for large grids)...")
for i in range(N):
    xi, yi, _ = points[i]
    for j in range(i + 1, N):
        xj, yj, _ = points[j]
        dx_ = xi - xj
        dy_ = yi - yj
        if dx_*dx_ + dy_*dy_ <= delta2:
            neighbor_pairs.append((i, j))
            N_total[i] += 1
            N_total[j] += 1

neighbor_pairs = np.array(neighbor_pairs, dtype=int)
total_bonds = neighbor_pairs.shape[0]
print(f"[INFO] Total bonds (before damage): {total_bonds}")

# -----------------------------
# 5. Randomly break bonds based on phi
# -----------------------------
rng = np.random.default_rng()
r   = rng.random(total_bonds)   # one random number per bond in [0,1)

broken_mask   = r < phi         # True => bond is broken
broken_pairs  = neighbor_pairs[broken_mask]
broken_bonds  = broken_pairs.shape[0]

for (i, j) in broken_pairs:
    N_broken[i] += 1
    N_broken[j] += 1

realized_porosity = broken_bonds / total_bonds if total_bonds > 0 else 0.0

print(f"[INFO] Broken bonds: {broken_bonds}")
print(f"[INFO] Realized global bond-porosity ≈ {realized_porosity:.3f}")

# -----------------------------
# 6. Compute local damage per point
# -----------------------------
damage = np.zeros(N, dtype=float)
mask_total = N_total > 0
damage[mask_total] = N_broken[mask_total] / N_total[mask_total]

print(f"[INFO] Damage stats: min={damage.min():.3f}, "
      f"max={damage.max():.3f}, mean={damage.mean():.3f}")

# -----------------------------
# 7. Write VTK point cloud
# -----------------------------
pc_filename = os.path.join(
    out_dir, f"rve_points_Lx{Lx:.2f}_phi{int(phi*100)}.vtk"
)

point_cloud = meshio.Mesh(
    points=points,
    cells=[("vertex", np.arange(N).reshape(-1, 1))],
    point_data={"damage": damage},
)

meshio.write(pc_filename, point_cloud)
print(f"[INFO] Wrote point-cloud VTK: {pc_filename}")

# -----------------------------
# 8. Build quad mesh & cell-wise damage (for FE / FEniCSx)
# -----------------------------
quad_cells = []
cell_damage = []

for j in range(Ny - 1):
    for i in range(Nx - 1):
        n0 = ij_to_id(i,   j)
        n1 = ij_to_id(i+1, j)
        n2 = ij_to_id(i+1, j+1)
        n3 = ij_to_id(i,   j+1)
        quad_cells.append([n0, n1, n2, n3])
        cd = 0.25 * (damage[n0] + damage[n1] + damage[n2] + damage[n3])
        cell_damage.append(cd)

quad_cells  = np.array(quad_cells, dtype=int)
cell_damage = np.array(cell_damage, dtype=float)

qm_vtk_filename  = os.path.join(
    out_dir, f"rve_quad_mesh_Lx{Lx:.2f}_phi{int(phi*100)}.vtk"
)
qm_xdmf_filename = os.path.join(
    out_dir, f"rve_quad_mesh_Lx{Lx:.2f}_phi{int(phi*100)}.xdmf"
)

quad_mesh = meshio.Mesh(
    points=points,
    cells=[("quad", quad_cells)],
    cell_data={"damage": [cell_damage]},
)

meshio.write(qm_vtk_filename, quad_mesh)
print(f"[INFO] Wrote quad mesh VTK: {qm_vtk_filename}")

# also XDMF (with HDF5) for dolfinx later
meshio.write(qm_xdmf_filename, quad_mesh)
print(f"[INFO] Wrote quad mesh XDMF: {qm_xdmf_filename}")

print("\n[DONE] RVE generation finished.")
print("       Files are in:", out_dir)
print("       You can now open the VTK files in ParaView,")
print("       and use the XDMF file in dolfinx for FE simulations.")
