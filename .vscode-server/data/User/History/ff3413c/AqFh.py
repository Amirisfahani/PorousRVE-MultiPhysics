import numpy as np
import meshio

# -----------------------------
# User parameters
# -----------------------------
Lx = 1.0      # domain length in x
Ly = 1.0      # domain length in y
dx = 0.05     # grid spacing
phi = 0.3     # target porosity (0..1) as probability of bond breaking
m   = 3.0     # horizon factor (delta = m * dx)

# -----------------------------
# 1. Build regular grid of points
# -----------------------------
Nx = int(np.floor(Lx / dx)) + 1
Ny = int(np.floor(Ly / dx)) + 1
N  = Nx * Ny

print(f"Grid: Nx = {Nx}, Ny = {Ny}, total points N = {N}")

# Create coordinates: points[i] = (x, y)
xs = np.linspace(0.0, Lx, Nx)
ys = np.linspace(0.0, Ly, Ny)
X, Y = np.meshgrid(xs, ys, indexing="xy")
points = np.column_stack([X.ravel(), Y.ravel(), np.zeros(N)])  # z=0 for 2D

# Helper to map (i,j) -> flat index
def ij_to_id(i, j):
    return j * Nx + i

# -----------------------------
# 2. Compute neighbors and bonds
# -----------------------------
delta  = m * dx
delta2 = delta * delta

N_total   = np.zeros(N, dtype=int)
N_broken  = np.zeros(N, dtype=int)

# First pass: find all neighbor pairs
neighbor_pairs = []  # list of (i,j) with i < j

print("Finding neighbors ...")
for i in range(N):
    xi, yi, _ = points[i]
    # Naive double loop (OK for small RVEs; for large N you would need KD-tree)
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
print(f"Total bonds (before damage): {total_bonds}")

# -----------------------------
# 3. Randomly break bonds to reach target porosity
# -----------------------------
rng = np.random.default_rng()  # modern NumPy RNG
r   = rng.random(total_bonds)  # random number per bond in [0,1)

broken_mask = r < phi  # True where bond is broken
broken_pairs = neighbor_pairs[broken_mask]

for (i, j) in broken_pairs:
    N_broken[i] += 1
    N_broken[j] += 1

broken_bonds = broken_pairs.shape[0]
realized_porosity = broken_bonds / total_bonds if total_bonds > 0 else 0.0
print(f"Broken bonds: {broken_bonds}")
print(f"Realized global porosity (bond-based) ≈ {realized_porosity:.3f}")

# -----------------------------
# 4. Compute local damage per point: d(i) = Nb(i) / N(i)
# -----------------------------
damage = np.zeros(N, dtype=float)
mask_total = N_total > 0
damage[mask_total] = N_broken[mask_total] / N_total[mask_total]

print(f"Damage stats: min={damage.min():.3f}, max={damage.max():.3f}, mean={damage.mean():.3f}")

# -----------------------------
# 5. Write point cloud VTK (like your C++ program)
# -----------------------------
point_cloud = meshio.Mesh(
    points=points,
    cells=[("vertex", np.arange(N).reshape(-1, 1))],
    point_data={"damage": damage},
)

meshio.write("rve_points_with_damage.vtk", point_cloud)
print("Wrote point-cloud VTK: rve_points_with_damage.vtk")

# -----------------------------
# 6. Build a quad mesh on the same grid (for FE)
# -----------------------------
# We'll use the regular grid connectivity:
# each cell has nodes: (i,j), (i+1,j), (i+1,j+1), (i,j+1)

quad_cells = []
cell_damage = []

for j in range(Ny - 1):
    for i in range(Nx - 1):
        n0 = ij_to_id(i,   j)
        n1 = ij_to_id(i+1, j)
        n2 = ij_to_id(i+1, j+1)
        n3 = ij_to_id(i,   j+1)
        quad_cells.append([n0, n1, n2, n3])
        # Cell damage = average of its 4 corner nodes
        cd = 0.25 * (damage[n0] + damage[n1] + damage[n2] + damage[n3])
        cell_damage.append(cd)

quad_cells  = np.array(quad_cells, dtype=int)
cell_damage = np.array(cell_damage, dtype=float)

print(f"Number of quad cells: {quad_cells.shape[0]}")

quad_mesh = meshio.Mesh(
    points=points,
    cells=[("quad", quad_cells)],
    cell_data={"damage": [cell_damage]},
)

meshio.write("rve_quad_mesh.vtk", quad_mesh)
print("Wrote quad mesh VTK: rve_quad_mesh.vtk")

# Optionally also write XDMF (better for dolfinx)
meshio.write("rve_quad_mesh.xdmf", quad_mesh)
print("Wrote quad mesh XDMF: rve_quad_mesh.xdmf")
