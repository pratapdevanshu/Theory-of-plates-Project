import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ----------------------------------------------------- Input -------------------------------------------------
# Material and geometry
E = 200000.0        # MPa
nu = 0.3            # Poisson's ratio

# -----------------------------------------------------
# User Input: Plate Dimensions & Load
# -----------------------------------------------------
print("\nEnter Plate Dimensions (in mm) and Load:")

a = float(input("Length in x-direction, a (mm): "))
b = float(input("Length in y-direction, b (mm): "))
h = float(input("Plate thickness, h (mm): "))

P0 = float(input("Central point load P0 (N): "))

print(f"\nPlate Dimensions: a = {a} mm, b = {b} mm, h = {h} mm")
print(f"Central Load Applied: {P0} N\n")

# ----------------------- CONSTANTS ------------------------
import numpy as np
pi = np.pi
D = E * h**3 / (12.0 * (1.0 - nu**2))     # flexural rigidity (N·mm)

print(f"Calculated Flexural Rigidity: D = {D:.4e} N·mm\n")


# ----------------------- CONSTANTS ------------------------
pi = np.pi
D = E * h**3 / (12.0 * (1.0 - nu**2))     # flexural rigidity (N·mm)


# -------------------------------------------------------------------------------------------------------------
# Build full-grid biharmonic operator A ≈ ∇^4 using L @ L
# Unknowns: ALL nodes (including boundaries)
# -------------------------------------------------------------------------------------------------------------
def build_biharmonic_full(Nx, Ny, a, b):
    """
    Build the ∇^4 operator on an Nx × Ny full grid (including boundaries).
    We first build a full-grid Laplacian L and then set A = L @ L.

    NOTE:
    - Boundary rows of A will later be overwritten to impose the desired BCs.
    """
    hx = a / (Nx - 1)
    hy = b / (Ny - 1)

    # 1D second-derivative matrices with placeholder Dirichlet rows
    Tx = np.zeros((Nx, Nx))
    for i in range(Nx):
        if i == 0 or i == Nx - 1:
            # boundary rows will be overwritten later anyway
            Tx[i, i] = 1.0
        else:
            Tx[i, i] = -2.0
            Tx[i, i - 1] = 1.0
            Tx[i, i + 1] = 1.0
    Tx /= hx**2

    Ty = np.zeros((Ny, Ny))
    for j in range(Ny):
        if j == 0 or j == Ny - 1:
            Ty[j, j] = 1.0
        else:
            Ty[j, j] = -2.0
            Ty[j, j - 1] = 1.0
            Ty[j, j + 1] = 1.0
    Ty /= hy**2

    Ix = np.eye(Nx)
    Iy = np.eye(Ny)

    # 2D Laplacian on full grid
    L = np.kron(Iy, Tx) + np.kron(Ty, Ix)

    # Biharmonic operator
    A = L @ L

    return A, hx, hy


# -------------------------------------------------------------------------------------------------------------
# Apply boundary conditions: user-input (any combination of S/C)
# -------------------------------------------------------------------------------------------------------------
def apply_3S_1C_bc(A, rhs, Nx, Ny, hx, hy, clamped_edge="bottom"):
    """
    Modify matrix A and rhs to impose:
      - simply supported edge: w = 0
      - clamped edge: w = 0 and slope normal = 0 (approx).

    On FIRST CALL it will ask:
      Bottom (y=0):  S or C
      Top    (y=b):  S or C
      Left   (x=0):  S or C
      Right  (x=a):  S or C

    Then it reuses the same BCs for all later calls.

    You can choose ANY combination:
      - 0 clamped (all S)
      - 1 clamped (3S+1C)
      - 2, 3, or 4 clamped edges
    """

    # ---------- Ask once & cache ----------
    if not hasattr(apply_3S_1C_bc, "initialized"):
        print("\nEnter boundary condition for each edge:")
        print("Use 'S' for simply supported (w = 0)")
        print("Use 'C' for clamped (w = 0 and zero normal slope)\n")

        bc_bottom = input("  Bottom edge (y = 0)   [S/C]: ").strip().upper()
        bc_top    = input("  Top edge    (y = b)   [S/C]: ").strip().upper()
        bc_left   = input("  Left edge   (x = 0)   [S/C]: ").strip().upper()
        bc_right  = input("  Right edge  (x = a)   [S/C]: ").strip().upper()

        def norm_bc(bc, default="S", name=""):
            if bc not in ("S", "C"):
                print(f"  Invalid '{bc}' for {name}, using '{default}' instead.")
                return default
            return bc

        bc_bottom = norm_bc(bc_bottom, "S", "bottom")
        bc_top    = norm_bc(bc_top,    "S", "top")
        bc_left   = norm_bc(bc_left,   "S", "left")
        bc_right  = norm_bc(bc_right,  "S", "right")

        # Just info (no error)
        num_clamped = sum(bc == "C" for bc in [bc_bottom, bc_top, bc_left, bc_right])
        print(f"\n[Info] You selected {num_clamped} clamped edge(s) and {4 - num_clamped} simply supported edge(s).\n")

        apply_3S_1C_bc.bc_bottom = bc_bottom
        apply_3S_1C_bc.bc_top    = bc_top
        apply_3S_1C_bc.bc_left   = bc_left
        apply_3S_1C_bc.bc_right  = bc_right
        apply_3S_1C_bc.initialized = True

    # Use cached BCs
    bc_bottom = apply_3S_1C_bc.bc_bottom
    bc_top    = apply_3S_1C_bc.bc_top
    bc_left   = apply_3S_1C_bc.bc_left
    bc_right  = apply_3S_1C_bc.bc_right

    # Helper to convert (i, j) -> 1D index (row-major)
    def idx(i, j):
        return j * Nx + i

    # --- 1) w = 0 on ALL four edges (applies to both S and C) ---
    for i in range(Nx):
        # bottom j=0
        k = idx(i, 0)
        A[k, :] = 0.0
        A[k, k] = 1.0
        rhs[k] = 0.0

        # top j=Ny-1
        k = idx(i, Ny - 1)
        A[k, :] = 0.0
        A[k, k] = 1.0
        rhs[k] = 0.0

    for j in range(Ny):
        # left i=0
        k = idx(0, j)
        A[k, :] = 0.0
        A[k, k] = 1.0
        rhs[k] = 0.0

        # right i=Nx-1
        k = idx(Nx - 1, j)
        A[k, :] = 0.0
        A[k, k] = 1.0
        rhs[k] = 0.0

    # --- 2) Add clamped (zero slope) condition on any edge that is 'C' ---

    # Bottom clamped: y = 0 → enforce slope at j=1
    if bc_bottom == "C":
        for i in range(Nx):
            k1 = idx(i, 1)
            k0 = idx(i, 0)
            A[k1, :] = 0.0
            A[k1, k1] = 1.0 / hy
            A[k1, k0] = -1.0 / hy
            rhs[k1] = 0.0

    # Top clamped: y = b → enforce slope at j=Ny-2
    if bc_top == "C":
        for i in range(Nx):
            k_in = idx(i, Ny - 2)
            k_b = idx(i, Ny - 1)
            A[k_in, :] = 0.0
            A[k_in, k_in] = 1.0 / hy
            A[k_in, k_b] = -1.0 / hy
            rhs[k_in] = 0.0

    # Left clamped: x = 0 → enforce slope at i=1
    if bc_left == "C":
        for j in range(Ny):
            k1 = idx(1, j)
            k0 = idx(0, j)
            A[k1, :] = 0.0
            A[k1, k1] = 1.0 / hx
            A[k1, k0] = -1.0 / hx
            rhs[k1] = 0.0

    # Right clamped: x = a → enforce slope at i=Nx-2
    if bc_right == "C":
        for j in range(Ny):
            k_in = idx(Nx - 2, j)
            k_b = idx(Nx - 1, j)
            A[k_in, :] = 0.0
            A[k_in, k_in] = 1.0 / hx
            A[k_in, k_b] = -1.0 / hx
            rhs[k_in] = 0.0


# -------------------------------------------------------------------------------------------------------------
# Convergence study
# -------------------------------------------------------------------------------------------------------------

# ---- User input for N_list ----
print("\nEnter N values for convergence study should be less than 100(comma separated)")
print("Example: 9, 13, 17, 21, 25")
N_input = input("N values: ")

try:
    N_list = [int(x.strip()) for x in N_input.split(",") if x.strip() != ""]
except Exception:
    raise ValueError("Invalid input! Please enter comma-separated integers only.")

if len(N_list) == 0:
    raise ValueError("No valid N values entered!")

print(f"\nUsing grid sizes: {N_list}\n")

Pxy_vals = []
Wxy_vals = []
Mxx_vals = []
Myy_vals = []
Mxy_vals = []
Qxz_vals = []
Qyz_vals = []

for N in N_list:
    Nx = Ny = N
    A, hx, hy = build_biharmonic_full(Nx, Ny, a, b)

    # RHS: central point load approximated over central cell
    rhs = np.zeros(Nx * Ny)

    ic = Nx // 2
    jc = Ny // 2
    centre_idx = jc * Nx + ic
    rhs[centre_idx] = P0 / (D * hx * hy)

    # Impose BCs (interactive: asks only on first call)
    apply_3S_1C_bc(A, rhs, Nx, Ny, hx, hy)

    # Solve
    w_vec = np.linalg.solve(A, rhs)

    # Reshape to full field
    W = w_vec.reshape((Ny, Nx))

    # Centre values
    W_c = W[jc, ic]

    # Compute derivatives for moments/shears
    d2wdx2 = np.zeros_like(W)
    d2wdy2 = np.zeros_like(W)

    d2wdx2[:, 1:-1] = (W[:, 0:-2] - 2.0 * W[:, 1:-1] + W[:, 2:]) / hx**2
    d2wdy2[1:-1, :] = (W[0:-2, :] - 2.0 * W[1:-1, :] + W[2:, :]) / hy**2

    d2wdxdy = np.zeros_like(W)
    d2wdxdy[1:-1, 1:-1] = (
        W[2:, 2:] - W[2:, 0:-2] - W[0:-2, 2:] + W[0:-2, 0:-2]
    ) / (4.0 * hx * hy)

    Mxx = -D * (d2wdx2 + nu * d2wdy2)
    Myy = -D * (d2wdy2 + nu * d2wdx2)
    Mxy = -D * (1.0 - nu) * d2wdxdy

    dMxxdx = np.zeros_like(W)
    dMxydy = np.zeros_like(W)
    dMyydy = np.zeros_like(W)
    dMxydx = np.zeros_like(W)

    dMxxdx[:, 1:-1] = (Mxx[:, 2:] - Mxx[:, 0:-2]) / (2.0 * hx)
    dMxydy[1:-1, :] = (Mxy[2:, :] - Mxy[0:-2, :]) / (2.0 * hy)
    dMyydy[1:-1, :] = (Myy[2:, :] - Myy[0:-2, :]) / (2.0 * hy)
    dMxydx[:, 1:-1] = (Mxy[:, 2:] - Mxy[:, 0:-2]) / (2.0 * hx)

    Qxz = dMxxdx + dMxydy
    Qyz = dMyydy + dMxydx

    Mxx_c = Mxx[jc, ic]
    Myy_c = Myy[jc, ic]
    Mxy_c = Mxy[jc, ic]
    Qxz_c = Qxz[jc, ic]
    Qyz_c = Qyz[jc, ic]

    Pxy_vals.append(P0)
    Wxy_vals.append(W_c)
    Mxx_vals.append(Mxx_c)
    Myy_vals.append(Myy_c)
    Mxy_vals.append(Mxy_c)
    Qxz_vals.append(Qxz_c)
    Qyz_vals.append(Qyz_c)

# --- Small helpers for printing labels from BCs ---
def bc_full(bc):
    return "Simply Supported (w=0)" if bc == "S" else "Clamped (w=0, slope≈0)"

def bc_short(bc):
    return "SS" if bc == "S" else "C"

bc_bottom = apply_3S_1C_bc.bc_bottom
bc_top    = apply_3S_1C_bc.bc_top
bc_left   = apply_3S_1C_bc.bc_left
bc_right  = apply_3S_1C_bc.bc_right

# --- Output convergence table ---
print("\nFinite Difference Method – Rectangular Plate")
print(f"BCs:")
print(f"  Bottom (y=0): {bc_full(bc_bottom)}")
print(f"  Top    (y=b): {bc_full(bc_top)}")
print(f"  Left   (x=0): {bc_full(bc_left)}")
print(f"  Right  (x=a): {bc_full(bc_right)}")
print(f"\nLoad: central point load P0 = {P0:.2f} N\n")

print("Convergence at plate centre (x=a/2, y=b/2)")
print("N = total grid points in each direction (including boundaries)\n")
for N, P1, W1, M11, M22, M12, Q13, Q23 in zip(
        N_list, Pxy_vals, Wxy_vals, Mxx_vals, Myy_vals, Mxy_vals, Qxz_vals, Qyz_vals):
    print(f"N = {N:3d}    P0 = {P1:10.2f} N  Wxy = {W1:10.4e}  "
          f"Mxx = {M11:10.4e}  Myy = {M22:10.4e}  Mxy = {M12:10.4e}  "
          f"Qxz = {Q13:10.4e}  Qyz = {Q23:10.4e}")

# -------------------------------------------------------------------------------------------------------------
# Full-field plots for finest grid
# -------------------------------------------------------------------------------------------------------------
N_plot = N_list[-1]
Nx = Ny = N_plot
A, hx, hy = build_biharmonic_full(Nx, Ny, a, b)
rhs = np.zeros(Nx * Ny)

ic = Nx // 2
jc = Ny // 2
centre_idx = jc * Nx + ic
rhs[centre_idx] = P0 / (D * hx * hy)

apply_3S_1C_bc(A, rhs, Nx, Ny, hx, hy)

w_vec = np.linalg.solve(A, rhs)
W = w_vec.reshape((Ny, Nx))

# Recompute derivatives, moments, and shears
d2wdx2 = np.zeros_like(W)
d2wdy2 = np.zeros_like(W)

d2wdx2[:, 1:-1] = (W[:, 0:-2] - 2.0 * W[:, 1:-1] + W[:, 2:]) / hx**2
d2wdy2[1:-1, :] = (W[0:-2, :] - 2.0 * W[1:-1, :] + W[2:, :]) / hy**2

d2wdxdy = np.zeros_like(W)
d2wdxdy[1:-1, 1:-1] = (
    W[2:, 2:] - W[2:, 0:-2] - W[0:-2, 2:] + W[0:-2, 0:-2]
) / (4.0 * hx * hy)

Mxx = -D * (d2wdx2 + nu * d2wdy2)
Myy = -D * (d2wdy2 + nu * d2wdx2)
Mxy = -D * (1.0 - nu) * d2wdxdy

dMxxdx = np.zeros_like(W)
dMxydy = np.zeros_like(W)
dMyydy = np.zeros_like(W)
dMxydx = np.zeros_like(W)

dMxxdx[:, 1:-1] = (Mxx[:, 2:] - Mxx[:, 0:-2]) / (2.0 * hx)
dMxydy[1:-1, :] = (Mxy[2:, :] - Mxy[0:-2, :]) / (2.0 * hy)
dMyydy[1:-1, :] = (Myy[2:, :] - Myy[0:-2, :]) / (2.0 * hy)
dMxydx[:, 1:-1] = (Mxy[:, 2:] - Mxy[:, 0:-2]) / (2.0 * hx)

Qxz = dMxxdx + dMxydy
Qyz = dMyydy + dMxydx

# Grid for plotting
x_coords = np.linspace(0, a, Nx)
y_coords = np.linspace(0, b, Ny)
Xg, Yg = np.meshgrid(x_coords, y_coords)

bc_label = (
    f"Bottom {bc_short(bc_bottom)}, "
    f"Top {bc_short(bc_top)}, "
    f"Left {bc_short(bc_left)}, "
    f"Right {bc_short(bc_right)}"
)

# ------------------------------------ Deflection plots ------------------------------------
z_scale = 10.0  # purely for visual exaggeration
Z = W * z_scale

fig = plt.figure(figsize=(12, 5))

# Contour of deflection
ax1 = fig.add_subplot(1, 2, 1)
cp = ax1.contourf(x_coords, y_coords, W, 40, cmap='jet')
fig.colorbar(cp, ax=ax1, label='w(x,y) [mm]')
ax1.set_title(f'Deflection Contour (FDM, central point load)\nBCs: {bc_label}')
ax1.set_xlabel('x (mm)')
ax1.set_ylabel('y (mm)')
ax1.axis('equal')

# 3D wireframe
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_wireframe(Xg, Yg, Z, linewidth=0.5)
ax2.set_title(f'3D Deflection (×{z_scale})\nBCs: {bc_label}')
ax2.set_xlabel('x (mm)')
ax2.set_ylabel('y (mm)')
ax2.set_zlabel('w [mm]')
ax2.set_box_aspect([a, b, np.max(Z) - np.min(Z)])

plt.tight_layout()

# ------------------------------------ Moment plots ------------------------------------
z_scale_m = 1e-6
fields = [Mxx, Myy, Mxy]
titles = ['Mxx', 'Myy', 'Mxy']

fig = plt.figure(figsize=(10, 12))
for k, (F, t) in enumerate(zip(fields, titles)):
    F = np.array(F)
    Zm = F * z_scale_m

    ax1 = fig.add_subplot(3, 2, 2*k + 1)
    c = ax1.contourf(x_coords, y_coords, F, 40, cmap='jet')
    fig.colorbar(c, ax=ax1, label=f'{t} [N·mm/mm]')
    ax1.set_title(f'{t} Contour (FDM, central point load)\nBCs: {bc_label}')
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    ax1.axis('equal')

    ax2 = fig.add_subplot(3, 2, 2*k + 2, projection='3d')
    ax2.plot_wireframe(Xg, Yg, Zm, lw=0.5, rstride=2, cstride=2)
    ax2.set_title(f'{t} 3D (×{z_scale_m})')
    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('y (mm)')
    ax2.set_zlabel(t)
    ax2.set_box_aspect([a, b, 0.25 * max(a, b)])

plt.tight_layout()

# ------------------------------------ Shear plots ------------------------------------
z_scale_q = 1e-6
fields_q = [Qxz, Qyz]
titles_q = ['Qxz', 'Qyz']

fig = plt.figure(figsize=(10, 8))
for k, (F, t) in enumerate(zip(fields_q, titles_q)):
    F = np.array(F)
    Zq = F * z_scale_q

    ax1 = fig.add_subplot(2, 2, 2*k + 1)
    c = ax1.contourf(x_coords, y_coords, F, 40, cmap='jet')
    fig.colorbar(c, ax=ax1, label=f'{t} [N/mm]')
    ax1.set_title(f'{t} Contour (FDM)\nBCs: {bc_label}')
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    ax1.axis('equal')

    ax2 = fig.add_subplot(2, 2, 2*k + 2, projection='3d')
    ax2.plot_wireframe(Xg, Yg, Zq, lw=0.5, rstride=2, cstride=2)
    ax2.set_title(f'{t} 3D (×{z_scale_q})')
    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('y (mm)')
    ax2.set_zlabel(t)
    ax2.set_box_aspect([a, b, 0.25 * max(a, b)])

plt.tight_layout()
plt.show()