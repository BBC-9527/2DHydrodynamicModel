import taichi as ti
import numpy as np
import threading

# Initialize Taichi
# We let Taichi choose the best backend (CUDA, Vulkan, OpenGL, or CPU)reset_sources
INIT_BACKEND = "Unknown"
try:
    ti.init(arch=ti.gpu, default_fp=ti.f64)
    INIT_BACKEND = "GPU (Auto-detected)"
except Exception as e:
    print(f"Warning: GPU not available or failed to initialize ({e}). Falling back to CPU.")
    ti.init(arch=ti.cpu, default_fp=ti.f64)
    INIT_BACKEND = "CPU (Fallback)"

@ti.data_oriented
class SWEModelGPU:
    def get_backend_name(self):
        return INIT_BACKEND

    def __init__(self):
        self.lock = threading.Lock()
        self.rows = 0
        self.cols = 0
        self.dx = 1.0
        self.dy = 1.0
        self.g = 9.81
        
        # Fields (initialized in init)
        self.h = None
        self.z = None
        self.u = None
        self.v = None
        self.n = None
        
        self.bc_type = None
        self.bc_val = None
        
        self.h_new = None
        self.hu_new = None
        self.hv_new = None

    def init(self, elevation, roughness, dx, dy):
        self.rows, self.cols = elevation.shape
        self.dx = dx
        self.dy = dy
        
        # Keep a CPU copy of elevation for easy access (e.g. by Coupler)
        self.elevation_np = elevation.astype(np.float64)
        
        # Allocate fields
        shape = (self.rows, self.cols)
        self.h = ti.field(dtype=ti.f64, shape=shape)
        self.z = ti.field(dtype=ti.f64, shape=shape)
        self.u = ti.field(dtype=ti.f64, shape=shape)
        self.v = ti.field(dtype=ti.f64, shape=shape)
        self.n = ti.field(dtype=ti.f64, shape=shape)
        
        self.bc_type = ti.field(dtype=ti.i32, shape=shape)
        self.bc_val = ti.field(dtype=ti.f64, shape=shape)
        
        # Source Term (m^3/s)
        self.q_src = ti.field(dtype=ti.f64, shape=shape)
        
        # Momentum Source Terms (m^4/s^2)
        self.mx_src = ti.field(dtype=ti.f64, shape=shape)
        self.my_src = ti.field(dtype=ti.f64, shape=shape)
        
        self.h_new = ti.field(dtype=ti.f64, shape=shape)
        self.hu_new = ti.field(dtype=ti.f64, shape=shape)
        self.hv_new = ti.field(dtype=ti.f64, shape=shape)
        
        # Coupling Buffers (Max 65536 nodes)
        self.max_couplers = 65536
        self.cp_r = ti.field(dtype=ti.i32, shape=self.max_couplers)
        self.cp_c = ti.field(dtype=ti.i32, shape=self.max_couplers)
        self.cp_h = ti.field(dtype=ti.f64, shape=self.max_couplers)
        self.cp_z = ti.field(dtype=ti.f64, shape=self.max_couplers)
        self.cp_u = ti.field(dtype=ti.f64, shape=self.max_couplers)
        self.cp_v = ti.field(dtype=ti.f64, shape=self.max_couplers)
        
        self.cp_q = ti.field(dtype=ti.f64, shape=self.max_couplers)
        self.cp_mx = ti.field(dtype=ti.f64, shape=self.max_couplers)
        self.cp_my = ti.field(dtype=ti.f64, shape=self.max_couplers)
        
        # Visualization Velocity Buffers
        self.cp_vis_u = ti.field(dtype=ti.f64, shape=self.max_couplers)
        self.cp_vis_v = ti.field(dtype=ti.f64, shape=self.max_couplers)
        
        # Initialize data
        self.z.from_numpy(elevation.astype(np.float64))
        self.n.from_numpy(roughness.astype(np.float64))
        
        self.h.fill(0.0)
        self.u.fill(0.0)
        self.v.fill(0.0)
        self.bc_type.fill(0)
        self.bc_val.fill(0.0)
        self.q_src.fill(0.0)
        self.mx_src.fill(0.0)
        self.my_src.fill(0.0)
        
        self.h_new.fill(0.0)
        self.hu_new.fill(0.0)
        self.hv_new.fill(0.0)

    @property
    def elevation_flat(self):
        """Returns flattened elevation array (for compatibility with Coupler)"""
        return self.elevation_np.flatten()

    def set_water_surface(self, ws_grid):
        self.set_water_surface_kernel(ws_grid)

    @ti.kernel
    def set_water_surface_kernel(self, ws_grid: ti.types.ndarray()):
        for i, j in self.h:
            h_val = ws_grid[i, j] - self.z[i, j]
            if h_val < 0:
                h_val = 0.0
            self.h[i, j] = h_val
            self.u[i, j] = 0.0
            self.v[i, j] = 0.0

    def set_boundary_conditions(self, bc_type_grid, bc_val_grid):
        self.bc_type.from_numpy(bc_type_grid.astype(np.int32))
        self.bc_val.from_numpy(bc_val_grid.astype(np.float64))

    @ti.kernel
    def reset_sources_kernel(self):
        for i, j in self.q_src:
            self.q_src[i, j] = 0.0
            self.mx_src[i, j] = 0.0
            self.my_src[i, j] = 0.0

    def reset_sources(self):
        self.reset_sources_kernel()

    def set_source(self, r, c, q_val):
        """Set source term for cell (r,c) in m^3/s"""
        # Taichi field access from python (slow for bulk, ok for single updates)
        self.q_src[r, c] = q_val

    def set_source_momentum(self, r, c, mx, my):
        """Set momentum source term for cell (r,c) in m^4/s^2"""
        self.mx_src[r, c] = mx
        self.my_src[r, c] = my

    @ti.kernel
    def gather_coupling_kernel(self, count: ti.i32):
        for i in range(count):
            r = self.cp_r[i]
            c = self.cp_c[i]
            # Fix 1.1: Coupling Index Check
            if r >= 0 and r < self.rows and c >= 0 and c < self.cols:
                self.cp_h[i] = self.h[r, c]
                self.cp_z[i] = self.z[r, c]
                self.cp_u[i] = self.u[r, c]
                self.cp_v[i] = self.v[r, c]
            else:
                self.cp_h[i] = 0.0
                self.cp_z[i] = 0.0
                self.cp_u[i] = 0.0
                self.cp_v[i] = 0.0

    def update_coupling_indices(self, rows, cols):
        """
        Update cached coupling indices on GPU.
        """
        count = len(rows)
        if count > self.max_couplers:
            print(f"[Warning] Coupling node count {count} exceeds buffer {self.max_couplers}. Truncating.")
            count = self.max_couplers
            rows = rows[:count]
            cols = cols[:count]
            
        padded_rows = np.zeros(self.max_couplers, dtype=np.int32)
        padded_cols = np.zeros(self.max_couplers, dtype=np.int32)
        padded_rows[:count] = rows.astype(np.int32)
        padded_cols[:count] = cols.astype(np.int32)
        
        self.cp_r.from_numpy(padded_rows)
        self.cp_c.from_numpy(padded_cols)
        self.cp_count = count
        self.cp_indices_cached = True

    def get_coupling_values(self, rows=None, cols=None, use_cache=False):
        """
        Batch get values for coupling.
        rows, cols: numpy arrays of indices (Required if use_cache=False)
        use_cache: If True, use previously updated indices
        Returns: (h, z, u, v) as numpy arrays
        """
        if use_cache:
            if not hasattr(self, 'cp_indices_cached') or not self.cp_indices_cached:
                 # Fallback if cache not ready
                 if rows is None or cols is None:
                     return np.array([]), np.array([]), np.array([]), np.array([])
                 use_cache = False
            else:
                 count = self.cp_count
        
        if not use_cache:
            if rows is None or cols is None:
                return np.array([]), np.array([]), np.array([]), np.array([])
                
            count = len(rows)
            if count > self.max_couplers:
                print(f"[Warning] Coupling node count {count} exceeds buffer {self.max_couplers}. Truncating.")
                count = self.max_couplers
                rows = rows[:count]
                cols = cols[:count]
                
            # upload indices (Must pad to full size for from_numpy)
            padded_rows = np.zeros(self.max_couplers, dtype=np.int32)
            padded_cols = np.zeros(self.max_couplers, dtype=np.int32)
            padded_rows[:count] = rows.astype(np.int32)
            padded_cols[:count] = cols.astype(np.int32)
            
            self.cp_r.from_numpy(padded_rows)
            self.cp_c.from_numpy(padded_cols)
        
        try:
            with self.lock:
                self.gather_coupling_kernel(count)
                
                # download results (only first 'count' elements)
                h = self.cp_h.to_numpy()[:count]
                z = self.cp_z.to_numpy()[:count]
                u = self.cp_u.to_numpy()[:count]
                v = self.cp_v.to_numpy()[:count]
                
                return h, z, u, v
        except Exception as e:
            print(f"[SWEModelGPU] Error in get_coupling_values: {e}")
            return np.array([]), np.array([]), np.array([]), np.array([])

    @ti.kernel
    def scatter_coupling_kernel(self, count: ti.i32):
        for i in range(count):
            r = self.cp_r[i]
            c = self.cp_c[i]
            # Fix 1.1: Coupling Index Check
            if r >= 0 and r < self.rows and c >= 0 and c < self.cols:
                self.q_src[r, c] += self.cp_q[i]
                self.mx_src[r, c] += self.cp_mx[i]
                self.my_src[r, c] += self.cp_my[i]

    @ti.kernel
    def scatter_coupling_velocities_kernel(self, count: ti.i32):
        for i in range(count):
            r = self.cp_r[i]
            c = self.cp_c[i]
            # Fix 1.1: Coupling Index Check
            if r >= 0 and r < self.rows and c >= 0 and c < self.cols:
                self.u[r, c] = self.cp_vis_u[i]
                self.v[r, c] = self.cp_vis_v[i]

    def update_coupling_sources(self, q, mx, my):
        """
        Update coupling sources using cached indices.
        Only uploads q, mx, my. Assumes update_coupling_indices was called.
        """
        if not hasattr(self, 'cp_indices_cached') or not self.cp_indices_cached:
             print("[Warning] update_coupling_sources called without cached indices. Ignoring.")
             return

        count = self.cp_count
        
        # Ensure input matches cached count
        # (Relaxed check: allow input to be larger, truncate)
        if len(q) != count:
            # If input is smaller, we might have issue. If larger, we truncate.
            pass
            
        padded_q = np.zeros(self.max_couplers, dtype=np.float64)
        padded_mx = np.zeros(self.max_couplers, dtype=np.float64)
        padded_my = np.zeros(self.max_couplers, dtype=np.float64)
        
        limit = min(count, len(q))
        padded_q[:limit] = q[:limit].astype(np.float64)
        padded_mx[:limit] = mx[:limit].astype(np.float64)
        padded_my[:limit] = my[:limit].astype(np.float64)
        
        self.cp_q.from_numpy(padded_q)
        self.cp_mx.from_numpy(padded_mx)
        self.cp_my.from_numpy(padded_my)
        
        self.scatter_coupling_kernel(count)

    def set_coupling_sources(self, rows, cols, q, mx, my):
        """
        Batch set sources for coupling.
        """
        count = len(rows)
        if count > self.max_couplers:
            # Fix 1.2: Warning for truncation
            print(f"[Warning] Coupling source count {count} exceeds buffer {self.max_couplers}. Truncating.")
            count = self.max_couplers
            rows = rows[:count]
            cols = cols[:count]
            q = q[:count]
            mx = mx[:count]
            my = my[:count]
            
        padded_rows = np.zeros(self.max_couplers, dtype=np.int32)
        padded_cols = np.zeros(self.max_couplers, dtype=np.int32)
        padded_q = np.zeros(self.max_couplers, dtype=np.float64)
        padded_mx = np.zeros(self.max_couplers, dtype=np.float64)
        padded_my = np.zeros(self.max_couplers, dtype=np.float64)
        
        padded_rows[:count] = rows.astype(np.int32)
        padded_cols[:count] = cols.astype(np.int32)
        padded_q[:count] = q.astype(np.float64)
        padded_mx[:count] = mx.astype(np.float64)
        padded_my[:count] = my.astype(np.float64)
        
        self.cp_r.from_numpy(padded_rows)
        self.cp_c.from_numpy(padded_cols)
        self.cp_q.from_numpy(padded_q)
        self.cp_mx.from_numpy(padded_mx)
        self.cp_my.from_numpy(padded_my)
        
        self.scatter_coupling_kernel(count)

    def set_coupling_velocities(self, rows, cols, u, v):
        """
        Batch set velocities for visualization.
        """
        count = len(rows)
        if count > self.max_couplers:
            # Fix 1.2: Warning for truncation
            print(f"[Warning] Coupling velocity count {count} exceeds buffer {self.max_couplers}. Truncating.")
            count = self.max_couplers
            rows = rows[:count]
            cols = cols[:count]
            u = u[:count]
            v = v[:count]
            
        padded_rows = np.zeros(self.max_couplers, dtype=np.int32)
        padded_cols = np.zeros(self.max_couplers, dtype=np.int32)
        padded_u = np.zeros(self.max_couplers, dtype=np.float64)
        padded_v = np.zeros(self.max_couplers, dtype=np.float64)
        
        padded_rows[:count] = rows.astype(np.int32)
        padded_cols[:count] = cols.astype(np.int32)
        padded_u[:count] = u.astype(np.float64)
        padded_v[:count] = v.astype(np.float64)
            
        self.cp_r.from_numpy(padded_rows)
        self.cp_c.from_numpy(padded_cols)
        self.cp_vis_u.from_numpy(padded_u)
        self.cp_vis_v.from_numpy(padded_v)
        
        self.scatter_coupling_velocities_kernel(count)

    def get_results(self):
        # Return numpy arrays
        # Use simple try-except to handle potential Taichi sync/access issues
        try:
            with self.lock:
                h_np = self.h.to_numpy()
                u_np = self.u.to_numpy()
                v_np = self.v.to_numpy()
                return (h_np, u_np, v_np)
        except Exception as e:
            print(f"[SWEModelGPU] Error in get_results: {e}")
            import traceback
            traceback.print_exc()
            # Return zeros if failed
            return (np.zeros((self.rows, self.cols)), 
                    np.zeros((self.rows, self.cols)), 
                    np.zeros((self.rows, self.cols)))

    @ti.kernel
    def get_max_vals_kernel(self) -> ti.types.vector(2, ti.f64):
        max_h = 0.0
        max_v = 0.0
        for i, j in self.h:
            ti.atomic_max(max_h, self.h[i, j])
            v_mag = ti.sqrt(self.u[i, j]**2 + self.v[i, j]**2)
            ti.atomic_max(max_v, v_mag)
        return ti.Vector([max_h, max_v])

    def get_stability_info(self):
        vals = self.get_max_vals_kernel()
        return vals[0], vals[1]

    @ti.func
    def compute_flux(self, h_L, u_L, v_L, h_R, u_R, v_R, is_x_dir):
        # Return f_h, f_hu, f_hv
        f_h = 0.0
        f_hu = 0.0
        f_hv = 0.0
        
        if h_L > 1e-6 or h_R > 1e-6:
            c_L = ti.sqrt(self.g * h_L)
            c_R = ti.sqrt(self.g * h_R)
            
            u_n_L = u_L if is_x_dir else v_L
            u_n_R = u_R if is_x_dir else v_R
            
            S_L = min(u_n_L - c_L, u_n_R - c_R)
            S_R = max(u_n_L + c_L, u_n_R + c_R)
            
            if S_L >= 0:
                f_h = h_L * u_n_L
                f_hu = h_L * u_L * u_n_L + (0.5 * self.g * h_L * h_L if is_x_dir else 0.0)
                f_hv = h_L * v_L * u_n_L + (0.0 if is_x_dir else 0.5 * self.g * h_L * h_L)
            elif S_R <= 0:
                f_h = h_R * u_n_R
                f_hu = h_R * u_R * u_n_R + (0.5 * self.g * h_R * h_R if is_x_dir else 0.0)
                f_hv = h_R * v_R * u_n_R + (0.0 if is_x_dir else 0.5 * self.g * h_R * h_R)
            else:
                F_h_L = h_L * u_n_L
                F_hu_L = h_L * u_L * u_n_L + (0.5 * self.g * h_L * h_L if is_x_dir else 0.0)
                F_hv_L = h_L * v_L * u_n_L + (0.0 if is_x_dir else 0.5 * self.g * h_L * h_L)
                
                F_h_R = h_R * u_n_R
                F_hu_R = h_R * u_R * u_n_R + (0.5 * self.g * h_R * h_R if is_x_dir else 0.0)
                F_hv_R = h_R * v_R * u_n_R + (0.0 if is_x_dir else 0.5 * self.g * h_R * h_R)
                
                denom = S_R - S_L
                if denom < 1e-6:
                    denom = 1e-6
                    
                f_h = (S_R * F_h_L - S_L * F_h_R + S_L * S_R * (h_R - h_L)) / denom
                f_hu = (S_R * F_hu_L - S_L * F_hu_R + S_L * S_R * (h_R * u_R - h_L * u_L)) / denom
                f_hv = (S_R * F_hv_L - S_L * F_hv_R + S_L * S_R * (h_R * v_R - h_L * v_L)) / denom
                
        return f_h, f_hu, f_hv

    @ti.kernel
    def run_step_kernel(self, dt: ti.f64):
        # 1. Update State (FVM)
        for i, j in self.h:
            # Skip inactive
            if self.bc_type[i, j] == -1:
                continue
                
            # Current State
            h_c = self.h[i, j]
            
            # Init Flux Sums (Net Flux into cell)
            net_f_h = 0.0
            net_f_hu = 0.0
            net_f_hv = 0.0
            
            # --- Right Face (i, j) <-> (i, j+1) ---
            z_L = self.z[i, j]
            h_L_raw = self.h[i, j]
            u_L = self.u[i, j]
            v_L = self.v[i, j]
            
            z_R = 0.0
            h_R_raw = 0.0
            u_R = 0.0
            v_R = 0.0
            has_neighbor = False
            
            if j < self.cols - 1:
                if self.bc_type[i, j+1] != -1:
                    has_neighbor = True
                    z_R = self.z[i, j+1]
                    h_R_raw = self.h[i, j+1]
                    u_R = self.u[i, j+1]
                    v_R = self.v[i, j+1]
                    
            if not has_neighbor:
                # Wall / Domain Boundary
                # Fix 2.1: Momentum Reflection (Ghost Cell: u_R = -u_L)
                z_R = z_L
                h_R_raw = h_L_raw
                u_R = -u_L
                v_R = v_L # Slip
                
                # Outflow (Type 2) -> Transmissive
                if self.bc_type[i, j] == 2:
                    u_R = u_L
                    v_R = v_L
                
            # Hydrostatic Reconstruction
            z_face = max(z_L, z_R)
            h_L = max(0.0, z_L + h_L_raw - z_face)
            h_R = max(0.0, z_R + h_R_raw - z_face)
            
            # Flux
            fx_h, fx_hu, fx_hv = self.compute_flux(
                h_L, u_L, v_L, h_R, u_R, v_R, 1
            )
            net_f_h -= fx_h / self.dx
            net_f_hu -= fx_hu / self.dx
            net_f_hv -= fx_hv / self.dx
            
            # Fix 2.2: Correct Bed Slope Source Term (User suggestion: -g * h * dz/dx)
            # Using centered difference for h and z at face
            slope_src = -self.g * 0.5 * (h_L + h_R) * (z_R - z_L)
            net_f_hu += slope_src / self.dx

            # --- Left Face (i, j-1) <-> (i, j) ---
            z_R = self.z[i, j]
            h_R_raw = self.h[i, j]
            u_R = self.u[i, j]
            v_R = self.v[i, j]
            
            z_L = 0.0
            h_L_raw = 0.0
            u_L = 0.0
            v_L = 0.0
            has_neighbor = False
            
            if j > 0:
                if self.bc_type[i, j-1] != -1:
                    has_neighbor = True
                    z_L = self.z[i, j-1]
                    h_L_raw = self.h[i, j-1]
                    u_L = self.u[i, j-1]
                    v_L = self.v[i, j-1]
                    
            if not has_neighbor:
                # Wall
                z_L = z_R
                h_L_raw = h_R_raw
                u_L = -u_R
                v_L = v_R
                
                # Outflow (Type 2) -> Transmissive
                if self.bc_type[i, j] == 2:
                    u_L = u_R
                    v_L = v_R
                
            z_face = max(z_L, z_R)
            h_L = max(0.0, z_L + h_L_raw - z_face)
            h_R = max(0.0, z_R + h_R_raw - z_face)
            
            fx_h, fx_hu, fx_hv = self.compute_flux(
                h_L, u_L, v_L, h_R, u_R, v_R, 1
            )
            net_f_h += fx_h / self.dx
            net_f_hu += fx_hu / self.dx
            net_f_hv += fx_hv / self.dx
            
            slope_src = -self.g * 0.5 * (h_L + h_R) * (z_R - z_L)
            net_f_hu += slope_src / self.dx

            # --- Bottom Face (i+1, j) <-> (i, j) ---
            # Flux Down (Out of cell i,j to i+1,j)
            z_T = self.z[i, j]
            h_T_raw = self.h[i, j]
            u_T = self.u[i, j]
            v_T = self.v[i, j]
            
            z_B = 0.0
            h_B_raw = 0.0
            u_B = 0.0
            v_B = 0.0
            has_neighbor = False
            
            if i < self.rows - 1:
                if self.bc_type[i+1, j] != -1:
                    has_neighbor = True
                    z_B = self.z[i+1, j]
                    h_B_raw = self.h[i+1, j]
                    u_B = self.u[i+1, j]
                    v_B = self.v[i+1, j]
                    
            if not has_neighbor:
                # Wall
                z_B = z_T
                h_B_raw = h_T_raw
                u_B = u_T # Slip u
                v_B = -v_T # Reflect v
                
                # Outflow (Type 2) -> Transmissive
                if self.bc_type[i, j] == 2:
                    u_B = u_T
                    v_B = v_T
                
            z_face = max(z_T, z_B)
            h_T = max(0.0, z_T + h_T_raw - z_face)
            h_B = max(0.0, z_B + h_B_raw - z_face)
            
            fy_h, fy_hu, fy_hv = self.compute_flux(
                h_T, u_T, v_T, h_B, u_B, v_B, 0
            )
            net_f_h -= fy_h / self.dy
            net_f_hu -= fy_hu / self.dy
            net_f_hv -= fy_hv / self.dy
            
            slope_src = -self.g * 0.5 * (h_T + h_B) * (z_B - z_T)
            net_f_hv += slope_src / self.dy

            # --- Top Face (i-1, j) <-> (i, j) ---
            # Flux Up (Into cell i,j from i-1,j)
            z_B = self.z[i, j]
            h_B_raw = self.h[i, j]
            u_B = self.u[i, j]
            v_B = self.v[i, j]
            
            z_T = 0.0
            h_T_raw = 0.0
            u_T = 0.0
            v_T = 0.0
            has_neighbor = False
            
            if i > 0:
                if self.bc_type[i-1, j] != -1:
                    has_neighbor = True
                    z_T = self.z[i-1, j]
                    h_T_raw = self.h[i-1, j]
                    u_T = self.u[i-1, j]
                    v_T = self.v[i-1, j]
            
            if not has_neighbor:
                # Wall
                z_T = z_B
                h_T_raw = h_B_raw
                u_T = u_B
                v_T = -v_B
                
                # Outflow (Type 2) -> Transmissive
                if self.bc_type[i, j] == 2:
                    u_T = u_B
                    v_T = v_B
                
            z_face = max(z_T, z_B)
            h_T = max(0.0, z_T + h_T_raw - z_face)
            h_B = max(0.0, z_B + h_B_raw - z_face)
            
            fy_h, fy_hu, fy_hv = self.compute_flux(
                h_T, u_T, v_T, h_B, u_B, v_B, 0
            )
            net_f_h += fy_h / self.dy
            net_f_hu += fy_hu / self.dy
            net_f_hv += fy_hv / self.dy
            
            slope_src = -self.g * 0.5 * (h_T + h_B) * (z_B - z_T)
            net_f_hv += slope_src / self.dy
            
            # FIX 3.2: Duplicate Boundary Pressure block removed

            # Update conservative variables
            self.h_new[i, j] = self.h[i, j] + dt * net_f_h
            self.hu_new[i, j] = self.h[i, j] * self.u[i, j] + dt * net_f_hu
            self.hv_new[i, j] = self.h[i, j] * self.v[i, j] + dt * net_f_hv

        # 2. Apply Friction and Update State
        for i, j in self.h:
            if self.bc_type[i, j] == -1:
                continue
                
            # FIX 3.1: Boundary Handling - Do not force overwrite for Type 2
            # For Type 1 (Inflow), if we want to prescribe level, we might still need to force it
            # OR we rely on the ghost state logic if we implement it fully for BCs.
            # Currently I only implemented Wall Ghost State.
            # Ideally, Inflow should also be handled by Ghost State.
            # Given the complexity, I will keep the overwrite for Inflow (Type 1) for now as it's standard Dirichlet.
            # But for Outflow (Type 2), user asked to "not force water level".
            
            if self.bc_type[i, j] == 1:
                target_h = self.bc_val[i, j] - self.z[i, j]
                if target_h < 0: target_h = 0.0
                self.h[i, j] = target_h
                # Allow velocity to evolve? User complaint was "destroy conservation".
                # If I don't set u=0, it's better.
                # self.u[i, j] = 0.0
                # self.v[i, j] = 0.0
                continue

            if self.bc_type[i, j] == 2:
                # Outflow: Transmissive / Zero Gradient
                # Do nothing here. The flux calculation naturally allows flow out
                # because the neighbor (if missing) is treated as Wall currently in my code above?
                # Wait, if neighbor is missing (edge), I treated it as Wall.
                # That prevents outflow!
                # I need to handle Type 2 in the flux loop if I want it to work as Outflow.
                # BUT, checking bc_type inside flux loop is expensive/messy?
                # No, I should do it.
                # Since I didn't add it in the flux loop above, I should rely on the update loop modification?
                # No, if flux is Wall, no flow out. Update loop can't fix that.
                # I must fix the Flux Loop for Type 2 Outflow.
                pass
            
            # Normal or Source
            h_new = self.h_new[i, j]

            # Apply Type 3 (Inflow Rate m/s)
            if self.bc_type[i, j] == 3:
                # Apply Type 3 (Flux Q m/s)
                # If Q is positive, it's inflow (add water).
                # If Q is negative, it's outflow (remove water).
                # bc_val[i, j] is already scaled to m/s (Flux) by server if needed,
                # BUT wait, server sends Q (m3/s) or velocity?
                # Server sends: init_val = feat.value / area (m/s).
                # So bc_val is vertical velocity (m/s).
                # dH = v * dt.
                h_new += self.bc_val[i, j] * dt
                
                # If we are removing water (Outflow), we should ensure H doesn't go negative
                if h_new < 0: h_new = 0.0

            # Apply Source Term (m^3/s)
            if self.q_src[i, j] != 0:
                dh_src = (self.q_src[i, j] * dt) / (self.dx * self.dy)
                h_new += dh_src
            
            # Apply Momentum Source (Input is Acceleration m/s^2)
            # FIX 5.2: Momentum Source Physics
            # dhu = acc * h * dt
            if self.mx_src[i, j] != 0 or self.my_src[i, j] != 0:
                h_val = max(h_new, 1e-6) # Use new depth or old? usually old or new.
                dhu_src = self.mx_src[i, j] * h_val * dt
                dhv_src = self.my_src[i, j] * h_val * dt
                self.hu_new[i, j] += dhu_src
                self.hv_new[i, j] += dhv_src

            # FIX 6.1: NaN Check Extended
            if h_new != h_new or self.hu_new[i, j] != self.hu_new[i, j] or self.hv_new[i, j] != self.hv_new[i, j]:
                 self.h[i, j] = 0.0
                 self.u[i, j] = 0.0
                 self.v[i, j] = 0.0
                 continue

            if h_new < 1e-6:
                self.h[i, j] = 0.0
                self.u[i, j] = 0.0
                self.v[i, j] = 0.0
            else:
                hu = self.hu_new[i, j]
                hv = self.hv_new[i, j]
                u = hu / h_new
                v = hv / h_new
                
                # Friction (Semi-implicit)
                # FIX 5.1: Manning Gravity
                # FIX 4.2: Friction Numerical Stability (min depth)
                V_mag = ti.sqrt(u*u + v*v)
                n_val = self.n[i, j]
                # cf = n^2 * g * |V| / h^(4/3)
                cf = (n_val * n_val * self.g * V_mag) / ti.pow(max(h_new, 1e-4), 4.0/3.0)
                factor = 1.0 / (1.0 + dt * cf) # g is already in cf
                
                self.h[i, j] = h_new
                self.u[i, j] = u * factor
                self.v[i, j] = v * factor

    def step(self, dt, check_stability=True):
        with self.lock:
            # FIX 4.1: CFL Check & Sub-stepping
            remaining_dt = dt
            
            # Add a safety limit to prevent infinite loop
            max_substeps = 100
            substeps_count = 0
            
            while remaining_dt > 1e-6:
                current_step = remaining_dt
                
                if check_stability:
                    max_h, max_v = self.get_stability_info()
                    c = np.sqrt(self.g * max_h) + max_v
                    if c > 1e-6:
                        max_dt = min(self.dx, self.dy) / c * 0.5 # CFL=0.5
                        if current_step > max_dt:
                            current_step = max_dt
                
                self.run_step_kernel(current_step)
                remaining_dt -= current_step
                
                substeps_count += 1
                if substeps_count >= max_substeps:
                     print(f"[Warning] Max substeps ({max_substeps}) reached in GPU step. Remaining dt: {remaining_dt}")
                     break
