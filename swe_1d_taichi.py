import taichi as ti
import numpy as np

@ti.data_oriented
class SWEModel1D:
    def __init__(self, num_cells, dx):
        self.num_cells = num_cells
        self.dx = dx
        self.g = 9.81
        
        # Physical Fields
        self.h = ti.field(dtype=ti.f64, shape=num_cells)
        self.u = ti.field(dtype=ti.f64, shape=num_cells)
        self.z = ti.field(dtype=ti.f64, shape=num_cells) # Bed Elevation
        self.n = ti.field(dtype=ti.f64, shape=num_cells) # Manning's n
        self.width = ti.field(dtype=ti.f64, shape=num_cells) # Channel Width
        
        # Intermediate Fields
        self.h_new = ti.field(dtype=ti.f64, shape=num_cells)
        self.hu_new = ti.field(dtype=ti.f64, shape=num_cells)
        
        # Boundary Conditions (Scalar fields)
        # Types: 0=Wall, 1=Fixed Depth (H), 2=Fixed Discharge (Q), 3=Transmissive (Open)
        self.bc_left_type = ti.field(dtype=ti.i32, shape=())
        self.bc_left_val = ti.field(dtype=ti.f64, shape=())
        self.bc_right_type = ti.field(dtype=ti.i32, shape=())
        self.bc_right_val = ti.field(dtype=ti.f64, shape=())

        # Lateral Inflow Source (for Coupling) - m^3/s
        self.q_lat = ti.field(dtype=ti.f64, shape=num_cells)

    def init(self, z_arr, n_arr, w_arr, h_arr):
        """Initialize the 1D model from numpy arrays"""
        if len(z_arr) != self.num_cells:
            raise ValueError(f"Array length mismatch. Expected {self.num_cells}, got {len(z_arr)}")
            
        self.z.from_numpy(z_arr.astype(np.float64))
        self.n.from_numpy(n_arr.astype(np.float64))
        self.width.from_numpy(w_arr.astype(np.float64))
        self.h.from_numpy(h_arr.astype(np.float64))
        self.u.fill(0.0)
        self.q_lat.fill(0.0)
        
        # Default Boundaries: Closed Left, Open Right
        self.bc_left_type[None] = 0
        self.bc_left_val[None] = 0.0
        self.bc_right_type[None] = 3
        self.bc_right_val[None] = 0.0

    def set_source(self, index, Q_val):
        """Set lateral inflow source at cell index (m^3/s)"""
        self.q_lat[index] = Q_val


    def set_boundary(self, side, b_type, val=0.0):
        """
        Set boundary condition.
        side: 'left' or 'right'
        b_type: 0=Wall, 1=H, 2=Q, 3=Open
        val: Value for H or Q
        """
        if side == 'left':
            self.bc_left_type[None] = int(b_type)
            self.bc_left_val[None] = float(val)
        elif side == 'right':
            self.bc_right_type[None] = int(b_type)
            self.bc_right_val[None] = float(val)

    def get_results(self):
        """Return (h, u) as numpy arrays"""
        return self.h.to_numpy(), self.u.to_numpy()

    @ti.func
    def compute_flux(self, h_L, u_L, h_R, u_R):
        """Compute HLL Flux at an interface"""
        f_h = 0.0
        f_hu = 0.0
        
        # Wet/Dry check
        if h_L > 1e-6 or h_R > 1e-6:
            c_L = ti.sqrt(self.g * h_L)
            c_R = ti.sqrt(self.g * h_R)
            
            # Wave speeds
            S_L = min(u_L - c_L, u_R - c_R)
            S_R = max(u_L + c_L, u_R + c_R)
            
            # Fluxes
            if S_L >= 0:
                f_h = h_L * u_L
                f_hu = h_L * u_L * u_L + 0.5 * self.g * h_L * h_L
            elif S_R <= 0:
                f_h = h_R * u_R
                f_hu = h_R * u_R * u_R + 0.5 * self.g * h_R * h_R
            else:
                # HLL Intermediate
                F_h_L = h_L * u_L
                F_hu_L = h_L * u_L * u_L + 0.5 * self.g * h_L * h_L
                
                F_h_R = h_R * u_R
                F_hu_R = h_R * u_R * u_R + 0.5 * self.g * h_R * h_R
                
                denom = S_R - S_L
                if denom < 1e-6: denom = 1e-6
                
                f_h = (S_R * F_h_L - S_L * F_h_R + S_L * S_R * (h_R - h_L)) / denom
                f_hu = (S_R * F_hu_L - S_L * F_hu_R + S_L * S_R * (h_R * u_R - h_L * u_L)) / denom
                
        return f_h, f_hu

    @ti.kernel
    def step_kernel(self, dt: ti.f64):
        # Loop over cells to compute net flux
        for i in range(self.num_cells):
            h_c = self.h[i]
            
            # --- Left Interface (i-1, i) ---
            flux_in_h = 0.0
            flux_in_hu = 0.0
            
            if i == 0:
                # Left Boundary
                b_type = self.bc_left_type[None]
                h_L = h_c
                u_L = -self.u[i] # Default Wall
                
                if b_type == 1: # Fixed H
                    h_L = self.bc_left_val[None] - self.z[i]
                    if h_L < 0: h_L = 0.0
                    u_L = self.u[i] # Assume smooth velocity? Or 0?
                elif b_type == 2: # Fixed Q
                    # Q = h * u * width
                    Q_val = self.bc_left_val[None]
                    w = self.width[i]
                    if h_c > 1e-3:
                        u_L = Q_val / (h_c * w)
                        h_L = h_c # Assume depth matches?
                    else:
                        u_L = 0.0
                        h_L = h_c
                elif b_type == 3: # Open
                    h_L = h_c
                    u_L = self.u[i]
                    
                # Compute Flux with Ghost State
                f_h, f_hu = self.compute_flux(h_L, u_L, h_c, self.u[i])
                flux_in_h += f_h
                flux_in_hu += f_hu
                
            else:
                # Internal Interface
                # Reconstruct (Hydrostatic)
                z_L = self.z[i-1]
                z_R = self.z[i]
                z_face = max(z_L, z_R)
                
                h_L = max(0.0, z_L + self.h[i-1] - z_face)
                h_R = max(0.0, z_R + self.h[i] - z_face)
                
                f_h, f_hu = self.compute_flux(h_L, self.u[i-1], h_R, self.u[i])
                flux_in_h += f_h
                flux_in_hu += f_hu

            # --- Right Interface (i, i+1) ---
            flux_out_h = 0.0
            flux_out_hu = 0.0
            
            if i == self.num_cells - 1:
                # Right Boundary
                b_type = self.bc_right_type[None]
                h_R = h_c
                u_R = -self.u[i] # Wall
                
                if b_type == 1: # Fixed H
                    h_R = self.bc_right_val[None] - self.z[i]
                    if h_R < 0: h_R = 0.0
                    u_R = self.u[i]
                elif b_type == 2: # Fixed Q
                    Q_val = self.bc_right_val[None]
                    w = self.width[i]
                    if h_c > 1e-3:
                        u_R = Q_val / (h_c * w)
                        h_R = h_c
                    else:
                        u_R = 0.0
                        h_R = h_c
                elif b_type == 3: # Open
                    h_R = h_c
                    u_R = self.u[i]
                
                f_h, f_hu = self.compute_flux(h_c, self.u[i], h_R, u_R)
                flux_out_h += f_h
                flux_out_hu += f_hu
                
            else:
                # Internal Interface
                z_L = self.z[i]
                z_R = self.z[i+1]
                z_face = max(z_L, z_R)
                
                h_L = max(0.0, z_L + self.h[i] - z_face)
                h_R = max(0.0, z_R + self.h[i+1] - z_face)
                
                f_h, f_hu = self.compute_flux(h_L, self.u[i], h_R, self.u[i+1])
                flux_out_h += f_h
                flux_out_hu += f_hu
                
            # Update Conservative Variables
            # d(h)/dt + d(hu)/dx = 0
            # h_new = h - dt/dx * (flux_out - flux_in)
            
            self.h_new[i] = self.h[i] - (dt / self.dx) * (flux_out_h - flux_in_h)
            self.hu_new[i] = self.h[i] * self.u[i] - (dt / self.dx) * (flux_out_hu - flux_in_hu)
            
            # Apply Lateral Source
            # q_lat is in m^3/s
            # dVol = q_lat * dt
            # dh = dVol / Area = (q_lat * dt) / (dx * width)
            if self.q_lat[i] != 0:
                 dh_src = (self.q_lat[i] * dt) / (self.dx * self.width[i])
                 self.h_new[i] += dh_src
                 # Momentum? If lateral flow comes with 0 velocity (perpendicular),
                 # it adds mass but no momentum, effectively slowing down the flow (dilution of momentum).
                 # So we don't add to hu_new, just h_new. 
                 # When we divide by h_new later, u will decrease. Correct.

        # 2. Update with Friction and Dry Bed Check
        for i in range(self.num_cells):
            h_val = self.h_new[i]
            
            if h_val < 1e-6:
                self.h[i] = 0.0
                self.u[i] = 0.0
            else:
                hu = self.hu_new[i]
                u = hu / h_val
                
                # Friction: Sf = n^2 * u * |u| / h^(4/3)
                # Semi-implicit
                n_val = self.n[i]
                cf = (n_val * n_val * abs(u)) / ti.pow(h_val, 4.0/3.0)
                denom = 1.0 + dt * self.g * cf
                
                self.h[i] = h_val
                self.u[i] = u / denom

    def step(self, dt):
        self.step_kernel(dt)
