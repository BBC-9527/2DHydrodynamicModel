import ctypes
import numpy as np
import os
import sys

# Try to import global constants, otherwise default
try:
    from swe_1d import DRY_THRESHOLD, WET_THRESHOLD
except ImportError:
    DRY_THRESHOLD = 0.001
    WET_THRESHOLD = 0.01

# Advanced Tuning Constants
NOISE_FILTER_THRESHOLD_1D = 0.1   # Depth below which 1D is considered "dry/sensitive" for noise
NOISE_FILTER_THRESHOLD_2D = 0.5   # 2D Depth required to breach a dry 1D channel (anti-noise)
SHALLOW_WAVE_THRESHOLD = 0.3      # Depth below which 1D outflow is restricted to preserve wave front
INFLOW_DAMPING_RISE = 0.05        # Max rise (m) per step allowed for dry channel inflow

# C-Structs for Numerical Flux Coupling
class InterfaceState(ctypes.Structure):
    _fields_ = [
        ("h1", ctypes.c_double), ("z1", ctypes.c_double), ("u1", ctypes.c_double),
        ("h2", ctypes.c_double), ("z2", ctypes.c_double), ("u2", ctypes.c_double), ("v2", ctypes.c_double),
        ("z_bank", ctypes.c_double), ("nx", ctypes.c_double), ("ny", ctypes.c_double), ("length", ctypes.c_double),
        ("discharge_coeff", ctypes.c_double)
    ]

class CouplingFlux(ctypes.Structure):
    _fields_ = [
        ("mass_flux", ctypes.c_double),
        ("mom_flux_x", ctypes.c_double),
        ("mom_flux_y", ctypes.c_double)
    ]

class CouplingNode:
    def __init__(self, cell_1d_idx, cell_2d_coords, type='weir', model_1d=None, is_outlet=False, **params):
        """
        cell_1d_idx: Index of the 1D cell
        cell_2d_coords: (row, col) of the 2D cell
        type: 'weir' or 'orifice'
        model_1d: Optional reference to specific 1D model instance (for Networks)
        is_outlet: If True, this node is a global outlet of the 1D network (restricts inflow).
        params: 
            For weir: width, crest_level, coeff (Cd)
            For orifice: area, coeff (Cd), invert_level
        """
        self.idx_1d = cell_1d_idx
        self.r_2d = cell_2d_coords[0]
        self.c_2d = cell_2d_coords[1]
        self.type = type
        self.model_1d = model_1d
        self.is_outlet = is_outlet
        self.params = params
        self.current_q = 0.0
        
        # Precompute geometry for numerical flux (if applicable)
        self.nx = params.get('nx', 0.0) 
        self.ny = params.get('ny', 1.0) # Default pointing North?
        self.length = params.get('width', 10.0) # Interface length

class Coupler:
    def __init__(self, model_1d, model_2d, bc_type_grid=None):
        self.model_1d = model_1d
        self.model_2d = model_2d
        self.bc_type_grid = bc_type_grid
        self.nodes = []
        self.g = 9.81
        self.method = 'empirical' # 'empirical' or 'numerical_flux'
        
        # Load C++ DLL if available
        self.lib = None
        dll_path = os.path.join(os.path.dirname(__file__), "swe_coupling.dll")
        if os.path.exists(dll_path):
            try:
                self.lib = ctypes.CDLL(dll_path)
                self.lib.compute_coupling_flux.argtypes = [InterfaceState, ctypes.POINTER(CouplingFlux)]
                self.lib.compute_coupling_flux.restype = None
                print("[Coupler] Loaded swe_coupling.dll for numerical flux method.")
            except Exception as e:
                print(f"[Coupler] Failed to load swe_coupling.dll: {e}")

    def add_node(self, idx_1d, r_2d, c_2d, type='weir', model_1d=None, lateral=False, is_outlet=False, **params):
        """
        Add a coupling connection.
        """
        node = CouplingNode(idx_1d, (r_2d, c_2d), type, model_1d=model_1d, lateral=lateral, is_outlet=is_outlet, **params)
        self.nodes.append(node)
        self.cache_valid = False

    def set_global_cd(self, cd):
        """
        Update discharge coefficient for all coupling nodes.
        """
        for node in self.nodes:
            node.params['coeff'] = cd

    def set_method(self, method):
        if method not in ['empirical', 'numerical_flux']:
            print(f"[Coupler] Unknown method '{method}', keeping current.")
            return
        if method == 'numerical_flux' and self.lib is None:
            print("[Coupler] Numerical flux DLL not loaded, falling back to empirical.")
            self.method = 'empirical'
            return
        self.method = method
        print(f"[Coupler] Method set to: {self.method}")

    def _build_cache(self):
        """
        Build static cache for optimized coupling.
        """
        self.active_nodes = []
        self.active_rows = []
        self.active_cols = []
        self.active_idx1 = []
        
        # Mapping for output aggregation (Duplicate Handling)
        self.unique_map = [] # active_node_index -> unique_output_index
        unique_coords_map = {} # (r, c) -> unique_output_index
        self.unique_rows = []
        self.unique_cols = []
        
        # Group by 1D model for batch fetching
        self.model_1d_groups = {} 
        
        skipped_bounds = 0
        skipped_inactive = 0
        
        for i, node in enumerate(self.nodes):
            r2, c2 = int(node.r_2d), int(node.c_2d)
            if not (0 <= r2 < self.model_2d.rows and 0 <= c2 < self.model_2d.cols):
                skipped_bounds += 1
                continue
            
            if self.bc_type_grid is not None and self.bc_type_grid[r2, c2] == -1:
                skipped_inactive += 1
                continue
                
            self.active_nodes.append(node)
            self.active_rows.append(r2)
            self.active_cols.append(c2)
            self.active_idx1.append(int(node.idx_1d))
            
            # Handle duplicates
            if (r2, c2) not in unique_coords_map:
                unique_coords_map[(r2, c2)] = len(self.unique_rows)
                self.unique_rows.append(r2)
                self.unique_cols.append(c2)
            
            self.unique_map.append(unique_coords_map[(r2, c2)])
            
            model_1d = node.model_1d if node.model_1d is not None else self.model_1d
            if model_1d not in self.model_1d_groups:
                self.model_1d_groups[model_1d] = []
            self.model_1d_groups[model_1d].append(len(self.active_nodes) - 1)
            
        print(f"[Coupler] Cache built. Total Nodes: {len(self.nodes)} | Active: {len(self.active_nodes)} | Unique Cells: {len(self.unique_rows)}")

        self.active_rows = np.array(self.active_rows, dtype=np.int32)
        self.active_cols = np.array(self.active_cols, dtype=np.int32)
        self.active_idx1 = np.array(self.active_idx1, dtype=np.int32)
        
        self.unique_rows = np.array(self.unique_rows, dtype=np.int32)
        self.unique_cols = np.array(self.unique_cols, dtype=np.int32)
        self.unique_map = np.array(self.unique_map, dtype=np.int32)
        
        # Optimize: Cache indices on GPU
        # We cache the UNIQUE indices because that's what we update
        if hasattr(self.model_2d, 'update_coupling_indices'):
            self.model_2d.update_coupling_indices(self.unique_rows, self.unique_cols)
        
        self.cache_valid = True

    def compute_exchange_optimized(self, dt):
        """
        Optimized compute_exchange using batch GPU transfer and cached mappings.
        With enhanced stability: NaN checks, Flux Damping, and Threshold Unification.
        """
        try:
            if not hasattr(self, 'cache_valid') or not self.cache_valid:
                self._build_cache()
                
            if not self.active_nodes:
                if not hasattr(self, '_last_empty_warn'):
                    print("[Coupler] Warning: No active coupling nodes found in cache. Coupling skipped.")
                    self._last_empty_warn = True
                return
            else:
                self._last_empty_warn = False

            # 2. Batch Fetch 2D Values (Using cached rows/cols)
            if hasattr(self.model_2d, 'update_coupling_indices'):
                 # Returns values for UNIQUE cells
                 h_vals_unique, z_vals_unique, u_vals_unique, v_vals_unique = self.model_2d.get_coupling_values(use_cache=True)
                 
                 if len(h_vals_unique) > 0:
                     max_idx = len(h_vals_unique)
                     # [FIX] Safety clip for indices
                     valid_map = np.clip(self.unique_map, 0, max_idx - 1)
                     
                     h_vals = h_vals_unique[valid_map]
                     # [FIX] Clamp small 2D depths to avoid noise-driven flux
                     h_vals[h_vals < DRY_THRESHOLD] = 0.0
                     
                     z_vals = z_vals_unique[valid_map]
                     u_vals = u_vals_unique[valid_map]
                     v_vals = v_vals_unique[valid_map]
                 else:
                     h_vals = np.zeros(len(self.active_nodes))
                     z_vals = np.zeros(len(self.active_nodes))
                     u_vals = np.zeros(len(self.active_nodes))
                     v_vals = np.zeros(len(self.active_nodes))
            else:
                 h_vals, z_vals, u_vals, v_vals = self.model_2d.get_coupling_values(self.active_rows, self.active_cols)
            
            # [FIX] Global NaN Sanitization for 2D inputs
            h_vals = np.nan_to_num(h_vals, nan=0.0)
            z_vals = np.nan_to_num(z_vals, nan=0.0)
            u_vals = np.nan_to_num(u_vals, nan=0.0)
            v_vals = np.nan_to_num(v_vals, nan=0.0)

            # 3. Compute Flux
            model_cache = {}
            src_1d = {}
            
            # Prepare Unique Output Arrays
            num_unique = len(self.unique_rows)
            unique_q = np.zeros(num_unique, dtype=np.float64)
            unique_mx = np.zeros(num_unique, dtype=np.float64)
            unique_my = np.zeros(num_unique, dtype=np.float64)
            
            # Pre-fetch all 1D results
            for model_1d in self.model_1d_groups:
                try:
                    h_1d_arr, u_1d_arr = model_1d.get_results()
                    z_1d = model_1d.z
                except Exception as e:
                    print(f"[Coupler] Error getting results from 1D model: {e}")
                    h_1d_arr, u_1d_arr, z_1d = None, None, None
                model_cache[model_1d] = (h_1d_arr, u_1d_arr, z_1d)

            for i, node in enumerate(self.active_nodes):
                if i >= len(h_vals): break
                
                h2 = float(h_vals[i])
                z2 = float(z_vals[i])
                u2 = float(u_vals[i])
                v2 = float(v_vals[i])
                
                model_1d = node.model_1d if node.model_1d is not None else self.model_1d
                if model_1d is None: continue
                
                h_1d_arr, u_1d_arr, z_1d = model_cache.get(model_1d, (None, None, None))
                idx1 = int(node.idx_1d)
                
                if h_1d_arr is None:
                    continue
                    
                if not (0 <= idx1 < len(h_1d_arr)):
                    continue

                try:
                    h1 = float(h_1d_arr[idx1])
                    # [FIX] NaN safety for 1D
                    if np.isnan(h1): h1 = 0.0 
                    if h1 < DRY_THRESHOLD: h1 = 0.0 # Clamp small 1D depth
                    
                    u1 = float(u_1d_arr[idx1])
                    if np.isnan(u1): u1 = 0.0 
                    
                    if np.ndim(z_1d) == 0: z1 = float(z_1d)
                    else: z1 = float(z_1d[idx1])
                    
                    # --- Flux Calculation ---
                    Q = 0.0
                    mx = 0.0
                    my = 0.0
                    
                    # Compute WSE and Crest
                    wse_1d = z1 + h1
                    wse_2d = z2 + h2
                    
                    # [FIX] Prioritize crest_level if provided (e.g. from bank_height)
                    crest = float(node.params.get('crest_level', -9999.0))
                    if crest < -9000.0:
                        crest = max(z1, z2)
                    
                    # [FIX] Active Status Check: If 1D node is dry and below crest, check if it should be active
                    # This prevents dry "puddles" from coupling unless significant water is present
                    is_dry_1d = h1 < DRY_THRESHOLD
                    is_dry_2d = h2 < DRY_THRESHOLD
                    
                    if is_dry_1d and is_dry_2d:
                        continue

                    # [FIX] Prevent unphysical flow from dry cells (Ghost Water)
                    # If 2D is dry, it cannot provide water to 1D, even if Z2 > Z1
                    if is_dry_2d and wse_2d > wse_1d:
                        continue
                    # If 1D is dry, it cannot provide water to 2D
                    if is_dry_1d and wse_1d > wse_2d:
                        continue
                        
                    # [FIX] Strict Bank Blocking
                    if wse_1d < (crest - 1e-3) and wse_2d < (crest - 1e-3):
                        continue

                    # [FIX] Outlet Logic: Prevent backflow into dry outlet
                    # If this is the downstream end, and it's dry, don't let 2D water enter.
                    if getattr(node, 'is_outlet', False) and is_dry_1d:
                        if wse_2d > wse_1d:
                             continue

                    # [FIX] General Dry Channel Protection (Noise Filter)
                    # If 1D is dry, require significant 2D depth to initiate inflow
                    if is_dry_1d and (wse_2d > wse_1d):
                        if h2 < NOISE_FILTER_THRESHOLD_2D:
                             continue

                    if self.method == 'numerical_flux' and self.lib:
                        state = InterfaceState()
                        state.h1 = h1
                        state.z1 = z1
                        if getattr(node, 'lateral', False):
                            state.u1 = 0.0
                        else:
                            state.u1 = u1
                        state.h2 = h2
                        state.z2 = z2
                        state.u2 = u2
                        state.v2 = v2
                        
                        state.z_bank = crest
                        state.nx = float(node.nx)
                        state.ny = float(node.ny)
                        state.length = float(node.length)
                        state.discharge_coeff = float(node.params.get('coeff', 0.4))
                        
                        flux = CouplingFlux()
                        self.lib.compute_coupling_flux(state, ctypes.byref(flux))
                        
                        Q = flux.mass_flux
                        
                        # [FIX] Outflow Protection (Prevent Wave Front Suicide)
                        if Q > 0: # 1D -> 2D
                            if h1 < SHALLOW_WAVE_THRESHOLD: 
                                Q = 0.0
                        
                        # [FIX] Inflow Damping
                        if Q < 0 and h1 < NOISE_FILTER_THRESHOLD_1D:
                            if hasattr(model_1d, 'w') and hasattr(model_1d, 'dx'):
                                width_1d = float(model_1d.w[idx1])
                                dx_1d = float(model_1d.dx)
                                max_inflow = (width_1d * dx_1d * INFLOW_DAMPING_RISE) / dt
                                if abs(Q) > max_inflow:
                                    Q = -max_inflow
                                     
                        if Q > 0: # 1D -> 2D
                            nx = float(node.nx)
                            ny = float(node.ny)
                            tx = -ny
                            ty = nx
                            
                            u_t = 0.0
                            contact_len = float(node.length)
                            h_flow = h1 if h1 > DRY_THRESHOLD else DRY_THRESHOLD
                            u_n = Q / (contact_len * h_flow)
                            
                            u_in = u_n * nx + u_t * tx
                            v_in = u_n * ny + u_t * ty
                            
                            mx = Q * u_in
                            my = Q * v_in
                            
                        else: # 2D -> 1D
                            mx = Q * u2
                            my = Q * v2
                        
                    else:
                        # Empirical Method (Simplified for robustness)
                        if node.type == 'weir':
                            width = float(node.params.get('width', 1.0))
                            Cd = float(node.params.get('coeff', 0.4))
                            min_head = 0.001

                            if wse_1d > wse_2d:
                                head = wse_1d - crest
                                if head > min_head and h1 > DRY_THRESHOLD:
                                    Q = Cd * width * np.sqrt(2 * self.g) * np.power(head, 1.5)
                            elif wse_2d > wse_1d:
                                head = wse_2d - crest
                                if head > min_head and h2 > DRY_THRESHOLD:
                                    Q = -Cd * width * np.sqrt(2 * self.g) * np.power(head, 1.5)
                            
                            # Apply limits similar to numerical flux
                            if abs(Q) > 50000.0: Q = np.sign(Q) * 50000.0

                        elif node.type == 'orifice':
                            invert = float(node.params.get('invert_level', min(z1, z2)))
                            area = float(node.params.get('area', 1.0))
                            Cd = float(node.params.get('coeff', 0.6))

                            if (z1 + h1) > invert or (z2 + h2) > invert:
                                delta_h = (z1 + h1) - (z2 + h2)
                                Q = float(np.sign(delta_h) * Cd * area * np.sqrt(2 * self.g * abs(delta_h)))
                    
                    # Limits (CFL-like) - Moved here to apply to both Numerical and Empirical
                    if Q > 0:
                        if hasattr(model_1d, 'w') and hasattr(model_1d, 'dx'):
                            width_1d = float(model_1d.w[idx1])
                            dx_1d = float(model_1d.dx)
                            max_Q = 0.5 * (h1 * width_1d * dx_1d) / dt
                            if Q > max_Q:
                                ratio = max_Q / Q
                                Q = max_Q
                                mx *= ratio
                                my *= ratio
                    elif Q < 0:
                        if hasattr(self.model_2d, 'dx') and hasattr(self.model_2d, 'dy'):
                            dx_2d = float(self.model_2d.dx)
                            dy_2d = float(self.model_2d.dy)
                            max_Q = 0.5 * (h2 * dx_2d * dy_2d) / dt
                            if abs(Q) > max_Q:
                                ratio = max_Q / abs(Q)
                                Q = -max_Q
                                mx *= ratio
                                my *= ratio
                                
                    if abs(Q) > 50000.0:
                        ratio = 50000.0 / abs(Q)
                        Q = np.sign(Q) * 50000.0
                        mx *= ratio
                        my *= ratio

                    # Store Q in node for debugging/visualization
                    node.current_q = Q
                    
                    # --- Stability Check (Prevent Overshoot) ---
                    # (Keep existing stability logic)
                    A1_stab = 0.1 
                    if hasattr(model_1d, 'w') and hasattr(model_1d, 'dx'):
                         try: 
                             if 0 <= idx1 < len(model_1d.w):
                                 val = float(model_1d.w[idx1] * model_1d.dx)
                                 if val > 0: A1_stab = val
                         except: pass
                    elif hasattr(model_1d, 'area'):
                         try: 
                             if 0 <= idx1 < len(model_1d.area):
                                 val = float(model_1d.area[idx1])
                                 if val > 0: A1_stab = val
                         except: pass
                    
                    A2_stab = 1.0
                    if hasattr(self.model_2d, 'dx') and hasattr(self.model_2d, 'dy'):
                        A2_stab = float(self.model_2d.dx * self.model_2d.dy)

                    wse_1d_stab = z1 + h1
                    wse_2d_stab = z2 + h2
                    delta_h_stab = wse_1d_stab - wse_2d_stab
                    denom_stab = dt * (1.0/A1_stab + 1.0/A2_stab)
                    
                    if denom_stab > 1e-12:
                        max_Q_stab = abs(delta_h_stab) / denom_stab
                        if (Q > 0 and delta_h_stab < -0.001) or (Q < 0 and delta_h_stab > 0.001):
                             Q = 0.0
                             mx = 0.0
                             my = 0.0
                        elif Q > 0 and delta_h_stab > 0: 
                             if Q > max_Q_stab:
                                 ratio = max_Q_stab / Q
                                 Q = max_Q_stab
                                 mx *= ratio
                                 my *= ratio
                        elif Q < 0 and delta_h_stab < 0: 
                             if abs(Q) > max_Q_stab:
                                 ratio = max_Q_stab / abs(Q)
                                 Q = -max_Q_stab
                                 mx *= ratio
                                 my *= ratio
                    
                    # --- Hard Safety Limit for 1D Inflow (2D -> 1D) ---
                    if Q < 0: 
                        max_rise = 1.0 
                        max_Q_safety = (max_rise * A1_stab) / dt
                        if abs(Q) > max_Q_safety:
                            ratio = max_Q_safety / abs(Q)
                            Q = -max_Q_safety
                            mx *= ratio
                            my *= ratio

                    node.current_q = float(Q)
                    
                    # Accumulate 1D Source
                    src_1d[(model_1d, idx1)] = src_1d.get((model_1d, idx1), 0.0) + (-Q)
                    
                    # Accumulate to Unique Array
                    u_idx = self.unique_map[i]
                    # [FIX] Bounds Check for Unique Map
                    if 0 <= u_idx < num_unique:
                        unique_q[u_idx] += Q
                        unique_mx[u_idx] += mx
                        unique_my[u_idx] += my
                    
                except Exception:
                    node.current_q = 0.0
            
            # 4. Batch Apply 2D Sources
            if hasattr(self.model_2d, 'update_coupling_sources'):
                self.model_2d.update_coupling_sources(unique_q, unique_mx, unique_my)
            else:
                self.model_2d.set_coupling_sources(self.unique_rows, self.unique_cols, unique_q, unique_mx, unique_my)
            
            # 5. Apply 1D Sources
            for (model_1d, idx1), q1 in src_1d.items():
                try:
                    model_1d.set_source(idx1, float(q1))
                except Exception as e:
                    print(f"[Coupler] Failed to set source for 1D model: {e}")

        except Exception as e:
            print(f"[WARNING] compute_exchange_optimized failed: {e}")


    def sync_velocity_for_visualization(self):
        """
        Force 2D velocity to match 1D velocity direction for coupled nodes.
        This fixes the arrow direction in visualization.
        """
        try:
            if not self.model_2d or not hasattr(self.model_2d, 'set_coupling_velocities'):
                return

            if not hasattr(self, 'cache_valid') or not self.cache_valid:
                self._build_cache()

            rows = []
            cols = []
            us = []
            vs = []
            
            model_cache = {}
            
            nodes_to_use = self.active_nodes if hasattr(self, 'active_nodes') else self.nodes

            for node in nodes_to_use:
                # Skip if not active or not in grid
                if 'flow_angle' not in node.params:
                    continue
                
                model_1d = node.model_1d if node.model_1d is not None else self.model_1d
                if model_1d is None: continue
                
                idx1 = int(node.idx_1d)
                
                # Cache results
                if model_1d not in model_cache:
                    try:
                        _, u_arr = model_1d.get_results()
                        model_cache[model_1d] = u_arr
                    except:
                        model_cache[model_1d] = None
                
                u_arr = model_cache[model_1d]
                if u_arr is None or not (0 <= idx1 < len(u_arr)):
                    continue
                    
                u_1d = float(u_arr[idx1])
                
                # Get flow angle
                angle = float(node.params.get('flow_angle', 0.0))
                
                # Calculate 2D components
                # u_1d is signed. Positive = downstream (along angle). Negative = upstream (opposite).
                u_vis = u_1d * np.cos(angle)
                v_vis = u_1d * np.sin(angle)
                
                rows.append(node.r_2d)
                cols.append(node.c_2d)
                us.append(u_vis)
                vs.append(v_vis)
                
            if rows:
                self.model_2d.set_coupling_velocities(
                    np.array(rows), np.array(cols), 
                    np.array(us), np.array(vs)
                )
        except Exception as e:
            # Log error but don't crash
            print(f"[WARNING] sync_velocity_for_visualization failed: {e}")


    def compute_exchange(self, dt):
        """
        Compute exchange flow for all nodes and apply source terms.
        """
        if self.model_2d is None:
            return

        h_2d_arr, u_2d_arr, v_2d_arr = self.model_2d.get_results()
        z_2d = self.model_2d.elevation_flat.reshape((self.model_2d.rows, self.model_2d.cols))

        model_cache = {}
        src_1d = {}
        src_2d = {}
        mom_2d = {}

        for node in self.nodes:
            idx1 = int(node.idx_1d)
            r2, c2 = int(node.r_2d), int(node.c_2d)

            model_1d = node.model_1d if node.model_1d is not None else self.model_1d
            if model_1d is None:
                node.current_q = 0.0
                continue

            if not (0 <= r2 < self.model_2d.rows and 0 <= c2 < self.model_2d.cols):
                node.current_q = 0.0
                continue

            # Check Inactive Cell (AOI Mask)
            if self.bc_type_grid is not None:
                # Assuming bc_type_grid has same shape as model
                if self.bc_type_grid[r2, c2] == -1:
                    node.current_q = 0.0
                    continue

            cache_key = model_1d
            if cache_key not in model_cache:
                try:
                    h_1d_arr, u_1d_arr = model_1d.get_results()
                    z_1d = model_1d.z
                except Exception:
                    h_1d_arr, u_1d_arr, z_1d = None, None, None
                model_cache[cache_key] = (h_1d_arr, u_1d_arr, z_1d)

            h_1d_arr, u_1d_arr, z_1d = model_cache[cache_key]
            if h_1d_arr is None or u_1d_arr is None:
                node.current_q = 0.0
                continue

            if not (0 <= idx1 < len(h_1d_arr) and 0 <= idx1 < len(u_1d_arr)):
                node.current_q = 0.0
                continue

            try:
                h1 = float(h_1d_arr[idx1])
                u1 = float(u_1d_arr[idx1])
                if np.ndim(z_1d) == 0:
                    z1 = float(z_1d)
                else:
                    z1 = float(z_1d[idx1])

                h2 = float(h_2d_arr[r2, c2])
                z2 = float(z_2d[r2, c2])
                u2 = float(u_2d_arr[r2, c2])
                v2 = float(v_2d_arr[r2, c2])

                Q = 0.0
                mx = 0.0
                my = 0.0

                if self.method == 'numerical_flux' and self.lib:
                    # --- Pre-Check: Uphill Flooding & Dry Bed ---
                    # To align with Empirical method safeguards
                    wse_1d = z1 + h1
                    wse_2d = z2 + h2
                    crest = float(node.params.get('crest_level', max(z1, z2)))
                    
                    # 1. 1D -> 2D Flow Block
                    # If 2D is effectively dry, and 1D Water Level is below 2D Terrain/Crest, DO NOT allow flux.
                    # (Prevent water jumping up to a dry cell)
                    block_flux = False
                    if h2 < 0.001: 
                        if wse_1d < (z2 + 0.001): block_flux = True # Below terrain
                        if wse_1d < crest: block_flux = True # Below Bank
                    
                    # 2. 2D -> 1D Flow Block
                    # If 1D is dry, and 2D Water Level is below 1D Bed/Crest, DO NOT allow flux.
                    if h1 < 0.001:
                        if wse_2d < (z1 + 0.001): block_flux = True
                        if wse_2d < crest: block_flux = True

                    if not block_flux:
                        state = InterfaceState()
                        state.h1 = h1
                        state.z1 = z1
                        
                        # For lateral coupling, 1D velocity u1 is tangential to the interface.
                        # The normal velocity component from 1D side is 0.
                        # For end-coupling (river mouth), u1 is the normal velocity.
                        if getattr(node, 'lateral', False):
                            state.u1 = 0.0
                        else:
                            state.u1 = u1
                        
                        state.h2 = h2
                        state.z2 = z2
                        state.u2 = u2
                        state.v2 = v2

                        state.z_bank = crest
                        state.nx = float(node.nx)
                        state.ny = float(node.ny)
                        state.length = float(node.length)
                        state.discharge_coeff = float(node.params.get('coeff', 1.0))

                        flux = CouplingFlux()
                        self.lib.compute_coupling_flux(state, ctypes.byref(flux))

                        Q = float(flux.mass_flux)
                        
                        # --- Post-Process: Prevent Uphill Flow ---
                        # If River is below Bank (z2), Flow 1D->2D (Q > 0) is physically impossible via gravity/weir.
                        # Unless it's a jet, but here we assume shallow water.
                        if Q > 0 and wse_1d < (z2 - 0.01):
                            Q = 0.0
                        
                        # --- Recompute Momentum Flux (Advective Only) ---
                        # The DLL returns Total Momentum Flux (including Hydrostatic Pressure).
                        # For a Source Term injection, we should only inject Advective Momentum (Q * u).
                        # Injecting Pressure term causes "Ghost Force" and excessive water piling.
                        
                        mx = 0.0
                        my = 0.0
                        
                        # Normalize Normal Vector
                        nx = float(node.nx)
                        ny = float(node.ny)
                        norm = np.sqrt(nx*nx + ny*ny)
                        if norm > 1e-9:
                            nx /= norm
                            ny /= norm
                        tx = -ny
                        ty = nx
                        
                        if abs(Q) > 1e-9:
                            if Q > 0: # 1D -> 2D
                                # Flow entering 2D.
                                # Determine entering velocity vector.
                                # For lateral weir, u_longitudinal (tangential) is u1.
                                # u_normal is derived from Q.
                                
                                # 1. Tangential Component
                                u_t = 0.0
                                if getattr(node, 'lateral', False):
                                    u_t = u1 # Preserve longitudinal momentum
                                else:
                                    # End coupling: u1 is normal velocity. Tangential is 0?
                                    # Or u1 vector is (u1, 0) in 1D coords.
                                    # Let's assume u1 is purely normal for end coupling.
                                    u_t = 0.0
                                    
                                # 2. Normal Component
                                # Estimate flow area
                                contact_len = float(node.length)
                                # Effective depth on 1D side? Or 2D side?
                                # Use Upwind Depth
                                h_flow = h1 if h1 > 0.001 else 0.001
                                u_n = Q / (contact_len * h_flow)
                                
                                # 3. Combine
                                # v = u_n * n + u_t * t
                                u_in = u_n * nx + u_t * tx
                                v_in = u_n * ny + u_t * ty
                                
                                # Momentum Flux = Mass Flux * Velocity
                                mx = Q * u_in
                                my = Q * v_in
                                
                            else: # 2D -> 1D (Q < 0)
                                # Flow leaving 2D.
                                # Momentum is removed based on 2D velocity.
                                # This is a sink term.
                                # Sink momentum = Q * u_cell
                                mx = Q * u2
                                my = Q * v2
                        
                        # --- Post-Check: Stability Limiter ---
                        # Prevent excessive drainage in one step (similar to empirical method)
                        
                        # Limit 1D Drainage
                        if Q > 0: # 1D -> 2D
                            if hasattr(model_1d, 'w') and hasattr(model_1d, 'dx'):
                                width_1d = float(model_1d.w[idx1])
                                dx_1d = float(model_1d.dx)
                                # Limit to 50% of available volume
                                max_Q = 0.5 * (h1 * width_1d * dx_1d) / dt
                                if Q > max_Q:
                                    ratio = max_Q / Q
                                    Q = max_Q
                                    mx *= ratio
                                    my *= ratio
                        
                        # Limit 2D Drainage
                        elif Q < 0: # 2D -> 1D
                            if hasattr(self.model_2d, 'dx') and hasattr(self.model_2d, 'dy'):
                                dx_2d = float(self.model_2d.dx)
                                dy_2d = float(self.model_2d.dy)
                                # Limit to 50% of available volume
                                max_Q = 0.5 * (h2 * dx_2d * dy_2d) / dt
                                if abs(Q) > max_Q:
                                    ratio = max_Q / abs(Q)
                                    Q = -max_Q
                                    mx *= ratio
                                    my *= ratio
                                    
                        # Absolute Safety Clamp
                        if abs(Q) > 50000.0:
                            ratio = 50000.0 / abs(Q)
                            Q = np.sign(Q) * 50000.0
                            mx *= ratio
                            my *= ratio
                    else:
                        Q = 0.0
                        mx = 0.0
                        my = 0.0
                else:
                    wse_1d = z1 + h1
                    wse_2d = z2 + h2

                    if node.type == 'weir':
                        crest = float(node.params.get('crest_level', max(z1, z2)))
                        width = float(node.params.get('width', 1.0))
                        Cd = float(node.params.get('coeff', 0.4))

                        min_head = 0.001

                        if wse_1d > wse_2d:
                            head = wse_1d - crest
                            # Allow shallow overflow from 1D to 2D
                            if head > min_head and h1 > 0.001:
                                Q = Cd * width * np.sqrt(2 * self.g) * np.power(head, 1.5)

                                A1 = 100.0
                                if hasattr(model_1d, 'w') and hasattr(model_1d, 'dx'):
                                    try:
                                        A1 = float(model_1d.w[idx1] * model_1d.dx)
                                    except Exception:
                                        pass
                                elif hasattr(model_1d, 'nodes') and hasattr(model_1d.nodes[idx1], 'area'):
                                    try:
                                        A1 = float(model_1d.nodes[idx1].area)
                                    except Exception:
                                        pass
                                elif hasattr(model_1d, 'area'):
                                    try:
                                        A1 = float(model_1d.area)
                                    except Exception:
                                        pass
                                if A1 < 0.1:
                                    A1 = 100.0

                                A2 = 100.0
                                if hasattr(self.model_2d, 'dx') and hasattr(self.model_2d, 'dy'):
                                    A2 = float(self.model_2d.dx * self.model_2d.dy)

                                denom = dt * (1.0 / A1 + 1.0 / A2)
                                if denom > 1e-9:
                                    Q_eq = (wse_1d - wse_2d) / denom
                                    if Q > Q_eq:
                                        Q = Q_eq

                                if hasattr(model_1d, 'w') and hasattr(model_1d, 'dx'):
                                    width_1d = float(model_1d.w[idx1])
                                    dx_1d = float(model_1d.dx)
                                    max_Q = 0.5 * (h1 * width_1d * dx_1d) / dt
                                    if Q > max_Q:
                                        Q = max_Q
                        else:
                            head = wse_2d - crest
                            # Increase threshold to 0.02 to prevent noise from entering 1D
                            if head > min_head and h2 > 0.02:
                                Q = -Cd * width * np.sqrt(2 * self.g) * np.power(head, 1.5)

                                A1 = 100.0
                                if hasattr(model_1d, 'w') and hasattr(model_1d, 'dx'):
                                    try:
                                        A1 = float(model_1d.w[idx1] * model_1d.dx)
                                    except Exception:
                                        pass
                                elif hasattr(model_1d, 'nodes') and hasattr(model_1d.nodes[idx1], 'area'):
                                    try:
                                        A1 = float(model_1d.nodes[idx1].area)
                                    except Exception:
                                        pass
                                if A1 < 0.1:
                                    A1 = 100.0

                                A2 = 100.0
                                if hasattr(self.model_2d, 'dx') and hasattr(self.model_2d, 'dy'):
                                    A2 = float(self.model_2d.dx * self.model_2d.dy)

                                denom = dt * (1.0 / A1 + 1.0 / A2)
                                if denom > 1e-9:
                                    Q_eq = (wse_2d - wse_1d) / denom
                                    if Q < -Q_eq:
                                        Q = -Q_eq

                                if hasattr(self.model_2d, 'dx') and hasattr(self.model_2d, 'dy'):
                                    dx_2d = float(self.model_2d.dx)
                                    dy_2d = float(self.model_2d.dy)
                                    max_Q = 0.5 * (h2 * dx_2d * dy_2d) / dt
                                    if abs(Q) > max_Q:
                                        Q = -max_Q

                                if Q < -50000.0:
                                    Q = -50000.0
                    elif node.type == 'orifice':
                        invert = float(node.params.get('invert_level', min(z1, z2)))
                        area = float(node.params.get('area', 1.0))
                        Cd = float(node.params.get('coeff', 0.6))

                        if (z1 + h1) > invert or (z2 + h2) > invert:
                            delta_h = (z1 + h1) - (z2 + h2)
                            direction = np.sign(delta_h)
                            Q = float(direction * Cd * area * np.sqrt(2 * self.g * abs(delta_h)))

                # --- Stability Check (Prevent Overshoot) ---
                # Calculate areas if not already done
                A1_stab = 100.0
                if hasattr(model_1d, 'w') and hasattr(model_1d, 'dx'):
                     try: A1_stab = float(model_1d.w[idx1] * model_1d.dx)
                     except: pass
                elif hasattr(model_1d, 'area'):
                     try: A1_stab = float(model_1d.area[idx1])
                     except: pass
                
                A2_stab = 1.0
                if hasattr(self.model_2d, 'dx') and hasattr(self.model_2d, 'dy'):
                    A2_stab = float(self.model_2d.dx * self.model_2d.dy)

                # Equilibrium Logic:
                # Q*dt * (1/A1 + 1/A2) <= |(z1+h1) - (z2+h2)|
                wse_1d_stab = z1 + h1
                wse_2d_stab = z2 + h2
                delta_h_stab = wse_1d_stab - wse_2d_stab
                denom_stab = dt * (1.0/A1_stab + 1.0/A2_stab)
                
                if denom_stab > 1e-12:
                    max_Q_stab = abs(delta_h_stab) / denom_stab
                    
                    # Apply limit only if Q is driving towards equilibrium
                    if Q > 0 and delta_h_stab > 0: # 1D -> 2D
                         if Q > max_Q_stab:
                             ratio = max_Q_stab / Q
                             Q = max_Q_stab
                             mx *= ratio
                             my *= ratio
                    elif Q < 0 and delta_h_stab < 0: # 2D -> 1D
                         if abs(Q) > max_Q_stab:
                             ratio = max_Q_stab / abs(Q)
                             Q = -max_Q_stab
                             mx *= ratio
                             my *= ratio

                node.current_q = float(Q)

                src_1d[(model_1d, idx1)] = src_1d.get((model_1d, idx1), 0.0) + (-Q)
                src_2d[(r2, c2)] = src_2d.get((r2, c2), 0.0) + Q

                if self.method == 'numerical_flux' and self.lib and (mx != 0.0 or my != 0.0):
                    prev = mom_2d.get((r2, c2))
                    if prev is None:
                        mom_2d[(r2, c2)] = (mx, my)
                    else:
                        mom_2d[(r2, c2)] = (prev[0] + mx, prev[1] + my)
            except Exception:
                node.current_q = 0.0

        for (model_1d, idx1), q1 in src_1d.items():
            try:
                model_1d.set_source(idx1, float(q1))
            except Exception:
                pass

        for (r2, c2), q2 in src_2d.items():
            try:
                self.model_2d.set_source(int(r2), int(c2), float(q2))
            except Exception:
                pass

        if self.method == 'numerical_flux' and self.lib and mom_2d and hasattr(self.model_2d, 'set_source_momentum'):
            for (r2, c2), (mx, my) in mom_2d.items():
                try:
                    self.model_2d.set_source_momentum(int(r2), int(c2), float(mx), float(my))
                except Exception:
                    pass
