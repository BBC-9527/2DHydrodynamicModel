import numpy as np
try:
    from swe_1d import SWEModel1D, DRY_THRESHOLD, WET_THRESHOLD
except ImportError:
    from swe_1d import SWEModel1D
    DRY_THRESHOLD = 0.001
    WET_THRESHOLD = 0.01

class NetworkNode:
    def __init__(self, node_id, x, y, z, area=100.0):
        self.id = node_id
        self.x = x
        self.y = y
        self.z = z # Bed elevation
        self.h = 0.0 # Water depth relative to z
        self.area = area # Storage area (m2)
        
        # Connections
        # List of (edge_id, 'start' or 'end')
        self.connections = []
        
        # Boundary Conditions
        self.bc_type = None # 'inflow' or None
        self.bc_value = 0.0
        
        # Lateral Flux (from Coupling)
        self.lateral_flux = 0.0

    def set_source(self, idx, Q):
        """
        Add lateral source/sink flux (m3/s).
        idx is ignored for nodes.
        Q > 0 means INFLOW to node.
        """
        self.lateral_flux += Q

    def get_results(self):
        """Duck-typing for Coupler"""
        return np.array([self.h]), np.array([0.0])

class NetworkEdge:
    def __init__(self, edge_id, model, start_node_id, end_node_id, geometry=None):
        self.id = edge_id
        self.model = model # SWEModel1D instance
        self.start_node_id = start_node_id # Connected to Left (Index 0)
        self.end_node_id = end_node_id     # Connected to Right (Index N-1)
        self.geometry = geometry # Optional: List of [lat, lon] for visualization

class Network1D:
    def __init__(self):
        self.nodes = {} # id -> NetworkNode
        self.edges = {} # id -> NetworkEdge
        
    def add_node(self, node_id, x, y, z, area=None):
        if area is None:
            area = 500.0
        node = NetworkNode(node_id, x, y, z, area)
        self.nodes[node_id] = node
        
    def add_edge(self, edge_id, start_node_id, end_node_id, width, n, z_start, z_end, dx=10.0, width_end=None, geometry=None, length=None):
        # Calculate length from coordinates
        if start_node_id not in self.nodes or end_node_id not in self.nodes:
            raise ValueError("Nodes must exist before creating edge")
            
        n1 = self.nodes[start_node_id]
        n2 = self.nodes[end_node_id]
        
        if length is not None and length > 0:
            dist = length
        else:
            dist = np.sqrt((n2.x - n1.x)**2 + (n2.y - n1.y)**2)
        
        # Create Model
        num_cells = int(dist / dx)
        if num_cells < 2: num_cells = 2
        real_dx = dist / num_cells
        
        model = SWEModel1D(num_cells, real_dx)
        
        # Init Arrays
        # Z: Linear Interp
        z_arr = np.linspace(z_start, z_end, num_cells)
        
        # Check for Uphill Slope (Warning)
        if z_start < z_end:
            print(f"[WARNING] Edge {edge_id} has UPHILL slope (Start Z={z_start:.2f} < End Z={z_end:.2f}).")
            print("          Gravity will drive flow backwards (End -> Start). Verify this is intended.")
        
        n_arr = np.full(num_cells, n)
        
        # Width: Linear Interp or Constant
        if width_end is None:
            w_arr = np.full(num_cells, width)
            w_end_val = width
        else:
            w_arr = np.linspace(width, width_end, num_cells)
            w_end_val = width_end
            
        h_arr = np.full(num_cells, 0.0) # Start dry or init later
        
        model.init(z_arr, n_arr, w_arr, h_arr)
        
        edge = NetworkEdge(edge_id, model, start_node_id, end_node_id, geometry=geometry)
        self.edges[edge_id] = edge
        
        # Register connections
        self.nodes[start_node_id].connections.append((edge_id, 'start'))
        self.nodes[end_node_id].connections.append((edge_id, 'end'))
        
        # Update Node Area heuristic (add full cell area for better stability)
        # A node represents a junction. Its area should be proportional to connected channels.
        # Previously used 0.5 * area, but for large flux, larger area dampens rapid level changes.
        self.nodes[start_node_id].area += width * real_dx
        self.nodes[end_node_id].area += w_end_val * real_dx
        
    def set_node_h(self, node_id, h):
        if node_id in self.nodes:
            self.nodes[node_id].h = h
            
    def set_node_bc(self, node_id, bc_type, bc_value):
        if node_id in self.nodes:
            self.nodes[node_id].bc_type = bc_type
            self.nodes[node_id].bc_value = bc_value

    def set_initial_condition(self, h_val):
        """Set uniform initial water depth for the entire network"""
        # Set all nodes
        for node in self.nodes.values():
            node.h = h_val
            
        # Set all edges
        for edge in self.edges.values():
            h_arr = np.full(edge.model.num_cells, h_val)
            # Re-initialize model with existing Z, N, W and new H
            edge.model.init(edge.model.z, edge.model.n, edge.model.w, h_arr)

    def compute_hll_flux(self, h_L, u_L, h_R, u_R):
        """
        Compute HLL Mass Flux across an interface.
        Mimics the C++ solver's boundary flux calculation.
        """
        g = 9.81
        if h_L <= 1e-6 and h_R <= 1e-6:
            return 0.0
            
        c_L = np.sqrt(g * h_L)
        c_R = np.sqrt(g * h_R)
        
        # Wave speeds
        S_L = min(u_L - c_L, u_R - c_R)
        S_R = max(u_L + c_L, u_R + c_R)
        
        if S_L >= 0:
            return h_L * u_L
        elif S_R <= 0:
            return h_R * u_R
        else:
            F_h_L = h_L * u_L
            F_h_R = h_R * u_R
            # HLL Flux formula
            return (S_R * F_h_L - S_L * F_h_R + S_L * S_R * (h_R - h_L)) / (S_R - S_L)

    def run_step(self, dt, sim_time=None):
        # 1. Update Junctions (Mass Balance)
        # Calculate net flux into each node
        node_flux = {nid: 0.0 for nid in self.nodes}
        
        # Soft Start Factor (Ramp up Inflow)
        ramp_factor = 1.0
        if sim_time is not None and sim_time < 10.0:
            ramp_factor = sim_time / 10.0
            if ramp_factor < 0: ramp_factor = 0
        
        # Apply External Inflow
        for nid, node in self.nodes.items():
            if node.bc_type == 'inflow':
                node_flux[nid] += node.bc_value * ramp_factor
            
            # Add Lateral Flux from Coupling
            node_flux[nid] += node.lateral_flux
            # Reset for next step
            node.lateral_flux = 0.0

        for eid, edge in self.edges.items():
            h, u = edge.model.get_results()
            # [FIX] Update the Python model's cached state so external readers (like Coupler) see current values
            edge.model.h = h
            edge.model.u = u
            
            w = edge.model.w # Width array
            
            # Clamp velocity in dry cells to avoid ghost flux
            # If h < threshold, force u = 0 for flux calculation
            dry_mask = h < DRY_THRESHOLD
            u_clamped = u.copy()
            u_clamped[dry_mask] = 0.0
            
            n_start = self.nodes[edge.start_node_id]
            n_end = self.nodes[edge.end_node_id]

            # --- Left Boundary (Start Node) ---
            # Determine Boundary Depth (h_L) based on Node WSE
            wse_start = n_start.z + n_start.h
            z_c_start = edge.model.get_bed_elevation(0)
            
            # [FIX] Use DRY_THRESHOLD instead of WET_THRESHOLD to allow earlier inflow
            if n_start.h < DRY_THRESHOLD:
                h_bc_left = 0.0
            else:
                h_bc_left = max(0.0, wse_start - max(n_start.z, z_c_start))
            
            # Compute HLL Flux entering the edge (Positive = Into Edge)
            # L state = Boundary (Node), R state = Cell 0
            # C++ uses u_L = u_R (Zero order extrapolation)
            h_R_safe = max(h[0], 0.0)
            q_flux_left = self.compute_hll_flux(h_bc_left, u_clamped[0], h_R_safe, u_clamped[0])
            
            # Total Discharge = Flux * Width
            Q_left = q_flux_left * w[0]
            
            # Flow INTO node = - Q_left (since Q_left is into edge)
            node_flux[edge.start_node_id] -= Q_left
            
            # --- Right Boundary (End Node) ---
            # End Node (Right)
            wse_end = n_end.z + n_end.h
            z_c_end = edge.model.get_bed_elevation(edge.model.num_cells - 1)
            
            # [FIX] Use DRY_THRESHOLD instead of WET_THRESHOLD
            if n_end.h < DRY_THRESHOLD:
                h_bc_right = 0.0
            else:
                h_bc_right = max(0.0, wse_end - max(n_end.z, z_c_end))
                
            # Compute HLL Flux leaving the edge (Positive = Out of Edge? No, standard formula)
            # L state = Cell -1, R state = Boundary (Node)
            # C++ uses u_R = u_L (Zero order extrapolation)
            h_L_safe = max(h[-1], 0.0)
            q_flux_right = self.compute_hll_flux(h_L_safe, u_clamped[-1], h_bc_right, u_clamped[-1])
            
            # Total Discharge = Flux * Width
            Q_right = q_flux_right * w[-1]
            
            # Flow INTO node = + Q_right (Flow from Left to Right is positive)
            node_flux[edge.end_node_id] += Q_right
            
        # Update Node Water Levels
        for nid, node in self.nodes.items():
            # Handle Outflow Boundary (Sink / Fixed Head)
            if node.bc_type == 'outflow':
                # Do not update H. It stays at initial value (e.g. 0.0 or fixed level).
                # Flux entering the node is effectively removed from the system.
                continue
            
            # Safety check for NaN flux
            if np.isnan(node_flux[nid]):
                 # Log error but don't crash?
                 # print(f"[ERROR] Node {nid} flux is NaN! Resetting flux to 0.")
                 node_flux[nid] = 0.0

            # dH = (Q_net * dt) / Area
            dh = (node_flux[nid] * dt) / node.area
            node.h += dh
            if node.h < DRY_THRESHOLD: node.h = 0.0 # Clamp small negative or positive noise to 0
            if np.isnan(node.h): 
                 node.h = 0.0

            
            # DEBUG
            # if node.h > 0:
            #     print(f"[DEBUG] Node {nid} H={node.h:.4f}, Flux={node_flux[nid]:.4f}")

        # 2. Set Boundary Conditions for Edges
        for eid, edge in self.edges.items():
            n_start = self.nodes[edge.start_node_id]
            n_end = self.nodes[edge.end_node_id]
            
            # Set BCs
            # Fixed Depth = Node Water Level
            # IMPORTANT: Node H is depth relative to Node Z.
            # Channel Boundary H is depth relative to Channel Z.
            # Usually we match Water Surface Elevation (WSE).
            # WSE_node = n_start.z + n_start.h
            # H_channel_start = WSE_node - z_channel_start
            
            # Start Node (Left)
            # Check for Outflow BC (Transmissive)
            if n_start.bc_type == 'outflow':
                # Use Type 3 (Open/Transmissive)
                edge.model.set_boundary('left', 3, 0.0)
            else:
                # Normal Case or Inflow (Fixed Depth based on Node H)
                wse_start = n_start.z + n_start.h
                z_c_start = edge.model.get_bed_elevation(0)
                
                if n_start.h < DRY_THRESHOLD:
                    h_bc_left = 0.0
                else:
                    h_bc_left = max(0.0, wse_start - max(n_start.z, z_c_start))
                    
                edge.model.set_boundary('left', 1, h_bc_left)
                
            # End Node (Right)
            # Check for Outflow BC (Transmissive)
            if n_end.bc_type == 'outflow':
                # Use Type 3 (Open/Transmissive)
                edge.model.set_boundary('right', 3, 0.0)
            else:
                wse_end = n_end.z + n_end.h
                z_c_end = edge.model.get_bed_elevation(edge.model.num_cells - 1)
                
                if n_end.h < DRY_THRESHOLD:
                    h_bc_right = 0.0
                else:
                    h_bc_right = max(0.0, wse_end - max(n_end.z, z_c_end))
                
                edge.model.set_boundary('right', 1, h_bc_right)
            
        # 3. Run Steps with Adaptive Sub-stepping (CFL Stability)
        for edge in self.edges.values():
            # Get max wave speed for CFL check
            h, u = edge.model.get_results()
            
            # Safety check for NaNs
            if np.any(np.isnan(h)) or np.any(np.isnan(u)):
                print(f"[ERROR] Edge {edge.id} unstable (NaN detected)! Resetting to dry.")
                # Re-initialize to dry to clear internal state (C++ or Taichi)
                if hasattr(edge.model, 'init') and edge.model.z is not None:
                     zero_h = np.zeros(edge.model.num_cells, dtype=np.float64)
                     edge.model.init(edge.model.z, edge.model.n, edge.model.w, zero_h)
                else:
                     # Fallback for models that might expose arrays directly (Taichi)
                     if hasattr(edge.model, 'h'): edge.model.h[:] = 0.0
                     if hasattr(edge.model, 'u'): edge.model.u[:] = 0.0
                continue
                
            g = 9.81
            # c = sqrt(gh) + |u|
            # Add epsilon to avoid sqrt(negative)
            h_safe = np.maximum(h, 0.0)
            c_wave = np.sqrt(g * h_safe) + np.abs(u)
            max_c = np.max(c_wave)
            if max_c < 0.1: max_c = 0.1
            
            dx = edge.model.dx
            # Target CFL = 0.7 for safety
            dt_max = 0.7 * dx / max_c
            
            remaining_dt = dt
            while remaining_dt > 1e-9:
                sub_dt = min(remaining_dt, dt_max)
                edge.model.run_step(sub_dt)
                remaining_dt -= sub_dt
