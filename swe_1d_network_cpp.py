import ctypes
import os
import numpy as np

# Load DLL
dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "swe_1d_network.dll"))
try:
    lib = ctypes.CDLL(dll_path)
except OSError:
    print(f"Error: Could not load DLL at {dll_path}")
    lib = None

if lib:
    # Define Types
    lib.Network_Create.restype = ctypes.c_void_p
    lib.Network_AddReach.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, 
                                     np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
                                     np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
                                     np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
                                     np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
                                     np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
                                     np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')]
    lib.Network_AddJunction.argtypes = [ctypes.c_void_p, ctypes.c_int, 
                                        np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
                                        np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
                                        ctypes.c_int]
    lib.Network_SetJunctionSource.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
    lib.Network_Step.argtypes = [ctypes.c_void_p, ctypes.c_double]
    lib.Network_GetResults.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                       np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
                                       np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')]
    lib.Network_Delete.argtypes = [ctypes.c_void_p]

class CPPNode:
    def __init__(self, id, x, y, z):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.h = 0.0
        self.connections = []
        self.bc_type = None
        self.bc_value = 0.0

class CPPEdgeModel:
    def __init__(self, num_cells, z, w, n):
        self.num_cells = num_cells
        self.z = z
        self.w = w
        self.n = n
        self.h = np.zeros(num_cells)
        self.u = np.zeros(num_cells)
        
    def init(self, z, n, w, h):
        self.h[:] = h

class CPPEdge:
    def __init__(self, id, start, end, geometry, model):
        self.id = id
        self.start_node_id = start
        self.end_node_id = end
        self.geometry = geometry
        self.model = model
        self.dx = 10.0 
        self.width_end = None

class Network1D_CPP:
    def __init__(self):
        if not lib:
            raise RuntimeError("DLL not loaded")
        self.obj = lib.Network_Create()
        self.reaches = {} # id -> metadata
        self.junctions = {}
        
        # Buffer for construction
        self.nodes_data = {} # id -> {x, y, z}
        self.edges_data = {} # id -> {start, end, width, n, ...}
        self.adj_list = {} # node_id -> [(edge_id, is_start)]
        
        # Public Interface Mimic
        self.nodes = {} # id -> CPPNode
        self.edges = {} # id -> CPPEdge
        self.built = False
        
    def add_node(self, node_id, x, y, z, area=None):
        self.nodes_data[node_id] = {'x': x, 'y': y, 'z': z}
        if node_id not in self.adj_list:
            self.adj_list[node_id] = []
            
        self.nodes[node_id] = CPPNode(node_id, x, y, z)

    def add_edge(self, edge_id, start_node, end_node, width, n, z_start, z_end, dx=10.0, width_end=None, geometry=None, length=None):
        # Calculate length
        n1 = self.nodes_data[start_node]
        n2 = self.nodes_data[end_node]
        
        if length is None or length <= 0:
            length = np.sqrt((n1['x'] - n2['x'])**2 + (n1['y'] - n2['y'])**2)
        
        self.edges_data[edge_id] = {
            'start': start_node,
            'end': end_node,
            'width': width,
            'n': n,
            'z_start': z_start,
            'z_end': z_end,
            'dx': dx,
            'length': length
        }
        
        self.adj_list[start_node].append((edge_id, True)) # True = Edge Starts here
        self.adj_list[end_node].append((edge_id, False)) # False = Edge Ends here
        
        # Create Dummy Model for Visualization
        num_cells = int(length / dx)
        if num_cells < 2: num_cells = 2
        z_arr = np.linspace(z_start, z_end, num_cells)
        w_arr = np.full(num_cells, width)
        n_arr = np.full(num_cells, n)
        model = CPPEdgeModel(num_cells, z_arr, w_arr, n_arr)
        
        self.edges[edge_id] = CPPEdge(edge_id, start_node, end_node, geometry, model)
        
        # Connections for Node objects (Validation)
        self.nodes[start_node].connections.append((edge_id, 'start'))
        self.nodes[end_node].connections.append((edge_id, 'end'))

    def set_node_bc(self, node_id, bc_type, value):
        if node_id in self.nodes:
            self.nodes[node_id].bc_type = bc_type
            self.nodes[node_id].bc_value = value
            
    def set_initial_condition(self, h):
        for n in self.nodes.values(): n.h = h
        for e in self.edges.values(): e.model.h[:] = h

    def build(self):
        """Construct the C++ Network from buffered data"""
        if self.built: return
        
        # 1. Add Reaches
        for eid, data in self.edges_data.items():
            # Re-calculate to match exactly
            num_cells = int(data['length'] / data['dx'])
            if num_cells < 2: num_cells = 2
            
            dist = np.linspace(0, data['length'], num_cells)
            z = np.linspace(data['z_start'], data['z_end'], num_cells)
            w = np.full(num_cells, data['width'])
            n = np.full(num_cells, data['n'])
            
            # Use initial condition from Python object if set
            edge_obj = self.edges[eid]
            h0 = edge_obj.model.h
            if len(h0) != num_cells: h0 = np.zeros(num_cells)
            
            q0 = np.full(num_cells, 0.0)
            
            self.add_reach_cpp(eid, dist, z, w, n, h0, q0)
            
        # 2. Add Junctions
        for nid, conns in self.adj_list.items():
            # Filter connections only for edges that exist
            valid_conns = [c for c in conns if c[0] in self.edges_data]
            # Even if no connections (isolated node), we might want to add it if C++ supports it? 
            # No, C++ junction needs connections.
            if not valid_conns:
                continue
                
            r_ids = [c[0] for c in valid_conns]
            is_start = [1 if c[1] else 0 for c in valid_conns]
            
            self.add_junction_cpp(nid, r_ids, is_start)
            
        self.built = True

    def add_reach_cpp(self, reach_id, dist, z, w, n, h0, q0):
        dist = np.ascontiguousarray(dist, dtype=np.float64)
        z = np.ascontiguousarray(z, dtype=np.float64)
        w = np.ascontiguousarray(w, dtype=np.float64)
        n = np.ascontiguousarray(n, dtype=np.float64)
        h0 = np.ascontiguousarray(h0, dtype=np.float64)
        q0 = np.ascontiguousarray(q0, dtype=np.float64)
        
        num_nodes = len(dist)
        lib.Network_AddReach(self.obj, reach_id, num_nodes, dist, z, w, n, h0, q0)
        self.reaches[reach_id] = {'num_nodes': num_nodes}

    def add_junction_cpp(self, junc_id, reach_ids, is_start):
        r_ids = np.ascontiguousarray(reach_ids, dtype=np.int32)
        start = np.ascontiguousarray(is_start, dtype=np.int32)
        n = len(reach_ids)
        lib.Network_AddJunction(self.obj, junc_id, r_ids, start, n)
        self.junctions[junc_id] = {'reaches': reach_ids}

    def run_step(self, dt, sim_time=None):
        if not self.built: self.build()
        
        # Apply BCs
        for nid, node in self.nodes.items():
            if node.bc_type == 'inflow':
                lib.Network_SetJunctionSource(self.obj, nid, node.bc_value)
            elif node.bc_type == 'outflow':
                 lib.Network_SetJunctionSource(self.obj, nid, -node.bc_value)
            else:
                 lib.Network_SetJunctionSource(self.obj, nid, 0.0)

        lib.Network_Step(self.obj, dt)
        
        # Sync results back to Python objects for visualization
        for eid, edge in self.edges.items():
             h, q = self.get_reach_results(eid)
             if h is not None:
                 edge.model.h[:] = h
                 edge.model.u[:] = q # Store Q directly in U for now, or convert? Server expects U?
                 # Visualization uses h.
                 # If server uses u for vector, q is not u. u = q/A.
                 # Let's approx u
                 with np.errstate(divide='ignore', invalid='ignore'):
                    edge.model.u[:] = np.where(edge.model.h > 1e-3, q / (edge.model.w * edge.model.h), 0.0)

    def get_reach_results(self, reach_id):
        if reach_id not in self.reaches:
            return None, None
        n = self.reaches[reach_id]['num_nodes']
        h_out = np.zeros(n, dtype=np.float64)
        q_out = np.zeros(n, dtype=np.float64)
        lib.Network_GetResults(self.obj, reach_id, h_out, q_out)
        return h_out, q_out

    def __del__(self):
        if hasattr(self, 'obj') and self.obj:
            lib.Network_Delete(self.obj)
