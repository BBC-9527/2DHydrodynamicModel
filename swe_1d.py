import ctypes
import numpy as np
import os
import sys

# Global Constants for Consistency
DRY_THRESHOLD = 0.01  # Depth below which a cell is considered dry (User suggestion: 0.01m)
WET_THRESHOLD = 0.05  # Depth above which a cell is considered fully wet/flowing (User suggestion: 0.05m)
FLOW_THRESHOLD = 1e-6 # Minimal flow velocity/flux threshold

class SWEModel1D:
    def __init__(self, num_cells, dx, dll_path=None):
        if dll_path is None:
            # Default to swe_1d_core.dll in the current directory or src
            base_dir = os.path.dirname(os.path.abspath(__file__))
            dll_path = os.path.join(base_dir, "swe_1d_core.dll")
        
        if not os.path.exists(dll_path):
             # Try looking in src/ as well if not in root
             src_path = os.path.join(base_dir, "src", "swe_1d_core.dll")
             if os.path.exists(src_path):
                 dll_path = src_path
             else:
                 raise FileNotFoundError(f"DLL not found at {dll_path} or {src_path}")

        self.lib = ctypes.CDLL(dll_path)
        
        # New: Create Model Object (Object-Oriented API)
        try:
            self.lib.create_model.restype = ctypes.c_void_p
            self.obj = self.lib.create_model()
        except AttributeError:
             # Fallback for old DLL if user didn't recompile yet (safety check)
             print("Warning: create_model not found in DLL. Using old global API?")
             self.obj = None

        # init_model(void* ptr, int num_cells, double dx, double* elevation, double* roughness, double* width, double* initial_h)
        self.lib.init_model.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int, 
            ctypes.c_double, 
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        ]
        self.lib.init_model.restype = None

        # set_boundary_conditions(void* ptr, int left_type, double left_val, int right_type, double right_val)
        self.lib.set_boundary_conditions.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int, ctypes.c_double,
            ctypes.c_int, ctypes.c_double
        ]
        self.lib.set_boundary_conditions.restype = None

        # set_source(void* ptr, int index, double q_val)
        self.lib.set_source.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
        self.lib.set_source.restype = None

        # run_step(void* ptr, double dt)
        self.lib.run_step.argtypes = [ctypes.c_void_p, ctypes.c_double]
        self.lib.run_step.restype = None

        # get_results(void* ptr, double* out_h, double* out_u)
        self.lib.get_results.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        ]
        self.lib.get_results.restype = None
        
        # Destructor
        if hasattr(self.lib, 'delete_model'):
            self.lib.delete_model.argtypes = [ctypes.c_void_p]
            self.lib.delete_model.restype = None

        self.num_cells = num_cells
        self.dx = dx
        
        # Keep references to arrays to prevent garbage collection
        self.z = None
        self.n = None
        self.w = None
        self.h = None

    def __del__(self):
        if hasattr(self, 'obj') and self.obj:
            if hasattr(self.lib, 'delete_model'):
                self.lib.delete_model(self.obj)
            self.obj = None

    def init(self, z_arr, n_arr, w_arr, h_arr):
        """Initialize the 1D model from numpy arrays"""
        if len(z_arr) != self.num_cells:
            raise ValueError(f"Array length mismatch. Expected {self.num_cells}, got {len(z_arr)}")
            
        # Ensure contiguous C arrays
        self.z = np.ascontiguousarray(z_arr, dtype=np.float64)
        self.n = np.ascontiguousarray(n_arr, dtype=np.float64)
        self.w = np.ascontiguousarray(w_arr, dtype=np.float64)
        self.h = np.ascontiguousarray(h_arr, dtype=np.float64)
        
        self.lib.init_model(self.obj, self.num_cells, self.dx, self.z, self.n, self.w, self.h)
        
    def set_boundary(self, side, b_type, val=0.0):
        """
        Set boundary conditions.
        side: 'left' or 'right'
        b_type: 0=Wall, 1=Fixed Depth, 2=Fixed Discharge, 3=Open
        """
        # We need to store current values because C++ API sets both at once
        if not hasattr(self, '_bc_left'): self._bc_left = (0, 0.0)
        if not hasattr(self, '_bc_right'): self._bc_right = (3, 0.0)
        
        if side == 'left':
            self._bc_left = (b_type, val)
        elif side == 'right':
            self._bc_right = (b_type, val)
            
        self.lib.set_boundary_conditions(
            self.obj,
            self._bc_left[0], self._bc_left[1],
            self._bc_right[0], self._bc_right[1]
        )

    def set_source(self, index, Q_val):
        """Set lateral inflow source at cell index (m^3/s)"""
        self.lib.set_source(self.obj, index, Q_val)

    def run_step(self, dt):
        self.lib.run_step(self.obj, dt)
        
    def step(self, dt):
        self.run_step(dt)

    def get_results(self):
        h = np.zeros(self.num_cells, dtype=np.float64)
        u = np.zeros(self.num_cells, dtype=np.float64)
        self.lib.get_results(self.obj, h, u)
        return h, u
        
    def get_bed_elevation(self, idx):
        if self.z is not None:
            if idx >= 0 and idx < len(self.z):
                return self.z[idx]
        return 0.0
