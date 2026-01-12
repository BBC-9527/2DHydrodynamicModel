import ctypes
import numpy as np
import os

class SWEModel:
    def __init__(self, dll_path):
        self.lib = ctypes.CDLL(dll_path)
        
        # init_model(int rows, int cols, double dx, double dy, double* elevation, double* roughness)
        self.lib.init_model.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, 
                                        np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
                                        np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')]
        self.lib.init_model.restype = None

        # set_water_level(double* water_level)
        self.lib.set_water_level.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')]
        self.lib.set_water_level.restype = None

        # set_boundary_mask(int* bc_types, double* bc_values)
        self.lib.set_boundary_mask.argtypes = [np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
                                               np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')]
        self.lib.set_boundary_mask.restype = None

        # set_source(int index, double q)
        self.lib.set_source.argtypes = [ctypes.c_int, ctypes.c_double]
        self.lib.set_source.restype = None

        # set_source_momentum(int index, double mx, double my)
        try:
            self.lib.set_source_momentum.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double]
            self.lib.set_source_momentum.restype = None
        except AttributeError:
            print("[WARN] set_source_momentum not found in DLL. Update DLL for momentum coupling.")

        # reset_sources()
        try:
            self.lib.reset_sources.argtypes = []
            self.lib.reset_sources.restype = None
        except AttributeError:
            pass

        # run_step(double dt)
        self.lib.run_step.argtypes = [ctypes.c_double]
        self.lib.run_step.restype = None

        # get_results(double* out_h, double* out_u, double* out_v)
        self.lib.get_results.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
                                         np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
                                         np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')]
        self.lib.get_results.restype = None

        self.rows = 0
        self.cols = 0

    def init(self, elevation, roughness, dx, dy):
        self.rows, self.cols = elevation.shape
        # Flatten arrays for C++
        self.elevation_flat = np.ascontiguousarray(elevation.flatten(), dtype=np.float64)
        self.roughness_flat = np.ascontiguousarray(roughness.flatten(), dtype=np.float64)
        
        self.lib.init_model(self.rows, self.cols, dx, dy, self.elevation_flat, self.roughness_flat)

    def set_water_surface(self, ws_grid):
        ws_flat = np.ascontiguousarray(ws_grid.flatten(), dtype=np.float64)
        self.lib.set_water_level(ws_flat)

    def set_boundary_conditions(self, bc_type_grid, bc_val_grid):
        bc_type_flat = np.ascontiguousarray(bc_type_grid.flatten(), dtype=np.int32)
        bc_val_flat = np.ascontiguousarray(bc_val_grid.flatten(), dtype=np.float64)
        self.lib.set_boundary_mask(bc_type_flat, bc_val_flat)

    def set_source(self, r, c, q_val):
        """Set source term for cell (r,c) in m^3/s"""
        idx = r * self.cols + c
        self.lib.set_source(idx, q_val)

    def set_source_momentum(self, r, c, mx, my):
        """Set momentum source term for cell (r,c) in m^4/s^2 (Force/rho)"""
        if not hasattr(self.lib, 'set_source_momentum'):
            return
        idx = r * self.cols + c
        self.lib.set_source_momentum(idx, mx, my)

    def reset_sources(self):
        """Reset all source terms to 0"""
        if hasattr(self.lib, 'reset_sources'):
            self.lib.reset_sources()

    def step(self, dt):
        self.lib.run_step(dt)

    def get_results(self):
        h = np.zeros(self.rows * self.cols, dtype=np.float64)
        u = np.zeros(self.rows * self.cols, dtype=np.float64)
        v = np.zeros(self.rows * self.cols, dtype=np.float64)
        
        self.lib.get_results(h, u, v)
        
        return (h.reshape((self.rows, self.cols)), 
                u.reshape((self.rows, self.cols)), 
                v.reshape((self.rows, self.cols)))
