import sys
import os
import threading
import zipfile
import tempfile
import json
import shutil
try:
    import shapefile
except ImportError:
    shapefile = None


# Fix for PROJ: proj_create_from_database: Cannot find proj.db
# Always prefer bundled rasterio PROJ data first, then pyproj, then search.
try:
    # 1. Try rasterio bundled proj_data
    import importlib.util
    spec = importlib.util.find_spec("rasterio")
    found_bundled = False
    if spec and spec.origin:
        rasterio_dir = os.path.dirname(spec.origin)
        proj_data = os.path.join(rasterio_dir, 'proj_data')
        if os.path.exists(os.path.join(proj_data, 'proj.db')):
            os.environ['PROJ_LIB'] = proj_data
            print(f"DEBUG: Found rasterio bundled proj_data. Force set PROJ_LIB to {proj_data}")
            found_bundled = True

    if not found_bundled:
        # 2. Try pyproj
        import pyproj
        path = pyproj.datadir.get_data_dir()
        os.environ['PROJ_LIB'] = path
        print(f"DEBUG: pyproj imported. Force set PROJ_LIB to {path}")

except ImportError:
    pass # Fallback to search

if 'PROJ_LIB' not in os.environ or not os.path.exists(os.path.join(os.environ['PROJ_LIB'], 'proj.db')):
    # 3. Search for proj.db if still missing or invalid
        # Search for proj.db
        print("Searching for proj.db...")
        found = False
        for root, dirs, files in os.walk(sys.prefix):
            if 'proj.db' in files:
                os.environ['PROJ_LIB'] = root
                print(f"Found proj.db at {root}")
                found = True
                break
        if not found:
             import site
             try:
                 for site_pkg in site.getsitepackages():
                     for root, dirs, files in os.walk(site_pkg):
                         if 'proj.db' in files:
                             os.environ['PROJ_LIB'] = root
                             print(f"Found proj.db at {root}")
                             found = True
                             break
                     if found: break
             except: pass
print(f"DEBUG: Final PROJ_LIB = {os.environ.get('PROJ_LIB')}")
import io
import asyncio
import numpy as np
import rasterio
from rasterio import features, warp
from rasterio.crs import CRS
from typing import List, Optional
from pyproj import Transformer
from PIL import Image
# Dynamic Patch for Middleware Issue
# Detects if Middleware yields 3 items but FastAPI expects 2 (ValueError)
try:
    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from fastapi import FastAPI as CheckFastAPI
    
    # Check what Middleware yields
    m = Middleware(BaseHTTPMiddleware, dispatch=lambda x,y: x)
    if len(list(m)) == 3:
        # Middleware yields 3 items. Check if FastAPI crashes.
        app_check = CheckFastAPI()
        app_check.add_middleware(BaseHTTPMiddleware, dispatch=lambda x,y: x)
        try:
            app_check.build_middleware_stack()
        except ValueError as e:
            if "too many values" in str(e):
                print("Patching Starlette Middleware to yield 2 items for FastAPI compatibility")
                def patched_middleware_iter(self):
                    yield self.cls
                    yield self.kwargs
                Middleware.__iter__ = patched_middleware_iter
        except Exception:
            pass
except Exception as e:
    print(f"Warning: Middleware patch check failed: {e}")

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect, Request
from contextlib import asynccontextmanager
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import matplotlib.pyplot as plt
from matplotlib import cm
import asyncio
import builtins

# Global Loop Reference for Thread-Safe Logging
MAIN_LOOP = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MAIN_LOOP
    try:
        MAIN_LOOP = asyncio.get_running_loop()
    except RuntimeError:
        pass 
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- WebSocket & Logging Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()



# Override print to broadcast logs
original_print = builtins.print

def custom_print(*args, **kwargs):
    # Create the message string exactly as print would
    sep = kwargs.get('sep', ' ')
    end = kwargs.get('end', '\n')
    msg = sep.join(map(str, args)) + end
    
    # Broadcast to WebSockets
    try:
        # Check if we are in the event loop
        loop = asyncio.get_running_loop()
        loop.create_task(manager.broadcast(msg))
    except RuntimeError:
        # We are likely in a separate thread (e.g. saving file)
        # Use the captured MAIN_LOOP
        if MAIN_LOOP and MAIN_LOOP.is_running():
            asyncio.run_coroutine_threadsafe(manager.broadcast(msg), MAIN_LOOP)
        
    # Call original print
    original_print(*args, **kwargs)

builtins.print = custom_print
# -----------------------------------

# Add gui to path to import swe_wrapper
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "gui")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # For swe_gpu

from swe_wrapper import SWEModel
try:
    from swe_1d import SWEModel1D, DRY_THRESHOLD, WET_THRESHOLD
except ImportError:
    from swe_1d import SWEModel1D
    DRY_THRESHOLD = 0.001
    WET_THRESHOLD = 0.01
from swe_1d_network import Network1D
try:
    from swe_1d_network_cpp import Network1D_CPP
except ImportError:
    Network1D_CPP = None
    print("Warning: Network1D_CPP not available (DLL missing?)")
from coupler import Coupler

try:
    from swe_gpu import SWEModelGPU
    GPU_AVAILABLE = True
    print("GPU Acceleration Available (Taichi)")
except ImportError as e:
    print(f"GPU Acceleration NOT Available: {e}")
    GPU_AVAILABLE = False
    SWEModelGPU = None

from PIL import Image

import matplotlib

# Precompute colormaps for fast rendering
# 'terrain' for elevation
try:
    terrain_cmap = matplotlib.colormaps['terrain']
except:
    terrain_cmap = plt.get_cmap('terrain')
terrain_lut = (terrain_cmap(np.arange(256)) * 255).astype(np.uint8)

# 'Blues' for water depth (Better visual for water, distinct from terrain)
try:
    water_cmap = matplotlib.colormaps['Blues']
except:
    water_cmap = plt.get_cmap('Blues')
water_lut = (water_cmap(np.arange(256)) * 255).astype(np.uint8)

# Set alpha:
# For Blues, we want to see the color clearly.
water_lut[:, 3] = 220 # Slightly more opaque

# 0 depth (index 0) should be transparent
water_lut[0, :] = 0 
# And maybe the first few indices too to avoid noise
water_lut[0:2, :] = 0

@app.websocket("/ws/logs")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, maybe receive commands in future
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Global State (Single Session for Demo)
class SimulationState:
    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        self.model_1d = None # Legacy Single 1D Model
        self.network = None # New Network 1D
        self.coupler = None  # Coupler
        self.elevation = None
        self.original_elevation = None # Backup for reset
        self.nodata_mask = None # Boolean mask: True = NoData/Transparent
        self.h = None
        self.bc_type_grid = None
        self.bc_val_grid = None
        self.dx = 1.0
        self.dy = 1.0
        self.transform = None # Affine transform (3857 -> pixel)
        self.crs = None # CRS of the loaded DEM (defaults to 3857 if not loaded)
        self.bounds_4326 = None # [[min_lat, min_lon], [max_lat, max_lon]]
        self.inactive_1d_nodes = [] # List of (model, idx) for 1D cells outside AOI
        self.disable_1d = False # Explicit flag to disable 1D logic
        
        # Dynamic Boundaries List
        # Stores dicts: {id, type, mode, value, series, cells, count}
        self.boundaries = []
        
        # Simulation Loop Control
        self.is_running = False
        self.current_time = 0.0
        self.total_time = 3600.0 # Default 1 hour
        self.dt_save = 10.0 # Save interval in seconds
        self.save_index = 0
        self.geometry_1d = None # List of [lat, lon] for 1D nodes

sim_state = SimulationState()

# Mount static files
app.mount("/static", StaticFiles(directory="web/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("web/static/index.html", "r", encoding='utf-8') as f:
        return f.read()

@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools_dummy():
    """Silence 404 logs for Chrome DevTools probing"""
    return {}

@app.post("/init_test")
async def init_test():
    """Initialize with test terrain in China (Poyang Lake area)"""
    rows, cols = 100, 200
    
    # 1. Define real-world location (approx Poyang Lake)
    # Center: ~29.0N, 116.0E
    lon_origin = 116.0
    lat_origin = 29.0
    
    # Convert to EPSG:3857
    x_min, y_min_3857 = warp.transform(CRS.from_epsg(4326), CRS.from_epsg(3857), [lon_origin], [lat_origin])
    x_min = x_min[0]
    y_min_3857 = y_min_3857[0]
    
    # 2. Set resolution (meters)
    sim_state.dx = 50.0 # 50m resolution
    sim_state.dy = 50.0
    
    # Reset Boundaries
    sim_state.boundaries = []
    
    # Reset Simulation Time
    sim_state.current_time = 0.0
    sim_state.save_index = 0
    sim_state.is_running = False

    # 3. Create Transform
    # Rasterio reads top-left origin by default, but check transform
    height_meters = rows * sim_state.dy
    width_meters = cols * sim_state.dx
    y_max = y_min_3857 + height_meters
    
    from rasterio.transform import from_origin
    sim_state.transform = from_origin(x_min, y_max, sim_state.dx, sim_state.dy)
    sim_state.crs = CRS.from_epsg(3857)
    
    # 4. Generate Elevation (using grid indices to preserve shape)
    X_idx, Y_idx = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Channel with slope
    # Drop: 0.1m per cell * 200 cells = 20m drop
    elevation = 30.0 - 0.1 * X_idx 
    # Hump: (Y_idx - 50)^2. Max (50)^2 = 2500. * 0.005 = 12.5m
    elevation += 0.005 * (Y_idx - 50)**2
    
    sim_state.elevation = elevation.astype(np.float64)
    sim_state.original_elevation = sim_state.elevation.copy()
    sim_state.nodata_mask = np.zeros(sim_state.elevation.shape, dtype=bool) # No transparency for test terrain
    sim_state.bc_type_grid = np.zeros((rows, cols), dtype=np.int32)
    sim_state.bc_val_grid = np.zeros((rows, cols), dtype=np.float64)
    
    # 5. Calculate Lat/Lon bounds for Leaflet
    left, bottom, right, top = x_min, y_min_3857, x_min + width_meters, y_max
    min_lon, min_lat, max_lon, max_lat = warp.transform_bounds(
        CRS.from_epsg(3857), CRS.from_epsg(4326), left, bottom, right, top
    )
    sim_state.bounds_4326 = [[min_lat, min_lon], [max_lat, max_lon]]
    
    # Initialize Model
    if GPU_AVAILABLE:
        print("Initializing GPU Model (Taichi)...")
        sim_state.model = SWEModelGPU()
    else:
        print("Initializing CPU Model (DLL)...")
        dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "swe_core.dll"))
        if not os.path.exists(dll_path):
             raise HTTPException(status_code=500, detail="DLL not found")
             
        sim_state.model = SWEModel(dll_path)
    
    roughness = np.full((rows, cols), 0.03, dtype=np.float64)
    sim_state.roughness = roughness
    sim_state.model.init(sim_state.elevation, roughness, sim_state.dx, sim_state.dy)
    sim_state.model.set_water_surface(sim_state.elevation) # Dry start
    
    # Reset simulation time
    sim_state.current_time = 0.0
    sim_state.save_index = 0
    sim_state.is_running = False
    sim_state.model_1d = None
    sim_state.network = None
    sim_state.geometry_1d = None
    sim_state.coupler = None
    sim_state.disable_1d = False # Default to allowing 1D if added later
    
    return {
        "status": "initialized", 
        "rows": rows, 
        "cols": cols, 
        "bounds": sim_state.bounds_4326
    }

@app.post("/init_2d_simulation")
async def init_2d_simulation():
    """Set 2D Only Mode (1D Disabled)"""
    # 1. Reset Simulation State Completely
    sim_state.disable_1d = True
    sim_state.model_1d = None
    sim_state.network = None
    sim_state.coupler = None
    sim_state.elevation = None
    sim_state.original_elevation = None
    sim_state.model = None
    sim_state.boundaries = []
    sim_state.current_time = 0.0
    sim_state.save_index = 0
    sim_state.is_running = False
    
    print("Mode Set: 2D Only (1D Disabled). State cleared.")
    return {
        "status": "mode_set", 
        "mode": "2d_only",
        "message": "Mode set to 2D Only. Please load DEM and Boundaries."
    }

@app.post("/upload_dem")
async def upload_dem(file: UploadFile = File(...)):
    """Upload and load a GeoTIFF DEM. Must be EPSG:3857."""
    content = await file.read()
    
    with rasterio.MemoryFile(content) as memfile:
        with memfile.open() as dataset:
            # Check CRS
            if dataset.crs != CRS.from_epsg(3857):
                print(f"Warning: CRS is {dataset.crs}, expected EPSG:3857. Assuming valid or user handled.")
                # Ideally we should reproject here if not 3857, but user said "input data unified requirement EPSG:3857"
                # So we assume it IS 3857.
            
            elevation = dataset.read(1).astype(np.float64)
            # Handle NoData if present
            mask = None
            if dataset.nodata is not None:
                mask = (elevation == dataset.nodata)
                elevation[mask] = np.nan

            # Check for NaNs already in data
            if mask is None:
                mask = np.isnan(elevation)
            else:
                mask = mask | np.isnan(elevation)
            
            sim_state.nodata_mask = mask
            
            # Replace NaNs with min value or 0 for physics stability
            elevation = np.nan_to_num(elevation, nan=np.nanmin(elevation))
            
            sim_state.elevation = elevation
            sim_state.original_elevation = elevation.copy()
            rows, cols = elevation.shape
            
            sim_state.transform = dataset.transform
            sim_state.crs = dataset.crs
            dx = sim_state.transform[0]
            dy = -sim_state.transform[4] 
            
            if dx <= 0 or dy <= 0:
                dx, dy = 1.0, 1.0
            
            sim_state.dx = dx
            sim_state.dy = dy
            
            # Calculate Lat/Lon bounds for Leaflet
            # Transform bounds from source CRS (3857) to EPSG:4326
            left, bottom, right, top = dataset.bounds
            
            # warp.transform_bounds(src_crs, dst_crs, left, bottom, right, top)
            # 3857 -> 4326
            min_lon, min_lat, max_lon, max_lat = warp.transform_bounds(
                dataset.crs, CRS.from_epsg(4326), left, bottom, right, top
            )
            
            # Leaflet expects [[min_lat, min_lon], [max_lat, max_lon]]
            sim_state.bounds_4326 = [[min_lat, min_lon], [max_lat, max_lon]]
            
    # Reset Boundaries
    sim_state.boundaries = []

    rows, cols = sim_state.elevation.shape
    sim_state.bc_type_grid = np.zeros((rows, cols), dtype=np.int32)
    sim_state.bc_val_grid = np.zeros((rows, cols), dtype=np.float64)
    
    # Init Model
    if GPU_AVAILABLE:
        print("Initializing GPU Model (Taichi)...")
        sim_state.model = SWEModelGPU()
    else:
        print("Initializing CPU Model (DLL)...")
        dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "swe_core.dll"))
        sim_state.model = SWEModel(dll_path)
    
    roughness = np.full((rows, cols), 0.03, dtype=np.float64)
    sim_state.roughness = roughness
    sim_state.model.init(sim_state.elevation, roughness, sim_state.dx, sim_state.dy)
    sim_state.model.set_water_surface(sim_state.elevation)
    
    # Reset simulation time
    sim_state.current_time = 0.0
    sim_state.save_index = 0
    sim_state.is_running = False
    sim_state.model_1d = None
    sim_state.geometry_1d = None
    sim_state.coupler = None

    return {
        "status": "initialized", 
        "rows": rows, 
        "cols": cols, 
        "bounds": sim_state.bounds_4326
    }

@app.post("/init_coupling_test")
async def init_coupling_test():
    """Set Coupled 1D-2D Mode"""
    # 1. Enable 1D logic (allow creation)
    sim_state.disable_1d = False
    
    # 2. Reset Everything
    sim_state.model_1d = None
    sim_state.network = None
    sim_state.coupler = None
    sim_state.elevation = None
    sim_state.original_elevation = None
    sim_state.model = None
    sim_state.boundaries = []
    sim_state.current_time = 0.0
    sim_state.save_index = 0
    sim_state.is_running = False
    
    print("Mode Set: Coupled 1D-2D. State cleared.")
    return {
        "status": "mode_set", 
        "mode": "coupled",
        "message": "Mode set to Coupled 1D-2D. Please load DEM and Boundaries."
    }

class Channel1DRequest(BaseModel):
    geometry: List[List[float]] # List of [lat, lon]
    width: float
    n: float
    z_start: float
    z_end: float
    h_init: float
    dx: float

@app.post("/create_1d_channel")
async def create_1d_channel(req: Channel1DRequest):
    """
    Create a 1D model from a polyline geometry and parameters.
    Automatically detects coupling with 2D grid.
    Wraps the single channel into a Network1D for consistent handling.
    """
    if sim_state.model is None:
        raise HTTPException(status_code=400, detail="2D Model not initialized")
    
    print(f"Creating 1D Channel with {len(req.geometry)} points, dx={req.dx}")
    
    # Initialize Network
    sim_state.network = Network1D()
    sim_state.coupler = None # Reset coupler
    
    # 1. Transform Lat/Lon to EPSG:3857 (Meters)
    coords_ll = req.geometry # [lat, lon]
    
    # Create LineString-like structure and resample
    total_length = 0.0
    segments = [] # (length, x1, y1, x2, y2)
    
    points_proj = []
    src_crs = CRS.from_epsg(4326)
    dst_crs = CRS.from_epsg(3857)
    
    for lat, lon in coords_ll:
        x, y = warp.transform(src_crs, dst_crs, [lon], [lat])
        points_proj.append((x[0], y[0]))
        
    # Calculate segment lengths
    for i in range(len(points_proj) - 1):
        p1 = points_proj[i]
        p2 = points_proj[i+1]
        dist = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        segments.append({'len': dist, 'p1': p1, 'p2': p2})
        total_length += dist
        
    print(f"Total Channel Length: {total_length:.2f} m")
    
    # 2. Determine number of cells
    n_cells = int(np.ceil(total_length / req.dx))
    if n_cells < 2: n_cells = 2 # Minimum 2 cells
    
    real_dx = total_length / n_cells
    print(f"Discretizing into {n_cells} cells with dx={real_dx:.2f} m")
    
    # 3. Generate Cell Centers
    centers_x = []
    centers_y = []
    
    seg_idx = 0
    seg_accum_dist = 0.0
    
    for i in range(n_cells):
        target_dist = (i * real_dx) + (real_dx / 2.0)
        
        # Find which segment this falls into
        while seg_idx < len(segments):
            seg = segments[seg_idx]
            if target_dist <= (seg_accum_dist + seg['len']):
                local_dist = target_dist - seg_accum_dist
                t = local_dist / seg['len']
                x = seg['p1'][0] + t * (seg['p2'][0] - seg['p1'][0])
                y = seg['p1'][1] + t * (seg['p2'][1] - seg['p1'][1])
                centers_x.append(x)
                centers_y.append(y)
                break
            else:
                seg_accum_dist += seg['len']
                seg_idx += 1
                
    # 4. Add Nodes and Edge to Network
    # Node 0: Start
    start_pt = points_proj[0]
    sim_state.network.add_node(0, start_pt[0], start_pt[1], req.z_start)
    sim_state.network.set_node_bc(0, 'inflow', 0.0) # Default
    
    # Node 1: End
    end_pt = points_proj[-1]
    sim_state.network.add_node(1, end_pt[0], end_pt[1], req.z_end)
    sim_state.network.set_node_bc(1, 'outflow', 0.0) # Default
    
    # Edge 0
    # geometry expects [lat, lon]
    sim_state.network.add_edge(
        id=0,
        start_node_id=0,
        end_node_id=1,
        width=req.width,
        n=req.n,
        z_start=req.z_start,
        z_end=req.z_end,
        dx=real_dx,
        geometry=req.geometry,
        length=total_length
    )
    
    # Get the model created by add_edge
    edge = sim_state.network.edges[0]
    sim_state.model_1d = edge.model # For legacy compatibility if any
    
    # 5. Advanced Bed Elevation Interpolation (Draping)
    z_linear = np.linspace(req.z_start, req.z_end, n_cells)
    z_dem_sampled = np.zeros(n_cells)
    
    if sim_state.elevation is not None and sim_state.transform:
        rows_sample, cols_sample = rasterio.transform.rowcol(sim_state.transform, centers_x, centers_y)
        for i in range(n_cells):
            r, c = rows_sample[i], cols_sample[i]
            if 0 <= r < sim_state.model.rows and 0 <= c < sim_state.model.cols:
                z_dem_sampled[i] = float(sim_state.elevation[r, c])
            else:
                z_dem_sampled[i] = z_linear[i]
    else:
        z_dem_sampled = z_linear.copy()
            
    # Draping logic: z_river = min(z_linear, z_dem - 0.5)
    z_1d = np.minimum(z_linear, z_dem_sampled - 0.5) 
    
    # Re-init model with custom Z
    h_1d = np.ones(n_cells) * req.h_init
    # Re-apply init to edge model
    edge.model.init(z_1d, edge.model.n, edge.model.w, h_1d)
    
    # 6. Store Geometry for Visualization (Legacy)
    coords_1d_viz = []
    lon_out, lat_out = warp.transform(dst_crs, src_crs, centers_x, centers_y)
    for i in range(n_cells):
        coords_1d_viz.append([lat_out[i], lon_out[i]])
    sim_state.geometry_1d = coords_1d_viz
    
    # 7. Setup Coupler via Network function
    try:
        initialize_network_coupling()
    except Exception as e:
        print(f"[WARNING] Failed to auto-initialize coupling: {e}")
            
    return {
        "status": "success",
        "cells": n_cells,
        "coupled_nodes": len(sim_state.coupler.nodes) if sim_state.coupler else 0
    }

class NetworkNodeItem(BaseModel):
    id: int
    lat: float
    lon: float
    z: float
    is_inflow: bool = False # New field
    inflow_q: float = 0.0 # New field
    is_outflow: bool = False # New field

class NetworkEdgeItem(BaseModel):
    id: int
    start_id: int
    end_id: int
    width: float
    width_end: float = None
    n: float
    z_start: float
    z_end: float
    dx: float = 10.0
    geometry: List[List[float]] = None # [[lat, lon], ...]

class NetworkRequest(BaseModel):
    nodes: List[NetworkNodeItem]
    edges: List[NetworkEdgeItem]
    initial_h: float = 0.0
    use_cpp: bool = False

@app.post("/create_1d_network")
async def create_1d_network(req: NetworkRequest):
    if req.use_cpp and Network1D_CPP:
        sim_state.network = Network1D_CPP()
        print("[INFO] Using C++ 1D Network Engine (Implicit Preissmann)")
    else:
        sim_state.network = Network1D()
        print("[INFO] Using Python 1D Network Engine (Explicit HLL)")
        
    sim_state.coupler = None # Reset coupler when network changes
    
    # 1. Add Nodes
    src_crs = CRS.from_epsg(4326)
    dst_crs = sim_state.crs if sim_state.crs else CRS.from_epsg(3857)
    
    pending_bcs = [] # Store BC requests to apply AFTER edges are linked

    # Batch transform optimization could be done, but simple loop is fine for <1000 nodes
    for n in req.nodes:
        x, y = warp.transform(src_crs, dst_crs, [n.lon], [n.lat])
        
        # Auto-sample Z from DEM if provided Z is 0.0 (likely default from frontend)
        # This prevents massive head differences if user draws on high terrain (e.g. 300m) but sends Z=0.
        z_val = n.z
        if z_val == 0.0 and sim_state.elevation is not None:
             # Transform x,y to grid rows, cols
             r, c = rasterio.transform.rowcol(sim_state.transform, x[0], y[0])
             if 0 <= r < sim_state.model.rows and 0 <= c < sim_state.model.cols:
                 z_sampled = float(sim_state.elevation[r, c])
                 # Use sampled Z, maybe add a small offset?
                 z_val = z_sampled
                 # print(f"Node {n.id}: Auto-sampled Z={z_val:.2f}m (was 0.0)")
        
        sim_state.network.add_node(n.id, x[0], y[0], z_val)
        if n.is_inflow:
            pending_bcs.append((n.id, 'inflow', n.inflow_q))
        if n.is_outflow:
            pending_bcs.append((n.id, 'outflow', 0.0))
        
    # 2. Add Edges
    for e in req.edges:
        # Check if Z needs auto-sampling (if 0.0)
        # However, edges use Node Z usually? 
        # NetworkEdge uses z_start/z_end for linear interp.
        # If nodes were updated, we should ideally use Node Z.
        # But Network1D.add_edge takes z_start/z_end explicitly.
        # We should override them if they are 0 and nodes have valid Z.
        
        z_s = e.z_start
        z_e = e.z_end
        
        if z_s == 0.0 and e.start_id in sim_state.network.nodes:
            z_s = sim_state.network.nodes[e.start_id].z
            
        if z_e == 0.0 and e.end_id in sim_state.network.nodes:
            z_e = sim_state.network.nodes[e.end_id].z

        # Calculate Polyline Length if Geometry is available
        calc_length = None
        if e.geometry and len(e.geometry) >= 2:
             try:
                 # Assume geometry is [lat, lon]
                 g_lats = [p[0] for p in e.geometry]
                 g_lons = [p[1] for p in e.geometry]
                 
                 # Project to Meters (3857) - Re-use src_crs/dst_crs from Nodes section
                 xs_poly, ys_poly = warp.transform(src_crs, dst_crs, g_lons, g_lats)
                 
                 total_len = 0.0
                 for i in range(len(xs_poly)-1):
                     total_len += np.hypot(xs_poly[i+1]-xs_poly[i], ys_poly[i+1]-ys_poly[i])
                 
                 if total_len > 0:
                     calc_length = total_len
             except Exception as ex:
                 print(f"[WARNING] Failed to calculate length for Edge {e.id}: {ex}")
            
        sim_state.network.add_edge(e.id, e.start_id, e.end_id, e.width, e.n, z_s, z_e, e.dx, width_end=e.width_end, geometry=e.geometry, length=calc_length)
        
    # 2.5 Apply Boundary Conditions with Validation
    # Prevent assigning Inflow/Outflow to Internal Nodes (Junctions)
    for node_id, bc_type, val in pending_bcs:
        if node_id not in sim_state.network.nodes:
            continue
            
        node = sim_state.network.nodes[node_id]
        degree = len(node.connections)
        
        # Check if internal node
        if degree > 1:
            print(f"[WARNING] Ignoring {bc_type} BC on INTERNAL Node {node_id} (Degree {degree}). Internal nodes cannot be boundaries.")
        else:
            sim_state.network.set_node_bc(node_id, bc_type, val)
            print(f"[INFO] Set 1D Node {node_id} BC: {bc_type} = {val}")

    # 3. Initialize Water Depth
    if req.initial_h > 0:
        sim_state.network.set_initial_condition(req.initial_h)
        print(f"Network initialized with uniform water depth h={req.initial_h}m")
    
    # [FIX] Initialize Coupling IMMEDIATELY for multi-segment networks
    # This ensures that when the user starts the simulation, the coupler is ready.
    if sim_state.model is not None:
         # Need to ensure this function is available/defined
         # It is defined later in this file, so we can call it.
         try:
             initialize_network_coupling()
         except Exception as e:
             print(f"[WARNING] Failed to auto-initialize coupling: {e}")

    # Optional: If user intended initial_h as Water Surface Elevation (WSE),
    # we should calculate h = max(0, WSE - z).
    # Currently frontend sends "initial_h" which label says "Initial Depth".
    # We stick to Depth interpretation unless specified otherwise.
        
    return {"status": "created", "nodes": len(req.nodes), "edges": len(req.edges)}

@app.post("/reset")
async def reset_simulation():
    """
    Reset the simulation state (time, water depth, etc.) without reloading geometry.
    """
    print("Resetting simulation...")
    sim_state.current_time = 0.0
    sim_state.save_index = 0
    sim_state.is_running = False
    sim_state.coupler = None # Force re-initialization of coupling

    sim_state.boundaries = []
    if sim_state.bc_type_grid is not None:
        # sim_state.bc_type_grid.fill(0) # Keep AOI Mask on Reset!
        pass
    if sim_state.bc_val_grid is not None:
        sim_state.bc_val_grid.fill(0.0)
    
    # Reset 2D Model Water Depth
    with sim_state.lock:
        if sim_state.model and sim_state.elevation is not None:
            # Create zero depth array
            zeros = np.zeros_like(sim_state.elevation)
            sim_state.h = zeros # Update state tracker
            
            # Hard Reset: Re-initialize model to clear all internal state (sources, momentum, etc.)
            # This is safer than just setting water level, especially if DLL doesn't support reset_sources()
            if hasattr(sim_state.model, 'init'):
                 sim_state.model.init(sim_state.elevation, sim_state.roughness, sim_state.dx, sim_state.dy)
            else:
                 sim_state.model.set_water_surface(zeros)
            
            # Reset Source Terms (Just in case init doesn't clear them, though it should)
            if hasattr(sim_state.model, 'reset_sources'):
                sim_state.model.reset_sources()

            sim_state.model.set_boundary_conditions(sim_state.bc_type_grid, sim_state.bc_val_grid)
                
            print("2D Model reset (re-initialized).")
        
    # Reset 1D Network Water Depth
    if sim_state.network:
        # Reset nodes
        for node in sim_state.network.nodes.values():
            node.h = 0.0
        # Reset edges
        for edge in sim_state.network.edges.values():
            # Reset edge model state
            if edge.model:
                edge.model.h[:] = 0.0
                # Re-initialize C++ model with zero depth
                edge.model.init(edge.model.z, edge.model.n, edge.model.w, edge.model.h)
        print("1D Network water depth reset to 0.")
        
    return {"status": "reset", "message": "Simulation reset successfully"}

@app.post("/set_1d_node_bc")
async def set_1d_node_bc(request: Request):
    """
    Set Boundary Condition for a specific 1D node.
    """
    try:
        data = await request.json()
        node_id = int(data.get('node_id'))
        bc_type = data.get('type') # 'inflow'
        value = float(data.get('value', 0.0))
        
        if not sim_state.network:
             return JSONResponse({'status': 'error', 'message': 'Network not initialized'}, status_code=400)

        if node_id not in sim_state.network.nodes:
             return JSONResponse({'status': 'error', 'message': f'Node {node_id} not found'}, status_code=404)
             
        node = sim_state.network.nodes[node_id]
        # [FIX] Block setting BC on internal nodes
        if len(node.connections) > 1 and bc_type in ['inflow', 'outflow']:
             msg = f"Cannot set {bc_type} BC on INTERNAL Node {node_id} (Degree {len(node.connections)}). Internal nodes must be continuous."
             print(f"[WARNING] {msg}")
             # Return error or success with warning? 
             # Let's return error to prevent frontend from showing it as set.
             return JSONResponse({'status': 'error', 'message': msg}, status_code=400)
             
        sim_state.network.set_node_bc(node_id, bc_type, value)
        print(f"[INFO] Set 1D Node {node_id} BC: {bc_type} = {value}")
        
        return JSONResponse({'status': 'ok', 'node_id': node_id, 'bc': bc_type, 'value': value})
    except Exception as e:
        print(f"[ERROR] Failed to set 1D node BC: {e}")
        return JSONResponse({'status': 'error', 'message': str(e)}, status_code=500)

@app.get("/get_network_state")
async def get_network_state():
    if not sim_state.network:
        return {"nodes": [], "edges": []}
    
    src_crs = CRS.from_epsg(4326)
    dst_crs = CRS.from_epsg(3857)
        
    res_nodes = []
    for nid, node in sim_state.network.nodes.items():
        try:
            # Transform x, y (3857) -> lat, lon (4326)
            lon, lat = warp.transform(dst_crs, src_crs, [node.x], [node.y])
            h_val = float(node.h)
            if np.isnan(h_val) or np.isinf(h_val): h_val = 0.0
            
            res_nodes.append({
                "id": nid,
                "lat": lat[0],
                "lon": lon[0],
                "h": h_val
            })
        except Exception:
            pass
        
    res_edges = []
    for eid, edge in sim_state.network.edges.items():
        try:
            h, u = edge.model.get_results()
            z = edge.model.z
            
            # Ensure arrays and handle 0-dim or scalar cases
            if h is None: h = np.zeros(edge.model.num_cells)
            elif np.ndim(h) == 0: h = np.full(edge.model.num_cells, float(h))
            
            if u is None: u = np.zeros(edge.model.num_cells)
            elif np.ndim(u) == 0: u = np.full(edge.model.num_cells, float(u))
            
            if z is None: z = np.zeros(edge.model.num_cells)
            elif np.ndim(z) == 0: z = np.full(edge.model.num_cells, float(z))
            
            # Sanitize arrays for JSON
            h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
            u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
            z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            
            # If edge.geometry is None or empty, reconstruct it roughly from start/end nodes
            geo = edge.geometry
            if not geo:
                 try:
                      n_start = sim_state.network.nodes[edge.start_node_id]
                      n_end = sim_state.network.nodes[edge.end_node_id]
                      # Transform back to Lat/Lon
                      lon_s, lat_s = warp.transform(dst_crs, src_crs, [n_start.x], [n_start.y])
                      lon_e, lat_e = warp.transform(dst_crs, src_crs, [n_end.x], [n_end.y])
                      geo = [[lat_s[0], lon_s[0]], [lat_e[0], lon_e[0]]]
                 except:
                      geo = []

            res_edges.append({
                "id": eid,
                "h": h.tolist(),
                "u": u.tolist(),
                "z": z.tolist(),
                "geometry": geo # Pass back stored or reconstructed geometry
            })
        except Exception as e:
            print(f"[ERROR] Failed to serialize Edge {eid}: {e}")
            continue
        
    return {"nodes": res_nodes, "edges": res_edges}

class PathElevationRequest(BaseModel):
    geometry: List[List[float]] # List of [lat, lon]

@app.post("/query_path_elevation")
async def query_path_elevation(req: PathElevationRequest):
    """
    Query elevation along a path.
    Returns: { "start_z": float, "end_z": float, "min_z": float, "max_z": float }
    """
    if sim_state.model is None:
        raise HTTPException(status_code=400, detail="Model not initialized")
    
    if not req.geometry or len(req.geometry) < 2:
        return {"start_z": 0.0, "end_z": 0.0, "min_z": 0.0, "max_z": 0.0}

    # Extract Lat/Lon points
    lats = [p[0] for p in req.geometry]
    lons = [p[1] for p in req.geometry]
    
    # Transform to EPSG:3857
    src_crs = CRS.from_epsg(4326)
    dst_crs = CRS.from_epsg(3857)
    
    xs, ys = warp.transform(src_crs, dst_crs, lons, lats)
    
    # Resample path to ensure we catch variations
    # Calculate step size based on grid resolution
    dx = sim_state.dx
    step = dx / 2.0 # Sample twice per cell roughly
    
    sampled_xs = []
    sampled_ys = []
    
    for i in range(len(xs) - 1):
        p1 = (xs[i], ys[i])
        p2 = (xs[i+1], ys[i+1])
        dist = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        
        n_steps = max(1, int(dist / step))
        
        for k in range(n_steps + 1):
            t = k / n_steps
            mx = p1[0] + t * (p2[0] - p1[0])
            my = p1[1] + t * (p2[1] - p1[1])
            sampled_xs.append(mx)
            sampled_ys.append(my)
            
    # Get Grid Indices
    fwd = sim_state.transform
    rows, cols = rasterio.transform.rowcol(fwd, sampled_xs, sampled_ys)
    
    vals = []
    for r, c in zip(rows, cols):
        if 0 <= r < sim_state.model.rows and 0 <= c < sim_state.model.cols:
            vals.append(float(sim_state.elevation[r, c]))
            
    if not vals:
        return {"start_z": 0.0, "end_z": 0.0, "min_z": 0.0, "max_z": 0.0}
        
    return {
        "start_z": vals[0],
        "end_z": vals[-1],
        "min_z": min(vals),
        "max_z": max(vals)
    }

@app.post("/upload_boundary_file")
async def upload_boundary_file(
    file: UploadFile = File(...), 
    value: float = Form(0.0)
):
    """
    Upload a boundary file (.zip containing Shapefile, or .geojson).
    Applies it as a MASK (bc_type = -1).
    Assumes CRS is EPSG:4326 (Lat/Lon) or EPSG:3857.
    """
    if sim_state.model is None:
        raise HTTPException(status_code=400, detail="Model not initialized")
    
    if shapefile is None and file.filename.endswith('.zip'):
        raise HTTPException(status_code=500, detail="pyshp not installed, cannot read Shapefiles")

    temp_dir = tempfile.mkdtemp()
    try:
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        geometries = [] # List of GeoJSON-like dicts
        
        if file.filename.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find .shp file
            shp_path = None
            for root, dirs, files in os.walk(temp_dir):
                for f in files:
                    if f.endswith('.shp'):
                        shp_path = os.path.join(root, f)
                        break
                if shp_path: break
            
            if not shp_path:
                raise HTTPException(status_code=400, detail="No .shp file found in zip")
            
            sf = shapefile.Reader(shp_path)
            # Convert to GeoJSON
            # pyshp shape.__geo_interface__ returns a dict like {'type': 'Polygon', 'coordinates': ...}
            for shape in sf.shapes():
                if hasattr(shape, '__geo_interface__'):
                    geometries.append(shape.__geo_interface__)
                else:
                    # Fallback for old pyshp? 
                    # Assuming Polygon
                    parts = shape.parts
                    points = shape.points
                    # Basic reconstruction (simplified, handles single polygon)
                    geometries.append({
                        'type': 'Polygon',
                        'coordinates': [points] 
                    })
                    
        elif file.filename.endswith('.geojson') or file.filename.endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if data['type'] == 'FeatureCollection':
                for feature in data['features']:
                    geometries.append(feature['geometry'])
            elif data['type'] == 'Feature':
                geometries.append(data['geometry'])
            else:
                geometries.append(data) # Geometry directly
                
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        if not geometries:
            return {"status": "warning", "message": "No geometries found"}

        # Process Geometries
        # Need to ensure they are in Grid Pixels
        # Logic similar to set_boundary
        
        shapes_for_rasterize = []
        
        for geom in geometries:
            coords = geom['coordinates']
            geom_type = geom['type']
            
            # Handle Polygon (nested lists)
            # GeoJSON Polygon: [ [ [x,y], ... ], [hole] ]
            # We assume simple polygons or multipolygons
            
            # Helper to transform a ring of points
            def transform_ring(ring):
                lons = [p[0] for p in ring]
                lats = [p[1] for p in ring]
                
                # Check bounds to guess CRS (Simple heuristic)
                if abs(lons[0]) <= 180 and abs(lats[0]) <= 90:
                    # Assume 4326 -> 3857
                    xs, ys = warp.transform(CRS.from_epsg(4326), CRS.from_epsg(3857), lons, lats)
                else:
                    # Assume already 3857 or similar meters
                    xs, ys = lons, lats
                    
                # 3857 -> Pixels
                if sim_state.transform:
                    import rasterio.transform
                    rows_idx, cols_idx = rasterio.transform.rowcol(sim_state.transform, xs, ys)
                    return list(zip(cols_idx, rows_idx)) # (x, y) = (col, row) for rasterize
                else:
                    return list(zip(xs, ys))

            new_coords = []
            
            if geom_type == 'Polygon':
                for ring in coords:
                    new_coords.append(transform_ring(ring))
            elif geom_type == 'MultiPolygon':
                for poly in coords:
                    new_poly = []
                    for ring in poly:
                        new_poly.append(transform_ring(ring))
                    new_coords.append(new_poly)
            else:
                # Skip non-polygons for now
                continue
                
            shapes_for_rasterize.append(({'type': geom_type, 'coordinates': new_coords}, 1))

        if not shapes_for_rasterize:
            return {"status": "warning", "message": "No valid polygons found"}

        rows, cols = sim_state.elevation.shape
        
        # Rasterize
        # value 1 where polygon is
        mask = features.rasterize(
            shapes_for_rasterize, out_shape=(rows, cols), default_value=0, dtype=np.uint8
        )
        
        # Apply mask (val_type = -1 for mask)
        # We assume the uploaded file defines the VALID area or the MASK area?
        # User said: "control... calculation range".
        # Usually "Mask" implies the area to EXCLUDE or the area to INCLUDE?
        # In set_boundary, "Draw Mask" sets val_type = -1.
        # "Mask" usually means "Mask Out" (hide/disable).
        # But if user uploads a "Boundary", they might mean "Calculate INSIDE this boundary".
        # If I draw a polygon and call it "Mask", I usually mean "Don't calculate here" (like an island).
        # But "Boundary" usually means "Domain".
        # Let's look at `set_boundary`:
        # `if feat.type == 'mask': val_type = -1`
        # And `bc_type_grid[mask == 1] = val_type`
        # So where the polygon IS, it becomes -1 (Inactive).
        # So "Draw Mask" = "Draw Obstacle/Hole".
        
        # If the user uploads a "Boundary" (range), they likely mean "Active Area".
        # If they mean "Active Area", then everything OUTSIDE should be -1.
        # This is ambiguous.
        # "Upload boundary range" -> likely "Compute ONLY inside this".
        # So:
        # 1. Initialize bc_type to -1 (Inactive everywhere).
        # 2. Set INSIDE polygon to 0 (Active).
        # BUT `sim_state.bc_type_grid` is already initialized to 0 (Active everywhere).
        # If I want to restrict to a polygon, I should:
        # - Set entire grid to -1.
        # - Set polygon area to 0.
        
        # Let's assume the user wants to upload a "Mask" (Obstacle) for now, to be consistent with "Draw Mask".
        # OR, since the prompt says "control calculation range", maybe I should offer both?
        # For now, I'll stick to "Mask = Exclude" (Obstacle) because that's what `bc_type=-1` does in existing code.
        # Wait, if I want to define the *domain*, I usually define the outer boundary.
        # If I define the outer boundary, I want inside to be Active.
        # Let's add a parameter or just assume "Mask" means "Exclude" to match "Draw Mask".
        # Re-reading user: "control 2D hydrodynamic calculation range".
        # Usually implies defining the active domain.
        # If so, I should probably Inverse the mask?
        # Let's assume the Shapefile contains the ACTIVE domain (Polygon).
        # So I should set everything OUTSIDE the polygon to -1.
        # How to do that?
        # rasterize(..., default_value=0, fill=0) -> 1 inside.
        # So mask==0 is outside.
        # So bc_type[mask == 0] = -1.
        
        # Let's make a decision:
        # If the tool is "Mask" (Draw Mask), it usually adds an obstacle.
        # If the tool is "Boundary" (Upload Boundary), it defines the domain.
        # I will treat "Upload Boundary" as defining the ACTIVE DOMAIN.
        # So: Outside = Inactive (-1). Inside = Active (0).
        
        # Rasterize
        domain_mask = features.rasterize(
            shapes_for_rasterize, out_shape=(rows, cols), default_value=0, fill=0, dtype=np.uint8
        )
        
        # AOI Logic:
        # domain_mask == 1 (Inside), domain_mask == 0 (Outside)
        
        # 1. Inside: Activate (0) but preserve Inflow(1)/Outflow(2)
        inside_mask = (domain_mask == 1)
        target_cells = inside_mask & (sim_state.bc_type_grid <= 0)
        sim_state.bc_type_grid[target_cells] = 0
        
        # 2. Outside: Force Inactive (-1)
        # This enforces the AOI by disabling everything outside the polygon
        sim_state.bc_type_grid[domain_mask == 0] = -1
        
        count = np.sum(inside_mask)
        
        # Update Model
        if sim_state.model:
            sim_state.model.set_boundary_conditions(sim_state.bc_type_grid, sim_state.bc_val_grid)
            # Force apply mask immediately (clears water outside AOI)
            sim_state.model.step(0.0)

        return {"status": "ok", "cells_marked": int(count)}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

class BoundaryFeature(BaseModel):
    type: str # inflow, outflow, mask
    boundary_type: str = "H" # H (Water Level) or Q (Flow Rate)
    mode: str = "constant" # constant or series
    series_data: List[List[float]] = [] # [[t, v], ...]
    geometry_type: str = "polyline" # polyline, polygon, rectangle
    points: List[List[float]] # [[x1,y1], [x2,y2], ...]
    value: float = 0.0

@app.post("/clear_boundaries")
async def clear_boundaries():
    """Clear all user-defined boundaries"""
    sim_state.boundaries = []
    if sim_state.bc_type_grid is not None:
        sim_state.bc_type_grid.fill(0) # Reset to Normal (Active)
        sim_state.bc_val_grid.fill(0.0)
        
    # Update Model
    if sim_state.model:
        sim_state.model.set_boundary_conditions(sim_state.bc_type_grid, sim_state.bc_val_grid)
            
    return {"status": "cleared", "message": "All boundaries cleared."}

@app.post("/set_boundary")
async def set_boundary(feat: BoundaryFeature):
    print(f"Set boundary: type={feat.type}, val={feat.value}, pts={len(feat.points)}")
    if sim_state.model is None:
        raise HTTPException(status_code=400, detail="Model not initialized")
    
    rows, cols = sim_state.elevation.shape
    
    # Coordinates come as Lat/Lon from Leaflet (EPSG:4326)
    # Need to transform to EPSG:3857, then to Grid Pixels
    
    coords = feat.points # [[lon, lat], ...] or [[lat, lon]]? Leaflet sends [lng, lat] usually if we did points.map(p => [p.lng, p.lat])
    if len(coords) < 2:
         raise HTTPException(status_code=400, detail="Not enough points")
    
    # 1. Transform Lat/Lon (4326) -> Meters (3857)
    # Leaflet sends [lng, lat]
    print("Debug Coordinate Transform:")
    lons = [p[0] for p in coords]
    lats = [p[1] for p in coords]
    
    for i in range(min(3, len(coords))):
        print(f"  Input Pt {i}: ({lons[i]}, {lats[i]})")

    xs, ys = warp.transform(CRS.from_epsg(4326), CRS.from_epsg(3857), lons, lats)
    
    for i in range(min(3, len(xs))):
        print(f"  Proj Pt {i} (3857): ({xs[i]}, {ys[i]})")
    
    # 2. Transform Meters (3857) -> Pixels (Grid) using affine transform inverse
    if sim_state.transform:
        print(f"  Transform: {sim_state.transform}")
        import rasterio.transform
        rows_idx, cols_idx = rasterio.transform.rowcol(sim_state.transform, xs, ys)
        
        # Keep as (row, col) for correct grid indexing
        pixel_coords = list(zip(rows_idx, cols_idx))
        
        for i, (r, c) in enumerate(pixel_coords):
            if i < 3: print(f"  -> Pixel(Row, Col): ({r}, {c})")
    else:
        # Fallback if no transform (e.g. init_test)
        pixel_coords = coords 

    print(f"Mapped to {len(pixel_coords)} vertices in grid space")

    # Filter points to ensure they are within grid bounds
    rows, cols = sim_state.bc_type_grid.shape
    valid_points = []
    for r, c in pixel_coords:
        # rowcol returns ints, but let's be safe
        r, c = int(r), int(c)
        if 0 <= r < rows and 0 <= c < cols:
            valid_points.append((r, c))

    if not valid_points:
        print("WARNING: All boundary points are outside the grid!")
        return {"status": "warning", "message": "Boundary outside grid"}

    # Apply to grid manually (Bresenham-like or simple marking)
    # Rasterize sometimes misses thin lines if not configured perfectly
    count = 0
    
    # Define val_type based on feature type
    val_type = 0
    if feat.type == 'inflow': 
        # Check boundary_type (H or Q)
        if feat.boundary_type == 'Q':
            val_type = 3
        else:
            val_type = 1
    elif feat.type == 'outflow': 
        if feat.boundary_type == 'Q':
            val_type = 3 # Reuse Q flux type (will be negative)
        else:
            val_type = 2 # Transmissive
    elif feat.type == 'mask': val_type = -1
    elif feat.type == 'aoi': val_type = 0 # Active
    
    # Create a mask for the geometry (Polygon or Polyline)
    mask = np.zeros((rows, cols), dtype=np.uint8)
    
    if feat.geometry_type == "polygon":
        # Polygons are areas, so rasterize is appropriate
        poly_pts = [(c, r) for r, c in valid_points]
        if len(poly_pts) > 2:
             shapes = [({'type': 'Polygon', 'coordinates': [poly_pts]}, 1)]
             from rasterio import features
             mask = features.rasterize(
                 shapes, out_shape=(rows, cols), default_value=0, dtype=np.uint8
             )
    else:
        # Polyline - Draw lines on mask
        for i in range(len(valid_points) - 1):
            r0, c0 = valid_points[i]
            r1, c1 = valid_points[i+1]
            
            # Bresenham's Line Algorithm
            rr, cc = line(r0, c0, r1, c1)
            
            # Filter bounds
            valid_mask = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
            rr = rr[valid_mask]
            cc = cc[valid_mask]
            
            mask[rr, cc] = 1

    # Apply Logic based on Feature Type
    if feat.type == 'aoi':
         # AOI: Inside = Active (0), Outside = Inactive (-1)
         
         # Inside (1):
         # Activate cells that are currently Inactive (-1) or Normal (0).
         # CRITICAL: Do NOT overwrite existing Inflow (1) or Outflow (2) boundaries!
         inside_mask = (mask == 1)
         # Only update cells that are NOT special boundaries (bc_type <= 0)
         # This sets -1 to 0, and 0 to 0. Leaves 1 and 2 alone.
         target_cells = inside_mask & (sim_state.bc_type_grid <= 0)
         sim_state.bc_type_grid[target_cells] = 0
         
         # Outside (0) -> Inactive (-1)
         # Force everything outside to -1, but PRESERVE existing boundaries (Type > 0)
         outside_mask = (mask == 0)
         
         # Identify cells that are currently Normal (0) or Inactive (-1)
         # i.e., NOT special boundaries (1, 2, 3)
         non_boundary_cells = (sim_state.bc_type_grid <= 0)
         
         # Only overwrite non-boundary cells
         target_inactive = outside_mask & non_boundary_cells
         sim_state.bc_type_grid[target_inactive] = -1
         
         # Check if any boundary cells are outside AOI
         boundary_outside = outside_mask & (sim_state.bc_type_grid > 0)
         count_outside = np.sum(boundary_outside)
         if count_outside > 0:
             print(f"[WARNING] {count_outside} boundary cells are OUTSIDE the AOI but were preserved.")

         count = np.sum(inside_mask) # Count active cells inside
         
         # Check if we accidentally masked everything
         total_active = np.sum(sim_state.bc_type_grid == 0)
         if total_active == 0:
             print(f"[WARNING] AOI Mask resulted in 0 ACTIVE cells! Check AOI polygon coordinates. Mask Sum: {count}")
         else:
             print(f"[INFO] AOI Applied. Active Cells: {total_active}. Mask Inside: {count}")
         
    else:
         # Standard Mask (Exclude) or Inflow/Outflow
         affected_indices = (mask == 1)
         count = np.sum(affected_indices)
         
         if count > 0:
             sim_state.bc_type_grid[affected_indices] = val_type
             
             # Calculate Init Val
             init_val = feat.value
             
             # Special handling for Q (Flow Rate)
             if val_type == 3: # Q
                  area = count * sim_state.dx * sim_state.dy
                  if area > 0:
                      init_val = feat.value / area # m/s (Unit Discharge / Area)
                      print(f"[INFO] Boundary Q Setup: Value={feat.value:.2f} m3/s, Area={area:.2f} m2, Flux Vel={init_val:.4f} m/s")
                      
                      if abs(init_val) > 10.0:
                          print(f"[WARNING] High Flux Velocity detected! ({init_val:.4f} m/s). Check Input Value (Q) or Boundary Size.")
                      
                      # If outflow, ensure value is negative (Sink)
                      if feat.type == 'outflow':
                          init_val = -abs(init_val)
                  else:
                      init_val = 0
             
             sim_state.bc_val_grid[affected_indices] = init_val
             
             # Store Boundary Definition
             if feat.type in ['inflow', 'outflow']:
                 import uuid
                 b_id = str(uuid.uuid4())
                 
                 # Find (row, col) indices
                 rows_idx, cols_idx = np.where(affected_indices)
                 # Convert to list of tuples (row, col)
                 cells = []
                 for r_idx, c_idx in zip(rows_idx, cols_idx):
                     cells.append((int(r_idx), int(c_idx)))
                 
                 boundary_obj = {
                     "id": b_id,
                     "type": val_type, # 1=H, 2=Outflow, 3=Q
                     "mode": feat.mode,
                     "value": feat.value,
                     "series": feat.series_data,
                     "cells": cells, # Store indices to update later
                     "count": int(count)
                 }
                 sim_state.boundaries.append(boundary_obj)
                 
                 # Check for H < Terrain (Only for H type)
                 if val_type == 1:
                      min_z = np.min(sim_state.elevation[affected_indices])
                      if init_val < min_z:
                          print(f"WARNING: Inflow Water Level ({init_val:.2f}) is below min terrain ({min_z:.2f}) at boundary. No water will enter.")

    # Update C++ model
    if sim_state.model:
        sim_state.model.set_boundary_conditions(sim_state.bc_type_grid, sim_state.bc_val_grid)
        
        # Debug: Check elevation at boundary
        z_vals = []
        for r, c in valid_points:
             z_vals.append(sim_state.elevation[r, c])
        
        if z_vals:
            min_z, max_z = min(z_vals), max(z_vals)
            print(f"Boundary applied. Marked {count} cells.")
            print(f"  Boundary Type: {val_type} (1=H, 2=Out, 3=Q)")
            print(f"  Boundary Value (Raw): {feat.value:.2f}")
            if val_type == 3:
                # Calculate what init_val ended up being
                # (Assuming uniform grid, we can just take one value)
                if count > 0:
                    area = count * sim_state.dx * sim_state.dy
                    v_in = feat.value / area if area > 0 else 0
                    print(f"  Flow Rate Q: {feat.value:.2f} m3/s -> Velocity/Flux: {v_in:.4f} m/s (Area={area:.1f} m2)")
            
            print(f"  Terrain Elevation at Boundary: Min={min_z:.2f}, Max={max_z:.2f}")
            
            if val_type == 1 and feat.value <= min_z:
                print("  WARNING: Boundary value is LOWER than terrain! No water will enter.")
            elif val_type == 1 and feat.value < max_z:
                print("  WARNING: Boundary value is lower than some terrain parts.")
        else:
             print(f"Boundary updated in C++ model. Marked {count} cells.")
        
    return {"status": "ok", "cells_marked": int(count)}

def line(r0, c0, r1, c1):
    """Generate line pixel coordinates"""
    # Simple DDA or Bresenham implementation using numpy
    num = max(abs(r1 - r0), abs(c1 - c0)) + 1
    rr = np.linspace(r0, r1, num).astype(int)
    cc = np.linspace(c0, c1, num).astype(int)
    return rr, cc

def apply_aoi_mask(h_grid):
    """
    Force water depth to 0.0 where AOI mask is active (bc_type == -1).
    This ensures no water is shown outside the user-defined area,
    regardless of internal calculation state.
    """
    if sim_state.bc_type_grid is not None and h_grid is not None:
        if h_grid.shape == sim_state.bc_type_grid.shape:
            h_grid[sim_state.bc_type_grid == -1] = 0.0
    return h_grid

def overlay_1d_results(h_2d_in):
    """
    Overlay 1D model water depth onto 2D grid for visualization and results.
    Uses Network Geometry to ensure continuous visualization.
    """
    if not (sim_state.network and sim_state.transform) or sim_state.disable_1d:
        return h_2d_in
        
    h_out = h_2d_in.copy()
    
    # 1. Determine Sampling Step (same as coupling)
    dx_2d = abs(sim_state.transform[0])
    dy_2d = abs(sim_state.transform[4])
    step_size = min(dx_2d, dy_2d) * 0.5
    
    # Cache model results to avoid repeated calls
    model_results_cache = {}

    count_mapped = 0
    
    for eid, edge in sim_state.network.edges.items():
        model_1d = edge.model
        if not model_1d: continue
        
        # Get Results (Cached)
        if model_1d not in model_results_cache:
            try:
                h1d_arr, _ = model_1d.get_results()
                model_results_cache[model_1d] = h1d_arr
            except:
                model_results_cache[model_1d] = None
        
        h1d_arr = model_results_cache[model_1d]
        if h1d_arr is None: continue

        # 2. Generate Points
        xs = None
        ys = None
        ts = None
        
        if edge.geometry and len(edge.geometry) >= 2:
            try:
                # Geometry is [lat, lon] -> [lon, lat]
                g_lons = [p[1] for p in edge.geometry]
                g_lats = [p[0] for p in edge.geometry]
                
                # Project to Grid CRS (Assuming 3857 for now)
                xs_poly, ys_poly = warp.transform(CRS.from_epsg(4326), CRS.from_epsg(3857), g_lons, g_lats)
                
                # Sanity Check: Are points within Grid Bounds?
                # Calculate bounds of the grid in projected coords
                if sim_state.transform:
                     # transform * (0, 0) -> Top Left
                     # transform * (cols, rows) -> Bottom Right
                     x_tl, y_tl = sim_state.transform * (0, 0)
                     x_br, y_br = sim_state.transform * (sim_state.model.cols, sim_state.model.rows)
                     
                     min_x_grid, max_x_grid = min(x_tl, x_br), max(x_tl, x_br)
                     min_y_grid, max_y_grid = min(y_tl, y_br), max(y_tl, y_br)
                     
                     # Check if poly is completely outside
                     poly_min_x, poly_max_x = min(xs_poly), max(xs_poly)
                     poly_min_y, poly_max_y = min(ys_poly), max(ys_poly)
                     
                     if (poly_max_x < min_x_grid or poly_min_x > max_x_grid or
                         poly_max_y < min_y_grid or poly_min_y > max_y_grid):
                         # print(f"[DEBUG] Edge {eid} is outside grid bounds. Skipping Overlay.")
                         continue

                points_poly = list(zip(xs_poly, ys_poly))
                dists = [0.0]
                total_len = 0.0
                for i in range(len(points_poly)-1):
                    d = np.hypot(points_poly[i+1][0] - points_poly[i][0], points_poly[i+1][1] - points_poly[i][1])
                    total_len += d
                    dists.append(total_len)
                
                if total_len > 1e-6:
                    num_samples = int(np.ceil(total_len / step_size))
                    if num_samples < 2: num_samples = 2
                    d_targets = np.linspace(0, total_len, num_samples)
                    xs = np.interp(d_targets, dists, xs_poly)
                    ys = np.interp(d_targets, dists, ys_poly)
                    ts = d_targets / total_len
            except Exception:
                pass # Fallback to straight line
        
        if xs is None:
            # Fallback: Straight Line
            try:
                n_start = sim_state.network.nodes[edge.start_node_id]
                n_end = sim_state.network.nodes[edge.end_node_id]
                x1, y1 = n_start.x, n_start.y
                x2, y2 = n_end.x, n_end.y
                
                dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if dist < 1e-6: continue
                
                num_samples = int(np.ceil(dist / step_size))
                if num_samples < 2: num_samples = 2
                
                ts = np.linspace(0, 1.0, num_samples)
                xs = x1 + ts * (x2 - x1)
                ys = y1 + ts * (y2 - y1)
            except Exception:
                continue
            
        # 3. Map to Grid
        if xs is not None:
            try:
                rows, cols = rasterio.transform.rowcol(sim_state.transform, xs, ys)
                
                for i in range(len(rows)):
                    r, c = rows[i], cols[i]
                    if 0 <= r < h_out.shape[0] and 0 <= c < h_out.shape[1]:
                        # Map t to 1D index
                        t = ts[i]
                        idx_1d = int(t * model_1d.num_cells)
                        if idx_1d >= model_1d.num_cells: idx_1d = model_1d.num_cells - 1
                        if idx_1d < 0: idx_1d = 0
                        
                        if idx_1d < len(h1d_arr):
                            h_val = float(h1d_arr[idx_1d])
                            
                            # Get Channel Width
                            width = 10.0 # Default
                            if hasattr(model_1d, 'w') and len(model_1d.w) > idx_1d:
                                width = float(model_1d.w[idx_1d])
                            
                            # Get Bed Elevation for WSE
                            z_bed = 0.0
                            if hasattr(model_1d, 'z') and model_1d.z is not None and len(model_1d.z) > idx_1d:
                                z_bed = float(model_1d.z[idx_1d])
                            
                            wse_1d = z_bed + h_val

                            # Calculate buffer radius in cells
                            # Use max(dx, dy) to be conservative, or min to be precise?
                            # Using min(dx, dy) ensures we cover at least the width.
                            cell_size = min(dx_2d, dy_2d)
                            radius_cells = int(np.ceil((width / 2.0) / cell_size))
                            
                            # Iterate neighborhood to fill width
                            r_min = max(0, r - radius_cells)
                            r_max = min(h_out.shape[0], r + radius_cells + 1)
                            c_min = max(0, c - radius_cells)
                            c_max = min(h_out.shape[1], c + radius_cells + 1)
                            
                            for rr in range(r_min, r_max):
                                for cc in range(c_min, c_max):
                                    # Check Mask (AOI) - Don't draw if inactive
                                    if sim_state.bc_type_grid is not None:
                                        if sim_state.bc_type_grid[rr, cc] == -1:
                                            continue
                                            
                                    # Optional: Check circular distance for better look?
                                    # Or simple square for performance.
                                    # Let's do simple check to avoid square artifacts for large widths
                                    if radius_cells > 1:
                                        dr = abs(rr - r)
                                        dc = abs(cc - c)
                                        if np.hypot(dr, dc) > radius_cells:
                                            continue
                                    
                                    # [FIX] Force Dryness if 1D is Dry (User Requirement: Endpoint must be dry)
                                    if h_val <= DRY_THRESHOLD:
                                         h_out[rr, cc] = 0.0
                                         continue

                                    # Check WSE vs Terrain (Prevent Uphill Flooding)
                                    z_terrain = -9999.0
                                    if sim_state.elevation is not None:
                                         z_terrain = float(sim_state.elevation[rr, cc])
                                    
                                    h_overlay = wse_1d - z_terrain
                                    
                                    if h_overlay > DRY_THRESHOLD:
                                        if h_overlay > h_out[rr, cc]:
                                            h_out[rr, cc] = h_overlay
                                            count_mapped += 1
            except Exception:
                continue
    
    if count_mapped > 0 and sim_state.current_time % 10.0 < 0.1: # Log occasionally
        print(f"[Overlay] Mapped {count_mapped} 1D cells to 2D grid.")
        
    return h_out

def save_simulation_result(idx: int):
    """Save h, u, v to GeoTIFF with LZW compression"""
    if sim_state.model is None:
        return
        
    with sim_state.lock:
        try:
            h, u, v = sim_state.model.get_results()
        except Exception as e:
            print(f"Error getting results for save: {e}")
            return
    
    # Overlay 1D results for consistency
    h = overlay_1d_results(h)
    
    # Force AOI Mask on Result
    h = apply_aoi_mask(h)
    
    # Apply AOI Mask to Velocity as well (h is already masked in overlay_1d_results)
    if sim_state.bc_type_grid is not None:
        if u.shape == sim_state.bc_type_grid.shape:
            u[sim_state.bc_type_grid == -1] = 0.0
            v[sim_state.bc_type_grid == -1] = 0.0
    
    rows, cols = h.shape
    
    # Ensure results directory
    os.makedirs("results", exist_ok=True)
    filename = f"results/rst_{idx}.tif"
    
    # Determine transform and CRS
    transform = sim_state.transform
    crs = CRS.from_epsg(3857) # Default
    
    if transform is None:
        # Fallback for init_test
        from rasterio.transform import from_origin
        transform = from_origin(0, rows, sim_state.dx, sim_state.dy) # Simple cartesian
        crs = None # No CRS for simple test
        
    profile = {
        'driver': 'GTiff',
        'height': rows,
        'width': cols,
        'count': 3,
        'dtype': rasterio.float64,
        'crs': crs,
        'transform': transform,
        'nodata': -9999,
        'compress': 'lzw',  # Lossless compression
        'predictor': 2,      # Horizontal differencing (good for smooth data)
        'tiled': True
    }
    
    try:
        with rasterio.open(filename, 'w', **profile) as dst:
            dst.write(h, 1)
            dst.write(u, 2)
            dst.write(v, 3)
            dst.set_band_description(1, 'Water Depth')
            dst.set_band_description(2, 'Velocity X')
            dst.set_band_description(3, 'Velocity Y')
        print(f"Saved {filename} at time {sim_state.current_time:.2f}")
    except Exception as e:
        print(f"Error saving result: {e}")

def update_boundaries(current_time):
    """
    Update bc_val_grid based on time-varying boundaries.
    Returns True if grid was updated.
    """
    updated = False
    
    for b in sim_state.boundaries:
        val = 0.0
        
        # 1. Determine Value
        if b['mode'] == 'constant':
            val = b['value']
        elif b['mode'] == 'series':
            # Linear Interpolation
            series = b['series'] # [[t, v], ...]
            if not series:
                val = b['value'] # Fallback
            else:
                # Find interval
                # Assume sorted by time
                # If t < t0, use v0
                # If t > tn, use vn
                t0, v0 = series[0]
                if current_time <= t0:
                    val = v0
                elif current_time >= series[-1][0]:
                    val = series[-1][1]
                else:
                    # Interpolate
                    for i in range(len(series) - 1):
                        t_a, v_a = series[i]
                        t_b, v_b = series[i+1]
                        if t_a <= current_time <= t_b:
                            frac = (current_time - t_a) / (t_b - t_a) if t_b > t_a else 0
                            val = v0 = v_a + frac * (v_b - v_a)
                            break
        
        # 2. Adjust for Type
        final_val = val
        if b['type'] == 3: # Q (Flow Rate m3/s) -> Source Rate (m/s)
            # q = Q / Area
            area = b['count'] * sim_state.dx * sim_state.dy
            if area > 0:
                final_val = val / area
                # Only log if value is significant to avoid spam
                if abs(final_val) > 10.0:
                     # Using a simple counter or checking change might be better, but print is okay for now
                     # print(f"[WARNING] High Flux Velocity in Update: {final_val:.4f} m/s") 
                     pass
            else:
                final_val = 0.0
        
        # 3. Update Grid
        # Check if value changed significantly? 
        # For simplicity, just update.
        # cells is list of (r, c)
        # Using numpy advanced indexing is faster if we had r_arr, c_arr
        # But we stored list of tuples.
        # Let's convert to arrays for this boundary if not already?
        # Actually, let's just do:
        # rows = [c[0] for c in b['cells']]
        # cols = [c[1] for c in b['cells']]
        # sim_state.bc_val_grid[rows, cols] = final_val
        
        # Optimization: Store rows/cols arrays in boundary obj?
        # Yes, let's convert on the fly for now, optimization later.
        
        rows = [c[0] for c in b['cells']]
        cols = [c[1] for c in b['cells']]
        
        # Only update if cells exist
        if rows:
            sim_state.bc_val_grid[rows, cols] = final_val
            updated = True
            
    return updated

async def simulation_loop():
    print("Simulation loop started")
    
    # Log Backend Info
    if sim_state.model is not None and hasattr(sim_state.model, 'get_backend_name'):
         print(f"[INFO] Compute Backend: {sim_state.model.get_backend_name()}")
    elif sim_state.model is not None and GPU_AVAILABLE:
         # Should not happen if SWEModelGPU has get_backend_name
         print(f"[INFO] Compute Backend: GPU (Taichi)")
    elif sim_state.model is not None:
         print(f"[INFO] Compute Backend: CPU (DLL)")
    else:
         print(f"[INFO] Compute Backend: 1D Network Only")
    
    # Initial conservative time step
    if sim_state.model is not None:
        dt = min(0.1, 0.002 * sim_state.dx)
    else:
        dt = 0.1
    print(f"Initial time step dt = {dt:.4f}s")
    
    # Log active boundaries
    if sim_state.boundaries:
        print(f"Active Boundaries: {len(sim_state.boundaries)}")
        for i, b in enumerate(sim_state.boundaries):
            print(f"  B{i}: Type={b['type']}, Mode={b['mode']}, Cells={b['count']}")

    last_log_time = 0.0
    
    while sim_state.is_running and (sim_state.model is not None or sim_state.model_1d is not None or sim_state.network is not None):
        if sim_state.model is None and sim_state.model_1d is None and sim_state.network is None:
            sim_state.is_running = False
            break
        # Check total time
        if sim_state.current_time >= sim_state.total_time:
             print(f"[INFO] Simulation Limit Reached: Time {sim_state.current_time:.2f}s >= Total Time {sim_state.total_time:.2f}s")
             
             # Trigger final save
             if sim_state.model is not None:
                 print(f"[>>> FINAL SAVE <<<] Saving final result at {sim_state.current_time:.2f}s")
                 await asyncio.to_thread(save_simulation_result, sim_state.save_index)
                 sim_state.save_index += 1
             
             sim_state.is_running = False
             break
        
        # --- Adaptive Time Stepping & Stability Check ---
        dt_source = "BASE"
        steps_per_batch = 1 # Default batch size
        if sim_state.model is not None and hasattr(sim_state.model, 'get_stability_info'):
            try:
                max_h, max_v = sim_state.model.get_stability_info()
                
                # Check for Instability (NaN or Inf or Extreme Value)
                if np.isnan(max_h) or np.isinf(max_h) or max_h > 1e6:
                    print(f"[ERROR] Simulation unstable! Max Depth={max_h}. Stopping.")
                    sim_state.is_running = False
                    break
                
                # Warning for Extreme Depth (likely user error in input)
                if max_h > 100.0:
                    print(f"[WARNING] Extreme Depth Detected: {max_h:.2f}m. Pausing Simulation to prevent overflow. Check Boundary Inputs.")
                    sim_state.is_running = False
                    break
                
                # Compute CFL-based dt
                # dt < CFL * dx / (|u| + sqrt(gh))
                safe_h = max(0.0, max_h)
                c = np.sqrt(9.81 * safe_h)
                denom = c + max_v
                if denom < 1e-3: denom = 1e-3
                
                dt_cfl = 0.5 * sim_state.dx / denom
                
                # Smooth update or clamp
                # Clamp dt to [1e-4, 1.0]
                dt = max(1e-4, min(dt_cfl, 1.0))
                dt_source = "2D_CFL"
                
                # Adjust batch size to target ~0.1s of simulation per batch (or reasonable compute chunk)
                # If dt is small, we need more steps to make progress.
                # But we don't want to block the thread too long.
                # Let's cap steps_per_batch to 100 to ensure responsiveness.
                if dt < 0.01:
                    steps_per_batch = 50
                elif dt < 0.05:
                    steps_per_batch = 20
                else:
                    steps_per_batch = 10
                    
            except Exception as e:
                print(f"[WARNING] Stability check failed: {e}")

        dt_runtime_label = dt_source
        dt_limits = []

        if sim_state.model_1d is not None and not sim_state.disable_1d:
            try:
                h_1d, u_1d = sim_state.model_1d.get_results()
                h_safe = np.maximum(h_1d, 0.0)
                c = np.sqrt(9.81 * h_safe) + np.abs(u_1d)
                max_c = float(np.max(c)) if len(c) else 0.0
                if max_c < 0.1:
                    max_c = 0.1
                dx_1d = float(getattr(sim_state.model_1d, 'dx', sim_state.dx))
                dt_limits.append(("1D_CFL", 0.7 * dx_1d / max_c))
            except Exception:
                pass

        if sim_state.network is not None and not sim_state.disable_1d:
            try:
                for edge in sim_state.network.edges.values():
                    h_e, u_e = edge.model.get_results()
                    h_safe = np.maximum(h_e, 0.0)
                    c = np.sqrt(9.81 * h_safe) + np.abs(u_e)
                    max_c = float(np.max(c)) if len(c) else 0.0
                    if max_c < 0.1:
                        max_c = 0.1
                    dx_e = float(getattr(edge.model, 'dx', sim_state.dx))
                    dt_limits.append(("NET_CFL", 0.7 * dx_e / max_c))

                ramp_factor = 1.0
                if sim_state.current_time < 10.0:
                    ramp_factor = max(0.0, sim_state.current_time / 10.0)
                for node in sim_state.network.nodes.values():
                    if getattr(node, 'bc_type', None) == 'inflow':
                        q_in = abs(float(getattr(node, 'bc_value', 0.0))) * ramp_factor
                        if q_in > 1e-9:
                            area = float(getattr(node, 'area', 0.0))
                            if area > 1e-9:
                                dt_limits.append(("NODE_Q", 0.05 * area / q_in))
            except Exception:
                pass

        if dt_limits:
            limit_label, dt_limit = min(dt_limits, key=lambda x: x[1])
            dt_pre = float(dt)
            if dt_limit > 0:
                dt = max(1e-4, min(dt, float(dt_limit)))
                if dt < dt_pre:
                    dt_runtime_label = str(limit_label)
                if dt < 0.01:
                    steps_per_batch = max(steps_per_batch, 50)
                elif dt < 0.05:
                    steps_per_batch = max(steps_per_batch, 20)

        # ------------------------------------------------
        
        # General batching rule for efficiency
        if dt < 0.005: steps_per_batch = max(steps_per_batch, 50)
        elif dt < 0.02: steps_per_batch = max(steps_per_batch, 20)
        elif dt < 0.1: steps_per_batch = max(steps_per_batch, 5)

        # 1. Reset 2D Sources before Coupling (Critical for stability and cleanup)
        # This ensures that only currently active coupling nodes contribute to sources.
        if sim_state.model and hasattr(sim_state.model, 'reset_sources'):
            sim_state.model.reset_sources()
            
        # Update Boundaries (Time-Varying)
        if sim_state.model and update_boundaries(sim_state.current_time):
             sim_state.model.set_boundary_conditions(sim_state.bc_type_grid, sim_state.bc_val_grid)
        
        for _ in range(steps_per_batch):
            if not sim_state.is_running: break
            
            # --- Enforce AOI Mask on 1D Models ---
            # If 2D Mask exists, force mapped 1D cells to be dry if outside AOI
            
            # 1. Force explicitly marked inactive nodes (from initialization)
            if hasattr(sim_state, 'inactive_1d_nodes') and not sim_state.disable_1d:
                for model, idx in sim_state.inactive_1d_nodes:
                     # Safe access for different model backends (C++ vs Taichi)
                     if hasattr(model, 'h'):
                         try: model.h[idx] = 0.0
                         except: pass
                     if hasattr(model, 'u'):
                         try: model.u[idx] = 0.0
                         except: pass

            # 2. Check coupled nodes (Legacy check, in case some slipped through)
            if sim_state.bc_type_grid is not None and sim_state.coupler is not None and not sim_state.disable_1d:
                for node in sim_state.coupler.nodes:
                     r, c = int(node.r_2d), int(node.c_2d)
                     # Safety check bounds
                     if 0 <= r < sim_state.model.rows and 0 <= c < sim_state.model.cols:
                         if sim_state.bc_type_grid[r, c] == -1: # Inactive
                             # Zero out the 1D cell
                             if node.model_1d is not None:
                                 idx = int(node.idx_1d)
                                 if 0 <= idx < node.model_1d.num_cells:
                                     if hasattr(node.model_1d, 'h'):
                                         try: node.model_1d.h[idx] = 0.0
                                         except: pass
                                     if hasattr(node.model_1d, 'u'):
                                         try: node.model_1d.u[idx] = 0.0
                                         except: pass

            # --- Coupling Step ---
            with sim_state.lock:
                # Re-apply sources for this step
                if sim_state.model:
                    if hasattr(sim_state.model, 'reset_sources'):
                        sim_state.model.reset_sources()
                    elif sim_state.coupler and not sim_state.disable_1d:
                         # Fallback: Manually reset sources for coupled nodes if DLL doesn't support reset_sources
                         for node in sim_state.coupler.nodes:
                             sim_state.model.set_source(node.r_2d, node.c_2d, 0.0)
                             if hasattr(sim_state.model, 'set_source_momentum'):
                                 sim_state.model.set_source_momentum(node.r_2d, node.c_2d, 0.0, 0.0)

                if sim_state.coupler and not sim_state.disable_1d:
                    if hasattr(sim_state.coupler, 'compute_exchange_optimized'):
                        sim_state.coupler.compute_exchange_optimized(dt)
                    else:
                        sim_state.coupler.compute_exchange(dt)
                
                # --- 2D Step ---
                try:
                    if sim_state.model:
                        sim_state.model.step(dt)
                    
                    # --- 1D Step ---
                    if sim_state.model_1d and not sim_state.disable_1d:
                        sim_state.model_1d.step(dt)
                        # [FIX] Sanitize Legacy 1D Model State
                        if hasattr(sim_state.model_1d, 'h'):
                             sim_state.model_1d.h = np.nan_to_num(sim_state.model_1d.h, nan=0.0, posinf=0.0, neginf=0.0)
                             sim_state.model_1d.h[sim_state.model_1d.h < 0] = 0.0
                        if hasattr(sim_state.model_1d, 'u'):
                             sim_state.model_1d.u = np.nan_to_num(sim_state.model_1d.u, nan=0.0, posinf=0.0, neginf=0.0)

                    # --- Network Step ---
                    if sim_state.network and not sim_state.disable_1d:
                        sim_state.network.run_step(dt, sim_time=sim_state.current_time)
                        
                        # [FIX] Sanitize Network 1D Model State (Addressing User Point 4)
                        # Ensure no NaN or Negative depths propagate
                        for edge in sim_state.network.edges.values():
                             if edge.model:
                                 # Sanitize H
                                 if np.any(np.isnan(edge.model.h)) or np.any(edge.model.h < 0):
                                     edge.model.h = np.nan_to_num(edge.model.h, nan=0.0, posinf=0.0, neginf=0.0)
                                     edge.model.h[edge.model.h < 0] = 0.0
                                 # Sanitize U
                                 if np.any(np.isnan(edge.model.u)):
                                     edge.model.u = np.nan_to_num(edge.model.u, nan=0.0, posinf=0.0, neginf=0.0)

                    # [FIX] Enforce AOI Mask on 1D Nodes (Inactive Cleanup) - BEFORE Sync
                    if hasattr(sim_state, 'inactive_1d_nodes') and not sim_state.disable_1d:
                        for model, idx in sim_state.inactive_1d_nodes:
                            if model is not None:
                                if hasattr(model, 'h') and idx < len(model.h):
                                     try: model.h[idx] = 0.0
                                     except: pass
                                if hasattr(model, 'u') and idx < len(model.u):
                                     try: model.u[idx] = 0.0
                                     except: pass

                        # Sync 1D velocities to 2D grid for visualization
                        if sim_state.coupler and not sim_state.disable_1d and hasattr(sim_state.coupler, 'sync_velocity_for_visualization'):
                            try:
                                sim_state.coupler.sync_velocity_for_visualization()
                            except Exception as e:
                                print(f"[WARNING] Velocity sync failed: {e}")

                except Exception as step_e:
                    print(f"[ERROR] Simulation Step Failed: {step_e}")
                    import traceback
                    traceback.print_exc()
                    sim_state.is_running = False
                    break

                
            sim_state.current_time += dt
            
            # Check save condition inside the tight loop to be precise
            if sim_state.model is not None:
                target_time = sim_state.save_index * sim_state.dt_save
                if sim_state.current_time >= target_time:
                    print(f"[>>> SAVING <<<] Saving rst_{sim_state.save_index}.tif at {sim_state.current_time:.2f}s")
                    await asyncio.to_thread(save_simulation_result, sim_state.save_index)
                    sim_state.save_index += 1

        # Log progress every 5 seconds of simulation time
        if sim_state.current_time - last_log_time >= 5.0:
            # 2D Stats
            max_h_2d = 0.0
            vol_2d = 0.0
            
            if sim_state.model:
                try:
                    with sim_state.lock:
                        h_curr, _, _ = sim_state.model.get_results()
                    
                    if h_curr is None:
                        max_h_2d = 0.0
                    elif np.any(np.isnan(h_curr)):
                         max_h_2d = np.nan
                    else:
                        # Mask inactive cells for stats to avoid showing ghost values
                        if sim_state.bc_type_grid is not None:
                            mask = sim_state.bc_type_grid != -1
                            h_valid = h_curr[mask]
                        else:
                            h_valid = h_curr
                            
                        if h_valid.size > 0:
                            max_h_2d = np.max(h_valid)
                            vol_2d = np.sum(h_valid) * sim_state.dx * sim_state.dy
                        else:
                            max_h_2d = 0.0
                            vol_2d = 0.0
                            
                except Exception as e:
                    print(f"[Simulation] Error calculating 2D stats: {e}")
                    h_curr = None

            # 1D Network Stats
            max_h_1d = 0.0
            vol_1d = 0.0
            
            if sim_state.network and not sim_state.disable_1d:
                # Edges
                for edge in sim_state.network.edges.values():
                    h_edge, _ = edge.model.get_results()
                    if np.any(np.isnan(h_edge)):
                        max_h_1d = np.nan
                        break
                    if len(h_edge) > 0:
                        max_h_1d = max(max_h_1d, np.max(h_edge))
                        vol_1d += np.sum(h_edge * edge.model.w * edge.model.dx)
                
                # Nodes
                if not np.isnan(max_h_1d):
                    for node in sim_state.network.nodes.values():
                        max_h_1d = max(max_h_1d, node.h)
                        vol_1d += node.h * node.area

            if np.isnan(max_h_2d) or np.isnan(max_h_1d):
                print(f"[ERROR] Simulation unstable! NaN detected at t={sim_state.current_time:.2f}s. Stopping.")
                sim_state.is_running = False
                break

            print(f"[Progress] Time: {sim_state.current_time:.1f}s | dt: {dt:.4f}s ({dt_runtime_label}) | Steps/Batch: {steps_per_batch} | Max Depth 2D: {max_h_2d:.4f}m | Max Depth 1D: {max_h_1d:.4f}m | Vol 2D: {vol_2d:.0f} | Vol 1D: {vol_1d:.0f}")
            last_log_time = sim_state.current_time
            
        # Yield to event loop
        await asyncio.sleep(0.001) 
        
    print("Simulation loop stopped")

def initialize_network_coupling(bank_height: float = 2.0):
    """
    Initialize coupling for Network1D if exists.
    Iterates all edges and finds overlap with 2D grid.
    """
    # Only proceed if both Network and Model exist
    if not sim_state.network or not sim_state.model:
        return

    print(f"Initializing Network Coupling with Bank Height: {bank_height}m")
    
    # Reset inactive nodes list
    sim_state.inactive_1d_nodes = []
    
    # Check for potential initial spill
    max_init_h = 0.0
    for node in sim_state.network.nodes.values():
        if node.h > max_init_h: max_init_h = node.h
    
    if max_init_h >= bank_height:
        print(f"[WARNING] Initial Depth ({max_init_h}m) >= Bank Height ({bank_height}m). Immediate flooding will occur!")
    else:
        print(f"[INFO] Initial Depth ({max_init_h}m) is safely below Bank Height ({bank_height}m).")

    # Create or Reset Coupler
    sim_state.coupler = Coupler(None, sim_state.model, sim_state.bc_type_grid)
    
    connections_count = 0
    
    fwd = sim_state.transform
    rows, cols = sim_state.model.rows, sim_state.model.cols
    
    diag = float(np.hypot(float(sim_state.dx), float(sim_state.dy))) if hasattr(sim_state, "dx") and hasattr(sim_state, "dy") else 0.0
    max_map_dist = 2.5 * diag if diag > 0 else 0.0

    skipped_mask = 0
    skipped_map = 0

    # NOTE: We do NOT couple Nodes explicitly anymore.
    # The Edge Dense Sampling covers the endpoints (t=0 and t=1), which correspond to the Nodes.
    # This avoids "Double Coupling" (Node + Edge End) at the same location.
    # It also ensures we use the "Burned-in" Z values from the Edge loop for the junctions.

    # Pre-calculate Node Degrees to identify Network Outlets
    node_degree_end = {} # Count how many edges end at this node
    node_degree_start = {} # Count how many edges start at this node
    node_total_degree = {} # Total connections
    
    for nid in sim_state.network.nodes:
        node_degree_end[nid] = 0
        node_degree_start[nid] = 0
        node_total_degree[nid] = 0
        
    for edge in sim_state.network.edges.values():
        node_degree_start[edge.start_node_id] += 1
        node_degree_end[edge.end_node_id] += 1
        node_total_degree[edge.start_node_id] += 1
        node_total_degree[edge.end_node_id] += 1
        
    # Identify Outlet Nodes: Degree 1 AND is an End Node of an edge
    outlet_node_ids = set()
    for nid, degree in node_total_degree.items():
        if degree == 1 and node_degree_end[nid] == 1:
            outlet_node_ids.add(nid)
            print(f"[Network] Identified Outlet Node: {nid}")

    # [FIX] Track coupled cells to detect overlaps
    # Map (r, c) -> Set of Edge IDs
    coupled_cells_map = {}

    # 1. Iterate all edges with Dense Sampling (Line Rasterization)
    modified_terrain = False
    
    for eid, edge in sim_state.network.edges.items():
        # Get Start/End Coords from Nodes
        n_start = sim_state.network.nodes[edge.start_node_id]
        n_end = sim_state.network.nodes[edge.end_node_id]
        
        x1, y1 = n_start.x, n_start.y
        x2, y2 = n_end.x, n_end.y
        
        # Calculate Edge Length
        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if dist < 1e-6: continue

        # Determine Sampling Step based on 2D Grid Resolution
        # Use half of the smallest cell dimension to ensure coverage (Nyquist)
        # Assuming transform [0] and [4] are pixel sizes roughly
        dx_2d = abs(sim_state.transform[0])
        dy_2d = abs(sim_state.transform[4])
        step_size = min(dx_2d, dy_2d) * 0.5
        
        # Dense Sampling Points
        xs = None
        ys = None
        ts = None
        
        # Calculate Overall Edge Tangent and Normal (Approximation for Momentum Direction)
        # Even for Polylines, the 1D flow is conceptually "along the reach".
        # For curved channels, this is a simplification, but better than hardcoded North.
        dx_edge = x2 - x1
        dy_edge = y2 - y1
        dist_edge = np.sqrt(dx_edge**2 + dy_edge**2)
        
        nx_seg = 0.0
        ny_seg = 1.0 # Default
        
        if dist_edge > 1e-6:
            tx_seg = dx_edge / dist_edge
            ty_seg = dy_edge / dist_edge
            nx_seg = -ty_seg
            ny_seg = tx_seg

        # Try to use Edge Geometry (Polyline) if available
        use_geometry = False
        if edge.geometry and len(edge.geometry) >= 2:
            try:
                # Geometry from Frontend is [[lat, lng], ...]
                # We need [lng, lat] for projection (x=lng, y=lat for EPSG:4326)
                g_lons = [p[1] for p in edge.geometry]
                g_lats = [p[0] for p in edge.geometry]
                
                # Validation: Check for likely swapped coordinates
                max_lat = max(abs(l) for l in g_lats)
                if max_lat > 90.0:
                    print(f"[WARNING] Edge {eid} has Lat > 90 ({max_lat}). Swapped [Lon, Lat] detected? Flipping.")
                    g_lons, g_lats = g_lats, g_lons
                
                # Project to Meters (3857 or Custom CRS)
                target_crs = sim_state.crs if sim_state.crs else CRS.from_epsg(3857)
                xs_poly, ys_poly = warp.transform(CRS.from_epsg(4326), target_crs, g_lons, g_lats)
                
                # [FIX] Ensure Geometry Direction matches Start -> End Node
                # Calculate distance from first point to Start Node
                d_start = np.hypot(xs_poly[0] - x1, ys_poly[0] - y1)
                # Calculate distance from last point to Start Node
                d_end = np.hypot(xs_poly[-1] - x1, ys_poly[-1] - y1)
                
                # If last point is closer to Start Node than first point, REVERSE geometry
                if d_end < d_start:
                    print(f"[INFO] Edge {eid} Geometry appears reversed. Flipping to match Start Node.")
                    xs_poly = xs_poly[::-1]
                    ys_poly = ys_poly[::-1]
                    
                    # Update distances after flip
                    d_start = np.hypot(xs_poly[0] - x1, ys_poly[0] - y1)
                    d_end = np.hypot(xs_poly[-1] - x2, ys_poly[-1] - y2)

                # [FIX] Geometric Integrity Check
                # Verify that the geometry actually starts/ends near the Nodes.
                # If the geometry is offset significantly, it implies a projection or data error.
                GEOM_TOLERANCE = 50.0 # meters (generous to allow for manual digitization errors)
                
                d_start_check = np.hypot(xs_poly[0] - x1, ys_poly[0] - y1)
                d_end_check = np.hypot(xs_poly[-1] - x2, ys_poly[-1] - y2)
                
                if d_start_check > GEOM_TOLERANCE or d_end_check > GEOM_TOLERANCE:
                     print(f"[WARNING] Edge {eid} Geometry Mismatch!")
                     print(f"          Start Node ({x1:.1f}, {y1:.1f}) vs Poly Start ({xs_poly[0]:.1f}, {ys_poly[0]:.1f}) -> Diff: {d_start_check:.1f}m")
                     print(f"          End Node   ({x2:.1f}, {y2:.1f}) vs Poly End   ({xs_poly[-1]:.1f}, {ys_poly[-1]:.1f}) -> Diff: {d_end_check:.1f}m")
                     print(f"          This may cause 'Ghost Water' or incorrect coupling locations.")

                # Calculate cumulative distance along polyline
                points_poly = list(zip(xs_poly, ys_poly))
                dists = [0.0]
                total_len = 0.0
                for i in range(len(points_poly)-1):
                    d = np.hypot(points_poly[i+1][0] - points_poly[i][0], points_poly[i+1][1] - points_poly[i][1])
                    total_len += d
                    dists.append(total_len)
                
                if total_len > 1e-6:
                    # Number of samples
                    num_samples = int(np.ceil(total_len / step_size))
                    if num_samples < 2: num_samples = 2
                    
                    # Interpolate along the curve
                    # Target distances
                    d_targets = np.linspace(0, total_len, num_samples)
                    
                    xs = np.interp(d_targets, dists, xs_poly)
                    ys = np.interp(d_targets, dists, ys_poly)
                    
                    # Map t (0..1) relative to total length
                    ts = d_targets / total_len
                    use_geometry = True
                    print(f"[INFO] Used Polyline Geometry for Edge {eid}: {len(xs)} samples")
            except Exception as e:
                print(f"[WARNING] Failed to process geometry for Edge {eid}: {e}. Falling back to straight line.")
        
        if not use_geometry:
            # Fallback: Straight Line
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if dist < 1e-6: continue # Skip zero length edge
            
            num_samples = int(np.ceil(dist / step_size))
            if num_samples < 2: num_samples = 2
            
            ts = np.linspace(0, 1.0, num_samples)
            xs = x1 + ts * (x2 - x1)
            ys = y1 + ts * (y2 - y1)
        
        # Get Grid Indices for ALL samples
        fwd = sim_state.transform
        rows, cols = rasterio.transform.rowcol(fwd, xs, ys)
        
        # DEBUG: Log first few samples for the first edge to verify transform
        if eid == list(sim_state.network.edges.keys())[0]:
            print(f"[DEBUG] Edge {eid} Sampling:")
            print(f"  Transform: {fwd}")
            print(f"  Sample 0: x={xs[0]:.2f}, y={ys[0]:.2f} -> r={rows[0]}, c={cols[0]}")
            if len(xs) > 1:
                print(f"  Sample -1: x={xs[-1]:.2f}, y={ys[-1]:.2f} -> r={rows[-1]}, c={cols[-1]}")
            print(f"  Grid Shape: {sim_state.model.rows}x{sim_state.model.cols}")
            
        # Calculate Angles for each sample point
        angles = np.zeros_like(xs)
        if len(xs) >= 2:
            dxs = np.gradient(xs)
            dys = np.gradient(ys)
            angles = np.arctan2(dys, dxs)
        
        # Deduplicate (r, c) pairs to avoid adding same cell multiple times for same edge
        # But we need to map each unique cell to the corresponding 1D segment
        unique_cells = {} # (r, c) -> {'ts': [], 'angles': []}
        
        for i in range(len(rows)):
            r, c = rows[i], cols[i]
            if (r, c) not in unique_cells:
                unique_cells[(r, c)] = {'ts': [], 'angles': []}
            unique_cells[(r, c)]['ts'].append(ts[i])
            unique_cells[(r, c)]['angles'].append(angles[i])
            
        # Create Coupler Nodes
        for (r, c), data in unique_cells.items():
            t_list = data['ts']
            angle_list = data['angles']
            # Check validity
            if 0 <= r < sim_state.model.rows and 0 <= c < sim_state.model.cols:
                # Check if 2D cell is active (not masked)
                if sim_state.nodata_mask is not None and sim_state.nodata_mask[r, c]:
                    continue
                
                # NOTE: Skip inactive cells (bc_type == -1) to strictly enforce AOI.
                # If we don't couple, no water flows from 1D to 2D outside AOI.
                if sim_state.bc_type_grid is not None and sim_state.bc_type_grid[r, c] == -1:
                   # Identify 1D cell and mark as inactive
                   t_avg = sum(t_list) / len(t_list)
                   idx_1d = int(t_avg * edge.model.num_cells)
                   if idx_1d >= edge.model.num_cells: idx_1d = edge.model.num_cells - 1
                   sim_state.inactive_1d_nodes.append((edge.model, idx_1d))
                   
                   skipped_mask += 1
                   continue
                
                # Check distance (Map Check) - Optional but good for safety
                if max_map_dist > 0:
                    x_cell, y_cell = rasterio.transform.xy(fwd, r, c, offset='center')
                    # Find closest point on segment
                    t_avg = sum(t_list) / len(t_list)
                    px = x1 + t_avg * (x2 - x1)
                    py = y1 + t_avg * (y2 - y1)
                    if float(np.hypot(float(x_cell) - float(px), float(y_cell) - float(py))) > max_map_dist:
                        skipped_map += 1
                        continue

                # Map to 1D Model Cell
                # Use average t to find the "center" of this interaction on the 1D line
                t_avg = sum(t_list) / len(t_list)
                
                # 1D Model Cell Index
                idx_1d = int(t_avg * edge.model.num_cells)
                if idx_1d >= edge.model.num_cells: idx_1d = edge.model.num_cells - 1
                
                # Calculate Interaction Length (Effective Width)
                # Use 2x cell size to account for both banks spilling
                contact_len = 2.0 * min(dx_2d, dy_2d)
                
                z_1d = float(edge.model.z[idx_1d])
                z_2d = float(sim_state.elevation[r, c])
                
                # River Burn-in Logic
                # If 2D terrain is higher than 1D bed, it blocks flow.
                # We force 2D terrain to be at most 1D Bed (or slightly lower) to ensure connectivity.
                # Only apply if we are at the start of simulation (time=0) to avoid jumping during run.
                # AND only if cell is ACTIVE (bc_type != -1)
                is_active = True
                if sim_state.bc_type_grid is not None and sim_state.bc_type_grid[r, c] == -1:
                    is_active = False

                if sim_state.current_time == 0.0 and is_active:
                    # [DISABLED] Burn-in causes water accumulation in trenches (spurious nodes).
                    # We disable it by default. The user should provide a hydro-enforced DEM.
                    pass
                    # burn_depth = 0.1 # Ensure it's slightly lower than 1D bed
                    # if z_2d > z_1d:
                    #     # Limit burn-in depth to avoid deep canyons causing numerical instability
                    #     # If difference is > 50m (increased from 10m), assume it's a tunnel/pipe or mistake, and don't burn.
                    #     if (z_2d - z_1d) < 50.0:
                    #         new_z = z_1d - burn_depth
                    #         sim_state.elevation[r, c] = new_z
                    #         z_2d = new_z
                    #         modified_terrain = True
                    #     else:
                    #         print(f"[WARNING] Burn-in skipped at ({r}, {c}): 2D={z_2d:.2f}, 1D={z_1d:.2f}, Diff={z_2d-z_1d:.2f} > 50.0")
                
                # Use max(1D_bed, 2D_bed) as crest
                crest = max(z_1d, z_2d) + bank_height
                
                # Calculate Average Flow Angle (Vector Averaging)
                sin_sum = np.sum(np.sin(angle_list))
                cos_sum = np.sum(np.cos(angle_list))
                avg_angle = np.arctan2(sin_sum, cos_sum)
                
                # Check if this node maps to an Outlet
                is_outlet = False
                if idx_1d >= edge.model.num_cells - 1:
                     if edge.end_node_id in outlet_node_ids:
                         is_outlet = True

                # [FIX] Overlap Detection
                if (r, c) in coupled_cells_map:
                    existing_eids = coupled_cells_map[(r, c)]
                    for exist_eid in existing_eids:
                        # Check if edges are connected topologically
                        exist_edge = sim_state.network.edges[exist_eid]
                        # Connected if they share ANY node
                        is_connected = (edge.start_node_id == exist_edge.start_node_id or
                                        edge.start_node_id == exist_edge.end_node_id or
                                        edge.end_node_id == exist_edge.start_node_id or
                                        edge.end_node_id == exist_edge.end_node_id)
                        
                        if not is_connected:
                            print(f"[WARNING] SPATIAL OVERLAP DETECTED at ({r}, {c})!")
                            print(f"          Edge {eid} overlaps with Edge {exist_eid} but they are NOT connected.")
                            print(f"          This will cause 'Ghost Water' (teleportation). Check Polylines for crossing/touching.")
                else:
                    coupled_cells_map[(r, c)] = set()
                
                coupled_cells_map[(r, c)].add(eid)

                sim_state.coupler.add_node(
                    idx_1d=idx_1d,
                    r_2d=r,
                    c_2d=c,
                    type='weir',
                    model_1d=edge.model,
                    flow_angle=avg_angle, # Store angle
                    lateral=True, 
                    is_outlet=is_outlet, # Pass outlet flag
                    width=contact_len, # Contact length for flux calculation
                    channel_width=edge.model.w[idx_1d],
                    crest_level=crest,
                    coeff=0.4,
                    nx=nx_seg,
                    ny=ny_seg
                )
                connections_count += 1
                
    if skipped_mask or skipped_map:
        print(f"Coupling Initialized: {connections_count} connections created. Skipped masked={skipped_mask}, skipped mapping={skipped_map}.")
    else:
        print(f"Coupling Initialized: {connections_count} connections created.")

    # 3. Explicitly scan ALL 1D Nodes to mark those outside AOI as Inactive
    # This is more robust than edge sampling, ensuring no 1D node is missed.
    if sim_state.bc_type_grid is not None and not sim_state.disable_1d:
        print("[INFO] Scanning 1D nodes against AOI mask...")
        inactive_node_count = 0
        
        for eid, edge in sim_state.network.edges.items():
            model_1d = edge.model
            if not model_1d: continue
            
            # We need the geometry to map 1D index -> (x, y)
            # If geometry is available (lat/lon), we can interpolate
            # If not, we use straight line.
            
            # Re-calculate geometry points (xs, ys, dists)
            # This duplicates some logic but is safer to be self-contained
            
            # ... Or we can just use the sampled 'unique_cells' if we trust it?
            # No, 'unique_cells' is 2D-centric.
            # We need to iterate 0..num_cells-1
            
            # Generate geometry trace
            xs_trace = []
            ys_trace = []
            
            n_start = sim_state.network.nodes[edge.start_node_id]
            n_end = sim_state.network.nodes[edge.end_node_id]
            x1, y1 = n_start.x, n_start.y
            x2, y2 = n_end.x, n_end.y
            
            # Try polyline
            use_poly = False
            if edge.geometry and len(edge.geometry) >= 2:
                try:
                    g_lons = [p[1] for p in edge.geometry]
                    g_lats = [p[0] for p in edge.geometry]
                    target_crs = sim_state.crs if sim_state.crs else CRS.from_epsg(3857)
                    xs_poly, ys_poly = warp.transform(CRS.from_epsg(4326), target_crs, g_lons, g_lats)
                    
                    points_poly = list(zip(xs_poly, ys_poly))
                    dists_poly = [0.0]
                    total_len = 0.0
                    for i in range(len(points_poly)-1):
                        d = np.hypot(points_poly[i+1][0] - points_poly[i][0], points_poly[i+1][1] - points_poly[i][1])
                        total_len += d
                        dists_poly.append(total_len)
                    
                    if total_len > 1e-6:
                         # Interpolate for each node center
                         # Node i center is at (i + 0.5) * dx
                         dx_1d = model_1d.dx
                         for i in range(model_1d.num_cells):
                             dist_target = (i + 0.5) * dx_1d
                             # Clamp
                             if dist_target > total_len: dist_target = total_len
                             
                             xi = np.interp(dist_target, dists_poly, xs_poly)
                             yi = np.interp(dist_target, dists_poly, ys_poly)
                             xs_trace.append(xi)
                             ys_trace.append(yi)
                         use_poly = True
                except:
                    pass
            
            if not use_poly:
                # Straight line
                total_len = np.hypot(x2-x1, y2-y1)
                dx_1d = model_1d.dx
                if total_len > 1e-6:
                    for i in range(model_1d.num_cells):
                        dist_target = (i + 0.5) * dx_1d
                        t = dist_target / total_len
                        xi = x1 + t * (x2 - x1)
                        yi = y1 + t * (y2 - y1)
                        xs_trace.append(xi)
                        ys_trace.append(yi)
            
            # Now check these points against bc_type_grid
            if xs_trace:
                rows_idx, cols_idx = rasterio.transform.rowcol(sim_state.transform, xs_trace, ys_trace)
                
                # Debug: Log first few nodes of the first few edges
                if inactive_node_count == 0 and len(rows_idx) > 0:
                    print(f"[DEBUG] Edge {eid} Trace Check:")
                    for i in range(min(5, len(rows_idx))):
                        r, c = rows_idx[i], cols_idx[i]
                        val = -999
                        if 0 <= r < sim_state.model.rows and 0 <= c < sim_state.model.cols:
                            val = sim_state.bc_type_grid[r, c]
                        print(f"  Node {i}: (x={xs_trace[i]:.2f}, y={ys_trace[i]:.2f}) -> (r={r}, c={c}), bc_type={val}")

                for i in range(len(rows_idx)):
                    r, c = rows_idx[i], cols_idx[i]
                    if 0 <= r < sim_state.model.rows and 0 <= c < sim_state.model.cols:
                        if sim_state.bc_type_grid[r, c] == -1:
                            # Mark Inactive
                            # Check if already added? List is fine, we loop.
                            sim_state.inactive_1d_nodes.append((model_1d, i))
                            inactive_node_count += 1
                            
        print(f"[INFO] Marked {inactive_node_count} 1D nodes as Inactive (outside AOI).")

    # Re-initialize model if terrain was modified (Burn-in)
    if modified_terrain and sim_state.current_time == 0.0:
        print("[INFO] River Burn-in applied. Re-initializing 2D Model with modified terrain.")
        # Re-init model with new elevation
        # Note: This resets h, u, v to 0.0. 
        # Since we are at t=0, this is expected (unless user had custom initial water).
        # Assuming standard workflow: Load DEM -> Create Network -> Start.
        sim_state.model.init(sim_state.elevation, sim_state.roughness, sim_state.dx, sim_state.dy)


@app.post("/start")
async def start_simulation(dt_save: float = 10.0, total_time: float = 3600.0, coupling_cd: float = 0.4, coupling_method: str = "empirical", bank_height: float = 2.0):
    if sim_state.model is None and sim_state.network is None:
        raise HTTPException(status_code=400, detail="Model or Network not initialized")
    
    if sim_state.is_running:
        return {"status": "already running"}
        
    sim_state.dt_save = dt_save
    sim_state.total_time = total_time
    sim_state.is_running = True

    # Restore Original Elevation to prevent accumulation of Burn-in trenches
    if sim_state.original_elevation is not None:
        print("[INFO] Restoring Original Elevation before Burn-in...")
        sim_state.elevation = sim_state.original_elevation.copy()
        # Note: Model will be re-initialized in initialize_network_coupling if burn-in happens.
        # If no burn-in, we should re-init here?
        # Ideally, we should always ensure model has correct elevation.
        # But initialize_network_coupling handles it.
    
    # Update coupling parameters
    # Always re-initialize coupling for Network to respect new bank_height
    if sim_state.network and sim_state.model and not sim_state.disable_1d:
        initialize_network_coupling(bank_height=bank_height)
    # For legacy single channel (no network), we keep existing coupler if present
    elif sim_state.coupler is None and sim_state.model:
        # Should not happen for legacy as it creates coupler on creation
        pass

    if sim_state.coupler:
        sim_state.coupler.set_global_cd(coupling_cd)
        sim_state.coupler.set_method(coupling_method)
        print(f"[INFO] Coupling Method: {coupling_method}, Cd: {coupling_cd}")

    # Start background task
    asyncio.create_task(simulation_loop())
    
    return {"status": "started", "dt_save": dt_save, "total_time": total_time, "coupling_cd": coupling_cd, "coupling_method": coupling_method}

@app.post("/stop")
async def stop_simulation():
    if not sim_state.is_running:
        return {"status": "not running"}
        
    sim_state.is_running = False
    return {"status": "stopped", "current_time": sim_state.current_time}

@app.post("/step")
async def step():
    if sim_state.model is None:
        raise HTTPException(status_code=400, detail="Model not initialized")
    
    # Run 5 steps for speed
    with sim_state.lock:
        for _ in range(5):
            sim_state.model.step(0.1)
    
    return {"status": "stepped"}

@app.get("/view/elevation")
async def view_elevation(max_dim: int = 800):
    if sim_state.elevation is None:
        raise HTTPException(status_code=400, detail="No elevation data")
    
    # Prepare visualization data
    elev_to_plot = sim_state.elevation
    
    # Downsample if too large
    rows, cols = elev_to_plot.shape
    scale = 1.0
    if max(rows, cols) > max_dim:
        scale = max_dim / max(rows, cols)
        new_rows, new_cols = int(rows * scale), int(cols * scale)
        # Simple striding for speed (or use scipy.ndimage.zoom for quality)
        # Striding: elev_to_plot[::step, ::step]
        step = int(1/scale)
        if step < 1: step = 1
        elev_to_plot = elev_to_plot[::step, ::step]
        
        # Also downsample mask
        mask_to_plot = None
        if sim_state.nodata_mask is not None:
             mask_to_plot = sim_state.nodata_mask[::step, ::step]
    else:
        mask_to_plot = sim_state.nodata_mask

    if mask_to_plot is not None:
        elev_to_plot = np.ma.masked_where(mask_to_plot, elev_to_plot)
    
    # Render to PNG
    plt.figure(figsize=(8, 4))
    plt.imshow(elev_to_plot, cmap='terrain', origin='upper')
    plt.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

def render_map_image(h, elev, nodata_mask, max_dim=800, show_terrain=True, show_water=True):
    def downsample_max_2d(arr, step):
        if step <= 1:
            return arr
        r, c = arr.shape
        r2 = int(np.ceil(r / step) * step)
        c2 = int(np.ceil(c / step) * step)
        pad_r = r2 - r
        pad_c = c2 - c
        if pad_r or pad_c:
            arr = np.pad(arr, ((0, pad_r), (0, pad_c)), mode='constant', constant_values=0)
        return arr.reshape(r2 // step, step, c2 // step, step).max(axis=(1, 3))

    def downsample_mean_2d(arr, step):
        if step <= 1:
            return arr
        r, c = arr.shape
        r2 = int(np.ceil(r / step) * step)
        c2 = int(np.ceil(c / step) * step)
        pad_r = r2 - r
        pad_c = c2 - c
        if pad_r or pad_c:
            arr = np.pad(arr, ((0, pad_r), (0, pad_c)), mode='edge')
        return arr.reshape(r2 // step, step, c2 // step, step).mean(axis=(1, 3))

    rows, cols = h.shape
    step = 1
    if max(rows, cols) > max_dim:
        step = int(max(rows, cols) / max_dim)
        if step < 1:
            step = 1

    if show_water:
        h_safe = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        if step > 1:
            h_safe = downsample_max_2d(h_safe, step)
    else:
        h_safe = None

    if show_terrain:
        elev_ds = elev
        if step > 1:
            elev_ds = downsample_mean_2d(elev_ds, step)
    else:
        elev_ds = elev

    if nodata_mask is not None:
        mask_ds = nodata_mask
        if step > 1:
            mask_ds = downsample_max_2d(np.asarray(mask_ds, dtype=np.uint8), step).astype(bool)
            
        # [FIX] Ensure mask_ds shape matches h_safe/water_rgba shape exactly
        # downsample logic might produce slightly different shapes due to padding/reshape
        if show_water and h_safe is not None:
            target_shape = h_safe.shape
            if mask_ds.shape != target_shape:
                # Resize mask to match target shape if different
                # This can happen if nodata_mask has different initial dimensions or downsampling behaves differently
                # But here we assume nodata_mask has same initial dims as h.
                # If step > 1, downsample might have yielded different size if h and mask were processed differently?
                # No, they use same function.
                # HOWEVER, if h and mask have different input shapes?
                # In view_water_dynamic, h_sub and mask_sub are sliced from same indices.
                # BUT, if mask_sub is None, or sliced differently?
                
                # Let's crop or pad mask_ds to match h_safe
                mr, mc = mask_ds.shape
                tr, tc = target_shape
                
                if mr > tr: mask_ds = mask_ds[:tr, :]
                elif mr < tr: pass # Should not happen with same downsample logic unless input differed
                
                if mc > tc: mask_ds = mask_ds[:, :tc]
                elif mc < tc: pass
                
                # If still mismatch (e.g. smaller), we might need to be more aggressive or just ignore mask for those pixels
                if mask_ds.shape != target_shape:
                     # Create a new mask of correct shape
                     new_mask = np.zeros(target_shape, dtype=bool)
                     r_lim = min(mr, tr)
                     c_lim = min(mc, tc)
                     new_mask[:r_lim, :c_lim] = mask_ds[:r_lim, :c_lim]
                     mask_ds = new_mask

    else:
        mask_ds = None
    
    # --- Render Water ---
    if show_water:
        # Initialize water_rgba with zeros
        water_rgba = np.zeros((h_safe.shape[0], h_safe.shape[1], 4), dtype=np.uint8)
        
        # Custom Discrete Colormap
        # 0 ~ 0.5m: Light Blue (#87CEEB) -> [135, 206, 235]
        # 0.5 ~ 1m: Blue       (#00BFFF) -> [0, 191, 255]
        # 1 ~ 2m:   Dark Blue  (#0000CD) -> [0, 0, 205]
        # 2 ~ 3m:   Navy Blue  (#00008B) -> [0, 0, 139]
        # > 3m:     Purple Blue(#4B0082) -> [75, 0, 130]
        
        alpha = 220 # Visibility
        
        # We process from Deep to Shallow (order doesn't strictly matter with masks)
        
        # > 3m
        mask = h_safe > 3.0
        water_rgba[mask] = [75, 0, 130, alpha]
        
        # 2 ~ 3m
        mask = (h_safe > 2.0) & (h_safe <= 3.0)
        water_rgba[mask] = [0, 0, 139, alpha]
        
        # 1 ~ 2m
        mask = (h_safe > 1.0) & (h_safe <= 2.0)
        water_rgba[mask] = [0, 0, 205, alpha]
        
        # 0.5 ~ 1m
        mask = (h_safe > 0.5) & (h_safe <= 1.0)
        water_rgba[mask] = [0, 191, 255, alpha]
        
        # 0.05 ~ 0.5m
        mask = (h_safe > WET_THRESHOLD) & (h_safe <= 0.5)
        water_rgba[mask] = [135, 206, 235, alpha]

        # 0.001 ~ 0.05m
        mask = (h_safe > DRY_THRESHOLD) & (h_safe <= WET_THRESHOLD)
        water_rgba[mask] = [173, 216, 230, 160]
        
    else:
        water_rgba = np.zeros((elev_ds.shape[0], elev_ds.shape[1], 4), dtype=np.uint8)
    
    if not show_terrain:
        # If no terrain, just return water layer
        out_rgba = water_rgba.copy()
        
        # Apply NoData mask (make fully transparent)
        if mask_ds is not None:
            out_rgba[mask_ds] = 0
            
        img = Image.fromarray(out_rgba, 'RGBA')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return buf

    # --- Render Terrain ---
    # Sanitize elev for display
    elev_safe = np.nan_to_num(elev_ds, nan=np.nanmin(elev_ds) if not np.isnan(np.nanmin(elev_ds)) else 0)
    
    # Normalize elevation to 0-255 for LUT
    e_min = np.min(elev_safe)
    e_max = np.max(elev_safe)
    if e_max > e_min:
        elev_norm = ((elev_safe - e_min) / (e_max - e_min) * 255).astype(np.uint8)
    else:
        elev_norm = np.zeros_like(elev_safe, dtype=np.uint8)
    
    terrain_rgba = terrain_lut[elev_norm] # (rows, cols, 4)
    
    # --- Composite ---
    t_rgb = terrain_rgba[:, :, :3].astype(np.float32)
    w_rgb = water_rgba[:, :, :3].astype(np.float32)
    w_a = (water_rgba[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]
    
    out_rgb = w_rgb * w_a + t_rgb * (1.0 - w_a)
    
    out_a = np.full((water_rgba.shape[0], water_rgba.shape[1], 1), 255, dtype=np.uint8)
    
    if mask_ds is not None:
        out_a[mask_ds] = 0
    
    out_rgba = np.concatenate([out_rgb.astype(np.uint8), out_a], axis=2)
    
    img = Image.fromarray(out_rgba, 'RGBA')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf

@app.get("/debug/gap_stats")
async def debug_gap_stats(bbox: str | None = None):
    if sim_state.model is None:
        raise HTTPException(status_code=400, detail="Model not initialized")

    h_full, _, _ = sim_state.model.get_results()
    bc = sim_state.bc_type_grid
    nodata = sim_state.nodata_mask

    h = h_full
    bc_sub = bc
    nodata_sub = nodata

    if bbox is not None:
        try:
            min_lon, min_lat, max_lon, max_lat = map(float, bbox.split(','))
        except:
            raise HTTPException(status_code=400, detail="Invalid bbox format")

        transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
        min_x, min_y = transformer.transform(min_lon, min_lat)
        max_x, max_y = transformer.transform(max_lon, max_lat)

        row_min, col_min = rasterio.transform.rowcol(sim_state.transform, min_x, max_y)
        row_max, col_max = rasterio.transform.rowcol(sim_state.transform, max_x, min_y)

        full_rows, full_cols = sim_state.elevation.shape
        r_start = max(0, min(row_min, row_max))
        r_stop = min(full_rows, max(row_min, row_max))
        c_start = max(0, min(col_min, col_max))
        c_stop = min(full_cols, max(col_min, col_max))

        if r_start >= r_stop or c_start >= c_stop:
            return {"shape": [0, 0], "total": 0}

        h = h_full[r_start:r_stop, c_start:c_stop]
        bc_sub = bc[r_start:r_stop, c_start:c_stop]
        if nodata is not None:
            nodata_sub = nodata[r_start:r_stop, c_start:c_stop]

    h_safe = np.asarray(h, dtype=np.float64)
    nan_mask = np.isnan(h_safe)
    neg_mask = h_safe < 0
    h_safe = np.nan_to_num(h_safe, nan=0.0, posinf=0.0, neginf=0.0)

    total = int(h_safe.size)
    wet_gt_dry = int(np.sum(h_safe > DRY_THRESHOLD))
    wet_gt_wet = int(np.sum(h_safe > WET_THRESHOLD))
    shallow = int(np.sum((h_safe > DRY_THRESHOLD) & (h_safe <= WET_THRESHOLD)))
    tiny = int(np.sum((h_safe > 0.0) & (h_safe <= DRY_THRESHOLD)))

    bc_inactive = int(np.sum(bc_sub == -1)) if bc_sub is not None else 0
    bc_inflow = int(np.sum(bc_sub == 1)) if bc_sub is not None else 0
    bc_outflow = int(np.sum(bc_sub == 2)) if bc_sub is not None else 0

    nodata_count = int(np.sum(nodata_sub)) if nodata_sub is not None else 0

    return {
        "shape": [int(h_safe.shape[0]), int(h_safe.shape[1])],
        "total": total,
        "h_nan": int(np.sum(nan_mask)),
        "h_neg": int(np.sum(neg_mask)),
        "wet_gt_dry": wet_gt_dry,
        "wet_gt_wet": wet_gt_wet,
        "shallow": shallow,
        "tiny": tiny,
        "bc_inactive": bc_inactive,
        "bc_inflow": bc_inflow,
        "bc_outflow": bc_outflow,
        "nodata": nodata_count,
    }

@app.get("/view/water")
async def view_water(max_dim: int = 800, show_terrain: bool = True, show_water: bool = True):
    """
    Render map image. 
    Returns raw PNG data.
    """
    if sim_state.model is None:
        raise HTTPException(status_code=400, detail="Model not initialized")
        
    # Get Water Depth (h)
    with sim_state.lock:
        h, _, _ = sim_state.model.get_results()
    elev = sim_state.elevation
    
    # --- Overlay 1D Water for Visualization ---
    # This ensures the 1D river channel is visible even if it hasn't spilled to 2D
    h = overlay_1d_results(h)
    
    # Force AOI Mask
    h = apply_aoi_mask(h)
    
    # Ensure h is purely depth (already is from get_results)
    # If user asks: "Does this include terrain?" -> No, h is depth above terrain.
    # WSE = h + elev
    
    buf = render_map_image(h, elev, sim_state.nodata_mask, max_dim, show_terrain, show_water)
    return StreamingResponse(buf, media_type="image/png")

@app.get("/view/water_dynamic")
async def view_water_dynamic(bbox: str, width: int = 800, height: int = 600, show_terrain: bool = True, show_water: bool = True):
    """
    Render water/terrain for a specific bounding box (Dynamic Viewport).
    bbox: "min_lon,min_lat,max_lon,max_lat"
    """
    if sim_state.model is None:
        raise HTTPException(status_code=400, detail="Model not initialized")
    
    # Parse BBox
    try:
        min_lon, min_lat, max_lon, max_lat = map(float, bbox.split(','))
    except:
        raise HTTPException(status_code=400, detail="Invalid bbox format")

    # Transform BBox to Grid CRS (Assuming sim_state.transform is in Projected CRS, usually 3857)
    # Note: sim_state.crs might not be set explicitly, but usually it's 3857 for web maps.
    # Let's assume input is 4326, and we need to transform to 3857 to find grid indices.
    
    # Create transformer 4326 -> 3857
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    min_x, min_y = transformer.transform(min_lon, min_lat)
    max_x, max_y = transformer.transform(max_lon, max_lat)
    
    # Handle Y-axis direction (Map Y increases North, Grid Y might increase South)
    # rasterio.transform.rowcol handles this if transform is correct.
    
    # Get Grid Window
    # rowcol returns (row, col)
    row_min, col_min = rasterio.transform.rowcol(sim_state.transform, min_x, max_y) # Top-Left
    row_max, col_max = rasterio.transform.rowcol(sim_state.transform, max_x, min_y) # Bottom-Right
    
    # Ensure indices are within bounds
    full_rows, full_cols = sim_state.elevation.shape
    
    r_start = max(0, min(row_min, row_max))
    r_stop = min(full_rows, max(row_min, row_max))
    c_start = max(0, min(col_min, col_max))
    c_stop = min(full_cols, max(col_min, col_max))
    
    if r_start >= r_stop or c_start >= c_stop:
        # Outside of domain
        # Return transparent image
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
        
    # Calculate Actual Bounds of the window (to return to client)
    # transform * (col, row) -> (x, y) (Top-Left of pixel)
    # Top-Left of window
    x_min_act, y_max_act = sim_state.transform * (c_start, r_start)
    # Bottom-Right of window
    x_max_act, y_min_act = sim_state.transform * (c_stop, r_stop)
    
    # Transform back to 4326 for client
    inv_transformer = Transformer.from_crs("epsg:3857", "epsg:4326", always_xy=True)
    lon_min_act, lat_min_act = inv_transformer.transform(x_min_act, y_min_act)
    lon_max_act, lat_max_act = inv_transformer.transform(x_max_act, y_max_act)
    
    # Slice Data
    # Read h from model if running
    try:
        if sim_state.model:
             with sim_state.lock:
                 h_full, _, _ = sim_state.model.get_results()
        else:
             h_full = np.zeros_like(sim_state.elevation)
    except Exception as e:
        print(f"[Server] Error fetching results: {e}")
        h_full = np.zeros_like(sim_state.elevation)
    
    # Overlay 1D results for consistency (Critical for Zoom-in view)
    h_full = overlay_1d_results(h_full)
    
    # Force AOI Mask
    h_full = apply_aoi_mask(h_full)
    
    h_sub = h_full[r_start:r_stop, c_start:c_stop]
    elev_sub = sim_state.elevation[r_start:r_stop, c_start:c_stop]
    mask_sub = None
    if sim_state.nodata_mask is not None:
        mask_sub = sim_state.nodata_mask[r_start:r_stop, c_start:c_stop]
        
    # Render
    # We pass max_dim as max(width, height) to allow full resolution if needed
    # But render_map_image downsamples if data > max_dim.
    # Here we want to match screen resolution roughly.
    buf = render_map_image(h_sub, elev_sub, mask_sub, max_dim=max(width, height), show_terrain=show_terrain, show_water=show_water)
    
    headers = {
        "X-Image-Bounds": f"{lat_min_act},{lon_min_act},{lat_max_act},{lon_max_act}"
    }
    
    return StreamingResponse(buf, media_type="image/png", headers=headers)


def extract_velocity_vectors(h, u, v, transform, stride=10, min_h=0.01):
    rows, cols = h.shape
    vectors = []
    
    # Create grid of indices
    # We want center of cells
    r_idx = np.arange(0, rows, stride)
    c_idx = np.arange(0, cols, stride)
    
    # Meshgrid
    C, R = np.meshgrid(c_idx, r_idx)
    
    # Filter by depth
    mask = h[R, C] > min_h
    
    valid_r = R[mask]
    valid_c = C[mask]
    valid_u = u[valid_r, valid_c]
    valid_v = v[valid_r, valid_c]
    
    if len(valid_u) == 0:
        return []
        
    # Transform to Lat/Lon
    # transform * (c, r) gives x, y (projected)
    # rasterio.transform.xy returns x, y for center of pixel if offset='center'
    xs, ys = rasterio.transform.xy(transform, valid_r, valid_c, offset='center')
    
    # To Lat/Lon
    lons, lats = warp.transform(CRS.from_epsg(3857), CRS.from_epsg(4326), xs, ys)
    
    magnitudes = np.sqrt(valid_u**2 + valid_v**2)
    
    # Determine Y-direction mapping based on Geotransform
    # transform[4] (e) is pixel height.
    # If e < 0 (Standard North-Up), Row index increases Southwards.
    #   So Grid V (>0) means South. Map V (Northing) = -Grid V.
    # If e > 0 (South-Up or Cartesian), Row index increases Northwards.
    #   So Grid V (>0) means North. Map V (Northing) = Grid V.
    
    pixel_height = transform[4]
    
    # Check if we need to invert V for Map Angle calculation
    # Default to inverting (Standard GeoTIFF) if pixel_height is close to 0 or negative
    if pixel_height < 0:
        v_map = -valid_v
    else:
        v_map = valid_v
        
    angles = np.arctan2(v_map, valid_u) # Radians
    
    # Normalize data for JSON
    for i in range(len(lats)):
        vectors.append({
            "lat": lats[i],
            "lon": lons[i],
            "u": float(valid_u[i]),
            "v": float(v_map[i]), # Return Northing velocity for consistency
            "m": float(magnitudes[i]),
            "a": float(angles[i]) # Radians
        })
        
    return vectors

@app.get("/view/velocity")
async def view_velocity(stride: int = 10):
    if sim_state.model is None:
        return []
    with sim_state.lock:
        h, u, v = sim_state.model.get_results()
    # Check if u, v are valid (might be None if model doesn't support it)
    if u is None or v is None:
        return []

    # --- INJECT 1D VELOCITY ---
    if hasattr(sim_state, 'coupler') and sim_state.coupler:
        u = u.copy()
        v = v.copy() # Avoid modifying simulation state
        
        # We need to know pixel height sign to handle V direction correctly
        pixel_height = sim_state.transform[4]
        is_north_up = pixel_height < 0

        for node in sim_state.coupler.nodes:
            # Only if node has stored angle
            if 'flow_angle' in node.params:
                angle = node.params['flow_angle']
                r, c = node.r_2d, node.c_2d
                
                if 0 <= r < u.shape[0] and 0 <= c < u.shape[1]:
                     # Get 1D Velocity
                     if node.model_1d:
                         # Use node.idx_1d to get velocity
                         if hasattr(node.model_1d, 'u') and node.idx_1d < len(node.model_1d.u):
                             u_1d = float(node.model_1d.u[node.idx_1d])
                             
                             # Map 1D U (along channel) to 2D Grid Components
                             # Angle is relative to East (EPSG:3857 X)
                             u_east = u_1d * np.cos(angle)
                             u_north = u_1d * np.sin(angle)
                             
                             # Set Grid U (East)
                             u[r, c] = u_east
                             
                             # Set Grid V (Row Direction)
                             if is_north_up:
                                 # Grid V is South (+), North (-)
                                 # Map V (North) = u_north
                                 # Grid V = -Map V
                                 v[r, c] = -u_north
                             else:
                                 # Grid V is North (+)
                                 v[r, c] = u_north
    # ---------------------------
        
    # Apply AOI Mask to u, v
    h = apply_aoi_mask(h)
    u = apply_aoi_mask(u)
    v = apply_aoi_mask(v)
             
    return extract_velocity_vectors(h, u, v, sim_state.transform, stride)

@app.get("/view/result/{index}/velocity")
async def view_result_velocity(index: int, stride: int = 10):
    filename = f"results/rst_{index}.tif"
    if not os.path.exists(filename):
         return []
    
    try:
        with rasterio.open(filename) as src:
            if src.count < 3:
                return []
            h = src.read(1)
            u = src.read(2)
            v = src.read(3)
            transform = src.transform
            
            # Apply AOI Mask
            h = apply_aoi_mask(h)
            u = apply_aoi_mask(u)
            v = apply_aoi_mask(v)
            
            return extract_velocity_vectors(h, u, v, transform, stride)
    except:
        return []

@app.get("/view/1d_geometry")
async def view_1d_geometry():
    """Return 1D channel geometry (nodes)"""
    if sim_state.geometry_1d is not None:
        return {"geometry": sim_state.geometry_1d}
    
    # Fallback for Network: Concatenate edge geometries
    if sim_state.network is not None:
        geo_list = []
        try:
            sorted_edges = sorted(sim_state.network.edges.values(), key=lambda e: int(e.id))
        except:
            sorted_edges = sorted(sim_state.network.edges.values(), key=lambda e: str(e.id))
            
        for edge in sorted_edges:
            if edge.geometry:
                geo_list.extend(edge.geometry)
        
        if geo_list:
            return {"geometry": geo_list}

    return {"geometry": []}

@app.get("/view/1d_results")
async def get_1d_results():
    # Helper to sanitize arrays
    def sanitize(arr, expected_len=None):
        if arr is None:
            if expected_len: return np.zeros(expected_len)
            return np.array([])
        
        arr = np.asarray(arr)
        if arr.ndim == 0:
            if expected_len: return np.full(expected_len, arr)
            return np.array([arr])
            
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Legacy Single Model
    if sim_state.model_1d is not None:
        h, u = sim_state.model_1d.get_results()
        z = sim_state.model_1d.z
        
        # Ensure z matches h length
        n = len(h) if hasattr(h, '__len__') else 0
        
        return {
            "h": sanitize(h).tolist(),
            "u": sanitize(u).tolist(),
            "z": sanitize(z, n).tolist()
        }
    
    # Network 1D
    if sim_state.network is not None:
        h_list = []
        u_list = []
        z_list = []
        
        # Sort edges by ID to try and maintain some order
        try:
            sorted_edges = sorted(sim_state.network.edges.values(), key=lambda e: int(e.id))
        except:
            sorted_edges = sorted(sim_state.network.edges.values(), key=lambda e: str(e.id))
        
        for edge in sorted_edges:
            h_e, u_e = edge.model.get_results()
            z_e = edge.model.z
            
            # Sanitize and consistency check
            # We need to know the number of cells.
            n_cells = edge.model.num_cells
            
            h_e = sanitize(h_e, n_cells)
            u_e = sanitize(u_e, n_cells)
            z_e = sanitize(z_e, n_cells)
            
            h_list.append(h_e)
            u_list.append(u_e)
            z_list.append(z_e)
            
        if h_list:
            return {
                "h": np.concatenate(h_list).tolist(),
                "u": np.concatenate(u_list).tolist(),
                "z": np.concatenate(z_list).tolist()
            }
            
    return {"h": [], "u": [], "z": []}

@app.get("/results/count")
async def get_results_count():
    return {"count": sim_state.save_index}

@app.get("/view/result/{index}")
async def view_result(index: int, max_dim: int = 800, show_terrain: bool = True, show_water: bool = True):
    if sim_state.elevation is None:
        raise HTTPException(status_code=400, detail="Model not initialized")
    
    # Read result file
    filename = f"results/rst_{index}.tif"
    if not os.path.exists(filename):
         raise HTTPException(status_code=404, detail="Result not found")
         
    try:
        with rasterio.open(filename) as src:
            h = src.read(1) # Band 1 is depth
            
        # Force AOI Mask (even for loaded results)
        h = apply_aoi_mask(h)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading result: {e}")
        
    buf = render_map_image(h, sim_state.elevation, sim_state.nodata_mask, max_dim, show_terrain, show_water)
    return StreamingResponse(buf, media_type="image/png")

@app.get("/view/result/{index}/dynamic")
async def view_result_dynamic(index: int, bbox: str, width: int = 800, height: int = 600, show_terrain: bool = True, show_water: bool = True):
    if sim_state.elevation is None:
        raise HTTPException(status_code=400, detail="Model not initialized")
    
    filename = f"results/rst_{index}.tif"
    if not os.path.exists(filename):
         raise HTTPException(status_code=404, detail="Result not found")

    # Parse BBox
    try:
        min_lon, min_lat, max_lon, max_lat = map(float, bbox.split(','))
    except:
        raise HTTPException(status_code=400, detail="Invalid bbox format")

    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    min_x, min_y = transformer.transform(min_lon, min_lat)
    max_x, max_y = transformer.transform(max_lon, max_lat)
    
    row_min, col_min = rasterio.transform.rowcol(sim_state.transform, min_x, max_y)
    row_max, col_max = rasterio.transform.rowcol(sim_state.transform, max_x, min_y)
    
    full_rows, full_cols = sim_state.elevation.shape
    r_start = max(0, min(row_min, row_max))
    r_stop = min(full_rows, max(row_min, row_max))
    c_start = max(0, min(col_min, col_max))
    c_stop = min(full_cols, max(col_min, col_max))
    
    if r_start >= r_stop or c_start >= c_stop:
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
        
    # Calculate Actual Bounds
    x_min_act, y_max_act = sim_state.transform * (c_start, r_start)
    x_max_act, y_min_act = sim_state.transform * (c_stop, r_stop)
    
    inv_transformer = Transformer.from_crs("epsg:3857", "epsg:4326", always_xy=True)
    lon_min_act, lat_min_act = inv_transformer.transform(x_min_act, y_min_act)
    lon_max_act, lat_max_act = inv_transformer.transform(x_max_act, y_max_act)
    
    # Read h from file (sliced)
    try:
        with rasterio.open(filename) as src:
            window = rasterio.windows.Window(c_start, r_start, c_stop-c_start, r_stop-r_start)
            h_sub = src.read(1, window=window)
            
            # Apply Mask to Sub-window
            if sim_state.bc_type_grid is not None:
                bc_sub = sim_state.bc_type_grid[r_start:r_stop, c_start:c_stop]
                if h_sub.shape == bc_sub.shape:
                    h_sub[bc_sub == -1] = 0.0
                    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading result: {e}")
        
    elev_sub = sim_state.elevation[r_start:r_stop, c_start:c_stop]
    mask_sub = None
    if sim_state.nodata_mask is not None:
        mask_sub = sim_state.nodata_mask[r_start:r_stop, c_start:c_stop]
        
    buf = render_map_image(h_sub, elev_sub, mask_sub, max_dim=max(width, height), show_terrain=show_terrain, show_water=show_water)
    
    headers = {
        "X-Image-Bounds": f"{lat_min_act},{lon_min_act},{lat_max_act},{lon_max_act}"
    }
    return StreamingResponse(buf, media_type="image/png", headers=headers)

if __name__ == "__main__":
    import uvicorn
    # Port 8003 to avoid conflict
    uvicorn.run(app, host="127.0.0.1", port=8003)

if __name__ == "__main__":
    import uvicorn
    # Port 8003 to avoid conflict
    uvicorn.run(app, host="127.0.0.1", port=8003)
    import uvicorn
    # Port 8003 to avoid conflict
    uvicorn.run(app, host="127.0.0.1", port=8003)

if __name__ == "__main__":
    import uvicorn
    # Port 8003 to avoid conflict
    uvicorn.run(app, host="127.0.0.1", port=8003)
