import sys
import os
import numpy as np
import rasterio
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox,
                             QInputDialog, QToolBar, QAction, QComboBox)
from PyQt5.QtCore import QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import matplotlib.patches as patches

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from swe_wrapper import SWEModel

class HydroApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D Hydrodynamic Model (Simplified HEC-RAS)")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize Model
        dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "swe_core.dll"))
        if not os.path.exists(dll_path):
            QMessageBox.critical(self, "Error", f"Core DLL not found at: {dll_path}")
            sys.exit(1)
            
        self.model = SWEModel(dll_path)
        self.dt = 0.1
        self.simulation_running = False
        
        # State
        self.elevation = None
        self.h = None
        self.bc_type_grid = None
        self.bc_val_grid = None
        
        # UI Setup
        self.init_ui()
        
    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Toolbar for Tools
        self.toolbar = QToolBar()
        self.addToolBar(self.toolbar)
        
        self.action_view = QAction("View/Pan", self)
        self.action_view.setCheckable(True)
        self.action_view.setChecked(True)
        self.action_view.triggered.connect(lambda: self.set_tool("view"))
        self.toolbar.addAction(self.action_view)

        self.action_inflow = QAction("Draw Inflow", self)
        self.action_inflow.setCheckable(True)
        self.action_inflow.triggered.connect(lambda: self.set_tool("inflow"))
        self.toolbar.addAction(self.action_inflow)

        self.action_outflow = QAction("Draw Outflow", self)
        self.action_outflow.setCheckable(True)
        self.action_outflow.triggered.connect(lambda: self.set_tool("outflow"))
        self.toolbar.addAction(self.action_outflow)
        
        self.action_mask = QAction("Draw Inactive Mask", self)
        self.action_mask.setCheckable(True)
        self.action_mask.triggered.connect(lambda: self.set_tool("mask"))
        self.toolbar.addAction(self.action_mask)

        self.toolbar.addSeparator()
        self.action_clear_bc = QAction("Clear All BCs", self)
        self.action_clear_bc.triggered.connect(self.clear_bcs)
        self.toolbar.addAction(self.action_clear_bc)

        self.current_tool = "view"

        # Plot Area
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.layout.addWidget(self.canvas)
        
        # Selector
        self.rs = RectangleSelector(self.ax, self.on_select,
                                    useblit=True,
                                    button=[1], 
                                    minspanx=5, minspany=5,
                                    spancoords='pixels',
                                    interactive=True)
        self.rs.set_active(False)

        # Controls
        controls_layout = QHBoxLayout()
        
        self.btn_load = QPushButton("Load DEM")
        self.btn_load.clicked.connect(self.load_dem)
        controls_layout.addWidget(self.btn_load)

        self.btn_test = QPushButton("Load Test Terrain")
        self.btn_test.clicked.connect(self.load_test_terrain)
        controls_layout.addWidget(self.btn_test)

        self.btn_start = QPushButton("Start Simulation")
        self.btn_start.clicked.connect(self.toggle_simulation)
        self.btn_start.setEnabled(False)
        controls_layout.addWidget(self.btn_start)
        
        self.lbl_status = QLabel("Ready")
        controls_layout.addWidget(self.lbl_status)

        self.layout.addLayout(controls_layout)

        # Animation Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        
        self.img_plot = None
        self.bc_patches = []

    def set_tool(self, tool_name):
        self.current_tool = tool_name
        
        # Uncheck others
        self.action_view.setChecked(tool_name == "view")
        self.action_inflow.setChecked(tool_name == "inflow")
        self.action_outflow.setChecked(tool_name == "outflow")
        self.action_mask.setChecked(tool_name == "mask")
        
        if tool_name == "view":
            self.rs.set_active(False)
            self.lbl_status.setText("Mode: View/Pan")
        else:
            self.rs.set_active(True)
            self.lbl_status.setText(f"Mode: Draw {tool_name}")

    def on_select(self, eclick, erelease):
        if self.elevation is None: return
        
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        
        # Ensure coords are within bounds
        rows, cols = self.elevation.shape
        r1, r2 = min(y1, y2), max(y1, y2)
        c1, c2 = min(x1, x2), max(x1, x2)
        
        r1 = max(0, min(r1, rows-1))
        r2 = max(0, min(r2, rows-1))
        c1 = max(0, min(c1, cols-1))
        c2 = max(0, min(c2, cols-1))
        
        if self.current_tool == "inflow":
            val, ok = QInputDialog.getDouble(self, "Inflow Condition", "Water Surface Elevation (m):", 15.0, 0, 10000, 2)
            if ok:
                self.bc_type_grid[r1:r2, c1:c2] = 1
                self.bc_val_grid[r1:r2, c1:c2] = val
                self.add_bc_patch(c1, r1, c2-c1, r2-r1, 'red', alpha=0.3)
                
        elif self.current_tool == "outflow":
            self.bc_type_grid[r1:r2, c1:c2] = 2
            self.add_bc_patch(c1, r1, c2-c1, r2-r1, 'green', alpha=0.3)
            
        elif self.current_tool == "mask":
            self.bc_type_grid[r1:r2, c1:c2] = -1
            self.add_bc_patch(c1, r1, c2-c1, r2-r1, 'gray', alpha=0.5)
            
        # Update Model
        self.model.set_boundary_conditions(self.bc_type_grid, self.bc_val_grid)
        self.canvas.draw()

    def add_bc_patch(self, x, y, w, h, color, alpha):
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor=color, alpha=alpha)
        self.ax.add_patch(rect)
        self.bc_patches.append(rect)

    def clear_bcs(self):
        if self.elevation is None: return
        self.bc_type_grid.fill(0)
        self.bc_val_grid.fill(0)
        self.model.set_boundary_conditions(self.bc_type_grid, self.bc_val_grid)
        
        for p in self.bc_patches:
            p.remove()
        self.bc_patches.clear()
        self.canvas.draw()


    def load_dem(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open DEM", "", "GeoTIFF (*.tif *.tiff)")
        if file_path:
            try:
                with rasterio.open(file_path) as src:
                    self.elevation = src.read(1).astype(np.float64)
                    transform = src.transform
                    self.dx = transform[0]
                    self.dy = -transform[4] # Usually negative
                    
                    # Resize if too large for demo
                    if self.elevation.shape[0] > 500 or self.elevation.shape[1] > 500:
                        QMessageBox.warning(self, "Warning", "DEM is large. Performance may be slow.")
                
                self.setup_model()
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def load_test_terrain(self):
        # Create a channel with a dam
        rows, cols = 100, 200
        self.dx, self.dy = 1.0, 1.0
        x = np.linspace(0, cols*self.dx, cols)
        y = np.linspace(0, rows*self.dy, rows)
        X, Y = np.meshgrid(x, y)
        
        # Channel with slope
        self.elevation = 10.0 - 0.01 * X 
        # Add banks
        self.elevation += 0.05 * (Y - 50)**2
        
        self.setup_model()
        
        # Add Dam Break Initial Condition
        # Water level 15m on the left (x < 50), Dry on right
        water_surface = self.elevation.copy()
        mask = X < 50
        water_surface[mask] = np.maximum(water_surface[mask], 15.0)
        
        self.model.set_water_surface(water_surface)
        
        # Update plot initial
        h, u, v = self.model.get_results()
        self.h = h
        self.update_plot()
        self.lbl_status.setText("Test Terrain Loaded (Dam Break Setup)")

    def setup_model(self):
        if self.elevation is None: return
        
        rows, cols = self.elevation.shape
        roughness = np.full((rows, cols), 0.03, dtype=np.float64) # Manning n = 0.03
        
        self.model.init(self.elevation, roughness, self.dx, self.dy)
        
        # Init BC grids
        self.bc_type_grid = np.zeros((rows, cols), dtype=np.int32)
        self.bc_val_grid = np.zeros((rows, cols), dtype=np.float64)
        
        # Default dry
        self.model.set_water_surface(self.elevation) 
        
        self.btn_start.setEnabled(True)
        self.update_plot()

    def toggle_simulation(self):
        if self.simulation_running:
            self.timer.stop()
            self.simulation_running = False
            self.btn_start.setText("Start Simulation")
        else:
            self.timer.start(50) # 50ms
            self.simulation_running = True
            self.btn_start.setText("Stop Simulation")

    def update_simulation(self):
        # Run multiple sub-steps for stability if needed, but 1 step per frame for visual
        self.model.step(self.dt)
        h, u, v = self.model.get_results()
        self.h = h
        
        # Refresh plot
        self.update_plot_data()

    def update_plot(self):
        self.ax.clear()
        self.ax.set_title("Water Depth")
        
        # Plot terrain as background (hillshade-ish)
        self.ax.imshow(self.elevation, cmap='terrain', alpha=0.5, origin='upper')
        
        # Plot water depth
        if self.h is not None:
            # Mask zero depth
            h_masked = np.ma.masked_where(self.h < 0.01, self.h)
            self.img_plot = self.ax.imshow(h_masked, cmap='Blues', alpha=0.8, origin='upper', vmin=0, vmax=5)
            
        self.canvas.draw()

    def update_plot_data(self):
        if self.img_plot and self.h is not None:
             h_masked = np.ma.masked_where(self.h < 0.01, self.h)
             self.img_plot.set_data(h_masked)
             self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HydroApp()
    window.show()
    sys.exit(app.exec_())
