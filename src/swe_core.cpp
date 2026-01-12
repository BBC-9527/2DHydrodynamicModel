#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

#define EXPORT extern "C" __declspec(dllexport)

// Constants
const double G = 9.81;

struct Cell {
    double h;   // Water depth
    double z;   // Bed elevation
    double u;   // Velocity X
    double v;   // Velocity Y
    double n;   // Manning's n
    
    // Boundary & Mask
    int bc_type; // 0=Normal, 1=Inflow(Fixed WSE), 2=Outflow(Transmissive), -1=Inactive
    double bc_val; // Value for BC (e.g. WSE)
    
    // Source Term
    double q_src; // External source in m^3/s
    double mx_src; // Momentum source X (m^4/s^2)
    double my_src; // Momentum source Y (m^4/s^2)

    // New state (for double buffering)
    double h_new;
    double hu_new;
    double hv_new;
};

// Global Simulation State
int ROWS, COLS;
double DX, DY;
std::vector<Cell> GRID;

// Helper to get index
inline int idx(int r, int c) {
    if (r < 0) r = 0;
    if (r >= ROWS) r = ROWS - 1;
    if (c < 0) c = 0;
    if (c >= COLS) c = COLS - 1;
    return r * COLS + c;
}

EXPORT void init_model(int rows, int cols, double dx, double dy, double* elevation, double* roughness) {
    ROWS = rows;
    COLS = cols;
    DX = dx;
    DY = dy;
    GRID.resize(rows * cols);

    for (int i = 0; i < rows * cols; ++i) {
        GRID[i].z = elevation[i];
        GRID[i].n = roughness[i];
        GRID[i].h = 0.0;
        GRID[i].u = 0.0;
        GRID[i].v = 0.0;
        GRID[i].q_src = 0.0;
        GRID[i].mx_src = 0.0;
        GRID[i].my_src = 0.0;
        GRID[i].h_new = 0.0;
        GRID[i].hu_new = 0.0;
        GRID[i].hv_new = 0.0;
        GRID[i].bc_type = 0;
        GRID[i].bc_val = 0.0;
    }
}

EXPORT void set_boundary_mask(int* bc_types, double* bc_values) {
    for (int i = 0; i < ROWS * COLS; ++i) {
        GRID[i].bc_type = bc_types[i];
        GRID[i].bc_val = bc_values[i];
    }
}

EXPORT void set_source(int index, double q) {
    if (index >= 0 && index < ROWS * COLS) {
        GRID[index].q_src = q;
    }
}

EXPORT void set_source_momentum(int index, double mx, double my) {
    if (index >= 0 && index < ROWS * COLS) {
        GRID[index].mx_src = mx;
        GRID[index].my_src = my;
    }
}

EXPORT void reset_sources() {
    for (int i = 0; i < ROWS * COLS; ++i) {
        GRID[i].q_src = 0.0;
        GRID[i].mx_src = 0.0;
        GRID[i].my_src = 0.0;
    }
}

EXPORT void set_water_level(double* water_level) {
    // Initialize water level (H). Depth h = H - z.
    for (int i = 0; i < ROWS * COLS; ++i) {
        double h = water_level[i] - GRID[i].z;
        if (h < 0) h = 0;
        GRID[i].h = h;
        GRID[i].u = 0;
        GRID[i].v = 0;
    }
}

EXPORT void get_results(double* out_h, double* out_u, double* out_v) {
    for (int i = 0; i < ROWS * COLS; ++i) {
        out_h[i] = GRID[i].h;
        out_u[i] = GRID[i].u;
        out_v[i] = GRID[i].v;
    }
}

// HLL Flux Calculation
void compute_flux(double h_L, double u_L, double v_L, double h_R, double u_R, double v_R,
                  double& f_h, double& f_hu, double& f_hv, bool is_x_dir) {
    
    // Dry bed handling
    if (h_L <= 1e-6 && h_R <= 1e-6) {
        f_h = 0; f_hu = 0; f_hv = 0;
        return;
    }

    // Wave speeds
    double c_L = sqrt(G * h_L);
    double c_R = sqrt(G * h_R);
    
    double u_n_L = is_x_dir ? u_L : v_L;
    double u_n_R = is_x_dir ? u_R : v_R;
    
    double S_L = std::min(u_n_L - c_L, u_n_R - c_R);
    double S_R = std::max(u_n_L + c_L, u_n_R + c_R);

    if (S_L >= 0) {
        // Supercritical Left
        f_h = h_L * u_n_L;
        f_hu = h_L * u_L * u_n_L + (is_x_dir ? 0.5 * G * h_L * h_L : 0);
        f_hv = h_L * v_L * u_n_L + (is_x_dir ? 0 : 0.5 * G * h_L * h_L);
    } else if (S_R <= 0) {
        // Supercritical Right
        f_h = h_R * u_n_R;
        f_hu = h_R * u_R * u_n_R + (is_x_dir ? 0.5 * G * h_R * h_R : 0);
        f_hv = h_R * v_R * u_n_R + (is_x_dir ? 0 : 0.5 * G * h_R * h_R);
    } else {
        // Subcritical (HLL)
        double F_h_L = h_L * u_n_L;
        double F_hu_L = h_L * u_L * u_n_L + (is_x_dir ? 0.5 * G * h_L * h_L : 0);
        double F_hv_L = h_L * v_L * u_n_L + (is_x_dir ? 0 : 0.5 * G * h_L * h_L);

        double F_h_R = h_R * u_n_R;
        double F_hu_R = h_R * u_R * u_n_R + (is_x_dir ? 0.5 * G * h_R * h_R : 0);
        double F_hv_R = h_R * v_R * u_n_R + (is_x_dir ? 0 : 0.5 * G * h_R * h_R);

        f_h = (S_R * F_h_L - S_L * F_h_R + S_L * S_R * (h_R - h_L)) / (S_R - S_L);
        f_hu = (S_R * F_hu_L - S_L * F_hu_R + S_L * S_R * (h_R * u_R - h_L * u_L)) / (S_R - S_L);
        f_hv = (S_R * F_hv_L - S_L * F_hv_R + S_L * S_R * (h_R * v_R - h_L * v_L)) / (S_R - S_L);
    }
}

EXPORT void run_step(double dt) {
    // 1. Reset new state to current state & Apply Boundary Conditions
    for (int i = 0; i < ROWS * COLS; ++i) {
        
        // Inactive / Masked: Force dry
        if (GRID[i].bc_type == -1) {
            GRID[i].h = 0.0;
            GRID[i].u = 0.0;
            GRID[i].v = 0.0;
            GRID[i].h_new = 0.0;
            GRID[i].hu_new = 0.0;
            GRID[i].hv_new = 0.0;
            continue;
        }

        // Apply BCs immediately
        if (GRID[i].bc_type == 1) { // Inflow: Fixed WSE
            double target_h = GRID[i].bc_val - GRID[i].z;
            if (target_h < 0) target_h = 0;
            GRID[i].h = target_h;
            // Keep velocity as is or zero? Let's keep existing to allow momentum to develop, 
            // or zero for simple pool. Let's zero it for stability at source.
            GRID[i].u = 0; 
            GRID[i].v = 0;
        } 
        else if (GRID[i].bc_type == 2) { // Outflow: Transmissive (Zero Gradient roughly)
             // Handled during flux? Or just let it float.
             // Usually for transmissive, we copy neighbor values, but here FVM handles it naturally 
             // if we ensure it doesn't build up. 
             // Simple approach: Allow water to leave freely.
             // Actually, the FVM logic below computes fluxes. 
             // For BC cells, we should update them based on internal neighbors?
             // For simplicity in this mask-based approach, let's treat them as normal cells 
             // but enforce zero depth gradient? 
             // Let's leave them as normal cells for flux calc, but maybe dampen reflection.
        }

        GRID[i].h_new = GRID[i].h;
        GRID[i].hu_new = GRID[i].h * GRID[i].u;
        GRID[i].hv_new = GRID[i].h * GRID[i].v;
    }

    // 2. Compute Fluxes and Update
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            int i = idx(r, c);
            
            if (GRID[i].bc_type == -1) continue; // Inactive / Masked

            // --- X Flux (Right Face) ---
            if (c < COLS - 1) {
                int i_R = idx(r, c + 1);
                if (GRID[i_R].bc_type == -1) continue; // Don't flux into inactive

                // Reconstruction (Hydrostatic) - simplified
                // Just take cell values for now (First Order)
                double z_L = GRID[i].z;
                double z_R = GRID[i_R].z;
                double h_L = GRID[i].h;
                double h_R = GRID[i_R].h;
                
                // Surface reconstruction for well-balanced scheme
                double H_L = z_L + h_L;
                double H_R = z_R + h_R;
                double z_face = std::max(z_L, z_R);
                
                double h_L_eff = std::max(0.0, H_L - z_face);
                double h_R_eff = std::max(0.0, H_R - z_face);
                
                double f_h, f_hu, f_hv;
                compute_flux(h_L_eff, GRID[i].u, GRID[i].v, 
                             h_R_eff, GRID[i_R].u, GRID[i_R].v, 
                             f_h, f_hu, f_hv, true);
                
                // Update Left Cell (i)
                if (GRID[i].bc_type == 0 || GRID[i].bc_type == 2) {
                    GRID[i].h_new -= (dt / DX) * f_h;
                    GRID[i].hu_new -= (dt / DX) * f_hu;
                    GRID[i].hv_new -= (dt / DX) * f_hv;
                }
                
                // Update Right Cell (i_R)
                if (GRID[i_R].bc_type == 0 || GRID[i_R].bc_type == 2) {
                    GRID[i_R].h_new += (dt / DX) * f_h;
                    GRID[i_R].hu_new += (dt / DX) * f_hu;
                    GRID[i_R].hv_new += (dt / DX) * f_hv;
                }
            }

            // --- Y Flux (Bottom Face) ---
            if (r < ROWS - 1) {
                int i_D = idx(r + 1, c);
                if (GRID[i_D].bc_type == -1) continue; 

                double z_T = GRID[i].z;
                double z_B = GRID[i_D].z;
                double h_T = GRID[i].h;
                double h_B = GRID[i_D].h;
                
                double H_T = z_T + h_T;
                double H_B = z_B + h_B;
                double z_face = std::max(z_T, z_B);
                
                double h_T_eff = std::max(0.0, H_T - z_face);
                double h_B_eff = std::max(0.0, H_B - z_face);
                
                double f_h, f_hu, f_hv;
                compute_flux(h_T_eff, GRID[i].u, GRID[i].v, 
                             h_B_eff, GRID[i_D].u, GRID[i_D].v, 
                             f_h, f_hu, f_hv, false);
                             
                // Update Top Cell (i)
                if (GRID[i].bc_type == 0 || GRID[i].bc_type == 2) {
                    GRID[i].h_new -= (dt / DY) * f_h;
                    GRID[i].hu_new -= (dt / DY) * f_hu;
                    GRID[i].hv_new -= (dt / DY) * f_hv;
                }
                
                // Update Bottom Cell (i_D)
                if (GRID[i_D].bc_type == 0 || GRID[i_D].bc_type == 2) {
                    GRID[i_D].h_new += (dt / DY) * f_h;
                    GRID[i_D].hu_new += (dt / DY) * f_hu;
                    GRID[i_D].hv_new += (dt / DY) * f_hv;
                }
            }
        }
    }
    
    // 3. Update State & Add Source Terms (Friction)
    for (int i = 0; i < ROWS * COLS; ++i) {
        // Apply Source Term
        if (GRID[i].q_src != 0) {
            double dh_src = (GRID[i].q_src * dt) / (DX * DY);
            GRID[i].h_new += dh_src;
        }
        
        // Safety Check: Cap maximum depth to avoid explosion
        if (GRID[i].h_new > 5000.0) GRID[i].h_new = 5000.0; // Hard cap
        if (std::isnan(GRID[i].h_new)) GRID[i].h_new = 0.0;
        
        // Apply Momentum Source
        if (GRID[i].mx_src != 0 || GRID[i].my_src != 0) {
            double dhu_src = (GRID[i].mx_src * dt) / (DX * DY);
            double dhv_src = (GRID[i].my_src * dt) / (DX * DY);
            GRID[i].hu_new += dhu_src;
            GRID[i].hv_new += dhv_src;
        }

        // Enforce BCs again for Inflow (overwrite any flux updates)
        if (GRID[i].bc_type == 1) {
             double target_h = GRID[i].bc_val - GRID[i].z;
             if (target_h < 0) target_h = 0;
             GRID[i].h = target_h;
             GRID[i].u = 0;
             GRID[i].v = 0;
             continue;
        }
        if (GRID[i].bc_type == -1) continue;

        if (GRID[i].h_new < 1e-6) {
            GRID[i].h = 0;
            GRID[i].u = 0;
            GRID[i].v = 0;
            continue;
        }

        double h = GRID[i].h_new;
        double hu = GRID[i].hu_new;
        double hv = GRID[i].hv_new;
        double u = hu / h;
        double v = hv / h;
        
        // Manning Friction
        // S_f = n^2 * V * |V| / h^(4/3)
        // Semi-implicit update: u_new = u / (1 + dt * coeff)
        double V_mag = sqrt(u*u + v*v);
        double n = GRID[i].n;
        double cf = (n * n * V_mag) / pow(h, 4.0/3.0);
        double factor = 1.0 / (1.0 + dt * G * cf);
        
        GRID[i].h = h;
        GRID[i].u = u * factor;
        GRID[i].v = v * factor;
    }
}
