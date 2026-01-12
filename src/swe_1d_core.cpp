#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

#define EXPORT extern "C" __declspec(dllexport)

// Constants
const double G = 9.81;

struct Cell1D {
    double h;     // Water depth
    double z;     // Bed elevation
    double u;     // Velocity
    double n;     // Manning's n
    double width; // Channel width at this cell
    double q_lat; // Lateral inflow (m^3/s)

    // New state (for double buffering)
    double h_new;
    double hu_new;
};

class SWE1D_Solver {
public:
    int num_cells;
    double dx;
    std::vector<Cell1D> grid;

    // Boundary Conditions
    int bc_left_type = 0;
    double bc_left_val = 0.0;
    int bc_right_type = 3;
    double bc_right_val = 0.0;

    void init(int n_cells, double d_x, double* elevation, double* roughness, double* width, double* initial_h) {
        num_cells = n_cells;
        dx = d_x;
        grid.resize(num_cells);

        for (int i = 0; i < num_cells; ++i) {
            grid[i].z = elevation[i];
            grid[i].n = roughness[i];
            grid[i].width = width[i];
            grid[i].h = initial_h[i];
            grid[i].u = 0.0;
            grid[i].q_lat = 0.0;
            grid[i].h_new = 0.0;
            grid[i].hu_new = 0.0;
        }
    }

    void set_boundary(int l_type, double l_val, int r_type, double r_val) {
        bc_left_type = l_type;
        bc_left_val = l_val;
        bc_right_type = r_type;
        bc_right_val = r_val;
    }

    void set_source(int index, double q_val) {
        if (index >= 0 && index < num_cells) {
            grid[index].q_lat = q_val;
        }
    }

    void get_results(double* out_h, double* out_u) {
        for (int i = 0; i < num_cells; ++i) {
            out_h[i] = grid[i].h;
            out_u[i] = grid[i].u;
        }
    }
    
    // Helper: HLL Flux
    void compute_flux(double h_L, double u_L, double h_R, double u_R, double& f_h, double& f_hu) {
        if (h_L <= 1e-6 && h_R <= 1e-6) {
            f_h = 0; f_hu = 0; return;
        }
        double c_L = sqrt(G * h_L);
        double c_R = sqrt(G * h_R);
        double S_L = std::min(u_L - c_L, u_R - c_R);
        double S_R = std::max(u_L + c_L, u_R + c_R);

        if (S_L >= 0) {
            f_h = h_L * u_L;
            f_hu = h_L * u_L * u_L + 0.5 * G * h_L * h_L;
        } else if (S_R <= 0) {
            f_h = h_R * u_R;
            f_hu = h_R * u_R * u_R + 0.5 * G * h_R * h_R;
        } else {
            double F_h_L = h_L * u_L;
            double F_hu_L = h_L * u_L * u_L + 0.5 * G * h_L * h_L;
            double F_h_R = h_R * u_R;
            double F_hu_R = h_R * u_R * u_R + 0.5 * G * h_R * h_R;
            f_h = (S_R * F_h_L - S_L * F_h_R + S_L * S_R * (h_R - h_L)) / (S_R - S_L);
            f_hu = (S_R * F_hu_L - S_L * F_hu_R + S_L * S_R * (h_R * u_R - h_L * u_L)) / (S_R - S_L);
        }
    }

    void run_step(double dt) {
        std::vector<double> flux_h(num_cells + 1);
        std::vector<double> flux_hu(num_cells + 1);

        for (int i = 0; i <= num_cells; ++i) {
            double h_L, u_L, z_L, h_R, u_R, z_R;

            // Left State
            if (i == 0) {
                z_L = grid[0].z;
                if (bc_left_type == 0) { // Wall
                    h_L = grid[0].h; u_L = -grid[0].u;
                } else if (bc_left_type == 1) { // Fixed Depth
                    h_L = bc_left_val; u_L = grid[0].u;
                    if (h_L < 0) h_L = 0;
                } else if (bc_left_type == 2) { // Fixed Discharge (approx)
                    h_L = grid[0].h; u_L = (h_L > 1e-4) ? bc_left_val / h_L : 0.0;
                } else { // Transmissive
                    h_L = grid[0].h; u_L = grid[0].u;
                }
            } else {
                z_L = grid[i - 1].z;
                h_L = grid[i - 1].h; u_L = grid[i - 1].u;
            }

            // Right State
            if (i == num_cells) {
                z_R = grid[num_cells - 1].z;
                if (bc_right_type == 0) { // Wall
                    h_R = grid[num_cells - 1].h; u_R = -grid[num_cells - 1].u;
                } else if (bc_right_type == 1) { // Fixed Depth
                    h_R = bc_right_val; u_R = grid[num_cells - 1].u;
                } else { // Transmissive
                    h_R = grid[num_cells - 1].h; u_R = grid[num_cells - 1].u;
                }
            } else {
                z_R = grid[i].z;
                h_R = grid[i].h; u_R = grid[i].u;
            }

            double z_face = std::max(z_L, z_R);
            double h_Lr = std::max(0.0, (z_L + h_L) - z_face);
            double h_Rr = std::max(0.0, (z_R + h_R) - z_face);
            compute_flux(h_Lr, u_L, h_Rr, u_R, flux_h[i], flux_hu[i]);
        }

        // Update
        for (int i = 0; i < num_cells; ++i) {
            double dVol_src = 0.0;
            if (grid[i].q_lat != 0) {
                dVol_src = grid[i].q_lat * dt / (grid[i].width * dx);
            }

            // Determine Interface Widths for FVM (Conservation of Volume/Momentum)
            // Left Face (i)
            double w_face_L = grid[i].width;
            if (i > 0) w_face_L = 0.5 * (grid[i-1].width + grid[i].width);

            // Right Face (i+1)
            double w_face_R = grid[i].width;
            if (i < num_cells - 1) w_face_R = 0.5 * (grid[i].width + grid[i+1].width);
            
            // Fluxes integrated over width (Q and Momentum Flux)
            double F_mass_L = flux_h[i] * w_face_L;
            double F_mass_R = flux_h[i+1] * w_face_R;
            
            double F_mom_L = flux_hu[i] * w_face_L;
            double F_mom_R = flux_hu[i+1] * w_face_R;

            // Update Mass (Continuity)
            // B * dh/dt + d(Q)/dx = 0  => dh = - dt/(B*dx) * dQ
            grid[i].h_new = grid[i].h - (dt / (grid[i].width * dx)) * (F_mass_R - F_mass_L) + dVol_src;
            
            // Update Momentum
            // B * d(hu)/dt + d(Flux_hu * B)/dx = Source_Pressure
            // Source Pressure term due to width variation: 1/2 * g * h^2 * dB/dx
            // Approximation: dB/dx approx (w_face_R - w_face_L) / dx
            double pressure_source = 0.0;
            if (std::abs(w_face_R - w_face_L) > 1e-8) {
                pressure_source = 0.5 * G * grid[i].h * grid[i].h * (w_face_R - w_face_L) / dx;
            }

            grid[i].hu_new = grid[i].h * grid[i].u - (dt / (grid[i].width * dx)) * (F_mom_R - F_mom_L);

            // Add pressure source
            if (grid[i].width > 1e-6) {
                grid[i].hu_new += dt * pressure_source / grid[i].width;
            } 

            if (num_cells >= 2) {
                double dzdx = 0.0;
                if (i == 0) dzdx = (grid[1].z - grid[0].z) / dx;
                else if (i == num_cells - 1) dzdx = (grid[i].z - grid[i-1].z) / dx;
                else dzdx = (grid[i+1].z - grid[i-1].z) / (2.0 * dx);
                grid[i].hu_new += dt * (-G * grid[i].h * dzdx);
            }

            // Friction
            double h_new = grid[i].h_new;
            if (h_new < 1e-6) {
                grid[i].h_new = 0; grid[i].hu_new = 0;
            } else {
                double u_new_star = grid[i].hu_new / h_new;
                double n2 = grid[i].n * grid[i].n;
                double Sf = (n2 * u_new_star * std::abs(u_new_star)) / std::pow(h_new, 4.0/3.0);
                double u_new = u_new_star / (1.0 + G * n2 * std::abs(u_new_star) * dt / std::pow(h_new, 4.0/3.0));
                grid[i].hu_new = h_new * u_new;
            }
        }

        // Apply
        for (int i = 0; i < num_cells; ++i) {
            grid[i].h = grid[i].h_new;
            if (grid[i].h > 1e-6) grid[i].u = grid[i].hu_new / grid[i].h;
            else { grid[i].h = 0; grid[i].u = 0; }
        }
    }
};

extern "C" {
    EXPORT void* create_model() {
        return new SWE1D_Solver();
    }

    EXPORT void delete_model(void* ptr) {
        if (ptr) delete (SWE1D_Solver*)ptr;
    }

    EXPORT void init_model(void* ptr, int n, double dx, double* z, double* r, double* w, double* h) {
        if (ptr) ((SWE1D_Solver*)ptr)->init(n, dx, z, r, w, h);
    }

    EXPORT void set_boundary_conditions(void* ptr, int l_t, double l_v, int r_t, double r_v) {
        if (ptr) ((SWE1D_Solver*)ptr)->set_boundary(l_t, l_v, r_t, r_v);
    }

    EXPORT void set_source(void* ptr, int idx, double q) {
        if (ptr) ((SWE1D_Solver*)ptr)->set_source(idx, q);
    }

    EXPORT void run_step(void* ptr, double dt) {
        if (ptr) ((SWE1D_Solver*)ptr)->run_step(dt);
    }

    EXPORT void get_results(void* ptr, double* h, double* u) {
        if (ptr) ((SWE1D_Solver*)ptr)->get_results(h, u);
    }
}
