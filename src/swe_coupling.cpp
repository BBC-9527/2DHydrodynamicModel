#include <cmath>
#include <algorithm>
#include <iostream>

#define EXPORT extern "C" __declspec(dllexport)

const double G = 9.81;

// InterfaceState Struct
// Corresponds to Python InterfaceState
struct InterfaceState {
    // 1D State
    double h1;   // Depth (m)
    double z1;   // Bed Elevation (m)
    double u1;   // Velocity (m/s)

    // 2D State
    double h2;   // Depth (m)
    double z2;   // Bed Elevation (m)
    double u2;   // Velocity X (m/s)
    double v2;   // Velocity Y (m/s)

    // Geometry
    double z_bank; // Bank Crest Level (m)
    double nx;     // Normal X (1D -> 2D)
    double ny;     // Normal Y (1D -> 2D)
    double length; // Interface Length (m)
    
    // Params
    double discharge_coeff; // Cd (0.0 ~ 1.0)
};

// CouplingFlux Struct
// Corresponds to Python CouplingFlux
struct CouplingFlux {
    double mass_flux;  // Mass Flux (m^3/s)
    double mom_flux_x; // Momentum Flux X (N)
    double mom_flux_y; // Momentum Flux Y (N)
};

// Helper: HLL Riemann Solver
// Computes flux per unit length (f_h, f_hu)
void solve_hll_interface(double h_L, double u_n_L, double h_R, double u_n_R, 
                        double& f_h, double& f_hu) {
    
    // Dry threshold (consistent with main function)
    if (h_L <= 1e-3 && h_R <= 1e-3) {
        f_h = 0.0;
        f_hu = 0.0;
        return;
    }

    // Wave speeds
    double c_L = sqrt(G * h_L);
    double c_R = sqrt(G * h_R);

    // Signal speeds (S_L, S_R)
    double S_L = std::min(u_n_L - c_L, u_n_R - c_R);
    double S_R = std::max(u_n_L + c_L, u_n_R + c_R);

    // HLL Flux
    if (S_L >= 0) {
        // Supercritical -> Right
        f_h = h_L * u_n_L;
        f_hu = h_L * u_n_L * u_n_L + 0.5 * G * h_L * h_L;
    } 
    else if (S_R <= 0) {
        // Supercritical -> Left
        f_h = h_R * u_n_R;
        f_hu = h_R * u_n_R * u_n_R + 0.5 * G * h_R * h_R;
    } 
    else {
        // Subcritical / Transcritical
        double F_h_L = h_L * u_n_L;
        double F_hu_L = h_L * u_n_L * u_n_L + 0.5 * G * h_L * h_L;

        double F_h_R = h_R * u_n_R;
        double F_hu_R = h_R * u_n_R * u_n_R + 0.5 * G * h_R * h_R;

        f_h = (S_R * F_h_L - S_L * F_h_R + S_L * S_R * (h_R - h_L)) / (S_R - S_L);
        f_hu = (S_R * F_hu_L - S_L * F_hu_R + S_L * S_R * (h_R * u_n_R - h_L * u_n_L)) / (S_R - S_L);
    }
}

// Core Coupling Function
EXPORT void compute_coupling_flux(InterfaceState state, CouplingFlux* out_flux) {
    // 1. Sanity Check
    if (state.h1 < 0) state.h1 = 0;
    if (state.h2 < 0) state.h2 = 0;
    
    // Default Cd
    double Cd = (state.discharge_coeff > 0.0) ? state.discharge_coeff : 1.0;

    // Normalize Normal Vector
    double norm = sqrt(state.nx * state.nx + state.ny * state.ny);
    double nx = state.nx;
    double ny = state.ny;
    if (norm > 1e-9) {
        nx /= norm;
        ny /= norm;
    }
    
    // Tangential Vector (-ny, nx)
    double tx = -ny;
    double ty = nx;

    // 2. Reconstruct WSE
    double wse_1 = state.z1 + state.h1;
    double wse_2 = state.z2 + state.h2;

    // 3. Effective Depth above Bank
    double d_L = std::max(0.0, wse_1 - state.z_bank);
    double d_R = std::max(0.0, wse_2 - state.z_bank);

    // CRITICAL FIX: Clamp effective depth to 0 if actual water depth is negligible.
    // This prevents "Ghost Water" where high bed elevation (z > z_bank) creates fake effective depth even when dry.
    if (state.h1 <= 1e-3) d_L = 0.0;
    if (state.h2 <= 1e-3) d_R = 0.0;

    // Dry check (User suggested 1e-3 for stability in large scale)
    if (d_L <= 1e-3 && d_R <= 1e-3) {
        out_flux->mass_flux = 0.0;
        out_flux->mom_flux_x = 0.0;
        out_flux->mom_flux_y = 0.0;
        return;
    }

    // 4. Velocity Decomposition
    
    // 1D Side: The input u1 is usually longitudinal.
    // For lateral coupling, the normal velocity towards 2D is effectively 0 (or driven by pressure).
    // The caller should pass u1=0 for lateral coupling, or we assume it here if we want to enforce it.
    // However, for end-coupling (river mouth), u1 IS the normal velocity.
    // We rely on the caller to pass the correct NORMAL velocity component in state.u1.
    double u_n_L = state.u1; 
    double u_t_L = 0.0;

    // 2D Side: Project
    double u_n_R = state.u2 * nx + state.v2 * ny;
    double u_t_R = state.u2 * tx + state.v2 * ty;

    // 5. Solve Riemann Problem (Normal Direction)
    double flux_h_n, flux_mom_n;
    solve_hll_interface(d_L, u_n_L, d_R, u_n_R, flux_h_n, flux_mom_n);
    
    // Apply Cd
    flux_h_n *= Cd;
    flux_mom_n *= Cd;

    // 6. Tangential Momentum Flux (Upwind)
    double flux_mom_t = 0.0;
    if (flux_h_n > 0) {
        flux_mom_t = flux_h_n * u_t_L;
    } else {
        flux_mom_t = flux_h_n * u_t_R;
    }

    // 7. Total Flux (Multiply by Length)
    out_flux->mass_flux = flux_h_n * state.length;

    // 8. Project Momentum Flux back to Global (X, Y)
    // F_vector = F_n * n_vector + F_t * t_vector
    out_flux->mom_flux_x = (flux_mom_n * nx + flux_mom_t * tx) * state.length;
    out_flux->mom_flux_y = (flux_mom_n * ny + flux_mom_t * ty) * state.length;
}
