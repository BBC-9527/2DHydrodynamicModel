#ifndef SWE_1D_NETWORK_H
#define SWE_1D_NETWORK_H

#include <vector>
#include <cmath>
#include <map>

// Export macro
#ifdef _WIN32
    #define EXPORT extern "C" __declspec(dllexport)
#else
    #define EXPORT extern "C"
#endif

// Constants
const double GRAVITY = 9.81;
const double TOLERANCE = 1e-4;
const int MAX_ITER = 20;
const double THETA = 0.6; // Preissmann weighting factor (0.5 - 1.0)

// Basic Structures
struct CrossSection {
    double distance; // Distance from reach start
    double z_bed;    // Bed elevation
    double width;    // Channel width (rectangular assumption for simplification, or top width)
    double n;        // Manning's roughness
};

struct NodeState {
    double h; // Water depth
    double Q; // Discharge
    double Z; // Water Surface Elevation (z_bed + h)
};

// Forward Declaration
class Reach;

// Junction Node
struct Junction {
    int id;
    double Z; // Water level at junction
    double external_Q; // External inflow/outflow (m3/s)
    std::vector<int> connected_reaches; // IDs of reaches
    std::vector<bool> is_start_node;    // True if reach starts at this junction
};

// Reach Class
class Reach {
public:
    int id;
    int num_nodes;
    std::vector<CrossSection> sections;
    std::vector<NodeState> current_state;
    std::vector<NodeState> next_state;
    
    // Boundary conditions for the current time step
    double bc_z_start, bc_z_end;
    bool use_z_start, use_z_end; // If false, assume Q boundary (not implemented fully in this simplified version)

    Reach(int id_val, int n_nodes);
    void set_geometry(const std::vector<double>& dist, const std::vector<double>& z, const std::vector<double>& w, const std::vector<double>& n);
    void set_initial_condition(const std::vector<double>& h, const std::vector<double>& q);
    
    // Core Solver Methods
    void preissmann_step(double dt);
    
    // Helpers
    double get_area(int i, double h) const;
    double get_perimeter(int i, double h) const;
};

// Network Manager
class NetworkSolver {
public:
    std::map<int, Reach> reaches;
    std::map<int, Junction> junctions;
    
    void add_reach(int id, int n_nodes, double* dist, double* z, double* w, double* n, double* h0, double* q0);
    void add_junction(int id, int* reach_ids, int* is_start, int n_reaches);
    
    // The "Hierarchical Solution" Step
    // 1. Guess Junction Z
    // 2. Solve Reaches
    // 3. Update Junction Z (Mass Balance)
    // 4. Repeat
    void step(double dt);
    
    // Data Access
    void get_reach_results(int reach_id, double* h_out, double* q_out);
    
    // BC
    void set_junction_source(int junction_id, double q);
};

// C-Style Export API
EXPORT void* Network_Create();
EXPORT void Network_AddReach(void* net, int id, int n_nodes, double* dist, double* z, double* w, double* n, double* h0, double* q0);
EXPORT void Network_AddJunction(void* net, int id, int* reach_ids, int* is_start, int n_reaches);
EXPORT void Network_SetJunctionSource(void* net, int junction_id, double q);
EXPORT void Network_Step(void* net, double dt);
EXPORT void Network_GetResults(void* net, int reach_id, double* h_out, double* q_out);
EXPORT void Network_Delete(void* net);

#endif // SWE_1D_NETWORK_H
