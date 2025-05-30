/*
Copyright 2025 D-Wave

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include <cstdint>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <limits>
#include "cpu_dsbm.h"

using namespace std;

// Performs a single run of simulated annealing with the given inputs.
// @param state a int8 array where each int8 holds the state of a
//        variable. Note that this will be used as the initial state of the
//        run.
// @param h vector of h or field value on each variable
// @param degrees the degree of each variable
// @param neighbors lists of the neighbors of each variable, such that 
//        neighbors[i][j] is the jth neighbor of variable i. Note
// @param neighbour_couplings same as neighbors, but instead has the J value.
//        neighbour_couplings[i][j] is the J value or weight on the coupling
//        between variables i and neighbors[i][j]. 
// @param sweeps_per_beta The number of sweeps to perform at each beta value.
//        Total number of sweeps is `sweeps_per_beta` * length of
//        `Hp_field`.
// @param Hp_field A list of the beta values to run `sweeps_per_beta`
//        sweeps at.
// @return Nothing, but `state` now contains the result of the run.

template<typename T>
void discrete_simulated_bifurcation_run(
    T* state_x,
    T* state_y,
    std::vector<T> & dstate_x,
    std::vector<T> & dstate_y,
    const std::vector<int>& degrees,
    const std::vector<vector<int>>& neighbors,
    const std::vector<vector<double>>& neighbour_couplings,
    const std::vector<double>& a_schedule,
    const double a0,
    const double c0,
    const double _Delta_t
) {
    const int num_vars = state_x.size();

    /*
      // Feature enhancement for O(conn.) speed up with dense matrices
      // We may wish to save a sign(state_x) variable separately and update delta_energy for higher efficiency
    for (int var = 0; var < num_vars; var++) {
      // Local fields
      delta_energy[var] = get_marginal_state_fieldC(var, state, h, degrees,
						    neighbors, neighbour_couplings,
						    state_to_costheta);
    } 
    */
    for (int a_idx = 0; a_idx < (int)a_schedule.size(); a_idx++) {
        for (int varI = 0; varI < num_vars; varI++) {
	  
        }
    }
}


// Perform simulated annealing on a general problem
// @param states a int8 array of size num_samples * number of variables in the
//        problem. Will be overwritten by this function as samples are filled
//        in. The initial state of the samples are used to seed the simulated
//        annealing runs.
// @param energies a double array of size num_samples. Will be overwritten by
//        this function as energies are filled in.
// @param num_samples the number of samples to get.
// @param h vector of h or field value on each variable
// @param coupler_starts an int vector containing the variables of one side of
//        each coupler in the problem
// @param coupler_ends an int vector containing the variables of the other side 
//        of each coupler in the problem
// @param coupler_weights a double vector containing the weights of the couplers
//        in the same order as coupler_starts and coupler_ends
// @param sweeps_per_beta The number of sweeps to perform at each beta value.
//        Total number of sweeps is `sweeps_per_beta` * length of
//        `Hp_field`.
// @param Hp_field A list of the beta values to run `sweeps_per_beta`
//        sweeps at.
// @param interrupt_callback A function that is invoked between each run of simulated annealing
//        if the function returns True then it will stop running.
// @param interrupt_function A pointer to contents that are passed to interrupt_callback.
// @return the number of samples taken. If no interrupt occured, will equal num_samples.
template <typename T>
int general_discrete_simulated_bifurcation_machine(
    T* states_x,
    T* states_y,
    const int num_samples,
    const int num_vars,
    const vector<int> coupler_starts,
    const vector<int> coupler_ends,
    const vector<double> coupler_weights,
    const vector<double> a_schedule,
    const double a0,
    const double c0,
    const double Delta_t,
    callback interrupt_callback,
    void * const interrupt_function
) {
    
    if (!((coupler_starts.size() == coupler_ends.size()) &&
	  (coupler_starts.size() == coupler_weights.size()))) {
        throw runtime_error("coupler vectors have mismatched lengths");
    }
    
    // degrees will be a vector of the degrees of each variable
    vector<int> degrees(num_vars, 0);
    // neighbors is a vector of vectors, such that neighbors[i][j] is the jth
    // neighbor of variable i
    vector<vector<int>> neighbors(num_vars);
    // neighbour_couplings is another vector of vectors with the same structure
    // except neighbour_couplings[i][j] is the weight on the coupling between i
    // and its jth neighbor
    vector<vector<double>> neighbour_couplings(num_vars);

    // build the degrees, neighbors, and neighbour_couplings vectors by
    // iterating over the inputted coupler vectors
    for (unsigned int cplr = 0; cplr < coupler_starts.size(); cplr++) {
        int u = coupler_starts[cplr];
        int v = coupler_ends[cplr];

        if ((u < 0) || (v < 0) || (u >= num_vars) || (v >= num_vars)) {
            throw runtime_error("coupler indexes contain an invalid variable");
        }

        // add v to u's neighbors list and vice versa
        neighbors[u].push_back(v);
        neighbors[v].push_back(u);
        // add the weights
        neighbour_couplings[u].push_back(coupler_weights[cplr]);
        neighbour_couplings[v].push_back(coupler_weights[cplr]);

        // increase the degrees of both variables
        degrees[u]++;
        degrees[v]++;
    }


    // get the sbm samples
    int sample = 0;
    std::vector<T> dstate_x(num_vars);  // Reusable buffer
    std::vector<T> dstate_y(num_vars);  // Reusable buffer
    while (sample < num_samples) {
        // states is a giant spin array that will hold the resulting states for
        // all the samples, so we need to get the location inside that vector
        // where we will store the sample for this sample
        double *state_x = states_x + sample*num_vars;
        double *state_y = states_y + sample*num_vars;
        // then do the actual sample. this function will modify state, storing
        // the sample there
        // Branching here is designed to make expicit compile time optimizations
	discrete_simulated_bifurcation_run(state_x, state_y,
					   dstate_x, dstate_y, degrees,
					   neighbors, neighbour_couplings,
					   a_schedule,
					   a0,
					   c0,
					   Delta_t);
	sample++;

        // if interrupt_function returns true, stop sampling
        if (interrupt_function && interrupt_callback(interrupt_function)) break;
    }

    // return the number of samples we actually took
    return sample;
}
