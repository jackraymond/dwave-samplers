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

// See https://www.science.org/doi/epdf/10.1126/sciadv.abe7953

inline double Equation17bracket(const double a0, const double atk, const double xitk, double c0, const vector<int> & neighbors, const vector<double>& neighbor_couplings, const vector<double> signxtk){
  
  double return_val = 0;
  for (auto j: neighbors){ // Tracking an effective field can speed up (in principle) when signxtk is slowly varying, in proportion to connectivity
      return_val += neighbor_couplings[j]*signxtk[j];
  }
  return -(a0 - atk)*xitk + c0*return_val;
}

void discrete_simulated_bifurcation_run(
    double* state_x,
    double* state_y,
    const int num_vars,
    const vector<vector<int>>& neighbors,
    const vector<vector<double>>& neighbor_couplings,
    const vector<double>& a_schedule,
    const double a0,
    const double c0,
    const double Delta_t
) {
    std::vector<double> signxtk(num_vars);
    for (int varI = 0; varI < num_vars; varI++) {
      signxtk[varI] = (state_x[varI] > 0) - (state_x[varI] < 0);
    }  
    for (auto atk: a_schedule){
        for (int varI = 0; varI < num_vars; varI++) {
            state_y[varI] += Delta_t*Equation17bracket(a0, atk, state_x[varI], c0, neighbors[varI], neighbor_couplings[varI], signxtk);  // Eq 17
        }
        for (int varI = 0; varI < num_vars; varI++) {
	    state_x[varI] += Delta_t*a0*state_y[varI];  // Eq 18
        }
	for (int varI = 0; varI < num_vars; varI++) {
	  signxtk[varI] = (state_x[varI] > 0) - (state_x[varI] < 0);
	  // Avoided if statements are assumed to help the compiler + efficiency:
	  double inrange = (abs(state_x[varI]) > 1);
	  state_y[varI] *= inrange;  // Zero out
	  state_x[varI] = inrange*state_x[varI] + (1 - inrange)*signxtk[varI]; // Threshold  
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
int general_discrete_simulated_bifurcation_machine(
    double* states_x,
    double* states_y,
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
    
    // neighbors is a vector of vectors, such that neighbors[i][j] is the jth
    // neighbor of variable i
    vector<vector<int>> neighbors(num_vars);
    // neighbor_couplings is another vector of vectors with the same structure
    // except neighbor_couplings[i][j] is the weight on the coupling between i
    // and its jth neighbor
    vector<vector<double>> neighbor_couplings(num_vars);

    // build the neighbors, and neighbor_couplings vectors by
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
        neighbor_couplings[u].push_back(coupler_weights[cplr]);
        neighbor_couplings[v].push_back(coupler_weights[cplr]);

    }


    // get the sbm samples
    int sample = 0;
    std::vector<double> dstate_x(num_vars);  // Reusable buffer
    std::vector<double> dstate_y(num_vars);  // Reusable buffer
    while (sample < num_samples) {
        // states is a giant spin array that will hold the resulting states for
        // all the samples, so we need to get the location inside that vector
        // where we will store the sample for this sample
        double *state_x = states_x + sample*num_vars;
        double *state_y = states_y + sample*num_vars;
        // then do the actual sample. this function will modify state, storing
        // the sample there
        // Branching here is designed to make expicit compile time optimizations
	discrete_simulated_bifurcation_run(state_x, state_y, num_vars,
					   neighbors, neighbor_couplings,
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
