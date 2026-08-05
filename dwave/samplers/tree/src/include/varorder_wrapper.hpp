// Copyright 2026 D-Wave Systems Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef INCLUDED_ORANG_VARORDER_WRAPPER_HPP
#define INCLUDED_ORANG_VARORDER_WRAPPER_HPP

#include <cstdint>
#include <random>
#include <set>
#include <string>
#include <stdexcept>
#include <vector>

#include <orang.h>

namespace dwave::samplers::tree {

namespace internal {

class UniformRng {
private:
  std::mt19937 engine_;
  std::uniform_real_distribution<double> distribution_;

public:
  explicit UniformRng(std::uint32_t seed) :
      engine_(seed),
      distribution_(0.0, 1.0) {}

  double operator()() {
    return distribution_(engine_);
  }
};

}  // namespace internal

class AdjacencyTask : public TaskBase {
public:
  explicit AdjacencyTask(const std::vector<std::vector<int>>& adjacency) {
    const Var numVars = static_cast<Var>(adjacency.size());

    domSizes_.assign(numVars, 2);

    std::set<Graph::adj_pair> adjSet;
    for (Var i = 0; i < numVars; ++i) {
      for (const int jRaw: adjacency[i]) {
        if (jRaw < 0) {
          throw std::invalid_argument("adjacency contains a negative vertex index");
        }

        const Var j = static_cast<Var>(jRaw);
        if (j >= numVars) {
          throw std::invalid_argument("adjacency contains a vertex index outside range");
        }

        if (i < j) {
          adjSet.insert(Graph::adj_pair(i, j));
        }
      }
    }

    graph_.setAdjacencies(adjSet, numVars);
  }
};

inline std::vector<int> greedyVarOrderAdjacency(
    const std::vector<std::vector<int>>& adjacency,
    double minComplexity,
    double maxComplexity,
    const std::vector<int>& clampRank,
    int heuristic,
    int seed,
    float selectionScale = 1.0f,
    double timeoutSeconds = 60.0,
    bool* timedOutOut = nullptr,
    std::string* timeoutMessageOut = nullptr) {

  if (heuristic < 0 || heuristic >= static_cast<int>(greedyvarorder::NUM_HEURISTICS)) {
    throw std::invalid_argument("invalid heuristic");
  }

  VarVector order;
  try {
    internal::UniformRng rng(static_cast<std::uint32_t>(seed));
    AdjacencyTask task(adjacency);

    order = greedyVarOrder(
        task,
        minComplexity,
        maxComplexity,
        clampRank,
        static_cast<greedyvarorder::Heuristics>(heuristic),
        rng,
        selectionScale,
        timeoutSeconds,
        timedOutOut,
        timeoutMessageOut);
  } catch (const Exception& e) {
    throw std::runtime_error(e.what());
  }

  std::vector<int> output;
  output.reserve(order.size());
  for (auto v: order) {
    output.push_back(static_cast<int>(v));
  }

  return output;
}

}  // namespace dwave::samplers::tree

#endif