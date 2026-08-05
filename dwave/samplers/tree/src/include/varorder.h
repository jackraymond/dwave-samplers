/**
# Copyright 2019 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# =============================================================================
*/
#ifndef INCLUDED_ORANG_VARORDER_H
#define INCLUDED_ORANG_VARORDER_H

#include <cstddef>
#include <cmath>
#include <algorithm>
#include <set>
#include <vector>
#include <utility>
#include <iterator>
#include <limits>
#include <memory>
#include <cassert>
#include <cstdint>
#include <chrono>
#include <functional>
#include <string>
#include <unordered_map>

#include <base.h>
#include <exception.h>
#include <graph.h>
#include <task.h>

namespace dwave::samplers::tree {

namespace greedyvarorder {
namespace internal {

struct Variable;
typedef std::shared_ptr<Variable> var_ptr;


struct Variable {
  const Var index;
  const double domSize;
  bool processed;
  int clampRank;
  double clampValue;
  double cost;
  double complexity;
  VarSet adjList;

  Variable(Var index0, const TaskBase& task, const std::vector<int>& clampRanks) :
    index(index0),
    domSize(task.domSize(index0)),
    processed(clampRanks[index0] < 0),
    clampRank(clampRanks[index0]),
    clampValue(),
    cost(),
    complexity(),
    adjList() {

    const Graph& g = task.graph();

    for (auto it = g.adjacencyBegin(index); it != g.adjacencyEnd(index); ++it) {
      if (clampRanks[*it] >= 0) {
        adjList.insert(*it);
      }
    }
  }

  Variable(Var index0, double domSize0, bool processed0, int clampRank0, double clampValue0,
      double cost0, double complexity0) :
        index(index0), domSize(domSize0), processed(processed0), clampRank(clampRank0), clampValue(clampValue0),
        cost(cost0), complexity(complexity0), adjList() {}

  static var_ptr upperBound(const Variable& var) {
    return std::make_shared<Variable>(std::numeric_limits<Var>::max(), var.domSize, var.processed, var.clampRank, var.clampValue,
                                      var.cost, var.complexity);
  }

  static var_ptr complexityUpperBound(double maxComplexity) {
    return std::make_shared<Variable>(std::numeric_limits<Var>::max(), 0.0, false, 0, 0.0,
                                      std::numeric_limits<double>::infinity(), maxComplexity);
  }

  static var_ptr clampRankUpperBound(int rank) {
    return std::make_shared<Variable>(std::numeric_limits<Var>::max(), 0.0, false, rank,
                                      -std::numeric_limits<double>::infinity(), 0.0, 0.0);
  }
};



//===================================================================================================================
//
//   C O M P A R I S O N   F U N C T O R S
//
//===================================================================================================================

/*
 * Var objects are sorted as follows:
 * 1. Processed variables appear last and are not sorted further.
 * 2. Within unprocessed variables, those whose complexity exceeds the maximum appear last and are not sorted further.
 * 3. Below-complexity-limit variables are sorted by increasing cost, with ties broken by variable index.
 */
class CostCmp {
private:
  double maxComplexity_;

public:
  CostCmp(double maxComplexity) : maxComplexity_(maxComplexity) {}

  bool operator()(const var_ptr& v1, const var_ptr& v2) const {
    return !v1->processed
        && (v2->processed
            || (v1->complexity <= maxComplexity_
                && (v2->complexity > maxComplexity_
                    || v1->cost < v2->cost
                    || (v1->cost == v2->cost && v1->index < v2->index))));
  }
};

struct ClampCmp {
  bool operator()(const var_ptr& v1, const var_ptr& v2) const {
    return !v1->processed
        && (v2->processed
            || v1->clampRank < v2->clampRank
            || (v1->clampRank == v2->clampRank
                && (v1->clampValue > v2->clampValue
                    || (v1->clampValue == v2->clampValue && v1->index < v2->index))));
  }
};



//===================================================================================================================
//
//   M U L T I - I N D E X   V A R I A B L E   C O N T A I N E R
//
//===================================================================================================================


/*
 // * Container of variables that provides multiple modes of access.  Random
 // * access is provided through a vector, while two sortings are also
 // * maintained through multisets.  Each internal container stores smart
 // * pointers to Variable instances.  Functions are provided for modifying an
 // * element specified by an iterator of any particular container.  Internally,
 // * modification is done by removing and adding back elements to the
 // * multisets, to preserve the order.
 */
class VarContainer {
public:
  std::vector<var_ptr> byIndex;
  std::multiset<var_ptr, CostCmp> byCost;
  std::multiset<var_ptr, ClampCmp> byClamp;

  VarContainer(const TaskBase& task, double maxComplexity, const std::vector<int>& clampRank)
    : byCost(CostCmp(maxComplexity)) {
    const Graph& g = task.graph();
    Var numVertices = g.numVertices();

    for (Var v = 0; v < numVertices; ++v) {
      add(std::make_shared<Variable>(v, task, clampRank));
    }
  }

  void add(var_ptr var) {
    byIndex.push_back(var);
    byCost.insert(var);
    byClamp.insert(var);
  }

  template<typename F>
  void modifyByIndex(std::vector<var_ptr>::iterator it, F& func) {
    func(**it);

    auto pos = find(byCost.begin(), byCost.end(), *it);
    assert(pos != byCost.end());
    byCost.erase(pos);
    byCost.insert(*it);

    pos = find(byClamp.begin(), byClamp.end(), *it);
    assert(pos != byClamp.end());
    byClamp.erase(pos);
    byClamp.insert(*it);
  }

  template<typename F>
  void modifyByCost(std::multiset<var_ptr, CostCmp>::iterator it, F func) {
    auto pos_index = find(byIndex.begin(), byIndex.end(), *it);
    func(**pos_index);

    byCost.erase(it);
    byCost.insert(*pos_index);

    auto pos = find(byClamp.begin(), byClamp.end(), *it);
    assert(pos != byClamp.end());
    byClamp.erase(pos);
    byClamp.insert(*pos_index);
  }

  template<typename F>
  void modifyByClamp(std::multiset<var_ptr, ClampCmp>::iterator it, F func) {
    auto pos_index = find(byIndex.begin(), byIndex.end(), *it);
    func(**pos_index);

    byClamp.erase(it);
    byClamp.insert(*pos_index);

    auto pos = find(byCost.begin(), byCost.end(), *it);
    assert(pos != byCost.end());
    byCost.erase(pos);
    byCost.insert(*pos_index);
  }

};



//===================================================================================================================
//
//   M O D I F I E R   F U N C T O R S
//
//===================================================================================================================


struct MarkAsProcessed {
  void operator()(Variable& var) const {
    var.processed = true;
  }
};

struct DecrementClampRank {
  void operator()(Variable& var) const {
    --var.clampRank;
  }
};

class ElimNeighbour {
private:
  const Var elimVar_;
  const VarSet& vars_;

public:
  ElimNeighbour(Var elimVar, const VarSet& vars = VarSet()) : elimVar_(elimVar), vars_(vars) {}
  void operator()(Variable& var) const {
    var.adjList.insert(vars_.begin(), vars_.end());
    var.adjList.erase(var.index);
    var.adjList.erase(elimVar_);
  }
};

class ClampNeighbour {
private:
  const Var clampVar_;

public:
  ClampNeighbour(Var clampVar) : clampVar_(clampVar) {}
  void operator()(Variable& var) const {
    var.adjList.erase(clampVar_);
  }
};



//===================================================================================================================
//
//   H E U R I S T I C - S P E C I F I C   S T U F F
//
//===================================================================================================================


//-------------------------------------------------------------------------------------------------------------------
// Var member data modifier functors
//-------------------------------------------------------------------------------------------------------------------

/*
 * These values are based on current contents of the variable's adjList and the adjList contents of its neighbours;
 * thus, this functor must be applied to all appropriate variables AFTER UpdateNeighbours has been applied to ALL
 * those same variables.
 *
 * Cost calculation is heuristic-dependent.  Derived classes exist for the different calculations.
 */
class UpdateVarData {
private:
  virtual void updateCost(Variable& var) const = 0;

protected:
  const std::vector<var_ptr>& varsByIndex_;

public:
  UpdateVarData(const std::vector<var_ptr>& varsByIndex) : varsByIndex_(varsByIndex) {}
  virtual ~UpdateVarData() {}

  void operator()(Variable& var) const {
    var.clampValue = static_cast<double>(var.domSize) * static_cast<double>(var.adjList.size());
    double p2Cplx = var.domSize;
    for (const auto &w: var.adjList) {
      p2Cplx *= varsByIndex_[w]->domSize;
    }
    static const double E_LOG2 = 1.4426950408889633;
    var.complexity = log(p2Cplx) * E_LOG2;
    updateCost(var);
  }
};

class UpdateMinDegreeVarData : public UpdateVarData {
private:
  virtual void updateCost(Variable& var) const {
    var.cost = static_cast<double>(var.adjList.size());
  }
public:
  UpdateMinDegreeVarData(std::vector<var_ptr>& varsByIndex) : UpdateVarData(varsByIndex) {}
};

class UpdateWeightedMinDegreeVarData : public UpdateVarData {
private:
  virtual void updateCost(Variable& var) const {
    var.cost = var.clampValue;
  }
public:
  UpdateWeightedMinDegreeVarData(const std::vector<var_ptr>& varsByIndex) : UpdateVarData(varsByIndex) {}
};

class UpdateMinFillVarData : public UpdateVarData {
private:
  virtual void updateCost(Variable& var) const {
    var.cost = 0.0;
    for (VarSet::const_iterator vAdjIter = var.adjList.begin(), vAdjEnd = var.adjList.end();
        vAdjIter != vAdjEnd; ++vAdjIter) {
      const Var u = *vAdjIter;
      const Variable* uVar = varsByIndex_[u].get();
      VarSet::const_iterator uAdjIter = uVar->adjList.upper_bound(u);
      VarSet::const_iterator uAdjEnd = uVar->adjList.end();
      VarSet::const_iterator vAdjIter2 = vAdjIter;
      ++vAdjIter2;
      while (vAdjIter2 != vAdjEnd) {
        if (uAdjIter == uAdjEnd || *vAdjIter2 < *uAdjIter) {
          ++var.cost;
          ++vAdjIter2;
        } else if (*uAdjIter < *vAdjIter2) {
          ++uAdjIter;
        } else {
          ++vAdjIter2;
          ++uAdjIter;
        }
      }
    }
  }
public:
  UpdateMinFillVarData(const std::vector<var_ptr>& varsByIndex) : UpdateVarData(varsByIndex) {}
};

class UpdateWeightedMinFillVarData : public UpdateVarData {
private:
  virtual void updateCost(Variable& var) const {
    var.cost = 0.0;
    for (VarSet::const_iterator vAdjIter = var.adjList.begin(), vAdjEnd = var.adjList.end();
        vAdjIter != vAdjEnd; ++vAdjIter) {
      const Var u = *vAdjIter;
      const Variable* uVar = varsByIndex_[u].get();
      VarSet::const_iterator uAdjIter = uVar->adjList.upper_bound(u);
      VarSet::const_iterator uAdjEnd = uVar->adjList.end();
      VarSet::const_iterator vAdjIter2 = vAdjIter;
      ++vAdjIter2;
      double cost = 0.0;
      while (vAdjIter2 != vAdjEnd) {
        if (uAdjIter == uAdjEnd || *vAdjIter2 < *uAdjIter) {
          cost += varsByIndex_[*vAdjIter2]->domSize;
          ++vAdjIter2;
        } else if (*uAdjIter < *vAdjIter2) {
          ++uAdjIter;
        } else {
          ++vAdjIter2;
          ++uAdjIter;
        }
      }
      var.cost += uVar->domSize * cost;
    }
  }
public:
  UpdateWeightedMinFillVarData(const std::vector<var_ptr>& varsByIndex) : UpdateVarData(varsByIndex) {}
};


//-------------------------------------------------------------------------------------------------------------------
// List-of-affected-variables functors
//-------------------------------------------------------------------------------------------------------------------

class AffectedVars {
private:
  virtual VarSet affectedVars(const Variable&) const = 0;
public:
  virtual ~AffectedVars() {}
  VarSet operator()(const Variable& var) const {
    return affectedVars(var);
  }
};

class MinDegreeAffectedVars : public AffectedVars {
private:
  virtual VarSet affectedVars(const Variable& var) const {
    return var.adjList;
  }
};

class MinFillAffectedVars : public AffectedVars {
private:
  const std::vector<var_ptr>& varsByIndex_;
  virtual VarSet affectedVars(const Variable& var) const {
    VarSet vars = var.adjList;
    for (const auto &u: var.adjList) {
      const Variable* uVar = varsByIndex_[u].get();
      vars.insert(uVar->adjList.begin(), uVar->adjList.end());
    }
    vars.erase(var.index);
    return vars;
  }

public:
  MinFillAffectedVars(const VarContainer& varContainer) : varsByIndex_(varContainer.byIndex) {}
};



//===================================================================================================================
//
//   R A N D O M   V A R I A B L E   S E L E C T O R
//
//===================================================================================================================

template<typename Iter, typename Rng>
Iter selectVar(Iter begin, Iter baseEnd, Iter finalEnd, Rng& rng, float selectionScale) {

  float baseRange = static_cast<float>(std::distance(begin, baseEnd));
  float totalRange = baseRange + static_cast<float>(std::distance(baseEnd, finalEnd));
  float selectionRange = std::min(baseRange * selectionScale, totalRange);
  float incr = std::floor(static_cast<float>(selectionRange * rng()));
  incr = std::max(incr, 0.0f);
  incr = std::min(incr, totalRange - 1);

  Iter ret = begin;
  std::advance(ret, incr);
  return ret;
}

inline std::size_t countBits(std::uint64_t value) {
  std::size_t count = 0;
  while (value) {
    value &= value - 1;
    ++count;
  }
  return count;
}

inline std::size_t leastSignificantBitIndex(std::uint64_t value) {
  std::size_t index = 0;
  while ((value & 1ULL) == 0ULL) {
    value >>= 1;
    ++index;
  }
  return index;
}

inline void eliminateVertex(
    std::vector<std::uint64_t>& adjacency,
    std::size_t vertex,
    std::uint64_t activeMask) {

  const std::uint64_t neighbours = adjacency[vertex] & activeMask;

  for (std::uint64_t left = neighbours; left; left &= (left - 1ULL)) {
    const std::size_t u = leastSignificantBitIndex(left);
    for (std::uint64_t right = left & (left - 1ULL); right; right &= (right - 1ULL)) {
      const std::size_t w = leastSignificantBitIndex(right);
      adjacency[u] |= (1ULL << w);
      adjacency[w] |= (1ULL << u);
    }
  }

  adjacency[vertex] = 0ULL;
  for (std::uint64_t it = neighbours; it; it &= (it - 1ULL)) {
    const std::size_t u = leastSignificantBitIndex(it);
    adjacency[u] &= ~(1ULL << vertex);
  }
}

inline double minFillCost(
    const std::vector<std::uint64_t>& adjacency,
    std::size_t vertex,
    std::uint64_t activeMask) {

  const std::uint64_t neighbours = adjacency[vertex] & activeMask;
  const std::size_t degree = countBits(neighbours);
  if (degree < 2) {
    return 0.0;
  }

  std::size_t existingEdges = 0;
  for (std::uint64_t left = neighbours; left; left &= (left - 1ULL)) {
    const std::size_t u = leastSignificantBitIndex(left);
    existingEdges += countBits((adjacency[u] & activeMask) & (left & (left - 1ULL)));
  }

  const std::size_t completeEdges = degree * (degree - 1) / 2;
  return static_cast<double>(completeEdges - existingEdges);
}

inline double eliminationComplexity(
    const std::vector<std::uint64_t>& adjacency,
    const std::vector<double>& domSizes,
    std::size_t vertex,
    std::uint64_t activeMask) {

  double p2Complexity = domSizes[vertex];
  for (std::uint64_t neighbours = adjacency[vertex] & activeMask; neighbours; neighbours &= (neighbours - 1ULL)) {
    const std::size_t u = leastSignificantBitIndex(neighbours);
    p2Complexity *= domSizes[u];
  }

  static const double E_LOG2 = 1.4426950408889633;
  return std::log(p2Complexity) * E_LOG2;
}

struct ExactStateKey {
  std::uint64_t activeMask;
  std::vector<std::uint64_t> adjacency;

  bool operator==(const ExactStateKey& other) const {
    return activeMask == other.activeMask && adjacency == other.adjacency;
  }
};

struct ExactStateHash {
  std::size_t operator()(const ExactStateKey& key) const {
    std::size_t h = std::hash<std::uint64_t>{}(key.activeMask);
    for (auto value: key.adjacency) {
      h ^= std::hash<std::uint64_t>{}(value + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
    }
    return h;
  }
};

inline std::pair<double, std::vector<std::size_t>> greedyMinFillUpperBound(
    const std::vector<std::uint64_t>& baseAdjacency,
    const std::vector<double>& domSizes) {

  constexpr std::size_t kMaskBits = std::numeric_limits<std::uint64_t>::digits;
  const std::size_t n = domSizes.size();
  std::vector<std::uint64_t> adjacency = baseAdjacency;
  std::uint64_t activeMask = (n == kMaskBits) ? std::numeric_limits<std::uint64_t>::max() : ((1ULL << n) - 1ULL);

  std::vector<std::size_t> order;
  order.reserve(n);
  double maxSeenComplexity = 0.0;

  while (activeMask) {
    std::size_t bestVertex = n;
    double bestFill = std::numeric_limits<double>::infinity();
    double bestComplexity = std::numeric_limits<double>::infinity();

    for (std::uint64_t bits = activeMask; bits; bits &= (bits - 1ULL)) {
      const std::size_t v = leastSignificantBitIndex(bits);
      const double fill = minFillCost(adjacency, v, activeMask & ~(1ULL << v));
      const double complexity = eliminationComplexity(adjacency, domSizes, v, activeMask & ~(1ULL << v));

      if (fill < bestFill || (fill == bestFill && (complexity < bestComplexity
          || (complexity == bestComplexity && v < bestVertex)))) {
        bestVertex = v;
        bestFill = fill;
        bestComplexity = complexity;
      }
    }

    maxSeenComplexity = std::max(maxSeenComplexity, bestComplexity);
    order.push_back(bestVertex);
    eliminateVertex(adjacency, bestVertex, activeMask & ~(1ULL << bestVertex));
    activeMask &= ~(1ULL << bestVertex);
  }

  return std::make_pair(maxSeenComplexity, order);
}

inline VarVector exactVarOrder(
    const TaskBase& task,
    double minComplexity,
    double maxComplexity,
    const std::vector<int>& clampRank,
    double timeoutSeconds,
    bool* timedOutOut = nullptr,
    std::string* timeoutMessageOut = nullptr) {

  // Implementation compromise notes:
  // - This is an exact branch-and-bound search with memoization over residual
  //   graph states, not the potential-maximal-clique/minimal-separator
  //   algorithm family from Fomin & Villanger (arXiv:0803.1321).
  // - Branching order is guided by local complexity/min-fill scores for speed,
  //   but those scores are heuristic only and do not change exactness.
  // - The optimized objective here is the maximum elimination complexity under
  //   maxComplexity constraints; min-fill is used for ordering/tie-breaking.

  if (!(timeoutSeconds > 0.0) || !std::isfinite(timeoutSeconds)) {
    throw InvalidArgumentException("EXACT heuristic timeoutSeconds must be a finite value greater than 0");
  }
  if (!std::isfinite(minComplexity)) {
    throw InvalidArgumentException("EXACT heuristic minComplexity must be finite");
  }
  if (minComplexity > maxComplexity) {
    throw InvalidArgumentException("EXACT heuristic minComplexity must be less than or equal to maxComplexity");
  }

  const Var numVars = task.numVars();
  std::vector<Var> activeVars;
  activeVars.reserve(numVars);

  for (Var v = 0; v < numVars; ++v) {
    if (clampRank[v] >= 0) {
      activeVars.push_back(v);
    }
  }

  if (activeVars.empty()) {
    return VarVector();
  }

  for (Var v: activeVars) {
    if (clampRank[v] != 0) {
      throw InvalidArgumentException("EXACT heuristic currently requires unclamped variables to have clampRank 0");
    }
  }

  const std::size_t n = activeVars.size();
  constexpr std::size_t kMaskBits = std::numeric_limits<std::uint64_t>::digits;
  if (n > kMaskBits) {
    throw InvalidArgumentException("EXACT heuristic supports at most 64 active variables");
  }

  std::vector<double> domSizes(n, 0.0);
  std::vector<std::uint64_t> baseAdjacency(n, 0ULL);
  std::vector<int> denseIndex(numVars, -1);

  for (std::size_t i = 0; i < n; ++i) {
    denseIndex[activeVars[i]] = static_cast<int>(i);
    domSizes[i] = task.domSize(activeVars[i]);
  }

  const Graph& g = task.graph();
  for (std::size_t i = 0; i < n; ++i) {
    for (auto it = g.adjacencyBegin(activeVars[i]); it != g.adjacencyEnd(activeVars[i]); ++it) {
      const int j = denseIndex[*it];
      if (j >= 0 && static_cast<std::size_t>(j) != i) {
        baseAdjacency[i] |= (1ULL << static_cast<std::size_t>(j));
      }
    }
  }

  bool targetReached = false;
  std::vector<std::size_t> incumbentOrder;
  double incumbentBest = std::numeric_limits<double>::infinity();
  {
    // Compromise for performance only: use a greedy incumbent to tighten
    // pruning early. Search still explores alternatives and remains exact.
    auto greedySeed = greedyMinFillUpperBound(baseAdjacency, domSizes);
    incumbentBest = greedySeed.first;
    incumbentOrder = std::move(greedySeed.second);
    if (incumbentBest <= minComplexity) {
      targetReached = true;
    }
  }

  std::unordered_map<ExactStateKey, double, ExactStateHash> memo;
  std::vector<std::size_t> currentOrder;
  currentOrder.reserve(n);

  const auto startTime = std::chrono::steady_clock::now();
  std::size_t statesVisited = 0;
  std::size_t maxDepthReached = 0;
  static constexpr std::size_t timeoutCheckInterval = 1024;
  bool timedOut = false;
  std::string timeoutMessage;

  std::function<void(const std::vector<std::uint64_t>&, std::uint64_t, double)> dfs;
  dfs = [&](const std::vector<std::uint64_t>& adjacency, std::uint64_t activeMask, double currentWorst) {
    if (timedOut || targetReached) {
      return;
    }

    ++statesVisited;
    maxDepthReached = std::max(maxDepthReached, currentOrder.size());
    if ((statesVisited % timeoutCheckInterval) == 0) {
      const auto now = std::chrono::steady_clock::now();
      const double elapsed = std::chrono::duration<double>(now - startTime).count();
      if (elapsed >= timeoutSeconds) {
        timeoutMessage =
            "EXACT heuristic timed out after " + std::to_string(timeoutSeconds) +
            " seconds; progress: visited " + std::to_string(statesVisited) +
            " states, reached depth " + std::to_string(maxDepthReached) +
            "/" + std::to_string(n);
        if (std::isfinite(incumbentBest)) {
          timeoutMessage += ", best upper bound complexity=" + std::to_string(incumbentBest);
        }
        timedOut = true;
        return;
      }
    }

    if (!activeMask) {
      if (currentWorst < incumbentBest) {
        incumbentBest = currentWorst;
        incumbentOrder = currentOrder;
        if (incumbentBest <= minComplexity) {
          targetReached = true;
        }
      }
      return;
    }

    if (currentWorst >= incumbentBest || currentWorst > maxComplexity) {
      return;
    }

    ExactStateKey key;
    key.activeMask = activeMask;
    key.adjacency = adjacency;
    auto memoIt = memo.find(key);
    if (memoIt != memo.end() && memoIt->second <= currentWorst) {
      return;
    }
    memo[key] = currentWorst;

    double bestPossibleNext = std::numeric_limits<double>::infinity();
    for (std::uint64_t bits = activeMask; bits; bits &= (bits - 1ULL)) {
      const std::size_t v = leastSignificantBitIndex(bits);
      bestPossibleNext = std::min(bestPossibleNext,
          eliminationComplexity(adjacency, domSizes, v, activeMask & ~(1ULL << v)));
    }
    if (std::max(currentWorst, bestPossibleNext) >= incumbentBest) {
      return;
    }

    struct Candidate {
      std::size_t vertex;
      double fill;
      double complexity;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(countBits(activeMask));

    for (std::uint64_t bits = activeMask; bits; bits &= (bits - 1ULL)) {
      const std::size_t v = leastSignificantBitIndex(bits);
      const std::uint64_t nextMask = activeMask & ~(1ULL << v);
      candidates.push_back({
          v,
          minFillCost(adjacency, v, nextMask),
          eliminationComplexity(adjacency, domSizes, v, nextMask)});
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
      if (a.complexity != b.complexity) return a.complexity < b.complexity;
      if (a.fill != b.fill) return a.fill < b.fill;
      return a.vertex < b.vertex;
    });

    for (const auto& candidate: candidates) {
      const double nextWorst = std::max(currentWorst, candidate.complexity);
      if (nextWorst >= incumbentBest || nextWorst > maxComplexity) {
        continue;
      }

      const std::uint64_t nextMask = activeMask & ~(1ULL << candidate.vertex);
      std::vector<std::uint64_t> nextAdjacency = adjacency;
      eliminateVertex(nextAdjacency, candidate.vertex, nextMask);

      currentOrder.push_back(candidate.vertex);
      dfs(nextAdjacency, nextMask, nextWorst);
      currentOrder.pop_back();
    }
  };

  const std::uint64_t fullMask = (n == kMaskBits) ? std::numeric_limits<std::uint64_t>::max() : ((1ULL << n) - 1ULL);
  dfs(baseAdjacency, fullMask, 0.0);

  if (timedOutOut != nullptr) {
    *timedOutOut = timedOut;
  }
  if (timeoutMessageOut != nullptr) {
    *timeoutMessageOut = timeoutMessage;
  }

  if (incumbentOrder.empty() || incumbentBest > maxComplexity) {
    throw InvalidArgumentException("EXACT heuristic could not find an elimination order under maxComplexity");
  }

  VarVector order;
  order.reserve(n);
  for (const auto idx: incumbentOrder) {
    order.push_back(activeVars[idx]);
  }

  return order;
}

} // namespace dwave::samplers::tree::greedyvarorder::internal




//===================================================================================================================
//
//   H E U R I S T I C   E N U M
//
//===================================================================================================================

enum Heuristics {
  MIN_DEGREE,
  WEIGHTED_MIN_DEGREE,
  MIN_FILL,
  WEIGHTED_MIN_FILL,
  EXACT,
  NUM_HEURISTICS
};

} // namespace dwave::samplers::tree::greedyvarorder



//===================================================================================================================
//
//   T H E   F U N C T I O N
//
//===================================================================================================================


template<typename Rng>
VarVector greedyVarOrder(
    const TaskBase& task,
    double minComplexity,
    double maxComplexity,
    const std::vector<int>& clampRank,
    greedyvarorder::Heuristics h,
    Rng& rng,
    float selectionScale = 1.0f,
    double timeoutSeconds = 60.0,
    bool* timedOutOut = nullptr,
    std::string* timeoutMessageOut = nullptr) {

  using std::floor;
  using std::advance;
  using std::distance;
  using namespace greedyvarorder::internal;

  if (task.numVars() != clampRank.size()) {
    throw InvalidArgumentException("clampRank size must equal the number of variables in task");
  }

  if (task.numVars() == 0) {
    return VarVector();
  }

  if (h == greedyvarorder::EXACT) {
    return exactVarOrder(task, minComplexity, maxComplexity, clampRank, timeoutSeconds, timedOutOut, timeoutMessageOut);
  }

  VarContainer vars(task, maxComplexity, clampRank);

  std::unique_ptr<UpdateVarData> updateCostPtr;
  std::unique_ptr<AffectedVars> affectedVarsPtr;
  switch (h) {
    case greedyvarorder::MIN_DEGREE:
      updateCostPtr.reset( new UpdateMinDegreeVarData(vars.byIndex) );
      affectedVarsPtr.reset( new MinDegreeAffectedVars() );
      break;
    case greedyvarorder::WEIGHTED_MIN_DEGREE:
      updateCostPtr.reset( new UpdateWeightedMinDegreeVarData(vars.byIndex) );
      affectedVarsPtr.reset( new MinDegreeAffectedVars() );
      break;
    case greedyvarorder::MIN_FILL:
      updateCostPtr.reset( new UpdateMinFillVarData(vars.byIndex) );
      affectedVarsPtr.reset( new MinFillAffectedVars(vars) );
      break;
    case greedyvarorder::WEIGHTED_MIN_FILL:
      updateCostPtr.reset( new UpdateWeightedMinFillVarData(vars.byIndex) );
      affectedVarsPtr.reset( new MinFillAffectedVars(vars) );
      break;
    default:
      throw InvalidArgumentException("Invalid heuristic");
  }

  for (auto iter = vars.byIndex.begin(); iter != vars.byIndex.end(); ++iter) {
    vars.modifyByIndex(iter, *updateCostPtr);
  }

  VarVector varOrder;
  int lastClampRank = -1;
  const var_ptr complexityUpper = Variable::complexityUpperBound(maxComplexity);

  for (;;) {

    auto minCostLower = vars.byCost.begin();
    if ((*minCostLower)->processed) {
      break;
    }

    if ((*minCostLower)->complexity <= maxComplexity) {
      auto pickedIter = selectVar(minCostLower, vars.byCost.upper_bound( Variable::upperBound(**minCostLower)),
          vars.byCost.upper_bound(complexityUpper), rng, selectionScale);

      const Variable& v = **pickedIter;
      varOrder.push_back(v.index);
      VarSet affectedVars = (*affectedVarsPtr)(v);
      ElimNeighbour elimNeighbour(v.index, v.adjList);

      vars.modifyByCost(pickedIter, MarkAsProcessed());

      for (const auto &uIndex: v.adjList) {
        vars.modifyByIndex(vars.byIndex.begin() + uIndex, elimNeighbour);
      }

      for (const auto &uIndex: affectedVars) {
        vars.modifyByIndex(vars.byIndex.begin() + uIndex, *updateCostPtr);
      }

    } else {
      if (lastClampRank >= 0) {
        auto clampIter = vars.byClamp.upper_bound( Variable::clampRankUpperBound(lastClampRank) );
        auto clampEnd = vars.byClamp.end();
        while (clampIter != clampEnd && !(*clampIter)->processed) {
          auto here = clampIter++;
          vars.modifyByClamp(here, DecrementClampRank());
        }
      }

      auto clampLower = vars.byClamp.begin();
      auto pickedIter = selectVar(clampLower,
                                  vars.byClamp.upper_bound( Variable::upperBound(**clampLower) ),
                                  vars.byClamp.upper_bound( Variable::clampRankUpperBound((*clampLower)->clampRank) ),
                                  rng, selectionScale);

      const Variable& v = **pickedIter;
      lastClampRank = v.clampRank;
      vars.modifyByClamp(pickedIter, MarkAsProcessed());
      ClampNeighbour clampNeighbour(v.index);

      for (const auto &uIndex: v.adjList) {
        vars.modifyByIndex(vars.byIndex.begin() + uIndex, clampNeighbour);
      }

      for (const auto &uIndex: v.adjList) {
        vars.modifyByIndex(vars.byIndex.begin() + uIndex, *updateCostPtr);
      }
    }
  }

  return varOrder;
}

} // namespace dwave::samplers::tree

#endif
