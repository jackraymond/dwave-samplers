# Copyright 2019 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from cython.operator cimport preincrement as inc, dereference as deref

from libcpp cimport bool as cppbool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set

import dimod
import numpy as np
import warnings
cimport dimod


__all__ = ['elimination_order_width', 'min_fill_heuristic', 'greedy_var_order']


cdef extern from "varorder_wrapper.hpp" namespace "dwave::samplers::tree":
    vector[int] greedyVarOrderAdjacency(
        const vector[vector[int]]& adjacency,
        double minComplexity,
        double maxComplexity,
        const vector[int]& clampRank,
        int heuristic,
        int seed,
        float selectionScale,
        double timeoutSeconds,
        cppbool* timedOutOut,
        string* timeoutMessageOut
    ) except +


cdef dict _HEURISTIC_TO_INT = {
    'MIN_DEGREE': 0,
    'WEIGHTED_MIN_DEGREE': 1,
    'MIN_FILL': 2,
    'WEIGHTED_MIN_FILL': 3,
    'EXACT': 4,
}


ctypedef unordered_map[Py_ssize_t, unordered_set[Py_ssize_t]] adj_t

cdef adj_t _cybqm_to_adj(dimod.cyBQM_float64 cybqm):
    cdef adj_t adj

    # add the nodes, even if they don't have any edges
    cdef Py_ssize_t vi
    for vi in range(cybqm.num_variables()):
        adj[vi]

    it = cybqm.data().cbegin_quadratic()
    while it != cybqm.data().cend_quadratic():
        adj[deref(it).u].insert(deref(it).v)
        adj[deref(it).v].insert(deref(it).u)
        inc(it)

    return adj


cdef vector[vector[int]] _adj_to_cpp(adj_t& adj, Py_ssize_t num_variables):
    cdef vector[vector[int]] cpp_adj
    cpp_adj.resize(num_variables)

    cdef Py_ssize_t vi
    cdef unordered_set[Py_ssize_t].iterator it
    cdef unordered_set[Py_ssize_t].iterator end
    for vi in range(num_variables):
        it = adj[vi].begin()
        end = adj[vi].end()
        while it != end:
            cpp_adj[vi].push_back(<int>deref(it))
            inc(it)

    return cpp_adj


cdef void _elim_adj(adj_t& adj, Py_ssize_t vi) except +:
    """Remove vi from adj and make its neighborhood a clique."""

    # make the neighborhood of vi a clique
    uit = adj[vi].begin()
    while uit != adj[vi].end():
        vit = uit
        inc(vit)  # no self-loops
        while vit != adj[vi].end():
            adj[deref(vit)].insert(deref(uit))
            adj[deref(uit)].insert(deref(vit))
            inc(vit)

        # remove vi from its neighbors
        adj[deref(uit)].erase(vi)

        inc(uit)

    # finally remove vi
    adj.erase(vi)


def elimination_order_width(bqm, order):
    """Calculates the width of the tree decomposition induced by a
    variable elimination order.

    order must contain exactly the variables of the bqm
    """

    if len(bqm) != len(order):
        raise ValueError("bqm and order must have the name variables")

    cdef dimod.cyBQM_float64 cybqm = dimod.as_bqm(bqm, dtype=float).data

    cdef adj_t adj = _cybqm_to_adj(cybqm)

    # if there is at least one node then the treewidth is at least 1
    cdef Py_ssize_t treewidth = cybqm.num_variables() > 0

    cdef Py_ssize_t vi
    for v in order:
        vi = cybqm.variables.index(v)

        if adj[vi].size() > treewidth:
            treewidth = adj[vi].size()

        _elim_adj(adj, vi)

    return treewidth


# dev note: adj is actually a const reference, but Cython does not like
# that (fixed in Cython 3.0)
cdef Py_ssize_t _min_num_edges(adj_t& adj):
    """Get the node that would need to add the fewest edges when eliminated.

    Only defined for len(adj) > 0.
    """
    # The goal is to go through each node in adj, and determine how many
    # edges we would need to add in order to eliminate it.

    # C++ lambdas don't work so well in Cython so we do 'min' the hard way...

    cdef Py_ssize_t min_num_edges = adj.size() * adj.size()
    cdef Py_ssize_t min_node  # our return value


    cdef Py_ssize_t num_edges
    cdef Py_ssize_t vi

    it = adj.begin()
    while it != adj.end():
        vi = deref(it).first

        # for all pairs of nodes in the neighborhood of vi, count the missing edges
        num_edges = 0
        uit = deref(it).second.begin()
        while uit != deref(it).second.end():
            vit = uit
            while vit != deref(it).second.end():
                if not adj[deref(vit)].count(deref(uit)):
                    num_edges += 1
                inc(vit)
            inc(uit)

        if num_edges < min_num_edges:
            min_num_edges = num_edges
            min_node = vi

        inc(it)

    return min_node


def min_fill_heuristic(bqm):
    """Compute an upper bound on the treewidth of the given bqm based on
    the min-fill heuristic for the elimination ordering.

    Args:
        bqm: a binary quadratic model

    Returns:
        A 2-tuple containing the bound on the treewidth and the elimination 
        order.

    """
    cdef dimod.cyBQM_float64 cybqm = dimod.as_bqm(bqm, dtype=float).data

    cdef adj_t adj = _cybqm_to_adj(cybqm)

    cdef vector[Py_ssize_t] order
    order.reserve(cybqm.num_variables())

    # if there is at least one node then the treewidth is at least 1
    cdef Py_ssize_t upper_bound = cybqm.num_variables() > 0

    cdef Py_ssize_t vi
    while adj.size():
        vi = _min_num_edges(adj)

        if adj[vi].size() > upper_bound:
            upper_bound = adj[vi].size()

        # remove vi from adj
        _elim_adj(adj, vi)

        order.push_back(vi)

    cdef Py_ssize_t i
    variables = []
    for i in range(order.size()):
        variables.append(cybqm.variables.at(order[i]))

    return upper_bound, variables


def greedy_var_order(bqm,
                     heuristic='MIN_FILL',
                     min_complexity=0.0,
                     max_complexity=float('inf'),
                     clamp_rank=None,
                     seed=None,
                     selection_scale=1.0,
                     timeout=60.0):
    """Compute an elimination order with the C++ greedyVarOrder implementation.

    Args:
        bqm:
            A binary quadratic model.

        heuristic:
            One of ``MIN_DEGREE``, ``WEIGHTED_MIN_DEGREE``, ``MIN_FILL``,
            ``WEIGHTED_MIN_FILL``, or ``EXACT``. ``EXACT`` uses an
            exact branch-and-bound implementation in ``varorder.h``.

            Compromises in ``EXACT``:
                - The method implements key features of the 
                  potential-maximal-clique/minimal-separator algorithm
                  arXiv:0803.1321, but with some simplifications so that
                  it does not achieve strictly best case scaling.
                - ``EXACT`` currently supports only unclamped/rank-0 active
                                    variables and is limited to at most 64 active variables.
                - The objective optimized by ``EXACT`` is minimizing the
                  maximum elimination complexity; min-fill scores are used only
                  for candidate ordering and initial upper bounds.

        max_complexity:
            Maximum allowed elimination complexity.

        min_complexity:
            Target elimination complexity for early stopping. For ``EXACT``,
            the search returns as soon as it finds an elimination order whose
            achieved upper bound on complexity is less than or equal to this
            value. Defaults to 0.0.

        clamp_rank:
            Optional clamp-rank data. If provided, must be a sequence of length
            ``num_variables`` in integer-variable order.

            Compromise:
                For ``EXACT``, nonzero entries for active variables are not
                supported by the current C++ implementation.

        seed:
            Optional RNG seed.
            For ``EXACT``, when provided, this seed is used to pseudo-randomly
            shuffle variable indexing before calling the C++ algorithm. This
            randomizes tie-breaking-sensitive outcomes while preserving the
            returned variable labels. When seed is None (default) the returned
            order is a deterministic function of the bqm.
            For non-``EXACT`` heuristics, this seed is forwarded directly to
            the C++ RNG.

        selection_scale:
            Selection scale forwarded to the C++ heuristic.

        timeout:
            Maximum runtime in seconds for the C++ search. Defaults to 60.
            For ``EXACT``, if the timeout is reached, a warning is emitted and
            the best elimination order found so far is returned together with a
            heuristic progress summary.

    Returns:
        A 2-tuple containing the achieved elimination complexity (treewidth)
        and the elimination order as a list of variables.
    """
    if isinstance(heuristic, str):
        heuristic_name = heuristic.upper()
        if heuristic_name not in _HEURISTIC_TO_INT:
            raise ValueError("unknown heuristic")
        heuristic_value = _HEURISTIC_TO_INT[heuristic_name]
    else:
        heuristic_value = int(heuristic)
        if heuristic_value < 0 or heuristic_value >= len(_HEURISTIC_TO_INT):
            raise ValueError("unknown heuristic")

    cdef adj_t adj
    cdef Py_ssize_t num_vars
    cdef list int_to_var
    cdef list new_to_old = []
    cdef list old_to_new = []
    cdef bint shuffle_for_exact = False
    cdef int exact_heuristic_value = _HEURISTIC_TO_INT['EXACT']

    if not isinstance(bqm, dimod.BinaryQuadraticModel):
        raise TypeError("bqm must be a dimod.BinaryQuadraticModel")

    cdef dimod.cyBQM_float64 cybqm = dimod.as_bqm(bqm, copy=False, dtype=float).data
    adj = _cybqm_to_adj(cybqm)
    num_vars = cybqm.num_variables()
    int_to_var = [cybqm.variables.at(i) for i in range(num_vars)]

    shuffle_for_exact = (seed is not None and heuristic_value == exact_heuristic_value)

    cdef vector[vector[int]] cpp_adj
    cdef Py_ssize_t vi
    cdef Py_ssize_t new_vi
    cdef unordered_set[Py_ssize_t].iterator it
    cdef unordered_set[Py_ssize_t].iterator end

    if shuffle_for_exact:
        perm = np.arange(num_vars, dtype=np.intc)
        np.random.default_rng(int(seed)).shuffle(perm)

        new_to_old = [0] * num_vars
        old_to_new = [0] * num_vars
        for new_vi in range(num_vars):
            vi = int(perm[new_vi])
            new_to_old[new_vi] = vi
            old_to_new[vi] = new_vi

        cpp_adj.resize(num_vars)
        for vi in range(num_vars):
            new_vi = old_to_new[vi]
            it = adj[vi].begin()
            end = adj[vi].end()
            while it != end:
                cpp_adj[new_vi].push_back(<int>old_to_new[deref(it)])
                inc(it)
    else:
        cpp_adj = _adj_to_cpp(adj, num_vars)

    cdef vector[int] clamp_rank_vec

    if clamp_rank is None:
        clamp_rank_vec.reserve(num_vars)
        for _ in range(num_vars):
            clamp_rank_vec.push_back(0)
    else:
        clamp_rank = list(clamp_rank)
        if len(clamp_rank) != num_vars:
            raise ValueError("clamp_rank length must equal number of variables")

        clamp_rank_vec.reserve(num_vars)
        if shuffle_for_exact:
            for new_vi in range(num_vars):
                clamp_rank_vec.push_back(int(clamp_rank[new_to_old[new_vi]]))
        else:
            for rank in clamp_rank:
                clamp_rank_vec.push_back(int(rank))

    cdef int cpp_seed
    cdef double min_complexity_value = float(min_complexity)
    cdef double timeout_seconds = float(timeout)
    cdef cppbool timed_out = False
    cdef string timeout_message
    if not np.isfinite(min_complexity_value):
        raise ValueError("min_complexity must be finite")
    if min_complexity_value > max_complexity:
        raise ValueError("min_complexity must be less than or equal to max_complexity")
    if timeout_seconds <= 0.0 or not np.isfinite(timeout_seconds):
        raise ValueError("timeout must be a finite number greater than 0")

    if seed is None:
        cpp_seed = int(np.random.randint(np.iinfo(np.intc).max, dtype=np.intc))
    else:
        cpp_seed = int(seed)

    cdef vector[int] order = greedyVarOrderAdjacency(
        cpp_adj,
        min_complexity_value,
        float(max_complexity),
        clamp_rank_vec,
        heuristic_value,
        cpp_seed,
        float(selection_scale),
        timeout_seconds,
        &timed_out,
        &timeout_message,
    )

    if timed_out:
        warnings.warn(timeout_message.decode('utf-8'), RuntimeWarning, stacklevel=2)

    cdef list variable_order
    if shuffle_for_exact:
        variable_order = [int_to_var[new_to_old[i]] for i in order]
    else:
        variable_order = [int_to_var[i] for i in order]

    return elimination_order_width(bqm, variable_order), variable_order
