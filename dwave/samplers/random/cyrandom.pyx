# Copyright 2022 D-Wave Systems Inc.
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

cimport cython

from libc.stdint cimport int8_t, uint8_t
from libcpp.algorithm cimport make_heap, push_heap, pop_heap
from libcpp.utility cimport move, pair
from libcpp.vector cimport vector

import dimod
cimport dimod
import numpy as np

# chrono is not included in Cython's libcpp. So we do it more manually
cdef extern from *:
    """
    #include <chrono>

    double realtime_clock() {
        auto t = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(t.time_since_epoch()).count();
    }
    """
    double realtime_clock()


# it would be nicer to use a struct, but OSX throws segfaults when sorting
# Cython-created structs. We should retest that when we switch to Cython 3.
ctypedef pair[double, vector[int8_t]] state_t


@cython.boundscheck(False)
@cython.wraparound(False)
def sample(
    dimod.cyBQM_float64 cybqm,
    Py_ssize_t num_reads,
    double time_limit,
    Py_ssize_t max_num_samples,
    Py_ssize_t batch_size,
    object seed,
):
    if cybqm.vartype() is not dimod.BINARY:
        raise ValueError("cybqm must be binary")
    cdef Py_ssize_t num_variables = cybqm.num_variables()

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if num_reads <= 0:
        raise ValueError("num_reads must be positive")
    if time_limit <= 0:
        raise ValueError("time_limit must be positive")
    if max_num_samples <= 0:
        raise ValueError("max_num_samples must be positive")

    # Get our rng
    rng = np.random.default_rng(seed)

    # Ok, time to start sampling so let's start the timer
    cdef double sampling_start_time = realtime_clock()
    cdef double sampling_stop_time = sampling_start_time + time_limit

    # Get a vector that will hold our samples. We'll be keeping them in a heap
    # so that we can keep the best as we go
    cdef vector[state_t] samples_heap

    # Track the total number of samples we drew
    cdef Py_ssize_t num_drawn = 0

    # Some cdefs we'll need inside the loop
    cdef const uint8_t[:, ::1] batch
    cdef vector[int8_t] sample

    # Until we run out of samples or time, draw random samples
    cdef bint stop = False
    while not stop:
        # Use NumPy to generate a batch of random samples. We do this in batches
        # to amortize the cost of calling a Python function. We could avoid the
        # Python overhead with a cimport but this way we avoid a compile-time
        # relationship with NumPy.
        batch = rng.integers(2, size=(batch_size, num_variables), dtype=np.uint8)

        for bi in range(batch_size):
            # If we've run out of time, don't parse the next sample
            if realtime_clock() >= sampling_stop_time:
                stop = True
                break

            # Copy the sample from an array into a vector
            sample.reserve(num_variables)
            for vi in range(num_variables):
                sample.emplace_back(batch[bi, vi])

            # Put that vector (with its energy) on the heap
            samples_heap.emplace_back(cybqm.data().energy(sample.begin()), move(sample))
            push_heap(samples_heap.begin(), samples_heap.end())

            # Increment the total number we've drawn
            num_drawn += 1

            # If we've drawn enough, exit early
            if num_drawn >= num_reads:
                stop = True
                break

            # Finally make sure out heap isn't getting larger than the max_num_samples
            if samples_heap.size() > <size_t>max_num_samples:
                pop_heap(samples_heap.begin(), samples_heap.end())
                samples_heap.pop_back()

    # sampling done!
    # time to construct the return objects
    cdef int8_t[:, ::1] samples = np.empty((samples_heap.size(), num_variables), dtype=np.int8)
    cdef double[::1] energies = np.empty(samples_heap.size(), dtype=np.double)
    for i in range(samples_heap.size()):
        for vi in range(num_variables):
            samples[i, vi] = samples_heap[i].second[vi]
        energies[i] = samples_heap[i].first

    return np.asarray(samples), np.asarray(energies), dict(num_reads=num_drawn)
