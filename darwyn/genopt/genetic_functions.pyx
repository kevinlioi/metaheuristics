cimport cython
import numpy as np
cimport numpy as np
ctypedef np.int32_t int_t
ctypedef np.float32_t double_t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef ordered_crossover(np.ndarray[int_t, ndim=1] solution1,
                        np.ndarray[int_t, ndim=1] solution2,
                        Py_ssize_t index0,
                        Py_ssize_t index1):
    cdef Py_ssize_t i, j, k
    cdef int_t num1, num2
    cdef set subsequence_set = set(solution1[index0:index1])
    cdef np.ndarray[int_t, ndim=1] child = solution1.copy()
    cdef np.ndarray[int_t, ndim=1] subsequence = solution1[index0:index1]

    i = 0 
    for j in range(solution1.shape[0]):
        num1 = solution2[j]
        if (i < index0) or (i >= index1):
            if num1 in subsequence_set:
                continue
            else:
                child[i] = num1
                i += 1
        else:
            for k in range(subsequence.shape[0]):
                num2 = subsequence[k]
                child[i] = num2
                i += 1
            if num1 not in subsequence_set:
                child[i] = num1
                i += 1
    return child


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef heuristic_crossover(np.ndarray[double_t, ndim=1] solution1,
                          np.ndarray[double_t, ndim=1] solution2,
                          double_t fitness1,
                          double_t fitness2,
                          np.ndarray[double_t, ndim=1] lb,
                          np.ndarray[double_t, ndim=1] ub,
                          np.ndarray[double_t, ndim=1] R):
    cdef np.ndarray[double_t, ndim=1] child, better_solution, worse_solution    
    cdef int_t variable_len, w
    cdef double_t r, child_value, better_value, worse_value
    cdef Py_ssize_t var_index

    child = solution1.copy()
    variable_len = lb.shape[0]
    if fitness1 < fitness2:
        better_solution = solution1
        worse_solution = solution2
    else:
        better_solution = solution2
        worse_solution = solution1

    for var_index in range(variable_len):
        better_value = better_solution[var_index]
        worse_value = worse_solution[var_index]
        r = R[var_index]
        w = 0
        child_value = r*(better_value - worse_value) + better_value
        while (lb[var_index] > child_value) or (child_value > ub[var_index]):
            if w == 10:
                child_value = better_value
                break
            else:
                r = r / 2
                child_value = r*(better_value - worse_value) + better_value
                w += 1
        child[var_index] = child_value
    return child


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mutate_random(np.ndarray[double_t, ndim=2] solutions,
                    np.ndarray[double_t, ndim=1] lb,
                    np.ndarray[double_t, ndim=1] ub,
                    np.ndarray[int_t, ndim=2] needs_mutation)
                    double_t mutation_radius:
    cdef np.ndarray[int_t, ndim=1] uniques = np.unique(needs_mutation[:, 1])
    cdef int_t i, var_index
    cdef np.ndarray[int_t, ndim=1] rows

    for i in range(len(uniques)):
        var_index = uniques[i]
        rows = needs_mutation[needs_mutation[:, 1] == var_index][:, 0]
        if mutation_radius == 0:
            solutions[rows, var_index] = np.random.uniform(low=lb[var_index],
                                                           high=ub[var_index],
                                                           size=rows.shape[0])
        else:
            for row in rows:
                solutions[row, var_index] = np.random.uniform(low=max(lb[var_index], solutions[row, var_index] - mutation_radius),
                                                              high=min(ub[var_index], solutions[row, var_index] + mutation_radius))
    return solutions


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef swap_mutate(np.ndarray[int_t, ndim=2] solutions,
                  np.ndarray[int_t, ndim=2] needs_mutation,
                  np.ndarray[int_t, ndim=1] random_var_indices):
    cdef Py_ssize_t i, j, k
    cdef int_t var_index, other_value, this_value, random_var_index, row
    cdef np.ndarray[int_t, ndim=1] rows
    cdef np.ndarray[int_t, ndim=1] uniques = np.unique(needs_mutation[:, 1])

    i = 0
    for k in range(uniques.shape[0]):
        var_index = uniques[k]
        rows = needs_mutation[needs_mutation[:, 1] == var_index][:, 0]
        for j in range(rows.shape[0]):
            row = rows[j]
            random_var_index = random_var_indices[i]
            i += 1
            if random_var_index == var_index:
                continue
            other_value = solutions[row, random_var_index]
            this_value = solutions[row, var_index]

            solutions[row, var_index] = other_value
            solutions[row, random_var_index] = this_value
    return solutions


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef procreate_sequence(np.ndarray[int_t, ndim=4] parents,
                         np.ndarray[int_t, ndim=3] island_solutions,
                         np.ndarray[int_t, ndim=2] ordered_crossover_indices,
                         int_t num_islands,
                         int_t death_count,
                         int_t variables_len):
    cdef np.ndarray[int_t, ndim=2] children = np.zeros((num_islands*death_count, variables_len)).astype(np.int32)
    cdef np.ndarray[int_t, ndim=1] parent1, parent2, child
    cdef int_t index0, index1, i, island_index, child_index, parent_island_index1, parent_index1, parent_island_index2, parent_index2

    i = 0
    for island_index in range(num_islands):
        for child_index in range(parents[island_index].shape[0]):
            (parent_island_index1, parent_index1), (parent_island_index2, parent_index2) = parents[island_index][child_index]
            parent1 = island_solutions[parent_island_index1][parent_index1]
            parent2 = island_solutions[parent_island_index2][parent_index2]
            index0, index1 = ordered_crossover_indices[i]
            child = ordered_crossover(parent1, parent2, index0, index1)
            children[i] = child
            i += 1
    return children


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef procreate_real_valued(np.ndarray[int_t, ndim=4] parents,
                            np.ndarray[double_t, ndim=3] island_solutions,
                            np.ndarray[double_t, ndim=2] island_fitnesses,
                            np.ndarray[double_t, ndim=2] heuristic_crossover_randoms,
                            np.ndarray[double_t, ndim=1] lb,
                            np.ndarray[double_t, ndim=1] ub,
                            int_t num_islands,
                            int_t death_count,
                            int_t variables_len):
    cdef np.ndarray[double_t, ndim=2] children = np.zeros((num_islands*death_count, variables_len)).astype(np.float32)
    cdef np.ndarray[double_t, ndim=1] parent1, parent2, child
    cdef int_t fitness1, fitness2, i, island_index, child_index, parent_island_index1, parent_index1, parent_island_index2, parent_index2

    i = 0
    for island_index in range(num_islands):
        for child_index in range(parents[island_index].shape[0]):
            (parent_island_index1, parent_index1), (parent_island_index2, parent_index2) = parents[island_index][child_index]
            parent1 = island_solutions[parent_island_index1][parent_index1]
            parent2 = island_solutions[parent_island_index2][parent_index2]

            fitness1 = island_fitnesses[parent_island_index1][parent_index1]
            fitness2 = island_fitnesses[parent_island_index2][parent_index2]

            child = heuristic_crossover(parent1, parent2, fitness1, fitness2, lb, ub, heuristic_crossover_randoms[i])
            children[i] = child
            i += 1
    return children


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef random_selection(int_t num_solutions,
                       int_t death_count,
                       int_t num_islands,
                       dict other_island_indices,
                       np.ndarray[int_t, ndim=2] parent_indices):
    cdef np.ndarray[int_t, ndim=1] indices = np.arange(num_solutions-death_count).astype(np.int32)
    cdef np.ndarray[int_t, ndim=4] parents = np.zeros((num_islands, death_count, 2, 2)).astype(np.int32)
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t child_index, island_index

    for island_index in range(num_islands):
        for child_index in range(death_count):
            # This sets the island indices for the parents. One from home, one from afar.
            parents[island_index][child_index][0][0] = island_index
            parents[island_index][child_index][1][0] = next(other_island_indices[island_index])
            
            # This sets the index of the parent within the islands chosen above
            # i keeps track of how many parents we've ticked so far
            parents[island_index][child_index][0][1] = parent_indices[i][0]
            parents[island_index][child_index][1][1] = parent_indices[i][1]
            i += 1
    return parents



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef random_selection_v2(int_t num_solutions,
                          int_t death_count,
                          int_t num_islands,
                          dict other_island_indices,
                          np.ndarray[int_t, ndim=2] parent_indices,
                          dict topology_network):
    cdef np.ndarray[int_t, ndim=1] indices = np.arange(num_solutions-death_count).astype(np.int32)
    cdef np.ndarray[int_t, ndim=4] parents = np.zeros((num_islands, death_count, 2, 2)).astype(np.int32)
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t child_index, island_index

    for island_index in range(num_islands):
        for child_index in range(death_count):
            # This sets the island indices for the parents. One from home, one from afar.
            parents[island_index][child_index][0][0] = island_index
            parents[island_index][child_index][1][0] = topology_network[island_index][next(other_island_indices[island_index])]
            
            # This sets the index of the parent within the islands chosen above
            # i keeps track of how many parents we've ticked so far
            parents[island_index][child_index][0][1] = parent_indices[i][0]
            parents[island_index][child_index][1][1] = parent_indices[i][1]
            i += 1
    return parents
