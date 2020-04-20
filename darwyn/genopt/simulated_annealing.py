import numpy as np
import math
from numba import jit
from scipy.optimize import root
import bisect


@jit
def domination_test(x, score):
    """Returns the part of Array that is dominating score and the part that is nondominated with score.
       Also returns the indices that are non dominated with score"""

    first_better = x[:, 0] > score[0]
    second_better = x[:, 1] > score[1]
    dominating_count = (first_better & second_better).sum()
    non_dominated_with_score = first_better != second_better
    return dominating_count, x[non_dominated_with_score], non_dominated_with_score


class SimulatedAnnealing:
    def __init__(self,
                 objectives,
                 max_epochs=150,
                 M=1000,
                 a=.925,
                 diagnostic_sample_size=1000,
                 num_solutions_to_return=np.inf,
                 save_attainment_snapshots=False):
        """
            max_iterations: This shouldn't be changed (optimized default). It is the max_iterations of the simulated annealing procedure to do.
            M: It is the number of neighbors to find during one iteration of the simulated annealing procedure.
            a: It is the cooling schedule, set between 0 and 1. The higher it is, the slower the cooling.
        """

        self.objectives = objectives
        self.max_epochs = max_epochs
        self.M = M
        self.a = a
        self.diagnostic_sample_size = diagnostic_sample_size
        self.num_solutions_to_return = num_solutions_to_return
        self.save_attainment_snapshots = save_attainment_snapshots
        self.attainment_snapshots = []

        self.attainment_sampling_uniforms = np.random.uniform(size=10000)
        self.energy_uniforms = (x for x in np.random.uniform(size=10000))

        self.current_solution = self.objectives.create_random_solution()
        self.pareto_fitnesses = np.array([self.objectives.fitnesses(self.current_solution)]).astype(np.float32)
        self.pareto_fitnesses_sorted = [list(self.pareto_fitnesses[:, 0]), list(self.pareto_fitnesses[:, 1])]
        self.current_score = self.pareto_fitnesses[0]
        self.pareto_solutions = [self.current_solution]
        self.attainment_surface = self.attainment_sampling()
        self.as_dominating_current = domination_test(self.attainment_surface, self.current_score)[0]

    def optimize(self):
        """Uses simulated annealing to find a good answer but not the most optimal (since that is impossible)"""

        T_0 = self.diagnostics()

        temp_iteration = 0
        while temp_iteration < self.max_epochs:
            T_k = self.a**temp_iteration*T_0
            for m in range(self.M):
                self.multiobj_iteration(T_k)

            if self.save_attainment_snapshots:
                self.attainment_snapshots.append(self.attainment_surface.copy())
            temp_iteration += 1
        abridged_solutions, abridged_pareto_fitnesses = self.create_final_solutions()
        return abridged_solutions, abridged_pareto_fitnesses

    def diagnostics(self):
        """Does 100k iterations where a neighbor is always accepted. This helps define what T_0 should be,
           and sets up the pareto frontier before the real algorithm begins.

           T_0 should be chosen such that -sigma/ln(b) = T_0, where sigma is the sample variance of the change
           in energy and b is the percentage of answers to be accepted at the beginning of the simulated annealing procedure."""

        energies = []
        for i in range(self.diagnostic_sample_size):
            E = self.multiobj_iteration(0, diagnostics=True)
            energies.append(E)
        
        # Initial temperature is set so that 95% of random moves would be accepted
        get_initial_temp = lambda T_0: np.mean(np.exp(-(np.array([x for x in energies if x > 0]))/T_0)) - .95
        T_0 = root(get_initial_temp, 3, method='lm')['x'][0]  # lm method is more robust with the starting point than the hybr method
        return T_0

    def multiobj_iteration(self, T_k, diagnostics=False):
        new_solution = self.objectives.neighbor(self.current_solution)

        new_score = self.objectives.fitnesses(new_solution)

        as_dominating_new = domination_test(self.attainment_surface, new_score)[0]
        f_x_new = as_dominating_new
        f_x_current = self.as_dominating_current

        # Always need to see the dominance of current and new
        total = len(self.attainment_surface) + 2.0
        if (self.current_score[0] > new_score[0]) & (self.current_score[1] > new_score[1]):
            f_x_new += 1
        elif (new_score[0] > self.current_score[0]) & (new_score[1] > self.current_score[1]):
            f_x_current += 1

        # The more the attainment surface dominates a point, the less likely it is chosen.
        E = (f_x_new - f_x_current) / total
        if diagnostics:
            condition = True
        else:
            try:
                condition = next(self.energy_uniforms) < np.exp(-E/T_k)
            except StopIteration:
                self.energy_uniforms = (x for x in np.random.uniform(size=10000))
                condition = next(self.energy_uniforms) < np.exp(-E/T_k)

        if condition:
            # We are switching from current to next, so check if current needs to be added to archive
            # We need to add current if 1) current is at least non-dominated or 2) current dominates all of archive
            archive_dominating_current, non_dominated_archive, non_dominated_bool = domination_test(self.pareto_fitnesses, self.current_score)
            L1 = archive_dominating_current
            L2 = len(non_dominated_archive)
            if (L1 == 0) or (L2 == L1 == 0):
                indices_to_remove = np.where(non_dominated_bool == False)[0]
                fitnesses_to_remove_from_sort = []
                for index in sorted(indices_to_remove, reverse=True):
                    fitnesses_to_remove_from_sort.append(self.pareto_fitnesses[index])
                    del self.pareto_solutions[index]
                fitnesses_to_remove_from_sort = np.array(fitnesses_to_remove_from_sort)

                self.pareto_solutions.append(self.current_solution.copy())
                self.pareto_fitnesses = np.concatenate((non_dominated_archive, self.current_score[np.newaxis, :]))

                if len(fitnesses_to_remove_from_sort) > 0:
                    remove_indices0 = np.intersect1d(self.pareto_fitnesses_sorted[0], fitnesses_to_remove_from_sort[:, 0], assume_unique=True, return_indices=True)[1]
                    remove_indices1 = np.intersect1d(self.pareto_fitnesses_sorted[1], fitnesses_to_remove_from_sort[:, 1], assume_unique=True, return_indices=True)[1]
                    for index in sorted(remove_indices0, reverse=True):
                        del self.pareto_fitnesses_sorted[0][index]
                    for index in sorted(remove_indices1, reverse=True):
                        del self.pareto_fitnesses_sorted[1][index]
                index1 = bisect.bisect_left(self.pareto_fitnesses_sorted[0], self.current_score[0])
                index2 = bisect.bisect_left(self.pareto_fitnesses_sorted[1], self.current_score[1])
                self.pareto_fitnesses_sorted[0].insert(index1, self.current_score[0])
                self.pareto_fitnesses_sorted[1].insert(index2, self.current_score[1])

                self.attainment_surface = self.attainment_sampling()
            self.current_score = new_score
            self.current_solution = new_solution
            self.as_dominating_current = domination_test(self.attainment_surface, self.current_score)[0]
        if diagnostics:
            return E

    def attainment_sampling(self, num_pts=200):
        """Creates the Attainment Surface of the current pareto frontier (pareto_fitnesses). It is effectively a
           uniform distribution along the pareto frontier, but each point is dominated by exactly one pareto frontier
           point along one dimension."""

        L1, L2 = self.pareto_fitnesses_sorted
        L1_rev, L2_rev = L1[::-1], L2[::-1]
        L = len(L1)

        v1 = np.empty((num_pts, 2))
        v2 = np.empty((num_pts, 2))

        if len(self.attainment_sampling_uniforms) < 2*num_pts:
            self.attainment_sampling_uniforms = np.random.uniform(size=10000)
        L1_min, L1_max, L2_min, L2_max = L1[0], L1[-1], L2[0], L2[-1]
        v1[:, 0] = (L1_max - L1_min)*self.attainment_sampling_uniforms[:num_pts] + L1_min
        v2[:, 1] = (L2_max - L2_min)*self.attainment_sampling_uniforms[num_pts:2*num_pts] + L2_min
        self.attainment_sampling_uniforms = self.attainment_sampling_uniforms[2*num_pts:]

        Indices1 = -(L - np.searchsorted(L1, v1[:, 0]))
        Indices2 = -(L - np.searchsorted(L2, v2[:, 1]))

        v1[:, 1] = [L2_rev[index] for index in Indices1]
        v2[:, 0] = [L1_rev[index] for index in Indices2]
        attainment_surface = np.vstack((v1[:, :2], v2[:, :2]))
        return attainment_surface

    def create_final_solutions(self):
        """The ArchiveArray is typically too large to have an analyst determine which specific solution to use.
           This sorts the archive array according to the schedule/FTE metric and then selects about 50 solutions
           uniformly."""

        sort_indices = np.argsort(self.pareto_fitnesses[:, 1])
        abridged_pareto_fitnesses = []
        abridged_pareto_solutions = []
        modulus = math.ceil(len(self.pareto_fitnesses)/min(self.num_solutions_to_return, len(self.pareto_fitnesses)))
        for i, index in enumerate(sort_indices):
            if i % modulus == 0:
                abridged_pareto_fitnesses.append(self.pareto_fitnesses[index])
                abridged_pareto_solutions.append(self.pareto_solutions[index])
        abridged_pareto_fitnesses = np.vstack(abridged_pareto_fitnesses)
        return np.array(abridged_pareto_solutions), abridged_pareto_fitnesses


class MultiObjective:
    def __init__(self, lb, ub, objective_function, neighborhood_radius):
        self.lb = lb.astype(np.float32)
        self.ub = ub.astype(np.float32)
        self.objective_function = objective_function
        self.neighborhood_radius = neighborhood_radius

        self.R = self.neighborhood_radius**-1
        self.bound_range = self.ub - self.lb
        self.d = len(lb)
        self.adjustments = self.create_random_numbers()

    def create_random_numbers(self):
        # More efficient to do many random numbers at once rather than one-by-one
        u = np.random.normal(0, 1, (10000, self.d))
        norm = np.sqrt(np.sum(np.square(u), axis=1))
        r = np.random.uniform(size=10000)**(1.0/self.d)
        adjustments = (x for x in (r[:, np.newaxis]*u/norm[:, np.newaxis]) / self.R)
        return adjustments

    def create_random_solution(self):
        solution = np.array([np.random.uniform(low=lb, high=self.ub[var_index])
                             for var_index, lb in enumerate(self.lb)])
        return solution

    def neighbor(self, solution):
        normalized_solution = 2*((solution - self.lb) / self.bound_range) - 1
        while True:
            try:
                adjustment = next(self.adjustments)
            except StopIteration:
                self.adjustments = self.create_random_numbers()
                adjustment = next(self.adjustments)
            new_normalized_solution = normalized_solution + adjustment
            in_bounds = (-1 <= new_normalized_solution).all() & (new_normalized_solution <= 1).all()
            if in_bounds:
                break
        new_solution = self.bound_range*((new_normalized_solution + 1) / 2) + self.lb
        return new_solution

    def fitnesses(self, solution):
        return self.objective_function(solution)
