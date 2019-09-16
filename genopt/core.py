import numpy as np
from collections import defaultdict
np.seterr(all='ignore')


#################################################
#################################################
# Island Genetic Algorithm

# Common Graphs
# nx.cycle_graph(n) is a ring topology
# nx.grid_2d_graph(n,n) nxn 2d lattice
# nx.hexagonal_lattice_graph(n,m) triangular lattices so that each "surrounded" node has 6 edges
# nx.connected_watts_strogatz_graph(n = 256, k = 10, p = .1) is supposed to model social networks


def networkx_topology(G):
    node_to_index = {}
    for i, node in enumerate(G.nodes):
        node_to_index[node] = i

    topology_network = defaultdict(lambda: [])
    for edge in G.edges:
        index0 = node_to_index[edge[0]]
        index1 = node_to_index[edge[1]]
        topology_network[index0].append(index1)
        topology_network[index1].append(index0)
    topology_network = [(index, connections) for index, connections in topology_network.items()]
    topology_network = sorted(topology_network, key = lambda x: x[0])
    topology_network = [x[1] for x in topology_network]
    return topology_network


class GeneticOptimizer:
    def __init__(self,
                 objective,
                 lb,
                 ub,
                 G,
                 num_solutions=128,
                 mutation_rate=.01,
                 max_iterations=3000,
                 num_islands=128,
                 death_rate=.5,
                 min_target=None,
                 args=None,
                 integer_solution=False):

        self.objective = objective
        self.lb = lb
        self.ub = ub
        self.G = G
        self.num_solutions = num_solutions
        self.mutation_rate = mutation_rate
        self.max_iterations = max_iterations
        self.num_islands = num_islands
        self.death_rate = death_rate
        self.min_target = min_target
        self.args = args
        self.integer_solution = integer_solution

        self.topology_network = networkx_topology(self.G)
        self.death_count = int(self.num_solutions*self.death_rate)
        
        total_death_count = int(self.death_count*self.num_islands)
        self.i1 = np.array([np.full((2, len(self.lb)), i) for i in range(total_death_count)])
        self.i3 = np.array([np.array([np.arange(len(self.lb)), np.arange(len(self.lb))])]*total_death_count)

        # Check for bad inputs
        if len(self.lb) != len(self.ub):
            raise Exception('lengths of ub and lb do not match')

        if not (0 < self.mutation_rate < 1):
            raise Exception('mutation_rate should be between 0 and 1')

        if type(self.lb) != np.ndarray:
            raise Exception('lb should be np.ndarray')

        if type(self.ub) != np.ndarray:
            raise Exception('ub should be np.ndarray')

    def GA(self):
        island_solutions = [self.create_random_solutions() for _ in range(self.num_islands)]
        if self.args is None:
            island_fitnesses = [self.objective(solutions) for solutions in island_solutions]
        else:
            island_fitnesses = [self.objective(solutions, **self.args) for solutions in island_solutions]

        self.metrics = []
        iterations = 0
        while iterations < self.max_iterations:
            if (iterations > 0) & (iterations % 5000 == 0):
                print(self.metrics[-1])

            if iterations % 1000 == 0:
                self.other_island_indices = {island_index: (x for x in np.random.choice(self.topology_network[island_index],
                                                                                        size=int( (1-self.death_rate)*self.num_solutions*1000 ),
                                                                                        replace=True))
                                             for island_index in range(self.num_islands)}

            # Order solutions by fitness
            argsorts = [np.argsort(island_fitnesses[island_index]) for island_index in range(self.num_islands)]
            island_solutions = [island_solutions[island_index][argsorts[island_index]] for island_index in range(self.num_islands)]
            island_fitnesses = [island_fitnesses[island_index][argsorts[island_index]] for island_index in range(self.num_islands)]

            # Natural selection: kill off worst self.death_rate of population
            island_solutions = [island_solutions[island_index][:-self.death_count] for island_index in range(self.num_islands)]
            island_fitnesses = [island_fitnesses[island_index][:-self.death_count] for island_index in range(self.num_islands)]

            # Choose parents
            parents = self.random_selection()

            # Procreation and mutation
            next_gen_solutions = self.procreate_and_mutate(parents, island_solutions, island_fitnesses)

            # Compute next gen fitness
            solutions_to_compute_fitness_for = []
            indices_for_computed_solutions = []
            [(solutions_to_compute_fitness_for.append(next_gen_solutions[island_index]),
              indices_for_computed_solutions.extend([island_index]*self.death_count))
             for island_index in range(self.num_islands)]
            if self.args is None:
                computed_fitnesses = self.objective(np.concatenate(solutions_to_compute_fitness_for))
            else:
                computed_fitnesses = self.objective(np.concatenate(solutions_to_compute_fitness_for), **self.args)

            # Apply next gen fitness
            next_gen_fitnesses = defaultdict(lambda: [])
            [next_gen_fitnesses[island_index].append(computed_fitnesses[index])
             for index, island_index in enumerate(indices_for_computed_solutions)]
            next_gen_fitnesses = [np.array(next_gen_fitnesses[island_index]) for island_index in range(self.num_islands)]

            # Concatenate survivors with next generation
            island_solutions = [np.concatenate((island_solutions[island_index], next_gen_solutions[island_index]))
                                for island_index in range(self.num_islands)]
            island_fitnesses = [np.concatenate((island_fitnesses[island_index], next_gen_fitnesses[island_index]))
                                for island_index in range(self.num_islands)]

            # Record results of generation
            self.metrics.append((np.min(island_fitnesses), np.percentile(island_fitnesses, 25), np.mean(island_fitnesses)))
            iterations += 1
            
            if self.min_target is not None:
                if self.metrics[-1] <= self.min_target:
                    break

        island_bests = [(island_index,
                         np.min(island_fitnesses[island_index]),
                         np.argmin(island_fitnesses[island_index]))
                         for island_index in range(self.num_islands)]
        best_island_index, best_fitness, best_solution_index = min(island_bests, key = lambda x: x[1])
        best_solution = island_solutions[best_island_index][best_solution_index]
        return (best_solution, best_fitness), self.metrics

    def random_selection(self):
        indices = [item for item in range(self.num_solutions-self.death_count)]
        parent_indices = [x for x in np.random.choice(indices, size = (self.death_count*self.num_islands, 2), replace = True)]
        
        parents = [[((island_index,
                      parent_indices[island_index*self.death_count+child_index][0]),
                    (next(self.other_island_indices[island_index]),
                      parent_indices[island_index*self.death_count+child_index][1]))
                     for child_index in range(self.death_count)]
                     for island_index in range(self.num_islands)]
        return parents

    def procreate_and_mutate(self, parents, island_solutions, island_fitnesses):
        all_sols = [[island_solutions[parent_island_index1][parent_index1],
                     island_solutions[parent_island_index2][parent_index2]]
                     for island_index in range(self.num_islands)
                     for child_index, ((parent_island_index1, parent_index1), (parent_island_index2, parent_index2)) in enumerate(parents[island_index])]

        all_fits = [[island_fitnesses[parent_island_index1][parent_index1],
                     island_fitnesses[parent_island_index2][parent_index2]]
                     for island_index in range(self.num_islands)
                     for child_index, ((parent_island_index1, parent_index1), (parent_island_index2, parent_index2)) in enumerate(parents[island_index])]

        all_sols = np.array(all_sols)
        argsort = np.argsort(all_fits)
        i2 = [[[asort[0]]*len(self.lb), [asort[1]]*len(self.lb)] for asort in argsort]
        all_sols = all_sols[self.i1, i2, self.i3]

        children = self.crossover_heuristic(all_sols)
        mutated_children = self.mutate_random(children, self.mutation_rate)
        next_gen_solutions = mutated_children.reshape(self.num_islands, self.death_count, len(self.lb))
        return next_gen_solutions

    def mutate_random(self, solutions, mutation_rate):
        random_uniforms = np.random.uniform(size=solutions.shape)
        needs_mutation = np.where(random_uniforms < mutation_rate)
        needs_mutation = np.array(needs_mutation).T
        for var_index in np.unique(needs_mutation[:, 1]):
            rows = needs_mutation[needs_mutation[:, 1] == var_index][:, 0]
            if self.integer_solution:
                solutions[rows, var_index] = np.round(np.random.uniform(low=self.lb[var_index], high=self.ub[var_index], size=len(rows)))
            else:
                solutions[rows, var_index] = np.random.uniform(low=self.lb[var_index], high=self.ub[var_index], size=len(rows))
        return solutions

    def crossover_heuristic(self, all_sols):
        w = 0
        R = np.random.uniform(size = (all_sols.shape[0], all_sols.shape[2]))
        stop_point = len(self.lb)*all_sols.shape[0]
        condition = True
        while condition:
            if w == 10:
                # Randomly reduce RemainingSols to one row by taking a random element from each column.
                first_dim_falses, second_dim_falses = np.where(valid_points == False)
                OG_points = all_sols[first_dim_falses, :, second_dim_falses]
                random_selection = np.random.permutation(OG_points.T).T[:, 0]
                children[not_valid_points] = random_selection
                break

            # It's possible R is such that the new point is out of bounds. We then halve R iteratively until
            # we are in bounds or reach 10 iterations, in which case we just randomly combine solutions.
            better = all_sols[:, 0]
            worse = all_sols[:, 1]
            children = R*(better - worse) + better

            valid_points = np.logical_and(self.lb <= children, children <= self.ub)
            not_valid_points = ~valid_points
            R[not_valid_points] = R[not_valid_points]/2.0
            w += 1
            condition = np.sum(np.sum(valid_points)) != stop_point

        if self.integer_solution:
            return np.round(children)
        else:
            return children

    def create_random_solutions(self):
        """Generates self.num_solutions random solutions."""

        random_solutions = []
        for index in range(len(self.lb)):
            solution = np.random.uniform(low=self.lb[index], high=self.ub[index], size=(self.num_solutions, 1))
            random_solutions.append(solution)

        if self.integer_solution:
            return np.round(np.hstack(random_solutions))
        else:
            return np.hstack(random_solutions)


class GeneticOptimizer_v2:
    def __init__(self,
                 objective,
                 lb,
                 ub,
                 G,
                 num_solutions=128,
                 mutation_rate=.01,
                 max_iterations=3000,
                 num_islands=128,
                 death_rate=.5,
                 min_target=None,
                 args=None,
                 integer_solution=False,
                 variable_multiplier=1):

        self.objective = objective
        self.lb = lb
        self.ub = ub
        self.G = G
        self.num_solutions = num_solutions
        self.mutation_rate = mutation_rate
        self.max_iterations = max_iterations
        self.num_islands = num_islands
        self.death_rate = death_rate
        self.min_target = min_target
        self.args = args
        self.integer_solution = integer_solution
        self.variable_multiplier = variable_multiplier

        # We will have repeats of the variables
        self.lb = np.tile(self.lb, self.variable_multiplier)
        self.ub = np.tile(self.ub, self.variable_multiplier)

        self.topology_network = networkx_topology(self.G)
        self.death_count = int(self.num_solutions*self.death_rate)

        total_death_count = int(self.death_count*self.num_islands)
        self.i1 = np.array([np.full((2, len(self.lb)), i) for i in range(total_death_count)])
        self.i3 = np.array([np.array([np.arange(len(self.lb)), np.arange(len(self.lb))])]*total_death_count)

        # Check for bad inputs
        if len(self.lb) != len(self.ub):
            raise Exception('lengths of ub and lb do not match')

        if not (0 < self.mutation_rate < 1):
            raise Exception('mutation_rate should be between 0 and 1')

        if type(self.lb) != np.ndarray:
            raise Exception('lb should be np.ndarray')

        if type(self.ub) != np.ndarray:
            raise Exception('ub should be np.ndarray')

    def create_random_phenotype_solution_indices(self, count, phenotype_length, full_island):
        if full_island:
            # at the start the whole island needs it regardless of death
            solutions_needed = int(count*self.num_solutions)
        else:
            # percentage of the island needs it cause of death
            solutions_needed = int(count*self.num_solutions*self.death_rate)

        variable_random_indices = {var_index: (item for item in np.random.choice(a=[var_index+M*phenotype_length
                                                                                    for M in range(self.variable_multiplier)],
                                                                                 replace=True,
                                                                                 size=solutions_needed))
                                   for var_index in range(phenotype_length)}

        def boolean_generator(arr):
            empty = np.zeros(len(self.lb), dtype=bool)
            empty[arr] = True
            return empty

        solution_random_indices = (boolean_generator(np.array([next(variable_random_indices[var_index])
                                                               for var_index in range(phenotype_length)]))
                                   for _ in range(solutions_needed))

        if full_island:
            phenotype_matrices = (np.vstack([next(solution_random_indices)
                                             for _ in range(int(solutions_needed/count))])
                                  for _ in range(count))
            return phenotype_matrices
        return solution_random_indices

    def GA(self):
        island_solutions = [self.create_random_solutions() for _ in range(self.num_islands)]
        phenotype_length = int(len(island_solutions[0][0])/self.variable_multiplier)

        phenotype_island_matrices = self.create_random_phenotype_solution_indices(len(island_solutions), phenotype_length, full_island=True)
        island_solutions_phenotypes = [island_solution[next(phenotype_island_matrices)].reshape((island_solution.shape[0],
                                                                                                phenotype_length))
                                       for island_solution in island_solutions]

        if self.args is None:
            island_fitnesses = [self.objective(solutions) for solutions in island_solutions_phenotypes]
        else:
            island_fitnesses = [self.objective(solutions, **self.args) for solutions in island_solutions_phenotypes]

        self.metrics = []
        iterations = 0
        while iterations < self.max_iterations:
            if (iterations > 0) & (iterations % 5000 == 0):
                print(self.metrics[-1])

            if iterations % 1000 == 0:
                self.other_island_indices = {island_index: (x for x in np.random.choice(self.topology_network[island_index],
                                                                                        size=int( (1-self.death_rate)*self.num_solutions*1000 ),
                                                                                        replace=True))
                                             for island_index in range(self.num_islands)}
                phenotype_solution_indices = self.create_random_phenotype_solution_indices(len(island_solutions)*1000, phenotype_length, full_island=False)

            # Order solutions by fitness
            argsorts = [np.argsort(island_fitnesses[island_index]) for island_index in range(self.num_islands)]
            island_solutions = [island_solutions[island_index][argsorts[island_index]] for island_index in range(self.num_islands)]
            island_fitnesses = [island_fitnesses[island_index][argsorts[island_index]] for island_index in range(self.num_islands)]

            # Natural selection: kill off worst self.death_rate of population
            island_solutions = [island_solutions[island_index][:-self.death_count] for island_index in range(self.num_islands)]
            island_fitnesses = [island_fitnesses[island_index][:-self.death_count] for island_index in range(self.num_islands)]

            # Choose parents
            parents = self.random_selection()

            # Procreation and mutation
            next_gen_solutions = self.procreate_and_mutate(parents, island_solutions, island_fitnesses)

            # Compute next gen fitness
            solutions_to_compute_fitness_for = []
            indices_for_computed_solutions = []
            [(solutions_to_compute_fitness_for.append(next_gen_solutions[island_index]),
              indices_for_computed_solutions.extend([island_index]*self.death_count))
             for island_index in range(self.num_islands)]

            solutions_to_compute_fitness_for_phenotypes = [solution[next(phenotype_solution_indices)]
                                                           for solution in np.concatenate(solutions_to_compute_fitness_for)]

            if self.args is None:
                computed_fitnesses = self.objective(solutions_to_compute_fitness_for_phenotypes)
            else:
                computed_fitnesses = self.objective(solutions_to_compute_fitness_for_phenotypes, **self.args)

            # Apply next gen fitness
            next_gen_fitnesses = defaultdict(lambda: [])
            [next_gen_fitnesses[island_index].append(computed_fitnesses[index])
             for index, island_index in enumerate(indices_for_computed_solutions)]
            next_gen_fitnesses = [np.array(next_gen_fitnesses[island_index]) for island_index in range(self.num_islands)]

            # Concatenate survivors with next generation
            island_solutions = [np.concatenate((island_solutions[island_index], next_gen_solutions[island_index]))
                                for island_index in range(self.num_islands)]
            island_fitnesses = [np.concatenate((island_fitnesses[island_index], next_gen_fitnesses[island_index]))
                                for island_index in range(self.num_islands)]

            # Record results of generation
            self.metrics.append((np.min(island_fitnesses), np.percentile(island_fitnesses, 25), np.mean(island_fitnesses)))
            iterations += 1

            if self.min_target is not None:
                if self.metrics[-1][0] <= self.min_target:
                    break

        island_bests = [(island_index,
                         np.min(island_fitnesses[island_index]),
                         np.argmin(island_fitnesses[island_index]))
                        for island_index in range(self.num_islands)]
        best_island_index, best_fitness, best_solution_index = min(island_bests, key = lambda x: x[1])
        best_solution = island_solutions[best_island_index][best_solution_index]
        return (best_solution, best_fitness), self.metrics

    def random_selection(self):
        indices = [item for item in range(self.num_solutions-self.death_count)]
        parent_indices = [x for x in np.random.choice(indices, size = (self.death_count*self.num_islands, 2), replace = True)]

        parents = [[((island_index,
                      parent_indices[island_index*self.death_count+child_index][0]),
                     (next(self.other_island_indices[island_index]),
                      parent_indices[island_index*self.death_count+child_index][1]))
                    for child_index in range(self.death_count)]
                   for island_index in range(self.num_islands)]
        return parents

    def procreate_and_mutate(self, parents, island_solutions, island_fitnesses):
        all_sols = [[island_solutions[parent_island_index1][parent_index1],
                     island_solutions[parent_island_index2][parent_index2]]
                    for island_index in range(self.num_islands)
                    for child_index, ((parent_island_index1, parent_index1), (parent_island_index2, parent_index2)) in enumerate(parents[island_index])]

        all_fits = [[island_fitnesses[parent_island_index1][parent_index1],
                     island_fitnesses[parent_island_index2][parent_index2]]
                    for island_index in range(self.num_islands)
                    for child_index, ((parent_island_index1, parent_index1), (parent_island_index2, parent_index2)) in enumerate(parents[island_index])]

        all_sols = np.array(all_sols)
        argsort = np.argsort(all_fits)
        i2 = [[[asort[0]]*len(self.lb), [asort[1]]*len(self.lb)] for asort in argsort]
        all_sols = all_sols[self.i1, i2, self.i3]

        children = self.crossover_heuristic(all_sols)
        mutated_children = self.mutate_random(children, self.mutation_rate)
        next_gen_solutions = mutated_children.reshape(self.num_islands, self.death_count, len(self.lb))
        return next_gen_solutions

    def mutate_random(self, solutions, mutation_rate):
        random_uniforms = np.random.uniform(size=solutions.shape)
        needs_mutation = np.where(random_uniforms < mutation_rate)
        needs_mutation = np.array(needs_mutation).T
        for var_index in np.unique(needs_mutation[:, 1]):
            rows = needs_mutation[needs_mutation[:, 1] == var_index][:, 0]
            if self.integer_solution:
                solutions[rows, var_index] = np.round(np.random.uniform(low=self.lb[var_index], high=self.ub[var_index], size=len(rows)))
            else:
                solutions[rows, var_index] = np.random.uniform(low=self.lb[var_index], high=self.ub[var_index], size=len(rows))
        return solutions

    def crossover_heuristic(self, all_sols):
        w = 0
        R = np.random.uniform(size = (all_sols.shape[0], all_sols.shape[2]))
        stop_point = len(self.lb)*all_sols.shape[0]
        condition = True
        while condition:
            if w == 10:
                # Randomly reduce RemainingSols to one row by taking a random element from each column.
                first_dim_falses, second_dim_falses = np.where(valid_points == False)
                OG_points = all_sols[first_dim_falses, :, second_dim_falses]
                random_selection = np.random.permutation(OG_points.T).T[:, 0]
                children[not_valid_points] = random_selection
                break

            # It's possible R is such that the new point is out of bounds. We then halve R iteratively until
            # we are in bounds or reach 10 iterations, in which case we just randomly combine solutions.
            better = all_sols[:, 0]
            worse = all_sols[:, 1]
            children = R*(better - worse) + better

            valid_points = np.logical_and(self.lb <= children, children <= self.ub)
            not_valid_points = ~valid_points
            R[not_valid_points] = R[not_valid_points]/2.0
            w += 1
            condition = np.sum(np.sum(valid_points)) != stop_point

        if self.integer_solution:
            return np.round(children)
        else:
            return children

    def create_random_solutions(self):
        """Generates self.num_solutions random solutions."""

        random_solutions = []
        for index in range(len(self.lb)):
            solution = np.random.uniform(low=self.lb[index], high=self.ub[index], size=(self.num_solutions, 1))
            random_solutions.append(solution)

        if self.integer_solution:
            return np.round(np.hstack(random_solutions))
        else:
            return np.hstack(random_solutions)
