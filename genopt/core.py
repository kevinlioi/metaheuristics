import numpy as np
from collections import defaultdict
import genetic_functions
np.seterr(all='ignore')


#################################################
#################################################
# Island Genetic Algorithm


def networkx_topology(G):
    """Common Graphs
       nx.cycle_graph(n) is a ring topology
       nx.grid_2d_graph(n,n) nxn 2d lattice
       nx.hexagonal_lattice_graph(n,m) triangular lattices so that each "surrounded" node has 6 edges
       nx.connected_watts_strogatz_graph(n = 256, k = 10, p = .1) is supposed to model social networks"""

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


# crossover_heuristic, mutate_random, and create random solutions require integer_solution. Just round stuffs.
class GeneticOptimizer:
    def __init__(self,
                 objective,
                 G,
                 problem_type,
                 variables=None,
                 lb=None,
                 ub=None,
                 num_solutions=8,
                 mutation_rate=.01,
                 max_iterations=3000,
                 num_islands=128,
                 death_rate=.5,
                 min_target=None,
                 args=None,
                 integer_solution=False):

        self.objective = objective
        self.G = G
        self.problem_type = problem_type
        self.lb = lb.astype(np.float32)
        self.ub = ub.astype(np.float32)
        self.variables = variables
        self.num_solutions = num_solutions
        self.mutation_rate = mutation_rate
        self.max_iterations = max_iterations
        self.num_islands = num_islands
        self.death_rate = death_rate
        self.min_target = min_target
        self.args = args
        self.integer_solution = integer_solution
        if self.problem_type == 'real_valued':
            self.num_vars = len(self.lb)
        elif self.problem_type == 'sequence':
            self.num_vars = len(variables)

        self.death_count = int(self.num_solutions*self.death_rate)
        self.topology_network = networkx_topology(self.G)

    def random_number_factory(self):
        other_island_indices = {island_index: (x for x in np.random.choice(self.topology_network[island_index],
                                                                           size=int((1-self.death_rate)*self.num_solutions*1000),
                                                                           replace=True).astype(np.int32))
                                for island_index in range(self.num_islands)}

        # Compute parent indices ahead of time
        indices = [item for item in range(self.num_solutions-self.death_count)]
        parent_indices = np.random.choice(indices, size=(1000, self.death_count*self.num_islands, 2), replace=True).astype(np.int32)

        # Compute mutation ahead of time
        mutations = {}
        random_uniforms = np.random.uniform(size=(1000, self.num_islands*self.death_count, self.num_vars))
        mutations['needs_mutation'] = [np.array(np.where(x < self.mutation_rate)).T.astype(np.int32) for x in random_uniforms]
        mutations['lengths'] = [len(x) for x in mutations['needs_mutation']]
        mutations['random_mutation_var_indices'] = np.random.choice(range(self.num_vars), replace=True, size=sum(mutations['lengths'])).astype(np.int32)
        mutations['current_mutations_count'] = 0

        # Compute ordered_crossover_indices ahead of time
        if self.problem_type == 'real_valued':
            crossover_random_nums = np.random.uniform(low=0, high=1, size=(1000, self.num_islands*self.death_count, self.num_vars)).astype(np.float32)
        elif self.problem_type == 'sequence':
            crossover_random_nums = np.sort(np.random.choice(range(self.num_vars), replace=True, size=(1000, self.num_islands*self.death_count, 2))).astype(np.int32)
        return other_island_indices, parent_indices, crossover_random_nums, mutations

    def GA(self):
        if self.problem_type == 'sequence':
            island_solutions = [np.vstack([np.random.choice(self.variables, replace=False, size=len(self.variables))
                                           for _ in range(self.num_solutions)]).astype(np.int32)
                                for _ in range(self.num_islands)]
        if self.problem_type == 'real_valued':
            island_solutions = np.array([[[np.random.uniform(low=self.lb[var_index], high=self.ub[var_index])
                                           for var_index in range(self.num_vars)]
                                          for _ in range(self.num_solutions)]
                                         for __ in range(self.num_islands)]).astype(np.float32)

        if self.args is None:
            island_fitnesses = [[self.objective(solution) for solution in solutions] for solutions in island_solutions]
            island_fitnesses = np.array(island_fitnesses).reshape((self.num_islands, self.num_solutions)).astype(np.float32)
        else:
            island_fitnesses = [[self.objective(solution, **self.args) for solution in solutions] for solutions in island_solutions]
            island_fitnesses = np.array(island_fitnesses).reshape((self.num_islands, self.num_solutions)).astype(np.float32)

        self.metrics = []
        iterations = 0
        while iterations < self.max_iterations:
            if iterations % 1000 == 0:
                # Should consider making a "random number factory" function. Then alter on it could be adopted to take a stream or something clever.
                other_island_indices, parent_indices, crossover_random_nums, mutations = self.random_number_factory()
                if iterations > 0:
                    print(self.metrics[-1])

            # Order solutions by fitness
            argsorts = [np.argsort(island_fitnesses[island_index]) for island_index in range(self.num_islands)]
            island_solutions = [island_solutions[island_index][argsorts[island_index]] for island_index in range(self.num_islands)]
            island_fitnesses = [island_fitnesses[island_index][argsorts[island_index]] for island_index in range(self.num_islands)]

            # Natural selection: kill off worst self.death_rate of population
            island_solutions = [island_solutions[island_index][:-self.death_count] for island_index in range(self.num_islands)]
            island_fitnesses = [island_fitnesses[island_index][:-self.death_count] for island_index in range(self.num_islands)]

            # Choose parents
            parents = genetic_functions.random_selection(self.num_solutions, self.death_count, self.num_islands, other_island_indices, parent_indices[iterations % 1000])

            # Procreation and mutation
            L = mutations['lengths'][iterations % 1000]
            C = mutations['current_mutations_count']
            if self.problem_type == 'sequence':
                children = genetic_functions.procreate_sequence(parents=parents,
                                                                island_solutions=np.array(island_solutions).astype(np.int32),
                                                                ordered_crossover_indices=crossover_random_nums[iterations % 1000],
                                                                num_islands=self.num_islands,
                                                                death_count=self.death_count,
                                                                variables_len=len(self.variables))
                mutated_children = genetic_functions.swap_mutate(solutions=children,
                                                                 needs_mutation=mutations['needs_mutation'][iterations % 1000],
                                                                 random_var_indices=mutations['random_mutation_var_indices'][C:C+L])
            elif self.problem_type == 'real_valued':
                children = genetic_functions.procreate_real_valued(parents=parents,
                                                                   island_solutions=np.array(island_solutions).astype(np.float32),
                                                                   island_fitnesses=np.array(island_fitnesses).astype(np.float32),
                                                                   heuristic_crossover_randoms=crossover_random_nums[iterations % 1000],
                                                                   lb=self.lb,
                                                                   ub=self.ub,
                                                                   num_islands=self.num_islands,
                                                                   death_count=self.death_count,
                                                                   variables_len=self.num_vars)
                mutated_children = genetic_functions.mutate_random(solutions=children,
                                                                   lb=self.lb,
                                                                   ub=self.ub,
                                                                   needs_mutation=mutations['needs_mutation'][iterations % 1000])

            next_gen_solutions = mutated_children.reshape(self.num_islands, self.death_count, self.num_vars)
            mutations['current_mutations_count'] += L

            # Compute next gen fitness
            solutions_to_compute_fitness_for = []
            indices_for_computed_solutions = []
            [(solutions_to_compute_fitness_for.append(next_gen_solutions[island_index]),
              indices_for_computed_solutions.extend([island_index]*self.death_count))
             for island_index in range(self.num_islands)]
            if self.args is None:
                computed_fitnesses = self.objective(np.concatenate(solutions_to_compute_fitness_for))
            else:
                solutions_to_compute_fitness_for = np.concatenate(solutions_to_compute_fitness_for).astype(np.int32)
                computed_fitnesses = [self.objective(solution, **self.args)
                                      for solution in solutions_to_compute_fitness_for]

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
            self.metrics.append((np.min(island_fitnesses), np.percentile(island_fitnesses, 25), np.percentile(island_fitnesses, 50)))
            iterations += 1

        all_fitnesses = np.vstack(island_fitnesses)
        ind = np.unravel_index(np.argmin(all_fitnesses, axis=None), all_fitnesses.shape)
        best_solution = island_solutions[ind[0]][ind[1]]
        best_fitness = all_fitnesses[ind]
        return (best_solution, best_fitness), self.metrics

