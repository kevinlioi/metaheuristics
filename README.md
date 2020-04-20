# metaheuristics

Contains classes for two different metaheuristics; a genetic algorithm and simulated annealing.

The genetic algorithm can solve either real-valued optimization problems or sequence optimization problems. The novel aspect of this algorithm is abandoning island migration for island procreation, e.g. individuals of one island can only procreate with individuals of other islands. These islands are connected to one another through a networkx graph. An important result is that for most problems, a lattice is often best but for extremely difficult problems a social network (such as Watts-Strogatz) often performs better.

The simulated annealing algorithm solves multiobjective optimization problems, and has methods specific to real-valued problems. The method is based on the paper found in the references section below.

## GeneticOptimizer

### Dependencies
* numpy
* collections
* cython

### Arguments
*The GeneticOptimizer class requires several inputs:*

Argument                  | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Object&nbsp;Type&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;               | Default           | Description
------------------------- | :-------------------------: | :----------------:  | ----------------
objective                 | `python function`                     | *required*        | The objective function to be optimized.
G                         | `network X graph`         | *required*        | A graph that defines which islands can procreate with eachother. Nodes should be equal to num_islands argument.
problem_type              | `string`                  | *required*        | Either 'real_valued' or 'sequence'
variables                 | `list`                    | None              | If sequence problem_type, these are the objects to be sequenced optimally
lb                        | `np array of numbers`     | None              | Lower bounds for each variable
ub                        | `np array of numbers`     | None              | Upper bounds for each variable
num_solutions             | `integer`                 | 8                 | Number of solutions per island in the population
mutation_rate             | `float`                   | `.01`             | The rate at which genes are mutated.
max_iterations            | `integer`                 | 3000              | The total number of iterations (natural selection/procreation/mutation) to perform before termination
num_islands               | `integer`                 | 128               | The number of islands each with num_solutions in the population. There is no intra-island procreation to increase diversity.
death_rate                | `float`                   | .5                | The percentage of the population killed off in each generation
min_target                | `float`                   | None              | If not None, if the algorithm achieves a score below min_target, algorithm terminates early
mutation_radius           | `float`                   | 0                 | Only relevant if problem_type is real_valued. If 0, mutation is uniform random between lb and ub. If >0, defines a radius for the uniform random mutation from where the gene currently is.
args                      | `dictionary`              | None              | Additional, static arguments for the objective if applicable
integer_solution          | `boolean`                 | False             | If problem_type='real_valued', this forces each gene in a solution to be integer valued after procreation.
log_metrics               | `boolean`                 | True              | Records the best, 25th and 50th percentiles of the population at each generation


### Outputs
The outputs from `GeneticOptimizer.GA()` is (best solution, best fitness), metrics.
best_solution is a numpy array, best fitness is a float, and metrics is a list of tuples.

### Examples

###### Real-valued Optimization
Real-valued example. Relatively easy problem, does not require a huge population.

```
import networkx as nx
def styblinski_tang(solution):
    return np.sum(solution**4 - 16*solution**2 + 5*solution)/2


(best_solution, best_fitness), metrics = GeneticOptimizer(objective=styblinski_tang,
                                                          G=nx.grid_2d_graph(2, 2),
                                                          problem_type='real_valued',
                                                          variables=None,
                                                          lb=np.array([-5]*10),
                                                          ub=np.array([5]*10),
                                                          num_solutions=16,
                                                          mutation_rate=.01,
                                                          max_iterations=750,
                                                          num_islands=4,
                                                          death_rate=.5,
                                                          min_target=None,
                                                          mutation_radius=0,
                                                          args=None,
                                                          integer_solution=False,
                                                          log_metrics=True).GA()

```

###### Sequence Optimization
Sequence example: order 100 numbers

Note the almost absurd difficulty of the problem. There are more possible sequences of 100 items than there are atoms in the universe. For this reason, the island count, members per island, and number of generations are very high and computation takes some time. Higher values for these achieves better probability of optimal solution.
   
The experiment takes a while, but it does 30 runs of lattice and social network topologies to reveal that the latter is better. Better to do in parallel but I wanted to keep the dependencies simple

```
import networkx as nx
import matplotlib.pyplot as plt


def order_numbers(solution):
    """Given a list of numbers from 0 to n, the optimal solution is to sequence the
       numbers in either increasing or decreasing order. The optimal value is n."""
    return np.sum(np.abs(np.diff(solution)))


variables = list(range(100))
np.random.shuffle(variables)

GO = GeneticOptimizer(objective=order_numbers,
                      G=nx.grid_2d_graph(20, 20),
                      problem_type='sequence',
                      variables=variables,
                      lb=None,
                      ub=None,
                      num_solutions=32,
                      mutation_rate=.01,
                      max_iterations=750,
                      num_islands=400,
                      death_rate=.5,
                      min_target=None,
                      mutation_radius=0,
                      args=None,
                      integer_solution=False,
                      log_metrics=True)
experiment_lattice = [GO.GA() for _ in range(30)]

GO = GeneticOptimizer(objective=order_numbers,
                      G=nx.connected_watts_strogatz_graph(n=400, k=10, p=.1),
                      problem_type='sequence',
                      variables=variables,
                      lb=None,
                      ub=None,
                      num_solutions=32,
                      mutation_rate=.01,
                      max_iterations=750,
                      num_islands=400,
                      death_rate=.5,
                      min_target=None,
                      mutation_radius=0,
                      args=None,
                      integer_solution=False,
                      log_metrics=True)
experiment_social_network = [GO.GA() for _ in range(30)]

plt.plot(np.mean([np.array(x[1])[:, 0] for x in experiment_lattice], axis=0), color='blue', label='lattice')
plt.plot(np.mean([np.array(x[1])[:, 0] for x in experiment_social_network], axis=0), color='green', label='social network')
plt.xlabel('generation')
plt.ylabel('best individual fitness')
plt.title('lattice vs social network -- average of 30 runs')
plt.legend()
```

## Simulated Annealing for Multi-Objectives

### Dependencies 
* numpy
* math
* numba
* bisect
* scipy

### Arguments
*The SimulatedAnnealing class requires several inputs:*

Argument                  | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Object&nbsp;Type&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;               | Default           | Description
------------------------- | :-------------------------: | :----------------:  | ----------------
objectives                | `python class   `           | *required*          | The objective functions class to be optimized. Has neighbor and fitness methods.
max_epochs                | `integer`                   | 150                 | The number of temperature iterations to do 
M                         | `integer`                   | 1000                | The number of iterations per temperature iteration
a                         | `float`                     | .925                | The exponential decay rate.
diagnostic_sample_size    | `integer`                   | 1000                | The number of iterations (like M) to do as a warm up phase to determine how many iterations to do based on how the objective function changes and the chosen a value.
num_solutions_to_return   | `integer of np.int`         | np.inf              | The number of solutions to return. If inf, return everything in the pareto frontier that was found.
save_attainment_snapshots | `boolean`                   | False               | Saves pareto frontier at each temperature horizon. Saves in the attainment_snapshots attribute after algorithm runs

### Outputs
pareto_solutions and pareto_fitnesses, both numpy arrays.

### Examples
###### Real-valued Optimization
The MultiObjective class is itself an example on how to use this and can be found along with SimulatedAnnealing class. It models having a neighbor and fitness function, so that you can do other kinds of problems with two competing objectives.
```
def kursawe(solution):
    """lb = np.array([-5, -5, -5])
       ub = np.array([5, 5, 5])"""
  
    s1 = -10*np.exp(-.2*np.sqrt(solution[0]**2 + solution[1]**2))
    s2 = -10*np.exp(-.2*np.sqrt(solution[1]**2 + solution[2]**2))
    fitness1 = s1 + s2
    fitness2 = np.sum(np.abs(solution)**.8 + 5*np.sin(solution**3))
    return np.array([-fitness1, -fitness2])
  
  
objectives = MultiObjective(lb=np.array([-5]*3),
                            ub=np.array([5]*3), 
                            objective_function=kursawe,
                            neighborhood_radius=.001)
SA = SimulatedAnnealing(objectives=objectives,
                        max_epochs=1500,
                        M=3000,
                        a=.999,
                        diagnostic_sample_size=1500,
                        save_attainment_snapshots=False)
pareto_solutions, pareto_fitnesses = SA.optimize()
  
plt.scatter(-pareto_fitnesses[:, 0], -pareto_fitnesses[:, 1], s=1)
```

## Reference
Kevin I. Smith, Richard M. Everson, Jonathan E. Fieldsend, "Dominance-Based
Multi-Objective Simulated Annealing." IEEE Transactions on Evolutionary Computation, 2008, https://ore.exeter.ac.uk/repository/bitstream/handle/10871/15260/Dominance-Based%20Multi-Objective%20Simulated%20Annealing.pdf;jsessionid=64435C0B8B0218B1042160A3353B1D41?sequence=5

