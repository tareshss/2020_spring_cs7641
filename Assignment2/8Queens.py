import random
import mlrose
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from textwrap import wrap

seed = 903387974
random.seed(seed)


def main():
    # fitness = mlrose.Queens()
    # problem = mlrose.DiscreteOpt(length=8, fitness_fn=fitness, maximize=False, max_val=8)
    schedule = mlrose.ExpDecay()
    # Check function is working correctly
    init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    # The fitness of this state should be 22
    fitness_cust = mlrose.CustomFitness(queens_max)
    # Define optimization problem object
    problem_cust = mlrose.DiscreteOpt(length=8, fitness_fn=fitness_cust, maximize=True, max_val=8)

    # Solve using genetic algorithm
    _, best_fitness, ga_curves = mlrose.genetic_alg(problem_cust, mutation_prob=0.2, max_attempts=100, max_iters=20000,
                                             random_state=seed, curve=True)
    # Solve using MIMIC
    _, best_fitness, mimic_curves = mlrose.mimic(problem_cust, pop_size=200, keep_pct=0.2, max_attempts=100,
                                                 max_iters=20000, curve=True, random_state=seed, fast_mimic=False)

    _, best_fitness, sa_curves = mlrose.simulated_annealing(problem_cust, max_attempts=100, max_iters=20000,
                                                            curve=True, random_state=seed)
    _, best_fitness, rhc_curves = mlrose.random_hill_climb(problem_cust, max_attempts=100, max_iters=20000, curve=True,
                                               random_state=seed)

    zippedlist = list(zip(ga_curves, mimic_curves, sa_curves, rhc_curves))
    df_TSP = pd.DataFrame(zippedlist, columns=['ga', 'mimic', 'sa', 'rhc'])
    df_TSP.index += 1
    ax = df_TSP.plot()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig('./graphs/' + '8Queens' + '_' + str(timestr) + '.png')



# Define alternative N-Queens fitness function for maximization problem
def queens_max(state):
    # Initialize counter
    fitness = 0

    # For all pairs of queens
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):

            # Check for horizontal, diagonal-up and diagonal-down attacks
            if (state[j] != state[i]) \
                    and (state[j] != state[i] + (j - i)) \
                    and (state[j] != state[i] - (j - i)):
                # If no attacks, then increment counter
                fitness += 1

    return fitness


if __name__ == "__main__":
    main()
