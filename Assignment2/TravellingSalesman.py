import mlrose_hiive
import numpy as np
import sys
sys.path.append("../..")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from textwrap import wrap
import random

# plt.switch_backend('agg')

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

seed = 903387974
random.seed(seed)


def main():
    # Create list of city coordinates
    X = 10
    Y = 10
    possible_coordinates = [(x, y) for x in range(X) for y in range(1, Y + 1)]
    for(i in range(10,50,10)):
        coords_list = random.sample(possible_coordinates, 50)
    # coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]
    # Define optimization problem object
    maximize = True
    if maximize:
        label = "Maximize"
    else:
        label = "Minimize"

    problem_no_fit = mlrose_hiive.TSPOpt(length=len(coords_list), coords=coords_list, maximize=maximize)

    ga_tuning = []
    mimic_tuning = []
    ga_mimic_hyper_params = [(150, 0.3), (150, 0.5), (150, 0.1), (200, 0.03), (200, 0.05), (200, 0.1), (250, 0.03),
                             (250, 0.05), (250, 0.01), (275, 0.03), (275, 0.5), (275, 0.10), (300, 0.03), (300, 0.5),
                             (300, 0.10)]

    for p, m in ga_mimic_hyper_params:
        _, best_fitness_ga = mlrose_hiive.genetic_alg(problem_no_fit, pop_size=p, mutation_prob=m, max_attempts=100,
                                                max_iters=1000, random_state=seed)
        ga_tuning.append((p, m, best_fitness_ga))
        _, best_fitness_mimic = mlrose_hiive.mimic(problem_no_fit, pop_size=p, keep_pct=m, max_attempts=100, max_iters=1000,
                                       curve=False, random_state=seed, fast_mimic=False)
        mimic_tuning.append((p, m, best_fitness_mimic))

    plot_fitness(f'GA for TSP ' + label + ' ' + str(len(coords_list)), ga_tuning, 'population', 'mutation', 'fitness')
    plot_fitness('MIMIC for TSP', ga_tuning, 'population', 'keep_pct', 'fitness')

    # Solve using genetic algorithm
    _, best_fitness, ga_curves = mlrose_hiive.genetic_alg(problem_no_fit, mutation_prob=0.2, max_attempts=100,
                                                          max_iters=20000,
                                                    random_state=seed, curve=True)
    # Solve using MIMIC
    _, best_fitness, mimic_curves = mlrose_hiive.mimic(problem_no_fit, pop_size=200, keep_pct=0.2, max_attempts=100,
                                                 max_iters=20000, curve=True, random_state=seed, fast_mimic=False)

    _, best_fitness, sa_curves = mlrose_hiive.simulated_annealing(problem_no_fit, max_attempts=100, max_iters=20000,
                                                            curve=True, random_state=seed)
    _, best_fitness, rhc_curves = mlrose_hiive.random_hill_climb(problem_no_fit, max_attempts=100, max_iters=20000,
                                                                 curve=True, random_state=seed)

    zippedlist = list(zip(ga_curves, mimic_curves, sa_curves, rhc_curves))
    df_TSP = pd.DataFrame(zippedlist, columns=['ga', 'mimic', 'sa', 'rhc'])
    df_TSP.index += 1
    ax = df_TSP.plot()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig('./graphs/' + 'TSP' + '_' + str(timestr) + '.png')


def plot_fitness(name, values, param1, param2, param3):
    df = pd.DataFrame(values, columns=[param1, param2, param3])
    df = df.pivot(index=param1, columns=param2, values=param3) \
        .reset_index()
    df = df.melt(param1, var_name=param2, value_name=param3)
    g = sns.catplot(x=param1, y=param3, hue=param2, data=df)
    title = f"Plot for a %s" % "\n".join(wrap(f"{name}", width=150))
    g.fig.suptitle(title)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig('./graphs/' + name.replace(" ", "_") + '_' + str(timestr) + '.png')
    plt.close()

if __name__ == "__main__":
    main()
