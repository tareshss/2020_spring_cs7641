import sys
import numpy as np
sys.path.append("../..")
import random
from mlrose_hiive.runners import GARunner, MIMICRunner, RHCRunner, SARunner
from mlrose_hiive.generators import KnapsackGenerator
# plt.switch_backend('agg')

seed = 903387974
random.seed(seed)

def main():
    experiment_name_ga = 'Knapsack_GA'
    OUTPUT_DIRECTORY = './output'
    problem = KnapsackGenerator.generate(number_of_items_types=20, seed=seed)
    ga = GARunner(problem=problem,
                  experiment_name=experiment_name_ga,
                  output_directory=OUTPUT_DIRECTORY,
                  seed=seed,
                  iteration_list=2 ** np.arange(12),
                  max_attempts=200,
                  population_sizes=[150, 200, 250, 300],
                  mutation_rates=[0.01, 0.02, 0.03, 0.1, 0.2, 0.3, 0.4])
    ga.run()
    experiment_name_sa = 'Knapsack_SA'
    sa = SARunner(problem=problem,
                  experiment_name=experiment_name_sa,
                  output_directory=OUTPUT_DIRECTORY,
                  seed=seed,
                  iteration_list=2 ** np.arange(14),
                  max_attempts=200,
                  temperature_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000])

    sa.run()
    experiment_name_mimic = 'Knapsack_MIMIC'
    mmc = MIMICRunner(problem=problem,
                      experiment_name=experiment_name_mimic,
                      output_directory=OUTPUT_DIRECTORY,
                      seed=seed,
                      iteration_list=2 ** np.arange(12),
                      max_attempts=200,
                      population_sizes=[150, 200, 250, 300],
                      keep_percent_list=[0.01, 0.02, 0.03, 0.1, 0.2, 0.3, 0.4])
    mmc.run()
    experiment_name_rhc = 'Knapsack_RHC'
    rhc = RHCRunner(problem=problem,
                    experiment_name=experiment_name_rhc,
                    output_directory=OUTPUT_DIRECTORY,
                    seed=seed,
                    iteration_list=2 ** np.arange(12),
                    max_attempts=5000,
                    restart_list=[25, 75, 100])
    rhc.run()

if __name__ == "__main__":
    main()
