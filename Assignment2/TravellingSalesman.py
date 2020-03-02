import numpy as np
import sys
import sys

import numpy as np
sys.path.append("../..")
import random
from mlrose_hiive.runners import GARunner, MIMICRunner, RHCRunner, SARunner
from mlrose_hiive.generators import TSPGenerator

# plt.switch_backend('agg')

seed = 903387974
random.seed(seed)


def main():
    experiment_name_ga = 'TSP_Maximize_GA'
    OUTPUT_DIRECTORY = './output'
    problem = TSPGenerator.generate(seed=seed, number_of_cities=50, maximize=True)
    ga = GARunner(problem=problem,
                  experiment_name=experiment_name_ga,
                  output_directory=OUTPUT_DIRECTORY,
                  seed=seed,
                  iteration_list=2 ** np.arange(12),
                  max_attempts=200,
                  population_sizes=[150, 200, 250, 300],
                  mutation_rates=[0.01, 0.02, 0.03, 0.1, 0.2, 0.3, 0.4])
    ga.run()
    experiment_name_mimic = 'TSP_Maximize_MIMIC'
    mmc = MIMICRunner(problem=problem,
                      experiment_name=experiment_name_mimic,
                      output_directory=OUTPUT_DIRECTORY,
                      seed=seed,
                      iteration_list=2 ** np.arange(12),
                      max_attempts=200,
                      population_sizes=[150, 200, 250, 300],
                      keep_percent_list=[0.01, 0.02, 0.03, 0.1, 0.2, 0.3, 0.4])
    mmc.run()
    experiment_name_rhc = 'TSP_Maximize_RHC'
    rhc = RHCRunner(problem=problem,
                    experiment_name=experiment_name_rhc,
                    output_directory=OUTPUT_DIRECTORY,
                    seed=seed,
                    iteration_list=2 ** np.arange(12),
                    max_attempts=200,
                    restart_list=[25, 75, 100])
    rhc.run()
    experiment_name_sa = 'TSP_Maximize_SA'
    sa = SARunner(problem=problem,
                  experiment_name=experiment_name_sa,
                  output_directory=OUTPUT_DIRECTORY,
                  seed=seed,
                  iteration_list=2 ** np.arange(14),
                  max_attempts=200,
                  temperature_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000])

    sa.run()

    OUTPUT_DIRECTORY = './output/N_CITIES'
    for i in range(10, 100):
        problem = TSPGenerator.generate(seed=seed, number_of_cities=i, maximize=True)
        experiment_name_cities_ga = str(i) + '_' + 'City_Fitness_ga'
        ga = GARunner(problem=problem,
                      experiment_name=experiment_name_cities_ga,
                      output_directory=OUTPUT_DIRECTORY,
                      seed=seed,
                      iteration_list=2 ** np.arange(12),
                      max_attempts=200,
                      population_sizes=[300],
                      mutation_rates=[0.03])
        ga.run()
        experiment_name_cities_MIMIC = str(i) + '_' + 'City_Fitness_MIMIC'
        mmc = MIMICRunner(problem=problem,
                          experiment_name=experiment_name_cities_MIMIC,
                          output_directory=OUTPUT_DIRECTORY,
                          seed=seed,
                          iteration_list=2 ** np.arange(12),
                          max_attempts=200,
                          population_sizes=[300],
                          keep_percent_list=[0.2])
        mmc.run()
        experiment_name_cities_RHC = str(i) + '_' + 'City_Fitness_RHC'
        rhc = RHCRunner(problem=problem,
                        experiment_name=experiment_name_cities_RHC,
                        output_directory=OUTPUT_DIRECTORY,
                        seed=seed,
                        iteration_list=2 ** np.arange(12),
                        max_attempts=200,
                        restart_list=[25])
        rhc.run()
        experiment_name_cities_sa = str(i) + '_' + 'City_Fitness_SA'
        sa = SARunner(problem=problem,
                      experiment_name=experiment_name_cities_sa,
                      output_directory=OUTPUT_DIRECTORY,
                      seed=seed,
                      iteration_list=2 ** np.arange(14),
                      max_attempts=200,
                      temperature_list=[10])

        sa.run()


if __name__ == "__main__":
    main()
