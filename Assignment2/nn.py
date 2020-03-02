import sys
from sklearn.ensemble import AdaBoostClassifier
from pathlib import Path
sys.path.append("../..")
import logging
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import train_test_split, learning_curve
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import time
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from textwrap import wrap
import datetime
import mlrose_hiive

import seaborn as sns
from mlrose_hiive.runners import SKMLPRunner
from mlrose_hiive.neural import NeuralNetwork
from mlrose_hiive.algorithms.decay import GeomDecay


LOG_FILENAME = datetime.datetime.now().strftime('logfile_%H_%M_%d_%m_%Y.log')

logger: logging.getLogger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.hasHandlers():
    fh = logging.FileHandler(LOG_FILENAME)
    # sh = logging.StreamHandler()
    fmt = logging.Formatter(fmt="[%(asctime)s] [%(process)d] %(name)-12s %(levelname)-8s %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False

verbose = False

data_directory = Path("data/")
credit_card_file = 'UCI_Credit_Card.csv'

def main():
    seed = 903387974
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if seed is None:
        seed = np.random.randint(0, (2 ** 31) - 1)
        logger.info("Using seed {}".format(seed))
    df_credit = pd.read_csv(data_directory/credit_card_file)

    # code copied from https://www.kaggle.com/kernels/scriptcontent/2094910/download
    df_credit = df_credit.rename(columns={'default.payment.next.month': 'def_pay',
                            'PAY_0': 'PAY_1'})

    features_credit = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'PAY_2',
                'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
                'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
                'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    x_scaler = StandardScaler()
    df_credit[features_credit] = x_scaler.fit_transform(df_credit[features_credit])

    x_scaler = StandardScaler()
    df_credit[features_credit] = x_scaler.fit_transform(df_credit[features_credit])

    # model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000, random_state=seed)
    x_train, x_test, y_train, y_test = train_test_split(df_credit[features_credit], df_credit['def_pay'], test_size=0.2,
                                                        random_state=seed)

    # One hot encode target values
    one_hot = OneHotEncoder()

    y_train_hot = one_hot.fit_transform(y_train.values.reshape(-1, 1)).todense()
    y_test_hot = one_hot.transform(y_test.values.reshape(-1, 1)).todense()

    grid_search_parameters = ({
        'hidden_layer_sizes': [(10, 10, 10)],
        'activation': ['tanh']
    })
    curves = []

    algorithms = [('random_hill_climb', 20000), ('simulated_annealing', 20000)
                  , ('gradient_descent', 50), ('genetic_alg', 50)]
    restarts = 0,
    schedule = GeomDecay(init_temp=10),
    pop_size = 200,
    mutation_prob = 0.1,
    max_attempts = 10,
    for algo, iterations in algorithms:
        model = NeuralNetwork(hidden_nodes=[10, 10, 10], activation='tanh',
                              algorithm=algo, restarts=100,
                              pop_size=250, mutation_prob=0.03,
                              schedule=GeomDecay(init_temp=10),
                              max_iters=iterations, bias=True, is_classifier=True,
                              learning_rate=0.001, early_stopping=False,
                              clip_max=5, max_attempts=100, random_state=seed,
                              curve=True)
        start_time = time.time()
        model.fit(x_train, y_train_hot)
        logger.info(f"fit time {algo} {iterations} {str(round(time.time() - start_time, 2))}")
        start_time = time.time()
        # Predict labels for train set and assess accuracy
        y_train_pred = model.predict(x_train)
        logger.info(f"Train predict time {algo} {iterations} {str(round(time.time() - start_time, 2))}")
        y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
        logger.info(f"Train Accuracy {algo} {iterations} {y_train_accuracy}")
        # Predict labels for test set and assess accuracy
        start_time = time.time()
        y_test_pred = model.predict(x_test)
        logger.info(f"Test predict time {algo} {iterations} {str(round(time.time() - start_time, 2))}")
        y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

        logger.info(f"Test Accuracy {algo} {iterations} {y_test_accuracy}")
        curves.append(model.fitness_curve)

    algo_list = [algorithm[0] for algorithm in algorithms]
    df_curves = pd.DataFrame({algo_list[0]: pd.Series(curves[0]), algo_list[1]: pd.Series(curves[1]),
                              algo_list[2]: pd.Series(curves[2]),
                              algo_list[3]: pd.Series(curves[3])})
    df_curves.fillna(method='ffill', inplace=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    df_curves.to_csv(f'./output/df_curves_{timestr}.csv', index=False)
    ax = df_curves.plot()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig('./graphs/' + 'Credit_All_Algorithms' + '_' + str(timestr) + '.png')

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
