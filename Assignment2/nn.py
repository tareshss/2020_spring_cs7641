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

    # runner = SKMLPRunner(x_train, y_train_hot, x_test, y_test_hot, 'NN', seed,
    #                      iteration_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
    #                      grid_search_parameters=grid_search_parameters, output_directory='./output')
    # runner.run()
    # # (self, x_train, y_train, x_test, y_test, experiment_name, seed, iteration_list,
    # #  grid_search_parameters, grid_search_scorer_method=skmt.balanced_accuracy_score,
    # #  early_stopping=True, max_attempts=500, n_jobs=1, cv=5,
    # #  generate_curves=True, output_directory=None, replay=False, ** kwargs):
    # model = NeuralNetwork(hidden_nodes=[10, 10, 10], activation='tanh',
    #                                    algorithm='random_hill_climb',
    #                                    max_iters=200, bias=True, is_classifier=True,
    #                                    learning_rate=0.001, early_stopping=False,
    #                                    clip_max=5, max_attempts=100, random_state=seed, curve=True)
    #
    # model.fit(x_train, y_train_hot)
    # Predict labels for train set and assess accuracy
        # y_train_pred = model.predict(x_train)
        #
        # y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
        #
        # logger.info(y_train_accuracy)
        #
        # # Predict labels for test set and assess accuracy
        # y_test_pred = model.predict(x_test)
        #
        # y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
        #
        # logger.info(y_test_accuracy)

        # model.fitness_curve

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


# code copied from https://www.dataquest.io/blog/learning-curves-machine-learning/
def learning_curves(name, estimator, data, features, target, train_sizes, cv):

    train_sizes, train_scores, validation_scores = learning_curve(estimator, data[features], data[target],
                                                                 train_sizes=train_sizes, cv=cv,
                                                                 scoring ='neg_mean_squared_error')
    train_scores_mean = -train_scores.mean(axis=1)
    validation_scores_mean = -validation_scores.mean(axis=1)
    plt.plot(train_sizes, train_scores_mean, label='Training error')
    plt.plot(train_sizes, validation_scores_mean, label='Validation error')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    # title = 'Learning curves for a ' + str(estimator).split('(')[0] + ' model'
    title = f"Learning curves for a\n%s" % "\n".join(wrap(f"{name}", width=60))
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.title(title, fontsize=14, y=1.03)
    plt.margins(0)
    plt.legend()
    # plt.show()
    plt.savefig('./learning_curves/' + name + '_' + str(timestr) + '.png')
    plt.close()

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes_credit : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(20, 5))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes.legend(loc="best")

    # # Plot n_samples vs fit_times
    # axes[1].grid()
    # axes[1].plot(train_sizes, fit_times_mean, 'o-')
    # axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
    #                      fit_times_mean + fit_times_std, alpha=0.1)
    # axes[1].set_xlabel("Training examples")
    # axes[1].set_ylabel("fit_times")
    # axes[1].set_title("Scalability of the model")
    #
    # # Plot fit_time vs score
    # axes[2].grid()
    # axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    # axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
    #                      test_scores_mean + test_scores_std, alpha=0.1)
    # axes[2].set_xlabel("fit_times")
    # axes[2].set_ylabel("Score")
    # axes[2].set_title("Performance of the model")

    return plt


if __name__ == "__main__":
    main()
