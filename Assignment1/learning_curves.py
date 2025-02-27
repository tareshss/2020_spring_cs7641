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
from sklearn.preprocessing import StandardScaler
from textwrap import wrap
import datetime

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
pen_digits_file = 'pendigits_csv.csv'


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

    train_sizes_credit = [100, 500, 2000, 5000, 8000, 10000, 12000, 15000, 18000, 20000, 24000]
    x_scaler = StandardScaler()
    df_credit[features_credit] = x_scaler.fit_transform(df_credit[features_credit])

    # load pen digits

    df_pen = pd.read_csv(data_directory/pen_digits_file)
    features_pen = ['input1', 'input2', 'input3', 'input4', 'input5', 'input6', 'input7', 'input8', 'input9',
                    'input10', 'input11', 'input12', 'input13', 'input14', 'input15', 'input16']

    df_pen[features_pen] = x_scaler.fit_transform(df_pen[features_pen])
    train_sizes_pen = [100, 500, 1000, 1500, 3000, 4000, 5000, 6000, 7000, 8793]

    run_estimators(credit_card_file, df_credit, features_credit, 'def_pay', seed, train_sizes_credit)
    run_estimators(pen_digits_file, df_pen, features_pen, 'class', seed, train_sizes_pen)


def run_estimators(fileName, df, features, target, seed, train_sizes):
    models = [
        ('Decision Tree', DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=seed,
                                                 splitter='random')),
        ('Ada Boost', AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='gini',
                                                                               max_depth=10,
                                                                               random_state=seed,
                                                                               splitter='random'),
                                         random_state=seed)),
        ('KNN', KNeighborsClassifier()),
        ('NN', MLPClassifier(hidden_layer_sizes=(50, 50, 50), max_iter=1000, random_state=seed)),
        ('SVM rbf', SVC(kernel='rbf')),
        ('SVM sigmoid', SVC(kernel='sigmoid'))
    ]
    # model = DecisionTreeClassifier(max_depth=10, random_state=14)
    # fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    for name, model in models:
        # timestr = time.strftime("%Y%m%d-%H%M%S")
        logger.info(f"Calling Trainer {name} {fileName}")
        learning_curves(name + ' ' + fileName, model, df, features, target, train_sizes, 5)
    pass


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
