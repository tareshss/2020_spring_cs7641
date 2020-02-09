import itertools
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
from sklearn.metrics import mean_squared_error, classification_report
import seaborn as sns
plt.switch_backend('agg')
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from collections import OrderedDict
from textwrap import wrap
import datetime

seed = 903387974

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


# copied from https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
# Utility function to report best scores
def report(results, n_top=1):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            logger.info("Model with rank: {0}".format(i))
            logger.info("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            logger.info("Parameters: {0}".format(results['params'][candidate]))
            logger.info("")

def main():

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    df_credit = pd.read_csv(data_directory / credit_card_file)

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

    # load pen digits
    df_pen = pd.read_csv(data_directory / pen_digits_file)
    features_pen = ['input1', 'input2', 'input3', 'input4', 'input5', 'input6', 'input7', 'input8', 'input9',
                    'input10', 'input11', 'input12', 'input13', 'input14', 'input15', 'input16']

    df_pen[features_pen] = x_scaler.fit_transform(df_pen[features_pen])

    train_sizes_pen = [100, 500, 1000, 1500, 3000, 4000, 5000, 6000, 7000, 8793]
    run_grid_search(pen_digits_file, df_pen, features_pen, 'class', train_sizes_pen)

    train_sizes_credit = [100, 500, 2000, 5000, 8000, 10000, 12000, 15000, 18000, 20000, 24000]
    run_grid_search(credit_card_file, df_credit, features_credit, 'def_pay', train_sizes_credit)


def run_grid_search(filename, df, features, pred, train_sizes):

    # Split 20% of data set for testing
    x_train, x_test, y_train, y_test = train_test_split(
        df[features], df[pred], test_size=0.2, random_state=seed)

    models = [
        (DecisionTreeClassifier(),
         'Decision Tree',
         OrderedDict([('min_samples_leaf', np.arange(1, 10)), ('max_depth', np.arange(1, 21, 2))])),
        (AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), random_state=seed),
         'Ada Boost',
         OrderedDict([('n_estimators', np.arange(10, 100, 10)), ('base_estimator__splitter', ["best", "random"])])),
        (KNeighborsClassifier(),
         'KNN',
         OrderedDict([('n_neighbors', np.arange(1, 10)), ('weights', ["uniform", "distance"])])),
        (MLPClassifier(max_iter=1000),
         'NN', OrderedDict([('hidden_layer_sizes',
                             [(10, 10, 10), (20, 20, 50), (20, 20, 100), (20, 100, 50), (30, 10, 50), (30, 30, 100),
                              (30, 40, 10), (100, 100, 50), (100, 100, 100)]),
                            ('activation', ['tanh', 'relu'])])),
        (SVC(kernel='rbf'),
         'SVM rbf',
         OrderedDict([('C', [0.1, 1, 10, 100, 1000]),
                      ('gamma', [1, 0.1, 0.01, 0.001, 0.0001])])),
        (SVC(kernel='sigmoid'),
         'SVM sigmoid',
         OrderedDict([('C', [0.1, 1, 10, 100, 1000]),
                      ('gamma', [1, 0.1, 0.01, 0.001, 0.0001])]))
              ]

    for model, name, ordered_dict in models:
        logger.info(f"Calling GridSearch On {name} {filename}")
        items = list(ordered_dict.items())
        param_1_name = items[0][0]
        param_1 = items[0][1]

        param_2_name = items[1][0]
        param_2 = items[1][1]

        param_grid = dict(ordered_dict)
        grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                                  n_iter=len(param_1) * len(param_2), cv=5, random_state=seed, n_jobs=-1,
                                  scoring='neg_mean_squared_error')
        grid.fit(x_train, y_train)
        plt.clf()
        plot_grid(name + ' ' + filename, grid, param_1_name, param_2_name)
        best = grid.best_estimator_
        logger.info(best)

        y_true, y_pred_test = y_test, grid.best_estimator_.predict(x_test)
        report(grid.cv_results_)
        test_report = classification_report(y_true, y_pred_test, output_dict=True)
        df_test_report = pd.DataFrame(test_report).transpose()
        # save report to disk
        timestr = time.strftime("%Y%m%d-%H%M%S")
        reports_folder = Path("reports/")

        test_report_name = f"{name}_{filename}_Test_" + str(timestr) + '.csv'
        reports_file_path = reports_folder / test_report_name
        df_test_report.to_csv(reports_file_path)

        #plot learing curves
        plt.clf()
        learning_curves(name + ' ' + filename, model, df, features, pred, train_sizes, 5)


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
    title = f"Best Fit Learning Curve %s" % "\n".join(wrap(f"{name}", width=100))
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.subplots_adjust(top=0.95)
    plt.title(title)
    plt.margins(0)
    plt.legend()
    # plt.show()
    plt.savefig('./grid_search_learning_curves/' + name + '_' + str(timestr) + '.png')
    plt.close()


def plot_grid(name, grid, param1, param2):
    score = pd.DataFrame(grid.cv_results_)
    df_score = score[[f'param_' + param1, f'param_' + param2, 'mean_test_score']].copy()
    df_score['mean_test_score'] *= -1
    df_score = df_score.pivot(index=f'param_' + param1, columns=f'param_' + param2, values='mean_test_score') \
        .reset_index()
    df_score = df_score.melt(f'param_' + param1, var_name=param2, value_name='mean_test_score')
    g = sns.catplot(x=f"param_" + param1, y="mean_test_score", hue=param2, data=df_score)
    title = f"RandomizedSearchCV for a %s" % "\n".join(wrap(f"{name}", width=150))
    g.fig.suptitle(title)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig('./grid_search/' + name + '_' + str(timestr) + '.png')

if __name__ == "__main__":
    main()
