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
# plt.switch_backend('agg')
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from collections import OrderedDict
from textwrap import wrap

logger: logging.getLogger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.hasHandlers():
    sh = logging.StreamHandler()
    fmt = logging.Formatter(fmt="[%(asctime)s] [%(process)d] %(name)-12s %(levelname)-8s %(message)s")
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.propagate = False

verbose = False

data_directory = Path("data/")
credit_card_file = 'UCI_Credit_Card.csv'
pen_digits_file = 'pendigits_csv.csv'


# copied from https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    score = pd.DataFrame(cv_results).sort_values(by=f"param_" + name_param_2, ascending=False)

    score2 = pd.DataFrame(cv_results).sort_values(by="mean_test_score", ascending=False)

    scores_mean = score['mean_test_score'].to_numpy()
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2), len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label=name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    ax.legend(bbox_to_anchor=(1.1, 1.05))
    plt.show()
    # plt.savefig('./grid_search/' + name + '_' + str(timestr) + '.png')


def main():
    seed = 903387974
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if seed is None:
        seed = np.random.randint(0, (2 ** 31) - 1)
        logger.info("Using seed {}".format(seed))
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

    # Split 20% of data set for testing
    x_train_credit, x_test_credit, y_train_credit, y_test_credit = train_test_split(
        df_credit[features_credit], df_credit['def_pay'], test_size=0.2, random_state=seed)

    # load pen digits

    df_pen = pd.read_csv(data_directory / pen_digits_file)
    features_pen = ['input1', 'input2', 'input3', 'input4', 'input5', 'input6', 'input7', 'input8', 'input9',
                    'input10', 'input11', 'input12', 'input13', 'input14', 'input15', 'input16']

    df_pen[features_pen] = x_scaler.fit_transform(df_pen[features_pen])

    # Split 20% of data set for testing
    x_train_pen, x_test_pen, y_train_pen, y_test_pen = train_test_split(
        df_pen[features_pen], df_pen['class'], test_size=0.2, random_state=seed)

    run_grid_search(pen_digits_file, x_train_pen, y_train_pen, seed)
    run_grid_search(credit_card_file, x_train_credit, y_train_credit, seed)

    # best_tree = grid.best_estimator_
    # logger.info(best_tree)
    #
    # y_true_pen, y_pred_test = y_test_pen, grid.predict(x_test_pen)
    # report(grid.cv_results_)
    # logger.info(classification_report(y_true_pen, y_pred_test))
def run_grid_search(filename, x_train_pen, y_train_pen, seed):

    models = [
        (DecisionTreeClassifier(),
               'Decision Tree',
               OrderedDict([('min_samples_leaf', np.arange(1, 10)), ('max_depth', np.arange(1, 21, 2))]))
        #        (AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), random_state=seed),
        #         #        'Ada Boost',
        #         #        OrderedDict([('n_estimators', np.arange(10, 100, 10)),
        #         #                     ('base_estimator__splitter', ["best", "random"])])),
        # (KNeighborsClassifier(), 'KNN',  OrderedDict([('n_neighbors', np.arange(1, 10)),
        # ('weights', ["uniform","distance"])])),
        #          (MLPClassifier(max_iter=1000), 'NN',
        #           OrderedDict([('hidden_layer_sizes', [(50, 50, 50), (50, 100, 50), (100,)]),
        #                        ('activation', ['tanh', 'relu'])])),
        # (SVC(kernel='rbf'),
        #  'SVM rbf',
        #  OrderedDict([('C', [0.1, 1, 10, 100, 1000]),
        #               ('gamma', [1, 0.1, 0.01, 0.001, 0.0001])])),
        # (SVC(kernel='sigmoid'),
        #  'SVM sigmoid',
        #  OrderedDict([('C', [0.1, 1, 10, 100, 1000]),
        #               ('gamma', [1, 0.1, 0.01, 0.001, 0.0001])]))
              ]

    for model, name, ordered_dict in models:
        # timestr = time.strftime("%Y%m%d-%H%M`1`S")
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
        grid.fit(x_train_pen, y_train_pen)
        plot_grid(name + ' ' + filename, grid, param_1_name, param_2_name)


def plot_grid(name, grid, param1, param2):
    score = pd.DataFrame(grid.cv_results_)
    df_score = score[[f'param_' + param1, f'param_' + param2, 'mean_test_score']].copy()
    df_score['mean_test_score'] *= -1
    df_score = df_score.pivot(index=f'param_' + param1, columns=f'param_' + param2, values='mean_test_score') \
        .reset_index()
    df_score = df_score.melt(f'param_' + param1, var_name=param2, value_name='mean_test_score')
    g = sns.catplot(x=f"param_" + param1, y="mean_test_score", hue=param2, data=df_score)
    # leg = g._legend
    # leg.set_bbox_to_anchor([1, 1])  # coordinates of lower left of bounding box
    # leg._loc = 2  # if required you can set the loc
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    title = f"RandomizedSearchCV for a %s" % "\n".join(wrap(f"{name}", width=150))
    g.fig.suptitle(title)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig('./grid_search/' + name + '_' + str(timestr) + '.png')

if __name__ == "__main__":
    main()
