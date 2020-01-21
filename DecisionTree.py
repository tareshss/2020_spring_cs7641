import sys
sys.path.append("../..")
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import train_test_split, learning_curve
import matplotlib.pyplot as plt


logger: logging.getLogger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.hasHandlers():
    sh = logging.StreamHandler()
    fmt = logging.Formatter(fmt="[%(asctime)s] [%(process)d] %(name)-12s %(levelname)-8s %(message)s")
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.propagate = False

verbose = False
credit_card_path = './data/UCI_Credit_Card.csv'


def main():
    seed = 903387974
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if seed is None:
        seed = np.random.randint(0, (2 ** 31) - 1)
        logger.info("Using seed {}".format(seed))
    df = pd.read_csv(credit_card_path)

    # code copied from https://www.kaggle.com/kernels/scriptcontent/2094910/download
    df = df.rename(columns={'default.payment.next.month': 'def_pay',
                            'PAY_0': 'PAY_1'})

    y = df['def_pay'].copy()
    features = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'PAY_2',
                'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
                'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
                'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    X = df[features].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
    classifier = DecisionTreeClassifier(max_depth=10, random_state=14)
    # training the classifier
    classifier.fit(X_train, y_train)
    # do our predictions on the test
    predictions = classifier.predict(X_test)
    # see how good we did on the test
    logger.info(accuracy_score(y_true=y_test, y_pred=predictions))
    train_sizes = [10, 100, 500, 2000, 5000, 8000, 10000, 12000, 15000, 18000, 20000, 24000]

    model = DecisionTreeClassifier(max_depth=10, random_state=14)

    learning_curves(model, df, features, 'def_pay', train_sizes, 5)
    pass


# code copied from https://www.dataquest.io/blog/learning-curves-machine-learning/
def learning_curves(estimator, data, features, target, train_sizes, cv):

    train_sizes, train_scores, validation_scores = learning_curve(estimator, data[features], data[target],
                                                                 train_sizes=train_sizes, cv=cv,
                                                                 scoring ='neg_mean_squared_error')
    train_scores_mean = -train_scores.mean(axis=1)
    validation_scores_mean = -validation_scores.mean(axis=1)
    plt.plot(train_sizes, train_scores_mean, label='Training error')
    plt.plot(train_sizes, validation_scores_mean, label='Validation error')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    title = 'Learning curves for a ' + str(estimator).split('(')[0] + ' model'
    plt.title(title, fontsize=18, y=1.03)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
