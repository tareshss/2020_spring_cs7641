1. Git clone https://github.com/tareshss/2020_spring_cs7641.git

2. Install Anaconda

3. conda search -c bioconda -c conda-forge

4. conda env create -f environment.yml

5. Activate conda env

6. run python learning_curves.py to generate learning curves in learning_curves folder.

7. run grid_search.py to generate grid search graphs in grid_search folder along with best fit learning curves
    in grid_search_learning_curves.
    - Also generates accuracy reports in reports folder,showing the accuracies of the 20% holdout set
    - Also log files are generated with grid_search.py, show the best fit model parameters from grid search.

8.  Code copied from:
    - https://www.kaggle.com/lucabasa/credit-card-default-a-very-pedagogical-notebook
    - learning_curves from https://www.dataquest.io/blog/learning-curves-machine-learning/
    - Some basic charting from stackoverflow
    - classification_report, plot_learning_curve function from https://scikit-learn.org/

9.  Major Libraries SciKit-Learn, MatplotLib, Pandas and Numpy and Anaconda as package manager.
    For complete list of libraries look at req.txt