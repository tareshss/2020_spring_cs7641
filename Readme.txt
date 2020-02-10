1. Git clone https://github.com/tareshss/2020_spring_cs7641.git
2. Install Anaconda
3. conda create -n cs7641 --file req.txt
4. run python learning_curves.py to generate learning curves in learning_curves folder.
5. run grid_search.py to generate grid search graphs in grid_search folder along with best fit learning curves
    in grid_search_learning_curves. Also generates accuracy reports in reports folder.
6. log files are generated with grid_search.py showing the accuracies of the 20% holdout set and best fit model
    parameters.
7.  Code copied from:
    - https://www.kaggle.com/lucabasa/credit-card-default-a-very-pedagogical-notebook
    - learning_curves from https://www.dataquest.io/blog/learning-curves-machine-learning/
    - Some basic charting from stackoverflow
    - classification_report, plot_learning_curve function from https://scikit-learn.org/