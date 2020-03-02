1. git clone https://github.com/tareshss/2020_spring_cs7641.git

2. Navigate to Assignment2 Directory

3. Install Anaconda

4. conda search -c bioconda -c conda-forge

5. conda env create -f environment.yml

6. Activate conda env

7. pip install mlrose-hiive

8. Run
    a) python FlipFlop.py to generate data in the output folder.
    b) python Knapsack.py to generate data in the output folder.
    c) python TravellingSalesman.py to generate data in the output folder.
    d) python Graphs_FlipFlop.py to generate graphs in graphs folder from data in output folder.
    e) python Graphs_Knapsack.py to generate graphs in graphs folder from data in output folder.
    f) python Graphs_TravellingSalesman.py to generate graphs in graphs folder from data in output folder.
    g) python nn.py to generate logfile with accuracy and graphs in graphs folder
        Manually adjust max_iterations in algorithms
         [('random_hill_climb', 20000), ('simulated_annealing', 20000)
                  , ('gradient_descent', 50), ('genetic_alg', 50)]

9.  Major Libraries mlrose, mlrose-hiive, SciKit-Learn, MatplotLib, Pandas and Numpy and Anaconda as package manager.
    For complete list of libraries look at environment.yml