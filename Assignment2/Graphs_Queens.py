import pandas as pd
import time
import matplotlib.pyplot as plt
from functools import reduce
plt.switch_backend('agg')


def main():
    graph_GA()
    graph_MIMIC()
    graph_SA()
    graph_RHC()
    graph_timings()


def graph_GA():
    df = pd.read_csv("./output/Queens_GA/ga__Queens_GA__run_stats_df.csv")
    df_150_03 = df.loc[(df['Population Size'] == 150) & (df['Mutation Rate'] == 0.03)].copy()
    df_150_03 = df_150_03[['Iteration', 'Fitness']]
    df_150_03.rename(columns={'Fitness': 'Pop 150, Mutation 0.03'}, inplace=True)
    df_150_1 = df.loc[(df['Population Size'] == 150) & (df['Mutation Rate'] == 0.1)].copy()
    df_150_1 = df_150_1[['Iteration', 'Fitness']]
    df_150_1.rename(columns={'Fitness': 'Pop 150, Mutation 0.1'}, inplace=True)
    df_150_2 = df.loc[(df['Population Size'] == 150) & (df['Mutation Rate'] == 0.2)].copy()
    df_150_2 = df_150_2[['Iteration', 'Fitness']]
    df_150_2.rename(columns={'Fitness': 'Pop 150, Mutation 0.2'}, inplace=True)
    df_150_3 = df.loc[(df['Population Size'] == 150) & (df['Mutation Rate'] == 0.3)].copy()
    df_150_3 = df_150_3[['Iteration', 'Fitness']]
    df_150_3.rename(columns={'Fitness': 'Pop 150, Mutation 0.3'}, inplace=True)
    df_150_4 = df.loc[(df['Population Size'] == 150) & (df['Mutation Rate'] == 0.4)].copy()
    df_150_4 = df_150_4[['Iteration', 'Fitness']]
    df_150_4.rename(columns={'Fitness': 'Pop 150, Mutation 0.4'}, inplace=True)
    #### Pop 250
    df_250_03 = df.loc[(df['Population Size'] == 250) & (df['Mutation Rate'] == 0.03)].copy()
    df_250_03 = df_250_03[['Iteration', 'Fitness']]
    df_250_03.rename(columns={'Fitness': 'Pop 250, Mutation 0.03'}, inplace=True)
    df_250_1 = df.loc[(df['Population Size'] == 250) & (df['Mutation Rate'] == 0.1)].copy()
    df_250_1 = df_250_1[['Iteration', 'Fitness']]
    df_250_1.rename(columns={'Fitness': 'Pop 250, Mutation 0.1'}, inplace=True)
    df_250_2 = df.loc[(df['Population Size'] == 250) & (df['Mutation Rate'] == 0.2)].copy()
    df_250_2 = df_250_2[['Iteration', 'Fitness']]
    df_250_2.rename(columns={'Fitness': 'Pop 250, Mutation 0.2'}, inplace=True)
    df_250_3 = df.loc[(df['Population Size'] == 250) & (df['Mutation Rate'] == 0.3)].copy()
    df_250_3 = df_250_3[['Iteration', 'Fitness']]
    df_250_3.rename(columns={'Fitness': 'Pop 250, Mutation 0.3'}, inplace=True)
    df_250_4 = df.loc[(df['Population Size'] == 250) & (df['Mutation Rate'] == 0.4)].copy()
    df_250_4 = df_250_4[['Iteration', 'Fitness']]
    df_250_4.rename(columns={'Fitness': 'Pop 250, Mutation 0.4'}, inplace=True)
    #### Pop 300
    df_300_03 = df.loc[(df['Population Size'] == 300) & (df['Mutation Rate'] == 0.03)].copy()
    df_300_03 = df_300_03[['Iteration', 'Fitness']]
    df_300_03.rename(columns={'Fitness': 'Pop 300, Mutation 0.03'}, inplace=True)
    df_300_1 = df.loc[(df['Population Size'] == 300) & (df['Mutation Rate'] == 0.1)].copy()
    df_300_1 = df_300_1[['Iteration', 'Fitness']]
    df_300_1.rename(columns={'Fitness': 'Pop 300, Mutation 0.1'}, inplace=True)
    df_300_2 = df.loc[(df['Population Size'] == 300) & (df['Mutation Rate'] == 0.2)].copy()
    df_300_2 = df_300_2[['Iteration', 'Fitness']]
    df_300_2.rename(columns={'Fitness': 'Pop 300, Mutation 0.2'}, inplace=True)
    df_300_3 = df.loc[(df['Population Size'] == 300) & (df['Mutation Rate'] == 0.3)].copy()
    df_300_3 = df_300_3[['Iteration', 'Fitness']]
    df_300_3.rename(columns={'Fitness': 'Pop 300, Mutation 0.3'}, inplace=True)
    df_300_4 = df.loc[(df['Population Size'] == 300) & (df['Mutation Rate'] == 0.4)].copy()
    df_300_4 = df_300_4[['Iteration', 'Fitness']]
    df_300_4.rename(columns={'Fitness': 'Pop 300, Mutation 0.4'}, inplace=True)

    data_frames = [df_150_03, df_150_1, df_150_2, df_150_3, df_150_4,
                   df_250_03, df_250_1, df_250_2, df_250_3, df_250_4,
                   df_300_03, df_300_1, df_300_2, df_300_3, df_300_4]
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['Iteration'],
                                                    how='outer'), data_frames)
    ax = df_merged.plot(x='Iteration')
    NUM_COLORS = 20
    cm = plt.get_cmap('gist_rainbow')
    ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])

    plt.ylim(0, 10)
    plt.xlim(0, 150)
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 1.1), shadow=True, ncol=1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness")
    title = f"Queens GA"
    plt.title(title, fontsize=14, y=1.03)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig('./graphs/' + 'Queens_GA' + '_' + str(timestr) + '.png')
    plt.close()


def graph_MIMIC():
    df = pd.read_csv("./output/Queens_GA_MIMIC/mimic__Queens_GA_MIMIC__curves_df.csv")
    df_150_03 = df.loc[(df['Population Size'] == 150) & (df['Keep Percent'] == 0.03)].copy()
    df_150_03 = df_150_03[['Iteration', 'Fitness']]
    df_150_03.rename(columns={'Fitness': 'Pop 150, Keep Percent 0.03'}, inplace=True)
    df_150_1 = df.loc[(df['Population Size'] == 150) & (df['Keep Percent'] == 0.1)].copy()
    df_150_1 = df_150_1[['Iteration', 'Fitness']]
    df_150_1.rename(columns={'Fitness': 'Pop 150, Keep Percent 0.1'}, inplace=True)
    df_150_2 = df.loc[(df['Population Size'] == 150) & (df['Keep Percent'] == 0.2)].copy()
    df_150_2 = df_150_2[['Iteration', 'Fitness']]
    df_150_2.rename(columns={'Fitness': 'Pop 150, Keep Percent 0.2'}, inplace=True)
    df_150_3 = df.loc[(df['Population Size'] == 150) & (df['Keep Percent'] == 0.3)].copy()
    df_150_3 = df_150_3[['Iteration', 'Fitness']]
    df_150_3.rename(columns={'Fitness': 'Pop 150, Keep Percent 0.3'}, inplace=True)
    df_150_4 = df.loc[(df['Population Size'] == 150) & (df['Keep Percent'] == 0.4)].copy()
    df_150_4 = df_150_4[['Iteration', 'Fitness']]
    df_150_4.rename(columns={'Fitness': 'Pop 150, Keep Percent 0.4'}, inplace=True)
    #### Pop 250
    df_250_03 = df.loc[(df['Population Size'] == 250) & (df['Keep Percent'] == 0.03)].copy()
    df_250_03 = df_250_03[['Iteration', 'Fitness']]
    df_250_03.rename(columns={'Fitness': 'Pop 250, Keep Percent 0.03'}, inplace=True)
    df_250_1 = df.loc[(df['Population Size'] == 250) & (df['Keep Percent'] == 0.1)].copy()
    df_250_1 = df_250_1[['Iteration', 'Fitness']]
    df_250_1.rename(columns={'Fitness': 'Pop 250, Keep Percent 0.1'}, inplace=True)
    df_250_2 = df.loc[(df['Population Size'] == 250) & (df['Keep Percent'] == 0.2)].copy()
    df_250_2 = df_250_2[['Iteration', 'Fitness']]
    df_250_2.rename(columns={'Fitness': 'Pop 250, Keep Percent 0.2'}, inplace=True)
    df_250_3 = df.loc[(df['Population Size'] == 250) & (df['Keep Percent'] == 0.3)].copy()
    df_250_3 = df_250_3[['Iteration', 'Fitness']]
    df_250_3.rename(columns={'Fitness': 'Pop 250, Keep Percent 0.3'}, inplace=True)
    df_250_4 = df.loc[(df['Population Size'] == 250) & (df['Keep Percent'] == 0.4)].copy()
    df_250_4 = df_250_4[['Iteration', 'Fitness']]
    df_250_4.rename(columns={'Fitness': 'Pop 250, Keep Percent 0.4'}, inplace=True)
    #### Pop 300
    df_300_03 = df.loc[(df['Population Size'] == 300) & (df['Keep Percent'] == 0.03)].copy()
    df_300_03 = df_300_03[['Iteration', 'Fitness']]
    df_300_03.rename(columns={'Fitness': 'Pop 300, Keep Percent 0.03'}, inplace=True)
    df_300_1 = df.loc[(df['Population Size'] == 300) & (df['Keep Percent'] == 0.1)].copy()
    df_300_1 = df_300_1[['Iteration', 'Fitness']]
    df_300_1.rename(columns={'Fitness': 'Pop 300, Keep Percent 0.1'}, inplace=True)
    df_300_2 = df.loc[(df['Population Size'] == 300) & (df['Keep Percent'] == 0.2)].copy()
    df_300_2 = df_300_2[['Iteration', 'Fitness']]
    df_300_2.rename(columns={'Fitness': 'Pop 300, Keep Percent 0.2'}, inplace=True)
    df_300_3 = df.loc[(df['Population Size'] == 300) & (df['Keep Percent'] == 0.3)].copy()
    df_300_3 = df_300_3[['Iteration', 'Fitness']]
    df_300_3.rename(columns={'Fitness': 'Pop 300, Keep Percent 0.3'}, inplace=True)
    df_300_4 = df.loc[(df['Population Size'] == 300) & (df['Keep Percent'] == 0.4)].copy()
    df_300_4 = df_300_4[['Iteration', 'Fitness']]
    df_300_4.rename(columns={'Fitness': 'Pop 300, Keep Percent 0.4'}, inplace=True)

    data_frames = [df_150_03, df_150_1, df_150_2, df_150_3, df_150_4,
                   df_250_03, df_250_1, df_250_2, df_250_3, df_250_4,
                   df_300_03, df_300_1, df_300_2, df_300_3, df_300_4]
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['Iteration'],
                                                    how='outer'), data_frames)
    ax = df_merged.plot(x='Iteration')
    NUM_COLORS = 20
    cm = plt.get_cmap('gist_rainbow')
    ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])

    plt.ylim(0, 15)
    #     # plt.xlim(0, 50)
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 1.1), shadow=True, ncol=1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness")
    title = f"Queens MIMIC"
    plt.title(title, fontsize=14, y=1.03)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig('./graphs/' + 'Queens_MIMMIC' + '_' + str(timestr) + '.png')
    plt.close()


def graph_SA():
    df = pd.read_csv("./output/Queens_GA_SA/sa__Queens_GA_SA__curves_df.csv")
    df_temp_1 = df.loc[(df['Temperature'] == 1)].copy()
    df_temp_1 = df_temp_1[['Iteration', 'Fitness']]
    df_temp_1.rename(columns={'Fitness': 'Temperature 1'}, inplace=True)
    df_temp_10 = df.loc[(df['Temperature'] == 10)].copy()
    df_temp_10 = df_temp_10[['Iteration', 'Fitness']]
    df_temp_10.rename(columns={'Fitness': 'Temperature 10'}, inplace=True)
    df_temp_50 = df.loc[(df['Temperature'] == 50)].copy()
    df_temp_50 = df_temp_50[['Iteration', 'Fitness']]
    df_temp_50.rename(columns={'Fitness': 'Temperature 50'}, inplace=True)
    df_temp_100 = df.loc[(df['Temperature'] == 100)].copy()
    df_temp_100 = df_temp_100[['Iteration', 'Fitness']]
    df_temp_100.rename(columns={'Fitness': 'Temperature 100'}, inplace=True)
    df_temp_250 = df.loc[(df['Temperature'] == 250)].copy()
    df_temp_250 = df_temp_250[['Iteration', 'Fitness']]
    df_temp_250.rename(columns={'Fitness': 'Temperature 250'}, inplace=True)
    df_temp_500 = df.loc[(df['Temperature'] == 500)].copy()
    df_temp_500 = df_temp_500[['Iteration', 'Fitness']]
    df_temp_500.rename(columns={'Fitness': 'Temperature 500'}, inplace=True)
    df_temp_1000 = df.loc[(df['Temperature'] == 1000)].copy()
    df_temp_1000 = df_temp_1000[['Iteration', 'Fitness']]
    df_temp_1000.rename(columns={'Fitness': 'Temperature 1000'}, inplace=True)
    df_temp_2500 = df.loc[(df['Temperature'] == 2500)].copy()
    df_temp_2500 = df_temp_2500[['Iteration', 'Fitness']]
    df_temp_2500.rename(columns={'Fitness': 'Temperature 2500'}, inplace=True)
    df_temp_5000 = df.loc[(df['Temperature'] == 5000)].copy()
    df_temp_5000 = df_temp_5000[['Iteration', 'Fitness']]
    df_temp_5000.rename(columns={'Fitness': 'Temperature 5000'}, inplace=True)
    df_temp_10000 = df.loc[(df['Temperature'] == 10000)].copy()
    df_temp_10000 = df_temp_10000[['Iteration', 'Fitness']]
    df_temp_10000.rename(columns={'Fitness': 'Temperature 10000'}, inplace=True)

    data_frames = [df_temp_1, df_temp_10, df_temp_100, df_temp_250, df_temp_500,
                   df_temp_1000, df_temp_2500, df_temp_5000, df_temp_10000]

    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['Iteration'],
                                                    how='outer'), data_frames)
    ax = df_merged.plot(x='Iteration')
    NUM_COLORS = 9
    cm = plt.get_cmap('gist_rainbow')
    ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])
    # for i in range(NUM_COLORS):
    #     ax.plot(np.arange(10) * (i + 1))
    plt.ylim(0, 30)
    plt.xlim(0, 1500)
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 1.1), shadow=True, ncol=1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    title = f"Queens for SA"
    plt.title(title, fontsize=14, y=1.03)
    plt.savefig('./graphs/' + 'Queens_SA' + '_' + str(timestr) + '.png')
    plt.close()


def graph_RHC():
    df = pd.read_csv("./output/TSP_Maximize_RHC/RHC__TSP_Maximize_RHC__run_stats_df.csv")
    df_temp_25 = df.loc[(df['Restarts'] == 25) & (df['current_restart'] == 25)].copy()
    df_temp_25 = df_temp_25[['Iteration', 'Fitness']]
    df_temp_25.rename(columns={'Fitness': 'Restarts 25'}, inplace=True)
    df_temp_75 = df.loc[(df['Restarts'] == 75) & (df['current_restart'] == 74)].copy()
    df_temp_75 = df_temp_75[['Iteration', 'Fitness']]
    df_temp_75.rename(columns={'Fitness': 'Restarts 74'}, inplace=True)
    df_temp_100 = df.loc[(df['Restarts'] == 100) & (df['current_restart'] == 100)].copy()
    df_temp_100 = df_temp_100[['Iteration', 'Fitness']]
    df_temp_100.rename(columns={'Fitness': 'Restarts 100'}, inplace=True)

    data_frames = [df_temp_25, df_temp_75, df_temp_100]
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['Iteration'],
                                                    how='outer'), data_frames)
    ax = df_merged.plot(x='Iteration')

    NUM_COLORS = 3
    cm = plt.get_cmap('gist_rainbow')
    ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])

    # plt.ylim(0, 15)
    # plt.xlim(0, 50)
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 1.1), shadow=True, ncol=1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    title = f"Queens for RHC"
    plt.title(title, fontsize=14, y=1.03)
    plt.savefig('./graphs/' + 'Queens_RHC' + '_' + str(timestr) + '.png')
    plt.close()


def graph_timings():
    df_1 = pd.read_csv("./output/TSP_Maximize_GA/ga__TSP_Maximize_GA__run_stats_df.csv")
    df_2 = pd.read_csv("./output/TSP_Maximize_MIMIC/mimic__TSP_Maximize_MIMIC__run_stats_df.csv")
    df_3 = pd.read_csv("./output/TSP_Maximize_SA/sa__TSP_Maximize_SA__run_stats_df.csv")
    df_4 = pd.read_csv("./output/TSP_Maximize_RHC/RHC__TSP_Maximize_RHC__run_stats_df.csv")

    df_1_Best = df_1.loc[(df_1['Population Size'] == 300) & (df_1['Mutation Rate'] == 0.03)].copy()
    df_1_Best = df_1_Best[['Iteration', 'Time']]
    df_1_Best.rename(columns={'Time': 'GA'}, inplace=True)

    df_2_Best = df_2.loc[(df_2['Population Size'] == 300) & (df_2['Keep Percent'] == 0.2)].copy()
    df_2_Best = df_2_Best[['Iteration', 'Time']]
    df_2_Best.rename(columns={'Time': 'MIMIC'}, inplace=True)

    df_3_Best = df_3.loc[(df_3['Temperature'] == 10)].copy()
    df_3_Best = df_3_Best[['Iteration', 'Time']]
    df_3_Best.rename(columns={'Time': 'SA'}, inplace=True)

    df_4_Best = df_4.loc[(df_4['Restarts'] == 75) & (df_4['current_restart'] == 74)].copy()
    df_4_Best = df_4_Best[['Iteration', 'Time']]
    df_4_Best.rename(columns={'Time': 'RHC'}, inplace=True)

    data_frames = [df_1_Best, df_2_Best, df_3_Best, df_4_Best]
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['Iteration'],
                                                    how='outer'), data_frames)
    ax = df_merged.plot(x='Iteration')

    NUM_COLORS = 4
    cm = plt.get_cmap('gist_rainbow')
    ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])

    plt.ylim(0, 2000)
    plt.xlim(0, 2000)
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 1.1), shadow=True, ncol=1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Timings")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    title = f"Queens Timings for All 4 Algorithms"
    plt.title(title, fontsize=14, y=1.03)
    plt.savefig('./graphs/' + 'Queen_Timings' + '_' + str(timestr) + '.png')
    plt.close()


if __name__ == "__main__":
    main()
