__author__ = 'alexander_clegg'
import numpy as np
import os
import math

import pyPhysX.pyutils as pyutils

if __name__ == '__main__':
    prefix = "/home/alexander/Documents/dev/"
    folder = "AblationStudy_tshirtR/"

    inprefix = prefix + folder
    outprefix = prefix + folder + "out/"

    print("loading reward data")
    reward_filenames = ["rewardGraphData_baseline", "rewardGraphData_haptics", "rewardGraphData_geo", "rewardGraphData_oracle"]
    cumulative_reward_filenames = ["cumulativeRewardGraphData_baseline", "cumulativeRewardGraphData_haptics", "cumulativeRewardGraphData_geo", "cumulativeRewardGraphData_oracle"]
    labels = ["Complete", "Haptics", "Geodesic", "Task"]

    reward_data = []
    cumulative_reward_data = []

    for filename in reward_filenames:
        reward_data.append(pyutils.loadData2D(filename=inprefix+filename))

    for filename in cumulative_reward_filenames:
        cumulative_reward_data.append(pyutils.loadData2D(filename=inprefix+filename))

    #average over each timestep
    reward_avgs = []
    cumulative_reward_avgs = []

    print(len(reward_data[0][0]))

    for k in range(len(reward_data)):
        reward_avgs.append([])
        cumulative_reward_avgs.append([])
        #compute averages of these lists
        for i in range(len(reward_data[0][0])):
            reward_avgs[-1].append(0)
            cumulative_reward_avgs[-1].append(0)
            for j in range(len(reward_data[0])):
                reward_avgs[-1][-1] += reward_data[k][j][i]
                cumulative_reward_avgs[-1][-1] += cumulative_reward_data[k][j][i]
            reward_avgs[-1][-1] /= len(reward_data[0])
            cumulative_reward_avgs[-1][-1] /= len(cumulative_reward_data[0])

    #re-graph the averages
    reward_avg_Graph = pyutils.LineGrapher(title="Ablation Policy Rewards",legend=True)
    cumulative_reward_avg_Graph = pyutils.LineGrapher(title="Cumulative Reward",legend=True)

    reward_avg_Graph.xdata = np.arange(len(reward_data[0][0]))
    cumulative_reward_avg_Graph.xdata = np.arange(len(reward_data[0][0]))

    print(reward_avgs)

    for i in range(len(reward_avgs)):
        reward_avg_Graph.plotData(ydata=reward_avgs[i],label=labels[i])
        cumulative_reward_avg_Graph.plotData(ydata=cumulative_reward_avgs[i], label=labels[i])

    reward_avg_Graph.save(filename=outprefix+"reward_avg_Graph")
    cumulative_reward_avg_Graph.save(filename=outprefix+"cumulative_reward_avg_Graph")

    #print(len(reward_data))
    #print(reward_data)