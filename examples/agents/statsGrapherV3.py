__author__ = 'alexander_clegg'
import numpy as np
import os
import math

import pyPhysX.pyutils as pyutils

if __name__ == '__main__':
    prefix = "/home/alexander/Documents/dev/"
    folder = ""

    inprefix = prefix + folder
    outprefix = prefix + folder + "assistive_out/"

    print("loading assistive data")
    progress_filenames = ["limbProgressGraphData", "limbProgressGraphData_assistive"]
    poseError_filenames = ["restPoseErrorGraphData", "restPoseErrorGraphData_assistive"]
    labels = ["Linear", "RL Policy"]

    progress_data = []
    restPose_data = []

    for filename in progress_filenames:
        progress_data.append(pyutils.loadData2D(filename=inprefix+filename))

    for filename in poseError_filenames:
        restPose_data.append(pyutils.loadData2D(filename=inprefix+filename))

    #average over each timestep
    progress_avgs = []
    poseError_avgs = []

    print(len(progress_data[0][0]))

    for k in range(len(progress_data)):
        progress_avgs.append([])
        poseError_avgs.append([])
        #compute averages of these lists
        for i in range(len(progress_data[0][0])):
            progress_avgs[-1].append(0)
            poseError_avgs[-1].append(0)
            for j in range(len(progress_data[0])):
                progress_avgs[-1][-1] += progress_data[k][j][i]
                poseError_avgs[-1][-1] += restPose_data[k][j][i]
            progress_avgs[-1][-1] /= len(progress_data[0])
            poseError_avgs[-1][-1] /= len(restPose_data[0])

    #re-graph the averages
    progress_avg_Graph = pyutils.LineGrapher(title="Limb Progress", legend=True)
    poseError_avg_Graph = pyutils.LineGrapher(title="Rest Pose Error", legend=True)

    progress_avg_Graph.xdata = np.arange(len(progress_data[0][0]))
    poseError_avg_Graph.xdata = np.arange(len(restPose_data[0][0]))

    #print(reward_avgs)

    for i in range(len(progress_avgs)):
        progress_avg_Graph.plotData(ydata=progress_avgs[i],label=labels[i])
        poseError_avg_Graph.plotData(ydata=poseError_avgs[i], label=labels[i])

    progress_avg_Graph.save(filename=outprefix+"progress_avg_Graph")
    poseError_avg_Graph.save(filename=outprefix+"poseError_avg_Graph")

    #print(len(reward_data))
    #print(reward_data)