__author__ = 'alexander_clegg'
import numpy as np
import os
import math

import pyPhysX.pyutils as pyutils

if __name__ == '__main__':

    graphMinDeformation = False

    #inprefix = "/home/alexander/Downloads/graphdata/"
    #outprefix = "/home/alexander/Downloads/graphdata/out/"
    prefix = "/home/alexander/Documents/"
    #folder = "graphoutput_mode7/"
    #folder = "graphoutput_nohaptics/"
    #folder = "graphoutput_notask/"
    #folder = "graphoutput_mode5/"
    folder = "graphoutput_mode6/"
    output_postfix = ""

    inprefix = prefix+folder
    outprefix = prefix+folder+"out/"

    progress_filename = inprefix + "armprogress.txt"
    deformation_filename = inprefix + "deformation.txt"
    haptics_indices = [13,14,15,16,17,18,19,20,21]
    haptics_filenames = []
    for ix in haptics_indices:
        haptics_filenames.append(inprefix+"haptics"+str(ix)+".txt")

    print("loading progress data")
    progress_data = pyutils.loadData2D(filename=progress_filename)
    print("loading deformation data")
    deformation_data = pyutils.loadData2D(filename=deformation_filename)
    print("loading haptics data")
    haptics_data = []
    for i,ix in enumerate(haptics_indices):
        haptics_data.append(pyutils.loadData2D(filename=haptics_filenames[i]))

    print("getting mean and std-dev")
    #get data means and std deviation
    progress_mean = pyutils.computListsMean(lists=progress_data)
    progress_stdDev = pyutils.computeListsStdDev(lists=progress_data, mean=progress_mean)
    deformation_mean = pyutils.computListsMean(deformation_data)
    deformation_stdDev = pyutils.computeListsStdDev(lists=deformation_data, mean=deformation_mean)
    haptics_means = []
    haptics_stdDevs = []
    for hd in haptics_data:
        haptics_means.append(pyutils.computListsMean(lists=hd))
        haptics_stdDevs.append(pyutils.computeListsStdDev(lists=hd, mean=haptics_means[-1]))

    #print relevant scalars
    defSum = 0
    for d in deformation_mean:
        defSum += d
    meanDef = defSum/len(deformation_mean)
    defSum = 0
    for d in deformation_mean:
        defSum += (d-meanDef)**2
    stdDevDef = math.sqrt(defSum/len(deformation_mean))
    print("Mean Deformation: " + str(meanDef))
    print("StdDev Deformation: " + str(stdDevDef))

    progSum = 0
    for d in progress_mean:
        progSum += d
    meanProg = progSum / len(progress_mean)
    progSum = 0
    for d in progress_mean:
        progSum += (d-meanProg)**2
    stdDevProg = math.sqrt(progSum / len(progress_mean))
    print("Mean Progress: " + str(meanProg))
    print("StdDev Progress: " + str(stdDevProg))
    progSum = 0
    for rollout in progress_data:
        progSum += (rollout[-1]-progress_mean[-1])**2
    stdDevEndProg = math.sqrt(progSum / len(progress_data))
    print("Mean End Progress: " + str(progress_mean[-1]))
    print("StdDev End Progress: " + str(stdDevEndProg))

    hapticSum = 0
    instances = 0
    for hm in haptics_means:
        for val in hm:
            hapticSum += val
            instances += 1.0
    meanHaptics = hapticSum / instances
    print("Mean Haptic Perception Magnitude: " + str(meanHaptics))

    #print("sorting data")
    #progressIXs = pyutils.getListsIXByLastElement(progress_data)
    print("creating graphs")
    progressGraph = pyutils.LineGrapher(title="Arm Progress Mean", ylims=(-1.0,1.0))
    deformationGraph = pyutils.LineGrapher(title="Max Deformation Mean", ylims=(0,50))
    cumulativeHapticsGraph = pyutils.LineGrapher(title="All Haptic Means", ylims=(0,1.), legend=True)
    hapticsGraphs = []
    for ix in haptics_indices:
        hapticsGraphs.append(pyutils.LineGrapher(title="Haptic Sensor " + str(ix) + " Mean", ylims=(0,2.5)))

    print("regraphing data")
    progressGraph.xdata = np.arange(len(progress_data[0]))
    deformationGraph.xdata = np.arange(len(progress_data[0]))
    cumulativeHapticsGraph.xdata = np.arange(len(progress_data[0]))
    for g in hapticsGraphs:
        g.xdata = np.arange(len(progress_data[0]))

    color = np.zeros(3)#pyutils.heatmapColor(minimum=0, maximum=1.0, value=float(i)/(len(progressIXs)-1.0))
    progressGraph.plotData(ydata=progress_mean, error=progress_stdDev, color=color)
    deformationGraph.plotData(ydata=deformation_mean, error=deformation_stdDev, color=color)
    for ix,g in enumerate(hapticsGraphs):
        g.plotData(ydata=haptics_means[ix], error=haptics_stdDevs[ix], color=color)
        cumulativeHapticsGraph.plotData(ydata=haptics_means[ix], error=haptics_stdDevs[ix], label=str(haptics_indices[ix]))

    if graphMinDeformation:
        minDeformation = []
        for l in deformation_data:
            for ix,val in enumerate(l):
                if len(minDeformation) <= ix:
                    minDeformation.append(val)
                elif minDeformation[ix] > val:
                    minDeformation[ix] = val
        deformationGraph.plotData(ydata=minDeformation, color=np.array([0.,1.0,0.]))

    print("saving data")
    if not os.path.exists(outprefix):
        os.makedirs(outprefix)
    progressGraph.save(filename=outprefix+"armprogressmean"+output_postfix+".png")
    deformationGraph.save(filename=outprefix+ "deformationmean"+output_postfix+".png")
    cumulativeHapticsGraph.save(filename=outprefix+ "hapticmeans"+output_postfix+".png")
    for i,ix in enumerate(haptics_indices):
        hapticsGraphs[i].save(filename=outprefix+ "hapticsensor"+str(ix)+"mean"+output_postfix+".png")
