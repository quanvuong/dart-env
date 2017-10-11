__author__ = 'alexander_clegg'
import numpy as np

import pyPhysX.pyutils as pyutils

if __name__ == '__main__':
    inprefix = "/home/alexander/Downloads/graphdata/"
    outprefix = "/home/alexander/Downloads/graphdata/out/"
    output_postfix = "out"


    trial = "close linear"
    output_postfix = "_closelinear"

    #trial = "mode 7_54"
    #output_postfix = "_mode7"

    #trial = "haptic unaware baseline"
    #output_postfix = "_nohaptics"

    #trial = "no oracle baseline"
    #output_postfix = "_nooracle"

    #trial = "side linear"
    #output_postfix = "_sidelinear"

    progress_filename = inprefix +trial+"/armprogress.txt"
    deformation_filename = inprefix +trial+"/deformation.txt"

    print("loading progress data")
    progress_data = pyutils.loadData2D(filename=progress_filename)
    print("loading deformation data")
    deformation_data = pyutils.loadData2D(filename=deformation_filename)

    print("sorting data")
    progressIXs = pyutils.getListsIXByLastElement(progress_data)

    progressGraph = pyutils.LineGrapher(title="Arm Progress", ylims=(-1.0,1.0))
    deformationGraph = pyutils.LineGrapher(title="Max Deformation", ylims=(0,50))

    print("regraphing data")
    progressGraph.xdata = np.arange(len(progress_data[0]))
    deformationGraph.xdata = np.arange(len(progress_data[0]))
    for i,ix in enumerate(progressIXs):
        print(" " + str(i))
        color = pyutils.heatmapColor(minimum=0, maximum=1.0, value=float(i)/(len(progressIXs)-1.0))
        progressGraph.plotData(ydata=progress_data[ix], color=color)
        deformationGraph.plotData(ydata=deformation_data[ix], color=color)

    print("saving data")
    progressGraph.save(filename=outprefix+"/armprogress"+output_postfix+".png")
    deformationGraph.save(filename=outprefix+ "/deformation"+output_postfix+".png")
