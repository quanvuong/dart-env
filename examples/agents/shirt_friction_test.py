__author__ = 'alexander_clegg'

import gym
import numpy as np
import time

import pickle

import pyPhysX.pyutils as pyutils

if __name__ == '__main__':

    if False:
        #graphing lines
        #default colors (nice mute palette)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        #lowDistance graph
        '''lineGrapher = pyutils.LineGrapher(title="Mean Lowest Target Distance")
        cutoff = [np.ones(20)*0.1, "success", 'blue']
        data = [
                [pyutils.loadList("lowestdistances_haptics_reacher.txt"), "reacher(h)", colors[0]],
                #[pyutils.loadList("lowestdistances_nohaptics_reacher.txt"), "reacher", colors[1]],
                #[pyutils.loadList("lowestdistances_haptics_shirt.txt"), "shirt(h)", colors[2]],
                #[pyutils.loadList("lowestdistances_nohaptics_shirt.txt"), "shirt", colors[3]],
                [pyutils.loadList("lowestdistances_haptics_noisereacher.txt"), "noise reacher(h)", colors[4]],
                #[pyutils.loadList("lowestdistances_nohaptics_noisereacher.txt"), "noise reacher", colors[5]],
                #[pyutils.loadList("lowestdistances_haptics_noiseshirt.txt"), "noise shirt(h)", colors[6]],
                #[pyutils.loadList("lowestdistances_nohaptics_noiseshirt.txt"), "noise shirt", colors[7]]
            ]'''

        #maxDeformation graph
        '''lineGrapher = pyutils.LineGrapher(title="Max Deformation")
        cutoff = [np.ones(20) * 15.0, "success", 'blue']
        data = [
            #[pyutils.loadList("maxDeformation_haptics_reacher.txt"), "reacher(h)", colors[0]],
            #[pyutils.loadList("maxDeformation_nohaptics_reacher.txt"), "reacher", colors[1]],
            [pyutils.loadList("maxDeformation_haptics_shirt.txt"), "shirt(h)", colors[2]],
            #[pyutils.loadList("maxDeformation_nohaptics_shirt.txt"), "shirt", colors[3]],
            #[pyutils.loadList("maxDeformation_haptics_noisereacher.txt"), "noise reacher(h)", colors[4]],
            #[pyutils.loadList("maxDeformation_nohaptics_noisereacher.txt"), "noise reacher", colors[5]],
            [pyutils.loadList("maxDeformation_haptics_noiseshirt.txt"), "noise shirt(h)", colors[6]],
            #[pyutils.loadList("maxDeformation_nohaptics_noiseshirt.txt"), "noise shirt", colors[7]]
        ]'''

        #completion time graph
        lineGrapher = pyutils.LineGrapher(title="Mean Runtime")
        cutoff = [np.ones(20) * 499.0, "success", 'blue']
        data = [
            #[pyutils.loadList("runtimes_haptics_reacher.txt"), "reacher(h)", colors[0]],
            [pyutils.loadList("runtimes_nohaptics_reacher.txt"), "reacher", colors[1]],
            #[pyutils.loadList("runtimes_haptics_shirt.txt"), "shirt(h)", colors[2]],
            #[pyutils.loadList("runtimes_nohaptics_shirt.txt"), "shirt", colors[3]],
            #[pyutils.loadList("runtimes_haptics_noisereacher.txt"), "noise reacher(h)", colors[4]],
            #[pyutils.loadList("runtimes_nohaptics_noisereacher.txt"), "noise reacher", colors[5]],
            #[pyutils.loadList("runtimes_haptics_noiseshirt.txt"), "noise shirt(h)", colors[6]],
            #[pyutils.loadList("runtimes_nohaptics_noiseshirt.txt"), "noise shirt", colors[7]]
        ]

        showStdDev = True

        #set the x axis
        xdata = np.arange(20)*0.05
        lineGrapher.xdata =xdata

        #plot the data
        lineGrapher.plotData(cutoff[0], cutoff[1], color=cutoff[2])
        for d in data:
            print("error = " + str(d[0][1]))
            if showStdDev:
                lineGrapher.plotData(d[0][0], d[1], color=d[2], error=d[0][1])
            else:
                lineGrapher.plotData(d[0][0], d[1], color=d[2])
        lineGrapher.update()

        #lineGrapher.save(filename="MaxDeformationShirt.png")
        #lineGrapher.save(filename="MeanLowestDistance.png")
        #lineGrapher.save(filename="MeanRuntime.png")

        #loop until interupt
        try:
            while(True):
                lineGrapher.update()
                time.sleep(0.1)
        except KeyboardInterrupt:
            exit()
    #done graphing

    policy = None

    #Upper Body reacher
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/experiment_2017_06_06_upperBodyReacher_arm2/policy.pkl"
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/experiment_2017_06_22_UpperBodyShirtArm2/params.pkl"
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/experiment_2017_06_26_UpperBodyReacher2ndArm/params.pkl"
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/experiment_2017_06_27_UpperBodyReacherShirt2ndArm/params.pkl"
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/experiment_2017_06_29_mirrorwarmstart/params.pkl"
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/experiment_2017_06_30_relaxedwarmstart/params11.pkl"
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/experiment_2017_07_13_hapticNoiseTest_reacher2/params24.pkl"
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/experiment_2017_07_14_posthapticNoise_Shirtreacher/params.pkl"

    prefix = "/home/alexander/Documents/dev/rllab/data/local/experiment/"
    trial = None
    #the following have all been converted to policy.pkl files
    #trial = "experiment_2017_07_14_posthapticNoise_Shirtreacher"
    #trial = "experiment_2017_07_13_hapticNoiseTest_reacher2"
    #trial = "experiment_2017_06_22_UpperBodyShirtArm2"
    #trial = "experiment_2017_06_06_upperBodyReacher_arm2"

    env = gym.make('DartClothShirtReacher-v1')

    if trial is not None and policy is None:
        policy = pickle.load(open(prefix+trial+"/policy.pkl", "rb"))

    print("about to run")
    paused = False
    #time.sleep(0.5)
    for i in range(200):
        o = env.reset()
        env.render()
        time.sleep(0.5)
        for j in range(500):
            #a = np.array([0.,0.,0.,0.,0.])
            a = np.zeros(11) #22 dof upper body
            #a = np.ones(22)
            #a[0] = 1
            #a += np.random.uniform(-1,1,22)
            if policy is not None:
                a, a_info = policy.get_action(o)
            #a = np.array([-1,-0,-0,-0,-0.])
            #if j < 9999: #add this for voxel testing
            done = False
            if not paused:
                s_info = env.step(a)
                o = s_info[0]
            #print(o)
                done = s_info[2]
            env.render()
            if done is True:
                time.sleep(0.5)
                break
            

    env.render(close=True)
    

