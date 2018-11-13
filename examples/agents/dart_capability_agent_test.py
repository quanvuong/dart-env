__author__ = 'alexander_clegg'

import gym
import numpy as np
import time

import pickle
import joblib

import pyPhysX.pyutils as pyutils
import os

from rllab import spaces
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
import lasagne.layers as L

if __name__ == '__main__':

    filename = None
    filename2 = None

    prefix = os.path.dirname(os.path.abspath(__file__))
    prefix = os.path.join(prefix, '../../../rllab/data/local/experiment/')

    #envName = 'DartSawyerRigid-v4'
    #envName = 'DartIiwaRigid-v1'
    #envName = 'DartIiwaGown-v1'
    #envName = 'DartIiwaGownAssist-v1'
    #envName = 'DartIiwaCapability-v1'
    envName = 'DartIiwaBezier-v1'

    env = gym.make(envName)

    reloaderTest = False

    if reloaderTest:
        print("reloader test")
        #env.close()
        trials = 0
        failures = 0
        failureRate = 0
        numReloaderSamples = 500
        for i in range(numReloaderSamples):
            trials += 1
            try:
                print("entering try")
                envc = gym.make('DartSawyerRigid-v4')
                #envc = gym.make('DartSawyer-v3')
                #envc = gym.make('DartClothUpperBodyDataDrivenDropGrip-v1')
                #print("reseting")
                #envc.reset()
                #print("stepping")
                #envc.step(action=np.zeros(envc.act_dim))
                #envc.close()
                print("exiting try")
            except:
                failures += 1
            print("----------------------------------------------")
            print("----------------------------------------------")
            print("Number of failures detected: " + str(failures))
            print("Failure rate detected: " + str(failures/trials))
            print("----------------------------------------------")
            print("----------------------------------------------")
        print("done reloader test:")
        print("Number of trials run: " + str(trials))
        print("Number of failures detected: " + str(failures))
        print("Failure rate detected: " + str(failures / trials))
        exit(0)

    useMeanPolicy = True

    print("about to run")
    paused = False
    #useMeanPolicy = True $set mean policy usage
    time.sleep(0.5)
    cumulativeFPS = 0
    completedRollouts = 0 #counts rollouts which were not terminated early
    successfulTrials = 0
    failureTrials = 0
    env.render()
    #time.sleep(30.0) #window setup time for recording
    #o = env.reset()

    for i in range(110):
        #print("here")
        o = env.reset()
        #envFilename = env.getFile()
        #print(envFilename)
        env.render()
        #time.sleep(0.5)
        rolloutHorizon = 10000
        rolloutHorizon = 60
        #rolloutHorizon = 10000
        if paused is True:
            rolloutHorizon = 10000
        startTime = time.time()
        #for j in range(rolloutHorizon):
        #start_pose = np.array(env.robot_skeleton.q[6:])
        while(env.numSteps < rolloutHorizon):
            #if j%(rolloutHorizon/10) == 0:
            #    print("------- Checkpoint: " + str(j/(rolloutHorizon/10)) + "/10 --------")
            a = np.zeros(env.act_dim) #22 dof upper body, ?? dof full body

            done = False
            if not paused or env.numSteps == 0:# or j==0:
                s_info = env.step(a)
                o = s_info[0]
                done = s_info[2]
                #print(s_info)
                #print(o)
            env.render()

            j = env.numSteps
            if done is True:
                print("killed at step " + str(j))
                cumulativeFPS += (j+1)/(time.time()-startTime)
                print("framerate = " + str((j+1) / (time.time() - startTime)))
                print("average FPS: " + str(cumulativeFPS / (i + 1)))
                print("episode reward = " + str(env.rewardsData.cumulativeReward))
                #if
                time.sleep(0.5)
                break
            if j == rolloutHorizon-1:
                #print("startTime = " + str(startTime))
                #print("endTime = " + str(time.time()))
                #print("totalTime = " + str(time.time()-startTime))
                cumulativeFPS += rolloutHorizon/(time.time()-startTime)
                print("framerate = " + str(rolloutHorizon/(time.time()-startTime)))
                print("average FPS: " + str(cumulativeFPS/(i+1)))
                print("total rollout time: " + str(time.time()-startTime))
            #    print("Time terminate")
            #paused = True
    env.render(close=True)
    

