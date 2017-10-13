__author__ = 'alexander_clegg'

import gym
import numpy as np
import time

import pickle
import joblib

import pyPhysX.pyutils as pyutils
import os

if __name__ == '__main__':
    filename = None
    filename2 = None

    prefix = os.path.dirname(os.path.abspath(__file__))
    prefix = os.path.join(prefix, '../../../rllab/data/local/experiment/')

    trial = None

    #trial = "experiment_2017_10_12_upright_gravity"
    #trial = "experiment_2017_10_12_displacer_gravity"
    trial = "experiment_2017_10_12_displacer_nogravity"
    #trial = "experiment_2017_10_12_upreacher_gravity"

    #trial = "experiment_2017_09_10_mode7_gripcover"
    #trial = "experiment_2017_09_06_lineargownclose"
    #trial = "experiment_2017_09_12_linearside_warmstart"
    #trial = "experiment_2017_09_11_mode7_nooraclebaseline"
    #trial = "experiment_2017_09_11_mode7_nohapticsbaseline"

    loadSave = False

    if loadSave is True:
        import tensorflow as tf
        if trial is not None:
            #load the params.pkl file and save a policy.pkl file
            with tf.Session() as sess:
                print("trying to load the params.pkl file")
                data = joblib.load(prefix+trial+"/params.pkl")
                print("loaded the pkl file")
                policy = data['policy']
                pickle.dump(policy, open(prefix+trial+"/policy.pkl", "wb"))
                print("saved the policy")
                exit()

    print("about to make")

    #construct env
    #env = gym.make('DartClothSphereTube-v1')
    #env = gym.make('DartReacher-v1')
    #env = gym.make('DartClothReacher-v2') #one arm reacher
    #env = gym.make('DartClothReacher-v3') #one arm reacher with target spline
    #env = gym.make('DartClothPoseReacher-v1')  #pose reacher
    #env = gym.make('DartClothSleeveReacher-v1')
    #env = gym.make('DartClothShirtReacher-v1')
    #env = gym.make('DartMultiAgent-v1')
    #env = gym.make('DartClothTestbed-v1')
    #env = gym.make('DartClothGrippedTshirt-v1') #no spline
    #env = gym.make('DartClothGrippedTshirt-v2') #1st arm
    #env = gym.make('DartClothGrippedTshirt-v3') #2nd arm
    env = gym.make('DartClothEndEffectorDisplacer-v1') #both arms

    policy = None
    if trial is not None and policy is None:
        policy = pickle.load(open(prefix+trial+"/policy.pkl", "rb"))

    print("about to run")
    paused = False
    time.sleep(0.5)
    for i in range(10000):
        o = env.reset()
        #envFilename = env.getFile()
        #print(envFilename)
        env.render()
        #time.sleep(0.5)
        rolloutHorizon = 40
        #rolloutHorizon = 100000
        if paused is True:
            rolloutHorizon = 10000
        for j in range(rolloutHorizon):
            a = np.zeros(22) #22 dof upper body
            #a = np.ones(22)
            #a += np.random.uniform(-1,1,22)
            if policy is not None:
                a, a_info = policy.get_action(o)
            done = False
            if not paused:
                s_info = env.step(a)
                o = s_info[0]
                done = s_info[2]
                #print(o)
            env.render()
            if done is True:
                time.sleep(0.5)
                break
            #if j == rolloutHorizon-1:
            #    print("Time terminate")
            #paused = True
    env.render(close=True)
    

