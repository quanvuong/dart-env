__author__ = 'alexander_clegg'

import gym
import numpy as np
import time

import pickle
import joblib

import pyPhysX.pyutils as pyutils
import os

if __name__ == '__main__':

    prefix = os.path.dirname(os.path.abspath(__file__))
    prefix = os.path.join(prefix, '../../../rllab/data/local/experiment/')

    env = gym.make('DartClothUpperBodyDataDrivenTshirtMaster-v1') #master Tshirt dressing controller

    for i in range(50):
        o = env.reset()
        env.render()

        rolloutHorizon = 20000
        startTime = time.time()
        for j in range(rolloutHorizon):
            a = np.zeros(22) #placeholder action since master env contains its own policy

            done = False
            s_info = env.step(a)
            o = s_info[0]
            done = s_info[2]
            env.render()
            if done is True:
                print("killed at step " + str(j))
                break
    env.render(close=True)
    

