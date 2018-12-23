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

    trial = None

    print("about to make")

    envName = 'DartFlockAgent-v1'
    env = gym.make(envName)

    useMeanPolicy = True

    #print("policy time")
    policy = None
    if trial is not None and policy is None:
        policy = pickle.load(open(prefix+trial+"/policy.pkl", "rb"))
        print(policy)
        useMeanPolicy = True #always use mean if we loaded the policy

    #initialize an empty test policy
    if True and policy is None:
        env2 = normalize(GymEnv(envName, record_log=False, record_video=False))
        #env2 = normalize(GymEnv('DartSawyerRigidAssist-v1', record_log=False, record_video=False))
        policy = GaussianMLPPolicy(
            env_spec=env2.spec,
            # The neural network policy should have two hidden layers, each with 32 hidden units.
            hidden_sizes=(64, 64),
            #hidden_sizes=(128, 64),
            #init_std=0.2 #exploration scaling
            #init_std=0.15 #exploration scaling #human
            init_std=0.1 #robot
        )
        all_param_values = L.get_all_param_values(policy._mean_network.output_layer)
        #output bias scaling
        #all_param_values[4] *= 0.01 #human
        all_param_values[4] *= 0.002 #robot
        L.set_all_param_values(policy._mean_network.output_layer, all_param_values)
        env2._wrapped_env.env._render(close=True)
        pickle.dump(policy, open(prefix+"flockingdefault/policy.pkl", "wb"))
        print("saved the policy")
        useMeanPolicy = False #don't use the mean when we want to test a fresh policy initialization

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
        rolloutHorizon = 600
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

            #print(a)
            if policy is not None:
                action, a_info = policy.get_action(o)
                #print(a_info['mean'])
                a = action
                if useMeanPolicy:
                    a = a_info['mean']
                as_ub = np.ones(env.action_space.shape)
                action_space = spaces.Box(-1 * as_ub, as_ub)
                lb, ub = action_space.bounds
                scaled_action = lb + (a + 1.) * 0.5 * (ub - lb)
                scaled_action = np.clip(scaled_action, lb, ub)
                a=scaled_action
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
                #print("episode reward = " + str(env.rewardsData.cumulativeReward))
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
    

