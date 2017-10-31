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

    trial = "experiment_2017_10_26_tshirt_nograv_complete"
    #trial = "experiment_2017_10_26_tshirt_grav_complete"

    #trial = "experiment_2017_10_18_displacerR_mod_prevT"
    #trial = "experiment_2017_10_18_displacerR_mod"
    #trial = "experiment_2017_10_18_gravity_displacerR_prevTau"

    #trial = "experiment_2017_10_17_nogravity_reacherR"
    #trial = "experiment_2017_10_17_gravity_reacherR"
    #trial = "experiment_2017_10_17_nogravity_displacerR"
    #trial = "experiment_2017_10_17_gravity_displacerR"

    #trial = "experiment_2017_10_13_displacer_gravity_noT_randpose"
    #trial = "experiment_2017_10_13_displacer_nogravity_noT"
    #trial = "experiment_2017_10_13_displacer_gravity_noT"

    #trial = "experiment_2017_10_12_upright_gravity"
    #trial = "experiment_2017_10_12_displacer_gravity"
    #trial = "experiment_2017_10_12_displacer_nogravity"
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
    #env = gym.make('DartClothEndEffectorDisplacer-v1') #both arms
    #env = gym.make('DartClothJointLimitsTest-v1')
    #env = gym.make('DartClothGownDemo-v1')
    #env = gym.make('DartClothUpperBodyDataDriven-v1')
    env = gym.make('DartClothUpperBodyDataDrivenTshirt-v1')

    policy = None
    if trial is not None and policy is None:
        policy = pickle.load(open(prefix+trial+"/policy.pkl", "rb"))

    print("about to run")
    paused = False
    time.sleep(0.5)
    cumulativeFPS = 0
    completedRollouts = 0 #counts rollouts which were not terminated early
    for i in range(10000):
        #print("here")
        o = env.reset()
        #envFilename = env.getFile()
        #print(envFilename)
        env.render()
        #time.sleep(0.5)
        rolloutHorizon = 400
        #rolloutHorizon = 10000
        if paused is True:
            rolloutHorizon = 10000
        startTime = time.time()
        for j in range(rolloutHorizon):
            #if j%(rolloutHorizon/10) == 0:
            #    print("------- Checkpoint: " + str(j/(rolloutHorizon/10)) + "/10 --------")
            a = np.zeros(22) #22 dof upper body

            #a = np.ones(22)
            a += np.random.uniform(-1,1,len(a))
            #a[:11] = np.zeros(11)
            #a += np.random.randint(3, size=len(a))-np.ones(len(a))
            '''
            if j==0:
                a = np.array([-1,-0.5935519015,-0.6243126472,-1,-0.3540411152,-0.8545956428,-0.1052807823,-0.6650868959,1,-1,-0.4370771514,-1,-0.2656309561,-0.7392283111,1,-0.4849024561,-0.4222881197,-1,-1,-0.1260703,1,0.3853144958,])
            elif j==1:
                a = np.array([1,0.08203909,-0.4428489711,0.3709899779,-0.1139084987,0.8878356518,-0.3833406323,0.9175109866,-1,-0.7288833698,-0.3778503588,-0.0617086992,-1,0.5471811498,1,-1,-0.4266441964,-1,0.2783551927,0.0862617301,0.5444295707,0.7144071905])
            elif j==2:
                a = np.array([-1,1,-0.6846910321,0.5774784709,-0.7145496691,-0.7416754164,-1,0.9724756555,-1,1,-1,-0.9628565439,-1,1,-0.9544127885,1,0.5642344238,-0.1455457015,-0.3926989475,-1,1,-0.0842431477])
            elif j==3:
                a = np.array([1,0.6312312283,-0.6876604936,0.5467897784,-0.9867554189,-1,-1,0.314068975,0.2136389088,-1,1,-1,-0.1857029911,0.933112181,-1,-0.9219502237,0.7421179613,1,-1,0.0583067668,1,-0.3022806922])
            '''
            #print(a)
            if policy is not None:
                a, a_info = policy.get_action(o)
            done = False
            if not paused or j==0:
                s_info = env.step(a)
                o = s_info[0]
                done = s_info[2]
                #print(o)
            env.render()
            if done is True:
                print("killed at step " + str(j))
                cumulativeFPS += (j+1)/(time.time()-startTime)
                print("framerate = " + str((j+1) / (time.time() - startTime)))
                print("average FPS: " + str(cumulativeFPS / (i + 1)))
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
    

