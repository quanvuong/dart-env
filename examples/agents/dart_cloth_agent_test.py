__author__ = 'alexander_clegg'

import gym
import numpy as np
import time

import pickle
import joblib
import tensorflow as tf

import pyPhysX.pyutils as pyutils
import os

if __name__ == '__main__':
    #load policy
    
    filename = None
    filename2 = None

    prefix = os.path.dirname(os.path.abspath(__file__))
    prefix = os.path.join(prefix, '../../../rllab/data/local/experiment/')

    trial = None
    #trial = "experiment_2017_09_11_mode7_nooraclebaseline"
    #trial = "experiment_2017_09_11_mode7_nohapticsbaseline"
    #trial = "experiment_2017_09_10_mode7_gripcover" #good (may be overtrained. FYI)
    #trial = "experiment_2017_09_09_mode7_torquerange" #meh
    #trial = "experiment_2017_09_09_largestatichighy_oracleonly" #bad
    #trial = "experiment_2017_09_08_closelinear_oracleonly" #bad
    #trial = "experiment_2017_09_08_gownfixed_mode7" #best w/ grip cover on.
    trial = "experiment_2017_09_07_gownlinearside"

    #trial = "experiment_2017_09_06_fixedgownwide"
    #trial = "experiment_2017_09_06_lineargownclose"
    #trial = "experiment_2017_09_06_linearfixedsmall_baseline" #oddly doesn't work

    #trial = "experiment_2017_09_05_gownlargerange_contactIDS"
    #trial = "experiment_2017_09_06_gownlargerange_geofix"
    #trial = "experiment_2017_09_06_gownlargerange_contactIDS_geovec"


    #trial = "experiment_2017_09_03_gownlinear_noterm_coll_largetorque"
    #trial = "experiment_2017_09_03_gownlargerange_noterm_coll_largetorque_gripcover"
    #trial = "experiment_2017_09_03_gownlargedist_noterm_coll_largetorque"

    #trial = "experiment_2017_09_02_gownsmalldist_noterm_coll_gripbox" #best?

    #trial = "experiment_2017_09_02_gownsmalldist_term_coll_gripbox"
    #trial = "experiment_2017_09_02_gowndistexpanded_noterm_bodycollision"
    #trial = "experiment_2017_09_01_gripped2nd_geoclamped"
    #trial = "experiment_2017_09_01_gowndistribution_noterm" #successful controller
    #trial = "experiment_2017_08_31_gripped_2ndarm_geospline_clamped"
    #trial = "experiment_2017_08_31_gripped1st_spline_notclamped_nogeo"
    #trial = "experiment_2017_08_31_gowndistribution_warmstart"
    #trial = "experiment_2017_08_30_fixedgown_geo_reinitialized"
    #trial = "experiment_2017_08_30_fixedgown_geo"
    #trial = "experiment_2017_08_30_gripped_1st_splineprog_clamped"
    #trial = "experiment_2017_08_29_assistivelinear_torqueclamp"
    #trial = "experiment_2017_08_29_clamped1stspline"
    #trial = "experiment_2017_08_29_gown_fixed_clamped"
    #trial = "experiment_2017_08_29_clamped1stspline"
    #trial = "experiment_2017_08_29_gripped1st_retrial_clamped"
    #trial = "experiment_2017_08_29_assistivelinear"
    #trial = "experiment_2017_08_28_gripped2ndarm_easy_contactReward"
    #trial = "experiment_2017_08_28_gripped1starm_phaseobs_retrial"

    loadSave = False

    if loadSave is True:
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

    policy = None
    policy2 = None

    #policy = pickle.load(open(filename, "rb"))

    if filename2 is not None:
        with tf.Session() as sess:
            data = joblib.load(filename2)
            policy2 = data['policy']
            #loadenv = data['env']
        print(policy2)
    #pfile = open(filename, 'r+')

    if policy is not None and False:
        print("")
        print("------------------------------------------------")
        print("Picking apart the policy now")
        print("policy: " + str(policy))
        print("policy._mean_network: " + str(policy._mean_network))
        print("policy._mean_network.layers: " + str(policy._mean_network.layers))
        numParams = 0
        for l in policy._mean_network.layers:
            print(" " + str(l))
            print(" " + str(l.get_params()))
            for i in l.get_params():
                print("     " + str(i))
                print("     " + str(i.get_value()))
                '''
                for p in i.get_value():
                    print("       p=" + str(p) + ": ")
                    for j in p:
                        numParams += 1
                        print("         j=" + str(j))
                '''
                '''for p in range(len(i.get_value())):
                    print("       p=" + str(p) + ": ")
                    for j in range(len(i.get_value()[p])):
                        numParams += 1
                        print("         j=" + str(j) + ": " + str(i.get_value()[p][j]))
                '''
        print("Counted " + str(numParams) + " total params.")
        print(pyutils.getPolicyParams(policy, layer=1, biases=True, weights=False))
        #pyutils.barGraph(list=pyutils.getPolicyParams(policy, layer=1, biases=False, weights=True),filename="barGraph1")
        print("")
        print("input weights for 1st input: ")
        #weights = pyutils.getPolicyInputWeights(policy, 72, 137)
        #weights2 = pyutils.getPolicyInputWeights(policy2, 72, 137)
        weights = pyutils.getPolicyInputWeights(policy, 0, 66)
        weights2 = pyutils.getPolicyInputWeights(policy2, 0, 66)
        #weights = pyutils.getPolicyInputWeights(policy, 0, 0)
        weightsdiff = pyutils.getDiffMagnitude(weights, weights2)
        print(weights)
        print("num weights = " + str(len(weights)))
        pyutils.barGraph(list=weightsdiff, filename="weights_diff_pose_reachers")
        pyutils.barGraph(list=weightsdiff, filename="weights_diff_pose_reachers")
        pyutils.barGraph(list=weightsdiff, filename="weights_diff_pose_reachers")
        '''
        layer2weights = policy._mean_network.layers[1].get_params()[0].get_value()

        print("layer2weights (" + str(len(layer2weights)) + ") = " + str(layer2weights))

        flattenedWeights = []
        for w in layer2weights:
            print(w)
            flattenedWeights.append(w)

        print("flattenedWeights (" + str(len(flattenedWeights)) + ") = " + str(flattenedWeights))
        pyutils.barGraph(list=flattenedWeights, filename="barGraph")
        '''
        print("------------------------------------------------")
        print("")
        exit()
        #pyutils.barGraph()

    #save the policy
    '''
    if policy is not None:
        pickle.dump(policy, open("/home/ubuntu/policy.pkl", "wb"))
        print("saved the policy")
        exit()
    '''

    print("about to make")

    #construct env
    #env = gym.make('DartClothSphereTube-v1')
    #env = gym.make('DartReacher-v1')
    #env = gym.make('DartClothReacher-v2') #one arm reacher
    #env = gym.make('DartClothReacher-v3') #one arm reacher with target spline
    #env = gym.make('DartClothPoseReacher-v1')  #pose reacher
    #env = gym.make('DartClothSleeveReacher-v1')
    #env = gym.make('DartClothShirtReacher-v1')
    env = gym.make('DartClothGownDemo-v1')
    #env = gym.make('DartClothTestbed-v1')
    #env = gym.make('DartClothGrippedTshirt-v1') #no spline
    #env = gym.make('DartClothGrippedTshirt-v2') #1st arm
    #env = gym.make('DartClothGrippedTshirt-v3') #2nd arm

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
        rolloutHorizon = 400
        #rolloutHorizon = 100000
        if paused is True:
            rolloutHorizon = 10000
        for j in range(rolloutHorizon):
            a = np.zeros(11) #22 dof upper body
            #a = np.ones(22)
            a += np.random.uniform(-1,1,11)
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
    env.render(close=True)
    

