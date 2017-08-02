__author__ = 'alexander_clegg'

import gym
import numpy as np
import time

import pickle
import joblib
import tensorflow as tf

import pyPhysX.pyutils as pyutils

if __name__ == '__main__':
    #load policy
    
    filename = None
    filename2 = None
    
    #standard reacher: moved cloth 
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/reacher_sphere_movedcloth_2017_04_28_10_02_27_0001/params.pkl"
    
    #standard reacher: no simulated cloth
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/reacher_sphere_noclothsim_2017_04_29_13_31_56_0001/params.pkl"
    
    #self terminating (1st try w/ small penalty for deformation)
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/reacher_selfterminating_2017_04_26_21_00_09_0001/params.pkl"
    
    #cloth trial 1?
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/reacher_cloth1st(slow)_2017_04_27_11_06_16_0001/params.pkl"
    
    #cloth trial alternate reward
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/reacher_cloth_alternatereward_2017_04_28_00_12_56_0001/params.pkl"
    
    #hemisphere no cloth test
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/experiment_2017_05_01_noclothhemisphere_statereward/params.pkl"
    
    #sphere samples with cloth test (-2000 penalty)
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/reacher_sphere_cloth(-2000 penalty)_2017_04_30_12_15_00_0001/params.pkl"
    
    #Cloth reacher new reward
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/experiment_2017_05_03_clothreacher_disprewardtuned/params.pkl"
    
    #warm start
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/experiment_2017_05_04_clothreacher_warmstart_hemispherestatereward/params.pkl"
    
    #vanilla reacher new reward
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/experiment_2017_05_02_reacher_displacementreward_nocloth/params.pkl"
    
    #warm start stabilizer trial:
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/experiment_2017_05_04_reacher_warmstart_sphere_state2stable_prox/params.pkl"
    
    #warm start moving cloth trial 1 (~600 iter)
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/experiment_2017_05_04_reacher_warmstart_stablesphere2movingcloth/params.pkl"
    
    #warm start moving cloth trial 2 (1500 iter)
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/experiment_2017_05_08_reacher_warmstart_stablesphere2movingcloth3/params.pkl"
    
    #sleeve reacher experiment 1
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/experiment_2017_05_10_sleevereacher1/params.pkl"
    
    #new arm reacher experiment 2
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/experiment_2017_05_12_newArmReacher2/params.pkl"
    
    #stable reacher (new arm)
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/experiment_2017_05_14_newArmReacher_stable_warmstart2/params.pkl"
    
    #new arm sleeve reacher
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/experiment_2017_05_14_newArmSleeveReacher_stablewarmstart/params.pkl"
    
    #shirt reacher
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/experiment_2017_05_15_shirtReacher_stablewarmstart/params.pkl"
    
    #9 dof arm reacher
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/experiment_2017_05_16_9dofReacher/params.pkl"
    
    #9 dof arm stable reacher
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/experiment_2017_05_16_9dofReacher_stable/params.pkl"
    
    #Upper Body reacher
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/experiment_2017_06_06_upperBodyReacher_arm2/policy.pkl"
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/experiment_2017_06_22_UpperBodyShirtArm2/params.pkl"
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/experiment_2017_06_26_UpperBodyReacher2ndArm/params.pkl"
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/experiment_2017_06_27_UpperBodyReacherShirt2ndArm/params.pkl"
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/experiment_2017_06_29_mirrorwarmstart/params.pkl"
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/experiment_2017_06_30_relaxedwarmstart/params11.pkl"
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/experiment_2017_07_13_hapticNoiseTest_reacher2/params24.pkl"
    #filename = "/home/alexander/Documents/dev/rllab/data/local/experiment/experiment_2017_07_14_posthapticNoise_Shirtreacher/params.pkl"
    #filename = "/home/aclegg3/Documents/dev/rllab/data/local/experiment/experiment_2017_07_19_posereacher/params.pkl"
    #filename = "/home/aclegg3/Documents/dev/rllab/data/local/experiment/experiment_2017_07_23_posereacher4/params.pkl"
    #filename = "/home/aclegg3/Documents/dev/rllab/data/local/experiment/experiment_2017_07_24_posereacher5_q/params.pkl"
    #filename = "/home/aclegg3/Documents/dev/rllab/data/local/experiment/experiment_2017_07_25_posereacher6_q_normerror_prox/params.pkl"
    #filename = "/home/aclegg3/Documents/dev/rllab/data/local/experiment/experiment_2017_07_27_posereacher7_q_normerror_prox_notau/params.pkl"
    #filename = "/home/aclegg3/Documents/dev/rllab/data/local/experiment/experiment_2017_07_31_posereacher8_q_normerror_prox_notau_nohaptics/params.pkl"
    filename = "/home/aclegg3/Documents/dev/rllab/data/local/experiment/experiment_2017_08_01_posereacher8_q_normerror_prox_notau_nohaptics_cont1/params.pkl"
    #filename2 = "/home/alexander/Documents/dev/rllab/data/local/experiment/experiment_2017_07_13_hapticNoiseTest_reacher2/params.pkl"
    #filename2 = "/home/alexander/Documents/dev/rllab/data/local/experiment/experiment_2017_07_14_posthapticNoise_Shirtreacher/params.pkl"

    policy = None
    policy2 = None

    
    if filename is not None:
        with tf.Session() as sess:
            data = joblib.load(filename)
            policy = data['policy']
            #loadenv = data['env']
        print(policy)
    
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
    #load from AWS trial policy.pkl
    #policy = pickle.load( open(filename, "rb") )

    print("about to make")

    #construct env
    #env = gym.make('DartClothSphereTube-v1')
    #env = gym.make('DartReacher-v1')
    #env = gym.make('DartClothReacher-v2') #one arm reacher
    env = gym.make('DartClothPoseReacher-v1')  #pose reacher
    #env = gym.make('DartClothSleeveReacher-v1')
    #env = gym.make('DartClothShirtReacher-v1')
    #env = gym.make('DartClothGownDemo-v1')
    #env = gym.make('DartClothTestbed-v1')
    #env.render()
    #time.sleep(4)
    #print("done init")
    #env.reset()
    #env.render()
    #time.sleep(1)
    #Cloth sphere testing

    #policy = pickle.load(open(filename, "rb"))

    '''
    #save the policy in JSON
    print("Saving policy")
    pyutils.GaussianMLPPolicy_toJSON(policy=policy, label="test save", description="This is the description text", filename="JSONtest")

    #load the policy from JSON
    print("Loading Policy")
    pyutils.GaussianMLPPolicy_fromJSON(filename="JSONtest")
    #TODO
    exit()
    '''

    '''
    ppos = np.array([0.1,0,0])
    for i in range(1000):
        #print("ppos = " + str(ppos))
        a = ppos/(-np.linalg.norm(ppos))
        #print("a = " + str(a))
        ppos = env.step(a)[0][:3]
        #print(env.step([0,0.98,0.1]))
        env.render()
    '''
    '''        
    for i in range(1000):
        env.reset()
        env.render()
        time.sleep(0.5)
        ppos = np.array([0.1,0,0])
        for j in range(100):
            a = ppos/(-np.linalg.norm(ppos))
            ppos = env.step(a)[0][:3]
            env.render()
            time.sleep(0.1)
        #time.sleep(0.5)
    '''
    print("about to run")
    paused = False
    time.sleep(0.5)
    for i in range(10000):
        #print("about to reset")
        o = env.reset()
        #print("done reset")
        #print("v: " + str(o))
        env.render()
        time.sleep(0.5)
        #time.sleep(0.5)
        for j in range(500):
            #a = np.array([0.,0.,0.,0.,0.])
            #a = np.zeros(11) #22 dof upper body
            a = np.ones(22)
            #a[0] = 1
            a += np.random.uniform(-1,1,22)
            '''if(i < 22):
                a[i] += 1
            elif(i<44):
                a[i-22] -= 1'''
            #a = np.array([0.0,0.0,0.0,0.0,-0.1,0.1,0.1,0.0,0.0]) #9 dofs for new arm
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
            #print("o = " + str(o))
            #time.sleep(0.1)
            #if i > 3400:
            #if j > 9999 or j < 100: #add this for voxel testing
            env.render()
            #if (j % 100) == 0:
            #    print(j)
            if done is True:
                time.sleep(0.5)
                break
            #exit()
        #print("complete")
        #time.sleep(5)
            

    env.render(close=True)
    

