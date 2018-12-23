__author__ = 'yuwenhao'

import gym
import numpy as np
import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        env = gym.make(sys.argv[1])
    else:
        env = gym.make('DartFlockAgent-v1')

    #env.env.disableViewer = False
    useMeanPolicy = True

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
        useMeanPolicy = False

    env.reset()

    for i in range(1000):
        env.step(env.action_space.sample())
        env.render()

    env.render(close=True)