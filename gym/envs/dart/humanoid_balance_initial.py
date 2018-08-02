import numpy as np
from gym import utils
from gym.envs.dart import dart_env

class DartHumanoidBalanceEnv(dart_env.DartEnv, utils.EzPickle):
    # initialization
    def __init__(self):
        pass

    # take a step forward in time by executing action
    def step(self, action):
        pass

    # reset the rollout
    def reset_model(self):
        pass