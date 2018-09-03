import numpy as np
from gym import utils
from gym.envs.dart import dart_env


class DartHumanoidBalanceEnv(dart_env.DartEnv, utils.EzPickle):
    # initialization
    def __init__(self):
        # define action space, humanoid has 23 actuated dofs
        control_bounds = np.array([[1.0] * 23, [-1.0] * 23])

        # set observation dimension, humanoid has (23+6)*2
        observation_dim = 56

        # initialize dart-env
        dart_env.DartEnv.__init__(self, 'kima/kima_human_balance.skel', frame_skip=4, observation_size=observation_dim,
                                  action_bounds=control_bounds)

        # enable self-collision check
        self.robot_skeleton.set_self_collision_check(True)

        utils.EzPickle.__init__(self)

    # take a step forward in time by executing action
    def step(self, action):
        # advance the simulation
        tau = np.zeros(29)
        tau[6:] = np.clip(action, -1, 1)*200
        self.do_simulation(tau, self.frame_skip)

        # calculate reward
        reward = 3.0 - np.square(action).sum()

        # termination criteria
        s = self.state_vector()
        height = self.robot_skeleton.bodynode('head').com()[1]
        done = not (np.isfinite(s).all() and (np.abs(s) < 100).all() and
                    (height - self.init_height > -0.35) and
                    np.abs(self.robot_skeleton.q[5]) < 1.0 and np.abs(self.robot_skeleton.q[4]) < 1.0 and
                    np.abs(self.robot_skeleton.q[3]) < 1.0)

        # get next observation
        observation = np.zeros(56)
        observation[0] = self.state_vector()[1]
        observation[1:] = self.state_vector()[3:]

        return observation, reward, done, {}

    # reset the rollout
    def reset_model(self):
        self.dart_world.reset()

        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.05, high=.05, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        self.init_height = self.robot_skeleton.bodynode('head').com()[1]

        observation = np.zeros(56)
        observation[0] = self.state_vector()[1]
        observation[1:] = self.state_vector()[3:]

        return observation

    def viewer_setup(self):
        if not self.disableViewer:
            self._get_viewer().scene.tb.trans[2] = -3.5




