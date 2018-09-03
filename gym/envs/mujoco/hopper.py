import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.dart.parameter_managers import *

class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.train_UP = False
        self.noisy_input = False

        self.resample_MP = False  # whether to resample the model paraeters
        self.param_manager = mjHopperManager(self)
        self.velrew_weight = 1.0

        mujoco_env.MujocoEnv.__init__(self, 'hopper.xml', 4)

        utils.EzPickle.__init__(self)

    def pad_action(self, a):
        full_ac = np.zeros(len(self.sim.data.qpos))
        full_ac[3:] = a
        return full_ac

    def unpad_action(self, a):
        return a[3:]

    def advance(self, a):
        self.do_simulation(a, self.frame_skip)

    def about_to_contact(self):
        return False

    def post_advance(self):
        pass

    def terminated(self):
        s = self.state_vector()
        height, ang = self.sim.data.qpos[1:3]
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .8))
        return done

    def pre_advance(self):
        self.posbefore = self.sim.data.qpos[0]

    def reward_func(self, a):
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - self.posbefore) / self.dt * self.velrew_weight
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        joint_limit_penalty = 0
        if np.abs(self.sim.data.qpos[-2]) < 0.05:
            joint_limit_penalty += 1.5
        #reward -= 5e-1 * joint_limit_penalty
        return reward

    def step(self, a):
        self.pre_advance()
        self.advance(a)
        self.post_advance()

        reward = self.reward_func(a)

        done = self.terminated()

        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        state = np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])
        if self.train_UP:
            state = np.concatenate([state, self.param_manager.get_simulator_parameters()])
        if self.noisy_input:
            state = state + np.random.normal(0, .01, len(state))
        return state

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)

        if self.resample_MP:
            self.param_manager.resample_parameters()

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
