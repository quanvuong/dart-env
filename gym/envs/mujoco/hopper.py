import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
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
                    (height > .7) and (abs(ang) < .4))

        return done

    def pre_advance(self):
        self.posbefore = self.sim.data.qpos[0]

    def reward_func(self, a):
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - self.posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
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
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
