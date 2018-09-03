import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class Walker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)

    def pre_advance(self):
        self.posbefore = self.sim.data.qpos[0]

    def advance(self, a):
        self.do_simulation(a, self.frame_skip)

    def reward_func(self, a):
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 3.0
        reward = ((posafter - self.posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        return reward

    def post_advance(self):
        pass

    def terminated(self):
        posafter, height, ang = self.sim.data.qpos[0:3]
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        return done

    def step(self, a):
        self.pre_advance()
        self.advance(a)
        reward = self.reward_func(a)

        # uncomment to enable knee joint limit penalty
        '''joint_limit_penalty = 0
        for j in [-2, -5]:
            if (self.model.jnt_range[j][0] - self.model.data.qpos[j]) > -0.05:
                joint_limit_penalty += abs(1.5)
            if (self.model.jnt_range[j][1] - self.model.data.qpos[j]) < 0.05:
                joint_limit_penalty += abs(1.5)
        reward -= 5e-1*joint_limit_penalty'''

        done = self.terminated()
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], qvel]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
