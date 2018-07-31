__author__ = 'yuwenhao'

import numpy as np
from gym import utils
from gym.envs.dart import dart_env
import joblib
import os

import time

import pydart2 as pydart


class DartHumanBalanceEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0] * 23, [-1.0] * 23])
        self.action_scale = 200#np.array([60.0, 200, 60, 100, 80, 60, 60, 200, 60, 100, 80, 60, 150, 150, 100, 15,80,15, 30, 15,80,15, 30])
        obs_dim = 57

        dart_env.DartEnv.__init__(self, 'kima/kima_human_edited.skel', 15, obs_dim, self.control_bounds,
                                      disableViewer=True, dt=0.002)

        self.push_direction = np.array([1, 0, 0])
        self.push_strength = 100.0
        self.push_target = 'thorax'

        self.robot_skeleton.set_self_collision_check(True)

        for i in range(0, len(self.dart_world.skeletons[0].bodynodes)):
            self.dart_world.skeletons[0].bodynodes[i].set_friction_coeff(1)
        for i in range(0, len(self.dart_world.skeletons[1].bodynodes)):
            self.dart_world.skeletons[1].bodynodes[i].set_friction_coeff(1)

        # self.dart_world.set_collision_detector(3)

        for bn in self.robot_skeleton.bodynodes:
            if len(bn.shapenodes) > 0:
                if hasattr(bn.shapenodes[0].shape, 'size'):
                    shapesize = bn.shapenodes[0].shape.size()
                    print('density of ', bn.name, ' is ', bn.mass()/np.prod(shapesize))
                if hasattr(bn.shapenodes[0].shape, 'radius') and hasattr(bn.shapenodes[0].shape, 'height'):
                    radius = bn.shapenodes[0].shape.radius()
                    height = bn.shapenodes[0].shape.height()
                    print('density of ', bn.name, ' is ', bn.mass()/(height * np.pi * radius ** 2))
        print('Total mass: ', self.robot_skeleton.mass())

        utils.EzPickle.__init__(self)

    def do_simulation(self, tau, n_frames):
        for _ in range(n_frames):
            if self.cur_step < 30:
                self.robot_skeleton.bodynode(self.push_target).add_ext_force(self.push_direction * self.push_strength)
            self.robot_skeleton.set_forces(tau)
            self.dart_world.step()

    def advance(self, clamped_control):
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[6:] = clamped_control * self.action_scale

        self.do_simulation(tau, self.frame_skip)

    def _step(self, a):
        self.advance(np.clip(a, -1, 1))

        height = self.robot_skeleton.bodynode('head').com()[1]

        alive_bonus = 3.0

        reward = alive_bonus - np.square(a).sum() * 0.1

        self.cur_step += 1

        s = self.state_vector()


        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height - self.init_height > -0.35) and
                    np.abs(self.robot_skeleton.q[5]) < 0.7 and np.abs(self.robot_skeleton.q[4]) < 0.7 and
                    np.abs(self.robot_skeleton.q[3]) < 0.7)

        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self):
        state = np.concatenate([
            self.robot_skeleton.q[1:],
            self.robot_skeleton.dq,
        ])

        return state

    def reset_model(self):
        self.dart_world.reset()
        self.total_work = 0

        init_q = self.robot_skeleton.q
        init_dq = self.robot_skeleton.dq

        qpos = init_q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = init_dq + self.np_random.uniform(low=-.05, high=.05, size=self.robot_skeleton.ndofs)

        self.set_state(qpos, qvel)
        self.cur_step = 0

        self.init_height = self.robot_skeleton.bodynode('head').com()[1]

        self.push_direction = np.random.uniform(-1, 1, 3)
        self.push_direction[1] = 0
        self.push_direction[0] = np.abs(self.push_direction[0])
        self.push_direction /= np.linalg.norm(self.push_direction)
        self.push_strength = (np.random.random() + 1.0) * 100

        return self._get_obs()

    def viewer_setup(self):
        if not self.disableViewer:
            # self.track_skeleton_id = 0
            self._get_viewer().scene.tb.trans[2] = -5.5
