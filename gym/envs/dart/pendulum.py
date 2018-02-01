__author__ = 'yuwenhao'

import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from gym.envs.dart.parameter_managers import CartPoleManager
import os
import joblib
from gym.envs.dart.sub_tasks import *
import copy


class DartPendulumEnv(dart_env.DartEnv):
    def __init__(self):
        self.control_bounds = np.array([[1.0], [-1.0]])
        self.action_scale = np.array([2])

        self.use_disc_ref_policy = False
        self.disc_ref_weight = 0.001
        self.disc_funcs = [] # vfunc, obs_disc, act_disc, state_filter, state_unfilter

        self.cur_step = 0

        obs_dim = 2

        dart_env.DartEnv.__init__(self, ['pendulum.skel'], 2, obs_dim, self.control_bounds, dt=0.01,
                                  disableViewer=True)

        self.dart_world = self.dart_worlds[0]
        self.robot_skeleton = self.dart_world.skeletons[-1]

        utils.EzPickle.__init__(self)

    def _step(self, a):
        clamped_control = np.copy(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]

        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[0] = clamped_control[0] * self.action_scale[0]

        state_act = np.concatenate([self.state_vector(), tau / 40.0])
        state_pre = np.copy(self.state_vector())

        self.do_simulation(tau, self.frame_skip)

        ob = self._get_obs()

        ang = self.robot_skeleton.q[0]

        ang_proc = (np.abs(ang) % (2 * np.pi))
        ang_proc = np.min([ang_proc, (2 * np.pi) - ang_proc])

        alive_bonus = 1.0
        ang_cost = 1.0 * (ang_proc ** 2)
        quad_ctrl_cost = 0.01 * np.square(a).sum()

        reward = alive_bonus - ang_cost - quad_ctrl_cost
        if ang_proc < 0.5:
            reward += np.max([5 - (ang_proc) * 4, 0]) + np.max([3 - np.abs(self.robot_skeleton.dq[0]), 0])

        done = abs(self.robot_skeleton.dq[0]) > 35

        self.cur_step += 1

        envinfo = {}

        if self.use_disc_ref_policy:
            if self.disc_funcs[1](self.disc_funcs[3](self.state_vector())) in self.disc_funcs[0]:
                ref_reward = self.disc_ref_weight * self.disc_funcs[0][self.disc_funcs[1](self.disc_funcs[3](self.state_vector()))]
                envinfo['sub_disc_ref_reward'] = ref_reward

        return ob, reward, done, envinfo

    def _get_obs(self):
        state = np.concatenate([self.robot_skeleton.q, self.robot_skeleton.dq]).ravel()

        ang = self.robot_skeleton.q[0]
        ang_proc = (ang % (2 * np.pi))
        if ang_proc > np.pi:
            ang_proc -= 2 * np.pi

        state[0] = ang_proc

        return state

    def reset_model(self):
        self.total_dist = []
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.1, high=.1, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        if self.np_random.uniform(0, 1) > 0.5:
            qpos[0] += np.pi
        else:
            qpos[0] += -np.pi

        self.set_state(qpos, qvel)

        self.cur_step = 0

        return self._get_obs()

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -4.5
        self._get_viewer().scene.tb._set_theta(0)
        self.track_skeleton_id = 0
