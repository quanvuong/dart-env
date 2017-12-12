import numpy as np
from gym import utils
from gym.envs.dart import dart_env

from gym.envs.dart.parameter_managers import *
from gym.envs.dart.sub_tasks import *
import copy

import joblib, os


class DartHopperRSSEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
        self.action_scale = np.array([200.0, 200.0, 200.0])
        self.train_UP = False
        self.noisy_input = False
        self.avg_div = 0

        self.resample_MP = True  # whether to resample the model paraeters
        self.train_mp_sel = False
        self.perturb_MP = False

        self.slippery_surface = False
        self.slippery_coef = 0.3
        obs_dim = 11
        self.param_manager = hopperContactMassManager(self)
        self.param_manager.activated_param = [0]
        self.param_manager.controllable_param = [0]
        self.param_manager.range = [0.3, 1.0]

        self.state_index = 0

        if self.train_UP:
            obs_dim += self.param_manager.param_dim

        if self.avg_div > 1:
            obs_dim += self.avg_div

        self.dyn_models = [None]
        self.dyn_model_id = 0
        self.base_path = None
        self.transition_locator = None
        self.baseline = None

        self.t = 0

        self.total_dist = []

        dart_env.DartEnv.__init__(self, ['hopper_rss.skel'], 4, obs_dim, self.control_bounds, disableViewer=True)

        self.param_manager.set_simulator_parameters([1.0])
        self.current_param = self.param_manager.get_simulator_parameters()

        self.dart_world = self.dart_worlds[0]
        self.robot_skeleton = self.dart_world.skeletons[-1]

        utils.EzPickle.__init__(self)

    def do_simulation(self, tau, n_frames):
        self.robot_skeleton.set_forces(tau)

        for _ in range(n_frames):
            self.dart_world.step()

    def advance(self, a):
        clamped_control = np.array(a)
        '''for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]'''
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[3:] = clamped_control * self.action_scale

        self.do_simulation(tau, self.frame_skip)

    def _step(self, a):
        if self.slippery_surface:
            if self.robot_skeleton.com()[0] > 20 and self.robot_skeleton.com()[0] < 30:
                self.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(self.slippery_coef)
            else:
                self.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(0.9)

        self.t += self.dt
        prev_obs = self._get_obs()
        pre_state = [self.state_vector()]
        if self.train_UP:
            pre_state.append(self.param_manager.get_simulator_parameters())
        posbefore = self.robot_skeleton.com()[0]
        state_act = np.concatenate([self.state_vector(), np.clip(a, -1, 1)])
        state_pre = np.copy(self.state_vector())
        self.advance(a)
        ang = self.robot_skeleton.q[2]
        posafter = self.robot_skeleton.com()[0]
        height = self.robot_skeleton.bodynodes[2].com()[1]

        alive_bonus = 1.0
        reward = (posafter - posbefore) / (self.dt)
        reward += alive_bonus
        reward -= 0.003 * np.square(a).sum()

        s = self.state_vector()
        done = not (
        np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (np.abs(self.robot_skeleton.dq) < 100).all() and
        (height > 0.7) and (abs(ang) < .2))

        ob = self._get_obs()

        if self.perturb_MP:
            # bounded random walk of mp
            rdwk_step = 0.005
            bound_size = 0.05
            mp = self.param_manager.get_simulator_parameters() + self.np_random.uniform(-rdwk_step, rdwk_step, len(
                self.param_manager.get_simulator_parameters()))
            for dim in range(len(self.current_param)):
                if mp[dim] > self.current_param[dim] + bound_size:
                    dist = mp[dim] - self.current_param[dim] - bound_size
                    samp_range = 2 * rdwk_step - dist
                    mp[dim] -= dist + self.np_random.uniform(0, samp_range)
                elif mp[dim] < self.current_param[dim] - bound_size:
                    dist = self.current_param[dim] - bound_size - mp[dim]
                    samp_range = 2 * rdwk_step - dist
                    mp[dim] += dist + self.np_random.uniform(0, samp_range)
            self.param_manager.set_simulator_parameters(mp)

        self.cur_step += 1

        return ob, reward, done, {'model_parameters': self.param_manager.get_simulator_parameters(),
                                  'vel_rew': (posafter - posbefore) / self.dt, 'action_rew': 1e-3 * np.square(a).sum(), 'done_return': done,
                                  'state_act': state_act, 'next_state': self.state_vector() - state_pre,
                                  'dyn_model_id': self.dyn_model_id, 'state_index': self.state_index}

    def _get_obs(self):
        state = np.concatenate([
            self.robot_skeleton.q[1:],
            self.robot_skeleton.dq,
        ])
        state[0] = self.robot_skeleton.bodynodes[2].com()[1]
        if self.train_UP:
            state = np.concatenate([state, self.param_manager.get_simulator_parameters()])
        if self.noisy_input:
            state = state + np.random.normal(0, .01, len(state))
        if self.train_mp_sel:
            state = np.concatenate([state, [np.random.random()]])

        return state

    def reset_model(self):
        for world in self.dart_worlds:
            world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        if self.resample_MP:
            self.param_manager.resample_parameters()
            self.current_param = self.param_manager.get_simulator_parameters()

        # Split the mp space by left and right for now
        if self.train_UP:
            self.state_index = 0

        self.state_action_buffer = []  # for UPOSI

        state = self._get_obs()

        self.cur_step = 0

        self.total_dist = []

        self.t = 0
        return state

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -5.5

