import numpy as np
from gym import utils
from gym.envs.dart import dart_env

from gym.envs.dart.parameter_managers import *
from gym.envs.dart.sub_tasks import *
import copy

import joblib, os

class DartHopperEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0, 1.0, 1.0],[-1.0, -1.0, -1.0]])
        self.action_scale = np.array([200.0, 200.0, 200.0])
        self.train_UP = False
        self.noisy_input = False
        obs_dim = 11

        self.resample_MP = False  # whether to resample the model paraeters
        self.param_manager = hopperContactMassManager(self)

        self.dyn_models = [None]
        self.dyn_model_id = 0
        self.base_path = None
        self.transition_locator = None
        self.baseline = None

        self.t = 0

        self.total_dist = []

        dart_env.DartEnv.__init__(self, ['hopper_capsule.skel', 'hopper_box.skel', 'hopper_ellipsoid.skel'], 4, obs_dim, self.control_bounds, disableViewer=True)

        self.current_param = self.param_manager.get_simulator_parameters()

        self.dart_worlds[0].set_collision_detector(3)
        self.dart_worlds[1].set_collision_detector(0)
        self.dart_worlds[2].set_collision_detector(1)

        self.dart_world=self.dart_worlds[0]
        self.robot_skeleton=self.dart_world.skeletons[-1]

        # setups for articunet
        self.state_dim = 32
        self.enc_net = []
        self.act_net = []
        self.vf_net = []
        self.merg_net = []
        self.net_modules = []
        self.net_vf_modules = []
        self.generic_modules = []

        # build dynamics model
        self.generic_modules.append([0, [self.state_dim] * 3, self.state_dim, 128, 2, 'world_node1'])
        self.generic_modules.append([3, [self.state_dim] * 3, self.state_dim, 128, 2, 'revolute_node1'])
        self.generic_modules.append([6, [self.state_dim] * 3, self.state_dim, 128, 2, 'pole_node1'])
        self.generic_modules.append([0, [self.state_dim] * 2, 6, 128, 2, 'pole_predictor'])
        self.generic_modules.append([0, [self.state_dim] * 2, 2, 128, 2, 'revolute_predictor'])

        self.generic_modules.append([0, [self.state_dim] * 3, self.state_dim, 128, 2, 'world_node2'])
        self.generic_modules.append([3, [self.state_dim] * 3, self.state_dim, 128, 2, 'revolute_node2'])
        self.generic_modules.append([6, [self.state_dim] * 3, self.state_dim, 128, 2, 'pole_node2'])

        # first pass
        self.net_modules.append([[4, 5, 6, 7, 8, 9], 2, [None, None, None]])  # [world, jnt, bdnd]
        self.net_modules.append([[10, 11, 12, 13, 14, 15], 2, [None, None, None]])
        self.net_modules.append([[0, 2, 16], 1, [None, None, [0]]])
        self.net_modules.append([[1, 3, 17], 1, [None, None, [0, 1]]])
        self.net_modules.append([[], 0, [None, [2, 3], [0, 1]]])

        # second pass
        self.net_modules.append([[4, 5, 6, 7, 8, 9], 2 + 5, [[4], [2, 3], [1]]])  # [world, jnt, bdnd]
        self.net_modules.append([[10, 11, 12, 13, 14, 15], 2 + 5, [[4], [3], [0]]])
        self.net_modules.append([[0, 2, 16], 1 + 5, [[4], [2, 3], [5]]])
        self.net_modules.append([[1, 3, 17], 1 + 5, [[4], [2, 3], [5, 6]]])
        self.net_modules.append([[], 0 + 5, [[4], [7, 8], [5, 6]]])

        # pass to predictor
        self.net_modules.append([[], 4, [[9], [7]]])
        self.net_modules.append([[], 4, [[9], [8]]])
        self.net_modules.append([[], 3, [[9], [5]]])
        self.net_modules.append([[], 3, [[9], [6]]])

        self.net_modules.append([[], None, [[10], [11], [12], [13]], None, False])
        self.reorder_output = np.array([0, 2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.int32)

        utils.EzPickle.__init__(self)


    def advance(self, a):
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[3:] = clamped_control * self.action_scale

        self.do_simulation(tau, self.frame_skip)

    def _step(self, a):
        self.t += self.dt
        prev_obs = self._get_obs()
        pre_state = [self.state_vector()]
        if self.train_UP:
            pre_state.append(self.param_manager.get_simulator_parameters())
        posbefore = self.robot_skeleton.q[0]
        heightbefore = self.robot_skeleton.q[1]
        state_act = np.concatenate([self.state_vector(), np.clip(a,-1,1)])
        state_pre = np.copy(self.state_vector())
        self.advance(a)
        posafter, heightafter, ang = self.robot_skeleton.q[0,1,2]
        height = self.robot_skeleton.bodynodes[2].com()[1]

        fall_on_ground = False
        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()
            if contact.bodynode1 != self.robot_skeleton.bodynodes[-1] and contact.bodynode2 != self.robot_skeleton.bodynodes[-1]:
                fall_on_ground = True

        joint_limit_penalty = 0
        for j in [-2]:
            if (self.robot_skeleton.q_lower[j] - self.robot_skeleton.q[j]) > -0.05:
                joint_limit_penalty += abs(1.5)
            if (self.robot_skeleton.q_upper[j] - self.robot_skeleton.q[j]) < 0.05:
                joint_limit_penalty += abs(1.5)

        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        reward -= 5e-1 * joint_limit_penalty

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (np.abs(self.robot_skeleton.dq) < 100).all() and
                    (height > self.height_threshold_low) and (abs(ang) < .2))

        ob = self._get_obs()

        self.cur_step += 1


        envinfo = {'model_parameters': self.param_manager.get_simulator_parameters(), 'vel_rew': (posafter - posbefore) / self.dt,
         'action_rew': 1e-3 * np.square(a).sum(), 'forcemag': 1e-7 * total_force_mag, 'done_return': done,
         'state_act': state_act, 'next_state': self.state_vector() - state_pre,
         'plot_info':[self.robot_skeleton.C[0], self.robot_skeleton.C[1]]}

        return ob, reward, done, envinfo

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            self.robot_skeleton.dq,
        ])
        state[0] = self.robot_skeleton.bodynodes[2].com()[1]
        if self.train_UP:
            state = np.concatenate([state, self.param_manager.get_simulator_parameters()])
        if self.noisy_input:
            state = state + np.random.normal(0, .01, len(state))

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

        self.state_action_buffer = [] # for delay

        state = self._get_obs()

        self.cur_step = 0

        self.height_threshold_low = 0.56*self.robot_skeleton.bodynodes[2].com()[1]
        self.t = 0

        return state

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -5.5
