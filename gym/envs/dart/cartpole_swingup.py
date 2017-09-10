__author__ = 'yuwenhao'

import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from gym.envs.dart.parameter_managers import CartPoleManager
import os
import joblib
from gym.envs.dart.sub_tasks import *
import copy


class DartCartPoleSwingUpEnv(dart_env.DartEnv):
    def __init__(self):
        self.control_bounds = np.array([[1.0], [-1.0]])
        self.action_scale = np.array([40])
        self.train_UP = False
        self.resample_MP = False  # whether to resample the model paraeters
        self.train_mp_sel = False
        self.perturb_MP = False
        self.avg_div = 0
        self.param_manager = CartPoleManager(self)

        self.split_task_test = True
        self.tasks = TaskList(1)
        self.tasks.add_world_choice_tasks([0])
        # self.tasks.add_range_param_tasks([0, [[0.0, 0.5], [0.5, 1.0], [0.0, 0.5], [0.5, 1.0]]])
        # self.tasks.add_range_param_tasks([2, [[0.0, 0.5], [0.5, 1.0], [0.5, 1.0], [0.0, 0.5]]])

        self.cur_step = 0

        modelpath = os.path.join(os.path.dirname(__file__), "models")
        # self.upselector = joblib.load(os.path.join(modelpath, 'UPSelector_jug_sd9_mass_2seg.pkl'))

        obs_dim = 4
        if self.train_UP:
            obs_dim += self.param_manager.param_dim
        if self.train_mp_sel:
            obs_dim += 5

        self.juggling = False
        if self.juggling:
            obs_dim += 4
            self.action_scale *= 2

        self.dyn_models = [None]
        self.dyn_model_id = 0
        self.base_path = None
        self.transition_locator = None

        if self.split_task_test:
            obs_dim += self.tasks.task_input_dim()
        if self.avg_div > 1:
            obs_dim += self.avg_div

        dart_env.DartEnv.__init__(self, ['cartpole_swingup.skel', 'cartpole_swingup_variation1.skel',
                                         'cartpole_swingup_variation2.skel'], 2, obs_dim, self.control_bounds, dt=0.01,
                                  disableViewer=True)
        self.current_param = self.param_manager.get_simulator_parameters()
        # self.dart_world.skeletons[1].bodynodes[0].set_friction_coeff(0.2)
        # self.dart_world.skeletons[1].bodynodes[0].set_restitution_coeff(0.7)
        # self.dart_world.skeletons[-1].bodynodes[-1].set_restitution_coeff(0.7)

        self.dart_world = self.dart_worlds[0]
        self.robot_skeleton = self.dart_world.skeletons[-1]

        utils.EzPickle.__init__(self)

    def _step(self, a):
        # if a[0] > self.control_bounds[0][0] or a[0] < self.control_bounds[1][0]:
        #    a[0] = np.sign(a[0])
        # if np.abs(a[0]) > 1:
        #    a[0] = np.sign(a[0])

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

        if self.dyn_model_id == 0 or self.dyn_models[self.dyn_model_id - 1] is None:
            self.do_simulation(tau, self.frame_skip)
        elif self.dyn_models[self.dyn_model_id - 1] is not None and self.base_path is None:
            # new_state = self.dyn_models[self.dyn_model_id-1].do_simulation(self.state_vector(), tau, self.frame_skip)
            # self.set_state_vector(new_state)
            self.do_simulation(tau, self.frame_skip)
            dist = 0
        elif self.dyn_models[self.dyn_model_id - 1] is not None:
            self.do_simulation(tau, self.frame_skip)
            dist = 0
            '''cur_state = self.state_vector()
            tau /= 40.0
            cur_act = tau
            if self.transition_locator is None:
                base_state_act = self.base_path['env_infos']['state_act'][self.cur_step]
                base_state = base_state_act[0:len(cur_state)]
                base_act = base_state_act[-len(cur_act):]
                base_next_state = base_state+self.base_path['env_infos']['next_state'][self.cur_step]
            else:
                query = self.transition_locator.kneighbors([np.concatenate([cur_state, cur_act])])
                dist = query[0][0][0]
                ind = query[1][0][0]
                base_state_act = self.transition_locator._fit_X[ind]
                base_state = base_state_act[0:len(cur_state)]
                base_act = base_state_act[-len(cur_act):]
                base_next_state = base_state + self.transition_locator._y[ind]
                self.total_dist.append(dist)

            new_state = self.dyn_models[self.dyn_model_id-1].do_simulation_corrective(base_state, base_act, \
                                            self.frame_skip, base_next_state, cur_state - base_state, cur_act-base_act)
            #new_state = base_next_state + 0.1 * (new_state - base_next_state)/np.linalg.norm(new_state - base_next_state)
            self.set_state_vector(new_state)'''

        ob = self._get_obs()

        ang = self.robot_skeleton.q[1]

        ang_proc = (np.abs(ang) % (2 * np.pi))
        ang_proc = np.min([ang_proc, (2 * np.pi) - ang_proc])

        if not self.juggling:
            alive_bonus = 6.0
        else:
            alive_bonus = 4.0
        ang_cost = 1.0 * (ang_proc ** 2)
        quad_ctrl_cost = 0.01 * np.square(a).sum()
        com_cost = 2.0 * np.abs(self.robot_skeleton.q[0]) ** 2

        reward = alive_bonus - ang_cost - quad_ctrl_cost - com_cost
        if ang_proc < 0.5:
            reward += np.max([5 - (ang_proc) * 4, 0]) + np.max([3 - np.abs(self.robot_skeleton.dq[1]), 0])

        done = abs(self.robot_skeleton.dq[1]) > 35 or abs(self.robot_skeleton.q[0]) > 2.0

        if self.juggling:
            if self.dart_world.skeletons[1].com()[1] < -1.0:  # or self.dart_world.skeletons[2].com()[1] < 0.1:
                # reward -= 50
                done = True

        if self.perturb_MP:
            self.param_manager.set_simulator_parameters(
                self.current_param + np.random.uniform(-0.01, 0.01, len(self.current_param)))

        if self.dyn_model_id != 0:
            reward *= 1.0
        self.cur_step += 1
        '''if self.base_path is not None and self.dyn_model_id != 0 and self.transition_locator is None:
            if len(self.base_path['env_infos']['state_act']) <= self.cur_step:
                done = True
        if self.dyn_model_id != 0 and self.transition_locator is not None:
            if dist > 0.5:
                done = True'''

        if self.cur_step * self.dt < 1.3 and self.juggling:
            self.dart_world.skeletons[1].set_positions(self.jug_pos)
            self.dart_world.skeletons[1].set_velocities(self.jug_vel)
        if self.cur_step * self.dt < 2.3 and self.juggling:
            self.dart_world.skeletons[2].set_positions(self.jug_pos2)
            self.dart_world.skeletons[2].set_velocities(self.jug_vel2)

        return ob, reward, done, {'model_parameters': self.param_manager.get_simulator_parameters(),
                                  'state_act': state_act, 'next_state': self.state_vector() - state_pre
            , 'dyn_model_id': self.dyn_model_id, 'state_index': self.state_index}

    def _get_obs(self):
        state = np.concatenate([self.robot_skeleton.q, self.robot_skeleton.dq]).ravel()

        ang = self.robot_skeleton.q[1]
        ang_proc = (ang % (2 * np.pi))
        if ang_proc > np.pi:
            ang_proc -= 2 * np.pi

        state[1] = ang_proc

        if self.juggling:
            state = np.concatenate([state, self.dart_world.skeletons[1].com()[[0, 1]],
                                    self.dart_world.skeletons[1].com_velocity()[[0, 1]]])  # ,\
            # self.dart_world.skeletons[2].com()[[0, 1]], self.dart_world.skeletons[2].com_velocity()[[0, 1]]])

        if self.train_UP:
            state = np.concatenate([state, self.param_manager.get_simulator_parameters()])

        if self.train_mp_sel:
            state = np.concatenate([state, [self.rand]])

        if self.split_task_test:
            state = np.concatenate([state, self.tasks.get_task_inputs(self.state_index)])

        if self.avg_div > 1:
            return_state = np.zeros(len(state) + self.avg_div)
            return_state[0:len(state)] = state
            return_state[len(state) + self.state_index] = 1.0
            return return_state

        return state

    def reset_model(self):
        self.total_dist = []
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.1, high=.1, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        if self.np_random.uniform(0, 1) > 0.5:
            qpos[1] += np.pi
        else:
            qpos[1] += -np.pi

        # qpos[1]+=self.np_random.uniform(low=-np.pi, high=np.pi, size=1)

        if self.resample_MP:
            self.param_manager.resample_parameters()

        self.current_param = self.param_manager.get_simulator_parameters()
        self.state_index = self.dyn_model_id

        if self.train_UP:
            self.state_index = 0
            if len(self.param_manager.get_simulator_parameters()) > 1:
                if self.param_manager.get_simulator_parameters()[0] < 0.5 and \
                                self.param_manager.get_simulator_parameters()[1] >= 0.5:
                    self.state_index = 1
                elif self.param_manager.get_simulator_parameters()[0] >= 0.5 and \
                                self.param_manager.get_simulator_parameters()[1] < 0.5:
                    self.state_index = 2
                elif self.param_manager.get_simulator_parameters()[0] >= 0.5 and \
                                self.param_manager.get_simulator_parameters()[1] >= 0.5:
                    self.state_index = 3
            if self.upselector is not None:
                self.state_index = self.upselector.classify([self.param_manager.get_simulator_parameters()],
                                                            stoch=False)

        if not self.juggling:
            self.dart_world.skeletons[1].set_positions([0, 0, 0, 100, 0, 0])
            self.dart_world.skeletons[2].set_positions([0, 0, 0, 200, 0, 0])
        else:
            self.jug_pos = self.dart_world.skeletons[1].q + self.np_random.uniform(low=-.1, high=.1, size=6)
            self.jug_vel = self.dart_world.skeletons[1].dq + self.np_random.uniform(low=-.01, high=.01, size=6)
            self.jug_pos[-1] = 0
            self.jug_vel[-1] = 0
            self.jug_pos[-3] = self.np_random.uniform(low=0.05, high=0.5)
            self.dart_world.skeletons[1].set_positions(self.jug_pos)
            self.dart_world.skeletons[1].set_velocities(self.jug_vel)
            # self.dart_world.skeletons[2].set_positions([0,0,0,200, 0, 0])
            self.jug_pos2 = self.dart_world.skeletons[2].q + self.np_random.uniform(low=-.1, high=.1, size=6)
            self.jug_vel2 = self.dart_world.skeletons[2].dq + self.np_random.uniform(low=-.01, high=.01, size=6)
            self.jug_pos2[-1] = 0
            self.jug_vel2[-1] = 0
            self.jug_pos2[-3] = self.np_random.uniform(low=-0.5, high=-0.05)
            self.jug_pos2[-3] = 100
            self.dart_world.skeletons[2].set_positions(self.jug_pos2)
            self.dart_world.skeletons[2].set_velocities(self.jug_vel2)

        if self.split_task_test:
            self.state_index = np.random.randint(self.tasks.task_num)
            world_choice, pm_id, pm_val, jt_id, jt_val = self.tasks.resample_task(self.state_index)
            if self.dart_world != self.dart_worlds[world_choice]:
                self.dart_world = self.dart_worlds[world_choice]
                self.robot_skeleton = self.dart_world.skeletons[-1]
                self.dart_world.skeletons[1].set_positions([0, 0, 0, 100, 0, 0])
                self.dart_world.skeletons[2].set_positions([0, 0, 0, 200, 0, 0])
                qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.1, high=.1, size=self.robot_skeleton.ndofs)
                qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01,
                                                                       size=self.robot_skeleton.ndofs)
                if self.np_random.uniform(0, 1) > 0.5:
                    qpos[1] += np.pi
                else:
                    qpos[1] += -np.pi
                self.set_state(qpos, qvel)
                if not self.disableViewer:
                    self._get_viewer().sim = self.dart_world
            self.param_manager.controllable_param = pm_id
            self.param_manager.set_simulator_parameters(np.array(pm_val))
            for ind, jtid in enumerate(jt_id):
                self.robot_skeleton.joints[jtid].set_position_upper_limit(0, jt_val[ind][1])
                self.robot_skeleton.joints[jtid].set_position_lower_limit(0, jt_val[ind][0])

        if self.train_mp_sel:
            self.rand = np.random.random()
        self.set_state(qpos, qvel)

        # if self.base_path is not None and self.dyn_model_id != 0:
        #    base_len = len(self.base_path)
        #    base_state = self.base_path['env_infos']['state_act'][np.random.randint(base_len-3)][0:len(self.state_vector())]
        #    self.set_state_vector(base_state + self.np_random.uniform(low=-0.01, high=0.01, size=len(base_state)))

        self.cur_step = 0

        return self._get_obs()

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -4.5
        self._get_viewer().scene.tb._set_theta(0)
        self.track_skeleton_id = 0
