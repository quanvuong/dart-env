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
        self.action_scale = np.array([200, 200, 200])
        self.train_UP = False
        self.noisy_input = False
        self.avg_div = 0

        self.resample_MP = False  # whether to resample the model paraeters
        self.train_mp_sel = False
        self.perturb_MP = False
        obs_dim = 11
        self.param_manager = hopperContactMassManager(self)

        self.split_task_test = True
        self.learn_diff_style = False
        self.learn_forwardbackward = False
        self.tasks = TaskList(2)
        self.tasks.add_world_choice_tasks([1,1])
        #self.tasks.add_fix_param_tasks([0, [0.1, 1.0]])
        #self.tasks.add_fix_param_tasks([1, [0.3, 0.0]])
        #self.tasks.add_fix_param_tasks([2, [0.3, 0.6]])
        #self.tasks.add_fix_param_tasks([4, [0.4, 0.7]])
        self.tasks.add_range_param_tasks([1, [[0.0,0.2], [0.8,1.0]]], expand=0.0)
        #self.tasks.add_range_param_tasks([2, [[0.4, 0.5]]])
        #self.tasks.add_joint_limit_tasks([-2, [[-2.61799, 0], [0, 2.61799]]])
        self.task_expand_flag = False

        self.upselector = None
        modelpath = os.path.join(os.path.dirname(__file__), "models")
        #self.upselector = joblib.load(os.path.join(modelpath, 'UPSelector_torso_lrange35_sd17_3seg.pkl'))

        #self.param_manager.sampling_selector = upselector
        #self.param_manager.selector_target = 2

        if self.train_UP:
            obs_dim += self.param_manager.param_dim
        if self.train_mp_sel:
            obs_dim += 1
        if self.split_task_test:
            obs_dim += self.tasks.task_input_dim()
        if self.avg_div > 1:
            obs_dim += self.avg_div

        self.dyn_models = [None]
        self.dyn_model_id = 0
        self.base_path = None
        self.transition_locator = None
        self.baseline = None

        self.total_dist = []

        dart_env.DartEnv.__init__(self, ['hopper_capsule.skel', 'hopper_box.skel', 'hopper_ellipsoid.skel', 'hopper_hybrid.skel'], 4, obs_dim, self.control_bounds, disableViewer=True)

        self.current_param = self.param_manager.get_simulator_parameters()

        self.dart_worlds[0].set_collision_detector(3)
        self.dart_worlds[1].set_collision_detector(0)
        self.dart_worlds[2].set_collision_detector(1)
        self.dart_worlds[3].set_collision_detector(1)

        self.dart_world=self.dart_worlds[0]
        self.robot_skeleton=self.dart_world.skeletons[-1]
        '''self.current_param = self.param_manager.get_simulator_parameters()
        curcontparam = copy.copy(self.param_manager.controllable_param)
        self.param_manager.controllable_param = [1]
        self.param_manager.set_simulator_parameters([1.0])
        self.param_manager.controllable_param = curcontparam'''

        #if self.learn_diff_style:
        #    for world in self.dart_worlds:
        #        world.skeletons[-1].joints[-3].set_position_upper_limit(0, 2.5)
        #        world.skeletons[-1].joints[-3].set_position_lower_limit(0, -0.0)

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

        if self.dyn_model_id == 0 or self.dyn_models[self.dyn_model_id-1] is None:
            self.do_simulation(tau, self.frame_skip)
        elif self.dyn_models[self.dyn_model_id-1] is not None and self.base_path is None:
            self.total_dist.append(0)
            new_state = self.dyn_models[self.dyn_model_id-1].do_simulation(self.state_vector(), a, self.frame_skip)
            self.set_state_vector(new_state)
        elif self.dyn_models[self.dyn_model_id-1] is not None:
            #self.total_dist.append(np.linalg.norm(self.base_path['env_infos']['state_act'][self.cur_step] - np.concatenate([self.state_vector(), clamped_control])))
            self.total_dist.append(0)
            #ref_state = self.base_path['env_infos']['state_act'][self.cur_step][0:len(self.state_vector())] + self.base_path['env_infos']['next_state'][self.cur_step]
            #new_state = self.dyn_models[self.dyn_model_id-1].do_simulation(self.state_vector(), clamped_control, self.frame_skip)
            #self.set_state_vector(new_state)
            tau = np.zeros(self.robot_skeleton.ndofs)
            tau[3:] = clamped_control * self.action_scale
            self.do_simulation(tau, self.frame_skip)
            #diff_size = np.linalg.norm(self.state_vector() - ref_state)
            #move_direction = (self.state_vector() - ref_state) / np.linalg.norm(self.state_vector() - ref_state)
            #if diff_size > 0.01:
            #    self.set_state_vector(ref_state + 0.01 * move_direction)

            #print('diff: ', np.linalg.norm(new_state - self.state_vector()))

            '''cur_state = self.state_vector()
            cur_act = a
            if self.transition_locator is None:
                base_state_act = self.base_path['env_infos']['state_act'][self.cur_step]
                base_state = base_state_act[0:len(cur_state)]
                base_act = base_state_act[-len(cur_act):]
                base_next_state = base_state+self.base_path['env_infos']['next_state'][self.cur_step]
                self.total_dist.append(np.linalg.norm(base_state_act-np.concatenate([cur_state,cur_act])))
            else:
                query = self.transition_locator.kneighbors([np.concatenate([cur_state, cur_act])])
                dist = query[0][0][0]
                #print('distance: ', dist)
                ind = query[1][0][0]
                base_state_act = self.transition_locator._fit_X[ind]
                base_state = base_state_act[0:len(cur_state)]
                base_act = base_state_act[-len(cur_act):]
                base_next_state = base_state + self.transition_locator._y[ind]
                self.total_dist.append(dist)
            new_state = self.dyn_models[self.dyn_model_id-1].do_simulation_corrective(base_state, base_act, \
                                            self.frame_skip, base_next_state, cur_state - base_state, cur_act-base_act)
            self.set_state_vector(new_state)'''

    def _step(self, a):
        prev_obs = self._get_obs()
        pre_state = [self.state_vector()]
        if self.train_UP:
            pre_state.append(self.param_manager.get_simulator_parameters())
        posbefore = self.robot_skeleton.q[0]
        state_act = np.concatenate([self.state_vector(), np.clip(a,-1,1)])
        state_pre = np.copy(self.state_vector())
        self.advance(a)
        posafter,ang = self.robot_skeleton.q[0,2]
        height = self.robot_skeleton.bodynodes[2].com()[1]

        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()

        joint_limit_penalty = 0
        for j in [-2]:
            if (self.robot_skeleton.q_lower[j] - self.robot_skeleton.q[j]) > -0.05:
                joint_limit_penalty += abs(1.5)
            if (self.robot_skeleton.q_upper[j] - self.robot_skeleton.q[j]) < 0.05:
                joint_limit_penalty += abs(1.5)

        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        if self.state_index == 1 and self.learn_forwardbackward:
            reward *= -1
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        reward -= 5e-1 * joint_limit_penalty
        #reward -= 1e-7 * total_force_mag

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (np.abs(self.robot_skeleton.dq) < 100).all()\
                      and (height > self.height_threshold_low) and height < 1.8 and (abs(ang) < .4))
        #if not((height > .7) and (height < 1.8) and (abs(ang) < .4)):
        #    reward -= 1

        if self.learn_diff_style:
            if self.state_index == 0: # avoid contact
                if height > self.height_threshold_low + 0.15:
                    done = True
            elif self.state_index == 1: # encourage contact
                if height < self.height_threshold_low - 0.1 or height > 1.65:
                    done = True 
 
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
            # simply add noise
            #self.param_manager.set_simulator_parameters(self.current_param + np.random.uniform(-0.01, 0.01, len(self.current_param)))

        self.cur_step += 1
        '''if self.base_path is not None and self.dyn_model_id != 0 and self.transition_locator is None:
            if self.cur_step - self.start_step > 9000 or len(self.base_path['env_infos']['state_act']) <= self.cur_step:
                done = True
        if self.dyn_model_id != 0:
            if self.total_dist[-1] > 2.0:
                done = True'''

        '''if done and self.baseline is not None and self.dyn_model_id != 0:
            fake_obs = np.copy(prev_obs)
            fake_obs[-1] = 0
            fake_obs[-2] = 1 # use this to query the normal baseline value
            fake_path = {'observations': [fake_obs], 'rewards':[reward]}
            reward = self.baseline.predict(fake_path)[-1]'''
        if self.dyn_model_id != 0:
            reward *= 0.5

        return ob, reward, done, {'model_parameters':self.param_manager.get_simulator_parameters(), 'vel_rew':(posafter - posbefore) / self.dt, 'action_rew':1e-3 * np.square(a).sum(), 'forcemag':1e-7*total_force_mag, 'done_return':done,
                                  'state_act': state_act, 'next_state':self.state_vector()-state_pre, 'dyn_model_id':self.dyn_model_id, 'state_index':self.state_index}

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            np.clip(self.robot_skeleton.dq,-10,10)
        ])
        state[0] = self.robot_skeleton.bodynodes[2].com()[1]
        if self.train_UP:
            state = np.concatenate([state, self.param_manager.get_simulator_parameters()])
        if self.noisy_input:
            state = state + np.random.normal(0, .01, len(state))
        if self.train_mp_sel:
            state = np.concatenate([state, [np.random.random()]])

        if self.split_task_test:
            state = np.concatenate([state, self.tasks.get_task_inputs(self.state_index)])

        if self.avg_div > 1:
            return_state = np.zeros(len(state) + self.avg_div)
            return_state[0:len(state)] = state
            return_state[len(state) + self.state_index] = 1
            return return_state

        return state

    def get_reward(self, statevec_before, act, statevec_after, multiplier):
        self.set_state_vector(statevec_before)
        posbefore = self.robot_skeleton.q[0]
        self.set_state_vector(statevec_after)
        posafter,ang = self.robot_skeleton.q[0,2]

        joint_limit_penalty = 0
        for j in [-2]:
            if (self.robot_skeleton.q_lower[j] - self.robot_skeleton.q[j]) > -0.05:
                joint_limit_penalty += abs(1.5)
            if (self.robot_skeleton.q_upper[j] - self.robot_skeleton.q[j]) < 0.05:
                joint_limit_penalty += abs(1.5)

        alive_bonus = 1.0
        reward = 0.6*(posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(act).sum()
        reward -= 5e-1 * joint_limit_penalty

        return reward * multiplier

    def reset_model(self):
        for world in self.dart_worlds:
            world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        if self.resample_MP:
            self.param_manager.resample_parameters()
            #self.param_manager.set_simulator_parameters(np.array([0.6, 0.5]) + self.np_random.uniform(low=-0.05, high=0.05, size=2))
            #self.param_manager.set_simulator_parameters(np.array([0.6, 0.5]))
            self.current_param = self.param_manager.get_simulator_parameters()
            #self.param_manager.set_simulator_parameters(mp)


        self.state_index = self.dyn_model_id

        # Split the mp space by left and right for now
        if self.train_UP:
            self.state_index = 0
            if len(self.param_manager.get_simulator_parameters()) > 1:
                if self.param_manager.get_simulator_parameters()[0] < 0.5 and self.param_manager.get_simulator_parameters()[1] >= 0.5:
                    self.state_index = 1
                elif self.param_manager.get_simulator_parameters()[0] >= 0.5 and self.param_manager.get_simulator_parameters()[1] < 0.5:
                    self.state_index = 2
                elif self.param_manager.get_simulator_parameters()[0] >= 0.5 and self.param_manager.get_simulator_parameters()[1] >= 0.5:
                    self.state_index = 3
            if self.upselector is not None:
                self.state_index = self.upselector.classify([self.param_manager.get_simulator_parameters()], False)

        if self.split_task_test:
            if self.task_expand_flag:
                self.tasks.expand_range_param_tasks()
                self.task_expand_flag = False
            self.state_index = np.random.randint(self.tasks.task_num)
            world_choice, pm_id, pm_val, jt_id, jt_val = self.tasks.resample_task(self.state_index)
            if self.dart_world != self.dart_worlds[world_choice]:
                self.dart_world = self.dart_worlds[world_choice]
                self.robot_skeleton = self.dart_world.skeletons[-1]
                qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
                qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
                self.set_state(qpos, qvel)
                if not self.disableViewer:
                    self._get_viewer().sim = self.dart_world
            self.param_manager.controllable_param = pm_id
            self.param_manager.set_simulator_parameters(np.array(pm_val))
            for ind, jtid in enumerate(jt_id):
                self.robot_skeleton.joints[jtid].set_position_upper_limit(0, jt_val[ind][1])
                self.robot_skeleton.joints[jtid].set_position_lower_limit(0, jt_val[ind][0])

        self.state_action_buffer = [] # for UPOSI

        state = self._get_obs()

        self.cur_step = 0

        if self.base_path is not None and self.dyn_model_id != 0:
            base_len = len(self.base_path['env_infos']['state_act'])
            self.cur_step = 0#np.random.randint(base_len-1)
            self.start_step = self.cur_step
            base_state = self.base_path['env_infos']['state_act'][self.cur_step][0:len(self.state_vector())]
            self.set_state_vector(base_state + self.np_random.uniform(low=-0.01, high=0.01, size=len(base_state)))

        self.total_dist = []

        self.height_threshold_low = 0.56*self.robot_skeleton.bodynodes[2].com()[1]

        return state

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -5.5
