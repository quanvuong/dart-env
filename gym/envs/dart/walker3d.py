__author__ = 'yuwenhao'

import numpy as np
from gym import utils
from gym.envs.dart import dart_env
import joblib
import os

from gym.envs.dart.parameter_managers import *
import time

class DartWalker3dEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0]*15,[-1.0]*15])
        self.action_scale = np.array([150.0]*15)
        self.action_scale[[-1,-2,-7,-8]] = 60
        self.action_scale[[0, 1, 2]] = 100
        #self.action_scale = np.array([40,80,150, 150,60,80,120,60,60, 150,60,80,120,60,60])
        self.action_scale = np.array([200, 200, 200, 250, 60, 80, 100, 60, 60, 250, 60, 80, 100, 60, 60])
        obs_dim = 41

        self.t = 0
        self.target_vel = 1.0
        self.init_tv = 1.0
        self.final_tv = 4.0
        self.tv_endtime = 2.5
        self.smooth_tv_change = True
        self.rand_target_vel = False
        self.init_push = False
        self.enforce_target_vel = True
        self.running_avg_rew_only = True
        self.avg_rew_weighting = []
        self.vel_cache = []

        self.hard_enforce = False
        self.treadmill = False
        self.treadmill_vel = -1.0
        self.base_policy = None
        modelpath = os.path.join(os.path.dirname(__file__), "models")
        self.cur_step = 0
        self.stepwise_rewards = []
        self.conseq_limit_pen = 0 # number of steps lying on the wall
        self.constrain_2d = True
        self.init_balance_pd = 2000.0
        self.init_vel_pd = 2000.0
        self.end_balance_pd = 2000.0
        self.end_vel_pd = 2000.0
        self.pd_vary_end = self.target_vel * 6.0
        self.current_pd = self.init_balance_pd
        self.vel_enforce_kp = self.init_vel_pd

        self.energy_weight = 0.1

        self.local_spd_curriculum = True
        self.anchor_kp = np.array([2000, 2000])
        self.curriculum_step_size = 0.1 # 10%
        self.min_curriculum_step = 50 # include (0, 0) if distance between anchor point and origin is smaller than this value

        # state related
        self.contact_info = np.array([0, 0])
        self.include_additional_info = True
        if self.include_additional_info:
            obs_dim += len(self.contact_info)
        if self.rand_target_vel or self.smooth_tv_change:
            obs_dim += 1

        self.curriculum_id = 0
        self.spd_kp_candidates = None
        self.param_manager = walker3dManager(self)

        if self.treadmill:
            dart_env.DartEnv.__init__(self, 'walker3d_treadmill.skel', 15, obs_dim, self.control_bounds, disableViewer=True)
        else:
            dart_env.DartEnv.__init__(self, 'walker3d_waist.skel', 15, obs_dim, self.control_bounds,
                                      disableViewer=True, dt=0.002)

        #self.dart_world.set_collision_detector(3)
        self.robot_skeleton.set_self_collision_check(True)

        for i in range(1, len(self.dart_world.skeletons[0].bodynodes)):
            self.dart_world.skeletons[0].bodynodes[i].set_friction_coeff(0)

        for i in range(0, len(self.dart_world.skeletons[0].bodynodes)):
            self.dart_world.skeletons[0].bodynodes[i].set_friction_coeff(20)
        for i in range(0, len(self.dart_world.skeletons[1].bodynodes)):
            self.dart_world.skeletons[1].bodynodes[i].set_friction_coeff(20)

        self.sim_dt = self.dt / self.frame_skip

        for bn in self.robot_skeleton.bodynodes:
            if len(bn.shapenodes) > 0:
                shapesize = bn.shapenodes[0].shape.size()
                print('density of ', bn.name, ' is ', bn.mass()/np.prod(shapesize))
        print('Total mass: ', self.robot_skeleton.mass())

        utils.EzPickle.__init__(self)


    def _bodynode_spd(self, bn, kp, dof, target_vel=None):
        self.Kp = kp
        self.Kd = kp * self.sim_dt
        if target_vel is not None:
            self.Kd = self.Kp
            self.Kp *= 0

        invM = 1.0 / (bn.mass() + self.Kd * self.sim_dt)
        p = -self.Kp * (bn.C[dof] + bn.dC[dof] * self.sim_dt)
        if target_vel is None:
            target_vel = 0.0
        d = -self.Kd * (bn.dC[dof] - target_vel)
        qddot = invM * (-bn.C[dof] + p + d)
        tau = p + d - self.Kd * (qddot) * self.sim_dt
        return tau

    def do_simulation(self, tau, n_frames):
        for _ in range(n_frames):
            if self.constrain_2d:
                force = self._bodynode_spd(self.robot_skeleton.bodynode('h_pelvis'), self.current_pd, 2)
                self.robot_skeleton.bodynode('h_pelvis').add_ext_force(np.array([0, 0, force]))

            if self.enforce_target_vel and not self.hard_enforce:
                force = self._bodynode_spd(self.robot_skeleton.bodynode('h_pelvis'), self.vel_enforce_kp, 0, self.target_vel)
                self.robot_skeleton.bodynode('h_pelvis').add_ext_force(np.array([force, 0, 0]))

            self.robot_skeleton.set_forces(tau)
            self.dart_world.step()

    def advance(self, a):
        if self.base_policy is not None:
            full_a = np.zeros(np.max([len(self.base_action_indices), len(self.delta_action_indices)]))
            base_action = self.base_policy.get_action(self._get_obs())[1]['mean']
            full_a[self.base_action_indices] += base_action
            full_a[self.delta_action_indices] += a
            a = np.copy(full_a)
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[6:] = clamped_control * self.action_scale

        if self.enforce_target_vel:
            if self.hard_enforce and self.treadmill:
                current_dq_tread = self.dart_world.skeletons[0].dq
                current_dq_tread[0] = self.treadmill_vel# * np.min([self.t/4.0, 1.0])
                self.dart_world.skeletons[0].dq = current_dq_tread
            elif self.hard_enforce:
                current_dq = self.robot_skeleton.dq
                current_dq[0] = self.target_vel
                self.robot_skeleton.dq = current_dq
        self.do_simulation(tau, self.frame_skip)

    def _step(self, a):
        if self.smooth_tv_change:
            self.target_vel = (np.min([self.t, self.tv_endtime]) / self.tv_endtime) * (self.final_tv - self.init_tv) + self.init_tv

        pre_state = [self.state_vector()]

        posbefore = self.robot_skeleton.bodynodes[0].com()[0]
        self.advance(np.copy(a))

        posafter = self.robot_skeleton.bodynodes[1].com()[0]
        height = self.robot_skeleton.bodynodes[1].com()[1]
        side_deviation = self.robot_skeleton.bodynodes[1].com()[2]
        angle = self.robot_skeleton.q[3]

        pos_val = np.min([np.max([0, posafter]), self.pd_vary_end])
        self.current_pd = self.init_balance_pd + (self.end_balance_pd - self.init_balance_pd)/self.pd_vary_end*pos_val
        self.vel_enforce_kp = self.init_vel_pd + (self.end_vel_pd - self.init_vel_pd) / self.pd_vary_end*pos_val

        upward = np.array([0, 1, 0])
        upward_world = self.robot_skeleton.bodynodes[1].to_world(np.array([0, 1, 0])) - self.robot_skeleton.bodynodes[1].to_world(np.array([0, 0, 0]))
        upward_world /= np.linalg.norm(upward_world)
        ang_cos_uwd = np.dot(upward, upward_world)
        ang_cos_uwd = np.arccos(ang_cos_uwd)

        forward = np.array([1, 0, 0])
        forward_world = self.robot_skeleton.bodynodes[1].to_world(np.array([1, 0, 0])) - self.robot_skeleton.bodynodes[1].to_world(np.array([0, 0, 0]))
        forward_world /= np.linalg.norm(forward_world)
        ang_cos_fwd = np.dot(forward, forward_world)
        ang_cos_fwd = np.arccos(ang_cos_fwd)

        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        self_colliding = False
        self.contact_info = np.array([0, 0])
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()
            if contact.skel_id1 == contact.skel_id2:
                self_colliding = True
            if contact.skel_id1 + contact.skel_id2 == 1:
                if contact.bodynode1 == self.robot_skeleton.bodynode('h_foot_left') or contact.bodynode2 == \
                        self.robot_skeleton.bodynode('h_foot_left'):
                    self.contact_info[0] = 1
                if contact.bodynode1 == self.robot_skeleton.bodynode('h_foot') or contact.bodynode2 == \
                        self.robot_skeleton.bodynode('h_foot'):
                    self.contact_info[1] = 1

        alive_bonus = 4.0
        vel = (posafter - posbefore) / self.dt
        self.vel_cache.append(vel)
        if self.running_avg_rew_only:
            if self.t < self.tv_endtime:
                self.avg_rew_weighting.append(0.3)
            else:
                self.avg_rew_weighting.append(1)

        if len(self.vel_cache) > int(1.0/self.dt) and (self.running_avg_rew_only):
            self.vel_cache.pop(0)
            self.avg_rew_weighting.pop(0)

        vel_rew = 0
        if not self.treadmill:
            if self.running_avg_rew_only:
                append_vel = np.zeros(int(1.0/self.dt) - len(self.vel_cache))
                vel_rew = -3.0 * np.mean(np.append(np.array(self.avg_rew_weighting) * np.abs(np.array(self.vel_cache) - self.target_vel), append_vel))
            else:
                vel_diff = np.abs(self.target_vel - vel)
        else:
            if self.running_avg_rew_only:
                append_vel = np.ones(int(1.0/self.dt) - len(self.vel_cache)) * (self.target_vel + self.treadmill_vel)
                vel_rew = -3.0 * (np.abs(self.target_vel + self.treadmill_vel - np.mean(np.append(self.vel_cache, append_vel))))
            else:
                vel_rew = -3.0 * (np.abs(self.target_vel + self.treadmill_vel - vel))

        action_pen = self.energy_weight * np.abs(a).sum()
        #action_pen = 5e-3 * np.sum(np.square(a)* self.robot_skeleton.dq[6:]* actuator_pen_multiplier)
        deviation_pen = 3 * abs(side_deviation)
        reward = vel_rew + alive_bonus - action_pen - deviation_pen
        pos_rew = alive_bonus - deviation_pen
        neg_pen = vel_rew - action_pen


        #reward -= 1e-7 * total_force_mag

        #div = self.get_div()
        #reward -= 1e-1 * np.min([(div**2), 10])

        self.t += self.dt
        self.cur_step += 1

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > 1.0) and (height < 2.0) and (abs(ang_cos_uwd) < 2.0) and (abs(ang_cos_fwd) < 2.0)
                    and np.abs(angle) < 1.3 and np.abs(self.robot_skeleton.q[5]) < 0.4 and np.abs(side_deviation) < 0.9)

        self.stepwise_rewards.append(reward)

        '''if self.treadmill:
            if np.abs(self.robot_skeleton.q[0]) > 0.4:
                done = True'''

        #if self.conseq_limit_pen > 20:
        #    done = True

        #if done:
        #    reward = 0

        broke_sim = False
        if not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all()):
            broke_sim = True

        ob = self._get_obs()
 
        return ob, reward, done, {'pos_rew':pos_rew, 'neg_pen': neg_pen, 'broke_sim':broke_sim, 'pre_state':pre_state, 'vel_rew':vel_rew, 'action_pen':action_pen, 'deviation_pen':deviation_pen, 'curriculum_id':self.curriculum_id, 'curriculum_candidates':self.spd_kp_candidates, 'done_return':done, 'dyn_model_id':0, 'state_index':0}

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            self.robot_skeleton.dq,
        ])
        if self.include_additional_info:
            state = np.concatenate([state, self.contact_info])

        if self.rand_target_vel or self.smooth_tv_change:
            state = np.concatenate([state, [self.target_vel]])

        return state

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.05, high=.05, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.05, high=.05, size=self.robot_skeleton.ndofs)
        sign = np.sign(np.random.uniform(-1, 1))
        #qpos[9] = sign * self.np_random.uniform(low=0.1, high=0.15, size=1)
        #qpos[15] = -sign * self.np_random.uniform(low=0.1, high=0.15, size=1)
        qpos[11] = sign * self.np_random.uniform(low=-0.1, high=-0.05, size=1)
        qpos[17] = -sign * self.np_random.uniform(low=0.05, high=0.1, size=1)

        if self.rand_target_vel:
            self.target_vel = np.random.uniform(0.8, 2.5)

        if self.local_spd_curriculum:
            self.spd_kp_candidates = [self.anchor_kp]
            #axis1_cand = np.max([0, self.anchor_kp[0] - self.min_curriculum_step + np.random.uniform(-5, 5)])#self.anchor_kp[0] - np.min([self.min_curriculum_step, self.anchor_kp[0] * (self.curriculum_step_size+np.random.uniform(-0.01, 0.01))])
            #axis2_cand = np.max([0, self.anchor_kp[1] - self.min_curriculum_step + np.random.uniform(-5, 5)])#self.anchor_kp[1] - np.min([self.min_curriculum_step, self.anchor_kp[1] * (self.curriculum_step_size + np.random.uniform(-0.01, 0.01))])
            #self.spd_kp_candidates.append(np.array([self.anchor_kp[0], axis2_cand]))
            #self.spd_kp_candidates.append(np.array([axis1_cand, self.anchor_kp[1]]))
            #self.spd_kp_candidates.append(np.array([axis1_cand, axis2_cand]))
            #if np.linalg.norm(self.anchor_kp) < self.min_curriculum_step:
            #    self.spd_kp_candidates.append(np.array([0, 0]))
            self.curriculum_id = np.random.randint(len(self.spd_kp_candidates))
            chosen_curriculum = self.spd_kp_candidates[self.curriculum_id]
            self.init_balance_pd = chosen_curriculum[0]
            self.end_balance_pd = chosen_curriculum[0]
            self.init_vel_pd = chosen_curriculum[1]
            self.end_vel_pd = chosen_curriculum[1]

        if self.init_push:
            qpos[0] = self.target_vel
        self.set_state(qpos, qvel)
        self.t = 0
        self.cur_step = 0
        self.stepwise_rewards = []

        self.init_pos = self.robot_skeleton.q[0]

        self.vel_cache = []

        self.avg_rew_weighting = []

        self.conseq_limit_pen = 0
        self.current_pd = self.init_balance_pd
        self.vel_enforce_kp = self.init_vel_pd

        self.contact_info = np.array([0, 0])

        return self._get_obs()

    def viewer_setup(self):
        if not self.disableViewer:
            #self.track_skeleton_id = 0
            self._get_viewer().scene.tb.trans[2] = -5.5

