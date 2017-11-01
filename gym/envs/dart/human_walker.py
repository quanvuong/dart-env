__author__ = 'yuwenhao'

import numpy as np
from gym import utils
from gym.envs.dart import dart_env
import joblib
import os

from gym.envs.dart.parameter_managers import *
import time

import pydart2 as pydart


class DartHumanWalkerEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0] * 23, [-1.0] * 23])
        self.action_scale = np.array([200, 200, 200, 100, 60, 60, 200, 200, 200, 100, 60, 60, 200, 200, 200, 10,80,10, 50, 10,80,10, 50])
        obs_dim = 57

        self.t = 0
        self.target_vel = 1.3
        self.init_tv = 1.0
        self.final_tv = 3.0
        self.tv_endtime = 2.0
        self.smooth_tv_change = False

        self.rand_target_vel = False
        self.init_push = False
        self.enforce_target_vel = True
        self.hard_enforce = False
        self.treadmill = False
        self.treadmill_vel = -1.0
        self.base_policy = None

        self.cur_step = 0
        self.stepwise_rewards = []
        self.conseq_limit_pen = 0  # number of steps lying on the wall
        self.constrain_2d = True

        self.init_balance_pd = 6000.0
        self.init_vel_pd = 3000.0
        self.end_balance_pd = 6000.0
        self.end_vel_pd = 3000.0

        self.pd_vary_end = self.target_vel * 6.0
        self.current_pd = self.init_balance_pd
        self.vel_enforce_kp = self.init_vel_pd

        self.local_spd_curriculum = True
        self.anchor_kp = np.array([2000, 2000])
        self.curriculum_step_size = 0.1  # 10%
        self.min_curriculum_step = 50  # include (0, 0) if distance between anchor point and origin is smaller than this value

        # state related
        self.contact_info = np.array([0, 0])
        self.include_additional_info = True
        if self.include_additional_info:
            obs_dim += len(self.contact_info)
        if self.rand_target_vel:
            obs_dim += 1

        self.curriculum_id = 0
        self.spd_kp_candidates = None

        dart_env.DartEnv.__init__(self, 'kima/kima_human_edited.skel', 15, obs_dim, self.control_bounds,
                                      disableViewer=True, dt=0.002)

        # add human joint limit
        '''skel = self.robot_skeleton
        world = self.dart_world
        leftarmConstraint = pydart.constraints.HumanArmJointLimitConstraint(skel.joint('j_bicep_left'),
                                                                            skel.joint('j_forearm_left'), False)
        rightarmConstraint = pydart.constraints.HumanArmJointLimitConstraint(skel.joint('j_bicep_right'),
                                                                             skel.joint('j_forearm_right'), True)
        leftlegConstraint = pydart.constraints.HumanLegJointLimitConstraint(skel.joint('j_thigh_left'),
                                                                            skel.joint('j_shin_left'),
                                                                            skel.joint('j_heel_left'), False)
        rightlegConstraint = pydart.constraints.HumanLegJointLimitConstraint(skel.joint('j_thigh_right'),
                                                                             skel.joint('j_shin_right'),
                                                                             skel.joint('j_heel_right'), True)
        leftarmConstraint.add_to_world(world)
        rightarmConstraint.add_to_world(world)
        leftlegConstraint.add_to_world(world)
        rightlegConstraint.add_to_world(world)'''

        self.robot_skeleton.set_self_collision_check(False)

        for i in range(0, len(self.dart_world.skeletons[0].bodynodes)):
            self.dart_world.skeletons[0].bodynodes[i].set_friction_coeff(2)
        for i in range(0, len(self.dart_world.skeletons[1].bodynodes)):
            self.dart_world.skeletons[1].bodynodes[i].set_friction_coeff(2)

        # self.dart_world.set_collision_detector(3)

        self.sim_dt = self.dt / self.frame_skip

        utils.EzPickle.__init__(self)

    # only 1d
    def _spd(self, target_q, id, kp, target_dq=0.0):
        self.Kp = kp
        self.Kd = kp * self.sim_dt
        if target_dq > 0:
            self.Kd = self.Kp
            self.Kp *= 0

        invM = 1.0 / (self.robot_skeleton.M[id][id] + self.Kd * self.sim_dt)
        if target_dq == 0:
            p = -self.Kp * (self.robot_skeleton.q[id] + self.robot_skeleton.dq[id] * self.sim_dt - target_q[id])
        else:
            p = 0
        d = -self.Kd * (self.robot_skeleton.dq[id] - target_dq)
        qddot = invM * (-self.robot_skeleton.c[id] + p + d)
        tau = p + d - self.Kd * (qddot) * self.sim_dt

        return tau

    def _bodynode_spd(self, bn, kp, dof, target_vel=0.0):
        self.Kp = kp
        self.Kd = kp * self.sim_dt
        if target_vel > 0:
            self.Kd = self.Kp
            self.Kp *= 0

        invM = 1.0 / (bn.mass() + self.Kd * self.sim_dt)
        p = -self.Kp * (bn.C[dof] + bn.dC[dof] * self.sim_dt)
        d = -self.Kd * (bn.dC[dof] - target_vel)
        qddot = invM * (-bn.C[dof] + p + d)
        tau = p + d - self.Kd * (qddot) * self.sim_dt

        return tau

    def do_simulation(self, tau, n_frames):
        for _ in range(n_frames):
            if self.constrain_2d:
                #force = self._bodynode_spd(self.robot_skeleton.bodynode('thorax'), self.current_pd, 2)
                #self.robot_skeleton.bodynode('thorax').add_ext_force(np.array([0, 0, force]))
                force = self._bodynode_spd(self.robot_skeleton.bodynode('pelvis'), self.current_pd, 2)
                self.robot_skeleton.bodynode('pelvis').add_ext_force(np.array([0, 0, force]))
                # tq = self.robot_skeleton.q
                # tq[2] = 0
                # if _ % 5 == 0:
                #    spdtau = self._spd(tq, 2, self.current_pd)
                # tau[2] = spdtau

            if self.enforce_target_vel and not self.hard_enforce:
                #force = self._bodynode_spd(self.robot_skeleton.bodynode('thorax'), self.vel_enforce_kp, 0, self.target_vel)
                #self.robot_skeleton.bodynode('thorax').add_ext_force(np.array([force, 0, 0]))
                force = self._bodynode_spd(self.robot_skeleton.bodynode('pelvis'), self.vel_enforce_kp, 0,
                                           self.target_vel)

                self.robot_skeleton.bodynode('pelvis').add_ext_force(np.array([force, 0, 0]))
                '''tq2 = self.robot_skeleton.q
                tq2[0] = pos_before + self.dt * self.target_vel
                if _ % 5 == 0:
                    spdtau2 = self._spd(tq2, 0, self.vel_enforce_kp, self.target_vel)
                tau[0] = spdtau2'''
            self.robot_skeleton.set_forces(tau)
            self.dart_world.step()
            s = self.state_vector()
            if not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all()):
                break

    def advance(self, a):
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
                current_dq_tread[0] = self.treadmill_vel  # * np.min([self.t/4.0, 1.0])
                self.dart_world.skeletons[0].dq = current_dq_tread
            elif self.hard_enforce:
                current_dq = self.robot_skeleton.dq
                current_dq[0] = self.target_vel
                self.robot_skeleton.dq = current_dq
        self.do_simulation(tau, self.frame_skip)

    def _step(self, a):
        posbefore = self.robot_skeleton.bodynode('pelvis').com()[0]
        self.advance(np.copy(a))

        posafter = self.robot_skeleton.bodynode('pelvis').com()[0]
        height = self.robot_skeleton.bodynode('head').com()[1]
        side_deviation = self.robot_skeleton.bodynode('head').com()[2]
        angle = self.robot_skeleton.q[3]

        pos_val = np.min([np.max([0, posafter]), self.pd_vary_end])
        self.current_pd = self.init_balance_pd + (
                                                 self.end_balance_pd - self.init_balance_pd) / self.pd_vary_end * pos_val
        self.vel_enforce_kp = self.init_vel_pd + (self.end_vel_pd - self.init_vel_pd) / self.pd_vary_end * pos_val
        # print(self.current_pd)
        # smoothly increase the target velocity
        if self.smooth_tv_change:
            self.target_vel = (np.min([self.t, self.tv_endtime]) / self.tv_endtime) * (self.final_tv - self.init_tv) + self.init_tv

        upward = np.array([0, 1, 0])
        upward_world = self.robot_skeleton.bodynode('head').to_world(np.array([0, 1, 0])) - self.robot_skeleton.bodynode('head').to_world(np.array([0, 0, 0]))
        upward_world /= np.linalg.norm(upward_world)
        ang_cos_uwd = np.dot(upward, upward_world)
        ang_cos_uwd = np.arccos(ang_cos_uwd)

        forward = np.array([1, 0, 0])
        forward_world = self.robot_skeleton.bodynode('head').to_world(np.array([1, 0, 0])) - self.robot_skeleton.bodynode('head').to_world(np.array([0, 0, 0]))
        forward_world /= np.linalg.norm(forward_world)
        ang_cos_fwd = np.dot(forward, forward_world)
        ang_cos_fwd = np.arccos(ang_cos_fwd)

        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        self.contact_info = np.array([0, 0])
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()
            if contact.skel_id1 == contact.skel_id2:
                self_colliding = True
            if contact.skel_id1 + contact.skel_id2 == 1:
                if contact.bodynode1 == self.robot_skeleton.bodynode('l-foot') or contact.bodynode2 == self.robot_skeleton.bodynode('l-foot'):
                    self.contact_info[0] = 1
                if contact.bodynode1 == self.robot_skeleton.bodynode('r-foot') or contact.bodynode2 == self.robot_skeleton.bodynode('r-foot'):
                    self.contact_info[1] = 1


        alive_bonus = 4.0
        vel = (posafter - posbefore) / self.dt
        if not self.treadmill:
            vel_rew = 3 * ( - np.abs(self.target_vel - vel))  # 1.0 * (posafter - posbefore) / self.dt
        else:
            vel_rew = 2 * (self.target_vel - np.abs(self.target_vel + self.treadmill_vel - vel))

        # action_pen = 5e-1 * (np.square(a)* actuator_pen_multiplier).sum()
        action_pen = 0.4 * np.abs(a).sum()
        # action_pen = 5e-3 * np.sum(np.square(a)* self.robot_skeleton.dq[6:]* actuator_pen_multiplier)
        deviation_pen = 3 * abs(side_deviation)

        rot_pen = 3.0 * (abs(ang_cos_uwd))

        # penalize bending of spine
        spine_pen = 0.3 * np.sum(np.abs(self.robot_skeleton.q[[18, 19]])) + 0.01 * np.abs(self.robot_skeleton.q[20])
        reward = vel_rew + alive_bonus - action_pen - deviation_pen - rot_pen - spine_pen


        self.t += self.dt
        self.cur_step += 1

        s = self.state_vector()

        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height-self.init_height > -0.2) and (height - self.init_height < 1.0) and (abs(ang_cos_uwd) < 1.0) and (abs(ang_cos_fwd) < 2.0)
                    and np.abs(angle) < 1.3 and np.abs(self.robot_skeleton.q[5]) < 0.4 and np.abs(side_deviation) < 0.9)

        self.stepwise_rewards.append(reward)

        # if self.conseq_limit_pen > 20:
        #    done = True

        #if done:
        #    reward = 0

        ob = self._get_obs()

        broke_sim = False
        if not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all()):
            broke_sim = True

        return ob, reward, done, {'broke_sim': broke_sim, 'vel_rew': vel_rew, 'action_pen': action_pen,
                                  'deviation_pen': deviation_pen, 'curriculum_id': self.curriculum_id,
                                  'curriculum_candidates': self.spd_kp_candidates, 'done_return': done,
                                  'dyn_model_id': 0, 'state_index': 0}

    def _get_obs(self):
        state = np.concatenate([
            self.robot_skeleton.q[1:],
            self.robot_skeleton.dq,
        ])

        if self.include_additional_info:
            state = np.concatenate([state, self.contact_info])

        if self.rand_target_vel:
            state = np.concatenate([state, [self.target_vel]])

        return state

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.05, high=.05, size=self.robot_skeleton.ndofs)

        if self.rand_target_vel:
            self.target_vel = np.random.uniform(0.8, 2.5)

        if self.local_spd_curriculum:
            self.spd_kp_candidates = [self.anchor_kp]
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

        self.conseq_limit_pen = 0
        self.current_pd = self.init_balance_pd
        self.vel_enforce_kp = self.init_vel_pd

        self.contact_info = np.array([0, 0])

        self.init_height = self.robot_skeleton.bodynode('head').C[1]

        return self._get_obs()

    def viewer_setup(self):
        if not self.disableViewer:
            self._get_viewer().scene.tb.trans[2] = -5.5
