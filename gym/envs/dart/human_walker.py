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
        self.action_scale = np.array([60.0, 100, 60, 100, 80, 60, 60, 100, 60, 100, 80, 60, 150, 150, 100, 5,15,5, 3, 5,15,5, 3])
        self.action_scale *= 1.5
        self.action_penalty_weight = np.array([1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        obs_dim = 57

        self.t = 0
        self.target_vel = 5.5
        self.init_tv = 5.5
        self.final_tv = 5.5
        self.tv_endtime = 0.04
        self.smooth_tv_change = True
        self.running_average_velocity = False
        self.running_avg_rew_only = True
        self.avg_rew_weighting = []
        self.vel_cache = []
        self.init_pos = 0

        self.rand_target_vel = False
        self.init_push = False
        self.enforce_target_vel = True
        self.hard_enforce = True
        self.treadmill = True
        self.treadmill_vel = -self.init_tv
        self.treadmill_init_tv = -5.5
        self.treadmill_final_tv = -5.5
        self.treadmill_tv_endtime = 0.04

        self.base_policy = None
        self.push_target = 'pelvis'

        self.constrain_dcontrol = 1.0
        self.previous_control = None

        self.energy_weight = 0.25

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
        self.anchor_kp = np.array([2000.0, 67.0])
        self.curriculum_step_size = 0.1  # 10%
        self.min_curriculum_step = 50  # include (0, 0) if distance between anchor point and origin is smaller than this value

        # state related
        self.contact_info = np.array([0, 0])
        self.include_additional_info = True
        if self.include_additional_info:
            obs_dim += len(self.contact_info)
        if self.running_average_velocity or self.smooth_tv_change:
            obs_dim += 1

        self.curriculum_id = 0
        self.spd_kp_candidates = None

        self.vel_reward_weight = 3.0

        self.init_qs = [
            np.array( [ -2.89238718e-01, -9.29257100e-02, -3.07078440e-02, -4.15327171e-02,
   5.83924834e-02, -4.71606015e-02,  1.48977210e-03, -5.44134982e-01,
  -1.81086954e-03, -3.82748081e-02, -1.77467849e-01,  4.85611185e-01,
  -1.02180162e-04,  4.94809618e-01,  5.51014204e-03,  5.59560832e-01,
  -7.15421935e-01,  2.93405226e-01,  4.02499653e-02,  5.73677976e-02,
  -5.27050402e-02, -7.42991206e-02,  9.99610887e-01,  2.01051844e-01,
  -2.10429626e-03,  3.83956893e-01, -6.40007362e-02, -2.00071433e-01,
   5.19437974e-01])]

        self.init_dqs = [
            np.array( [  1.01851403e+00, -1.23959308e+00, -2.08295926e-01,  1.48034588e+00,
  -1.75699809e-01, -3.25052801e-01,  8.52433124e-11,  3.62135638e+00,
  -1.34838363e-09, -3.87428667e-09, -3.76459443e+00, -5.65576426e-01,
  -6.15563156e-11, -2.51946361e+00,  7.51502416e-10,  1.50267900e+01,
   2.56216163e+00,  2.74806714e+00,  4.50750935e-01,  2.44003327e-01,
  -6.18723651e-01, -9.07565503e-01, -9.30654030e-02,  4.90717744e-10,
  -3.14452742e-10,  5.39206331e-01,  4.53671153e+00, -5.00044783e-10,
   9.49501133e-01])]

        if self.treadmill:
            dart_env.DartEnv.__init__(self, 'kima/kima_human_edited_treadmill.skel', 15, obs_dim, self.control_bounds,
                                      disableViewer=True, dt=0.002)
        else:
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

        self.robot_skeleton.set_self_collision_check(True)

        for i in range(0, len(self.dart_world.skeletons[0].bodynodes)):
            self.dart_world.skeletons[0].bodynodes[i].set_friction_coeff(20)
        for i in range(0, len(self.dart_world.skeletons[1].bodynodes)):
            self.dart_world.skeletons[1].bodynodes[i].set_friction_coeff(20)

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

        d = -self.Kd * (bn.dC[dof] - target_vel * 1.0)  # compensate for average velocity match

        qddot = invM * (-bn.C[dof] + p + d)
        tau = p + d - self.Kd * (qddot) * self.sim_dt

        return tau

    def do_simulation(self, tau, n_frames):
        for _ in range(n_frames):
            if self.constrain_2d:
                force = self._bodynode_spd(self.robot_skeleton.bodynode(self.push_target), self.current_pd, 2)
                self.robot_skeleton.bodynode(self.push_target).add_ext_force(np.array([0, 0, force]))

            if self.enforce_target_vel:# and not self.hard_enforce:
                tvel = self.target_vel
                if self.treadmill:
                    tvel += self.treadmill_vel
                force = self._bodynode_spd(self.robot_skeleton.bodynode(self.push_target), self.vel_enforce_kp, 0,
                                           tvel)
                self.robot_skeleton.bodynode(self.push_target).add_ext_force(np.array([force, 0, 0]))
            self.robot_skeleton.set_forces(tau)
            self.dart_world.step()
            s = self.state_vector()
            if not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all()):
                break

    def advance(self, clamped_control):
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
        # smoothly increase the target velocity
        if self.smooth_tv_change:
            self.target_vel = (np.min([self.t, self.tv_endtime]) / self.tv_endtime) * (
                    self.final_tv - self.init_tv) + self.init_tv
            self.treadmill_vel = (np.min([self.t, self.treadmill_tv_endtime]) / self.treadmill_tv_endtime) * (
                    self.treadmill_final_tv - self.treadmill_init_tv) + self.treadmill_init_tv


        posbefore = self.robot_skeleton.bodynode(self.push_target).com()[0]

        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
            if self.previous_control is not None:
                if clamped_control[i] > self.previous_control[i] + self.constrain_dcontrol:
                    clamped_control[i] = self.previous_control[i] + self.constrain_dcontrol
                elif clamped_control[i] < self.previous_control[i] - self.constrain_dcontrol:
                    clamped_control[i] = self.previous_control[i] - self.constrain_dcontrol
        self.advance(np.copy(clamped_control))

        posafter = self.robot_skeleton.bodynode(self.push_target).com()[0]
        height = self.robot_skeleton.bodynode('head').com()[1]
        side_deviation = self.robot_skeleton.bodynode('head').com()[2]
        angle = self.robot_skeleton.q[3]

        #pos_val = np.min([np.max([0, posafter]), self.pd_vary_end])
        self.current_pd = self.init_balance_pd #+ (
                                             #        self.end_balance_pd - self.init_balance_pd) / self.pd_vary_end * pos_val
        self.vel_enforce_kp = self.init_vel_pd #+ (self.end_vel_pd - self.init_vel_pd) / self.pd_vary_end * pos_val

        upward = np.array([0, 1, 0])
        upward_world = self.robot_skeleton.bodynode('head').to_world(
            np.array([0, 1, 0])) - self.robot_skeleton.bodynode('head').to_world(np.array([0, 0, 0]))
        upward_world /= np.linalg.norm(upward_world)
        ang_cos_uwd = np.dot(upward, upward_world)
        ang_cos_uwd = np.arccos(ang_cos_uwd)

        forward = np.array([1, 0, 0])
        forward_world = self.robot_skeleton.bodynode('head').to_world(
            np.array([1, 0, 0])) - self.robot_skeleton.bodynode('head').to_world(np.array([0, 0, 0]))
        forward_world /= np.linalg.norm(forward_world)
        ang_cos_fwd = np.dot(forward, forward_world)
        ang_cos_fwd = np.arccos(ang_cos_fwd)

        lateral = np.array([0, 0, 1])
        lateral_world = self.robot_skeleton.bodynode('head').to_world(
            np.array([0, 0, 1])) - self.robot_skeleton.bodynode('head').to_world(np.array([0, 0, 0]))
        lateral_world /= np.linalg.norm(lateral_world)
        ang_cos_ltl = np.dot(lateral, lateral_world)
        ang_cos_ltl = np.arccos(ang_cos_ltl)

        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        self.contact_info = np.array([0, 0])
        for contact in contacts:
            if contact.skel_id1 == contact.skel_id2:
                self_colliding = True
            if contact.skel_id1 + contact.skel_id2 == 1:
                if contact.bodynode1 == self.robot_skeleton.bodynode(
                        'l-foot') or contact.bodynode2 == self.robot_skeleton.bodynode('l-foot'):
                    self.contact_info[0] = 1
                    total_force_mag += np.linalg.norm(contact.force)
                if contact.bodynode1 == self.robot_skeleton.bodynode(
                        'r-foot') or contact.bodynode2 == self.robot_skeleton.bodynode('r-foot'):
                    self.contact_info[1] = 1
                    total_force_mag += np.linalg.norm(contact.force)

        alive_bonus = 4.0
        vel = (posafter - posbefore) / self.dt
        vel_rew = 0.0
        self.vel_cache.append(vel)
        if self.running_average_velocity or self.running_avg_rew_only:
            if self.t < self.tv_endtime:
                self.avg_rew_weighting.append(0.3)
            else:
                self.avg_rew_weighting.append(1)


        vel_rew_scale = 1.0
        if len(self.vel_cache) > int(1.0/self.dt) and (self.running_average_velocity or self.running_avg_rew_only):
            self.vel_cache.pop(0)
            self.avg_rew_weighting.pop(0)
        else:
            vel_rew_scale = 0.1#np.min([len(self.vel_cache) * self.dt, 1.0])

        if not self.treadmill:
            if self.reference_trajectory is not None:
                vel_rew = - np.exp(3.0*(np.abs(self.reference_trajectory[self.cur_step][0] - self.robot_skeleton.com()[0]**2)))
            elif self.running_average_velocity or self.running_avg_rew_only:
                append_vel = np.zeros(int(1.0/self.dt) - len(self.vel_cache))
                vel_rew = -3.0 * np.mean(np.append(np.array(self.avg_rew_weighting) * np.abs(np.array(self.vel_cache) - self.target_vel), append_vel))
            else:
                vel_diff = np.abs(self.target_vel - vel)
        else:
            if self.running_average_velocity or self.running_avg_rew_only:
                append_vel = np.ones(int(1.0/self.dt) - len(self.vel_cache)) * (self.target_vel + self.treadmill_vel)
                vel_rew = -3.0 *vel_rew_scale* (np.abs(self.target_vel + self.treadmill_vel - np.mean(np.append(self.vel_cache, append_vel))))
            else:
                vel_rew = -3.0 * (np.abs(self.target_vel + self.treadmill_vel - vel))
        # vel_rew *= 0
        # action_pen = 5e-1 * (np.square(a)* actuator_pen_multiplier).sum()
        action_pen = self.energy_weight * np.abs(a * self.action_penalty_weight).sum()
        # action_pen += 0.02 * np.sum(np.abs(a* self.robot_skeleton.dq[6:]))
        deviation_pen = 3 * abs(side_deviation)

        rot_pen = 1.0 * (abs(ang_cos_uwd)) + 0.1 * (abs(ang_cos_fwd)) + 1.5 * (abs(ang_cos_ltl))
        # penalize bending of spine
        spine_pen = 1.0 * np.sum(np.abs(self.robot_skeleton.q[[18, 19]])) + 0.01 * np.abs(self.robot_skeleton.q[20])

        spine_pen += 0.05 * np.sum(np.abs(self.robot_skeleton.q[[8, 14]]))
        reward = vel_rew + alive_bonus - action_pen - deviation_pen - rot_pen - spine_pen
        pos_rew = vel_rew + alive_bonus - deviation_pen - rot_pen - spine_pen
        neg_pen = - action_pen
        self.t += self.dt

        self.cur_step += 1

        s = self.state_vector()

        height_in_range = (height - self.init_height > -0.4) and (height - self.init_height < 1.0)
        ang_in_range = (abs(ang_cos_uwd) < 1.0) and (abs(ang_cos_fwd) < 2.0) and np.abs(angle) < 1.3 and np.abs(
            self.robot_skeleton.q[5]) < 0.4
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height - self.init_height > -0.3) and (height - self.init_height < 1.0) and (
                    abs(ang_cos_uwd) < 1.0) and (abs(ang_cos_fwd) < 1.3)
                    and np.abs(angle) < 1.3 and np.abs(self.robot_skeleton.q[5]) < 0.4 and np.abs(side_deviation) < 0.9)

        self.stepwise_rewards.append(reward)

        self.previous_control = clamped_control

        # if self.conseq_limit_pen > 20:
        #    done = True

        #    reward = 0
        # if done:
        #    print(height_in_range, ang_in_range, np.abs(side_deviation) < 0.9)

        ob = self._get_obs()

        broke_sim = False
        if not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all()):
            broke_sim = True
        final_reward = 0.0
        return ob, reward, done, {'broke_sim': broke_sim, 'vel_rew': vel_rew, 'action_pen': action_pen,
                                  'deviation_pen': deviation_pen, 'curriculum_id': self.curriculum_id,
                                  'curriculum_candidates': self.spd_kp_candidates, 'done_return': done,
                                  'dyn_model_id': 0, 'state_index': 0, 'com': self.robot_skeleton.com(),
                                  'pos_rew': pos_rew, 'neg_pen': neg_pen}

    def _get_obs(self):
        state = np.concatenate([
            self.robot_skeleton.q[1:],
            self.robot_skeleton.dq,
        ])

        if self.include_additional_info:
            state = np.concatenate([state, self.contact_info])

        if self.rand_target_vel or self.smooth_tv_change:
            state = np.concatenate([state, [self.target_vel]])

        if self.running_average_velocity:
            state = np.concatenate([state, [(self.robot_skeleton.q[0] - self.init_pos) / self.t]])

        return state

    def reset_model(self):
        self.dart_world.reset()

        init_q = self.robot_skeleton.q
        init_dq = self.robot_skeleton.dq
        if len(self.init_dqs) > 0:
            init_pid = np.random.randint(len(self.init_qs))
            init_q = self.init_qs[init_pid]
            init_dq = self.init_dqs[init_pid]

        qpos = init_q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = init_dq + self.np_random.uniform(low=-.05, high=.05, size=self.robot_skeleton.ndofs)

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
            qvel[0] = self.target_vel
        self.set_state(qpos, qvel)
        self.t = self.dt
        self.cur_step = 0
        self.stepwise_rewards = []

        self.init_pos = self.robot_skeleton.q[0]

        self.conseq_limit_pen = 0
        self.current_pd = self.init_balance_pd
        self.vel_enforce_kp = self.init_vel_pd

        self.contact_info = np.array([0, 0])

        self.previous_control = None

        self.init_height = self.robot_skeleton.bodynode('head').C[1]
        self.moving_bin = None

        self.vel_cache = []

        self.avg_rew_weighting = []

        self.moving_bin = None
        self.reference_trajectory = None
        return self._get_obs()

    def viewer_setup(self):
        if not self.disableViewer:
            # self.track_skeleton_id = 0
            self._get_viewer().scene.tb.trans[2] = -5.5
