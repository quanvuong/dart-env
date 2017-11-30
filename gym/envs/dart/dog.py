__author__ = 'yuwenhao'


import numpy as np
from gym import utils
from gym.envs.dart import dart_env

from gym.envs.dart.parameter_managers import *


class DartDogEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0] * 65, [-1.0] * 65])
        self.action_scale = np.array(
            [100, 10, 30, 100, 10, 30, # femur
              60, 50, 40, 60, 50, 40, 60, 50, 40, 60, 50, 40, 60, 50, 40, # back
             100, 10, 30, 100 ,10, 30, # front legs
              60, 60, 40, 40, 20, 20, 10, 10,
             50, 40, 30, 50, 40, 30, 50, 40, 30, 50, 40, 30, # neck and head
             60, 60, 50, 50, 40, 40, # rear leag
             5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 # tails
             ])

        obs_dim = 136

        self.t = 0
        self.target_vel = 1.0
        self.init_tv = 2.5
        self.final_tv = 2.5
        self.tv_endtime = 4.0
        self.smooth_tv_change = False
        self.init_pos = 0

        self.enforce_target_vel = True

        self.cur_step = 0
        self.stepwise_rewards = []
        self.constrain_2d = True
        self.init_balance_pd = 4000.0
        self.init_vel_pd = 2000.0
        self.end_balance_pd = 4000.0
        self.end_vel_pd = 2000.0

        self.pd_vary_end = self.target_vel * 6.0
        self.current_pd = self.init_balance_pd
        self.vel_enforce_kp = self.init_vel_pd

        self.local_spd_curriculum = False
        self.anchor_kp = np.array([12, 11])

        # state related
        self.contact_info = np.array([0, 0, 0, 0])
        self.include_additional_info = True
        if self.include_additional_info:
            obs_dim += len(self.contact_info)

        self.curriculum_id = 0
        self.spd_kp_candidates = None

        self.vel_reward_weight = 0.0

        dart_env.DartEnv.__init__(self, 'dog/dog.skel', 15, obs_dim, self.control_bounds,
                                  disableViewer=True, dt=0.002)

        for i in range(0, len(self.dart_world.skeletons[0].bodynodes)):
            self.dart_world.skeletons[0].bodynodes[i].set_friction_coeff(20)
        for i in range(0, len(self.dart_world.skeletons[1].bodynodes)):
            self.dart_world.skeletons[1].bodynodes[i].set_friction_coeff(20)

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
                force = self._bodynode_spd(self.robot_skeleton.bodynode('pelvis'), self.current_pd, 2)
                self.robot_skeleton.bodynode('pelvis').add_ext_force(np.array([0, 0, force]))

            if self.enforce_target_vel:
                force = self._bodynode_spd(self.robot_skeleton.bodynode('pelvis'), self.vel_enforce_kp, 0,
                                           self.target_vel)
                self.robot_skeleton.bodynode('pelvis').add_ext_force(np.array([force, 0, 0]))
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
            self.target_vel = (np.min([self.t, self.tv_endtime]) / self.tv_endtime) * (
            self.final_tv - self.init_tv) + self.init_tv

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
        self.contact_info = np.array([0, 0, 0, 0])
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()
            if contact.skel_id1 == contact.skel_id2:
                self_colliding = True
            if contact.skel_id1 + contact.skel_id2 == 1:
                if contact.bodynode1 == self.robot_skeleton.bodynode(
                        'lFingers') or contact.bodynode2 == self.robot_skeleton.bodynode('lFingers'):
                    self.contact_info[0] = 1
                if contact.bodynode1 == self.robot_skeleton.bodynode(
                        'rFingers') or contact.bodynode2 == self.robot_skeleton.bodynode('rFingers'):
                    self.contact_info[1] = 1
                if contact.bodynode1 == self.robot_skeleton.bodynode(
                        'lToes') or contact.bodynode2 == self.robot_skeleton.bodynode('lToes'):
                    self.contact_info[2] = 1
                if contact.bodynode1 == self.robot_skeleton.bodynode(
                        'rToes') or contact.bodynode2 == self.robot_skeleton.bodynode('rToes'):
                    self.contact_info[3] = 1

        alive_bonus = 4.0
        vel = (posafter - posbefore) / self.dt

        vel_diff = np.abs(self.target_vel - vel)
        vel_rew = - 0.2 * self.vel_reward_weight * vel_diff
        if vel_diff > 0.2 * np.abs(self.target_vel):
            vel_rew += - self.vel_reward_weight * (vel_diff - 0.2 * np.abs(self.target_vel))

        action_pen = 0.1 * np.abs(a).sum()
        deviation_pen = 3 * abs(side_deviation)

        rot_pen = 1.0 * (abs(ang_cos_uwd)) + 1.0 * (abs(ang_cos_fwd))  # + 0.5 * (abs(ang_cos_ltl))
        # penalize bending of spine
        spine_pen = 0.1 * np.sum(np.abs(self.robot_skeleton.q[12:29]))
        #spine_pen += 0.05 * np.sum(np.abs(self.robot_skeleton.q[[8, 14]]))
        reward = vel_rew + alive_bonus - action_pen - deviation_pen - rot_pen - spine_pen

        self.t += self.dt
        self.cur_step += 1

        s = self.state_vector()

        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height - self.init_height > -0.2) and (height - self.init_height < 1.0) and (
                    abs(ang_cos_uwd) < 1.0) and (abs(ang_cos_fwd) < 2.0)
                    and np.abs(angle) < 1.3 and np.abs(self.robot_skeleton.q[5]) < 0.4 and np.abs(side_deviation) < 0.9)

        self.stepwise_rewards.append(reward)

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

        return state

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.05, high=.05, size=self.robot_skeleton.ndofs)


        if self.local_spd_curriculum:
            self.spd_kp_candidates = [self.anchor_kp]
            self.curriculum_id = np.random.randint(len(self.spd_kp_candidates))
            chosen_curriculum = self.spd_kp_candidates[self.curriculum_id]
            self.init_balance_pd = chosen_curriculum[0]
            self.end_balance_pd = chosen_curriculum[0]
            self.init_vel_pd = chosen_curriculum[1]
            self.end_vel_pd = chosen_curriculum[1]

        self.set_state(qpos, qvel)
        self.t = self.dt
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
            self._get_viewer().scene.tb.trans[2] = -2.5