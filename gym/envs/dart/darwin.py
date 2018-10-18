__author__ = 'yuwenhao'

import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from scipy.optimize import minimize
from pydart2.collision_result import CollisionResult
from pydart2.bodynode import BodyNode
import pydart2.pydart2_api as papi
import random
from random import randrange
import pickle
import copy


class DartDarwinTrajEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):

        obs_dim = 48

        self.control_bounds = np.array([np.ones(20, ), np.ones(20, )])
        # LOWER CONTROL BOUNDS
        # LEFT HANDS

        self.control_bounds[0, 0] = (2518 - 2048) * 0.088 * np.pi / 180  # - np.pi/8
        self.control_bounds[0, 1] = (2248 - 2048) * 0.088 * np.pi / 180  # - np.pi/2
        self.control_bounds[0, 2] = (1712 - 2048) * 0.088 * np.pi / 180  # - np.pi/4
        # RIGHT HANDS
        self.control_bounds[0, 3] = (1498 - 2048) * 0.088 * np.pi / 180  # - np.pi/8
        self.control_bounds[0, 4] = (1845 - 2048) * 0.088 * np.pi / 180  # - np.pi/8
        self.control_bounds[0, 5] = (2381 - 2048) * 0.088 * np.pi / 180  # - np.pi/4
        # head
        self.control_bounds[0, 6] = -0.01
        self.control_bounds[0, 7] = -0.01
        # LEFT LEG
        self.control_bounds[0, 8] = -np.pi / 10
        self.control_bounds[0, 9] = -0.20
        self.control_bounds[0, 10] = -0.25
        self.control_bounds[0, 11] = -0.60
        self.control_bounds[0, 12] = -0.40
        self.control_bounds[0, 13] = -0.40

        self.control_bounds[0, 14] = -np.pi / 10
        self.control_bounds[0, 15] = -0.15
        self.control_bounds[0, 16] = -0.40
        self.control_bounds[0, 17] = -0.40
        self.control_bounds[0, 18] = -0.25  # (2389-2048)*(np.pi/180)*0.088 - 0.050
        self.control_bounds[0, 19] = -0.30  # (2057-2048)*(np.pi/180)*0.088 - 0.050

        ### UPPER BOUNDS
        self.control_bounds[1, 0] = (2518 - 2048) * 0.088 * np.pi / 180  # + np.pi/8
        self.control_bounds[1, 1] = (2248 - 2048) * 0.088 * np.pi / 180  # + np.pi/8
        self.control_bounds[1, 2] = (1712 - 2048) * 0.088 * np.pi / 180  # + np.pi/4
        # RIGHT HANDS
        self.control_bounds[1, 3] = (1498 - 2048) * 0.088 * np.pi / 180  # + np.pi/8
        self.control_bounds[1, 4] = (1845 - 2048) * 0.088 * np.pi / 180  # + np.pi/2
        self.control_bounds[1, 5] = (2381 - 2048) * 0.088 * np.pi / 180  # + np.pi/4
        # head
        self.control_bounds[1, 6] = 0.01
        self.control_bounds[1, 7] = 0.01
        # LEFT LEG
        self.control_bounds[1, 8] = np.pi / 10
        self.control_bounds[1, 9] = 0.15  # (2052-2048)*(np.pi/180)*0.088 + np.pi/100
        self.control_bounds[1, 10] = 0.40  # 0.68 + 0.076
        self.control_bounds[1, 11] = 0.40  # -0.93 + 0.076
        self.control_bounds[1, 12] = 0.25  # (1707-2048)*(np.pi/180)*0.088 + 0.050
        self.control_bounds[1, 13] = 0.30  # (2039-2048)*(np.pi/180)*0.088 + 0.050

        self.control_bounds[1, 14] = np.pi / 10
        self.control_bounds[1, 15] = 0.20  # (2044-2048)*(np.pi/180)*0.088 + np.pi/25
        self.control_bounds[1, 16] = 0.25  # -0.68 + 0.076
        self.control_bounds[1, 17] = 0.60  # 0.93 + 0.076
        self.control_bounds[1, 18] = 0.40  # (2389-2048)*(np.pi/180)*0.088 +0.050
        self.control_bounds[1, 19] = 0.40  # (2057-2048)*(np.pi/180)*0.088 + 0.050

        self.control_limits_low = np.copy(self.control_bounds[0, :])
        self.control_limits_high = np.copy(self.control_bounds[1, :])
        self.control_bounds[0] = [-1.0] * 20
        self.control_bounds[1] = [1.0] * 20

        self.target_vel = 0.0
        self.init_tv = 0.0
        self.final_tv = 0.45
        self.tv_endtime = 0.001
        self.smooth_tv_change = True
        self.avg_rew_weighting = []
        self.vel_cache = []
        self.target_vel_cache = []

        self.alive_bonus = 5.0
        self.energy_weight = 0.005
        self.work_weight = 0.01
        self.vel_reward_weight = 10.0
        self.pose_weight = 0.2

        self.assist_timeout = 10.0
        self.assist_schedule = [[0.0, [20000, 20000]], [3.0, [15000, 15000]], [6.0, [11250.0, 11250.0]]]
        self.init_balance_pd = 20000.0
        self.init_vel_pd = 20000.0

        self.cur_step = 0

        self.torqueLimits = 3.5


        self.t = 0
        # self.dt = 0.002
        self.itr = 0
        self.sol = 0
        self.rankleFlag = False
        self.rhandFlag = False
        self.lhandFlag = False
        self.lankleFlag = False
        self.preverror = np.zeros(26, )
        self.edot = np.zeros(26, )
        self.target = np.zeros(26, )
        self.ndofs = np.zeros(26, )  # self.robot_skeleton.ndofs
        self.tau = np.zeros(26, )
        self.init = np.zeros(26, )
        self.sum = 0
        self.count = 0
        self.dumpTorques = False
        self.dumpActions = False
        self.f1 = np.array([0.])
        self.f2 = np.array([0.])

        self.include_obs_history = 1
        self.include_act_history = 0
        obs_dim *= self.include_obs_history
        obs_dim += len(self.control_bounds[0]) * self.include_act_history
        obs_perm_base = np.array(
            [-3, -4, -5, -0.0001, -1, -2, 6, 7, -14, -15, -16, -17, -18, -19, -8, -9, -10, -11, -12, -13,
             -23, -24, -25, -20, -21, -22, 26, 27, -34, -35, -36, -37, -38, -39, -28, -29, -30, -31, -32, -33,
             -40, 41, 42, -43, 44, -45, 46, -47])
        act_perm_base = np.array(
            [-3, -4, -5, -0.0001, -1, -2, 6, 7, -14, -15, -16, -17, -18, -19, -8, -9, -10, -11, -12, -13])
        self.obs_perm = np.copy(obs_perm_base)

        for i in range(self.include_obs_history - 1):
            self.obs_perm = np.concatenate(
                [self.obs_perm, np.sign(obs_perm_base) * (np.abs(obs_perm_base) + len(self.obs_perm))])
        for i in range(self.include_act_history):
            self.obs_perm = np.concatenate(
                [self.obs_perm, np.sign(act_perm_base) * (np.abs(act_perm_base) + len(self.obs_perm))])
        self.act_perm = np.copy(act_perm_base)

        dart_env.DartEnv.__init__(self, ['darwinmodel/ground1.urdf', 'darwinmodel/darwin_nocollision.URDF', 'darwinmodel/robotis_op2.urdf'], 15, obs_dim,
                                  self.control_bounds, disableViewer=True)

        self.dart_world.set_gravity([0, 0, -9.81])

        self.dupSkel = self.dart_world.skeletons[1]
        self.dupSkel.set_mobile(False)

        self.dart_world.set_collision_detector(0)

        self.robot_skeleton.set_self_collision_check(False)

        self.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(2.0)
        for bn in self.robot_skeleton.bodynodes:
            bn.set_friction_coeff(2.0)

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

    def advance(self, a):
        clamped_control = np.array(a)

        for i in range(len(clamped_control)):

            if clamped_control[i] > self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
            if clamped_control[i] < self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]

        clamped_control = (clamped_control + 1.0) * 0.5 * (
                self.control_limits_high - self.control_limits_low) + self.control_limits_low

        self.target[6:] = self.init_q[
                          6:] + clamped_control  # self.ref_traj[self.count,:] + (self.init[6:])# + #*self.action_scale# + self.ref_trajectory_right[self.count_right,6:]# +

        self.dupSkel.set_positions(self.target)
        self.dupSkel.set_velocities(self.target*0)

        for i in range(self.frame_skip):

            self.tau[6:] = self.PID()
            self.tau[0:6] *= 0.0

            if self.dumpTorques:
                with open("torques.txt", "ab") as fp:
                    np.savetxt(fp, np.array([self.tau]), fmt='%1.5f')

            if self.dumpActions:
                with open("targets_from_net.txt", 'ab') as fp:
                    np.savetxt(fp, np.array([[self.target[6], self.robot_skeleton.q[6]]]), fmt='%1.5f')

            if self.t < self.assist_timeout:
                force = self._bodynode_spd(self.robot_skeleton.bodynode('MP_NECK'), self.current_pd, 1)
                self.robot_skeleton.bodynode('MP_NECK').add_ext_force(np.array([0, force, 0]))

                force = self._bodynode_spd(self.robot_skeleton.bodynode('MP_NECK'), self.vel_enforce_kp, 0, self.target_vel)
                self.robot_skeleton.bodynode('MP_NECK').add_ext_force(np.array([force, 0, 0]))

            self.robot_skeleton.set_forces(self.tau)
            self.dart_world.step()


    def PID(self):
        # print("#########################################################################3")

        self.kp = [2.1, 1.79, 4.93,
                   2.0, 2.02, 1.98,
                   2.2, 2.06,
                   148, 152, 150, 136, 153, 102,
                   151, 151.4, 150.45, 151.36, 154, 105.2]
        self.kd = [0.21, 0.23, 0.22,
                   0.25, 0.21, 0.26,
                   0.28, 0.213
            , 0.192, 0.198, 0.22, 0.199, 0.02, 0.01,
                   0.53, 0.27, 0.21, 0.205, 0.022, 0.056]

        self.kp = [item for item in self.kp]
        self.kd = [item for item in self.kd]

        q = self.robot_skeleton.q
        qdot = self.robot_skeleton.dq
        tau = np.zeros(26, )
        for i in range(6, 26):
            # print(q.shape)
            #self.edot[i] = ((q[i] - self.target[i]) -
            #                self.preverror[i]) / self.dt
            tau[i] = -self.kp[i - 6] * \
                     (q[i] - self.target[i]) - \
                     self.kd[i - 6] * qdot[i]
            #self.preverror[i] = (q[i] - self.target[i])

        torqs = self.ClampTorques(tau)

        return torqs[6:]
        # return tau[6:]

    def ClampTorques(self, torques):
        torqueLimits = self.torqueLimits

        for i in range(6, 26):
            if torques[i] > torqueLimits:  #
                torques[i] = torqueLimits
            if torques[i] < -torqueLimits:
                torques[i] = -torqueLimits

        return torques

    def step(self, a):
        if self.smooth_tv_change:
            self.target_vel = (np.min([self.t, self.tv_endtime]) / self.tv_endtime) * (
                    self.final_tv - self.init_tv) + self.init_tv

        self.current_pd = self.init_balance_pd
        self.vel_enforce_kp = self.init_vel_pd

        if len(self.assist_schedule) > 0:
            for sch in self.assist_schedule:
                if self.t > sch[0]:
                    self.current_pd = sch[1][0]
                    self.vel_enforce_kp = sch[1][1]

        posbefore = self.robot_skeleton.bodynode('MP_BODY').C[0]
        self.advance(a)
        posafter = self.robot_skeleton.bodynode('MP_BODY').C[0]

        vel = (posafter - posbefore) / self.dt / 1.0
        self.vel_cache.append(vel)
        self.target_vel_cache.append(self.target_vel)

        if len(self.vel_cache) > int(0.5 / self.dt / 1.0):
            self.vel_cache.pop(0)
            self.target_vel_cache.pop(0)

        vel_rew = -self.vel_reward_weight * np.abs(np.mean(self.target_vel_cache) - np.mean(self.vel_cache))
        if self.t < self.tv_endtime:
            vel_rew *= 0.5

        pose_math_rew = np.sum(
            np.abs(np.array(self.init_q - self.robot_skeleton.q)[[7, 10, 14, 15, 19, 20, 21, 25]]) ** 2)

        reward = -self.energy_weight * np.sum(
            self.tau) ** 2 + vel_rew + self.alive_bonus - pose_math_rew * self.pose_weight
        reward -= self.work_weight * np.dot(self.tau, self.robot_skeleton.dq)

        foot_down_l = self.robot_skeleton.bodynodes[20].to_world([-1, 0, 0]) - self.robot_skeleton.bodynodes[
            20].to_world([0, 0, 0])
        foot_down_r = self.robot_skeleton.bodynodes[26].to_world([1, 0, 0]) - self.robot_skeleton.bodynodes[
            26].to_world([0, 0, 0])
        foot_ang_l = np.arccos(np.dot(foot_down_l.T, [0, 0, -1]))
        foot_ang_r = np.arccos(np.dot(foot_down_r.T, [0, 0, -1]))
        reward -= (foot_ang_l ** 2 + foot_ang_r ** 2) * 0.15

        foot_height = np.max([self.robot_skeleton.bodynodes[26].C[2], self.robot_skeleton.bodynodes[20].C[2]])
        reward += np.min([(foot_height - self.init_footheight), 0.05]) * 20

        s = self.state_vector()
        com_height = self.robot_skeleton.bodynodes[0].com()[2]
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 200).all() and (com_height > -0.45) and (
                self.robot_skeleton.q[0] > -0.6) and (self.robot_skeleton.q[0] < 0.6) and (
                            abs(self.robot_skeleton.q[1]) < 0.50) and (abs(self.robot_skeleton.q[2]) < 0.50))

        self.fall_on_ground = False
        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        permitted_contact_bodies = [self.robot_skeleton.bodynodes[-1], self.robot_skeleton.bodynodes[-2],
                                    self.robot_skeleton.bodynodes[-7], self.robot_skeleton.bodynodes[-8]]
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()
            if contact.bodynode1 not in permitted_contact_bodies and contact.bodynode2 not in permitted_contact_bodies:
                self.fall_on_ground = True
        if self.fall_on_ground:
            done = True

        if done:
            reward = 0

        self.t += self.dt * 1.0
        self.cur_step += 1

        ob = self._get_obs()

        return ob, reward, done, {'avg_vel': np.mean(self.vel_cache), 'avg_vel2': (posafter - self.init_position_x) / self.t}

    def _get_obs(self):
        # phi = np.array([self.count/self.ref_traj.shape[0]])
        # print(ContactList.shape)
        state = np.concatenate([self.robot_skeleton.q[6:], self.robot_skeleton.dq[6:]])

        body_com = self.robot_skeleton.bodynode('MP_BODY').C

        bodyvel = np.dot(self.robot_skeleton.bodynode('MP_BODY').world_jacobian(
            offset=self.robot_skeleton.bodynode('MP_BODY').local_com()), self.robot_skeleton.dq)
        state = np.concatenate([state, body_com[1:], bodyvel])

        return state

    def reset_model(self):
        self.dart_world.reset()
        qpos = np.zeros(
            self.robot_skeleton.ndofs)  # self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005,
                                                               size=self.robot_skeleton.ndofs)  # np.zeros(self.robot_skeleton.ndofs) #

        qpos[1] = 0.20
        qpos[5] = -0.36

        # LEFT HAND
        qpos[6] = (2518 - 2048) * (np.pi / 180) * 0.088# + np.random.uniform(low=-0.01, high=0.01, size=1)
        qpos[7] = (2248 - 2048) * (np.pi / 180) * 0.088# + np.random.uniform(low=-0.01, high=0.01, size=1)
        qpos[8] = (1712 - 2048) * (np.pi / 180) * 0.088# + np.random.uniform(low=-0.01, high=0.01, size=1)

        # RIGHT HAND
        qpos[9] = (1498 - 2048) * (np.pi / 180) * 0.088# + np.random.uniform(low=-0.01, high=0.01, size=1)
        qpos[10] = (1845 - 2048) * (np.pi / 180) * 0.088# + np.random.uniform(low=-0.01, high=0.01, size=1)
        qpos[11] = (2381 - 2048) * (np.pi / 180) * 0.088# + np.random.uniform(low=-0.01, high=0.01, size=1)

        # LEFT LEG
        qpos[14] = 0  # yaw
        qpos[15] = (2052 - 2048) * (np.pi / 180) * 0.088# + np.random.uniform(low=-0.01, high=0.01, size=1)
        qpos[16] = 0.68# + np.random.uniform(low=-0.01, high=0.01, size=1)
        qpos[17] = -0.88# + np.random.uniform(low=-0.01, high=0.01, size=1)
        qpos[18] = (1707 - 2048) * (np.pi / 180) * 0.088# + np.random.uniform(low=-0.01, high=0.01, size=1)
        qpos[19] = (2039 - 2048) * (np.pi / 180) * 0.088# + np.random.uniform(low=-0.01, high=0.01, size=1)

        # RIGHT LEG
        qpos[20] = 0
        qpos[21] = (2044 - 2048) * (np.pi / 180) * 0.088# + np.random.uniform(low=-0.01, high=0.01, size=1)
        qpos[22] = -0.68# + np.random.uniform(low=-0.01, high=0.01, size=1)
        qpos[23] = 0.88# + np.random.uniform(low=-0.01, high=0.01, size=1)
        qpos[24] = (2389 - 2048) * (np.pi / 180) * 0.088# + np.random.uniform(low=-0.01, high=0.01, size=1)
        qpos[25] = (2057 - 2048) * (np.pi / 180) * 0.088# + np.random.uniform(low=-0.01, high=0.01, size=1)
        print(repr(qpos))

        # set the mass
        '''
        for body in self.robot_skeleton.bodynodes:
            mass = body.m
            mass = np.random.uniform(low=mass-mass/50,high=mass+mass/50,size=1)[0]
            body.set_mass(mass)

            Inertia = body.I
            for i in range(3):
                Inertia[i,i] += np.random.uniform(low=0.9,high=1.1,size=1)[0]*Inertia[i,i]

            body.set_inertia(Inertia)

        for i in range(self.robot_skeleton.ndofs):
            j = self.robot_skeleton.dof(i)
            j.set_damping_coefficient(np.random.uniform(low=0.5,high=1.5,size=1)[0])
            j.set_spring_stiffness(np.random.uniform(low=1.0,high=2.0,size=1)[0])

        '''
        qpos[6:] += np.random.uniform(low=-0.01, high=0.01, size=20)
        self.init_q = np.copy(qpos)
        # self.target = qpos
        self.count = 0
        self.set_state(qpos, qvel)
        f1 = np.random.uniform(low=-15, high=15., size=1)
        f2 = np.random.uniform(low=-15, high=15., size=1)
        #self.robot_skeleton.bodynodes[0].add_ext_force([f1, f2, 0], [0, 0, 0.0])
        self.t = 0
        for i in range(6, self.robot_skeleton.ndofs):
            j = self.robot_skeleton.dof(i)
            j.set_damping_coefficient(0.515)
        self.init_position_x = self.robot_skeleton.bodynode('MP_BODY').C[0]
        self.init_footheight = -0.339
        return self._get_obs()

    def viewer_setup(self):
        if not self.disableViewer:
            self._get_viewer().scene.tb.trans[0] = 0.0
            self._get_viewer().scene.tb.trans[2] = -1.5
            self._get_viewer().scene.tb.trans[1] = 0.0
            self._get_viewer().scene.tb.theta = 80
            self._get_viewer().scene.tb.phi = 0

        return 0
