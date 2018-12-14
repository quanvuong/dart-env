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
import copy, os
from gym.envs.dart.dc_motor import DCMotor

from gym.envs.dart.darwin_utils import *
from gym.envs.dart.parameter_managers import *

class DartDarwinNewFootEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):

        obs_dim = 32

        self.imu_input_step = 1   # number of imu steps as input
        self.imu_cache = []
        self.accum_imu_input = True
        self.include_accelerometer = True
        self.accumulated_imu_info = np.zeros(9)  # start from 9 zeros, so it doesn't necessarily correspond to
                                                 # absolute physical quantity
        if not self.include_accelerometer:
            self.accumulated_imu_info = np.zeros(3)

        self.root_input = False    # whether to include root dofs in the obs

        self.action_filtering = 5 # window size of filtering, 0 means no filtering
        self.action_filter_cache = []

        self.obs_cache = []
        self.multipos_obs = 2 # give multiple steps of position info instead of pos + vel
        if self.multipos_obs > 0:
            obs_dim = 16 * self.multipos_obs

        self.imu_offset = np.array([0,0,-0.06]) # offset in the MP_BODY node for imu measurements
        self.mass_ratio = 1.0
        self.kp_ratio = 1.0
        self.kd_ratio = 1.0
        self.imu_offset_deviation = np.array([0,0,0])

        self.use_discrete_action = True

        self.param_manager = darwinSquatParamManager(self)

        self.use_delta_action = 0.0    # if > 0, use delta action from the current state as the next target
        self.use_DCMotor = False
        self.use_spd = False
        self.train_UP = False
        self.noisy_input = False
        self.resample_MP = False
        self.randomize_timestep = False

        self.kp = None
        self.kd = None

        if self.use_DCMotor:
            self.motors = DCMotor(0.0107, 8.3, 12, 193)

        if self.include_accelerometer:
            obs_dim += self.imu_input_step * 6
        else:
            obs_dim += self.imu_input_step * 3
        if self.accum_imu_input:
            if self.include_accelerometer:
                obs_dim += 9   # accumulated position, linear velocity and orientation, angular velocity is not used as
                           # it is included in the gyro data
            else:
                obs_dim += 3

        if self.root_input:
            obs_dim += 12

        if self.train_UP:
            obs_dim += len(self.param_manager.activated_param)

        self.control_bounds = np.array([-np.ones(16, ), np.ones(16, )])

        # normal pose
        self.permitted_contact_ids = [-1, -2, -7, -8]
        self.init_root_pert = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.target_vel = 0.0
        self.init_tv = 0.0
        self.final_tv = 0.25
        self.tv_endtime = 0.001
        self.avg_rew_weighting = []
        self.vel_cache = []
        self.target_vel_cache = []

        self.assist_timeout = 0.0
        self.assist_schedule = [[0.0, [20000, 20000]], [3.0, [15000, 15000]], [6.0, [11250.0, 11250.0]]]

        self.alive_bonus = 5.0
        self.energy_weight = 0.0015
        self.work_weight = 0.005
        self.vel_reward_weight = 5.0
        self.pose_weight = 0.2

        self.cur_step = 0

        self.torqueLimits = 5.0

        self.t = 0
        self.target = np.zeros(22, )
        self.tau = np.zeros(22, )
        self.init = np.zeros(22, )

        self.include_obs_history = 1
        self.include_act_history = 0
        obs_dim *= self.include_obs_history
        obs_dim += len(self.control_bounds[0]) * self.include_act_history
        obs_perm_base = np.array(
            [-3, -4, -5, -0.0001, -1, -2, -6, 7, -12, -13, -14, -15, -8, -9, -10, -11,
             -19, -20, -21, -16, -17, -18, -22, 23, -28, -29, -30, -31, -24, -25, -26, -27])
        act_perm_base = np.array(
            [-3, -4, -5, -0.0001, -1, -2, -6, 7, -12, -13, -14, -15, -8, -9, -10, -11])

        if self.include_accelerometer:
            for i in range(self.imu_input_step):
                beginid = len(obs_perm_base)
                obs_perm_base = np.concatenate([obs_perm_base, [-beginid, beginid+1, beginid+2, -beginid-3, beginid+4, -beginid-5]])
            if self.accum_imu_input:
                beginid = len(obs_perm_base)
                obs_perm_base = np.concatenate(
                    [obs_perm_base, [-beginid, beginid + 1, beginid + 2, -beginid-3, beginid + 4, beginid + 5,
                                     -beginid - 6, beginid + 7, -beginid - 8]])
        else:
            for i in range(self.imu_input_step):
                beginid = len(obs_perm_base)
                obs_perm_base = np.concatenate([obs_perm_base, [-beginid-1, beginid+2, -beginid-3]])
            if self.accum_imu_input:
                beginid = len(obs_perm_base)
                obs_perm_base = np.concatenate(
                    [obs_perm_base, [-beginid-1, beginid+2, -beginid-3]])

        if self.root_input:
            beginid = len(obs_perm_base)
            obs_perm_base = np.concatenate([obs_perm_base, [-beginid, beginid+1, -beginid-2, beginid+3, -beginid-4, beginid+5,
                                                            -beginid-6, beginid + 7, -beginid - 8, beginid + 9,
                                                            -beginid - 10, beginid + 11]])
        if self.train_UP:
            obs_dim += np.concatenate([obs_perm_base, np.arange(len(self.param_manager.activated_param), len(self.param_manager.activated_param)+3)])

        self.obs_perm = np.copy(obs_perm_base)

        for i in range(self.include_obs_history - 1):
            self.obs_perm = np.concatenate(
                [self.obs_perm, np.sign(obs_perm_base) * (np.abs(obs_perm_base) + len(self.obs_perm))])
        for i in range(self.include_act_history):
            self.obs_perm = np.concatenate(
                [self.obs_perm, np.sign(act_perm_base) * (np.abs(act_perm_base) + len(self.obs_perm))])
        self.act_perm = np.copy(act_perm_base)

        if self.use_discrete_action:
            from gym import spaces
            self.action_space = spaces.MultiDiscrete([11] * 16)

        dart_env.DartEnv.__init__(self, ['darwinmodel/ground1.urdf', 'darwinmodel/darwin_nocollision_newfoot.URDF', 'darwinmodel/coord.urdf', 'darwinmodel/robotis_op2_newfoot.urdf'], 15, obs_dim,
                                  self.control_bounds, dt=0.002, disableViewer=True, action_type="continuous" if not self.use_discrete_action else "discrete")

        self.orig_bodynode_masses = [bn.mass() for bn in self.robot_skeleton.bodynodes]

        self.dart_world.set_gravity([0, 0, -9.81])

        self.dupSkel = self.dart_world.skeletons[1]
        self.dupSkel.set_mobile(False)

        self.dart_world.set_collision_detector(0)

        self.robot_skeleton.set_self_collision_check(True)

        collision_filter = self.dart_world.create_collision_filter()
        collision_filter.add_to_black_list(self.robot_skeleton.bodynode('MP_PELVIS_L'),
                                           self.robot_skeleton.bodynode('MP_THIGH2_L'))
        collision_filter.add_to_black_list(self.robot_skeleton.bodynode('MP_PELVIS_R'),
                                           self.robot_skeleton.bodynode('MP_THIGH2_R'))
        collision_filter.add_to_black_list(self.robot_skeleton.bodynode('MP_ARM_HIGH_L'),
                                           self.robot_skeleton.bodynode('l_hand'))
        collision_filter.add_to_black_list(self.robot_skeleton.bodynode('MP_ARM_HIGH_R'),
                                           self.robot_skeleton.bodynode('r_hand'))

        self.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(5.0)
        for bn in self.robot_skeleton.bodynodes:
            bn.set_friction_coeff(5.0)
        self.robot_skeleton.bodynode('l_hand').set_friction_coeff(0.0)
        self.robot_skeleton.bodynode('r_hand').set_friction_coeff(0.0)

        self.add_perturbation = True
        self.perturbation_parameters = [0.02, 1, 1, 40]  # probability, magnitude, bodyid, duration
        self.perturbation_duration = 40

        for i in range(6, self.robot_skeleton.ndofs):
            j = self.robot_skeleton.dof(i)
            j.set_damping_coefficient(0.515)
            j.set_coulomb_friction(0.0)
        self.init_position_x = self.robot_skeleton.bodynode('MP_BODY').C[0]

        # set joint limits according to the measured one
        for i in range(6, self.robot_skeleton.ndofs):
            self.robot_skeleton.dofs[i].set_position_lower_limit(JOINT_LOW_BOUND_NEWFOOT[i - 6] - 0.01)
            self.robot_skeleton.dofs[i].set_position_upper_limit(JOINT_UP_BOUND_NEWFOOT[i - 6] + 0.01)

        self.permitted_contact_bodies = [self.robot_skeleton.bodynodes[id] for id in self.permitted_contact_ids]

        print('Total mass: ', self.robot_skeleton.mass())
        print('Permitted ground contact: ', self.permitted_contact_bodies)

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

    def get_imu_data(self):
        tinv = np.linalg.inv(self.robot_skeleton.bodynode('MP_BODY').T[0:3, 0:3])

        if self.include_accelerometer:
            acc = np.dot(self.robot_skeleton.bodynode('MP_BODY').linear_jacobian_deriv(
                    offset=self.robot_skeleton.bodynode('MP_BODY').local_com()+self.imu_offset+self.imu_offset_deviation, full=True),
                             self.robot_skeleton.dq) + np.dot(self.robot_skeleton.bodynode('MP_BODY').linear_jacobian(
                    offset=self.robot_skeleton.bodynode('MP_BODY').local_com()+self.imu_offset+self.imu_offset_deviation, full=True), self.robot_skeleton.ddq)
            #acc -= self.dart_world.gravity()
            lacc = np.dot(tinv, acc)
            # Correction for Darwin hardware
            lacc = np.array([lacc[1], lacc[0], -lacc[2]])

        angvel = self.robot_skeleton.bodynode('MP_BODY').com_spatial_velocity()[0:3]
        langvel = np.dot(tinv, angvel)

        # Correction for Darwin hardware
        langvel = np.array([-langvel[0], -langvel[1], langvel[2]])[[2,1,0]]

        if self.include_accelerometer:
            imu_data = np.concatenate([lacc, langvel])
        else:
            imu_data = langvel

        imu_data += np.random.normal(0, 0.001, len(imu_data))

        return imu_data

    def integrate_imu_data(self):
        imu_data = self.get_imu_data()
        if self.include_accelerometer:
            self.accumulated_imu_info[3:6] += imu_data[0:3] * self.dt
            self.accumulated_imu_info[0:3] += self.accumulated_imu_info[3:6] * self.dt
            self.accumulated_imu_info[6:] += imu_data[3:] * self.dt
        else:
            self.accumulated_imu_info += imu_data * self.dt

    def advance(self, a):
        clamped_control = np.array(a)

        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
            if clamped_control[i] < self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]

        if self.use_delta_action > 0:
            self.target[6:] = np.clip(clamped_control * self.use_delta_action + self.robot_skeleton.q[6:],
                                      CONTROL_LOW_BOUND_NEWFOOT, CONTROL_UP_BOUND_NEWFOOT)
        else:
            self.target[6:] = (clamped_control + 1.0) / 2.0 * (CONTROL_UP_BOUND_NEWFOOT - CONTROL_LOW_BOUND_NEWFOOT) + CONTROL_LOW_BOUND_NEWFOOT

        dup_pos = np.copy(self.target)
        dup_pos[4] = 0.5
        self.dupSkel.set_positions(dup_pos)
        self.dupSkel.set_velocities(self.target*0)

        if self.add_perturbation:
            if self.perturbation_duration == 0:
                self.perturb_force *= 0
                if np.random.random() < self.perturbation_parameters[0]:
                    axis_rand = np.random.randint(0, 2, 1)[0]
                    direction_rand = np.random.randint(0, 2, 1)[0] * 2 - 1
                    self.perturb_force[axis_rand] = direction_rand * self.perturbation_parameters[1]
                    self.perturbation_duration = self.perturbation_parameters[3]
            else:
                self.perturbation_duration -= 1


        for i in range(self.frame_skip):
            self.tau[6:] = self.PID()
            self.tau[0:6] *= 0.0

            if self.add_perturbation:
                self.robot_skeleton.bodynodes[self.perturbation_parameters[2]].add_ext_force(self.perturb_force)

            if self.t < self.assist_timeout:
                force = self._bodynode_spd(self.robot_skeleton.bodynode('MP_BODY'), self.current_pd, 1)
                self.robot_skeleton.bodynode('MP_BODY').add_ext_force(np.array([0, force, 0]))

                force = self._bodynode_spd(self.robot_skeleton.bodynode('MP_BODY'), self.vel_enforce_kp, 0, self.target_vel)
                self.robot_skeleton.bodynode('MP_BODY').add_ext_force(np.array([force, 0, 0]))

            self.robot_skeleton.set_forces(self.tau)
            self.dart_world.step()


    def PID(self):
        # print("#########################################################################3")

        if self.use_DCMotor:
            if self.kp is not None:
                kp = self.kp * self.kp_ratio
                kd = self.kd * self.kd_ratio
            else:
                kp = np.array([4]*16) * self.kp_ratio
                kd = np.array([0.032]*16) * self.kd_ratio
            pwm_command = -1 * kp * (self.robot_skeleton.q[6:] - self.target[6:]) - kd * self.robot_skeleton.dq[6:]
            tau = np.zeros(22, )
            tau[6:] = self.motors.get_torque(pwm_command, self.robot_skeleton.dq[6:])
        elif self.use_spd:
            if self.kp is not None:
                kp = np.array([self.kp]*22) * self.kp_ratio
                kd = np.array([self.kd]*22) * self.kd_ratio
                kp[0:6] *= 0
                kd[0:6] *= 0
            else:
                kp = np.array([0]*22) * self.kp_ratio
                kd = np.array([0.0]*22) * self.kd_ratio

            p = -kp * (self.robot_skeleton.q + self.robot_skeleton.dq * self.sim_dt - self.target)
            d = -kd * self.robot_skeleton.dq
            qddot = np.linalg.solve(self.robot_skeleton.M + np.diagflat(kd) * self.sim_dt, -self.robot_skeleton.c + p + d + self.robot_skeleton.constraint_forces())
            tau = p + d - kd * qddot * self.sim_dt
            tau[0:6] *= 0
        else:
            if self.kp is not None:
                kp = self.kp * self.kp_ratio
                kd = self.kd * self.kd_ratio
            else:
                kp = np.array([2.1+10, 1.79+10, 4.93+10,
                           2.0+10, 2.02+10, 1.98+10,
                           2.2+10, 2.06+10,
                           148, 152, 150, 136,
                           151, 151.4, 150.45, 151.36]) * self.kp_ratio
                kd = np.array([0.021, 0.023, 0.022,
                           0.025, 0.021, 0.026,
                           0.028, 0.0213
                    , 0.192, 0.198, 0.22, 0.199,
                           0.53, 0.27, 0.21, 0.205]) * self.kd_ratio

            q = self.robot_skeleton.q
            qdot = self.robot_skeleton.dq
            tau = np.zeros(22, )
            tau[6:] = -kp * (q[6:] - self.target[6:]) - kd * qdot[6:]

        torqs = self.ClampTorques(tau)

        return torqs[6:]

    def ClampTorques(self, torques):
        torqueLimits = self.torqueLimits

        for i in range(6, 22):
            if torques[i] > torqueLimits:  #
                torques[i] = torqueLimits
            if torques[i] < -torqueLimits:
                torques[i] = -torqueLimits

        return torques

    def step(self, a):
        if self.use_discrete_action:
            a = a * 1.0/ np.floor(self.action_space.nvec/2.0) - 1.0

        self.action_filter_cache.append(a)
        if len(self.action_filter_cache) > self.action_filtering:
            self.action_filter_cache.pop(0)
        if self.action_filtering > 0:
            a = np.mean(self.action_filter_cache, axis=0)

        self.action_buffer.append(np.copy(a))

        self.target_vel = (np.min([self.t, self.tv_endtime]) / self.tv_endtime) * (
                self.final_tv - self.init_tv) + self.init_tv

        self.current_pd = self.assist_schedule[0][1][0]
        self.vel_enforce_kp = self.assist_schedule[0][1][1]
        if len(self.assist_schedule) > 0:
            for sch in self.assist_schedule:
                if self.t > sch[0]:
                    self.current_pd = sch[1][0]
                    self.vel_enforce_kp = sch[1][1]

        xpos_before = self.robot_skeleton.q[3]
        self.advance(a)
        xpos_after = self.robot_skeleton.q[3]

        self.integrate_imu_data()

        # reward
        vel = (xpos_after - xpos_before) / self.dt
        self.vel_cache.append(vel)
        self.target_vel_cache.append(self.target_vel)

        if len(self.vel_cache) > int(0.5 / self.dt / 1.0):
            self.vel_cache.pop(0)
            self.target_vel_cache.pop(0)

        vel_rew = -self.vel_reward_weight * np.abs(np.mean(self.target_vel_cache) - np.mean(self.vel_cache))
        if self.t < self.tv_endtime:
            vel_rew *= 0.5

        pose_math_rew = np.sum(
            np.abs(np.array(self.init_q - self.robot_skeleton.q)[6:]) ** 2)

        reward = -self.energy_weight * np.sum(
            self.tau) ** 2 + vel_rew + self.alive_bonus - pose_math_rew * self.pose_weight
        reward -= self.work_weight * np.dot(self.tau, self.robot_skeleton.dq) - np.abs(self.robot_skeleton.q[4])

        s = self.state_vector()
        com_height = self.robot_skeleton.bodynodes[0].com()[2]
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 200).all() and (com_height > -0.45) and
                    (abs(self.robot_skeleton.q[0]) < 0.8) and (
                            abs(self.robot_skeleton.q[1]) < 0.8) and (abs(self.robot_skeleton.q[2]) < 0.8))

        self.fall_on_ground = False
        self_colliding = False
        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0

        ground_bodies = [self.dart_world.skeletons[0].bodynodes[0]]
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()
            if contact.bodynode1 not in self.permitted_contact_bodies and contact.bodynode2 not in self.permitted_contact_bodies:
                if contact.bodynode1 in ground_bodies or contact.bodynode2 in ground_bodies:
                    self.fall_on_ground = True
            if contact.bodynode1.skel == contact.bodynode2.skel:
                self_colliding = True


        if self.fall_on_ground:
            done = True

        if self_colliding:
            done = True

        if done:
            reward = 0

        self.t += self.dt * 1.0
        self.cur_step += 1

        self.imu_cache.append(self.get_imu_data())

        ob = self._get_obs()

        c = self.robot_skeleton.bodynode('MP_BODY').to_world(self.imu_offset+self.robot_skeleton.bodynode('MP_BODY').local_com()+self.imu_offset_deviation)
        #self.dart_world.skeletons[2].q = np.array([0, 0, 0, c[0], c[1], c[2]])


        return ob, reward, done, {'avg_vel': np.mean(self.vel_cache)}

    def _get_obs(self):
        # phi = np.array([self.count/self.ref_traj.shape[0]])
        # print(ContactList.shape)
        state = np.concatenate([self.robot_skeleton.q[6:], self.robot_skeleton.dq[6:]])
        if self.multipos_obs > 0:
            state = self.robot_skeleton.q[6:]

            self.obs_cache.append(state)
            while len(self.obs_cache) < self.multipos_obs:
                self.obs_cache.append(state)
            if len(self.obs_cache) > self.multipos_obs:
                self.obs_cache.pop(0)

            state = np.concatenate(self.obs_cache)

        for i in range(self.imu_input_step):
            state = np.concatenate([state, self.imu_cache[-i]])

        if self.accum_imu_input:
            state = np.concatenate([state, self.accumulated_imu_info])

        if self.root_input:
            state = np.concatenate([state, self.robot_skeleton.q[0:6], self.robot_skeleton.dq[0:6]])

        if self.train_UP:
            UP = self.param_manager.get_simulator_parameters()
            state = np.concatenate([state, UP])

        if self.noisy_input:
            state = state + np.random.normal(0, .01, len(state))



        return state

    def reset_model(self):
        self.dart_world.reset()
        qpos = np.zeros(
            self.robot_skeleton.ndofs)  # self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005,
                                                               size=self.robot_skeleton.ndofs)  # np.zeros(self.robot_skeleton.ndofs) #


        qpos[6:] += np.random.uniform(low=-0.01, high=0.01, size=16)
        # self.target = qpos
        self.count = 0
        qpos[0:6] += self.init_root_pert
        self.set_state(qpos, qvel)

        q = self.robot_skeleton.q
        q[5] += -0.33 - np.min([self.robot_skeleton.bodynodes[-1].C[2], self.robot_skeleton.bodynodes[-8].C[2]])
        self.robot_skeleton.q = q

        self.init_q = np.copy(self.robot_skeleton.q)

        self.t = 0

        self.action_buffer = []

        self.accumulated_imu_info = np.zeros(9)
        if not self.include_accelerometer:
            self.accumulated_imu_info = np.zeros(3)

        if self.resample_MP:
            self.param_manager.resample_parameters()
            self.current_param = self.param_manager.get_simulator_parameters()

        if self.randomize_timestep:
            self.frame_skip = np.random.randint(20, 40)
            self.dart_world.dt = 0.05 / self.frame_skip

        self.imu_cache = []
        for i in range(self.imu_input_step):
            self.imu_cache.append(self.get_imu_data())
        self.obs_cache = []

        if self.resample_MP or self.mass_ratio != 0:
            # set the ratio of the mass
            for i in range(len(self.robot_skeleton.bodynodes)):
                self.robot_skeleton.bodynodes[i].set_mass(self.orig_bodynode_masses[i] * self.mass_ratio)

        self.dart_world.skeletons[2].q = [0,0,0, 100, 100, 100]


        return self._get_obs()

    def viewer_setup(self):
        if not self.disableViewer:
            self._get_viewer().scene.tb.trans[0] = 0.0
            self._get_viewer().scene.tb.trans[2] = -1.5
            self._get_viewer().scene.tb.trans[1] = 0.0
            self._get_viewer().scene.tb.theta = 80
            self._get_viewer().scene.tb.phi = 0

        return 0
