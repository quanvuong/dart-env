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

class DartDarwinSquatEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):

        obs_dim = 40

        self.imu_input_step = 0   # number of imu steps as input
        self.imu_cache = []

        self.action_filtering = 5 # window size of filtering, 0 means no filtering
        self.action_filter_cache = []

        self.obs_cache = []
        self.multipos_obs = 2 # give multiple steps of position info instead of pos + vel
        if self.multipos_obs > 0:
            obs_dim = 20 * self.multipos_obs

        self.imu_offset = np.array([0,0,-0.06]) # offset in the MP_BODY node for imu measurements
        self.mass_ratio = 1.0
        self.kp_ratio = 1.0
        self.kd_ratio = 1.0
        self.imu_offset_deviation = np.array([0,0,0])

        self.use_discrete_action = True

        self.param_manager = darwinSquatParamManager(self)

        self.use_DCMotor = False
        self.use_spd = False
        self.train_UP = False
        self.noisy_input = False
        self.resample_MP = True
        self.randomize_timestep = True
        self.load_keyframe_from_file = False
        self.forward_reward = 0.0
        self.kp = None
        self.kd = None

        if self.use_DCMotor:
            self.motors = DCMotor(0.0107, 8.3, 12, 193)

        obs_dim += self.imu_input_step * 3

        if self.train_UP:
            obs_dim += len(self.param_manager.activated_param)

        self.control_bounds = np.array([-np.ones(20, ), np.ones(20, )])

        self.pose_squat_val = np.array([2509, 2297, 1714, 1508, 1816, 2376,
                                    2047, 2171,
                                    2032, 2039, 2795, 648, 1231, 2040,   2041, 2060, 1281, 3448, 2855, 2073])

        self.pose_stand_val = np.array([1500, 2048, 2048, 2500, 2048, 2048,
                                        2048, 2048,
                                        2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048])

        self.pose_left_stand_val = np.array([1500, 2048, 2048, 2500, 2048, 2048,
                                        2048, 2048,
                                         2032, 2039, 2795, 568, 1231, 2040, 2048, 2048, 2048, 2048, 2048, 2048])

        self.pose_right_stand_val = np.array([1500, 2048, 2048, 2500, 2048, 2048,
                                        2048, 2048,
                                        2048, 2048, 2048, 2048, 2048, 2048, 2041, 2060, 1281, 3525, 2855, 2073])

        self.pose_squat = VAL2RADIAN(self.pose_squat_val)
        self.pose_stand = VAL2RADIAN(self.pose_stand_val)
        self.pose_left_stand = VAL2RADIAN(self.pose_left_stand_val)
        self.pose_right_stand = VAL2RADIAN(self.pose_right_stand_val)

        self.interp_sch = [[0.0, self.pose_stand],
                           [2.0, self.pose_squat],
                           [4.0, self.pose_stand],
                           [5.0, self.pose_stand],
                           [5.5, self.pose_squat],
                           [6.0, self.pose_stand],
                           [8.0, self.pose_stand],
                           ]
        '''[[0.0, self.pose_stand],
           [1.5, self.pose_squat],
           [2.5, self.pose_stand],
           [3.0, self.pose_squat],
           [3.3, self.pose_stand],
           [3.6, self.pose_squat], ]'''

        if self.load_keyframe_from_file:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", 'darwinmodel/rig_keyframe.txt')
            rig_keyframe = np.loadtxt(fullpath)
            #self.interp_sch = [[0.0, rig_keyframe[0]],
            #              [2.0, rig_keyframe[1]],
            #              [6.0, rig_keyframe[1]]]
            self.interp_sch = [[0.0, rig_keyframe[0]]]
            interp_time = 0.5
            for i in range(10):
                for k in range(1, len(rig_keyframe)):
                    self.interp_sch.append([interp_time, rig_keyframe[k]])
                    interp_time += 0.5
            self.interp_sch.append([interp_time, rig_keyframe[0]])

        # single leg stand
        #self.permitted_contact_ids = [-7, -8] #[-1, -2, -7, -8]
        #self.init_root_pert = np.array([0.0, 1.2, 0.0, 0.0, 0.0, 0.0])

        # crawl
        self.permitted_contact_ids = [-1, -2, -7, -8, 6, 11]  # [-1, -2, -7, -8]
        self.init_root_pert = np.array([0.0, 1.35, 0.0, 0.0, 0.0, 0.0])

        # normal pose
        self.permitted_contact_ids = [-1, -2, -7, -8]
        self.init_root_pert = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.delta_angle_scale = 0.3

        self.alive_bonus = 5.0
        self.energy_weight = 0.015
        self.work_weight = 0.05
        self.pose_weight = 0.2

        self.cur_step = 0

        self.torqueLimits = 10.0

        self.t = 0
        self.target = np.zeros(26, )
        self.tau = np.zeros(26, )
        self.init = np.zeros(26, )

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

        if self.use_discrete_action:
            from gym import spaces
            self.action_space = spaces.MultiDiscrete([11] * 20)

        dart_env.DartEnv.__init__(self, ['darwinmodel/ground1.urdf', 'darwinmodel/darwin_nocollision.URDF', 'darwinmodel/coord.urdf', 'darwinmodel/robotis_op2.urdf'], 25, obs_dim,
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
        collision_filter.add_to_black_list(self.robot_skeleton.bodynode('MP_TIBIA_R'),
                                           self.robot_skeleton.bodynode('MP_ANKLE2_R'))
        collision_filter.add_to_black_list(self.robot_skeleton.bodynode('MP_TIBIA_L'),
                                           self.robot_skeleton.bodynode('MP_ANKLE2_L'))

        self.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(2.0)
        for bn in self.robot_skeleton.bodynodes:
            bn.set_friction_coeff(2.0)
        self.robot_skeleton.bodynode('l_hand').set_friction_coeff(2.0)
        self.robot_skeleton.bodynode('r_hand').set_friction_coeff(2.0)

        self.add_perturbation = False
        self.perturbation_parameters = [0.02, 1, 1, 40]  # probability, magnitude, bodyid, duration
        self.perturbation_duration = 40

        for i in range(6, self.robot_skeleton.ndofs):
            j = self.robot_skeleton.dof(i)
            j.set_damping_coefficient(0.515)
            j.set_coulomb_friction(0.0)
        self.init_position_x = self.robot_skeleton.bodynode('MP_BODY').C[0]

        # set joint limits according to the measured one
        for i in range(6, self.robot_skeleton.ndofs):
            self.robot_skeleton.dofs[i].set_position_lower_limit(JOINT_LOW_BOUND[i - 6] - 0.01)
            self.robot_skeleton.dofs[i].set_position_upper_limit(JOINT_UP_BOUND[i - 6] + 0.01)

        self.permitted_contact_bodies = [self.robot_skeleton.bodynodes[id] for id in self.permitted_contact_ids]

        print('Total mass: ', self.robot_skeleton.mass())

        utils.EzPickle.__init__(self)

    def get_imu_data(self):
        '''acc = np.dot(self.robot_skeleton.bodynode('MP_BODY').linear_jacobian_deriv(
                offset=self.robot_skeleton.bodynode('MP_BODY').local_com()+self.imu_offset+self.imu_offset_deviation, full=True),
                         self.robot_skeleton.dq) + np.dot(self.robot_skeleton.bodynode('MP_BODY').linear_jacobian(
                offset=self.robot_skeleton.bodynode('MP_BODY').local_com()+self.imu_offset+self.imu_offset_deviation, full=True), self.robot_skeleton.ddq)

        acc -= self.dart_world.gravity()'''
        angvel = self.robot_skeleton.bodynode('MP_BODY').com_spatial_velocity()[0:3]

        tinv = np.linalg.inv(self.robot_skeleton.bodynode('MP_BODY').T[0:3, 0:3])

        #lacc = np.dot(tinv, acc)
        langvel = np.dot(tinv, angvel)

        # Correction for Darwin hardware
        #lacc = np.array([lacc[1], lacc[0], -lacc[2]])
        #langvel = np.array([-langvel[0], -langvel[1], langvel[2]])[[2,1,0]]

        #imu_data = np.concatenate([lacc, langvel])
        imu_data = langvel

        imu_data += np.random.normal(0, 0.001, len(imu_data))

        return imu_data

    def advance(self, a):
        clamped_control = np.array(a)

        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
            if clamped_control[i] < self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]

        self.ref_target = self.interp_sch[0][1]

        for i in range(len(self.interp_sch)-1):
            if self.t >= self.interp_sch[i][0] and self.t < self.interp_sch[i+1][0]:
                ratio = (self.t - self.interp_sch[i][0]) / (self.interp_sch[i+1][0] - self.interp_sch[i][0])
                self.ref_target = ratio * self.interp_sch[i+1][1] + (1 - ratio) * self.interp_sch[i][1]
        if self.t > self.interp_sch[-1][0]:
            self.ref_target = self.interp_sch[-1][1]

        self.target[6:] = self.ref_target + clamped_control * self.delta_angle_scale
        self.target[6:] = np.clip(self.target[6:], JOINT_LOW_BOUND, JOINT_UP_BOUND)


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
            self.robot_skeleton.set_forces(self.tau)
            self.dart_world.step()


    def PID(self):
        # print("#########################################################################3")

        if self.use_DCMotor:
            if self.kp is not None:
                kp = self.kp * self.kp_ratio
                kd = self.kd * self.kd_ratio
            else:
                kp = np.array([4]*20) * self.kp_ratio
                kd = np.array([0.032]*20) * self.kd_ratio
            pwm_command = -1 * kp * (self.robot_skeleton.q[6:] - self.target[6:]) - kd * self.robot_skeleton.dq[6:]
            tau = np.zeros(26, )
            tau[6:] = self.motors.get_torque(pwm_command, self.robot_skeleton.dq[6:])
        elif self.use_spd:
            if self.kp is not None:
                kp = np.array([self.kp]*26) * self.kp_ratio
                kd = np.array([self.kd]*26) * self.kd_ratio
                kp[0:6] *= 0
                kd[0:6] *= 0
            else:
                kp = np.array([0]*26) * self.kp_ratio
                kd = np.array([0.0]*26) * self.kd_ratio

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
                           148, 152, 150, 136, 153, 102,
                           151, 151.4, 150.45, 151.36, 154, 105.2]) * self.kp_ratio * 2
                kd = np.array([0.021, 0.023, 0.022,
                           0.025, 0.021, 0.026,
                           0.028, 0.0213
                    , 0.192, 0.198, 0.22, 0.199, 0.02, 0.01,
                           0.53, 0.27, 0.21, 0.205, 0.022, 0.056]) * self.kd_ratio / 10

            q = self.robot_skeleton.q
            qdot = self.robot_skeleton.dq
            tau = np.zeros(26, )
            tau[6:] = -kp * (q[6:] - self.target[6:]) - kd * qdot[6:]

        torqs = self.ClampTorques(tau)

        return torqs[6:]

    def ClampTorques(self, torques):
        torqueLimits = self.torqueLimits

        for i in range(6, 26):
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

        xpos_before = self.robot_skeleton.q[3]
        self.advance(a)
        xpos_after = self.robot_skeleton.q[3]

        pose_math_rew = np.sum(
            np.abs(np.array(self.ref_target - self.robot_skeleton.q[6:])) ** 2)

        reward = -self.energy_weight * np.sum(
            self.tau ** 2) + self.alive_bonus - pose_math_rew * self.pose_weight
        reward -= self.work_weight * np.dot(self.tau, self.robot_skeleton.dq)
        #reward -= 0.5 * np.sum(np.abs(self.robot_skeleton.dC))

        reward += self.forward_reward * (xpos_after - xpos_before) / self.dt

        s = self.state_vector()
        com_height = self.robot_skeleton.bodynodes[0].com()[2]
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 200).all())

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
        if self.t > self.interp_sch[-1][0] + 2:
            done = True
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


        return ob, reward, done, {}

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

        #qpos[1] += np.random.uniform(-0.2, 0.2)

        # LEFT HAND
        qpos[6:] = self.interp_sch[0][1]

        qpos[6:] += np.random.uniform(low=-0.01, high=0.01, size=20)
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
