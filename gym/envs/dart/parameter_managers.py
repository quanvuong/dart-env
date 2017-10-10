__author__ = 'yuwenhao'

import numpy as np
from gym.envs.dart import dart_env

##############################################################################################################
################################  Hopper #####################################################################
##############################################################################################################


class hopperContactMassManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.2, 1.0] # friction range
        '''self.restitution_range = [0.0, 0.05]
        self.torso_mass_range = [2.5, 4.5]
        self.foot_mass_range = [4.0, 6.0]
        self.power_range = [170, 230]'''
        self.restitution_range = [0.0, 0.3]
        self.torso_mass_range = [2.0, 35.0]
        self.foot_mass_range = [2.0, 25.0]
        self.power_range = [150, 320]
        self.ankle_range = [40, 300]
        self.activated_param = [2,3]
        self.controllable_param = [2,3]
        
        self.binned_param = 2 # don't bin if = 0

        self.param_dim = len(self.activated_param)
        self.sampling_selector = None
        self.selector_target = -1

    def get_simulator_parameters(self):
        cur_friction = self.simulator.dart_world.skeletons[0].bodynodes[0].friction_coeff()
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        cur_rest = self.simulator.dart_world.skeletons[0].bodynodes[0].restitution_coeff()
        restitution_param = (cur_rest - self.restitution_range[0]) / (self.restitution_range[1] - self.restitution_range[0])

        cur_mass = self.simulator.robot_skeleton.bodynodes[2].m
        mass_param = (cur_mass - self.torso_mass_range[0]) / (self.torso_mass_range[1] - self.torso_mass_range[0])

        cur_ft_mass = self.simulator.robot_skeleton.bodynodes[-1].m
        ft_mass_param = (cur_ft_mass - self.foot_mass_range[0]) / (self.foot_mass_range[1] - self.foot_mass_range[0])

        cur_power = self.simulator.action_scale
        power_param = (cur_power[0] - self.power_range[0]) / (self.power_range[1] - self.power_range[0])

        cur_ank_power = self.simulator.action_scale[2]
        ank_power_param = (cur_ank_power - self.ankle_range[0]) / (self.ankle_range[1] - self.ankle_range[0])

        params = np.array([friction_param, restitution_param, mass_param, ft_mass_param, power_param, ank_power_param])[self.activated_param]
        if self.binned_param > 0:
            for i in range(len(params)):
                params[i] = int(params[i] / (1.0 / self.binned_param)) * (1.0/self.binned_param) + 0.5 / self.binned_param
        return params

    def set_simulator_parameters(self, x):
        cur_id = 0
        if 0 in self.controllable_param:
            friction = x[cur_id] * (self.range[1] - self.range[0]) + self.range[0]
            self.simulator.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(friction)
            cur_id += 1
        if 1 in self.controllable_param:
            restitution = x[cur_id] * (self.restitution_range[1] - self.restitution_range[0]) + self.restitution_range[0]
            self.simulator.dart_world.skeletons[0].bodynodes[0].set_restitution_coeff(restitution)
            self.simulator.dart_world.skeletons[1].bodynodes[-1].set_restitution_coeff(1.0)
            cur_id += 1
        if 2 in self.controllable_param:
            mass = x[cur_id] * (self.torso_mass_range[1] - self.torso_mass_range[0]) + self.torso_mass_range[0]
            self.simulator.robot_skeleton.bodynodes[2].set_mass(mass)
            cur_id += 1
        if 3 in self.controllable_param:
            ft_mass = x[cur_id] * (self.foot_mass_range[1] - self.foot_mass_range[0]) + self.foot_mass_range[0]
            self.simulator.robot_skeleton.bodynodes[-1].set_mass(ft_mass)
            cur_id += 1
        if 4 in self.controllable_param:
            power = x[cur_id] * (self.power_range[1] - self.power_range[0]) + self.power_range[0]
            self.simulator.action_scale = np.array([power, power, power])
            cur_id += 1
        if 5 in self.controllable_param:
            ankpower = x[cur_id] * (self.ankle_range[1] - self.ankle_range[0]) + self.ankle_range[0]
            self.simulator.action_scale[2] = ankpower
            cur_id += 1

    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, len(self.get_simulator_parameters()))
        if self.sampling_selector is not None:
            while not self.sampling_selector.classify(np.array([x])) == self.selector_target:
                x = np.random.uniform(0, 1, len(self.get_simulator_parameters()))
        self.set_simulator_parameters(x)


class walker3dManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0, 1500]  # lateral kp
        self.velkp_range = [0, 1500] # forward kp
        self.activated_param = [0]
        self.controllable_param = [0]

        self.param_dim = len(self.activated_param)

    def get_simulator_parameters(self):
        cur_kp = self.simulator.init_balance_pd
        kp_param = (cur_kp - self.range[0]) / (self.range[1] - self.range[0])

        cur_vel_kp = self.simulator.init_vel_pd
        vel_kp_param = (cur_vel_kp - self.velkp_range[0]) / (self.velkp_range[1] - self.velkp_range[0])

        params = np.array([kp_param, vel_kp_param])[self.activated_param]
        return params

    def set_simulator_parameters(self, x):
        cur_id = 0
        if 0 in self.controllable_param:
            kp = x[cur_id] * (self.range[1] - self.range[0]) + self.range[0]
            self.simulator.init_balance_pd = kp
            cur_id += 1
        if 1 in self.controllable_param:
            kp = x[cur_id] * (self.velkp_range[1] - self.velkp_range[0]) + self.velkp_range[0]
            self.simulator.init_vel_pd = kp
            cur_id += 1


class hopperBackPackManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.0, 60.0] # backpack mass range
        self.slope = [-0.5, 0.5]
        self.activated_param = [0]
        self.controllable_param = [0]

        self.param_dim = len(self.activated_param)
        self.sampling_selector = None
        self.selector_target = -1

    def get_simulator_parameters(self):
        cur_bpmass = self.simulator.robot_skeleton.bodynodes[3].m
        bpmass_param = (cur_bpmass - self.range[0]) / (self.range[1] - self.range[0])

        transform = self.simulator.dart_world.skeletons[0].bodynodes[0].shapenodes[0].relative_transform()
        cur_angle = np.arcsin(transform[1, 0])
        angle_param = (cur_angle - self.slope[0]) / (self.slope[1] - self.slope[0])

        return np.array([bpmass_param, angle_param])[self.activated_param]

    def set_simulator_parameters(self, x):
        cur_id = 0
        if 0 in self.controllable_param:
            bpmass = x[cur_id] * (self.range[1] - self.range[0]) + self.range[0]
            self.simulator.robot_skeleton.bodynodes[3].set_mass(bpmass)
            cur_id += 1
        if 1 in self.controllable_param:
            angle = x[cur_id] * (self.slope[1] - self.slope[0]) + self.slope[0]
            rot = np.identity(4)
            rot[0, 0] = np.cos(angle)
            rot[0, 1] = -np.sin(angle)
            rot[1, 0] = np.sin(angle)
            rot[1, 1] = np.cos(angle)
            self.simulator.dart_world.skeletons[0].bodynodes[0].shapenodes[0].set_relative_transform(rot)
            self.simulator.dart_world.skeletons[0].bodynodes[0].shapenodes[1].set_relative_transform(rot)
            cur_id += 1
            # rotate the ankle so that it doesn't collide with the ground
            cq = self.simulator.robot_skeleton.q
            cq[5] += angle
            self.simulator.robot_skeleton.set_positions(cq)


    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, len(self.get_simulator_parameters()))
        #if self.sampling_selector is not None:
        #    while not self.sampling_selector.classify(np.array([x])) == self.selector_target:
        #        x = np.random.uniform(0, 1, len(self.get_simulator_parameters()))

        self.set_simulator_parameters(x)

class hopperContactMassRoughnessManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.3, 1.0] # friction range
        self.torso_mass_range = [3.0, 6.0]
        self.roughness_range = [-0.05, -0.02] # height of the obstacles
        self.param_dim = 3


    def get_simulator_parameters(self):
        cur_friction = self.simulator.dart_world.skeletons[0].bodynodes[0].friction_coeff()
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        cur_mass = self.simulator.robot_skeleton.bodynodes[2].m
        mass_param = (cur_mass - self.torso_mass_range[0]) / (self.torso_mass_range[1] - self.torso_mass_range[0])

        cq = self.simulator.dart_world.skeletons[0].q
        cur_height = cq[10]
        roughness_param = (cur_height - self.roughness_range[0]) / (self.roughness_range[1] - self.roughness_range[0])

        return np.array([friction_param, mass_param, roughness_param])

    def set_simulator_parameters(self, x):
        friction = x[0] * (self.range[1] - self.range[0]) + self.range[0]
        for i in range(len(self.simulator.dart_world.skeletons[0].bodynodes)):
            self.simulator.dart_world.skeletons[0].bodynodes[i].set_friction_coeff(friction)

        mass = x[1] * (self.torso_mass_range[1] - self.torso_mass_range[0]) + self.torso_mass_range[0]
        self.simulator.robot_skeleton.bodynodes[2].set_mass(mass)

        obs_height = x[2] * (self.roughness_range[1] - self.roughness_range[0]) + self.roughness_range[0]
        cq = self.simulator.dart_world.skeletons[0].q
        cq[10] = obs_height
        cq[16] = obs_height
        cq[22] = obs_height
        cq[28] = obs_height
        cq[34] = obs_height
        cq[40] = obs_height
        self.simulator.dart_world.skeletons[0].q = cq

    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, len(self.get_simulator_parameters()))
        #x = np.random.normal(0, 0.2, self.param_dim) % 1
        x[2] = np.max([x[2]*1.5,1.0])
        self.set_simulator_parameters(x)

        if len(self.simulator.dart_world.skeletons[0].bodynodes) >= 7:
            cq = self.simulator.dart_world.skeletons[0].q
            pos = []
            pos.append(np.random.random()-0.5)
            for i in range(5):
                pos.append((pos[i] + 1.0/6.0)%1)
            np.random.shuffle(pos)
            cq[9] = pos[0]
            cq[15] = pos[1]
            cq[21] = pos[2]
            cq[27] = pos[3]
            cq[33] = pos[4]
            cq[39] = pos[5]
            self.simulator.dart_world.skeletons[0].q = cq

class hopperContactAllMassManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.3, 1.0] # friction range
        self.mass_range = [-1.0, 1.0]
        self.param_dim = 2
        self.initial_mass = []
        for i in range(4):
            self.initial_mass.append(simulator.robot_skeleton.bodynodes[2+i].m)

    def get_simulator_parameters(self):
        cur_friction = self.simulator.dart_world.skeletons[0].bodynodes[0].friction_coeff()
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        cur_mass1 = self.simulator.robot_skeleton.bodynodes[2].m - self.initial_mass[0]
        mass_param1 = (cur_mass1 - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0])
        cur_mass2 = self.simulator.robot_skeleton.bodynodes[3].m - self.initial_mass[1]
        mass_param2 = (cur_mass2 - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0])
        cur_mass3 = self.simulator.robot_skeleton.bodynodes[4].m - self.initial_mass[2]
        mass_param3 = (cur_mass3 - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0])
        cur_mass4 = self.simulator.robot_skeleton.bodynodes[5].m - self.initial_mass[3]
        mass_param4 = (cur_mass4 - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0])

        return np.array([friction_param, mass_param1])#, mass_param2, mass_param3, mass_param4])

    def set_simulator_parameters(self, x):
        friction = x[0] * (self.range[1] - self.range[0]) + self.range[0]
        self.simulator.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(friction)

        mass = x[1] * (self.mass_range[1] - self.mass_range[0]) + self.mass_range[0] + self.initial_mass[0]
        self.simulator.robot_skeleton.bodynodes[2].set_mass(mass)

        '''mass = x[2] * (self.mass_range[1] - self.mass_range[0]) + self.mass_range[0] + self.initial_mass[1]
        self.simulator.robot_skeleton.bodynodes[3].set_mass(mass)

        mass = x[3] * (self.mass_range[1] - self.mass_range[0]) + self.mass_range[0] + self.initial_mass[2]
        self.simulator.robot_skeleton.bodynodes[4].set_mass(mass)

        mass = x[4] * (self.mass_range[1] - self.mass_range[0]) + self.mass_range[0] + self.initial_mass[3]
        self.simulator.robot_skeleton.bodynodes[5].set_mass(mass)'''

    def resample_parameters(self):
        #x = np.random.uniform(0, 1, len(self.get_simulator_parameters()))
        x = np.random.normal(0, 0.2, 2) % 1
        self.set_simulator_parameters(x)

class hopperContactMassFootUpperLimitManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.3, 1.0] # friction range
        self.torso_mass_range = [3.0, 6.0]
        self.limit_range = [-0.2, 0.2]
        self.param_dim = 2+1
        self.initial_up_limit = self.simulator.robot_skeleton.joints[-3].position_upper_limit(0)

    def get_simulator_parameters(self):
        cur_friction = self.simulator.dart_world.skeletons[0].bodynodes[0].friction_coeff()
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        cur_mass = self.simulator.robot_skeleton.bodynodes[2].m
        mass_param = (cur_mass - self.torso_mass_range[0]) / (self.torso_mass_range[1] - self.torso_mass_range[0])

        # use upper limit of
        limit_diff1 = self.simulator.robot_skeleton.joints[-3].position_upper_limit(0) - self.initial_up_limits
        limit_diff1 = (limit_diff1 - self.limit_range[0]) / (self.limit_range[1] - self.limit_range[0])

        return np.array([friction_param, mass_param, limit_diff1])

    def set_simulator_parameters(self, x):
        friction = x[0] * (self.range[1] - self.range[0]) + self.range[0]
        self.simulator.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(friction)

        mass = x[1] * (self.torso_mass_range[1] - self.torso_mass_range[0]) + self.torso_mass_range[0]
        self.simulator.robot_skeleton.bodynodes[2].set_mass(mass)

        limit_diff1 = x[2] * (self.limit_range[1] - self.limit_range[0]) + self.limit_range[0] + self.initial_up_limits

        self.simulator.robot_skeleton.joints[-3].set_position_upper_limit(0, limit_diff1)

    def resample_parameters(self):
        #x = np.random.uniform(0, 1, len(self.get_simulator_parameters()))
        x = np.random.normal(0, 0.2, 2) % 1
        self.set_simulator_parameters(x)

class hopperContactMassAllLimitManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.3, 1.0] # friction range
        self.torso_mass_range = [3.0, 6.0]
        self.limit_range = [-0.3, 0.3]
        self.param_dim = 2+4
        self.initial_up_limits = []
        self.initial_low_limits = []
        for i in range(3):
            self.initial_up_limits.append(simulator.robot_skeleton.joints[-3+i].position_upper_limit(0))
            self.initial_low_limits.append(simulator.robot_skeleton.joints[-3+i].position_lower_limit(0))

    def get_simulator_parameters(self):
        cur_friction = self.simulator.dart_world.skeletons[0].bodynodes[0].friction_coeff()
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        cur_mass = self.simulator.robot_skeleton.bodynodes[2].m
        mass_param = (cur_mass - self.torso_mass_range[0]) / (self.torso_mass_range[1] - self.torso_mass_range[0])

        # use upper limit of
        limit_diff1 = self.simulator.robot_skeleton.joints[-3].position_upper_limit(0) - self.initial_up_limits[0]
        limit_diff2 = self.simulator.robot_skeleton.joints[-2].position_upper_limit(0) - self.initial_up_limits[1]
        limit_diff3 = self.simulator.robot_skeleton.joints[-1].position_upper_limit(0) - self.initial_up_limits[2]
        limit_diff4 = self.simulator.robot_skeleton.joints[-1].position_lower_limit(0) - self.initial_low_limits[2]
        limit_diff1 = (limit_diff1 - self.limit_range[0]) / (self.limit_range[1] - self.limit_range[0])
        limit_diff2 = (limit_diff2 - self.limit_range[0]) / (self.limit_range[1] - self.limit_range[0])
        limit_diff3 = (limit_diff3 - self.limit_range[0]) / (self.limit_range[1] - self.limit_range[0])
        limit_diff4 = (limit_diff4 - self.limit_range[0]) / (self.limit_range[1] - self.limit_range[0])

        return np.array([friction_param, mass_param, limit_diff1, limit_diff2, limit_diff3, limit_diff4])

    def set_simulator_parameters(self, x):
        friction = x[0] * (self.range[1] - self.range[0]) + self.range[0]
        for i in range(len(self.simulator.dart_world.skeletons[0].bodynodes)):
            self.simulator.dart_world.skeletons[0].bodynodes[i].set_friction_coeff(friction)

        mass = x[1] * (self.torso_mass_range[1] - self.torso_mass_range[0]) + self.torso_mass_range[0]
        self.simulator.robot_skeleton.bodynodes[2].set_mass(mass)

        limit_diff1 = x[2] * (self.limit_range[1] - self.limit_range[0]) + self.limit_range[0] + self.initial_up_limits[0]
        limit_diff2 = x[3] * (self.limit_range[1] - self.limit_range[0]) + self.limit_range[0] + self.initial_up_limits[1]
        limit_diff3 = x[4] * (self.limit_range[1] - self.limit_range[0]) + self.limit_range[0] + self.initial_up_limits[2]
        limit_diff4 = x[5] * (self.limit_range[1] - self.limit_range[0]) + self.limit_range[0] + self.initial_low_limits[2]

        self.simulator.robot_skeleton.joints[-3].set_position_upper_limit(0, limit_diff1)
        self.simulator.robot_skeleton.joints[-2].set_position_upper_limit(0, limit_diff2)
        self.simulator.robot_skeleton.joints[-1].set_position_upper_limit(0, limit_diff3)
        self.simulator.robot_skeleton.joints[-1].set_position_lower_limit(0, limit_diff4)

    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, len(self.get_simulator_parameters()))
        #x = np.random.normal(0, 0.2, 2) % 1
        self.set_simulator_parameters(x)

##############################################################################################################
##############################################################################################################




##############################################################################################################
################################  CartPole SwingUp #####################################################################
##############################################################################################################
class CartPoleManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.05, 10.0] # mass range
        self.joint_damping = [0.0, 1.0]
        self.actuator_strength = [20, 50]
        self.attach_width = [0.2, 0.3]
        self.jug_mass = [0.2, 3.0]

        self.activated_param = [0]
        self.controllable_param = [0]
        self.param_dim = len(self.activated_param)
        self.sampling_selector = None
        self.selector_target = -1

    def get_simulator_parameters(self):
        cur_mass = self.simulator.dart_world.skeletons[-1].bodynodes[2].mass()
        mass_param = (cur_mass - self.range[0]) / (self.range[1] - self.range[0])

        cur_damping = self.simulator.robot_skeleton.joints[1].damping_coefficient(0)
        damping_param = (cur_damping - self.joint_damping[0]) / (self.joint_damping[1] - self.joint_damping[0])

        cur_strength = self.simulator.action_scale[0]
        strength_param = (cur_strength - self.actuator_strength[0]) / (self.actuator_strength[1] - self.actuator_strength[0])

        width = self.simulator.robot_skeleton.bodynodes[-1].shapenodes[0].shape.size()[0]
        width_param = (width - self.attach_width[0]) / (self.attach_width[1] - self.attach_width[0])

        jug_mass = self.simulator.dart_world.skeletons[1].bodynodes[0].mass()
        jug_mass_param = (jug_mass - self.jug_mass[0]) / (self.jug_mass[1] - self.jug_mass[0])

        return np.array([mass_param, damping_param, strength_param, width_param, jug_mass_param])[self.activated_param]

    def set_simulator_parameters(self, x):
        cur_id = 0
        if 0 in self.controllable_param:
            mass = x[cur_id] * (self.range[1] - self.range[0]) + self.range[0]
            self.simulator.dart_world.skeletons[-1].bodynodes[2].set_mass(mass)
            cur_id += 1
        if 1 in self.controllable_param:
            damping = x[cur_id] * (self.joint_damping[1] - self.joint_damping[0]) + self.joint_damping[0]
            self.simulator.robot_skeleton.joints[1].set_damping_coefficient(0, damping)
            cur_id += 1
        if 2 in self.controllable_param:
            strength = x[cur_id] * (self.actuator_strength[1] - self.actuator_strength[0]) + self.actuator_strength[0]
            self.simulator.action_scale[0] = strength
            cur_id += 1
        if 3 in self.controllable_param:
            width = x[cur_id] * (self.attach_width[1] - self.attach_width[0]) + self.attach_width[0]
            size = np.copy(self.simulator.robot_skeleton.bodynodes[-1].shapenodes[0].shape.size())
            size[0] = width
            for i in range(len(self.simulator.robot_skeleton.bodynodes[-1].shapenodes)):
                self.simulator.robot_skeleton.bodynodes[-1].shapenodes[i].shape.set_size(size)
            size = np.copy(size ** 2)
            mass = self.simulator.dart_world.skeletons[-1].bodynodes[2].mass()
            self.simulator.robot_skeleton.bodynodes[-1].set_inertia_entries(1.0/12*mass*(size[1]+size[2]), 1.0/12*mass*(size[0]+size[2]), 1.0/12*mass*(size[1]+size[0]))
            cur_id += 1
        if 4 in self.controllable_param:
            jug_mass = x[cur_id] * (self.jug_mass[1] - self.jug_mass[0]) + self.jug_mass[0]
            self.simulator.dart_world.skeletons[1].bodynodes[0].set_mass(jug_mass)
            self.simulator.dart_world.skeletons[2].bodynodes[0].set_mass(jug_mass)
            cur_id += 1

    def resample_parameters(self):
        x = np.random.uniform(0.0, 1.0, len(self.get_simulator_parameters()))
        if self.sampling_selector is not None:
            while not self.sampling_selector.classify(np.array([x])) == self.selector_target:
                x = np.random.uniform(0, 1, len(self.get_simulator_parameters()))
        self.set_simulator_parameters(x)

class walker2dParamManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.4, 1.0] # friction range
        self.restitution_range = [0.0, 0.3]
        self.backpack_mass_range = [0.05, 10.0]
        self.activated_param = [2]
        self.controllable_param = [2]

        self.param_dim = len(self.activated_param)
        self.sampling_selector = None
        self.selector_target = -1

    def get_simulator_parameters(self):
        cur_friction = self.simulator.dart_world.skeletons[0].bodynodes[0].friction_coeff()
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        cur_rest = self.simulator.dart_world.skeletons[0].bodynodes[0].restitution_coeff()
        restitution_param = (cur_rest - self.restitution_range[0]) / (self.restitution_range[1] - self.restitution_range[0])

        cur_mass = self.simulator.robot_skeleton.bodynodes[3].m
        mass_param = (cur_mass - self.backpack_mass_range[0]) / (self.backpack_mass_range[1] - self.backpack_mass_range[0])

        return np.array([friction_param, restitution_param, mass_param])[self.activated_param]

    def set_simulator_parameters(self, x):
        cur_id = 0
        if 0 in self.controllable_param:
            friction = x[cur_id] * (self.range[1] - self.range[0]) + self.range[0]
            self.simulator.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(friction)
            cur_id += 1
        if 1 in self.controllable_param:
            restitution = x[cur_id] * (self.restitution_range[1] - self.restitution_range[0]) + self.restitution_range[0]
            self.simulator.dart_world.skeletons[0].bodynodes[0].set_restitution_coeff(restitution)
            self.simulator.dart_world.skeletons[1].bodynodes[-1].set_restitution_coeff(1.0)
            cur_id += 1
        if 2 in self.controllable_param:
            mass = x[cur_id] * (self.backpack_mass_range[1] - self.backpack_mass_range[0]) + self.backpack_mass_range[0]
            self.simulator.robot_skeleton.bodynodes[3].set_mass(mass)
            cur_id += 1


    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, len(self.get_simulator_parameters()))
        if self.sampling_selector is not None:
            while not self.sampling_selector.classify(np.array([x])) == self.selector_target:
                x = np.random.uniform(0, 1, len(self.get_simulator_parameters()))
        self.set_simulator_parameters(x)
