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
        self.restitution_range = [0.0, 0.3]
        self.mass_range = [2.0, 15.0]
        self.damping_range = [0.5, 3.0]
        self.power_range = [100, 300]
        self.ankle_range = [60, 300]
        self.velrew_weight_range = [-1.0, 1.0]
        self.com_offset_range = [-0.05, 0.05]

        self.activated_param = [0,1]#[0, 2,3,4,5, 6,7,8, 9, 12,13,14,15]
        self.controllable_param = [0,1]#[0, 2,3,4,5, 6,7,8, 9, 12,13,14,15]
        
        self.binned_param = 0 # don't bin if = 0

        self.param_dim = len(self.activated_param)
        self.sampling_selector = None
        self.selector_target = -1

    def get_simulator_parameters(self):
        cur_friction = self.simulator.dart_world.skeletons[0].bodynodes[0].friction_coeff()
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        cur_rest = self.simulator.dart_world.skeletons[0].bodynodes[0].restitution_coeff()
        restitution_param = (cur_rest - self.restitution_range[0]) / (self.restitution_range[1] - self.restitution_range[0])

        mass_param = []
        for bid in range(2, 6):
            cur_mass = self.simulator.robot_skeleton.bodynodes[bid].m
            mass_param.append((cur_mass - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0]))

        damp_param = []
        for jid in range(3, 6):
            cur_damp = self.simulator.robot_skeleton.joints[jid].damping_coefficient(0)
            damp_param.append((cur_damp - self.damping_range[0]) / (self.damping_range[1] - self.damping_range[0]))

        cur_power = self.simulator.action_scale
        power_param = (cur_power[0] - self.power_range[0]) / (self.power_range[1] - self.power_range[0])

        cur_ank_power = self.simulator.action_scale[2]
        ank_power_param = (cur_ank_power - self.ankle_range[0]) / (self.ankle_range[1] - self.ankle_range[0])

        cur_velrew_weight = self.simulator.velrew_weight
        velrew_param = (cur_velrew_weight - self.velrew_weight_range[0]) / (self.velrew_weight_range[1] - self.velrew_weight_range[0])

        com_param = []
        for bid in range(2, 6):
            if bid != 5:
                cur_com = self.simulator.robot_skeleton.bodynodes[bid].local_com()[1] - self.simulator.initial_local_coms[bid][1]
            else:
                cur_com = self.simulator.robot_skeleton.bodynodes[bid].local_com()[0] - self.simulator.initial_local_coms[bid][0]
            com_param.append((cur_com - self.com_offset_range[0]) / (self.com_offset_range[1] - self.com_offset_range[0]))

        params = np.array([friction_param, restitution_param]+ mass_param + damp_param +
                          [power_param, ank_power_param, velrew_param] + com_param)[self.activated_param]
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
        for bid in range(2, 6):
            if bid in self.controllable_param:
                mass = x[cur_id] * (self.mass_range[1] - self.mass_range[0]) + self.mass_range[0]
                self.simulator.robot_skeleton.bodynodes[bid].set_mass(mass)
                cur_id += 1
        for jid in range(6, 9):
            if jid in self.controllable_param:
                damp = x[cur_id] * (self.damping_range[1] - self.damping_range[0]) + self.damping_range[0]
                self.simulator.robot_skeleton.joints[jid - 3].set_damping_coefficient(0, damp)
                cur_id += 1
        if 9 in self.controllable_param:
            power = x[cur_id] * (self.power_range[1] - self.power_range[0]) + self.power_range[0]
            self.simulator.action_scale = np.array([power, power, power])
            cur_id += 1
        if 10 in self.controllable_param:
            ankpower = x[cur_id] * (self.ankle_range[1] - self.ankle_range[0]) + self.ankle_range[0]
            self.simulator.action_scale[2] = ankpower
            cur_id += 1
        if 11 in self.controllable_param:
            velrew_weight = x[cur_id] * (self.velrew_weight_range[1] - self.velrew_weight_range[0]) + self.velrew_weight_range[0]
            self.simulator.velrew_weight = velrew_weight
            cur_id += 1

        for bid in range(2, 6):
            if bid+10 in self.controllable_param:
                com = x[cur_id] * (self.com_offset_range[1] - self.com_offset_range[0]) + self.com_offset_range[0]
                init_com = np.copy(self.simulator.initial_local_coms[bid])
                if bid != 5:
                    init_com[1] += com
                else:
                    init_com[0] += com
                self.simulator.robot_skeleton.bodynodes[bid].set_local_com(init_com)
                cur_id += 1


    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, len(self.get_simulator_parameters()))
        if self.sampling_selector is not None:
            while not self.sampling_selector.classify(np.array([x])) == self.selector_target:
                x = np.random.uniform(0, 1, len(self.get_simulator_parameters()))
        self.set_simulator_parameters(x)


class mjHopperManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.2, 1.0]  # friction range
        self.mass_range = [2.0, 20.0]
        self.damping_range = [0.15, 2.0]
        self.power_range = [150, 500]
        self.velrew_weight_range = [-1.0, 1.0]
        self.restitution_range = [0.5, 1.0]
        self.solimp_range = [0.8, 0.99]
        self.solref_range = [0.001, 0.02]
        self.armature_range = [0.05, 0.98]

        self.activated_param = [0, 1,2,3,4, 5,6,7, 8, 10, 11, 12, 13]
        self.controllable_param = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]

        self.param_dim = len(self.activated_param)
        self.sampling_selector = None
        self.selector_target = -1

    def get_simulator_parameters(self):
        cur_friction = self.simulator.model.geom_friction[-1][0]
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        mass_param = []
        for bid in range(1, 5):
            cur_mass = self.simulator.model.body_mass[bid]
            mass_param.append((cur_mass - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0]))

        damp_param = []
        for jid in range(3, 6):
            cur_damp = self.simulator.model.dof_damping[jid]
            damp_param.append((cur_damp - self.damping_range[0]) / (self.damping_range[1] - self.damping_range[0]))

        cur_power = self.simulator.model.actuator_gear[0][0]
        power_param = (cur_power - self.power_range[0]) / (self.power_range[1] - self.power_range[0])

        cur_velrew_weight = self.simulator.velrew_weight
        velrew_param = (cur_velrew_weight - self.velrew_weight_range[0]) / (
                self.velrew_weight_range[1] - self.velrew_weight_range[0])

        cur_restitution = self.simulator.model.geom_solref[-1][1]
        rest_param = (cur_restitution - self.restitution_range[0]) / (self.restitution_range[1] - self.restitution_range[0])

        cur_solimp = self.simulator.model.geom_solimp[-1][0]
        solimp_param = (cur_solimp - self.solimp_range[0]) / (self.solimp_range[1] - self.solimp_range[0])

        cur_solref = self.simulator.model.geom_solref[-1][0]
        solref_param = (cur_solref - self.solref_range[0]) / (self.solref_range[1] - self.solref_range[0])

        cur_armature = self.simulator.model.dof_armature[-1]
        armature_param = (cur_armature - self.armature_range[0]) / (self.armature_range[1] - self.armature_range[0])

        params = np.array([friction_param] + mass_param + damp_param + [power_param, velrew_param, rest_param
                                                                        ,solimp_param, solref_param, armature_param])[self.activated_param]
        return params

    def set_simulator_parameters(self, x):
        cur_id = 0
        if 0 in self.controllable_param:
            friction = x[cur_id] * (self.range[1] - self.range[0]) + self.range[0]
            self.simulator.model.geom_friction[-1][0] = friction
            cur_id += 1
        for bid in range(1, 5):
            if bid in self.controllable_param:
                mass = x[cur_id] * (self.mass_range[1] - self.mass_range[0]) + self.mass_range[0]
                self.simulator.model.body_mass[bid] = mass
                cur_id += 1
        for jid in range(5, 8):
            if jid in self.controllable_param:
                damp = x[cur_id] * (self.damping_range[1] - self.damping_range[0]) + self.damping_range[0]
                self.simulator.model.dof_damping[jid - 2] = damp
                cur_id += 1
        if 8 in self.controllable_param:
            power = x[cur_id] * (self.power_range[1] - self.power_range[0]) + self.power_range[0]
            self.simulator.model.actuator_gear[0][0] = power
            self.simulator.model.actuator_gear[1][0] = power
            self.simulator.model.actuator_gear[2][0] = power
            cur_id += 1
        if 9 in self.controllable_param:
            velrew_weight = x[cur_id] * (self.velrew_weight_range[1] - self.velrew_weight_range[0]) + \
                            self.velrew_weight_range[0]
            self.simulator.velrew_weight = velrew_weight
            cur_id += 1
        if 10 in self.controllable_param:
            restitution = x[cur_id] * (self.restitution_range[1] - self.restitution_range[0]) + \
                            self.restitution_range[0]
            for bn in range(len(self.simulator.model.geom_solref)):
                self.simulator.model.geom_solref[bn][1] = restitution
            cur_id += 1
        if 11 in self.controllable_param:
            solimp = x[cur_id] * (self.solimp_range[1] - self.solimp_range[0]) + \
                            self.solimp_range[0]
            for bn in range(len(self.simulator.model.geom_solimp)):
                self.simulator.model.geom_solimp[bn][0] = solimp
                self.simulator.model.geom_solimp[bn][1] = solimp
            cur_id += 1
        if 12 in self.controllable_param:
            solref = x[cur_id] * (self.solref_range[1] - self.solref_range[0]) + \
                            self.solref_range[0]
            for bn in range(len(self.simulator.model.geom_solref)):
                self.simulator.model.geom_solref[bn][0] = solref
            cur_id += 1
        if 13 in self.controllable_param:
            armature = x[cur_id] * (self.armature_range[1] - self.armature_range[0]) + \
                            self.armature_range[0]
            for dof in range(3, 6):
                self.simulator.model.dof_armature[dof] = armature
            cur_id += 1

    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, len(self.get_simulator_parameters()))
        if self.sampling_selector is not None:
            while not self.sampling_selector.classify(np.array([x])) == self.selector_target:
                x = np.random.uniform(0, 1, len(self.get_simulator_parameters()))
        self.set_simulator_parameters(x)


##############################################################################################################
################################  CartPole SwingUp #####################################################################
##############################################################################################################
class CartPoleManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.05, 10.0] # mass range
        self.joint_damping = [0.0, 1.0]
        self.actuator_strength = [20, 50]
        self.attach_width = [0.05, 0.3]
        self.cartmass = [0.05, 5.0]
        self.jug_mass = [0.2, 3.0]

        self.activated_param = [0, 1, 2, 3, 4]
        self.controllable_param = [0, 1, 2, 3, 4]
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

        cur_cartmass = self.simulator.dart_world.skeletons[-1].bodynodes[0].mass()
        cartmass_param = (cur_cartmass - self.cartmass[0]) / (self.cartmass[1] - self.cartmass[0])

        jug_mass = self.simulator.dart_world.skeletons[1].bodynodes[0].mass()
        jug_mass_param = (jug_mass - self.jug_mass[0]) / (self.jug_mass[1] - self.jug_mass[0])

        return np.array([mass_param, damping_param, strength_param, width_param, cartmass_param, jug_mass_param])[self.activated_param]

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
            mass = x[cur_id] * (self.cartmass[1] - self.cartmass[0]) + self.cartmass[0]
            self.simulator.dart_world.skeletons[-1].bodynodes[0].set_mass(mass)
            cur_id += 1
        if 5 in self.controllable_param:
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
        self.mass_range = [2.0, 10.0]
        self.damping_range = [0.5, 3.0]
        self.friction_range = [0.2, 1.0] # friction range
        self.restitution_range = [0.0, 0.3]
        self.power_range = [150, 250]
        self.up_noise_range = [0.0, 1.0]

        #self.obs_delay = [0.0, 2.5]
        #self.act_delay = [0.0, 2.5]
        self.activated_param = [0, 15, 16] #[0,1,2,3,4,5,6,  7,8,9,10,11,12,  13, 14, 15]
        self.controllable_param = [0, 15, 16] #[0,1,2,3,4,5,6,  7,8,9,10,11,12,  13, 14, 15]

        self.param_dim = len(self.activated_param)
        self.sampling_selector = None
        self.selector_target = -1

    def get_simulator_parameters(self):
        mass_param = []
        for bid in range(2, 9):
            cur_mass = self.simulator.robot_skeleton.bodynodes[bid].m
            mass_param.append((cur_mass - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0]))

        damp_param = []
        for jid in range(3, 9):
            cur_damp = self.simulator.robot_skeleton.joints[jid].damping_coefficient(0)
            damp_param.append((cur_damp - self.damping_range[0]) / (self.damping_range[1] - self.damping_range[0]))\

        cur_friction = self.simulator.dart_world.skeletons[0].bodynodes[0].friction_coeff()
        friction_param = (cur_friction - self.friction_range[0]) / (self.friction_range[1] - self.friction_range[0])

        cur_rest = self.simulator.dart_world.skeletons[0].bodynodes[0].restitution_coeff()
        restitution_param = (cur_rest - self.restitution_range[0]) / (self.restitution_range[1] - self.restitution_range[0])

        cur_power = self.simulator.action_scale
        power_param = (cur_power - self.power_range[0]) / (self.power_range[1] - self.power_range[0])

        cur_up_noise = self.simulator.UP_noise_level
        up_noise_param = (cur_up_noise - self.up_noise_range[0]) / (self.up_noise_range[1] - self.up_noise_range[0])

        return np.array(mass_param+damp_param+[friction_param, restitution_param, power_param, up_noise_param])[self.activated_param]

    def set_simulator_parameters(self, x):
        cur_id = 0

        for bid in range(0, 7):
            if bid in self.controllable_param:
                mass = x[cur_id] * (self.mass_range[1] - self.mass_range[0]) + self.mass_range[0]
                self.simulator.robot_skeleton.bodynodes[bid+2].set_mass(mass)
                cur_id += 1
        for jid in range(7, 13):
            if jid in self.controllable_param:
                damp = x[cur_id] * (self.damping_range[1] - self.damping_range[0]) + self.damping_range[0]
                self.simulator.robot_skeleton.joints[jid - 5].set_damping_coefficient(0, damp)
                cur_id += 1

        if 13 in self.controllable_param:
            friction = x[cur_id] * (self.friction_range[1] - self.friction_range[0]) + self.friction_range[0]
            self.simulator.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(friction)
            cur_id += 1
        if 14 in self.controllable_param:
            restitution = x[cur_id] * (self.restitution_range[1] - self.restitution_range[0]) + self.restitution_range[0]
            self.simulator.dart_world.skeletons[0].bodynodes[0].set_restitution_coeff(restitution)
            self.simulator.dart_world.skeletons[1].bodynodes[-1].set_restitution_coeff(1.0)
            cur_id += 1
        if 15 in self.controllable_param:
            power = x[cur_id] * (self.power_range[1] - self.power_range[0]) + self.power_range[0]
            self.simulator.action_scale = power
            cur_id += 1
        if 16 in self.controllable_param:
            up_noise = x[cur_id] * (self.up_noise_range[1] - self.up_noise_range[0]) + self.up_noise_range[0]
            self.simulator.UP_noise_level = up_noise
            cur_id += 1


    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, len(self.get_simulator_parameters()))
        self.set_simulator_parameters(x)
