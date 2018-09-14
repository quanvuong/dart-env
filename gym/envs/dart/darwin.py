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
import time

class DartDarwinTrajEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        
        
        obs_dim = 48

        self.control_bounds = np.array([np.ones(20,), np.ones(20,)])
        # LOWER CONTROL BOUNDS
        self.control_bounds[0] = [-1.0]*20
        self.control_bounds[1] = [1.0]*20

        self.control_limits_low = np.array([-np.pi/2, -np.pi/4,0,-np.pi/2,0,-np.pi/2,0.0,0.0,-np.pi/5,-np.pi/3,-np.pi/6,-np.pi/2,-np.pi/4,-np.pi/4,-np.pi/2,0,-np.pi/2,0,-np.pi/4,-np.pi/4])
        self.control_limits_high = np.array([np.pi/2,0,np.pi/2,np.pi/2,np.pi/4,0,0.0,0.0,np.pi/20,0,np.pi/2,0,np.pi/4,np.pi/4,np.pi/5,np.pi/3,np.pi/6,np.pi/2,np.pi/4,np.pi/4])

        self.ndofs = 26


        self.t = 0
        self.preverror = np.zeros(self.ndofs,)
        self.edot = np.zeros(self.ndofs,)
        self.target = np.zeros(self.ndofs,)
        self.tau = np.zeros(self.ndofs,)
        self.init = np.zeros(self.ndofs,)
        self.sum = 0
        self.count = 0
        self.dumpTorques = False
        self.dumpActions = False

        self.target_vel = 0.0
        self.init_tv = 0.0
        self.final_tv = 0.2
        self.tv_endtime = 0.01
        self.smooth_tv_change = True
        self.avg_rew_weighting = []
        self.vel_cache = []
        self.target_vel_cache = []

        self.alive_bonus = 4.0
        self.energy_weight = 0.01
        self.vel_reward_weight = 10.0

        self.assist_timeout = 10.0
        self.assist_schedule = [[0.0, [20000, 2000]], [3.0, [15000, 1500]], [6.0, [11250.0, 1125.0]]]
        self.init_balance_pd = 2000.0
        self.init_vel_pd = 2000.0

        self.cur_step = 0

        self.include_obs_history = 1
        self.include_act_history = 0
        obs_dim *= self.include_obs_history
        obs_dim += len(self.control_bounds[0]) * self.include_act_history
        obs_perm_base = np.array(
            [-3,-4,-5  ,-0.0001,-1,-2,  6,7,  -14,-15,-16,-17,-18,-19,   -8,-9,-10,-11,-12,-13,
             -23,-24,-25,  -20,-21,-22,  26,27,  -34,-35,-36,-37,-38,-39,  -28,-29,-30,-31,-32,-33,
             -40,41,   42,-43,44,-45,46,-47])
        act_perm_base = np.array([-3,-4,-5  ,-0.0001,-1,-2,  6,7,  -14,-15,-16,-17,-18,-19,   -8,-9,-10,-11,-12,-13])
        self.obs_perm = np.copy(obs_perm_base)

        for i in range(self.include_obs_history - 1):
            self.obs_perm = np.concatenate(
                [self.obs_perm, np.sign(obs_perm_base) * (np.abs(obs_perm_base) + len(self.obs_perm))])
        for i in range(self.include_act_history):
            self.obs_perm = np.concatenate(
                [self.obs_perm, np.sign(act_perm_base) * (np.abs(act_perm_base) + len(self.obs_perm))])
        self.act_perm = np.copy(act_perm_base)

        dart_env.DartEnv.__init__(self, ['darwinmodel/ground1.urdf','darwinmodel/robotis_op2.urdf'], 15, obs_dim, self.control_bounds, disableViewer=True)

        self.dart_world.set_gravity([0,0,-9.81])

        self.dart_world.set_collision_detector(0)

        self.robot_skeleton.set_self_collision_check(False)

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

    def advance(self,a):
        clamped_control = np.array(a)

        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
            if clamped_control[i] < self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]

        clamped_control = (clamped_control + 1.0 ) * 0.5 * (self.control_limits_high - self.control_limits_low) + self.control_limits_low

        self.target[6:]= clamped_control# + #*self.action_scale# + self.ref_trajectory_right[self.count_right,6:]# + 

        actions = np.zeros(26,)
        
        actions[6:] = copy.deepcopy(self.target[6:])

        for i in range(15):
            self.tau[6:] = self.PID()
            
            if self.t < self.assist_timeout:
                force = self._bodynode_spd(self.robot_skeleton.bodynode('MP_NECK'), self.current_pd, 1)
                self.robot_skeleton.bodynode('MP_NECK').add_ext_force(np.array([0, force, 0]))

                force = self._bodynode_spd(self.robot_skeleton.bodynode('MP_NECK'), self.vel_enforce_kp, 0, self.target_vel)
                self.robot_skeleton.bodynode('MP_NECK').add_ext_force(np.array([force, 0, 0]))

            self.robot_skeleton.set_forces(self.tau)

            self.dart_world.step()


        #self.do_simulation(self.tau, self.frame_skip)
    def PID(self):
        #print("#########################################################################3")

        self.kp = [2.1,1,79,4.93,2.0,2.02,1.98,2.2,2.06,148,152,150,136,153,147,151,151.4,150.45,151.36,154,150.2]
        self.kd = [0.21,0.23,0.22,0.25,0.21,0.26,0.28,0.213,0.192,0.198,0.22,0.199,0.2,0.5,0.53,0.27,0.21,0.205,0.52,0.56]

        
        
        #self.kp = [item  for item in self.kp]
        #self.kd = [item  for item in self.kd]

        q = self.robot_skeleton.q
        qdot = self.robot_skeleton.dq
        tau = np.zeros(26,)
        for i in range(6, 26):
            #print(q.shape)
            self.edot[i] = ((q[i] - self.target[i]) -
                self.preverror[i]) / self.dt
            tau[i] = -self.kp[i - 6] * \
                (q[i] - self.target[i]) - \
                self.kd[i - 6] *qdot[i]
            self.preverror[i] = (q[i] - self.target[i])
        
        torqs = self.ClampTorques(tau)
        
        return torqs[6:]
        

    def ClampTorques(self,torques):
        torqueLimits = 9.0
        
        for i in range(6,26):
            if torques[i] > torqueLimits:#
                torques[i] = torqueLimits 
            if torques[i] < -torqueLimits:
                torques[i] = -torqueLimits

        return torques



    def step(self, a):
        if self.smooth_tv_change:
            self.target_vel = (np.min([self.t, self.tv_endtime]) / self.tv_endtime) * (self.final_tv - self.init_tv) + self.init_tv

        self.current_pd = self.init_balance_pd
        self.vel_enforce_kp = self.init_vel_pd

        if len(self.assist_schedule) > 0:
            for sch in self.assist_schedule:
                if self.t > sch[0]:
                    self.current_pd = sch[1][0]
                    self.vel_enforce_kp = sch[1][1]


        posbefore = self.robot_skeleton.bodynode('MP_NECK').C[0]
        self.advance(a)
        posafter = self.robot_skeleton.bodynode('MP_NECK').C[0]

        vel = (posafter - posbefore) / self.dt
        self.vel_cache.append(vel)
        self.target_vel_cache.append(self.target_vel)

        if len(self.vel_cache) > int(2.0 / self.dt):
            self.vel_cache.pop(0)
            self.target_vel_cache.pop(0)

        vel_rew = -self.vel_reward_weight * np.abs(np.mean(self.target_vel_cache) - np.mean(self.vel_cache))
        if self.t < self.tv_endtime:
            vel_rew *= 0.5

        reward =  -self.energy_weight*np.sum(a)**2 + vel_rew + self.alive_bonus

        s = self.state_vector()
        com_height = self.robot_skeleton.bodynodes[0].com()[2]
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 200).all() and (com_height > -0.60) and (self.robot_skeleton.q[0] > -0.6) and (self.robot_skeleton.q[0]<0.6) and (abs(self.robot_skeleton.q[1]) < 0.50) and (abs(self.robot_skeleton.q[2]) < 0.50))
        
        if done:
            reward =0

        self.t += self.dt
        self.cur_step += 1
    
        ob = self._get_obs()

        return ob, reward, done, {'avg_vel':np.mean(self.vel_cache)}  #{'pre_state':pre_state, 'vel_rew':vel_rew, 'action_pen':action_pen, 'joint_pen':joint_pen, 'deviation_pen':deviation_pen, 'aux_pred':np.hstack([com_foot_offset1, com_foot_offset2, [reward]]), 'done_return':done}

    def _get_obs(self):
        
        state =  np.concatenate([self.robot_skeleton.q[6:],self.robot_skeleton.dq[6:]])

        state += np.random.uniform(-0.01, 0.01, len(state))

        #state = np.concatenate([self.robot_skeleton.q[6:], self.robot_skeleton.dq[6:]])
        #state[:20] +=  np.random.uniform(low=-0.01,high=0.01,size=20)
        #state[20:] +=  np.random.uniform(low=-0.05,high=0.05,size=1)

        body_com = self.robot_skeleton.bodynode('MP_NECK').C
        bodyvel = np.dot(self.robot_skeleton.bodynode('MP_NECK').world_jacobian(offset=self.robot_skeleton.bodynode('MP_NECK').local_com()), self.robot_skeleton.dq)
        state = np.concatenate([body_com[1:], bodyvel, state])

        return state

    def reset_model(self):
        self.dart_world.reset()
        qpos = np.zeros(self.robot_skeleton.ndofs) #self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = np.zeros(self.robot_skeleton.ndofs) +  self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs) #self.robot_skeleton.dq +
        
        qpos[1] = 0.25
        qpos[5] = -0.37
        #LEFT HAND
        qpos[6] = (2518-2048)*(np.pi/180)*0.088 + np.random.uniform(low=-0.01,high=0.01,size=1)
        qpos[7]  = (2248-2048)*(np.pi/180)*0.088+ np.random.uniform(low=-0.01,high=0.01,size=1)
        qpos[8] = (1712-2048)*(np.pi/180)*0.088+ np.random.uniform(low=-0.01,high=0.01,size=1)

        #RIGHT HAND 
        qpos[9] = (1498-2048)*(np.pi/180)*0.088+ np.random.uniform(low=-0.01,high=0.01,size=1)
        qpos[10] = (1845-2048)*(np.pi/180)*0.088+ np.random.uniform(low=-0.01,high=0.01,size=1)
        qpos[11]= (2381-2048)*(np.pi/180)*0.088+ np.random.uniform(low=-0.01,high=0.01,size=1)

        #LEFT LEG
        qpos[14] = 0 # yaw
        qpos[15] = (2052-2048)*(np.pi/180)*0.088+ np.random.uniform(low=-0.01,high=0.01,size=1)
        qpos[16] = 0.68 + np.random.uniform(low=-0.01,high=0.01,size=1)
        qpos[17] = -0.88 + np.random.uniform(low=-0.01,high=0.01,size=1)
        qpos[18] = (1707-2048)*(np.pi/180)*0.088+ np.random.uniform(low=-0.01,high=0.01,size=1)
        qpos[19] = (2039 - 2048)*(np.pi/180)*0.088 + np.random.uniform(low=-0.01,high=0.01,size=1)

        #RIGHT LEG               
        qpos[20] = 0
        qpos[21] = (2044-2048)*(np.pi/180)*0.088 + np.random.uniform(low=-0.01,high=0.01,size=1)
        qpos[22] = -0.68 + np.random.uniform(low=-0.01,high=0.01,size=1)
        qpos[23] = 0.88 + np.random.uniform(low=-0.01,high=0.01,size=1)
        qpos[24] = (2389-2048)*(np.pi/180)*0.088 + np.random.uniform(low=-0.01,high=0.01,size=1)
        qpos[25] = (2057 - 2048)*(np.pi/180)*0.088 + np.random.uniform(low=-0.01,high=0.01,size=1)

        self.init_q = qpos

        # set the mass
        '''
        for body in self.robot_skeleton.bodynodes:
            mass = body.m
            mass1 = np.random.uniform(low=mass-mass/50,high=mass+mass/50,size=1)[0]
            if mass1 < 0:
                body.set_mass(mass)
            else:
                body.set_mass(mass1)
            Inertia = body.I
            #for i in range(3):
            #    Inertia[i,i] += np.random.uniform(low=0.9,high=1.1,size=1)[0]*Inertia[i,i]

            #body.set_inertia(Inertia)
        '''
        for i in range(self.robot_skeleton.ndofs):
            j = self.robot_skeleton.dof(i)
            j.set_damping_coefficient(1.256)
            #j.set_spring_stiffness(np.random.uniform(low=1.0,high=2.0,size=1)[0])

        self.count=0
        self.set_state(qpos, qvel)
        self.t = 0
        self.cur_step = 0
        self.vel_cache = []
        self.target_vel_cache = []

        self.avg_rew_weighting = []

        return self._get_obs()

    def viewer_setup(self):
        if not self.disableViewer:
            self._get_viewer().scene.tb.trans[0] = 0.0
            self._get_viewer().scene.tb.trans[2] = -1.5
            self._get_viewer().scene.tb.trans[1] = 0.0
            self._get_viewer().scene.tb.theta=80
            self._get_viewer().scene.tb.phi = 0

        return 0
