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
        
        
        obs_dim = 40
        
        self.control_bounds = np.array([np.ones(20,), np.ones(20,)])
        # LOWER CONTROL BOUNDS
        #LEFT HANDS
        self.control_bounds[0,0] = (2518 - 2048)*0.088*np.pi/180# - np.pi/8
        self.control_bounds[0,1] = (2248 - 2048)*0.088*np.pi/180# - np.pi/2
        self.control_bounds[0,2] = (1712 - 2048)*0.088*np.pi/180# - np.pi/4
        #RIGHT HANDS
        self.control_bounds[0,3] = (1498 - 2048)*0.088*np.pi/180# - np.pi/8
        self.control_bounds[0,4] = (1845 - 2048)*0.088*np.pi/180# - np.pi/8
        self.control_bounds[0,5] = (2381 - 2048)*0.088*np.pi/180# - np.pi/4
        #head
        self.control_bounds[0,6] = -0.01
        self.control_bounds[0,7] = -0.01
        #LEFT LEG
        self.control_bounds[0,8] = 0#-np.pi/25
        self.control_bounds[0,9] = -0.159600#(2052-2048)*(np.pi/180)*0.088 - np.pi/25
        self.control_bounds[0,10] = 0.529600#0.68 - 0.076
        self.control_bounds[0,11] = -1.417000#-0.93 - 0.076
        self.control_bounds[0,12] = -0.822900#(1707-2048)*(np.pi/180)*0.088 - 0.50
        self.control_bounds[0,13] = -0.310000#(2039-2048)*(np.pi/180)*0.088 - 0.50

        self.control_bounds[0,14] = 0#-np.pi/25
        self.control_bounds[0,15] = -0.089000#(2044-2048)*(np.pi/180)*0.088 - np.pi/100
        self.control_bounds[0,16] = -0.936200#-0.68 - 0.076
        self.control_bounds[0,17] = 0.565500#0.93 - 0.076
        self.control_bounds[0,18] = (2389-2048)*(np.pi/180)*0.088 - 0.050#0.365700#
        self.control_bounds[0,19] = (2057-2048)*(np.pi/180)*0.088 - 0.050#-0.248600#

        ### UPPER BOUNDS
        self.control_bounds[1,0] = (2518 - 2048)*0.088*np.pi/180# + np.pi/8
        self.control_bounds[1,1] = (2248 - 2048)*0.088*np.pi/180# + np.pi/8
        self.control_bounds[1,2] = (1712 - 2048)*0.088*np.pi/180# + np.pi/4
        #RIGHT HANDS
        self.control_bounds[1,3] = (1498 - 2048)*0.088*np.pi/180# + np.pi/8
        self.control_bounds[1,4] = (1845 - 2048)*0.088*np.pi/180# + np.pi/2
        self.control_bounds[1,5] = (2381 - 2048)*0.088*np.pi/180# + np.pi/4
        #head
        self.control_bounds[1,6] = 0.01
        self.control_bounds[1,7] = 0.01
        #LEFT LEG
        self.control_bounds[1,8] = 0#np.pi/25
        self.control_bounds[1,9] = 0.11200#(2052-2048)*(np.pi/180)*0.088 + np.pi/100
        self.control_bounds[1,10] = 0.936200#0.68 + 0.076
        self.control_bounds[1,11] = -0.562400#-0.93 + 0.076
        self.control_bounds[1,12] = -0.339600#(1707-2048)*(np.pi/180)*0.088 + 0.050
        self.control_bounds[1,13] = 0.236300#(2039-2048)*(np.pi/180)*0.088 + 0.050

        self.control_bounds[1,14] = 0#np.pi/25
        self.control_bounds[1,15] = 0.138100#(2044-2048)*(np.pi/180)*0.088 + np.pi/25
        self.control_bounds[1,16] = -0.529600#-0.68 + 0.076
        self.control_bounds[1,17] = 1.415500#0.93 + 0.076
        self.control_bounds[1,18] = (2389-2048)*(np.pi/180)*0.088 +0.050 #0.833700#
        self.control_bounds[1,19] = (2057-2048)*(np.pi/180)*0.088 + 0.050 #0.280800#
        self.ndofs = 26
        '''
        self.control_bounds = np.array([np.ones(20,)*-0.1,np.ones(20,)*0.1])
        #LEFT HANDS
        self.control_bounds[0,0] = -0.1#(2518 - 2048)*0.088*np.pi/180# - np.pi/8
        self.control_bounds[0,1] = -0.1#(2248 - 2048)*0.088*np.pi/180# - np.pi/2
        self.control_bounds[0,2] = -0.1#(1712 - 2048)*0.088*np.pi/180# - np.pi/4
        #RIGHT HAND
        self.control_bounds[0,3] = -0.1#1498 - 2048)*0.088*np.pi/180# - np.pi/8
        self.control_bounds[0,4] = -0.1#1845 - 2048)*0.088*np.pi/180# - np.pi/8
        self.control_bounds[0,5] = -0.1#2381 - 2048)*0.088*np.pi/180# - np.pi/4
        #head
        self.control_bounds[0,6] = -0.1
        self.control_bounds[0,7] = -0.1


        #LEFT HANDS
        self.control_bounds[1,0] = 0.1#(2518 - 2048)*0.088*np.pi/180# - np.pi/8
        self.control_bounds[1,1] = 0.1#(2248 - 2048)*0.088*np.pi/180# - np.pi/2
        self.control_bounds[1,2] = 0.1#(1712 - 2048)*0.088*np.pi/180# - np.pi/4
        #RIGHT HANDS
        self.control_bounds[1,3] = 0.1#1498 - 2048)*0.088*np.pi/180# - np.pi/8
        self.control_bounds[1,4] = 0.1#1845 - 2048)*0.088*np.pi/180# - np.pi/8
        self.control_bounds[1,5] = 0.1#2381 - 2048)*0.088*np.pi/180# - np.pi/4
        #head
        self.control_bounds[1,6] = 0.1
        self.control_bounds[1,7] = 0.1

        self.control_bounds[0,8] = -0.5#-np.pi/25
        self.control_bounds[0,9] = -0.5#(2052-2048)*(np.pi/180)*0.088 - np.pi/25
        self.control_bounds[0,14] = -0.5#-np.pi/25
        self.control_bounds[0,15] = -0.5#(2044-2048)*(np.pi/180)*0.088 - np.pi/100

        self.control_bounds[1,8] = 0.5#-np.pi/25
        self.control_bounds[1,9] = 0.5#(2052-2048)*(np.pi/180)*0.088 - np.pi/25
        self.control_bounds[1,14] = 0.5#-np.pi/25
        self.control_bounds[1,15] = 0.5#(2044-2048)*(np.pi/180)*0.088 - np.pi/100
        '''
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
        
        self.ref_traj = np.zeros((1000,self.ndofs))
        
        '''
        test = 0
        if test == 1:
            path = "../Darwin/"
        else:

            path = "../../Darwin/"##"./"#
        
        with open(path+"States.txt","rb") as fp:
            self.ref_traj = np.loadtxt(fp)

        with open(path + "lfoot.txt","rb") as fp:
            self.lfoot = np.loadtxt(fp)

        with open(path + "rfoot.txt","rb") as fp:
            self.rfoot = np.loadtxt(fp)

        with open(path + "lhand.txt","rb") as fp:
            self.lhand = np.loadtxt(fp)

        with open(path + "rhand.txt","rb") as fp:
            self.rhand = np.loadtxt(fp)

        '''

        dart_env.DartEnv.__init__(self, ['darwinmodel/ground1.urdf','darwinmodel/robotis_op2.urdf'], 16, obs_dim, self.control_bounds, disableViewer=True)

        self.dart_world.set_gravity([0,0,-9.81])

        self.robot_skeleton.set_self_collision_check(False)

        utils.EzPickle.__init__(self)

    def advance(self,a):

        clamped_control = np.array(a)
        

        
        for i in range(len(clamped_control)):
            clamped_control[i] = ((self.control_bounds[0][i] - self.control_bounds[1][i])/(3))*(clamped_control[i] +1)  + self.control_bounds[1][i]
            if clamped_control[i] > self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
            if clamped_control[i] < self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]

        
        

          
        self.target[6:]= clamped_control# + #*self.action_scale# + self.ref_trajectory_right[self.count_right,6:]# + 

        actions = np.zeros(26,)
        
        actions[6:] = copy.deepcopy(self.target[6:])

        for i in range(15):
            self.tau[6:] = self.PID()
            
            q = self.robot_skeleton.q
            dq = self.robot_skeleton.dq

            if self.dumpTorques:
                with open("torques.txt","ab") as fp:
                    np.savetxt(fp,np.array([self.tau]),fmt='%1.5f')

            if self.dumpActions:
                with open("targets_from_net.txt",'ab') as fp:
                    np.savetxt(fp,np.array([[self.target[6],self.robot_skeleton.q[6]]]),fmt='%1.5f')
            

            self.robot_skeleton.set_forces(self.tau)
            #print("torques",self.tau[22])
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
        torqueLimits = 4.5
        
        for i in range(6,26):
            if torques[i] > torqueLimits:#
                torques[i] = torqueLimits 
            if torques[i] < -torqueLimits:
                torques[i] = -torqueLimits

        return torques



    def step(self, a):
        self.advance(a)
        
        Joint_weights = np.ones(20,)
        Joint_weights[[0,3,10,11,12,13,16,17,18,19]] = 1
        Weight_matrix = np.diag(Joint_weights)
        joint_diff = self.ref_traj[self.count,6:] - self.robot_skeleton.q[6:]#+ (self.init[6:])
        joint_pen = np.sum(joint_diff.T*Weight_matrix*joint_diff)
        joint_term = 10*np.asarray(np.exp(-joint_pen))
        root_vel_term = 10/(1 + 0.01*abs(np.sum(self.robot_skeleton.dq[:6])))
        root_pos_term = np.exp(-5*np.sum((self.ref_traj[self.count,[3,4,5]] - self.robot_skeleton.q[[3,4,5]])**2)**0.5)

        reward =  -1e-3*np.sum(a)**2 + joint_term + root_pos_term # + root_pos_term + root_vel_term



        s = self.state_vector()
        com_height = self.robot_skeleton.bodynodes[0].com()[2]
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 200).all() and (com_height > -0.60) and (self.robot_skeleton.q[0] > -0.6) and (self.robot_skeleton.q[0]<0.6) and (abs(self.robot_skeleton.q[1]) < 0.50) and (abs(self.robot_skeleton.q[2]) < 0.50))
        
        if done:
            reward =0

        

        self.count+=1
        if self.count >= self.ref_traj.shape[0]-1:
            done = True
    
        ob = self._get_obs()

        self.t+=1
        #done = False

        

        return ob, reward, done, {}  #{'pre_state':pre_state, 'vel_rew':vel_rew, 'action_pen':action_pen, 'joint_pen':joint_pen, 'deviation_pen':deviation_pen, 'aux_pred':np.hstack([com_foot_offset1, com_foot_offset2, [reward]]), 'done_return':done}

    def _get_obs(self):
        
        state =  np.concatenate([self.robot_skeleton.q[6:],self.robot_skeleton.dq[6:]])

        state[:20] +=  np.random.uniform(low=-0.01,high=0.01,size=20)
        state[20:] +=  np.random.uniform(low=-0.05,high=0.05,size=1)

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
        return self._get_obs()

    def viewer_setup(self):
        if not self.disableViewer:
            self._get_viewer().scene.tb.trans[0] = 0.0
            self._get_viewer().scene.tb.trans[2] = -1.5
            self._get_viewer().scene.tb.trans[1] = 0.0
            self._get_viewer().scene.tb.theta=80
            self._get_viewer().scene.tb.phi = 0

        return 0
