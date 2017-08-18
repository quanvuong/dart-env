# This environment is created by Alexander Clegg (alexanderwclegg@gmail.com)

import numpy as np
from gym import utils
from gym.envs.dart.dart_cloth_env import *
import random
import time
import math

from pyPhysX.colors import *
import pyPhysX.pyutils as pyutils

import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT

class DartClothReacherEnv3(DartClothEnv, utils.EzPickle):
    def __init__(self):
        self.target = np.array([0.8, -0.6, 0.6])
        self.arm = 1 #if 1, left arm (character's perspective), if 2, right

        #target spline
        self.targetSpline = pyutils.Spline()
        self.targetSplineTime = 0 #set to 0 in reset
        self.numTargetSplinePoints = 4
        self.targetSplineGlobalBounds = [np.array([0.75, 0.75, 0.35]),
                                         np.array([-0.75, -0.85, -0.85])]  # total drift allowed from origin for target orgs
        self.targetSplineLocalBounds = [np.array([0.35, 0.35, 0.35]),
                                        np.array([-0.35, -0.35, -0.35])]  # cartesian drift allowed b/t neighboring CPs
        self.targetSplineMinLocaldist = 0.2

        #debugging boxes for visualizing distributions
        self.drawDebuggingBoxes = True
        self.debuggingBoxes = [] #emptied in reset
        self.debuggingColors = [[0., 1, 0], [0, 0, 1.], [1., 0, 0], [1., 1., 0], [1., 0., 1.], [0, 1., 1.]]


        self.sensorNoise = 0.0
        self.stableReacher = False #if true, no termination reward for touching the target (must hover instead)

        #storage of rewards from previous step for rendering
        self.renderRewards = True
        self.cumulativeReward = 0
        self.pastReward = 0
        self.targetDistReward = 0
        self.splineReward = 0
        self.tauReward = 0
        self.velReward = 0
        self.accReward = 0
        self.prevDq = None

        self.restPoseActive = True
        self.restPoseWeight = np.ones(22)*0.05
        self.restPoseWeight[11:] *= 0. #eliminate uncontrolled pose penalty
        self.restPoseWeight[:2] *= 10 #stronger weight on torso begin upright
        self.restPoseWeight[2] *= 5 #strong weight on spine twist
        self.restPose = np.array([])
        self.restPoseReward = 0
        self.usePoseTarget = False #if true, rest pose is given in policy input

        self.interactiveTarget = False

        #5 dof reacher
        #self.action_scale = np.array([10, 10, 10, 10, 10])
        #self.control_bounds = np.array([[1.0, 1.0, 1.0, 1.0, 1.0],[-1.0, -1.0, -1.0, -1.0, -1.0]])
        
        #5 dof reacher
        #self.action_scale = np.array([ 10, 10, 10, 10 ,10])
        #self.control_bounds = np.array([[ 1.0, 1.0, 1.0, 1.0, 1.0],[ -1.0, -1.0, -1.0, -1.0, -1.0]])
        
        #9 dof reacher
        #self.action_scale = np.array([ 10, 10, 10, 10 ,10, 10, 10, 10, 10])
        #self.control_bounds = np.array([[ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],[ -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]])
        
        #22 dof upper body
        self.action_scale = np.ones(11)*10

        self.action_scale[0] = 250 #torso
        self.action_scale[1] = 250
        self.action_scale[2] = 250 #spine
        self.action_scale[3] = 150 #clav
        self.action_scale[4] = 150
        self.action_scale[5] = 60 #shoulder
        self.action_scale[6] = 60
        self.action_scale[7] = 50
        self.action_scale[8] = 30 #elbow
        self.action_scale[9] = 10  #wrist
        self.action_scale[10] = 10
        '''self.action_scale[11] = 150 #clav
        self.action_scale[12] = 150
        self.action_scale[13] = 60 #shoulder
        self.action_scale[14] = 60
        self.action_scale[15] = 50
        self.action_scale[16] = 50 #elbow
        self.action_scale[17] = 10 #wrist
        self.action_scale[18] = 10
        self.action_scale[19] = 10 #neck/head
        self.action_scale[20] = 10
        self.action_scale[21] = 10'''

        self.control_bounds = np.array([np.ones(11), np.ones(11)*-1])
        
        #autoT(au) is applied force at every step
        self.autoT = np.zeros(11)
        self.useAutoTau = False
        
        self.reset_number = 0 #debugging
        self.numSteps = 0
        
        self.doROM = False
        self.ROM_period = 200.0
        
        self.targetHistory = []
        self.successHistory = []
        
        #create cloth scene
        clothScene = pyphysx.ClothScene(step=0.01, sheet=True, sheetW=60, sheetH=15, sheetSpacing=0.025)
        
        #intialize the parent env
        observation_size = 66+66+6 #pose(sin,cos), pose vel, haptics
        if self.usePoseTarget is True:
            observation_size += 22

        DartClothEnv.__init__(self, cloth_scene=clothScene, model_paths='UpperBodyCapsules.skel', frame_skip=4,
                              observation_size=observation_size, action_bounds=self.control_bounds, disableViewer=True, visualize=False)

        #DartClothEnv.__init__(self, cloth_scene=clothScene, model_paths='UpperBodyCapsules.skel', frame_skip=4, observation_size=(66+66+6), action_bounds=self.control_bounds, visualize=False)

        self.restPose = np.array(self.robot_skeleton.q) #by default the rest pose is start pose. Change this in reset if desired.

        #TODO: additional observation size for force
        utils.EzPickle.__init__(self)
        
        self.clothScene.seedRandom(random.randint(1,1000))
        self.clothScene.setFriction(0, 1)
        
        #self.updateClothCollisionStructures(capsules=True, hapticSensors=True)
        
        self.simulateCloth = False
        self.sampleFromHemisphere = False
        self.rotateCloth = False
        self.randomRoll = False
        
        self.trackSuccess = False
        self.renderSuccess =True
        self.renderFailure = False
        self.successSampleRenderSize = 0.01
        
        self.renderDofs = True #if true, show dofs text 
        self.renderForceText = False
        
        #self.voxels = np.zeros(1000000) #100^3
        
        self.random_dir = np.array([0,0,1.])
        
        for i in range(len(self.robot_skeleton.bodynodes)):
            print(self.robot_skeleton.bodynodes[i])
            
        for i in range(len(self.robot_skeleton.dofs)):
            print(self.robot_skeleton.dofs[i])

    def limits(self, dof_ix):
        return np.array([self.robot_skeleton.dof(dof_ix).position_lower_limit(), self.robot_skeleton.dof(dof_ix).position_upper_limit()])

    def getRandomPose(self, excluded=None):
        #get a random skeleton pose by sampling between joint limits
        qpos = np.array(self.robot_skeleton.q)
        for i in range(len(self.robot_skeleton.dofs)):
            if excluded is not None: #check the optional excluded list and skip dofs which are listed
                isExcluded = False
                for e in excluded:
                    if e == i:
                        isExcluded = True
                if isExcluded:
                    continue
            lim = self.limits(i)
            qpos[i] = lim[0] + (lim[1]-lim[0])*random.random()
        return qpos
        
    def _step(self, a):
        #print("step")
        clamped_control = np.array(a)
        if self.useAutoTau is True:
            clamped_control = clamped_control + self.autoT
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.multiply(clamped_control, self.action_scale)

        fingertip = np.array([0.0, -0.06, 0.0])
        fingerNode = 8
        if self.arm == 2:
            fingerNode = 14
        wFingertip1 = self.robot_skeleton.bodynodes[fingerNode].to_world(fingertip)
        vec1 = self.target-wFingertip1

        while np.linalg.norm(vec1) < 0.1 and self.targetSplineTime < 1.0:
            self.targetSplineTime += 0.01
            self.target = self.targetSpline.pos(t=self.targetSplineTime)
            vec1 = self.target - wFingertip1
        self.dart_world.skeletons[0].q = [0, 0, 0, self.target[0], self.target[1], self.target[2]]

        #apply action and simulate
        if self.arm == 1:
            tau = np.concatenate([tau, np.zeros(11)])
        else:
            tau = np.concatenate([tau[:3], np.zeros(8), tau[3:], np.zeros(3)])
        self.do_simulation(tau, self.frame_skip)
        
        wFingertip2 = self.robot_skeleton.bodynodes[fingerNode].to_world(fingertip)
        vec2 = self.target-wFingertip2

        # distance to target penalty
        reward_dist = - np.linalg.norm(vec2)

        #spline progress reward
        splineTimeReward = self.targetSplineTime * 2.0
        if self.targetSplineTime >= 1.0 and -reward_dist <= 0.1:
            splineTimeReward = 20.0
        
        #force magnitude penalty    
        reward_ctrl = - np.linalg.norm(tau) * 0.0001

        qMask = np.ones(22)
        qMask[11:] *= 0.  # eliminate penalty to moving uncontrolled joints

        velocityPenalty = -np.linalg.norm(self.robot_skeleton.dq*qMask) *0.001

        accelerationPenalty = 0
        if self.prevDq is not None:
            accelerationPenalty = -np.linalg.norm((self.prevDq - self.robot_skeleton.dq)*qMask)*0.025
        
        #displacement toward target reward
        reward_progress = np.dot((wFingertip2 - wFingertip1), vec1/np.linalg.norm(vec1)) * 100

        #horizon length penalty
        alive_bonus = -0.001
        
        #proximity to target bonus
        reward_prox = 0
        if -reward_dist < 0.1:
            reward_prox = (0.1+reward_dist)*40

        #rest pose reward
        self.restPoseReward = 0
        if self.restPoseActive is True and len(self.restPose) == len(self.robot_skeleton.q):
            #print(np.linalg.norm(self.restPose - self.robot_skeleton.q))
            self.restPoseReward -= np.linalg.norm((self.restPose - self.robot_skeleton.q)*self.restPoseWeight)

        #total reward        
        reward = splineTimeReward + reward_dist + self.restPoseReward + velocityPenalty + accelerationPenalty + reward_ctrl

        #record rewards for debugging
        self.cumulativeReward += reward
        self.pastReward = reward
        self.targetDistReward = reward_dist
        self.splineReward = splineTimeReward
        self.tauReward = reward_ctrl
        self.velReward = velocityPenalty
        self.accReward = accelerationPenalty
        
        ob = self._get_obs()

        s = self.state_vector()

        #check termination conditions
        done = False
        if not np.isfinite(s).all():
            print("Infinite value detected..." + str(s))
            done = True
            reward -= 500
        elif self.stableReacher is True:
            a=0
            #this just prevents the target distance from termination
        #TODO: elif target spline time is max and target distance < 0.1

        #increment the step counter
        self.numSteps += 1

        return ob, reward, done, {}

    def _get_obs(self):
        f_size = 66
        '22x3 dofs, 22x3 sensors, 7x2 targets(toggle bit, cartesian, relative)'
        theta = self.robot_skeleton.q
        fingertip = np.array([0.0, -0.06, 0.0])

        vec = self.robot_skeleton.bodynodes[8].to_world(fingertip) - self.target
        if self.arm == 2:
            vec = self.robot_skeleton.bodynodes[14].to_world(fingertip) - self.target

        if self.simulateCloth is True:
            f = self.clothScene.getHapticSensorObs()#get force from simulation 
        elif self.sensorNoise is not 0:
            f = np.random.uniform(-self.sensorNoise, self.sensorNoise, f_size)
        else:
            f = np.zeros(f_size)

        obs = np.concatenate([np.cos(theta), np.sin(theta), self.robot_skeleton.dq])
        obs = np.concatenate([obs, vec, self.target])
        if self.usePoseTarget is True:
            obs = np.concatenate([obs, self.restPose])
        obs = np.concatenate([obs, f])

        return obs.ravel()

    def reset_model(self):
        self.targetSplineTime = 0.
        self.cumulativeReward = 0.
        self.debuggingBoxes = [pyutils.BoxFrame(c0=self.targetSplineGlobalBounds[0],
                                                c1=self.targetSplineGlobalBounds[1],
                                                org=np.zeros(3))]
        self.dart_world.reset()
        self.clothScene.reset()
        # move cloth out of arm range
        self.clothScene.translateCloth(0, np.array([-100.5, 0, 0]))
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)

        qpos[0] -= 0
        qpos[1] -= 0.
        qpos[2] += 0
        qpos[3] += 0.
        qpos[4] -= 0.
        # qpos[5] += 1
        qpos[5] -= 0.5

        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        self.prevDq = np.array(qvel)

        #set rest pose
        if self.restPoseActive is True:
            self.restPose = qpos
            self.restPose = np.array([0.001827724094, -0.00492083024962, 0.00269002874384, 0.00424451760036,
                                      0.00892373789407, -0.275006149068, 0.421834512474, 0.703907952432,
                                      1.83666666667, -0.00852162044016, 0.00391015137603, 0.00148231514513,
                                      -0.00964949311134, 0.00629198543994, 0.0058226973426, 0.00388720644317,
                                      0.0, 0.0073720446693, -0.000369173199674, 0.00479662592095, -0.00559061958513,
                                      0.000216556543277])
        
        #sampling in sphere
        reacher_range = 0.9
        while True:
            self.target = self.np_random.uniform(low=-reacher_range, high=reacher_range, size=3)
            #print('target = ' + str(self.target))
            if np.linalg.norm(self.target) < reacher_range: break

        self.dart_world.skeletons[0].q=[0, 0, 0, self.target[0], self.target[1], self.target[2]]

        #initial target is the 1st control point
        self.targetSpline.points = []
        dt = 1.0/(self.numTargetSplinePoints-1)
        for i in range(self.numTargetSplinePoints):
            if i == 0:
                self.targetSpline.insert(t=0., p=self.target)
            else:
                t = i*dt
                pos = self.targetSpline.pos(t=t)
                localDriftRange = np.array(self.targetSplineLocalBounds)
                localDriftRange[0] = np.minimum(localDriftRange[0],
                                                self.targetSplineGlobalBounds[0] - pos)
                localDriftRange[1] = np.maximum(localDriftRange[1],
                                                self.targetSplineGlobalBounds[1] - pos)
                self.debuggingBoxes.append(pyutils.BoxFrame(c0=localDriftRange[0],
                                                            c1=localDriftRange[1],
                                                            org=pos))
                tries = 0
                while True:
                    delta = np.array([random.uniform(localDriftRange[1][0], localDriftRange[0][0]),
                                      random.uniform(localDriftRange[1][1], localDriftRange[0][1]),
                                      random.uniform(localDriftRange[1][2], localDriftRange[0][2])])
                    tries += 1
                    if np.linalg.norm(delta) >= self.targetSplineMinLocaldist or tries > 100:
                        newpos = pos + delta
                        self.targetSpline.insert(t=t, p=newpos)
                        break

        self.reset_number += 1
        self.numSteps = 0
        
        obs = self._get_obs()

        return self._get_obs()

    def updateClothCollisionStructures(self, capsules=False, hapticSensors=False):
        a=0
        #collision spheres creation
        fingertip = np.array([0.0, -0.06, 0.0])
        z = np.array([0.,0,0])
        cs0 = self.robot_skeleton.bodynodes[1].to_world(z)
        cs1 = self.robot_skeleton.bodynodes[2].to_world(z)
        cs2 = self.robot_skeleton.bodynodes[16].to_world(z)
        cs3 = self.robot_skeleton.bodynodes[16].to_world(np.array([0,0.175,0]))
        cs4 = self.robot_skeleton.bodynodes[4].to_world(z)
        cs5 = self.robot_skeleton.bodynodes[6].to_world(z)
        cs6 = self.robot_skeleton.bodynodes[7].to_world(z)
        cs7 = self.robot_skeleton.bodynodes[8].to_world(z)
        cs8 = self.robot_skeleton.bodynodes[8].to_world(fingertip)
        cs9 = self.robot_skeleton.bodynodes[10].to_world(z)
        cs10 = self.robot_skeleton.bodynodes[12].to_world(z)
        cs11 = self.robot_skeleton.bodynodes[13].to_world(z)
        cs12 = self.robot_skeleton.bodynodes[14].to_world(z)
        cs13 = self.robot_skeleton.bodynodes[14].to_world(fingertip)
        csVars0 = np.array([0.15, -1, -1, 0,0,0])
        csVars1 = np.array([0.07, -1, -1, 0,0,0])
        csVars2 = np.array([0.1, -1, -1, 0,0,0])
        csVars3 = np.array([0.1, -1, -1, 0,0,0])
        csVars4 = np.array([0.065, -1, -1, 0,0,0])
        csVars5 = np.array([0.05, -1, -1, 0,0,0])
        csVars6 = np.array([0.0365, -1, -1, 0,0,0])
        csVars7 = np.array([0.04, -1, -1, 0,0,0])
        csVars8 = np.array([0.036, -1, -1, 0,0,0])
        csVars9 = np.array([0.065, -1, -1, 0,0,0])
        csVars10 = np.array([0.05, -1, -1, 0,0,0])
        csVars11 = np.array([0.0365, -1, -1, 0,0,0])
        csVars12 = np.array([0.04, -1, -1, 0,0,0])
        csVars13 = np.array([0.036, -1, -1, 0,0,0])
        collisionSpheresInfo = np.concatenate([cs0, csVars0, cs1, csVars1, cs2, csVars2, cs3, csVars3, cs4, csVars4, cs5, csVars5, cs6, csVars6, cs7, csVars7, cs8, csVars8, cs9, csVars9, cs10, csVars10, cs11, csVars11, cs12, csVars12, cs13, csVars13]).ravel()
        #collisionSpheresInfo = np.concatenate([cs0, csVars0, cs1, csVars1]).ravel()
        
        self.clothScene.setCollisionSpheresInfo(collisionSpheresInfo)
        
        if capsules is True:
            #collision capsules creation
            collisionCapsuleInfo = np.zeros((14,14))
            collisionCapsuleInfo[0,1] = 1
            collisionCapsuleInfo[1,2] = 1
            collisionCapsuleInfo[1,4] = 1
            collisionCapsuleInfo[1,9] = 1
            collisionCapsuleInfo[2,3] = 1
            collisionCapsuleInfo[4,5] = 1
            collisionCapsuleInfo[5,6] = 1
            collisionCapsuleInfo[6,7] = 1
            collisionCapsuleInfo[7,8] = 1
            collisionCapsuleInfo[9,10] = 1
            collisionCapsuleInfo[10,11] = 1
            collisionCapsuleInfo[11,12] = 1
            collisionCapsuleInfo[12,13] = 1
            self.clothScene.setCollisionCapsuleInfo(collisionCapsuleInfo)
            
        if hapticSensors is True:
            #hapticSensorLocations = np.concatenate([cs0, LERP(cs0, cs1, 0.33), LERP(cs0, cs1, 0.66), cs1, LERP(cs1, cs2, 0.33), LERP(cs1, cs2, 0.66), cs2, LERP(cs2, cs3, 0.33), LERP(cs2, cs3, 0.66), cs3])
            #hapticSensorLocations = np.concatenate([cs0, LERP(cs0, cs1, 0.25), LERP(cs0, cs1, 0.5), LERP(cs0, cs1, 0.75), cs1, LERP(cs1, cs2, 0.25), LERP(cs1, cs2, 0.5), LERP(cs1, cs2, 0.75), cs2, LERP(cs2, cs3, 0.25), LERP(cs2, cs3, 0.5), LERP(cs2, cs3, 0.75), cs3])
            hapticSensorLocations = np.concatenate([cs0, cs1, cs2, cs3, cs4, LERP(cs4, cs5, 0.33), LERP(cs4, cs5, 0.66), cs5, LERP(cs5, cs6, 0.33), LERP(cs5,cs6,0.66), cs6, cs7, cs8, cs9, LERP(cs9, cs10, 0.33), LERP(cs9, cs10, 0.66), cs10, LERP(cs10, cs11, 0.33), LERP(cs10, cs11, 0.66), cs11, cs12, cs13])
            self.clothScene.setHapticSensorLocations(hapticSensorLocations)
            
    def getViewer(self, sim, title=None, extraRenderFunc=None, inputFunc=None):
        return DartClothEnv.getViewer(self, sim, title, self.extraRenderFunction, self.inputFunc)
        
    def extraRenderFunction(self):
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3d(0,0,0)
        GL.glVertex3d(-1,0,0)
        GL.glEnd()

        # render debugging boxes
        if self.drawDebuggingBoxes:
            for ix, b in enumerate(self.debuggingBoxes):
                c = self.debuggingColors[ix]
                GL.glColor3d(c[0], c[1], c[2])
                b.draw()

        self.targetSpline.draw()
        
        if self.renderSuccess is True or self.renderFailure is True:
            for i in range(len(self.targetHistory)):
                p = self.targetHistory[i]
                s = self.successHistory[i]
                if (s and self.renderSuccess) or (not s and self.renderFailure):
                    #print("draw")
                    GL.glColor3d(1,0.,0)
                    if s is True:
                        GL.glColor3d(0,1.,0)
                    GL.glPushMatrix()
                    GL.glTranslated(p[0], p[1], p[2])
                    GLUT.glutSolidSphere(self.successSampleRenderSize, 10,10)
                    GL.glPopMatrix()

        #print("ID:" + str(self.clothScene.id))
        m_viewport = GL.glGetIntegerv(GL.GL_VIEWPORT)
        
        if self.renderRewards:
            self.clothScene.drawText(x=15, y=m_viewport[3] - 15 * 1, text="Cumulative Reward = " + str(self.cumulativeReward), color=(0.,0,0))
            self.clothScene.drawText(x=15, y=m_viewport[3] - 15 * 2, text="Past Reward = " + str(self.pastReward), color=(0.,0,0))
            self.clothScene.drawText(x=15, y=m_viewport[3] - 15 * 3, text="Spline Reward = " + str(self.splineReward), color=(0., 0, 0))
            self.clothScene.drawText(x=15, y=m_viewport[3] - 15 * 4, text="Target Dist Reward = " + str(self.targetDistReward), color=(0., 0, 0))
            self.clothScene.drawText(x=15, y=m_viewport[3] - 15 * 5, text="Rest-pose Reward = " + str(self.restPoseReward), color=(0., 0, 0))
            self.clothScene.drawText(x=15, y=m_viewport[3] - 15 * 6, text="Vel Reward = " + str(self.velReward), color=(0., 0, 0))
            self.clothScene.drawText(x=15, y=m_viewport[3] - 15 * 7, text="Accel Reward = " + str(self.accReward), color=(0., 0, 0))
            self.clothScene.drawText(x=15, y=m_viewport[3] - 15 * 8, text="Tau Reward = " + str(self.tauReward), color=(0., 0, 0))

            #self.clothScene.drawText(x=15, y=m_viewport[3]-15*4, text="Dist Reward 2 = " + str(self.distreward2), color=(0.,0,0))
            #self.clothScene.drawText(x=15, y=m_viewport[3]-15*5, text="Tau Reward = " + str(self.taureward), color=(0.,0,0))
            #self.clothScene.drawText(x=15, y=m_viewport[3]-15*6, text="Disp Reward 1 = " + str(self.dispreward1), color=(0.,0,0))
            #self.clothScene.drawText(x=15, y=m_viewport[3]-15*7, text="Vel Penalty = " + str(self.velocityPenalty), color=(0.,0,0))
            #self.clothScene.drawText(x=15, y=m_viewport[3]-15*7, text="Disp Reward 2 = " + str(self.dispreward2), color=(0.,0,0))
            #self.clothScene.drawText(x=15, y=m_viewport[3]-15*8, text="Prox Reward 1 = " + str(self.proxreward1), color=(0.,0,0))
            #self.clothScene.drawText(x=15, y=m_viewport[3]-15*9, text="Prox Reward 2 = " + str(self.proxreward2), color=(0.,0,0))
            #self.clothScene.drawText(x=15, y=m_viewport[3]-15*10, text="Rest-pose Reward = " + str(self.restPoseReward), color=(0., 0, 0))

        
        topLeft = np.array([2.,self.viewer.viewport[3]-10-130])
        self.viewer.interactors[4].topLeft = np.array(topLeft)
        self.viewer.interactors[4].boxesDefined = False
        renderUtils.renderDofs(robot=self.robot_skeleton, restPose=None, renderRestPose=False, _topLeft=topLeft)

    def inputFunc(self, repeat=False):
        pyutils.inputGenie(self, repeat)

    def viewer_setup(self):
        if self._get_viewer().scene is not None:
            self._get_viewer().scene.tb.trans[2] = -3.5
            self._get_viewer().scene.tb._set_theta(180)
            self._get_viewer().scene.tb._set_phi(180)
        self.track_skeleton_id = 0
        
        
def LERP(p0, p1, t):
    return p0 + (p1-p0)*t
    
