# This environment is created by Alexander Clegg (alexanderwclegg@gmail.com)
# Phase interpolate2: end of 1st sleeve to match grip

import numpy as np
from gym import utils
from gym.envs.dart.dart_cloth_env import *
from gym.envs.dart.upperbodydatadriven_cloth_base import *
import random
import time
import math

from pyPhysX.colors import *
import pyPhysX.pyutils as pyutils
from pyPhysX.pyutils import LERP
import pyPhysX.renderUtils
import pyPhysX.meshgraph as meshgraph
from pyPhysX.clothfeature import *

import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT

class DartClothUpperBodyDataDrivenClothPhaseInterpolate2Env(DartClothUpperBodyDataDrivenClothBaseEnv, utils.EzPickle):
    def __init__(self):
        #feature flags
        rendering = False
        clothSimulation = True
        renderCloth = True

        #observation terms
        self.contactIDInObs = True  # if true, contact ids are in obs
        self.hapticsInObs   = True  # if true, haptics are in observation
        self.prevTauObs     = False  # if true, previous action in observation

        #reward flags
        self.uprightReward              = True #if true, rewarded for 0 torso angle from vertical
        self.elbowFlairReward           = False
        self.deformationPenalty         = True
        self.restPoseReward             = True
        self.rightTargetReward          = True
        self.leftTargetReward           = True

        #other flags
        self.collarTermination = True  # if true, rollout terminates when collar is off the head/neck
        self.collarTerminationCD = 0 #number of frames to ignore collar at the start of simulation (gives time for the cloth to drop)
        self.hapticsAware       = True  # if false, 0's for haptic input
        self.loadTargetsFromROMPositions = False
        self.resetPoseFromROMPoints = False
        self.resetTime = 0

        #other variables
        self.handleNode = None
        self.updateHandleNodeFrom = 12  # left fingers
        self.prevTau = None
        self.maxDeformation = 30.0
        self.restPose = None
        self.localRightEfShoulder1 = None
        self.localLeftEfShoulder1 = None
        self.rightTarget = np.zeros(3)
        self.leftTarget = np.zeros(3)
        self.prevErrors = None #stores the errors taken from DART each iteration

        self.actuatedDofs = np.arange(22)
        observation_size = len(self.actuatedDofs)*3 #q(sin,cos), dq
        if self.prevTauObs:
            observation_size += len(self.actuatedDofs)
        if self.hapticsInObs:
            observation_size += 66
        if self.contactIDInObs:
            observation_size += 22
        if self.rightTargetReward:
            observation_size += 9
        if self.leftTargetReward:
            observation_size += 9

        DartClothUpperBodyDataDrivenClothBaseEnv.__init__(self,
                                                          rendering=rendering,
                                                          screensize=(1080,720),
                                                          clothMeshFile="tshirt_m.obj",
                                                          clothMeshStateFile = "objFile_1starmin.obj",
                                                          clothScale=1.4,
                                                          obs_size=observation_size,
                                                          simulateCloth=clothSimulation)

        # clothing features
        self.collarVertices = [117, 115, 113, 900, 108, 197, 194, 8, 188, 5, 120]
        self.targetGripVertices = [570, 1041, 285, 1056, 435, 992, 50, 489, 787, 327, 362, 676, 887, 54, 55]
        self.collarFeature = ClothFeature(verts=self.collarVertices, clothScene=self.clothScene)

        self.simulateCloth = clothSimulation
        if self.simulateCloth:
            self.handleNode = HandleNode(self.clothScene, org=np.array([0.05, 0.034, -0.975]))

        if not renderCloth:
            self.clothScene.renderClothFill = False
            self.clothScene.renderClothBoundary = False
            self.clothScene.renderClothWires = False

    def _getFile(self):
        return __file__

    def saveObjState(self):
        print("Trying to save the object state")
        self.clothScene.saveObjState("objState", 0)

    def updateBeforeSimulation(self):
        #any pre-sim updates should happen here
        fingertip = np.array([0.0, -0.065, 0.0])
        wRFingertip1 = self.robot_skeleton.bodynodes[7].to_world(fingertip)
        wLFingertip1 = self.robot_skeleton.bodynodes[12].to_world(fingertip)
        self.localRightEfShoulder1 = self.robot_skeleton.bodynodes[3].to_local(wRFingertip1)  # right fingertip in right shoulder local frame
        self.localLeftEfShoulder1 = self.robot_skeleton.bodynodes[8].to_local(wLFingertip1)  # left fingertip in left shoulder local frame
        #self.leftTarget = pyutils.getVertCentroid(verts=self.targetGripVertices, clothscene=self.clothScene) + pyutils.getVertAvgNorm(verts=self.targetGripVertices, clothscene=self.clothScene)*0.03

        if self.collarFeature is not None:
            self.collarFeature.fitPlane()

        # update handle nodes
        if self.handleNode is not None:
            if self.updateHandleNodeFrom >= 0:
                self.handleNode.setTransform(self.robot_skeleton.bodynodes[self.updateHandleNodeFrom].T)
            self.handleNode.step()

        a=0

    def checkTermination(self, tau, s, obs):
        #check the termination conditions and return: done,reward
        topHead = self.robot_skeleton.bodynodes[14].to_world(np.array([0, 0.25, 0]))
        bottomHead = self.robot_skeleton.bodynodes[14].to_world(np.zeros(3))
        bottomNeck = self.robot_skeleton.bodynodes[13].to_world(np.zeros(3))

        if np.amax(np.absolute(s[:len(self.robot_skeleton.q)])) > 10:
            print("Detecting potential instability")
            print(s)
            return True, -1500
        elif not np.isfinite(s).all():
            print("Infinite value detected..." + str(s))
            return True, -1500
        elif self.collarTermination and self.simulateCloth and self.collarTerminationCD < self.numSteps:
            if not (self.collarFeature.contains(l0=bottomNeck, l1=bottomHead)[0] or
                        self.collarFeature.contains(l0=bottomHead, l1=topHead)[0]):
                #print("collar term")
                return True, -1500

        return False, 0

    def computeReward(self, tau):
        #compute and return reward at the current state
        fingertip = np.array([0.0, -0.065, 0.0])
        wRFingertip2 = self.robot_skeleton.bodynodes[7].to_world(fingertip)
        wLFingertip2 = self.robot_skeleton.bodynodes[12].to_world(fingertip)
        localRightEfShoulder2 = self.robot_skeleton.bodynodes[3].to_local(wRFingertip2)  # right fingertip in right shoulder local frame
        localLeftEfShoulder2 = self.robot_skeleton.bodynodes[8].to_local(wLFingertip2)  # right fingertip in right shoulder local frame

        #store the previous step's errors
        #if self.reset_number > 0 and self.numSteps > 0:
        #    self.prevErrors = self.dart_world.getAllConstraintViolations()
            #print("after getAllCon..")
            #print("prevErrors: " +str(self.prevErrors))

        self.prevTau = tau

        clothDeformation = 0
        if self.simulateCloth:
            clothDeformation = self.clothScene.getMaxDeformationRatio(0)
            self.deformation = clothDeformation

        reward_clothdeformation = 0
        if self.deformationPenalty is True:
            # reward_clothdeformation = (math.tanh(9.24 - 0.5 * clothDeformation) - 1) / 2.0  # near 0 at 15, ramps up to -1.0 at ~22 and remains constant
            reward_clothdeformation = -(math.tanh(
                0.14 * (clothDeformation - 25)) + 1) / 2.0  # near 0 at 15, ramps up to -1.0 at ~22 and remains constant

        # force magnitude penalty
        reward_ctrl = -np.square(tau).sum()

        # reward for maintaining posture
        reward_upright = 0
        if self.uprightReward:
            reward_upright = -abs(self.robot_skeleton.q[0]) - abs(self.robot_skeleton.q[1])

        reward_restPose = 0
        if self.restPoseReward and self.restPose is not None:
            '''z = 0.5  # half the max magnitude (e.g. 0.5 -> [0,1])
            s = 1.0  # steepness (higher is steeper)
            l = 4.2  # translation
            dist = np.linalg.norm(self.robot_skeleton.q - self.restPose)
            reward_restPose = -(z * math.tanh(s * (dist - l)) + z)'''
            dist = np.linalg.norm(self.robot_skeleton.q - self.restPose)
            reward_restPose = -dist
            # print("distance: " + str(dist) + " -> " + str(reward_restPose))

        reward_rightTarget = 0
        if self.rightTargetReward:
            rDist = np.linalg.norm(self.rightTarget-wRFingertip2)
            reward_rightTarget = -rDist - rDist**2
            '''if rDist < 0.02:
                reward_rightTarget += 0.25'''

        reward_leftTarget = 0
        if self.leftTargetReward:
            lDist = np.linalg.norm(self.leftTarget - wLFingertip2)
            reward_leftTarget = -lDist - lDist**2
            '''if lDist < 0.02:
                reward_leftTarget += 0.25'''

        #print("reward_restPose: " + str(reward_restPose))
        #print("reward_leftTarget: " + str(reward_leftTarget))
        self.reward = reward_ctrl * 0 \
                      + reward_upright \
                      + reward_clothdeformation * 5 \
                      + reward_restPose \
                      + reward_rightTarget \
                      + reward_leftTarget
        return self.reward

    def _get_obs(self):
        f_size = 66
        '22x3 dofs, 22x3 sensors, 7x2 targets(toggle bit, cartesian, relative)'
        theta = np.zeros(len(self.actuatedDofs))
        dtheta = np.zeros(len(self.actuatedDofs))
        for ix, dof in enumerate(self.actuatedDofs):
            theta[ix] = self.robot_skeleton.q[dof]
            dtheta[ix] = self.robot_skeleton.dq[dof]

        fingertip = np.array([0.0, -0.065, 0.0])

        obs = np.concatenate([np.cos(theta), np.sin(theta), dtheta]).ravel()

        if self.prevTauObs:
            obs = np.concatenate([obs, self.prevTau])

        if self.hapticsInObs:
            f = None
            if self.simulateCloth and self.hapticsAware:
                f = self.clothScene.getHapticSensorObs()#get force from simulation
            else:
                f = np.zeros(f_size)
            obs = np.concatenate([obs, f]).ravel()

        if self.contactIDInObs:
            HSIDs = self.clothScene.getHapticSensorContactIDs()
            obs = np.concatenate([obs, HSIDs]).ravel()

        if self.rightTargetReward:
            efR = self.robot_skeleton.bodynodes[7].to_world(fingertip)
            obs = np.concatenate([obs, self.rightTarget, efR, self.rightTarget-efR]).ravel()

        if self.leftTargetReward:
            efL = self.robot_skeleton.bodynodes[12].to_world(fingertip)
            obs = np.concatenate([obs, self.leftTarget, efL, self.leftTarget-efL]).ravel()

        return obs

    def additionalResets(self):
        '''if self.resetTime > 0:
            print("reset " + str(self.reset_number) + " after " + str(time.time()-self.resetTime))
        '''
        self.resetTime = time.time()
        #do any additional resetting here
        fingertip = np.array([0, -0.065, 0])
        '''if self.simulateCloth:
            up = np.array([0,1.0,0])
            varianceR = pyutils.rotateY(((random.random()-0.5)*2.0)*0.3)
            adjustR = pyutils.rotateY(0.2)
            R = self.clothScene.rotateTo(v1=np.array([0,0,1.0]), v2=up)
            self.clothScene.translateCloth(0, np.array([-0.01, 0.0255, 0]))
            self.clothScene.rotateCloth(cid=0, R=R)
            self.clothScene.rotateCloth(cid=0, R=adjustR)
            self.clothScene.rotateCloth(cid=0, R=varianceR)'''
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-0.1, high=0.1, size=self.robot_skeleton.ndofs)

        '''if self.resetPoseFromROMPoints and len(self.ROMPoints) > 0:
            poseFound = False
            while not poseFound:
                ix = random.randint(0,len(self.ROMPoints)-1)
                qpos = self.ROMPoints[ix]
                efR = self.ROMPositions[ix][:3]
                efL = self.ROMPositions[ix][-3:]
                if efR[2] < 0 and efL[2] < 0: #half-plane constraint on end effectors
                    poseFound = True
        '''
        #Check the constrained population
        '''positive = 0
        for targets in self.ROMPositions:
            efR = targets[:3]
            efL = targets[-3:]
            if efR[2] < 0 and efL[2] < 0:
                positive += 1
        print("Valid Poses: " + str(positive) + " | ratio: " + str(positive/len(self.ROMPositions)))'''


        '''if self.loadTargetsFromROMPositions and len(self.ROMPositions) > 0:
            targetFound = False
            while not targetFound:
                ix = random.randint(0, len(self.ROMPositions) - 1)
                self.rightTarget = self.ROMPositions[ix][:3] + self.np_random.uniform(low=-0.01, high=0.01, size=3)
                self.leftTarget = self.ROMPositions[ix][-3:] + self.np_random.uniform(low=-0.01, high=0.01, size=3)
                if self.rightTarget[2] < 0 and self.leftTarget[2] < 0: #half-plane constraint on end effectors
                    targetFound = True
        self.set_state(qpos, qvel)'''

        self.loadCharacterState(filename="characterState_regrip")

        #find end effector targets and set restPose from solution
        fingertip = np.array([0.0, -0.065, 0.0])
        self.rightTarget = self.robot_skeleton.bodynodes[7].to_world(fingertip)
        self.leftTarget = self.robot_skeleton.bodynodes[12].to_world(fingertip)
        self.restPose = np.array(self.robot_skeleton.q)

        self.loadCharacterState(filename="characterState_1starmin")

        if self.handleNode is not None:
            self.handleNode.clearHandles()
            self.handleNode.addVertices(verts=[570, 1041, 285, 1056, 435, 992, 50, 489, 787, 327, 362, 676, 887, 54, 55])
            self.handleNode.setOrgToCentroid()
            if self.updateHandleNodeFrom >= 0:
                self.handleNode.setTransform(self.robot_skeleton.bodynodes[self.updateHandleNodeFrom].T)
            self.handleNode.recomputeOffsets()


        if self.simulateCloth:
            self.collarFeature.fitPlane()

        a=0

    def extraRenderFunction(self):
        renderUtils.setColor(color=[0.0, 0.0, 0])
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3d(0,0,0)
        GL.glVertex3d(-1,0,0)
        GL.glEnd()

        topHead = self.robot_skeleton.bodynodes[14].to_world(np.array([0, 0.25, 0]))
        bottomHead = self.robot_skeleton.bodynodes[14].to_world(np.zeros(3))
        bottomNeck = self.robot_skeleton.bodynodes[13].to_world(np.zeros(3))

        renderUtils.drawLineStrip(points=[bottomNeck, bottomHead, topHead])
        if self.collarFeature is not None:
            self.collarFeature.drawProjectionPoly()

        renderUtils.setColor([0,0,0])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[4].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[4].to_world(np.array([0.0,-0.3,-0.075]))])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[9].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[9].to_world(np.array([0.0,-0.3,-0.075]))])

        #render targets
        fingertip = np.array([0,-0.065,0])
        if self.rightTargetReward:
            efR = self.robot_skeleton.bodynodes[7].to_world(fingertip)
            renderUtils.setColor(color=[1.0,0,0])
            renderUtils.drawSphere(pos=self.rightTarget,rad=0.02)
            renderUtils.drawLineStrip(points=[self.rightTarget, efR])
        if self.leftTargetReward:
            efL = self.robot_skeleton.bodynodes[12].to_world(fingertip)
            renderUtils.setColor(color=[0, 1.0, 0])
            renderUtils.drawSphere(pos=self.leftTarget,rad=0.02)
            renderUtils.drawLineStrip(points=[self.leftTarget, efL])

        textHeight = 15
        textLines = 2

        if self.renderUI:
            renderUtils.setColor(color=[0.,0,0])
            self.clothScene.drawText(x=15., y=textLines*textHeight, text="Steps = " + str(self.numSteps), color=(0., 0, 0))
            textLines += 1
            self.clothScene.drawText(x=15., y=textLines*textHeight, text="Reward = " + str(self.reward), color=(0., 0, 0))
            textLines += 1
            self.clothScene.drawText(x=15., y=textLines * textHeight, text="Cumulative Reward = " + str(self.cumulativeReward), color=(0., 0, 0))
            textLines += 1
            if self.numSteps > 0:
                renderUtils.renderDofs(robot=self.robot_skeleton, restPose=None, renderRestPose=False)