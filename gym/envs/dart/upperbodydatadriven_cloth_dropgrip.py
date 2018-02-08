# This environment is created by Alexander Clegg (alexanderwclegg@gmail.com)

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

class DartClothUpperBodyDataDrivenClothDropGripEnv(DartClothUpperBodyDataDrivenClothBaseEnv, utils.EzPickle):
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
        self.restPoseReward             = False
        self.rightTargetReward          = False
        self.leftTargetReward           = True
        self.rightOrientationTargetReward = False
        self.leftOrientationTargetReward = True

        #other flags
        self.collarTermination = True  # if true, rollout terminates when collar is off the head/neck
        self.collarTerminationCD = 10 #number of frames to ignore collar at the start of simulation (gives time for the cloth to drop)
        self.hapticsAware       = True  # if false, 0's for haptic input
        self.loadTargetsFromROMPositions = False
        self.resetPoseFromROMPoints = False
        self.resetTime = 0
        self.graphTaskSuccess = False
        self.taskSuccessGraph = None
        if self.graphTaskSuccess:
            self.taskSuccessGraph = pyutils.LineGrapher(title="Task Success", numPlots=2)

        #other variables
        self.prevTau = None
        self.maxDeformation = 30.0
        self.restPose = None
        self.localRightEfShoulder1 = None
        self.localLeftEfShoulder1 = None
        self.rightTarget = np.zeros(3)
        self.leftTarget = np.zeros(3)
        self.rightOrientationTarget = np.zeros(3)
        self.leftOrientationTarget = np.zeros(3)
        self.localLeftOrientationTarget = np.zeros(3) #local in gripLFeature frame
        self.prevErrors = None #stores the errors taken from DART each iteration
        self.fingertip = np.array([0, -0.085, 0])

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
        if self.rightOrientationTargetReward:
            observation_size += 6
        if self.leftOrientationTargetReward:
            observation_size += 6

        DartClothUpperBodyDataDrivenClothBaseEnv.__init__(self,
                                                          rendering=rendering,
                                                          screensize=(1080,720),
                                                          clothMeshFile="tshirt_m.obj",
                                                          #clothMeshStateFile = "tshirt_regrip5.obj",
                                                          clothScale=1.4,
                                                          obs_size=observation_size,
                                                          simulateCloth=clothSimulation)

        # clothing features
        self.collarVertices = [117, 115, 113, 900, 108, 197, 194, 8, 188, 5, 120]
        #self.targetGripVertices = [570, 1041, 285, 1056, 435, 992, 50, 489, 787, 327, 362, 676, 887, 54, 55]
        self.targetGripVerticesL = [46, 437, 955, 1185, 47, 285, 711, 677, 48, 905, 1041, 49, 741, 889, 45]
        self.targetGripVerticesR = [905, 1041, 49, 435, 50, 570, 992, 1056, 51, 676, 283, 52, 489, 892, 362, 53]
        self.collarFeature = ClothFeature(verts=self.collarVertices, clothScene=self.clothScene)
        self.gripFeatureL = ClothFeature(verts=self.targetGripVerticesL, clothScene=self.clothScene, b1spanverts=[889,1041], b2spanverts=[47,677])
        self.gripFeatureR = ClothFeature(verts=self.targetGripVerticesR, clothScene=self.clothScene, b1spanverts=[362,889], b2spanverts=[51,992])

        self.simulateCloth = clothSimulation
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
        wRFingertip1 = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)
        wLFingertip1 = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
        self.localRightEfShoulder1 = self.robot_skeleton.bodynodes[3].to_local(wRFingertip1)  # right fingertip in right shoulder local frame
        self.localLeftEfShoulder1 = self.robot_skeleton.bodynodes[8].to_local(wLFingertip1)  # left fingertip in left shoulder local frame

        if self.collarFeature is not None:
            self.collarFeature.fitPlane()
        self.gripFeatureL.fitPlane()
        self.gripFeatureR.fitPlane()

        #self.leftTarget = pyutils.getVertCentroid(verts=self.targetGripVerticesL, clothscene=self.clothScene) + pyutils.getVertAvgNorm(verts=self.targetGripVerticesL, clothscene=self.clothScene)*0.03
        self.leftTarget = self.gripFeatureL.plane.org + self.gripFeatureL.plane.normal*0.03
        self.leftOrientationTarget = self.gripFeatureL.plane.toWorld(self.localLeftOrientationTarget)

        a=0

    def checkTermination(self, tau, s, obs):
        #check the termination conditions and return: done,reward
        topHead = self.robot_skeleton.bodynodes[14].to_world(np.array([0, 0.25, 0]))
        bottomHead = self.robot_skeleton.bodynodes[14].to_world(np.zeros(3))
        bottomNeck = self.robot_skeleton.bodynodes[13].to_world(np.zeros(3))

        if np.amax(np.absolute(s[:len(self.robot_skeleton.q)])) > 10:
            print("Detecting potential instability")
            print(s)
            return True, -20000
        elif not np.isfinite(s).all():
            print("Infinite value detected..." + str(s))
            return True, -20000
        elif self.collarTermination and self.simulateCloth and self.collarTerminationCD < self.numSteps:
            if not (self.collarFeature.contains(l0=bottomNeck, l1=bottomHead)[0] or
                        self.collarFeature.contains(l0=bottomHead, l1=topHead)[0]):
                print("collar term")
                return True, -20000

        return False, 0

    def computeReward(self, tau):
        #compute and return reward at the current state
        wRFingertip2 = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)
        wLFingertip2 = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
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
            reward_upright = max(-2.5, -abs(self.robot_skeleton.q[0]) - abs(self.robot_skeleton.q[1]))

        reward_restPose = 0
        if self.restPoseReward and self.restPose is not None:
            '''z = 0.5  # half the max magnitude (e.g. 0.5 -> [0,1])
            s = 1.0  # steepness (higher is steeper)
            l = 4.2  # translation
            dist = np.linalg.norm(self.robot_skeleton.q - self.restPose)
            reward_restPose = -(z * math.tanh(s * (dist - l)) + z)'''
            dist = np.linalg.norm(self.robot_skeleton.q - self.restPose)
            reward_restPose = max(-51, -dist)
            # print("distance: " + str(dist) + " -> " + str(reward_restPose))

        reward_rightTarget = 0
        if self.rightTargetReward:
            rDist = np.linalg.norm(self.rightTarget-wRFingertip2)
            reward_rightTarget = -rDist

        reward_leftTarget = 0
        if self.leftTargetReward:
            lDist = np.linalg.norm(self.leftTarget - wLFingertip2)
            reward_leftTarget = -lDist

        reward_rightOrientationTarget = 0
        if self.rightOrientationTargetReward:
            wRZero = self.robot_skeleton.bodynodes[7].to_world(np.zeros(3))
            efRV = wRFingertip2-wRZero
            efRDir = efRV/np.linalg.norm(efRV)
            reward_rightOrientationTarget = -(1 - self.rightOrientationTarget.dot(efRDir)) / 2.0 #[-1,0]

        reward_leftOrientationTarget = 0
        if self.leftOrientationTargetReward:
            wLZero = self.robot_skeleton.bodynodes[12].to_world(np.zeros(3))
            efLV = wLFingertip2 - wLZero
            efLDir = efLV / np.linalg.norm(efLV)
            reward_leftOrientationTarget = -(1 - self.leftOrientationTarget.dot(efLDir)) / 2.0  # [-1,0]

        if self.graphTaskSuccess:
            self.taskSuccessGraph.addToLinePlot([-reward_leftTarget,-reward_leftOrientationTarget])

        self.reward = reward_ctrl * 0 \
                      + reward_upright*10 \
                      + reward_clothdeformation * 400 \
                      + reward_restPose*0.3 \
                      + reward_rightTarget \
                      + reward_leftTarget*100.0 \
                      + reward_leftOrientationTarget*10.0 \
                      + reward_rightOrientationTarget*10.0
        return self.reward

    def _get_obs(self):
        f_size = 66
        '22x3 dofs, 22x3 sensors, 7x2 targets(toggle bit, cartesian, relative)'
        theta = np.zeros(len(self.actuatedDofs))
        dtheta = np.zeros(len(self.actuatedDofs))
        for ix, dof in enumerate(self.actuatedDofs):
            theta[ix] = self.robot_skeleton.q[dof]
            dtheta[ix] = self.robot_skeleton.dq[dof]

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
            efR = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)
            obs = np.concatenate([obs, self.rightTarget, efR, self.rightTarget-efR]).ravel()

        if self.leftTargetReward:
            efL = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
            obs = np.concatenate([obs, self.leftTarget, efL, self.leftTarget-efL]).ravel()

        if self.rightOrientationTargetReward:
            efRV = self.robot_skeleton.bodynodes[7].to_world(self.fingertip) - self.robot_skeleton.bodynodes[7].to_world(np.zeros(3))
            efRDir = efRV / np.linalg.norm(efRV)
            obs = np.concatenate([obs, self.rightOrientationTarget, efRDir]).ravel()

        if self.leftOrientationTargetReward:
            efLV = self.robot_skeleton.bodynodes[12].to_world(self.fingertip) - self.robot_skeleton.bodynodes[12].to_world(np.zeros(3))
            efLDir = efLV / np.linalg.norm(efLV)
            obs = np.concatenate([obs, self.leftOrientationTarget, efLDir]).ravel()

        return obs

    def additionalResets(self):
        '''if self.resetTime > 0:
            print("reset " + str(self.reset_number) + " after " + str(time.time()-self.resetTime))
        '''
        self.resetTime = time.time()

        if self.graphTaskSuccess:
            self.taskSuccessGraph.xdata = []
            self.taskSuccessGraph.yData = [[],[]]
            #self.taskSuccessGraph.close()
            #self.taskSuccessGraph = pyutils.LineGrapher(title="Task Success", numPlots=2)

        #do any additional resetting here
        if self.simulateCloth:
            up = np.array([0,1.0,0])
            '''vDir = pyutils.sampleDirections(num=1)[0]
            tries = 0
            while vDir.dot(up) < 0.95:
                tries += 1
                vDir = pyutils.sampleDirections(num=1)[0]
            print("took " + str(tries) + " tries to sample")
            varianceR = self.clothScene.rotateTo(v1=up, v2=vDir)'''
            varianceR = pyutils.rotateY(((random.random()-0.5)*2.0)*0.3)
            adjustR = pyutils.rotateY(0.2)
            R = self.clothScene.rotateTo(v1=np.array([0,0,1.0]), v2=up)
            self.clothScene.translateCloth(0, np.array([-0.01, 0.0255, 0]))
            self.clothScene.rotateCloth(cid=0, R=R)
            self.clothScene.rotateCloth(cid=0, R=adjustR)
            self.clothScene.rotateCloth(cid=0, R=varianceR)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-0.1, high=0.1, size=self.robot_skeleton.ndofs)
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        #qpos = np.array(
        #    [-0.0483053659505, 0.0321213273351, 0.0173036909392, 0.00486290205677, -0.00284350018845, -0.634602301004,
        #     -0.359172622713, 0.0792754054027, 2.66867203095, 0.00489456931428, 0.000476966442889, 0.0234663491334,
        #     -0.0254520098678, 0.172782859361, -1.31351102137, 0.702315566312, 1.73993331669, -0.0422811572637,
        #     0.586669332152, -0.0122329947565, 0.00179736869435, -8.0625896949e-05])

        if self.resetPoseFromROMPoints and len(self.ROMPoints) > 0:
            poseFound = False
            while not poseFound:
                ix = random.randint(0,len(self.ROMPoints)-1)
                qpos = self.ROMPoints[ix]
                efR = self.ROMPositions[ix][:3]
                efL = self.ROMPositions[ix][-3:]
                if efR[2] < 0 and efL[2] < 0: #half-plane constraint on end effectors
                    poseFound = True

        #Check the constrained population
        '''positive = 0
        for targets in self.ROMPositions:
            efR = targets[:3]
            efL = targets[-3:]
            if efR[2] < 0 and efL[2] < 0:
                positive += 1
        print("Valid Poses: " + str(positive) + " | ratio: " + str(positive/len(self.ROMPositions)))'''


        if self.loadTargetsFromROMPositions and len(self.ROMPositions) > 0:
            targetFound = False
            while not targetFound:
                ix = random.randint(0, len(self.ROMPositions) - 1)
                self.rightTarget = self.ROMPositions[ix][:3] + self.np_random.uniform(low=-0.01, high=0.01, size=3)
                self.leftTarget = self.ROMPositions[ix][-3:] + self.np_random.uniform(low=-0.01, high=0.01, size=3)
                if self.rightTarget[2] < 0 and self.leftTarget[2] < 0: #half-plane constraint on end effectors
                    targetFound = True
        self.set_state(qpos, qvel)
        self.restPose = qpos

        if self.simulateCloth:
            self.collarFeature.fitPlane()
            self.gripFeatureL.fitPlane(normhint=np.array([0, 0, -1.0]))
            self.gripFeatureR.fitPlane(normhint=np.array([0, 0, -1.0]))
            self.localLeftOrientationTarget = np.array([0.0, -1.0, 0.0])
            self.localLeftOrientationTarget = np.array([1.0, -1.0, -1.0])
            self.localLeftOrientationTarget /= np.linalg.norm(self.localLeftOrientationTarget)

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
            self.collarFeature.drawProjectionPoly(renderNormal=False, renderBasis=False)

        self.gripFeatureL.drawProjectionPoly(fill=False, fillColor=[0.0, 0.75, 0.75], vecLenScale=0.25)
        self.gripFeatureR.drawProjectionPoly(fill=False, fillColor=[1.0, 0.0, 1.0], vecLenScale=0.5, renderNormal=False, renderBasis=False)

        renderUtils.setColor([0,0,0])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[4].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[4].to_world(np.array([0.0,-0.3,-0.075]))])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[9].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[9].to_world(np.array([0.0,-0.3,-0.075]))])

        #render targets
        if self.rightTargetReward:
            efR = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)
            renderUtils.setColor(color=[1.0,0,0])
            renderUtils.drawSphere(pos=self.rightTarget,rad=0.02)
            renderUtils.drawLineStrip(points=[self.rightTarget, efR])
        if self.leftTargetReward:
            efL = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
            renderUtils.setColor(color=[0, 1.0, 0])
            renderUtils.drawSphere(pos=self.leftTarget,rad=0.02)
            renderUtils.drawLineStrip(points=[self.leftTarget, efL])


        #get end effector frame orientation and position error
        orientationError = 0
        efLV = self.robot_skeleton.bodynodes[12].to_world(self.fingertip) - self.robot_skeleton.bodynodes[12].to_world(np.zeros(3))
        efLDir = efLV/np.linalg.norm(efLV)
        efLDirLocal = self.gripFeatureL.plane.toLocal(efLDir)
        efLDirWorld = self.gripFeatureL.plane.toWorld(efLDirLocal)

        '''print("efLDir" + str(efLDir))
        print("efLDirLocal" + str(efLDirLocal))
        print("efLDirWorld" + str(efLDirWorld))
        print("error = " + str(np.linalg.norm(efLDirWorld-efLDir)))
        print("--------------")'''

        worldVec = self.gripFeatureL.plane.toWorld(self.localLeftOrientationTarget)
        renderUtils.setColor([0, 0.7, 0.7])
        renderUtils.drawArrow(p0=self.gripFeatureL.plane.org, p1=self.gripFeatureL.plane.org+worldVec*0.5)
        renderUtils.setColor([0.7, 0.0, 0.4])
        renderUtils.drawArrow(p0=self.gripFeatureL.plane.org, p1=self.gripFeatureL.plane.org+efLDirWorld*0.5)

        orientationError = (1-worldVec.dot(efLDirWorld))/2.0

        # render a spherical fixed point above the character
        renderUtils.setColor([0.0, 1.0, 1.0])
        fixedPoint = np.array([0, 0.65, 0])
        renderUtils.drawSphere(pos=fixedPoint, rad=0.045)

        #numCollisions = self.clothScene.getNumSelfCollisions()
        collisionLocations = self.clothScene.getSelfCollisionLocations(recompute=False)
        #print(collisionLocations)
        lines = []
        for i in range(int(len(collisionLocations)/3)):
            lines.append([fixedPoint, np.array(collisionLocations[i*3:i*3+3])])
        renderUtils.drawLines(lines)

        #print("orientation error = " + str(orientationError))

        renderUtils.drawProgressBar(topLeft=[600, self.viewer.viewport[3] - 12], h=16, w=60, progress=orientationError, color=[0.0, 3.0, 0])

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