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

class DartClothUpperBodyDataDrivenClothPhaseInterpolateJacket1Env(DartClothUpperBodyDataDrivenClothBaseEnv, utils.EzPickle):
    def __init__(self):
        #feature flags
        rendering = False
        clothSimulation = True
        renderCloth = True
        dt = 0.002
        frameskip = 5

        #observation terms
        self.contactIDInObs = True  # if true, contact ids are in obs
        self.hapticsInObs   = True  # if true, haptics are in observation
        self.prevTauObs     = False  # if true, previous action in observation

        #reward flags
        self.uprightReward              = True #if true, rewarded for 0 torso angle from vertical
        self.stableHeadReward           = True
        self.deformationPenalty         = True
        self.restPoseReward             = True
        self.rightTargetReward          = True
        self.leftTargetReward           = True
        self.clothPlacementReward       = True #reward closeness of efL to a vertex

        self.uprightRewardWeight        = 1
        self.stableHeadRewardWeight     = 1
        self.deformationPenaltyWeight   = 5
        self.restPoseRewardWeight       = 2
        self.rightTargetRewardWeight    = 1
        self.leftTargetRewardWeight     = 1
        self.clothPlacementRewardWeight  = 1

        #other flags
        self.collarTermination = False  # if true, rollout terminates when collar is off the head/neck
        self.collarTerminationCD = 0 #number of frames to ignore collar at the start of simulation (gives time for the cloth to drop)
        self.hapticsAware       = True  # if false, 0's for haptic input
        self.loadTargetsFromROMPositions = False
        self.resetPoseFromROMPoints = False
        self.resetTime = 0
        self.resetStateFromDistribution = True
        self.resetDistributionPrefix = "saved_control_states_jacket/enter_seq_transition"
        self.resetDistributionSize = 20
        self.state_save_directory = "saved_control_states/"

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
        self.previousDeformationReward = 0
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

        DartClothUpperBodyDataDrivenClothBaseEnv.__init__(self,
                                                          rendering=rendering,
                                                          screensize=(1280,720),
                                                          clothMeshFile="jacketmedium.obj",
                                                          clothMeshStateFile = "endJacketSleeveR.obj",
                                                          clothScale=np.array([0.7,0.7,0.5]),
                                                          obs_size=observation_size,
                                                          simulateCloth=clothSimulation,
                                                          dt=dt,
                                                          frameskip=frameskip)

        #define pose and obj files for reset
        '''self.resetStateFileNames = ["endDropGrip1"] #each state name 'n' should refer to both a character state file: " characterState_'n' " and the cloth state file: " 'n'.obj ".
        #load reset poses from file
        for name in self.resetStateFileNames:
            self.clothScene.addResetStateFrom(filename=name+".obj")'''

        #clothing features
        self.clothPlacementVert = 183
        self.sleeveRVerts = [46, 697, 1196, 696, 830, 812, 811, 717, 716, 718, 968, 785, 1243, 783, 1308, 883, 990, 739, 740, 742, 1318, 902, 903, 919, 737, 1218, 736, 1217]
        self.sleeveRMidVerts = [1054, 1055, 1057, 1058, 1060, 1061, 1063, 1052, 1051, 1049, 1048, 1046, 1045, 1043, 1042, 1040, 1039, 734, 732, 733]
        self.sleeveREndVerts = [228, 1059, 229, 1062, 230, 1064, 227, 1053, 226, 1050, 225, 1047, 224, 1044, 223, 1041, 142, 735, 141, 1056]
        self.CP0Feature = ClothFeature(verts=self.sleeveRVerts, clothScene=self.clothScene)
        self.CP1Feature = ClothFeature(verts=self.sleeveREndVerts, clothScene=self.clothScene)
        self.CP2Feature = ClothFeature(verts=self.sleeveRMidVerts, clothScene=self.clothScene)

        self.simulateCloth = clothSimulation
        if self.simulateCloth:
            self.handleNode = HandleNode(self.clothScene, org=np.array([0.05, 0.034, -0.975]))

        if not renderCloth:
            self.clothScene.renderClothFill = False
            self.clothScene.renderClothBoundary = False
            self.clothScene.renderClothWires = False

        for i in range(len(self.robot_skeleton.dofs)):
            self.robot_skeleton.dofs[i].set_damping_coefficient(3.0)

        # load rewards into the RewardsData structure
        if self.uprightReward:
            self.rewardsData.addReward(label="upright", rmin=-2.5, rmax=0, rval=0, rweight=self.uprightRewardWeight)

        if self.stableHeadReward:
            self.rewardsData.addReward(label="stable head",rmin=-1.2,rmax=0,rval=0, rweight=self.stableHeadRewardWeight)

        if self.deformationPenalty:
            self.rewardsData.addReward(label="deformation", rmin=-1.0, rmax=0, rval=0,
                                       rweight=self.deformationPenaltyWeight)

        if self.restPoseReward:
            self.rewardsData.addReward(label="rest pose", rmin=-51.0, rmax=0, rval=0,
                                       rweight=self.restPoseRewardWeight)

        if self.rightTargetReward:
            self.rewardsData.addReward(label="right ef", rmin=-1.5, rmax=0, rval=0,
                                       rweight=self.rightTargetRewardWeight)

        if self.leftTargetReward:
            self.rewardsData.addReward(label="left ef", rmin=-1.5, rmax=0, rval=0,
                                       rweight=self.leftTargetRewardWeight)

        if self.clothPlacementReward:
            self.rewardsData.addReward(label="cloth placement", rmin=-1.5, rmax=0, rval=0,
                                       rweight=self.clothPlacementRewardWeight)

    def _getFile(self):
        return __file__

    def updateBeforeSimulation(self):
        #any pre-sim updates should happen here
        #fingertip = np.array([0.0, -0.065, 0.0])
        wRFingertip1 = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)
        wLFingertip1 = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
        self.localRightEfShoulder1 = self.robot_skeleton.bodynodes[3].to_local(wRFingertip1)  # right fingertip in right shoulder local frame
        self.localLeftEfShoulder1 = self.robot_skeleton.bodynodes[8].to_local(wLFingertip1)  # left fingertip in left shoulder local frame
        #self.leftTarget = pyutils.getVertCentroid(verts=self.targetGripVertices, clothscene=self.clothScene) + pyutils.getVertAvgNorm(verts=self.targetGripVertices, clothscene=self.clothScene)*0.03

        self.CP0Feature.fitPlane()
        self.CP1Feature.fitPlane()
        self.CP2Feature.fitPlane()

        # update handle nodes
        if self.handleNode is not None:
            if self.updateHandleNodeFrom >= 0:
                self.handleNode.setTransform(self.robot_skeleton.bodynodes[self.updateHandleNodeFrom].T)
            self.handleNode.step()

        if self.numSteps > 1:
            self.handleNode.clearHandles()

        a=0

    def checkTermination(self, tau, s, obs):
        #check the termination conditions and return: done,reward
        topHead = self.robot_skeleton.bodynodes[14].to_world(np.array([0, 0.25, 0]))
        bottomHead = self.robot_skeleton.bodynodes[14].to_world(np.zeros(3))
        bottomNeck = self.robot_skeleton.bodynodes[13].to_world(np.zeros(3))

        if np.amax(np.absolute(s[:len(self.robot_skeleton.q)])) > 10:
            print("Detecting potential instability")
            print(s)
            return True, -2000
        elif not np.isfinite(s).all():
            print("Infinite value detected..." + str(s))
            return True, -2000
        elif self.simulateCloth: #terminate if sleeve slips off right arm
            sleeveSlipError = pyutils.limbFeatureProgress(
                limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesR[:3],
                                                  offset=None), feature=self.CP0Feature)
            if sleeveSlipError <= 0:
                return True, -2000
        return False, 0

    def computeReward(self, tau):
        #compute and return reward at the current state
        #fingertip = np.array([0.0, -0.065, 0.0])
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
        reward_record = []

        # force magnitude penalty
        reward_ctrl = -np.square(tau).sum()

        # reward for maintaining posture
        reward_upright = 0
        if self.uprightReward:
            reward_upright = max(-2.5, -abs(self.robot_skeleton.q[0]) - abs(self.robot_skeleton.q[1]))
            reward_record.append(reward_upright)

        reward_stableHead = 0
        if self.stableHeadReward:
            reward_stableHead = max(-1.2, -abs(self.robot_skeleton.q[19]) - abs(self.robot_skeleton.q[20]))
            reward_record.append(reward_stableHead)

        clothDeformation = 0
        if self.simulateCloth:
            clothDeformation = self.clothScene.getMaxDeformationRatio(0)
            self.deformation = clothDeformation

        reward_clothdeformation = 0
        if self.deformationPenalty is True:
            # reward_clothdeformation = (math.tanh(9.24 - 0.5 * clothDeformation) - 1) / 2.0  # near 0 at 15, ramps up to -1.0 at ~22 and remains constant
            reward_clothdeformation = -(math.tanh(
                0.14 * (clothDeformation - 25)) + 1) / 2.0  # near 0 at 15, ramps up to -1.0 at ~22 and remains constant
            reward_record.append(reward_clothdeformation)

        self.previousDeformationReward = reward_clothdeformation

        reward_restPose = 0
        if self.restPoseReward:
            if self.restPose is not None:
                '''z = 0.5  # half the max magnitude (e.g. 0.5 -> [0,1])
                s = 1.0  # steepness (higher is steeper)
                l = 4.2  # translation
                dist = np.linalg.norm(self.robot_skeleton.q - self.restPose)
                reward_restPose = -(z * math.tanh(s * (dist - l)) + z)'''
                dist = np.linalg.norm(self.robot_skeleton.q - self.restPose)
                reward_restPose = max(-51, -dist)
                # print("distance: " + str(dist) + " -> " + str(reward_restPose))
            reward_record.append(reward_restPose)

        reward_rightTarget = 0
        if self.rightTargetReward:
            rDist = np.linalg.norm(self.rightTarget - wRFingertip2)
            reward_rightTarget = -rDist - rDist**2
            reward_record.append(reward_rightTarget)

        reward_leftTarget = 0
        if self.leftTargetReward:
            lDist = np.linalg.norm(self.leftTarget - wLFingertip2)
            reward_leftTarget = -lDist - lDist**2
            reward_record.append(reward_leftTarget)

        reward_clothplacement = 0
        if self.clothPlacementReward:
            vpos = self.clothScene.getVertexPos(cid=0,vid=self.clothPlacementVert) - self.clothScene.getVertNormal(cid=0,vid=self.clothPlacementVert)*0.02
            lDist = np.linalg.norm(vpos - wLFingertip2)
            reward_clothplacement = -lDist - lDist ** 2
            reward_record.append(reward_clothplacement)

        # update the reward data storage
        self.rewardsData.update(rewards=reward_record)

        #print("reward_restPose: " + str(reward_restPose))
        #print("reward_leftTarget: " + str(reward_leftTarget))
        self.reward = reward_ctrl * 0 \
                      + reward_upright * self.uprightRewardWeight \
                      + reward_stableHead * self.stableHeadRewardWeight \
                      + reward_clothdeformation * self.deformationPenaltyWeight \
                      + reward_restPose * self.restPoseRewardWeight \
                      + reward_rightTarget * self.rightTargetRewardWeight\
                      + reward_leftTarget * self.leftTargetRewardWeight\
                      + reward_clothplacement * self.clothPlacementRewardWeight
        return self.reward

    def _get_obs(self):
        f_size = 66
        '22x3 dofs, 22x3 sensors, 7x2 targets(toggle bit, cartesian, relative)'
        theta = np.zeros(len(self.actuatedDofs))
        dtheta = np.zeros(len(self.actuatedDofs))
        for ix, dof in enumerate(self.actuatedDofs):
            theta[ix] = self.robot_skeleton.q[dof]
            dtheta[ix] = self.robot_skeleton.dq[dof]

        #fingertip = np.array([0.0, -0.065, 0.0])

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

        return obs

    def additionalResets(self):
        '''if self.resetTime > 0:
            print("reset " + str(self.reset_number) + " after " + str(time.time()-self.resetTime))
        '''
        self.resetTime = time.time()
        #do any additional resetting here
        #fingertip = np.array([0, -0.065, 0])
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
        qpos = np.array(
            [-0.0483053659505, 0.0321213273351, 0.0173036909392, 0.00486290205677, -0.00284350018845, -0.634602301004,
             -0.359172622713, 0.0792754054027, 2.66867203095, 0.00489456931428, 0.000476966442889, 0.0234663491334,
             -0.0254520098678, 0.172782859361, -1.31351102137, 0.702315566312, 1.73993331669, -0.0422811572637,
             0.586669332152, -0.0122329947565, 0.00179736869435, -8.0625896949e-05])
        #self.set_state(qpos, qvel)
        self.loadCharacterState("characterState_startJacketSleeveL")

        #find end effector targets and set restPose from solution
        #fingertip = np.array([0.0, -0.065, 0.0])
        self.rightTarget = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)
        self.leftTarget = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
        #print("right target: " + str(self.rightTarget))
        #print("left target: " + str(self.leftTarget))
        self.restPose = np.array(self.robot_skeleton.q)
        #print(self.restPose)

        if self.resetStateFromDistribution:
            if self.reset_number == 0: #load the distribution
                count = 0
                objfname_ix = self.resetDistributionPrefix + "%05d" % count
                while os.path.isfile(objfname_ix + ".obj"):
                    count += 1
                    #print(objfname_ix)
                    self.clothScene.addResetStateFrom(filename=objfname_ix+".obj")
                    objfname_ix = self.resetDistributionPrefix + "%05d" % count

            resetStateNumber = random.randint(0,self.resetDistributionSize-1)
            #resetStateNumber = 0
            #resetStateNumber = self.reset_number%self.resetDistributionSize
            #print(resetStateNumber)
            charfname_ix = self.resetDistributionPrefix + "_char%05d" % resetStateNumber
            self.clothScene.setResetState(cid=0, index=resetStateNumber)
            self.loadCharacterState(filename=charfname_ix)

        else:
            self.loadCharacterState(filename="characterState_endJacketSleeveR")

        if self.handleNode is not None:
            self.handleNode.clearHandles()
            self.handleNode.addVertices(verts=[727, 138, 728, 1361, 730, 961, 1213, 137, 724, 1212, 726, 960, 964, 729, 155, 772])
            self.handleNode.setOrgToCentroid()
            if self.updateHandleNodeFrom >= 0:
                self.handleNode.setTransform(self.robot_skeleton.bodynodes[self.updateHandleNodeFrom].T)
            self.handleNode.recomputeOffsets()

        if self.simulateCloth:
            self.CP0Feature.fitPlane()
            self.CP1Feature.fitPlane()
            self.CP2Feature.fitPlane()

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
        if self.CP0Feature is not None:
            self.CP0Feature.drawProjectionPoly()

        renderUtils.setColor([0,0,0])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[4].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[4].to_world(np.array([0.0,-0.3,-0.075]))])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[9].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[9].to_world(np.array([0.0,-0.3,-0.075]))])

        links = pyutils.getRobotLinks(self.robot_skeleton, pose=self.restPose)
        renderUtils.drawLines(lines=links)

        #render targets
        #fingertip = np.array([0,-0.065,0])
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

        if self.clothPlacementReward:
            efL = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
            renderUtils.setColor(color=[0, 1.0, 0])
            vpos = self.clothScene.getVertexPos(cid=0,vid=self.clothPlacementVert) - self.clothScene.getVertNormal(cid=0,vid=self.clothPlacementVert)*0.02
            renderUtils.drawSphere(pos=vpos, rad=0.02)
            renderUtils.drawLineStrip(points=[vpos, efL])

        m_viewport = self.viewer.viewport
        # print(m_viewport)
        self.rewardsData.render(topLeft=[m_viewport[2] - 410, m_viewport[3] - 15],
                                dimensions=[400, -m_viewport[3] + 30])

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
                renderUtils.renderDofs(robot=self.robot_skeleton, restPose=self.restPose, renderRestPose=True)

            renderUtils.drawProgressBar(topLeft=[600, self.viewer.viewport[3] - 30], h=16, w=60, progress=-self.previousDeformationReward, color=[1.0, 0.0, 0])
