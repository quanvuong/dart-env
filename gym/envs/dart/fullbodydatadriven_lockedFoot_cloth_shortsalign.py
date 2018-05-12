# This environment is created by Alexander Clegg (alexanderwclegg@gmail.com)

import numpy as np
from gym import utils
from gym.envs.dart.dart_cloth_env import *
from gym.envs.dart.fullbodydatadriven_lockedFoot_cloth_base import *
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

class DartClothFullBodyDataDrivenLockedFootClothShortsAlignEnv(DartClothFullBodyDataDrivenLockedFootClothBaseEnv, utils.EzPickle):
    def __init__(self):
        #feature flags
        rendering = False
        clothSimulation = True
        renderCloth = True
        self.renderRestPose = True
        frameskip = 4
        dt = 0.002

        # reward flags
        self.restPoseReward         = True
        self.stableCOMReward        = True
        # dressing reward flags
        self.waistContainmentReward = True
        self.deformationPenalty     = True
        self.waistHorizontalReward  = True #penalize normal of the waist feature from perpendicular to ground plane


        # reward weights
        self.restPoseRewardWeight  = 0.5
        self.stableCOMRewardWeight = 10
        # dressing reward weights
        self.waistContainmentRewardWeight = 5
        self.deformationPenaltyWeight = 5
        self.waistHorizontalReward = 1

        #termination conditions
        self.wrongEnterTermination = True  # terminate if the foot enters the pant legs
        self.stabilityTermination = False  # if COM outside stability region, terminate

        #other variables
        self.prevTau = None
        self.restPose = None
        self.fingertip = np.array([0, -0.08, 0])
        self.toeOffset = np.array([0, 0, -0.2])
        self.previousDeformationReward = 0

        # handle nodes
        self.handleNodes = []
        self.updateHandleNodesFrom = [13, 18]

        self.actuatedDofs = np.arange(34)
        observation_size = len(self.actuatedDofs) * 3  # q(sin,cos), dq
        if self.stableCOMReward:
            observation_size += 3 #world COM location
        if clothSimulation:
            observation_size += 40 * 3  # haptic sensor readings

        DartClothFullBodyDataDrivenLockedFootClothBaseEnv.__init__(self,
                                                                  rendering=rendering,
                                                                  screensize=(1280,920),
                                                                  clothMeshFile="shorts_med.obj",
                                                                  clothScale=np.array([0.9,0.9,0.9]),
                                                                  obs_size=observation_size,
                                                                  simulateCloth=clothSimulation,
                                                                  left_foot_locked = True,
                                                                  frameskip=frameskip,
                                                                  dt=dt)


        #define shorts garment features
        self.targetGripVerticesL = [85, 22, 13, 92, 212, 366]
        self.targetGripVerticesR = [5, 146, 327, 215, 112, 275]
        self.legEndVerticesL = [42, 384, 118, 383, 37, 164, 123, 404, 36, 406, 151, 338, 38, 235, 81, 266, 40, 247, 80, 263]
        self.legEndVerticesR = [45, 394, 133, 397, 69, 399, 134, 401, 20, 174, 127, 336, 21, 233, 79, 258, 29, 254, 83, 264]
        self.legMidVerticesR = [91, 400, 130, 396, 128, 162, 165, 141, 298, 417, 19, 219, 270, 427, 440, 153, 320, 71, 167, 30, 424]
        self.legMidVerticesL = [280, 360, 102, 340, 196, 206, 113, 290, 41, 178, 72, 325, 159, 147, 430, 291, 439, 55, 345, 125, 429]
        self.waistVertices = [215, 278, 110, 217, 321, 344, 189, 62, 94, 281, 208, 107, 188, 253, 228, 212, 366, 92, 160, 119, 230, 365, 77, 0, 104, 163, 351, 120, 295, 275, 112]
        #leg entrance L and R
        self.legStartVerticesR = [232, 421, 176, 82, 319, 403, 256, 314, 98, 408, 144, 26, 261, 84, 434, 432, 27, 369, 132, 157, 249, 203, 99, 184, 437]
        self.legStartVerticesL = [209, 197, 257, 68, 109, 248, 238, 357, 195, 108, 222, 114, 205, 86, 273, 35, 239, 137, 297, 183, 105]


        self.gripFeatureL = ClothFeature(verts=self.targetGripVerticesL, clothScene=self.clothScene)
        self.gripFeatureR = ClothFeature(verts=self.targetGripVerticesR, clothScene=self.clothScene)

        self.legEndFeatureL = ClothFeature(verts=self.legEndVerticesL, clothScene=self.clothScene)
        self.legEndFeatureR = ClothFeature(verts=self.legEndVerticesR, clothScene=self.clothScene)
        self.legMidFeatureL = ClothFeature(verts=self.legMidVerticesL, clothScene=self.clothScene)
        self.legMidFeatureR = ClothFeature(verts=self.legMidVerticesR, clothScene=self.clothScene)
        self.legStartFeatureR = ClothFeature(verts=self.legStartVerticesR, clothScene=self.clothScene)
        self.legStartFeatureL = ClothFeature(verts=self.legStartVerticesL, clothScene=self.clothScene)

        self.waistFeature = ClothFeature(verts=self.waistVertices, clothScene=self.clothScene)


        self.simulateCloth = clothSimulation

        # setup the handle nodes
        if self.simulateCloth:
            self.handleNodes.append(HandleNode(self.clothScene, org=np.array([0.05, 0.034, -0.975])))
            self.handleNodes.append(HandleNode(self.clothScene, org=np.array([0.05, 0.034, -0.975])))

        if not renderCloth:
            self.clothScene.renderClothFill = False
            self.clothScene.renderClothBoundary = False
            self.clothScene.renderClothWires = False

        if self.restPoseReward:
            self.rewardsData.addReward(label="rest pose", rmin=-51.0, rmax=0, rval=0, rweight=self.restPoseRewardWeight)

        if self.stableCOMReward:
            self.rewardsData.addReward(label="stable COM", rmin=-1.0, rmax=0, rval=0, rweight=self.stableCOMRewardWeight)

        if self.waistContainmentReward:
            self.rewardsData.addReward(label="waist containment", rmin=-1.0, rmax=1.0, rval=0, rweight=self.waistContainmentRewardWeight)

        if self.deformationPenalty:
            self.rewardsData.addReward(label="deformation", rmin=-1.0, rmax=0, rval=0, rweight=self.deformationPenaltyWeight)

        if self.waistHorizontalReward:
            self.rewardsData.addReward(label="waist horizontal", rmin=-1.0, rmax=1.0, rval=0, rweight=self.waistHorizontalReward)

    def _getFile(self):
        return __file__

    def updateBeforeSimulation(self):
        #any pre-sim updates should happen here

        self.gripFeatureL.fitPlane()
        self.gripFeatureR.fitPlane()
        self.legEndFeatureL.fitPlane()
        self.legEndFeatureR.fitPlane()
        self.legMidFeatureL.fitPlane()
        self.legMidFeatureR.fitPlane()
        self.legStartFeatureL.fitPlane()
        self.legStartFeatureR.fitPlane()
        self.waistFeature.fitPlane()

        # update handle nodes
        if len(self.handleNodes) > 1 and self.reset_number > 0:
            self.handleNodes[0].setTransform(self.robot_skeleton.bodynodes[self.updateHandleNodesFrom[0]].T)
            self.handleNodes[1].setTransform(self.robot_skeleton.bodynodes[self.updateHandleNodesFrom[1]].T)
            self.handleNodes[0].step()
            self.handleNodes[1].step()

        a=0

    def checkTermination(self, tau, s, obs):
        #check the termination conditions and return: done,reward

        #this is handled in the base env now...
        if np.amax(np.absolute(s[:len(self.robot_skeleton.q)])) > 3:
            print("Detecting potential instability")
            print(s)
            print(self.rewardTrajectory)
            print(self.stateTraj[-2:])
            return True, -500
        elif not np.isfinite(s).all():
            print("Infinite value detected..." + str(s))
            return True, -500

        # stability termination
        if self.stabilityTermination:
            #if not self.stableCOM:
            #TODO: this
            if False:
                return True, 0

        if self.wrongEnterTermination:
            limbInsertionErrorL = pyutils.limbFeatureProgress( limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesLegR, offset=self.toeOffset), feature=self.legEndFeatureL)
            limbInsertionErrorR = pyutils.limbFeatureProgress( limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesLegR, offset=self.toeOffset), feature=self.legEndFeatureR)
            if limbInsertionErrorL > 0 or limbInsertionErrorR > 0:
                return True, -1500


        return False, 0

    def computeReward(self, tau):
        #compute and return reward at the current state
        self.prevTau = tau
        reward_record = []

        reward_restPose = 0
        if self.restPoseReward and self.restPose is not None:
            dist = np.linalg.norm(self.robot_skeleton.q - self.restPose)
            reward_restPose = max(-51, -dist)
            reward_record.append(reward_restPose)

        reward_stableCOM = 0
        if self.stableCOMReward:
            footCOM = self.robot_skeleton.bodynodes[0].com()
            bodyCOM = self.robot_skeleton.com()
            projFootCOM = np.array([footCOM[0], footCOM[2]])
            projBodyCOM = np.array([bodyCOM[0], bodyCOM[2]])
            reward_stableCOM = max(-1, -np.linalg.norm(projFootCOM-projBodyCOM))
            reward_record.append(reward_stableCOM)

        reward_waistContainment = 0
        if self.waistContainmentReward:
            if self.simulateCloth:
                self.limbProgress = pyutils.limbFeatureProgress(
                    limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesLegR,
                                                      offset=self.toeOffset), feature=self.waistFeature)
                reward_waistContainment = self.limbProgress
                # print(reward_waistContainment)
                '''if reward_waistContainment <= 0:  # replace centroid distance penalty with border distance penalty
                    # distance to feature
                    distance2Feature = 999.0
                    toe = self.robot_skeleton.bodynodes[20].to_world(self.toeOffset)
                    for v in self.waistFeature.verts:
                        dist = np.linalg.norm(self.clothScene.getVertexPos(cid=0, vid=v) - toe)
                        if dist < distance2Feature:
                            distance2Feature = dist
                            reward_waistContainment = - distance2Feature'''
                            # print(reward_waistContainment)
            reward_record.append(reward_waistContainment)

        clothDeformation = 0
        if self.simulateCloth is True:
            clothDeformation = self.clothScene.getMaxDeformationRatio(0)
            self.deformation = clothDeformation

        reward_clothdeformation = 0
        if self.deformationPenalty is True:
            # reward_clothdeformation = (math.tanh(9.24 - 0.5 * clothDeformation) - 1) / 2.0  # near 0 at 15, ramps up to -1.0 at ~22 and remains constant
            reward_clothdeformation = -(math.tanh(
                0.14 * (clothDeformation - 25)) + 1) / 2.0  # near 0 at 15, ramps up to -1.0 at ~35 and remains constant
            reward_record.append(reward_clothdeformation)
        self.previousDeformationReward = reward_clothdeformation

        reward_waistHorizontal = 0
        if self.waistHorizontalReward:
            reward_waistHorizontal = self.waistFeature.plane.normal.dot(np.array([0,-1.0,0]))
            reward_record.append(reward_waistHorizontal)

        # update the reward data storage
        self.rewardsData.update(rewards=reward_record)

        self.reward = reward_restPose * self.restPoseRewardWeight\
                    + reward_stableCOM * self.stableCOMRewardWeight \
                    + reward_waistContainment * self.waistContainmentRewardWeight \
                    + reward_clothdeformation * self.deformationPenaltyWeight \
                    + reward_waistHorizontal * self.waistContainmentRewardWeight

        return self.reward

    def _get_obs(self):
        theta = np.zeros(len(self.actuatedDofs))
        dtheta = np.zeros(len(self.actuatedDofs))
        for ix, dof in enumerate(self.actuatedDofs):
            theta[ix] = self.robot_skeleton.q[dof]
            dtheta[ix] = self.robot_skeleton.dq[dof]

        obs = np.concatenate([np.cos(theta), np.sin(theta), dtheta]).ravel()

        if self.stableCOMReward:
            #print(self.robot_skeleton.com())
            obs = np.concatenate([obs, np.array(self.robot_skeleton.com())]).ravel()

        # haptic observations
        f = np.zeros(40 * 3)
        if self.simulateCloth:
            f = self.clothScene.getHapticSensorObs()
        obs = np.concatenate([obs, f]).ravel()

        return obs

    def additionalResets(self):
        #do any additional resetting here
        qpos = np.array([0.00643240781574, 0.00343494316121, -0.00158793848997, -0.00693905055509, -0.00728992955043, -1.20776102819e-05, 0.00183865259133, 0.0113234850122, -0.00952845282162, -5.34278022043e-05, -0.00586483901021, -0.00238655645607, 0.00737146821647, -0.00203432739702, -0.00346169932589, -0.00801843751728, 0.00870087139625, -0.00467091876336, 0.00581036298978, -0.111714904276, 1.54965814933, -0.0072440773253, -0.6, -0.00342547114134, 0.00815570604697, 0.00392769783193, -0.000885047826654, 0.11264932594, 1.55764492201, 0.0139947806921, -0.6, -0.00658075724121, -0.00602227969012, -0.00765689804013])
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-0.01, high=0.01, size=self.robot_skeleton.ndofs)
        qpos = qpos + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        #self.restPose = np.array(qpos)

        #rest pose is one foot
        self.restPose = np.array([0.402149945024, 0.26417742198, 0.694509869782, 0.0703040144608, 0.218999034272, 0.000129805833933, -0.388680623056, -1.57073590178, 0.431718882029, 1.70254936482, 0.00473949413903, -0.00927479809255, 0.136183050592, 0.297959293122, 0.00598932040189, -0.0101696256212, -0.0107320923733, 0.0036338670046, -0.00188298068761, 0.00402389215485, 1.69447708293, -0.00963879710017, -0.00346143295241, -0.00275771262968, -0.00832522365649, 0.00645267428891, -0.00205846347826, 0.00710408240934, 1.52907046286, 0.00590862627021, 9.77641432949e-05, 0.00808337306807, 0.00768925048198, 0.00379325587092])

        if self.simulateCloth:
            RX = pyutils.rotateX(-1.56)
            self.clothScene.rotateCloth(cid=0, R=RX)
            self.clothScene.translateCloth(cid=0, T=np.array([0.45, -0.7, -1.35]))
            #self.clothScene.translateCloth(0, np.array([0, 3.0, 0]))

            self.gripFeatureL.fitPlane()
            self.gripFeatureR.fitPlane()
            self.legEndFeatureL.fitPlane()
            self.legEndFeatureR.fitPlane()
            self.legMidFeatureL.fitPlane()
            self.legMidFeatureR.fitPlane()
            self.legStartFeatureL.fitPlane()
            self.legStartFeatureR.fitPlane()
            self.waistFeature.fitPlane()

            # sort out feature normals
            CPLE_CPLM = self.legEndFeatureL.plane.org - self.legMidFeatureL.plane.org
            CPLS_CPLM = self.legStartFeatureL.plane.org - self.legMidFeatureL.plane.org

            CPRE_CPRM = self.legEndFeatureR.plane.org - self.legMidFeatureR.plane.org
            CPRS_CPRM = self.legStartFeatureR.plane.org - self.legMidFeatureR.plane.org

            estimated_groin = (self.legStartFeatureL.plane.org + self.legStartFeatureR.plane.org) / 2.0
            CPW_EG = self.waistFeature.plane.org - estimated_groin

            if CPW_EG.dot(self.waistFeature.plane.normal) > 0:
                self.waistFeature.plane.normal *= -1.0

            if CPLS_CPLM.dot(self.legStartFeatureL.plane.normal) > 0:
                self.legStartFeatureL.plane.normal *= -1.0

            if CPLE_CPLM.dot(self.legEndFeatureL.plane.normal) < 0:
                self.legEndFeatureL.plane.normal *= -1.0

            if CPLE_CPLM.dot(self.legMidFeatureL.plane.normal) < 0:
                self.legMidFeatureL.plane.normal *= -1.0

            if CPRS_CPRM.dot(self.legStartFeatureR.plane.normal) > 0:
                self.legStartFeatureR.plane.normal *= -1.0

            if CPRE_CPRM.dot(self.legEndFeatureR.plane.normal) < 0:
                self.legEndFeatureR.plane.normal *= -1.0

            if CPRE_CPRM.dot(self.legMidFeatureR.plane.normal) < 0:
                self.legMidFeatureR.plane.normal *= -1.0

            if len(self.handleNodes) > 1:
                self.handleNodes[0].clearHandles()
                self.handleNodes[0].addVertices(verts=self.targetGripVerticesR)
                self.handleNodes[0].setOrgToCentroid()
                self.handleNodes[0].setTransform(self.robot_skeleton.bodynodes[self.updateHandleNodesFrom[0]].T)
                self.handleNodes[0].recomputeOffsets()

                self.handleNodes[1].clearHandles()
                self.handleNodes[1].addVertices(verts=self.targetGripVerticesL)
                self.handleNodes[1].setOrgToCentroid()
                self.handleNodes[1].setTransform(self.robot_skeleton.bodynodes[self.updateHandleNodesFrom[1]].T)
                self.handleNodes[1].recomputeOffsets()
        a=0

    def extraRenderFunction(self):
        renderUtils.setColor(color=[0.0, 0.0, 0])
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3d(0,0,0)
        GL.glVertex3d(-1,0,0)
        GL.glEnd()

        renderUtils.setColor([0,0,0])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[10].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[10].to_world(np.array([0.0,-0.3,-0.075]))])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[15].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[15].to_world(np.array([0.0,-0.3,-0.075]))])

        #render the body and foot coms
        footCOM = self.robot_skeleton.bodynodes[0].com()
        bodyCOM = self.robot_skeleton.com()

        lines = [
            [np.array([footCOM[0], 1.0, footCOM[2]]), np.array([footCOM[0], -2.0, footCOM[2]])],
            [np.array([bodyCOM[0], 1.0, bodyCOM[2]]), np.array([bodyCOM[0], -2.0, bodyCOM[2]])]
        ]
        #print(lines)
        renderUtils.setColor(color=[0.0, 2.0, 0])
        renderUtils.drawLines(lines=lines)

        if self.renderRestPose:
            renderUtils.setColor([0, 0, 0])
            links = pyutils.getRobotLinks(self.robot_skeleton, pose=self.restPose)
            renderUtils.drawLines(lines=links)

        self.gripFeatureL.drawProjectionPoly(renderNormal=False, renderBasis=False, fillColor=[1.0, 0.0, 0.0])
        self.gripFeatureR.drawProjectionPoly(renderNormal=False, renderBasis=False, fillColor=[0.0, 1.0, 0.0])
        self.legEndFeatureL.drawProjectionPoly(renderNormal=True, renderBasis=False, fillColor=[1.0, 0.0, 0.0])
        self.legEndFeatureR.drawProjectionPoly(renderNormal=True, renderBasis=False, fillColor=[0.0, 1.0, 0.0])
        self.legMidFeatureL.drawProjectionPoly(renderNormal=True, renderBasis=False, fillColor=[1.0, 0.0, 0.0])
        self.legMidFeatureR.drawProjectionPoly(renderNormal=True, renderBasis=False, fillColor=[0.0, 1.0, 0.0])
        self.legStartFeatureL.drawProjectionPoly(renderNormal=True, renderBasis=False, fillColor=[1.0, 0.0, 0.0])
        self.legStartFeatureR.drawProjectionPoly(renderNormal=True, renderBasis=False, fillColor=[0.0, 1.0, 0.0])
        self.waistFeature.drawProjectionPoly(renderNormal=True, renderBasis=False, fillColor=[0.0, 0.0, 1.0])

        renderUtils.setColor(color=[1.0,0,0])
        distance2Feature = 999.0
        pos = np.zeros(3)
        toe = self.robot_skeleton.bodynodes[6].to_world(self.toeOffset)
        lines = []
        if self.reset_number > 0:
            '''for v in self.waistFeature.verts:
                # print(v)
                dist = np.linalg.norm(self.clothScene.getVertexPos(cid=0, vid=v) - toe)
                if dist < distance2Feature:
                    distance2Feature = dist
                    pos = self.clothScene.getVertexPos(cid=0, vid=v)
                    # print(dist)'''
            pos = self.waistFeature.plane.org

            lines.append([pos, toe])

        renderUtils.drawLines(lines)

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
