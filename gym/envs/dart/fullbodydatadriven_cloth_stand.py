# This environment is created by Alexander Clegg (alexanderwclegg@gmail.com)

import numpy as np
from gym import utils
from gym.envs.dart.dart_cloth_env import *
from gym.envs.dart.fullbodydatadriven_cloth_base import *
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

class DartClothFullBodyDataDrivenClothStandEnv(DartClothFullBodyDataDrivenClothBaseEnv, utils.EzPickle):
    def __init__(self):
        #feature flags
        rendering = False
        clothSimulation = False
        renderCloth = True

        #reward flags
        self.restPoseReward         = True
        self.stabilityCOMReward     = True
        self.contactReward          = True
        self.COMHeightReward        = True
        self.aliveBonusReward       = True #rewards rollout duration to counter suicidal tendencies

        #reward weights
        self.restPoseRewardWeight       = 1
        self.stabilityCOMRewardWeight   = 1
        self.contactRewardWeight        = 1
        self.COMHeightRewardWeight      = 1
        self.aliveBonusRewardWeight     = 5

        #other flags
        self.stabilityTermination = True #if COM outside stability region, terminate #TODO: timed?
        self.contactTermiantion   = True #if anything except the feet touch the ground, terminate

        #other variables
        self.prevTau = None
        self.restPose = None
        self.stabilityPolygon = [] #an ordered point set representing the stability region of the desired foot contacts
        self.stabilityPolygonCentroid = np.zeros(3)
        self.projectedCOM = np.zeros(3)
        self.COMHeight = 0.0
        self.stableCOM = True
        self.numFootContacts = 0
        self.lFootContact = False
        self.rFootContact = False
        self.nonFootContact = False #used for detection of failure

        self.actuatedDofs = np.arange(34)
        observation_size = 0
        observation_size = 37 * 3 + 6 #q[:3], q[3:](sin,cos), dq
        observation_size += 3 # COM
        observation_size += 2 # binary contact per foot with ground


        DartClothFullBodyDataDrivenClothBaseEnv.__init__(self,
                                                          rendering=rendering,
                                                          screensize=(1080,920),
                                                          clothMeshFile="capri_med.obj",
                                                          clothScale=np.array([1.0,1.0,1.0]),
                                                          obs_size=observation_size,
                                                          simulateCloth=clothSimulation)


        self.simulateCloth = clothSimulation

        if not renderCloth:
            self.clothScene.renderClothFill = False
            self.clothScene.renderClothBoundary = False
            self.clothScene.renderClothWires = False

        if self.restPoseReward:
            self.rewardsData.addReward(label="restPose", rmin=-51.0, rmax=0, rval=0, rweight=self.restPoseRewardWeight)

        if self.stabilityCOMReward:
            self.rewardsData.addReward(label="stability", rmin=-0.5, rmax=0, rval=0, rweight=self.stabilityCOMRewardWeight)

        if self.contactReward:
            self.rewardsData.addReward(label="contact", rmin=0, rmax=1.0, rval=0, rweight=self.contactRewardWeight)

        if self.COMHeightReward:
            self.rewardsData.addReward(label="COM height", rmin=-1.0, rmax=0, rval=0, rweight=self.COMHeightRewardWeight)

        if self.aliveBonusReward:
            self.rewardsData.addReward(label="alive", rmin=0, rmax=1.0, rval=0, rweight=self.aliveBonusRewardWeight)


    def _getFile(self):
        return __file__

    def updateBeforeSimulation(self):
        #any pre-sim updates should happen here
        #update the stability polygon
        points = [
            self.robot_skeleton.bodynodes[17].to_world(np.array([-0.025, 0, 0.03])),  # l-foot_l-heel
            self.robot_skeleton.bodynodes[17].to_world(np.array([0.025, 0, 0.03])),  # l-foot_r-heel
            self.robot_skeleton.bodynodes[17].to_world(np.array([0, 0, -0.15])),  # l-foot_toe
            self.robot_skeleton.bodynodes[20].to_world(np.array([-0.025, 0, 0.03])),  # r-foot_l-heel
            self.robot_skeleton.bodynodes[20].to_world(np.array([0.025, 0, 0.03])),  # r-foot_r-heel
            self.robot_skeleton.bodynodes[20].to_world(np.array([0, 0, -0.15])),  # r-foot_toe
        ]
        points2D = []
        for point in points:
            points2D.append(np.array([point[0], point[2]]))

        hull = pyutils.convexHull2D(points2D)
        self.stabilityPolygon = []
        for point in hull:
            self.stabilityPolygon.append(np.array([point[0], -1.3, point[1]]))

        self.stabilityPolygonCentroid = pyutils.getCentroid(self.stabilityPolygon)

        self.projectedCOM = np.array(self.robot_skeleton.com())
        self.COMHeight = self.projectedCOM[1]
        self.projectedCOM[1] = -1.3

        #test COM containment
        self.stableCOM = pyutils.polygon2DContains(hull, np.array([self.projectedCOM[0], self.projectedCOM[2]]))
        #print("containedCOM: " + str(containedCOM))

        #analyze contacts
        self.lFootContact = False
        self.rFootContact = False
        self.numFootContacts = 0
        self.nonFootContact = False
        if self.dart_world is not None:
            if self.dart_world.collision_result is not None:
                for contact in self.dart_world.collision_result.contacts:
                    if contact.skel_id1 == 0:
                        if contact.bodynode_id2 == 17:
                            self.numFootContacts += 1
                            self.lFootContact = True
                        elif contact.bodynode_id2 == 20:
                            self.numFootContacts += 1
                            self.rFootContact = True
                        else:
                            self.nonFootContact = True
                    if contact.skel_id2 == 0:
                        if contact.bodynode_id2 == 17:
                            self.numFootContacts += 1
                            self.lFootContact = True
                        elif contact.bodynode_id2 == 20:
                            self.numFootContacts += 1
                            self.rFootContact = True
                        else:
                            self.nonFootContact = True
        a=0

    def checkTermination(self, tau, s, obs):
        #check the termination conditions and return: done,reward
        if not np.isfinite(s).all():
            print(self.rewardTrajectory)
            print(self.stateTraj[-2:])
            print("Infinite value detected..." + str(s))
            return True, -1500
        elif np.amax(np.absolute(s[:len(self.robot_skeleton.q)])) > 10:
            print("Detecting potential instability")
            print(s)
            print(self.rewardTrajectory)
            print(self.stateTraj[-2:])
            return True, -1500

        #stability termination
        if(not self.stableCOM):
            return True, -1500

        #contact termination
        if(self.nonFootContact):
            return True, -1500
        return False, 0

    def computeReward(self, tau):
        #compute and return reward at the current state
        self.prevTau = tau

        reward_record = []

        # reward rest pose standing
        reward_restPose = 0
        if self.restPoseReward and self.restPose is not None:
            dist = np.linalg.norm(self.robot_skeleton.q - self.restPose)
            reward_restPose = max(-51, -dist)
            reward_record.append(reward_restPose)

        #reward COM over stability region
        reward_stability = 0
        if self.stabilityCOMReward:
            #penalty for distance from projected COM to stability centroid
            reward_stability = -np.linalg.norm(self.stabilityPolygonCentroid - self.projectedCOM)
            reward_record.append(reward_stability)

        #reward # ground contact points
        reward_contact = 0
        if self.contactReward:
            reward_contact = self.numFootContacts/6.0 #maximum of 6 ground contact points with 3 spheres per foot
            reward_record.append(reward_contact)

        #reward COM height?
        reward_COMHeight = 0
        if self.COMHeightReward:
            reward_COMHeight = self.COMHeight
            reward_record.append(reward_COMHeight)

        reward_alive = 0
        if self.aliveBonusReward:
            reward_alive = 1.0
            reward_record.append(reward_alive)

        # update the reward data storage
        self.rewardsData.update(rewards=reward_record)

        self.reward = reward_contact * self.contactRewardWeight \
                    + reward_stability * self.stabilityCOMRewardWeight \
                    + reward_restPose * self.restPoseRewardWeight \
                    + reward_COMHeight * self.COMHeightRewardWeight \
                    + reward_alive * self.aliveBonusRewardWeight

        return self.reward

    def _get_obs(self):
        obs = np.zeros(self.obs_size)

        theta = np.array(self.robot_skeleton.q[3:])
        dq = np.array(self.robot_skeleton.dq)
        trans = np.array(self.robot_skeleton.q[:3])

        obs = np.concatenate([trans, np.cos(theta), np.sin(theta), dq], axis=1).ravel()

        #COM
        com = np.array(self.robot_skeleton.com()).ravel()
        obs = np.concatenate([obs, com]).ravel()

        #foot contacts
        if self.lFootContact:
            obs = np.concatenate([obs, [1.0]]).ravel()
        else:
            obs = np.concatenate([obs, [0.0]]).ravel()

        if self.rFootContact:
            obs = np.concatenate([obs, [1.0]]).ravel()
        else:
            obs = np.concatenate([obs, [0.0]]).ravel()

        #print(obs)

        return obs

    def additionalResets(self):
        #do any additional resetting here
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-0.01, high=0.01, size=self.robot_skeleton.ndofs)
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        self.restPose = qpos

        if self.simulateCloth:
            self.clothScene.translateCloth(0, np.array([0, 3.0, 0]))
        a=0

    def extraRenderFunction(self):
        renderUtils.setColor(color=[0.0, 0.0, 0])
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3d(0,0,0)
        GL.glVertex3d(-1,0,0)
        GL.glEnd()

        renderUtils.setColor([0,0,0])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[4].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[4].to_world(np.array([0.0,-0.3,-0.075]))])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[9].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[9].to_world(np.array([0.0,-0.3,-0.075]))])

        #compute the zero moment point

        #render the ideal stability polygon
        if len(self.stabilityPolygon) > 0:
            renderUtils.drawPolygon(self.stabilityPolygon)
        renderUtils.setColor([0.0,0,1.0])
        renderUtils.drawSphere(pos=self.projectedCOM)

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
                renderUtils.renderDofs(robot=self.robot_skeleton, restPose=None, renderRestPose=False)
