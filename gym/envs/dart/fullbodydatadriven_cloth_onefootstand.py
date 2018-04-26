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

class DartClothFullBodyDataDrivenClothOneFootStandEnv(DartClothFullBodyDataDrivenClothBaseEnv, utils.EzPickle):
    def __init__(self):
        #feature flags
        rendering = False
        clothSimulation = False
        renderCloth = True

        #reward flags
        self.restPoseReward             = True
        self.stabilityCOMReward         = True
        self.contactReward              = False
        self.flatFootReward             = True  # if true, reward the foot for being parallel to the ground
        self.COMHeightReward            = True
        self.aliveBonusReward           = True #rewards rollout duration to counter suicidal tendencies
        self.stationaryAnkleAngleReward = True #penalizes ankle joint velocity
        self.stationaryAnklePosReward   = True #penalizes planar motion of projected ankle point

        #reward weights
        self.restPoseRewardWeight               = 1
        self.stabilityCOMRewardWeight           = 5
        self.contactRewardWeight                = 1
        self.flatFootRewardWeight               = 4
        self.COMHeightRewardWeight              = 2
        self.aliveBonusRewardWeight             = 5
        self.stationaryAnkleAngleRewardWeight   = 0.025
        self.stationaryAnklePosRewardWeight     = 2

        #other flags
        self.stabilityTermination = True #if COM outside stability region, terminate #TODO: timed?
        self.contactTermination   = True #if anything except the feet touch the ground, terminate
        self.gravity = True

        #other variables
        self.prevTau = None
        self.restPose = None
        self.stabilityPolygon = [] #an ordered point set representing the stability region of the desired foot contacts
        self.stabilityPolygonCentroid = np.zeros(3)
        self.projectedCOM = np.zeros(3)
        self.COMHeight = 0.0
        self.stableCOM = True
        self.numFootContacts = 0
        self.footContact = False
        self.nonFootContact = False #used for detection of failure
        self.initialProjectedAnkle = np.zeros(3)
        self.footCOP = np.zeros(3)
        self.footNormForceMag = 0
        self.footBodyNode = 17 #17 left, 20 right
        self.ankleDofs = [32,33] #[32,33] left, [38,39] right

        self.actuatedDofs = np.arange(34)
        observation_size = 0
        observation_size = 37 * 3 + 6 #q[:3], q[3:](sin,cos), dq
        observation_size += 3 # COM
        observation_size += 1 # binary contact per foot with ground
        observation_size += 4 # feet COPs and norm force mags



        DartClothFullBodyDataDrivenClothBaseEnv.__init__(self,
                                                          rendering=rendering,
                                                          screensize=(1080,920),
                                                          clothMeshFile="capri_med.obj",
                                                          clothScale=np.array([1.0,1.0,1.0]),
                                                          obs_size=observation_size,
                                                          simulateCloth=clothSimulation,
                                                          gravity=self.gravity)


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

        if self.flatFootReward:
            self.rewardsData.addReward(label="flat foot", rmin=-1.0, rmax=0, rval=0, rweight=self.flatFootRewardWeight)

        if self.COMHeightReward:
            self.rewardsData.addReward(label="COM height", rmin=-1.0, rmax=0, rval=0, rweight=self.COMHeightRewardWeight)

        if self.aliveBonusReward:
            self.rewardsData.addReward(label="alive", rmin=0, rmax=1.0, rval=0, rweight=self.aliveBonusRewardWeight)

        if self.stationaryAnkleAngleReward:
            self.rewardsData.addReward(label="ankle angle", rmin=-40.0, rmax=0.0, rval=0, rweight=self.stationaryAnkleAngleRewardWeight)

        if self.stationaryAnklePosReward:
            self.rewardsData.addReward(label="ankle pos", rmin=-0.5, rmax=0.0, rval=0, rweight=self.stationaryAnklePosRewardWeight)


    def _getFile(self):
        return __file__

    def updateBeforeSimulation(self):
        #any pre-sim updates should happen here
        #update the stability polygon
        points = [
            self.robot_skeleton.bodynodes[self.footBodyNode].to_world(np.array([-0.035, 0, 0.03])),  # l-foot_l-heel
            self.robot_skeleton.bodynodes[self.footBodyNode].to_world(np.array([0.035, 0, 0.03])),  # l-foot_r-heel
            self.robot_skeleton.bodynodes[self.footBodyNode].to_world(np.array([0, 0, -0.15])),  # l-foot_toe
        ]

        self.stabilityPolygon = []
        hull = []
        for point in points:
            self.stabilityPolygon.append(np.array([point[0], -1.3, point[2]]))
            hull.append(np.array([point[0], point[2]]))

        self.stabilityPolygonCentroid = pyutils.getCentroid(self.stabilityPolygon)

        self.projectedCOM = np.array(self.robot_skeleton.com())
        self.COMHeight = self.projectedCOM[1]
        self.projectedCOM[1] = -1.3

        #test COM containment
        self.stableCOM = pyutils.polygon2DContains(hull, np.array([self.projectedCOM[0], self.projectedCOM[2]]))
        #print("containedCOM: " + str(containedCOM))

        #analyze contacts
        self.footContact = False
        self.numFootContacts = 0
        self.nonFootContact = False
        self.footCOP = np.zeros(3)
        self.footNormForceMag = 0
        if self.dart_world is not None:
            if self.dart_world.collision_result is not None:
                for contact in self.dart_world.collision_result.contacts:
                    if contact.skel_id1 == 0:
                        if contact.bodynode_id2 == self.footBodyNode:
                            self.numFootContacts += 1
                            self.footContact = True
                            self.footCOP += contact.p*abs(contact.f[1])
                            self.footNormForceMag += abs(contact.f[1])
                        else:
                            self.nonFootContact = True
                    if contact.skel_id2 == 0:
                        if contact.bodynode_id2 == self.footBodyNode:
                            self.numFootContacts += 1
                            self.lFootContact = True
                            self.footCOP += contact.p * abs(contact.f[1])
                            self.footNormForceMag += abs(contact.f[1])
                        else:
                            self.nonFootContact = True

        if self.footNormForceMag > 0:
            self.footCOP /= self.footNormForceMag

        a=0

    def checkTermination(self, tau, s, obs):
        #check the termination conditions and return: done,reward
        if not np.isfinite(s).all():
            #print(self.rewardTrajectory)
            #print(self.stateTraj[-2:])
            print("Infinite value detected..." + str(s))
            return True, -1500
        elif np.amax(np.absolute(s[:len(self.robot_skeleton.q)])) > 10:
            print("Detecting potential instability")
            print(s)
            print(self.rewardTrajectory)
            #print(self.stateTraj[-2:])
            return True, -1500
        elif np.amax(np.absolute(s[:len(self.robot_skeleton.q)]-self.stateTraj[-1])) > 1.0:
            print("Detecting potential instability via velocity: " + str(np.amax(np.absolute(s[:len(self.robot_skeleton.q)]-self.stateTraj[-1]))))
            print(s)
            print(self.stateTraj[-2:])
            print(self.rewardTrajectory)
            return True, -1500


        #print(np.amax(np.absolute(s[:len(self.robot_skeleton.q)]-self.stateTraj[-1])))

        #stability termination
        if self.stabilityTermination:
            if not self.stableCOM:
                return True, -1500

        #contact termination
        if self.contactTermination:
            if self.nonFootContact:
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
            reward_contact = self.numFootContacts/3.0 #maximum of 3 ground contact points with 3 spheres per foot
            reward_record.append(reward_contact)

        reward_flatFoot = 0
        if self.flatFootReward:
            up = np.array([0, 1.0, 0])
            footNorm = self.robot_skeleton.bodynodes[self.footBodyNode].to_world(up) - self.robot_skeleton.bodynodes[self.footBodyNode].to_world(np.zeros(3))
            footNorm = footNorm / np.linalg.norm(footNorm)

            reward_flatFoot += footNorm.dot(up) - 1.0

            reward_record.append(reward_flatFoot)

        #reward COM height?
        reward_COMHeight = 0
        if self.COMHeightReward:
            reward_COMHeight = self.COMHeight
            if(abs(self.COMHeight) > 1.0):
                reward_COMHeight = 0
            reward_record.append(reward_COMHeight)

        #accumulate reward for continuing to balance
        reward_alive = 0
        if self.aliveBonusReward:
            reward_alive = 1.0
            reward_record.append(reward_alive)

        #reward stationary ankle
        reward_stationaryAnkleAngle = 0
        if self.stationaryAnkleAngleReward:
            reward_stationaryAnkleAngle += -abs(self.robot_skeleton.dq[self.ankleDofs[0]])
            reward_stationaryAnkleAngle += -abs(self.robot_skeleton.dq[self.ankleDofs[1]])
            reward_stationaryAnkleAngle = -max(-40, reward_stationaryAnkleAngle)
            reward_record.append(reward_stationaryAnkleAngle)

        reward_stationaryAnklePos = 0
        if self.stationaryAnklePosReward:
            projectedAnkle = self.robot_skeleton.bodynodes[self.footBodyNode].to_world(np.zeros(3))
            projectedAnkle[1] = 0
            reward_stationaryAnklePos += max(-0.5, -np.linalg.norm(self.initialProjectedAnkle - projectedAnkle))
            reward_record.append(reward_stationaryAnklePos)

        # update the reward data storage
        self.rewardsData.update(rewards=reward_record)

        self.reward = reward_contact * self.contactRewardWeight \
                    + reward_stability * self.stabilityCOMRewardWeight \
                    + reward_restPose * self.restPoseRewardWeight \
                    + reward_COMHeight * self.COMHeightRewardWeight \
                    + reward_alive * self.aliveBonusRewardWeight \
                    + reward_stationaryAnkleAngle * self.stationaryAnkleAngleRewardWeight \
                    + reward_stationaryAnklePos * self.stationaryAnklePosRewardWeight \
                    + reward_flatFoot * self.flatFootRewardWeight

        return self.reward

    def _get_obs(self):
        obs = np.zeros(self.obs_size)

        orientation = np.array(self.robot_skeleton.q[:3])
        theta = np.array(self.robot_skeleton.q[6:])
        dq = np.array(self.robot_skeleton.dq)
        trans = np.array(self.robot_skeleton.q[3:6])

        obs = np.concatenate([np.cos(orientation), np.sin(orientation), trans, np.cos(theta), np.sin(theta), dq]).ravel()

        #COM
        com = np.array(self.robot_skeleton.com()).ravel()
        obs = np.concatenate([obs, com]).ravel()

        #foot contacts
        if self.footContact:
            obs = np.concatenate([obs, [1.0]]).ravel()
        else:
            obs = np.concatenate([obs, [0.0]]).ravel()

        #foot COP and norm force magnitude
        obs = np.concatenate([obs, self.footCOP, [self.footNormForceMag]]).ravel()

        #print(obs)

        return obs

    def additionalResets(self):
        #do any additional resetting here
        #TODO: set a one foot standing initial pose
        qpos = np.array([-0.00469234655801, -0.0218378114573, -0.011132330496, 0.00809830385355, 0.00051861417993, 0.0584867818269, 0.374712375814, 0.0522417260384, -0.00777676124956, 0.00230285789432, -0.00274958108859, -0.008064630425, 0.00247294825781, -0.0093978116532, 0.195632645271, -0.00276696945071, 0.0075491687512, -0.0116846422966, 0.00636619242284, 0.00767084047346, -0.00913509000374, 0.00857521738396, 0.199096855493, 0.00787726246678, -0.00760402683795, -0.00433642327146, 0.00802311463366, -0.00482248656677, 0.131248337324, -0.00662274635457, 0.00333416764933, 0.00546016678096, -0.00150775759695, -0.00861184703697, -0.000589790168521, -0.832681560131, 0.00976653127827, 2.24259637323, -0.00374506255585, -0.00244949106062])


        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-0.01, high=0.01, size=self.robot_skeleton.ndofs)
        #qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qpos = qpos + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        self.restPose = qpos

        if self.simulateCloth:
            self.clothScene.translateCloth(0, np.array([0, 3.0, 0]))

        #set on initialization and used to measure displacement
        self.initialProjectedAnkle = self.robot_skeleton.bodynodes[self.footBodyNode].to_world(np.zeros(3))
        self.initialProjectedAnkle[1] = 0
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

        #render center of pressure for each foot
        COPLines = [
            [self.footCOP, self.footCOP+np.array([0,self.footNormForceMag,0])]
                    ]

        renderUtils.setColor(color=[0.0,1.0,0])
        renderUtils.drawLines(COPLines)
        renderUtils.setColor(color=[0.0, 0.0, 1.0])
        renderUtils.drawLines([[self.robot_skeleton.com(), np.array([self.robot_skeleton.com()[0], -2.0, self.robot_skeleton.com()[2]])]])


        #compute the zero moment point
        if False:
            m = self.robot_skeleton.mass()
            g = self.gravity
            z = np.array([0,1.0,0]) #ground normal (must be opposite gravity)
            O = self.robot_skeleton.bodynodes[20].to_world(np.zeros(3)) #ankle
            O[1] = -1.12 #projection of ankle to ground elevation
            G = self.robot_skeleton.com()
            Hd = np.zeros(3)#angular momentum: sum of angular momentum of all bodies about O
            #angular momentum of a body: R(I wd - ( (I w) x w ) )
            #  where I is Intertia matrix, R is rotation matrix, w rotation rate, wd angular acceleration
            for node in self.robot_skeleton.bodynodes:
                I = node.I()
                R = node.T() #TODO: may need to extract R from this?
                #w = node. #angular velocity
                #wd =  #angular acceleration
                #TODO: combine

            #TODO: continue


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
