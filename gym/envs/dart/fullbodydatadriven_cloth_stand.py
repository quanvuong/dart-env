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
        renderCloth = False
        gravity = True
        SPDActionSpace = True
        frameskip = 15
        dt = 0.002

        #reward flags
        self.restPoseReward             = True
        self.stabilityCOMReward         = True
        self.stabilityZMPReward         = False
        self.contactReward              = False
        self.flatFootReward             = False #if true, reward the foot for being parallel to the ground
        self.COMHeightReward            = False
        self.aliveBonusReward           = True #rewards rollout duration to counter suicidal tendencies
        self.stationaryAnkleAngleReward = False #penalizes ankle joint velocity
        self.stationaryAnklePosReward   = False #penalizes planar motion of projected ankle point

        #reward weights
        self.restPoseRewardWeight               = 1
        self.stabilityCOMRewardWeight           = 2
        self.stabilityZMPRewardWeight           = 2
        self.contactRewardWeight                = 1
        self.flatFootRewardWeight               = 4
        self.COMHeightRewardWeight              = 2
        self.aliveBonusRewardWeight             = 10
        self.stationaryAnkleAngleRewardWeight   = 0.025
        self.stationaryAnklePosRewardWeight     = 2

        #other flags
        self.stabilityTermination = False #if COM outside stability region, terminate #TODO: timed?
        self.contactTermiantion   = True #if anything except the feet touch the ground, terminate
        self.COMHeightTermination = True #terminate if COM drops below a certain height

        self.COMMinHeight         = -0.6

        #other variables
        self.prevTau = None
        self.restPose = None
        self.stabilityPolygon = [] #an ordered point set representing the stability region of the desired foot contacts
        self.stabilityPolygonCentroid = np.zeros(3)
        self.projectedCOM = np.zeros(3)
        self.ZMP = np.zeros(2)
        self.COMHeight = 0.0
        self.stableCOM = True
        self.numFootContacts = 0
        self.lFootContact = False
        self.rFootContact = False
        self.nonFootContact = False #used for detection of failure
        self.nonFootContactNode = None
        self.initialProjectedRAnkle = np.zeros(3)
        self.initialProjectedLAnkle = np.zeros(3)
        self.leftCOP = np.zeros(3)
        self.leftNormForceMag = 0
        self.rightCOP = np.zeros(3)
        self.rightNormForceMag = 0

        self.actuatedDofs = np.arange(34)
        observation_size = 0
        observation_size = 34 * 3 + 6 #q[6:](sin,cos), dq
        observation_size += 3 # COM (root local 3D)
        observation_size += 1 # COM height
        observation_size += 3 # ZMP (3D root local)
        observation_size += 3 # stability polygon centroid (3D root local)
        observation_size += 2 # binary contact per foot with ground
        observation_size += 8 # feet COPs (root local) and norm force mags



        DartClothFullBodyDataDrivenClothBaseEnv.__init__(self,
                                                          rendering=rendering,
                                                          screensize=(1280,920),
                                                          clothMeshFile="capri_med.obj",
                                                          clothScale=np.array([1.0,1.0,1.0]),
                                                          obs_size=observation_size,
                                                          simulateCloth=clothSimulation,
                                                          gravity=gravity,
                                                          frameskip=frameskip,
                                                          dt=dt,
                                                          SPDActionSpace=SPDActionSpace)


        self.simulateCloth = clothSimulation

        if not renderCloth:
            self.clothScene.renderClothFill = False
            self.clothScene.renderClothBoundary = False
            self.clothScene.renderClothWires = False

        if self.restPoseReward:
            if self.SPDActionSpace:
                self.rewardsData.addReward(label="restPose(SPD)", rmin=-100.0, rmax=0, rval=0, rweight=self.restPoseRewardWeight)
            else:
                self.rewardsData.addReward(label="restPose", rmin=-51.0, rmax=0, rval=0, rweight=self.restPoseRewardWeight)

        if self.stabilityCOMReward:
            self.rewardsData.addReward(label="COM stability", rmin=-0.5, rmax=0, rval=0, rweight=self.stabilityCOMRewardWeight)

        if self.stabilityZMPReward:
            self.rewardsData.addReward(label="ZMP stability", rmin=-1.0, rmax=0, rval=0, rweight=self.stabilityZMPRewardWeight)

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

        self.ZMP = self.computeZMP()

        #test COM containment
        self.stableCOM = pyutils.polygon2DContains(hull, np.array([self.projectedCOM[0], self.projectedCOM[2]]))
        #print("containedCOM: " + str(containedCOM))

        #analyze contacts
        self.nonFootContactNode = []
        self.lFootContact = False
        self.rFootContact = False
        self.numFootContacts = 0
        self.nonFootContact = False
        self.leftCOP = np.zeros(3)
        self.leftNormForceMag = 0
        self.rightCOP = np.zeros(3)
        self.rightNormForceMag = 0
        if self.dart_world is not None:
            if self.dart_world.collision_result is not None:
                for contact in self.dart_world.collision_result.contacts:
                    if contact.skel_id1 == 0:
                        if contact.bodynode_id2 == 17:
                            self.numFootContacts += 1
                            self.lFootContact = True
                            self.leftCOP += contact.p*abs(contact.f[1])
                            self.leftNormForceMag += abs(contact.f[1])
                        elif contact.bodynode_id2 == 20:
                            self.numFootContacts += 1
                            self.rFootContact = True
                            self.rightCOP += contact.p * abs(contact.f[1])
                            self.rightNormForceMag += abs(contact.f[1])
                        else:
                            self.nonFootContact = True
                            self.nonFootContactNode.append(contact.bodynode_id2)
                    if contact.skel_id2 == 0:
                        if contact.bodynode_id1 == 17:
                            self.numFootContacts += 1
                            self.lFootContact = True
                            self.leftCOP += contact.p * abs(contact.f[1])
                            self.leftNormForceMag += abs(contact.f[1])
                        elif contact.bodynode_id1 == 20:
                            self.numFootContacts += 1
                            self.rFootContact = True
                            self.rightCOP += contact.p * abs(contact.f[1])
                            self.rightNormForceMag += abs(contact.f[1])
                        else:
                            self.nonFootContact = True
                            self.nonFootContactNode.append(contact.bodynode_id1)

        if self.leftNormForceMag > 0:
            self.leftCOP /= self.leftNormForceMag

        if self.rightNormForceMag > 0:
            self.rightCOP /= self.rightNormForceMag

        a=0

    def checkTermination(self, tau, s, obs):
        #check the termination conditions and return: done,reward
        if not np.isfinite(s).all():
            #print(self.rewardTrajectory)
            #print(self.stateTraj[-2:])
            print("Infinite value detected..." + str(s))
            return True, -0
        elif np.amax(np.absolute(s[:len(self.robot_skeleton.q)])) > 10:
            print("Detecting potential instability")
            print(s)
            print(self.rewardTrajectory)
            #print(self.stateTraj[-2:])
            return True, -0
        '''elif np.amax(np.absolute(s[:len(self.robot_skeleton.q)]-self.stateTraj[-1])) > 1.0:
            print("Detecting potential instability via velocity: " + str(np.amax(np.absolute(s[:len(self.robot_skeleton.q)]-self.stateTraj[-1]))))
            print(s)
            print(self.stateTraj[-2:])
            print(self.rewardTrajectory)
            return True, -1500'''


        #print(np.amax(np.absolute(s[:len(self.robot_skeleton.q)]-self.stateTraj[-1])))

        #stability termination
        if self.stabilityTermination:
            if(not self.stableCOM):
                return True, -0

        #contact termination
        if self.contactTermiantion:
            if(self.nonFootContact):
                #print("non Foot Contact with node: " + str(self.nonFootContactNode))
                return True, -0

        if self.COMHeightTermination:
            #print(self.robot_skeleton.com()[1])
            if self.robot_skeleton.com()[1] < self.COMMinHeight:
                return True, -0

        return False, 0

    def computeReward(self, tau):
        #compute and return reward at the current state
        self.prevTau = tau

        reward_record = []

        # reward rest pose standing
        reward_restPose = 0
        if self.restPoseReward and self.restPose is not None:
            dist = np.linalg.norm(self.robot_skeleton.q[6:] - self.restPose[6:])
            reward_restPose = max(-51, -dist)

            if self.SPDActionSpace:
                #also reward the policy for output similar to restPose
                taudist = np.linalg.norm(tau[6:] - self.restPose[6:])
                reward_restPose += max(-51, -taudist)

            reward_record.append(reward_restPose)

        #reward COM over stability region
        reward_COMstability = 0
        if self.stabilityCOMReward:
            #penalty for distance from projected COM to stability centroid
            reward_COMstability = -np.linalg.norm(self.stabilityPolygonCentroid - self.projectedCOM)
            reward_record.append(reward_COMstability)

        # reward COM over stability region
        reward_ZMPstability = 0
        if self.stabilityZMPReward:
            # penalty for distance from projected COM to stability centroid
            ZMP3D = np.array([self.ZMP[0], self.stabilityPolygonCentroid[1], self.ZMP[1]])
            reward_ZMPstability = -np.linalg.norm(self.stabilityPolygonCentroid - ZMP3D)
            reward_ZMPstability = max(reward_ZMPstability, -1.0) #clamp to  distance of 1
            reward_record.append(reward_ZMPstability)
            #print(reward_ZMPstability)

        #reward # ground contact points
        reward_contact = 0
        if self.contactReward:
            reward_contact = self.numFootContacts/6.0 #maximum of 6 ground contact points with 3 spheres per foot
            reward_record.append(reward_contact)

        reward_flatFoot = 0
        if self.flatFootReward:
            up = np.array([0, 1.0, 0])
            lFootNorm = self.robot_skeleton.bodynodes[17].to_world(up) - self.robot_skeleton.bodynodes[17].to_world(np.zeros(3))
            rFootNorm = self.robot_skeleton.bodynodes[20].to_world(up) - self.robot_skeleton.bodynodes[20].to_world(np.zeros(3))
            lFootNorm = lFootNorm / np.linalg.norm(lFootNorm)
            rFootNorm = rFootNorm / np.linalg.norm(rFootNorm)

            reward_flatFoot += lFootNorm.dot(up) - 1.0
            reward_flatFoot += rFootNorm.dot(up) - 1.0

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
            reward_stationaryAnkleAngle += -abs(self.robot_skeleton.dq[38])
            reward_stationaryAnkleAngle += -abs(self.robot_skeleton.dq[39])
            reward_stationaryAnkleAngle += -abs(self.robot_skeleton.dq[32])
            reward_stationaryAnkleAngle += -abs(self.robot_skeleton.dq[33])
            reward_stationaryAnkleAngle = max(-40, reward_stationaryAnkleAngle)
            reward_record.append(reward_stationaryAnkleAngle)

        reward_stationaryAnklePos = 0
        if self.stationaryAnklePosReward:
            projectedRAnkle = self.robot_skeleton.bodynodes[20].to_world(np.zeros(3))
            projectedLAnkle = self.robot_skeleton.bodynodes[17].to_world(np.zeros(3))
            projectedRAnkle[1] = 0
            projectedLAnkle[1] = 0
            reward_stationaryAnklePos += max(-0.5, -np.linalg.norm(self.initialProjectedRAnkle - projectedRAnkle))
            reward_stationaryAnklePos += max(-0.5, -np.linalg.norm(self.initialProjectedLAnkle - projectedLAnkle))
            reward_record.append(reward_stationaryAnklePos)

        # update the reward data storage
        self.rewardsData.update(rewards=reward_record)

        self.reward = reward_contact * self.contactRewardWeight \
                    + reward_COMstability * self.stabilityCOMRewardWeight \
                    + reward_ZMPstability * self.stabilityZMPRewardWeight \
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

        obs = np.concatenate([np.cos(theta), np.sin(theta), dq]).ravel()

        #COM
        com = np.array(self.robot_skeleton.com()).ravel()
        local_com = self.robot_skeleton.bodynodes[0].to_local(com)
        obs = np.concatenate([obs, local_com, [com[1]]]).ravel()

        #ZMP
        local_zmp = self.robot_skeleton.bodynodes[0].to_local(np.array([self.ZMP[0], -1.42, self.ZMP[1]]))
        obs = np.concatenate([obs, local_zmp]).ravel()

        #stability polygon centroid
        stabilityPolygonCentroid2D = np.array([self.stabilityPolygonCentroid[0], self.stabilityPolygonCentroid[2]])
        local_stabilityPolygonCentroid = self.robot_skeleton.bodynodes[0].to_local(np.array([self.stabilityPolygonCentroid[0], -1.42, self.stabilityPolygonCentroid[2]]))
        obs = np.concatenate([obs, local_stabilityPolygonCentroid]).ravel()

        #foot contacts
        if self.lFootContact:
            obs = np.concatenate([obs, [1.0]]).ravel()
        else:
            obs = np.concatenate([obs, [0.0]]).ravel()

        if self.rFootContact:
            obs = np.concatenate([obs, [1.0]]).ravel()
        else:
            obs = np.concatenate([obs, [0.0]]).ravel()

        #foot COP and norm force magnitude
        localLeftCOP = self.robot_skeleton.bodynodes[0].to_local(self.leftCOP)
        localRightCOP = self.robot_skeleton.bodynodes[0].to_local(self.rightCOP)

        obs = np.concatenate([obs, localLeftCOP, [self.leftNormForceMag], localRightCOP, [self.rightNormForceMag]]).ravel()

        #print(obs)

        if not np.isfinite(obs).all():
            print("non-finite observation")
            print(obs)

        return obs

    def additionalResets(self):
        #do any additional resetting here
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-0.01, high=0.01, size=self.robot_skeleton.ndofs)
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        self.restPose = qpos

        if self.simulateCloth:
            self.clothScene.translateCloth(0, np.array([0, 3.0, 0]))

        #set on initialization and used to measure displacement
        self.initialProjectedRAnkle = self.robot_skeleton.bodynodes[20].to_world(np.zeros(3))
        self.initialProjectedLAnkle = self.robot_skeleton.bodynodes[17].to_world(np.zeros(3))
        self.initialProjectedRAnkle[1] = 0
        self.initialProjectedLAnkle[1] = 0

        self.updateBeforeSimulation()
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
            [self.leftCOP, self.leftCOP+np.array([0,self.leftNormForceMag,0])],
            [self.rightCOP, self.rightCOP+np.array([0,self.rightNormForceMag,0])],
                    ]

        renderUtils.setColor(color=[1.0,1.0,0])
        renderUtils.drawLines(COPLines)


        #render the zero moment point
        if True:
            ZMP = self.computeZMP()
            renderUtils.setColor(color=[0, 1.0, 0.0])
            renderUtils.drawLines([[np.array([ZMP[0], 1.0, ZMP[1]]), np.array([ZMP[0], -2.0, ZMP[1]])]])
            renderUtils.drawSphere(pos=np.array([ZMP[0], self.projectedCOM[1], ZMP[1]]))

        #render the ideal stability polygon
        if len(self.stabilityPolygon) > 0:
            renderUtils.drawPolygon(self.stabilityPolygon)
        renderUtils.setColor([0.0,0,1.0])
        renderUtils.drawSphere(pos=self.projectedCOM)
        renderUtils.drawLines([[np.array([self.projectedCOM[0], 1.0, self.projectedCOM[2]]), np.array([self.projectedCOM[0], -2.0, self.projectedCOM[2]])]])

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
            self.clothScene.drawText(x=15., y=textLines*textHeight, text="Time = " + str(self.numSteps*self.frame_skip*self.dart_world.time_step()), color=(0., 0, 0))
            textLines += 1
            self.clothScene.drawText(x=15., y=textLines*textHeight, text="Reward = " + str(self.reward), color=(0., 0, 0))
            textLines += 1
            self.clothScene.drawText(x=15., y=textLines * textHeight, text="Cumulative Reward = " + str(self.cumulativeReward), color=(0., 0, 0))
            textLines += 1

            if self.numSteps > 0:
                tauLim = np.concatenate([np.zeros(6), self.action_scale])
                renderUtils.renderDofs(robot=self.robot_skeleton, restPose=self.restPose, renderRestPose=True, tau=self.prevTau, tauLim=tauLim)

                #grapher test
                '''i = int(self.numSteps/100)
                tauGraph = pyutils.LineGrapher("Tau["+str(i)+"]")
                data = []
                for s in range(self.numSteps):
                    data.append(self.actionTrajectory[s][i])
                tauGraph.plotData(ydata=data,label=str(i))'''
                #self.actionGraphFoci = [10]
                #tauGraph.update()
                #tauGraph.close()