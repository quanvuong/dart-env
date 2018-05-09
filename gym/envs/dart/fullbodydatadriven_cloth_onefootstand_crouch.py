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

class DartClothFullBodyDataDrivenClothOneFootStandCrouchEnv(DartClothFullBodyDataDrivenClothBaseEnv, utils.EzPickle):
    def __init__(self):
        #feature flags
        rendering = True
        clothSimulation = False
        renderCloth = False
        self.gravity = True
        self.rectFoot = False

        #reward flags
        self.restPoseReward             = True
        self.stabilityCOMReward         = True
        self.stabilityZMPReward         = True
        self.stabilityBonusReward       = True
        self.contactReward              = False
        self.flatFootReward             = False  # if true, reward the foot for being parallel to the ground
        self.COMHeightReward            = False
        self.aliveBonusReward           = True #rewards rollout duration to counter suicidal tendencies
        self.stationaryAnkleAngleReward = False #penalizes ankle joint velocity
        self.stationaryAnklePosReward   = False #penalizes planar motion of projected ankle point

        #reward weights
        self.restPoseRewardWeight               = 1
        self.stabilityCOMRewardWeight           = 7
        self.stabilityZMPRewardWeight           = 6
        self.stabilityBonusRewardWeight         = 4
        self.contactRewardWeight                = 1
        self.flatFootRewardWeight               = 4
        self.COMHeightRewardWeight              = 4
        self.aliveBonusRewardWeight             = 15
        self.stationaryAnkleAngleRewardWeight   = 0.025
        self.stationaryAnklePosRewardWeight     = 2

        #other flags
        self.stabilityTermination = True #if COM outside stability region, terminate #TODO: timed?
        self.maxUnstableDistance = 0.075
        self.contactTermination   = True #if anything except the feet touch the ground, terminate

        #other variables
        self.prevTau = None
        self.restPose = None
        self.stabilityPolygon = [] #an ordered point set representing the stability region of the desired foot contacts
        self.stabilityPolygonCentroid = np.zeros(3)
        self.projectedCOM = np.zeros(3)
        self.COMHeight = 0.0
        self.stableCOM = True
        self.stableZMP = True
        self.signedStabilityCOMDist = 0
        self.signedStabilityZMPDist = 0
        self.ZMP = np.zeros(2)
        self.numFootContacts = 0
        self.footContact = False
        self.nonFootContact = False #used for detection of failure
        self.initialProjectedAnkle = np.zeros(3)
        self.footCOP = np.zeros(3)
        self.footNormForceMag = 0
        self.footBodyNode = 17 #17 left, 20 right
        self.ankleDofs = [32,33] #[32,33] left, [38,39] right


        self.signedDistanceInObs = True
        self.actuatedDofs = np.arange(34)
        observation_size = 0
        observation_size = 37 * 3 + 6 #q[:3], q[3:](sin,cos), dq
        observation_size += 3 # COM
        observation_size += 2  # ZMP (2D)
        observation_size += 2  # stability polygon centroid (2D)
        observation_size += 1 # binary contact per foot with ground
        observation_size += 4 # feet COPs and norm force mags
        if self.signedDistanceInObs:
            observation_size += 2 #signed COM and ZMP polygon distance

        skelfile=None
        if self.rectFoot:
            skelfile = "FullBodyCapsules_datadriven_rectfoot.skel"

        DartClothFullBodyDataDrivenClothBaseEnv.__init__(self,
                                                          rendering=rendering,
                                                          screensize=(1280,920),
                                                          clothMeshFile="capri_med.obj",
                                                          clothScale=np.array([1.0,1.0,1.0]),
                                                          obs_size=observation_size,
                                                          simulateCloth=clothSimulation,
                                                          gravity=self.gravity,
                                                          frameskip=10,
                                                          skelfile=skelfile)


        self.simulateCloth = clothSimulation

        if not renderCloth:
            self.clothScene.renderClothFill = False
            self.clothScene.renderClothBoundary = False
            self.clothScene.renderClothWires = False

        if self.restPoseReward:
            self.rewardsData.addReward(label="restPose", rmin=-51.0, rmax=0, rval=0, rweight=self.restPoseRewardWeight)

        if self.stabilityCOMReward:
            self.rewardsData.addReward(label="stability", rmin=-0.5, rmax=0, rval=0, rweight=self.stabilityCOMRewardWeight)

        if self.stabilityZMPReward:
            self.rewardsData.addReward(label="ZMP stability", rmin=-1.0, rmax=0, rval=0, rweight=self.stabilityZMPRewardWeight)

        if self.stabilityBonusReward:
            self.rewardsData.addReward(label="Bonus stability", rmin=0.0, rmax=1.0, rval=0, rweight=self.stabilityBonusRewardWeight)

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

        self.ZMP = self.computeZMP()

        #test COM containment
        self.stableCOM = pyutils.polygon2DContains(hull, np.array([self.projectedCOM[0], self.projectedCOM[2]]))
        self.stableZMP = pyutils.polygon2DContains(hull, np.array([self.ZMP[0], self.ZMP[1]]))
        #print("containedCOM: " + str(containedCOM))

        #signed triangle distances
        self.signedStabilityCOMDist, closePCOM = pyutils.distToTriangle(self.stabilityPolygon[0], self.stabilityPolygon[1], self.stabilityPolygon[2], self.projectedCOM, signed=True)
        self.signedStabilityZMPDist, closePZMP = pyutils.distToTriangle(self.stabilityPolygon[0], self.stabilityPolygon[1], self.stabilityPolygon[2], np.array([self.ZMP[0], self.stabilityPolygon[1][1], self.ZMP[1]]), signed=True)

        self.signedStabilityCOMDist = max(-1.0, min(1.0, self.signedStabilityCOMDist))
        self.signedStabilityZMPDist = max(-1.0, min(1.0, self.signedStabilityZMPDist))

        #print("COM stability: " + str(self.signedStabilityCOMDist))
        #print("ZMP stability: " + str(self.signedStabilityZMPDist))

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
            return True, -2000
        elif np.amax(np.absolute(s[:len(self.robot_skeleton.q)])) > 10:
            print("Detecting potential instability")
            print(s)
            print(self.rewardTrajectory)
            #print(self.stateTraj[-2:])
            return True, -2000
        elif np.amax(np.absolute(s[:len(self.robot_skeleton.q)]-self.stateTraj[-1])) > 1.0:
            print("Detecting potential instability via velocity: " + str(np.amax(np.absolute(s[:len(self.robot_skeleton.q)]-self.stateTraj[-1]))))
            print(s)
            print(self.stateTraj[-2:])
            print(self.rewardTrajectory)
            return True, -2000


        #print(np.amax(np.absolute(s[:len(self.robot_skeleton.q)]-self.stateTraj[-1])))

        #stability termination
        if self.stabilityTermination:
            if not self.stableCOM:
                dist2tri, closeP = pyutils.distToTriangle(self.stabilityPolygon[0], self.stabilityPolygon[1], self.stabilityPolygon[2], self.projectedCOM)
                #print(dist2tri)
                #if np.linalg.norm(self.stabilityPolygonCentroid - self.projectedCOM) > self.maxUnstableDistance:
                if dist2tri > self.maxUnstableDistance:
                    return True, -2000

        #contact termination
        if self.contactTermination:
            if self.nonFootContact:
                return True, -2000
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

        # reward COM over stability region
        reward_ZMPstability = 0
        if self.stabilityZMPReward:
            # penalty for distance from projected COM to stability centroid
            ZMP3D = np.array([self.ZMP[0], self.stabilityPolygonCentroid[1], self.ZMP[1]])
            reward_ZMPstability = -np.linalg.norm(self.stabilityPolygonCentroid - ZMP3D)
            reward_ZMPstability = max(reward_ZMPstability, -1.0)  # clamp to  distance of 1
            reward_record.append(reward_ZMPstability)
            # print(reward_ZMPstability)

        reward_stabilityBonus = 0
        if self.stabilityBonusReward:
            if self.stableCOM:
                reward_stabilityBonus += 0.5
            if self.stableZMP:
                reward_stabilityBonus += 0.5
            reward_record.append(reward_stabilityBonus)

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
            reward_stationaryAnkleAngle = max(-40, reward_stationaryAnkleAngle)
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
                    + reward_ZMPstability * self.stabilityZMPRewardWeight \
                    + reward_stabilityBonus * self.stabilityBonusRewardWeight \
                    + reward_restPose * self.restPoseRewardWeight \
                    + reward_COMHeight * self.COMHeightRewardWeight \
                    + reward_alive * self.aliveBonusRewardWeight \
                    + reward_stationaryAnkleAngle * self.stationaryAnkleAngleRewardWeight \
                    + reward_stationaryAnklePos * self.stationaryAnklePosRewardWeight \
                    + reward_flatFoot * self.flatFootRewardWeight

        #print(self.reward)

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

        # ZMP
        obs = np.concatenate([obs, self.ZMP]).ravel()

        # stability polygon centroid
        stabilityPolygonCentroid2D = np.array([self.stabilityPolygonCentroid[0], self.stabilityPolygonCentroid[2]])
        obs = np.concatenate([obs, stabilityPolygonCentroid2D]).ravel()

        #foot contacts
        if self.footContact:
            obs = np.concatenate([obs, [1.0]]).ravel()
        else:
            obs = np.concatenate([obs, [0.0]]).ravel()

        #foot COP and norm force magnitude
        obs = np.concatenate([obs, self.footCOP, [self.footNormForceMag]]).ravel()

        if self.signedDistanceInObs:
            obs = np.concatenate([obs, [self.signedStabilityCOMDist], [self.signedStabilityZMPDist]]).ravel()

        #print(obs)

        return obs

    def additionalResets(self):
        #do any additional resetting here
        #TODO: set a one foot standing initial pose
        #qpos = np.array([-0.00469234655801, -0.0218378114573, -0.011132330496, 0.00809830385355, 0.00051861417993, 0.0584867818269, 0.374712375814, 0.0522417260384, -0.00777676124956, 0.00230285789432, -0.00274958108859, -0.008064630425, 0.00247294825781, -0.0093978116532, 0.195632645271, -0.00276696945071, 0.0075491687512, -0.0116846422966, 0.00636619242284, 0.00767084047346, -0.00913509000374, 0.00857521738396, 0.199096855493, 0.00787726246678, -0.00760402683795, -0.00433642327146, 0.00802311463366, -0.00482248656677, 0.131248337324, -0.00662274635457, 0.00333416764933, 0.00546016678096, -0.00150775759695, -0.00861184703697, -0.000589790168521, -0.832681560131, 0.00976653127827, 2.24259637323, -0.00374506255585, -0.00244949106062])
        #qpos = np.array([0.0846660806322, -0.187712686011, 0.161260303884, 0.0641489627886, -0.0186528700497, 0.0474250147964, 0.020015958046, 0.16770604227, -0.0108410165928, 0.0535965979519, -0.0197669537195, -0.196563691654, -0.296550499725, 0.184719231518, 0.239352890642, 0.0203682695093, -0.00774988189866, -0.0319199328056, 0.0143662479045, 0.0210962017901, -0.00469342183995, 0.00455195202435, 0.195713986333, 0.0158739145927, -0.00663330152384, -0.0022286922927, 0.0146798728224, 0.00511109556815, -0.100090281179, -0.199426956515, 0.145284176732, 0.59524411564, 0.341386269429, 0.148021694136, 0.00156540000676, -0.839133710152, 0.017241857498, 2.23305534804, -0.00869403893085, 0.00486575140185])
        #qpos = np.array([0.0792540309714, -0.198038537538, 0.165982043711, 0.057678066664, -0.03, 0.0514905008947, 0.0153889940281, 0.172754267613, -0.0152665902114, 0.0447139591458, -0.0159152223541, -0.741653453661, -0.291857490409, 0.073, 0.247712958964, 0.0230298051369, -0.0129663713662, -0.0251081943623, 0.0184011650614, 0.0284706137625, -0.0143437953932, 0.00253595897386, 0.202764558055, 0.008180767185, 0.00144036976378, -0.00599927843092, 0.010826449318, 0.00831071336219, -0.0983439895237, -0.189360110846, 0.291156844437, 0.58830057445, 0.337019304264, 0.151085855622, 0.00418796434677, -0.841518739136, 0.0266268945701, 2.23410594707, -0.0133781980567, 0.00155774102539])
        qpos = np.array([0.0780131557357, -0.142593660368, 0.143019989259, 0.144666815206, -0.035, 0.165147835811, 0.0162260416596, 0.19115105848, -0.0299336428088, 0.0445035430603, -0.025419636699, -0.878286887463, -0.485843951506, 0.239911240107, 1.48781704099, 0.00147260210175, -3.84887833923e-05, -0.0116786422327, 0.0287998551014, 0.424678918993, -1.20629912179, 0.675013212728, 0.936068431591, -0.118766088348, 0.130936683699, -0.00550651147978, 0.0111253708206, 0.000890767938847, -0.130121733054, -0.195712660157, 0.413533717103, 0.588166252597, 0.281757292531, 0.0899107535319, -0.625904521458, -1.56979781802, 0.202940224704, 2.14854759605, -0.171377608919, 0.163232950118])

        self.restPose = np.array(qpos)

        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-0.2, high=0.2, size=self.robot_skeleton.ndofs)
        #qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qpos = qpos + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)


        if self.simulateCloth:
            self.clothScene.translateCloth(0, np.array([0, 3.0, 0]))

        #set on initialization and used to measure displacement
        self.initialProjectedAnkle = self.robot_skeleton.bodynodes[self.footBodyNode].to_world(np.zeros(3))
        self.initialProjectedAnkle[1] = 0

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
            [self.footCOP, self.footCOP+np.array([0,self.footNormForceMag,0])]
                    ]

        renderUtils.setColor(color=[0.0,1.0,0])
        renderUtils.drawLines(COPLines)
        renderUtils.setColor(color=[0.0, 0.0, 1.0])
        renderUtils.drawLines([[self.robot_skeleton.com(), np.array([self.robot_skeleton.com()[0], -2.0, self.robot_skeleton.com()[2]])]])

        dist2tri, closeP = pyutils.distToTriangle(self.stabilityPolygon[0], self.stabilityPolygon[1], self.stabilityPolygon[2], self.projectedCOM)
        renderUtils.drawLines(lines=[[closeP, self.projectedCOM]])

        #compute the zero moment point
        if True:
            #print("--------------------------------------")
            #print("computing ZMP")
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
                I = node.I
                #R = node.T() #TODO: may need to extract R from this?
                #w = node. #angular velocity
                wd = node.com_spatial_acceleration() #angular acceleration
                w = node.com_spatial_velocity()
                H = node.com_angular_momentum()
                #G =
                #print("H: " + str(H))
                #print(w)
                #print(wd)
                #print(node.com_linear_acceleration())
                #TODO: combine

            #scalar ZMP computation:
            xZMP = 0
            zZMP = 0
            g = 9.8
            denom = self.robot_skeleton.mass() * g #mg
            for node in self.robot_skeleton.bodynodes:
                m = node.mass()
                G = node.com()
                H = node.com_angular_momentum()
                Gdd = node.com_linear_acceleration()
                xZMP += m*G[0]*g - (H[2] + m*(G[1]*Gdd[0] - G[0]*Gdd[1]))
                zZMP += m*G[2]*g - (H[0] + m*(G[2]*Gdd[1] - G[1]*Gdd[2]))

                denom += m*Gdd[1]

            xZMP /= denom
            zZMP /= denom

            ZMP = np.array([xZMP, -2.0, zZMP])
            renderUtils.setColor(color=[0,1.0,0.0])
            renderUtils.drawLines([[np.array([ZMP[0], 1.0, ZMP[2]]), ZMP]])


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
                renderUtils.renderDofs(robot=self.robot_skeleton, restPose=self.restPose, renderRestPose=True)
