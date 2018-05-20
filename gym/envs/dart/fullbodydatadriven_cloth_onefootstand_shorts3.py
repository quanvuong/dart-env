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

class DartClothFullBodyDataDrivenClothOneFootStandShorts3Env(DartClothFullBodyDataDrivenClothBaseEnv, utils.EzPickle):
    def __init__(self):
        #feature flags
        rendering = True
        clothSimulation = True
        renderCloth = True
        self.gravity = False
        SPDActionSpace = False
        frameskip = 5
        dt = 0.002
        leftFootLocked = True

        # observation terms
        self.featureInObs       = False  # if true, feature centroid location and displacement from ef are observed
        self.oracleInObs        = False  # if true, oracle vector is in obs
        self.contactIDInObs     = False  # if true, contact ids are in obs
        self.limbProgressInObs  = False  # if true, include waist and leg limb progress in the reward function

        #reward flags
        self.restPoseReward             = True
        self.restCOMsReward             = True #if True, penalize displacement between world targets and the positions of local offsets
        self.stabilityCOMReward         = True
        self.contactReward              = False
        self.flatFootReward             = True  # if true, reward the foot for being parallel to the ground
        self.COMHeightReward            = False
        self.aliveBonusReward           = True #rewards rollout duration to counter suicidal tendencies
        self.stationaryAnkleAngleReward = False #penalizes ankle joint velocity
        self.stationaryAnklePosReward   = False #penalizes planar motion of projected ankle point
        self.footPosReward              = True  # if true, reward the foot for being at a particular location (3 point frame)

        #dressing reward flags
        self.waistContainmentReward     = False
        self.deformationPenalty         = True
        self.footBetweenHandsReward     = False #reward foot between the hands
        self.limbProgressReward         = False  # if true, the (-inf, 1] plimb progress metric is included in reward
        self.oracleDisplacementReward   = False  # if true, reward ef displacement in the oracle vector direction
        self.contactGeoReward           = False  # if true, [0,1] reward for ef contact geo (0 if no contact, 1 if limbProgress > 0).

        #reward weights
        self.restPoseRewardWeight               = 0.5
        self.restCOMsRewardWeight               = 1
        self.stabilityCOMRewardWeight           = 5
        self.contactRewardWeight                = 1
        self.flatFootRewardWeight               = 4
        self.COMHeightRewardWeight              = 2
        self.aliveBonusRewardWeight             = 60
        self.stationaryAnkleAngleRewardWeight   = 0.025
        self.stationaryAnklePosRewardWeight     = 2
        self.footPosRewardWeight                = 20

        #dressing reward weights
        self.waistContainmentRewardWeight       = 10
        self.deformationPenaltyWeight           = 5.0
        self.footBetweenHandsRewardWeight       = 1
        self.limbProgressRewardWeight           = 15
        self.oracleDisplacementRewardWeight     = 50
        self.contactGeoRewardWeight             = 4

        #other flags
        self.stabilityTermination = False #if COM outside stability region, terminate #TODO: timed?
        self.contactTermination   = False #if anything except the feet touch the ground, terminate
        self.wrongEnterTermination= True #terminate if the foot enters the pant legs
        self.legOutTermination    = True #terminate if the leg is outside of the correct shorts features
        self.COMHeightTermination = False  # terminate if COM drops below a certain height
        self.footDistanceTermination = True

        self.resetStateFromDistribution = True
        self.resetDistributionPrefix = "saved_control_states_shorts/enter_seq_rlegdown"
        self.resetDistributionSize = 20

        self.COMMinHeight = -0.6

        #other variables
        self.prevTau = None
        self.restPose = None
        self.restCOMs = []
        self.footOffsets = [np.array([0, 0, -0.2]), np.array([0.05, 0, 0.03]), np.array([-0.05, 0, 0.03])] #local positions of the foot to query for foot location reward
        self.footTargets = [] #world foot position targets for the foo location reward
        self.targetCOM = np.zeros(3) #target for the center of the stability region
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
        self.footBodyNode = 20 #17 left, 20 right
        self.ankleDofs = [32,33] #[32,33] left, [38,39] right
        self.fingertip = np.array([0,-0.08,0])
        self.handsPlane = Plane()
        self.prevOracle = None
        self.efBeforeSim = np.zeros(3)
        self.prevWaistContainment = 0

        self.debugPoints = []

        #handle nodes
        self.handleNodes = []
        self.updateHandleNodesFrom = [7, 12]

        self.actuatedDofs = np.arange(34)
        observation_size = 0
        #observation_size = 37 * 3 + 6 #q[:3], q[3:](sin,cos), dq
        observation_size = 34 * 3 + 12  # q[6:](sin,cos), dq, trans, orient
        observation_size += 3 # COM
        #observation_size += 1 # binary contact per foot with ground
        #observation_size += 4 # feet COPs and norm force mags
        observation_size += 40*3 # haptic sensor readings
        if self.featureInObs:
            observation_size += 6
        if self.oracleInObs:
            observation_size += 3
        if self.contactIDInObs:
            observation_size += 40
        if self.limbProgressInObs:
            observation_size += 2



        DartClothFullBodyDataDrivenClothBaseEnv.__init__(self,
                                                          rendering=rendering,
                                                          screensize=(1280,920),
                                                          clothMeshFile="shorts_med.obj",
                                                          clothScale=np.array([0.9,0.9,0.9]),
                                                          obs_size=observation_size,
                                                          simulateCloth=clothSimulation,
                                                          gravity=self.gravity,
                                                          frameskip=frameskip,
                                                          dt=dt,
                                                          SPDActionSpace=SPDActionSpace,
                                                          lockedLFoot=leftFootLocked)

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

        #setup the handle nodes
        if self.simulateCloth:
            self.handleNodes.append(HandleNode(self.clothScene, org=np.array([0.05, 0.034, -0.975])))
            self.handleNodes.append(HandleNode(self.clothScene, org=np.array([0.05, 0.034, -0.975])))

        if not renderCloth:
            self.clothScene.renderClothFill = False
            self.clothScene.renderClothBoundary = False
            self.clothScene.renderClothWires = False

        if self.restPoseReward:
            self.rewardsData.addReward(label="restPose", rmin=-51.0, rmax=0, rval=0, rweight=self.restPoseRewardWeight)

        if self.restCOMsReward:
            self.rewardsData.addReward(label="rest COMs", rmin=-20.0, rmax=0, rval=0, rweight=self.restCOMsRewardWeight)

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

        if self.footPosReward:
            self.rewardsData.addReward(label="foot pos", rmin=-3.0, rmax=0.0, rval=0, rweight=self.footPosRewardWeight)

        if self.waistContainmentReward:
            self.rewardsData.addReward(label="waist containment", rmin=-1.0, rmax=1.0, rval=0, rweight=self.waistContainmentRewardWeight)

        if self.deformationPenalty:
            self.rewardsData.addReward(label="deformation", rmin=-1.0, rmax=0, rval=0, rweight=self.deformationPenaltyWeight)

        if self.footBetweenHandsReward:
            self.rewardsData.addReward(label="foot btw hands", rmin=-1.0, rmax=0, rval=0, rweight=self.footBetweenHandsRewardWeight)

        if self.limbProgressReward:
            self.rewardsData.addReward(label="limb progress", rmin=0.0, rmax=1.0, rval=0, rweight=self.limbProgressRewardWeight)

        if self.oracleDisplacementReward:
            self.rewardsData.addReward(label="oracle", rmin=-0.1, rmax=0.1, rval=0, rweight=self.oracleDisplacementRewardWeight)

        if self.contactGeoReward:
            self.rewardsData.addReward(label="contact geo", rmin=0, rmax=1.0, rval=0, rweight=self.contactGeoRewardWeight)

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

        self.debugPoints[-1] = np.array(self.stabilityPolygonCentroid)

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

        efl = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
        efr = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)

        self.handsPlane = fitPlane(points=[efl,efr,(efl+efr)/2.0 + np.array([0,1.0,0])], basis1hint=np.array([0,1.0,0]))

        self.efBeforeSim = self.robot_skeleton.bodynodes[20].to_world(self.toeOffset)
        a=0

    def checkTermination(self, tau, s, obs):
        #check the termination conditions and return: done,reward
        if not np.isfinite(s).all():
            #print(self.rewardTrajectory)
            #print(self.stateTraj[-2:])
            print("Infinite value detected..." + str(s))
            return True, 0
        elif np.amax(np.absolute(s[:len(self.robot_skeleton.q)])) > 10:
            print("Detecting potential instability")
            print(s)
            print(self.rewardTrajectory)
            #print(self.stateTraj[-2:])
            return True, 0
        '''elif np.amax(np.absolute(s[:len(self.robot_skeleton.q)]-self.stateTraj[-1])) > 1.0:
            print("Detecting potential instability via velocity: " + str(np.amax(np.absolute(s[:len(self.robot_skeleton.q)]-self.stateTraj[-1]))))
            print(s)
            print(self.stateTraj[-2:])
            print(self.rewardTrajectory)
            return True, -1500'''


        #print(np.amax(np.absolute(s[:len(self.robot_skeleton.q)]-self.stateTraj[-1])))

        #stability termination
        if self.stabilityTermination:
            if not self.stableCOM:
                return True, 0

        #contact termination
        if self.contactTermination:
            if self.nonFootContact:
                return True, 0

        if self.wrongEnterTermination:
            errors = []
            #errors.append(pyutils.limbFeatureProgress( limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesLegR, offset=self.toeOffset), feature=self.legEndFeatureL))
            #errors.append(pyutils.limbFeatureProgress( limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesLegR, offset=self.toeOffset), feature=self.legMidFeatureL))
            #errors.append(pyutils.limbFeatureProgress( limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesLegR, offset=self.toeOffset), feature=self.legStartFeatureL))
            errors.append(pyutils.limbFeatureProgress( limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesLegR, offset=self.toeOffset), feature=self.legEndFeatureR))
            errors.append(pyutils.limbFeatureProgress( limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesLegR, offset=self.toeOffset), feature=self.legMidFeatureR))
            errors.append(pyutils.limbFeatureProgress( limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesLegR, offset=self.toeOffset), feature=self.legStartFeatureR))
            #print(errors)
            for e in errors:
                if e > 0:
                    return True, -100

        if self.legOutTermination:
            errors = []
            #errors.append(pyutils.limbFeatureProgress( limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesLegR, offset=self.toeOffset), feature=self.legEndFeatureL))
            errors.append(pyutils.limbFeatureProgress( limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesLegR, offset=self.toeOffset), feature=self.legMidFeatureL))
            errors.append(pyutils.limbFeatureProgress( limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesLegR, offset=self.toeOffset), feature=self.legStartFeatureL))
            errors.append(pyutils.limbFeatureProgress( limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesLegR, offset=self.toeOffset), feature=self.waistFeature))

            for e in errors:
                if e < 0:
                    return True, -100
                    #if self.limbProgress > 0 and self.prevWaistContainment < 0:
                #entered the pant leg before the waist or pulled the waist off the leg via penetration
            #    return True, -100

        if self.footDistanceTermination:
            maxError = 0.0
            for ix, p in enumerate(self.footOffsets):
                pos = self.robot_skeleton.bodynodes[20].to_world(p)
                maxError = max(maxError, np.linalg.norm(pos - self.footTargets[ix]))
            #print(maxError)
            if maxError < 0.05:
                return True, 1000

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

        reward_restCOMs = 0
        if self.restCOMsReward:
            for ix, b in enumerate(self.robot_skeleton.bodynodes):
                reward_restCOMs -= np.linalg.norm(self.restCOMs[ix] - b.com())
            reward_restCOMs = max(reward_restCOMs, -20)
            reward_record.append(reward_restCOMs)

        #reward COM over stability region
        reward_stability = 0
        if self.stabilityCOMReward:
            #penalty for distance from projected COM to stability centroid
            reward_stability = -np.linalg.norm(self.targetCOM - self.projectedCOM)
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

        reward_footPos = 0
        if self.footPosReward:
            for ix,p in enumerate(self.footOffsets):
                pos = self.robot_skeleton.bodynodes[20].to_world(p)
                reward_footPos -= np.linalg.norm(pos-self.footTargets[ix])
            reward_record.append(reward_footPos)

        reward_waistContainment = 0
        if self.waistContainmentReward:
            if self.simulateCloth:
                self.prevWaistContainment = pyutils.limbFeatureProgress(
                    limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesLegR,
                                                      offset=self.toeOffset), feature=self.waistFeature)
                reward_waistContainment = self.prevWaistContainment
                #print(reward_waistContainment)
                '''if reward_waistContainment <= 0:  # replace centroid distance penalty with border distance penalty
                    #distance to feature
                    distance2Feature = 999.0
                    toe = self.robot_skeleton.bodynodes[20].to_world(self.toeOffset)
                    for v in self.waistFeature.verts:
                        dist = np.linalg.norm(self.clothScene.getVertexPos(cid=0, vid=v) - toe)
                        if dist < distance2Feature:
                            distance2Feature = dist
                            reward_waistContainment = - distance2Feature
                            #print(reward_waistContainment)'''
            reward_record.append(reward_waistContainment)

        clothDeformation = 0
        if self.simulateCloth is True:
            clothDeformation = self.clothScene.getMaxDeformationRatio(0)
            self.deformation = clothDeformation

        reward_clothdeformation = 0
        if self.deformationPenalty is True:
            # reward_clothdeformation = (math.tanh(9.24 - 0.5 * clothDeformation) - 1) / 2.0  # near 0 at 15, ramps up to -1.0 at ~22 and remains constant
            l = 5
            s = 0.4
            z = -0.5
            reward_clothdeformation = z * math.tanh(s * (clothDeformation - l)) + z
            reward_record.append(reward_clothdeformation)
        self.previousDeformationReward = reward_clothdeformation

        reward_footBetweenHands = 0
        if self.footBetweenHandsReward:
            f2D = self.handsPlane.get2D(p=self.robot_skeleton.bodynodes[20].com())
            efl = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
            efr = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)
            efl2D = self.handsPlane.get2D(p=efl)
            efr2D = self.handsPlane.get2D(p=efr)
            if (f2D[1] < efl2D[1]) is (f2D[1] < efr2D[1]): #not between the hands
                d = min(abs(f2D[1]-efl2D[1]), abs(f2D[1]-efr2D[1]))
                reward_footBetweenHands = -d
            if (f2D[0] < efl2D[0]) is (f2D[0] < efr2D[0]):  # not between the hands
                d = min(abs(f2D[0] - efl2D[0]), abs(f2D[0] - efr2D[0]))
                reward_footBetweenHands += -d

            reward_record.append(reward_footBetweenHands)

        reward_limbprogress = 0
        if self.limbProgressReward:
            if self.simulateCloth:
                self.limbProgress = pyutils.limbFeatureProgress(
                    limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesLegR,
                                                      offset=self.toeOffset), feature=self.legMidFeatureL)
                reward_limbprogress = self.limbProgress
                if reward_limbprogress < 0:  # remove euclidean distance penalty before containment
                    reward_limbprogress = 0
            reward_record.append(reward_limbprogress)

        reward_oracleDisplacement = 0
        if self.oracleDisplacementReward:
            if np.linalg.norm(self.prevOracle) > 0:
                # world_ef_displacement = wRFingertip2 - wRFingertip1
                ef = self.robot_skeleton.bodynodes[20].to_world(self.toeOffset)
                displacement = ef - self.efBeforeSim
                oracle0 = self.prevOracle
                reward_oracleDisplacement += displacement.dot(oracle0)
            reward_record.append(reward_oracleDisplacement)

        avgContactGeodesic = None
        if self.numSteps > 0 and self.simulateCloth:
            contactInfo = pyutils.getContactIXGeoSide(sensorix=39, clothscene=self.clothScene,
                                                      meshgraph=self.separatedMesh)
            if len(contactInfo) > 0:
                avgContactGeodesic = 0
                for c in contactInfo:
                    avgContactGeodesic += c[1]
                avgContactGeodesic /= len(contactInfo)

        reward_contactGeo = 0
        if self.contactGeoReward:
            if self.simulateCloth:
                if self.limbProgress > 0:
                    reward_contactGeo = 1.0
                elif avgContactGeodesic is not None:
                    reward_contactGeo = 1.0 - (avgContactGeodesic / self.separatedMesh.maxGeo)
                    # reward_contactGeo = 1.0 - minContactGeodesic / self.separatedMesh.maxGeo
            reward_record.append(reward_contactGeo)

        # update the reward data storage
        self.rewardsData.update(rewards=reward_record)

        self.reward = reward_contact * self.contactRewardWeight \
                    + reward_stability * self.stabilityCOMRewardWeight \
                    + reward_restPose * self.restPoseRewardWeight \
                    + reward_COMHeight * self.COMHeightRewardWeight \
                    + reward_alive * self.aliveBonusRewardWeight \
                    + reward_stationaryAnkleAngle * self.stationaryAnkleAngleRewardWeight \
                    + reward_stationaryAnklePos * self.stationaryAnklePosRewardWeight \
                    + reward_flatFoot * self.flatFootRewardWeight \
                    + reward_waistContainment * self.waistContainmentRewardWeight \
                    + reward_clothdeformation * self.deformationPenaltyWeight \
                    + reward_footBetweenHands * self.footBetweenHandsRewardWeight \
                    + reward_oracleDisplacement * self.oracleDisplacementRewardWeight \
                    + reward_limbprogress * self.limbProgressRewardWeight \
                    + reward_contactGeo * self.contactGeoRewardWeight \
                    + reward_footPos * self.footPosRewardWeight \
                    + reward_restCOMs * self.restCOMsRewardWeight

        return self.reward

    def _get_obs(self):
        obs = np.zeros(self.obs_size)

        orientation = np.array(self.robot_skeleton.q[:3])
        theta = np.array(self.robot_skeleton.q[6:])
        dq = np.array(self.robot_skeleton.dq)
        trans = np.array(self.robot_skeleton.q[3:6])

        #obs = np.concatenate([np.cos(orientation), np.sin(orientation), trans, np.cos(theta), np.sin(theta), dq]).ravel()
        obs = np.concatenate([np.cos(theta), np.sin(theta), dq, orientation, trans]).ravel()

        #COM
        com = np.array(self.robot_skeleton.com()).ravel()
        obs = np.concatenate([obs, com]).ravel()

        #foot contacts
        '''if self.footContact:
            obs = np.concatenate([obs, [1.0]]).ravel()
        else:
            obs = np.concatenate([obs, [0.0]]).ravel()'''

        #foot COP and norm force magnitude
        #obs = np.concatenate([obs, self.footCOP, [self.footNormForceMag]]).ravel()

        #haptic observations
        f = np.zeros(40*3)
        if self.simulateCloth:
            f = self.clothScene.getHapticSensorObs()
        obs = np.concatenate([obs, f]).ravel()

        if self.featureInObs:
            centroid = self.legMidFeatureL.plane.org
            ef = self.robot_skeleton.bodynodes[20].to_world(self.toeOffset)
            disp = centroid - ef
            obs = np.concatenate([obs, centroid, disp]).ravel()

        if self.oracleInObs:
            oracle = np.zeros(3)
            if self.reset_number == 0:
                a = 0  # nothing
            elif self.limbProgress > 0:
                oracle = self.legMidFeatureL.plane.normal
            else:
                minContactGeodesic, minGeoVix, _side = pyutils.getMinContactGeodesic(sensorix=39,
                                                                                     clothscene=self.clothScene,
                                                                                     meshgraph=self.separatedMesh,
                                                                                    returnOnlyGeo=False)
                if minGeoVix is None:
                    # new: oracle points to the waist feature centroid when not in contact with cloth
                    target = self.waistFeature.plane.org
                    ef = self.robot_skeleton.bodynodes[20].to_world(self.toeOffset)
                    vec = target - ef
                    oracle = vec / np.linalg.norm(vec)
                else:
                    vixSide = 0
                    if _side:
                        vixSide = 1
                    if minGeoVix >= 0:
                        oracle = self.separatedMesh.geoVectorAt(minGeoVix, side=vixSide)
            self.prevOracle = oracle
            obs = np.concatenate([obs, oracle]).ravel()

        #print(obs)
        if self.contactIDInObs:
            HSIDs = self.clothScene.getHapticSensorContactIDs()
            obs = np.concatenate([obs, HSIDs]).ravel()

        if self.limbProgressInObs:
            obs = np.concatenate([obs, [max(-1, self.limbProgress), max(-1, self.prevWaistContainment)]]).ravel()

        return obs

    def additionalResets(self):
        self.debugPoints.append(np.zeros(3))
        #do any additional resetting here
        #TODO: set a one foot standing initial pose
        #qpos = np.array([-0.00469234655801, -0.0218378114573, -0.011132330496, 0.00809830385355, 0.00051861417993, 0.0584867818269, 0.374712375814, 0.0522417260384, -0.00777676124956, 0.00230285789432, -0.00274958108859, -0.008064630425, 0.00247294825781, -0.0093978116532, 0.195632645271, -0.00276696945071, 0.0075491687512, -0.0116846422966, 0.00636619242284, 0.00767084047346, -0.00913509000374, 0.00857521738396, 0.199096855493, 0.00787726246678, -0.00760402683795, -0.00433642327146, 0.00802311463366, -0.00482248656677, 0.131248337324, -0.00662274635457, 0.00333416764933, 0.00546016678096, -0.00150775759695, -0.00861184703697, -0.000589790168521, -0.832681560131, 0.00976653127827, 2.24259637323, -0.00374506255585, -0.00244949106062])
        #qpos = np.array([0.0792540309714, -0.198038537538, 0.165982043711, 0.057678066664, -0.03, 0.0514905008947, 0.0153889940281, 0.172754267613, -0.0152665902114, 0.0447139591458, -0.0159152223541, -0.741653453661, -0.291857490409, 0.073, 0.247712958964, 0.0230298051369, -0.0129663713662, -0.0251081943623, 0.0184011650614, 0.0284706137625, -0.0143437953932, 0.00253595897386, 0.202764558055, 0.008180767185, 0.00144036976378, -0.00599927843092, 0.010826449318, 0.00831071336219, -0.0983439895237, -0.189360110846, 0.291156844437, 0.58830057445, 0.337019304264, 0.151085855622, 0.00418796434677, -0.841518739136, 0.0266268945701, 2.23410594707, -0.0133781980567, 0.00155774102539])
        #qpos = np.array([0.0876736007526, -0.197663127282, 0.0512116324906, 0.0890321935781, 0.00503663000394, 0.153578012064, 0.0355287603611, 0.188868068244, -0.0393309627431, 0.0468919964458, -0.0157749170449, -0.732901924741, -0.342059517398, 0.0771170149731, 1.63974967002, 0.0234712949278, -0.0123146387565, -0.0224550112983, 0.0314256125072, 0.0304105325139, -0.806838907433, 0.578199999561, 1.2304995663, 0.00308410667268, -0.0020098497117, 0.000671096336519, 0.0160262363305, 0.00983038089353, -0.0139367467948, -0.255746135035, 0.292140467717, 0.585292554732, 0.338902153816, 0.0897545118006, -0.637071641222, -1.56662542487, 0.209168756426, 2.14915980862, -0.177912854867, 0.164453921709])
        #qpos = np.array([0.112736818959, -0.184613672394, 0.0483281279476, 0.0977783961469, -0.0395725338171, 0.118916694347, 0.0225602781819, 0.186664042145, -0.0424497225658, 0.0513031599457, -0.0239023434306, -0.72814835681, -0.338360476232, 0.0678506261861, 1.64682143574, 0.0186850837192, -0.00930510664293, -0.0207839101411, 0.0371876854933, 0.0289668575664, -0.8132911622, 0.581233898831, 1.23126365295, -0.00548656272987, 0.00129006785898, -0.00292524994714, 0.00612636259669, 0.00288579869358, -0.0177117020336, -0.189844550703, 0.403929649975, 0.584144641901, 0.27891744078, 0.0979740469555, -0.635892041362, -1.56305168508, 0.199326525817, 2.15366260278, -0.178666316407, 0.159687854881])
        #qpos = np.array([0.0783082859519, -0.142335807127, 0.142293175071, 0.144942555142, 0.0133898601508, 0.165276500738, 0.0160630842837, 0.190633101962, -0.0300941169552, 0.0450348264227, -0.0254260608645, -0.878047724726, -0.485648077845, 0.239917328593, 1.4876567992, 0.00147541881953, -4.95678764178e-05, -0.0115193558397, 0.0292081008816, 0.424683550591, -1.20586786954, 0.674957465063, 0.935902478547, -0.118767946668, 0.130931065834, -0.00538714909167, 0.0113181295264, 0.000849328006241, -0.025344210202, -0.195371502228, 0.413691811409, 0.588269242275, 0.281748815666, 0.0899108878587, -0.626263215102, -1.5691826733, 0.202581615871, 2.1489384041, -0.171405162264, 0.163236501426])
        #qpos = np.array([0.0780131557357, -0.142593660368, 0.143019989259, 0.144666815206, -0.035, 0.165147835811, 0.0162260416596, 0.19115105848, -0.0299336428088, 0.0445035430603, -0.025419636699, -0.878286887463, -0.485843951506, 0.239911240107, 1.48781704099, 0.00147260210175, -3.84887833923e-05, -0.0116786422327, 0.0287998551014, 0.424678918993, -1.20629912179, 0.675013212728, 0.936068431591, -0.118766088348, 0.130936683699, -0.00550651147978, 0.0111253708206, 0.000890767938847, -0.130121733054, -0.195712660157, 0.413533717103, 0.588166252597, 0.281757292531, 0.0899107535319, -0.625904521458, -1.56979781802, 0.202940224704, 2.14854759605, -0.171377608919, 0.163232950118])

        '''self.restPose = np.array(self.robot_skeleton.q)
        self.restPose[1] = 0.5
        self.restPose[3] = 0.352
        self.restPose[5] = 0
        #self.restPose[5] = 0.56
        self.restPose[14] = 0.2
        self.restPose[22] = 0.2'''
        #self.restPose = np.array([0.415926215284, -0.208072254172, 0.0359992793634, 0.197008695274, -0.080946120072, 0.386294441391, 0.500259401372, 1.00291802213, 0.190366268989, -0.199667785215, -0.221454351362, -0.995964947924, -0.322756577729, -0.185862207419, 1.20874032662, -0.601246970796, -0.356215287304, 0.248648034524, -0.207736588973, 1.19045063115, -1.10962637351, 0.64449704768, 1.20892254234, -0.603004201007, 0.584056520122, -0.101334215359, -0.217255331061, 0.665068109449, -0.0431213403695, 0.120595975597, 0.558129378897, 0.522613519553, 0.186776361183, 0.00898025185228, -1.04489130048, -0.674308682483, 0.871716975402, 1.32610514665, -1.55570020123, 0.262473197491])
        #self.restPose = np.array([0.415926215284, -0.208072254172, 0.0359992793634, 0.197008695274, -0.080946120072, 0.386294441391, 0.500259401372, 1.00291802213, 0.190366268989, -0.199667785215, -0.221454351362, -0.995964947924, -0.322756577729, -0.185862207419, 1.20874032662, -0.601246970796, -0.356215287304, 0.248648034524, -0.207736588973, 1.19045063115, -1.10962637351, 0.64449704768, 1.20892254234, -0.603004201007, 0.584056520122, -0.101334215359, -0.217255331061, 0.665068109449, -0.0772335924034, 0.345777940801, 0.347712898192, 0.597222545088, 0.163680756136, -0.0716346721614, 0.00539065251275, 0.350484523151, 0.298952447783, 0.510535951384, 0.178043644112, -0.0715581001448])
        #self.restPose = np.array([0.525824064201, -0.399976484702, 0.0887478468173, 0.145089767505, -0.136825881358, 0.560569009075, 0.504712938155, 1.00752481112, 0.0945622609612, -0.228753234696, -0.237164704235, 0.0793722954261, 0.986201206344, -0.343858171372, 0.999108918639, -0.591312452645, -0.589003131028, 0.253242784195, -0.194353296464, 1.07117931579, -0.964339003354, 0.635248852788, 1.21447608173, -0.574513297921, 0.562541004218, 0.0181084859628, -0.321564671007, 0.532503853831, -0.14731292525, 0.0404158608684, 0.55367653044, 0.638011072879, 0.101733194299, 0.195334829555, -0.987216985687, -0.540165630908, 0.86912949813, 1.25936404052, -1.32657545849, 0.364504394379])
        self.restPose = np.array([0.451776148062, -0.408060690682, -0.0813767999963, 0.231914742755, -0.128517505377, 0.509980348602, 0.500012087933, 1.00350776151, 0.128344076103, -0.249493315526, -0.250963105237, -0.410587913964, -0.300922973715, -0.632096540698, 1.04160284967, -0.600296442311, -0.573678985941, 0.250153764375, -0.227018065035, 1.27197819319, -0.83950877621, 0.968879260074, 0.695461638617, -0.594546287862, 0.604086349948, -0.0750487146813, -0.238633047019, 0.646839623225, 0.00530710951417, -0.0079435054315, 0.646537024558, 0.619000285028, 0.131835107278, 0.182739902529, 0.00119200512317, -0.295788344638, 0.2387694539, 0.732666666667, -0.0594136415473, -0.0843937492205])

        self.robot_skeleton.set_positions(self.restPose)
        self.footTargets = []
        for p in self.footOffsets:
            self.footTargets.append(self.robot_skeleton.bodynodes[20].to_world(p))

        self.targetCOM = self.robot_skeleton.com()
        self.targetCOM[1] = -1.3

        self.restCOMs = []
        for b in self.robot_skeleton.bodynodes:
            self.restCOMs.append(b.com())

        #now set the real character state
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
            #resetStateNumber = self.reset_number%self.resetDistributionSize
            #resetStateNumber = 0
            #print("resetState: " + str(resetStateNumber))
            charfname_ix = self.resetDistributionPrefix + "_char%05d" % resetStateNumber
            self.clothScene.setResetState(cid=0, index=resetStateNumber)
            self.loadCharacterState(filename=charfname_ix)
            qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-0.2, high=0.2, size=self.robot_skeleton.ndofs)
            self.robot_skeleton.set_velocities(qvel)


        #qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-0.01, high=0.01, size=self.robot_skeleton.ndofs)
        #qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        #qpos = qpos + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        #self.set_state(qpos, qvel)


        #RX = pyutils.rotateX(-1.56)
        #self.clothScene.rotateCloth(cid=0, R=RX)
        #self.clothScene.translateCloth(cid=0, T=np.array([0.555, -0.5, -1.45]))
        #if self.simulateCloth:
            #self.clothScene.translateCloth(0, np.array([0, 3.0, 0]))
            #a=0

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

        if self.reset_number == 0 and self.simulateCloth:
            self.separatedMesh.initSeparatedMeshGraph()
            self.separatedMesh.updateWeights()
            # TODO: compute geodesic depends on cloth state! Deterministic initial condition required!
            # option: maybe compute this before setting the cloth state at all in initialization?
            self.separatedMesh.computeGeodesic(feature=self.legEndFeatureL, oneSided=True, side=1, normalSide=1)

        #set on initialization and used to measure displacement
        self.initialProjectedAnkle = self.robot_skeleton.bodynodes[self.footBodyNode].to_world(np.zeros(3))
        self.initialProjectedAnkle[1] = 0

        if self.limbProgressReward:
            self.limbProgress = pyutils.limbFeatureProgress(limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesLegR,offset=self.toeOffset), feature=self.legEndFeatureL)

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

        # rest COMs
        renderUtils.setColor(color=[1.0,1.0,0])
        restCOMsLines = []
        if self.restCOMsReward:
            for ix, b in enumerate(self.robot_skeleton.bodynodes):
                restCOMsLines.append([b.com(), self.restCOMs[ix]])
        renderUtils.drawLines(restCOMsLines)

        #render center of pressure for each foot
        COPLines = [
            [self.footCOP, self.footCOP+np.array([0,self.footNormForceMag,0])]
                    ]

        renderUtils.setColor(color=[0.0,1.0,0])
        renderUtils.drawLines(COPLines)
        renderUtils.setColor(color=[0.0, 0.0, 1.0])
        renderUtils.drawLines([[self.robot_skeleton.com(), np.array([self.robot_skeleton.com()[0], -2.0, self.robot_skeleton.com()[2]])]])

        renderUtils.setColor(color=[0.0, 1.0, 1.0])
        for p in self.debugPoints:
            renderUtils.drawSphere(pos=p)

        #compute the zero moment point
        if False:
            print("--------------------------------------")
            print("computing ZMP")
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
                print(w)
                print(wd)
                print(node.com_linear_acceleration())
                #TODO: combine

            #TODO: continue

        # render geodesic
        if False:
            for v in range(self.clothScene.getNumVertices()):
                side1geo = self.separatedMesh.nodes[v + self.separatedMesh.numv].geodesic
                side0geo = self.separatedMesh.nodes[v].geodesic

                pos = self.clothScene.getVertexPos(vid=v)
                norm = self.clothScene.getVertNormal(vid=v)
                renderUtils.setColor(
                    color=renderUtils.heatmapColor(minimum=0, maximum=self.separatedMesh.maxGeo,
                                                   value=self.separatedMesh.maxGeo - side0geo))
                renderUtils.drawSphere(pos=pos - norm * 0.01, rad=0.01)
                renderUtils.setColor(
                    color=renderUtils.heatmapColor(minimum=0, maximum=self.separatedMesh.maxGeo,
                                                   value=self.separatedMesh.maxGeo - side1geo))
                renderUtils.drawSphere(pos=pos + norm * 0.01, rad=0.01)

        #draw the oracle
        if self.oracleDisplacementReward:
            ef = self.robot_skeleton.bodynodes[20].to_world(self.toeOffset)
            renderUtils.drawArrow(p0=ef, p1=ef + self.prevOracle)

        if self.footPosReward:
            footTargetLines = []
            if self.footPosReward:
                if len(self.footTargets) == len(self.footOffsets):
                    renderUtils.setColor([0, 1.0, 0])
                    for p in self.footOffsets:
                        pos = self.robot_skeleton.bodynodes[20].to_world(p)
                        renderUtils.drawSphere(pos=pos)
                        footTargetLines.append([pos])
                    renderUtils.setColor([1.0, 0, 0])
                    for ix,t in enumerate(self.footTargets):
                        renderUtils.drawSphere(pos=t)
                        footTargetLines[ix].append(t)
            renderUtils.drawLines(footTargetLines)

        #draw the foot between hands info
        #self.handsPlane.draw()
        if self.footBetweenHandsReward:
            f2D = self.handsPlane.get2D(p=self.robot_skeleton.bodynodes[20].com())
            efl = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
            efr = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)
            efl2D = self.handsPlane.get2D(p=efl)
            efr2D = self.handsPlane.get2D(p=efr)
            renderUtils.setColor(color=[0,1,0])
            renderUtils.drawSphere(pos=self.handsPlane.org+self.handsPlane.basis2*f2D[1])
            renderUtils.setColor(color=[1, 0, 0])
            renderUtils.drawSphere(pos=self.handsPlane.org+self.handsPlane.basis2*efl2D[1])
            renderUtils.drawSphere(pos=self.handsPlane.org+self.handsPlane.basis2*efr2D[1])
            renderUtils.setColor(color=[0, 1, 0])
            renderUtils.drawSphere(pos=self.handsPlane.org + self.handsPlane.basis1 * f2D[0])
            renderUtils.setColor(color=[1, 0, 0])
            renderUtils.drawSphere(pos=self.handsPlane.org + self.handsPlane.basis1 * efl2D[0])
            renderUtils.drawSphere(pos=self.handsPlane.org + self.handsPlane.basis1 * efr2D[0])

        #render the ideal stability polygon
        if len(self.stabilityPolygon) > 0:
            renderUtils.drawPolygon(self.stabilityPolygon)
        renderUtils.setColor([0.0,0,1.0])
        renderUtils.drawSphere(pos=self.projectedCOM)
        renderUtils.setColor([0.0, 1.0, 1.0])
        renderUtils.drawSphere(pos=self.targetCOM)

        self.gripFeatureL.drawProjectionPoly(renderNormal=False, renderBasis=False, fillColor=[1.0,0.0,0.0])
        self.gripFeatureR.drawProjectionPoly(renderNormal=False, renderBasis=False, fillColor=[0.0,1.0,0.0])
        self.legEndFeatureL.drawProjectionPoly(renderNormal=True, renderBasis=False, fillColor=[1.0,0.0,0.0])
        self.legEndFeatureR.drawProjectionPoly(renderNormal=True, renderBasis=False, fillColor=[0.0,1.0,0.0])
        self.legMidFeatureL.drawProjectionPoly(renderNormal=True, renderBasis=False, fillColor=[1.0, 0.0, 0.0])
        self.legMidFeatureR.drawProjectionPoly(renderNormal=True, renderBasis=False, fillColor=[0.0, 1.0, 0.0])
        self.legStartFeatureL.drawProjectionPoly(renderNormal=True, renderBasis=False, fillColor=[1.0, 0.0, 0.0])
        self.legStartFeatureR.drawProjectionPoly(renderNormal=True, renderBasis=False, fillColor=[0.0, 1.0, 0.0])
        self.waistFeature.drawProjectionPoly(renderNormal=True, renderBasis=False, fillColor=[0.0, 0.0, 1.0])

        lines = []
        lines.append([self.robot_skeleton.bodynodes[18].to_world(np.zeros(3)), self.robot_skeleton.bodynodes[19].to_world(np.zeros(3))])
        lines.append([self.robot_skeleton.bodynodes[19].to_world(np.zeros(3)), self.robot_skeleton.bodynodes[20].to_world(np.zeros(3))])
        lines.append([self.robot_skeleton.bodynodes[20].to_world(np.zeros(3)), self.robot_skeleton.bodynodes[20].to_world(self.toeOffset)])

        distance2Feature = 999.0
        pos = np.zeros(3)
        toe = self.robot_skeleton.bodynodes[20].to_world(self.toeOffset)
        if self.reset_number > 0:
            '''for v in self.waistFeature.verts:
                #print(v)
                dist = np.linalg.norm(self.clothScene.getVertexPos(cid=0, vid=v) - toe)
                if dist < distance2Feature:
                    distance2Feature = dist
                    pos = self.clothScene.getVertexPos(cid=0, vid=v)
                    #print(dist)
            '''
            pos = np.array(self.waistFeature.plane.org)
            lines.append([pos, toe])

        renderUtils.drawLines(lines)

        #visualize bodynode coms
        if False:
            lines = []
            for node in self.robot_skeleton.bodynodes:
                com = node.com()
                s = 0.25
                lines.append([np.array([-s+com[0], com[1], com[2]]), np.array([s+com[0], com[1], com[2]])])
                lines.append([np.array([com[0], -s+com[1], com[2]]), np.array([com[0], s+com[1], com[2]])])
                lines.append([np.array([com[0], com[1], -s+com[2],]), np.array([com[0], com[1], s+com[2]])])
            renderUtils.drawLines(lines)

        m_viewport = self.viewer.viewport
        # print(m_viewport)
        self.rewardsData.render(topLeft=[m_viewport[2] - 410, m_viewport[3] - 15],
                                dimensions=[400, -m_viewport[3] + 30])

        links = pyutils.getRobotLinks(self.robot_skeleton, pose=self.restPose)
        renderUtils.drawLines(lines=links)

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
            self.clothScene.drawText(x=15., y=textLines * textHeight, text="max deformation = " + str(self.deformation), color=(0., 0, 0))
            textLines += 1


            if self.numSteps > 0:
                renderUtils.renderDofs(robot=self.robot_skeleton, restPose=self.restPose, renderRestPose=True)
