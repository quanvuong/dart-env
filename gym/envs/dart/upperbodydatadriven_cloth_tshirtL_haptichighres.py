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

class DartClothUpperBodyDataDrivenClothTshirtLHapticHighResEnv(DartClothUpperBodyDataDrivenClothBaseEnv, utils.EzPickle):
    def __init__(self):
        #feature flags
        rendering = True
        clothSimulation = True
        renderCloth = True

        #observation terms
        self.featureInObs   = True  # if true, feature centroid location and displacement from ef are observed
        self.oracleInObs    = True  # if true, oracle vector is in obs
        self.contactIDInObs = True  # if true, contact ids are in obs
        self.hapticsInObs   = True  # if true, haptics are in observation
        self.prevTauObs     = False  # if true, previous action in observation

        #reward flags
        self.uprightReward              = True  #if true, rewarded for 0 torso angle from vertical
        self.stableHeadReward           = True  # if True, rewarded for - head/torso angle
        self.elbowFlairReward           = False
        self.limbProgressReward         = True  # if true, the (-inf, 1] plimb progress metric is included in reward
        self.oracleDisplacementReward   = True  # if true, reward ef displacement in the oracle vector direction
        self.contactGeoReward           = True  # if true, [0,1] reward for ef contact geo (0 if no contact, 1 if limbProgress > 0).
        self.deformationPenalty         = True
        self.restPoseReward             = False

        # reward weights
        self.uprightRewardWeight = 1
        self.stableHeadRewardWeight = 1
        self.elbowFlairRewardWeight = 1
        self.limbProgressRewardWeight = 15
        self.oracleDisplacementRewardWeight = 50
        self.contactGeoRewardWeight = 4
        self.deformationPenaltyWeight = 1  # was 5... then 1...
        self.restPoseRewardWeight = 1

        #other flags
        self.hapticsAware       = True  # if false, 0's for haptic input
        self.collarTermination  = True  #if true, rollout terminates when collar is off the head/neck
        self.sleeveEndTerm      = True  #if true, terminate the rollout if the arm enters the end of sleeve feature before the beginning (backwards dressing)

        self.resetStateFromDistribution = True
        #self.resetDistributionPrefix = "saved_control_states/ltuck_narrow"
        #self.resetDistributionSize = 3
        #self.resetDistributionPrefix = "saved_control_states/ltuck_wide"
        #self.resetDistributionSize = 17 #3
        self.resetDistributionPrefix = "saved_control_states/enter_seq_lsleeve"
        self.resetDistributionSize = 20 #20
        self.state_save_directory = "saved_control_states/"

        #other variables
        self.prevTau = None
        self.elbowFlairNode = 5
        self.maxDeformation = 30.0
        self.restPose = None
        self.prevOracle = None
        self.localLeftEfShoulder1 = None
        self.limbProgress = 0
        self.previousDeformationReward = 0
        self.fingertip = np.array([0, -0.08, 0])
        self.previousContainmentTriangle = [np.zeros(3), np.zeros(3), np.zeros(3)]

        #high res haptic sensor variables
        self.hapticSensorLocations = [] #one entry for each body node
        self.hapticSensorLocationFile = "hapticSensorLocations" #if none, compute
        self.hapticSensorsPerNode = [0, 51, 0, 16, 24, 20, 9, 15, 23, 24, 13, 9, 15, 20, 25] #must be updated to reflect the number of sensors which will be loaded if not computing from scratch
        self.totalSensors = 0
        for sensors in self.hapticSensorsPerNode:
            self.totalSensors += sensors

        self.handleNode = None
        self.updateHandleNodeFrom = 7  # right fingers

        self.actuatedDofs = np.arange(22)
        observation_size = len(self.actuatedDofs)*3 #q(sin,cos), dq
        if self.prevTauObs:
            observation_size += len(self.actuatedDofs)
        if self.hapticsInObs:
            observation_size += 3*self.totalSensors
        if self.featureInObs:
            observation_size += 6
        if self.oracleInObs:
            observation_size += 3
        if self.contactIDInObs:
            observation_size += self.totalSensors

        DartClothUpperBodyDataDrivenClothBaseEnv.__init__(self,
                                                          rendering=rendering,
                                                          screensize=(1280,720),
                                                          clothMeshFile="tshirt_m.obj",
                                                          clothMeshStateFile="objFile_startarm2.obj",
                                                          clothScale=1.4,
                                                          obs_size=observation_size)

        self.loadCharacterState(filename="characterState_startarm2")

        #clothing features
        #self.sleeveRVerts = [2580, 2495, 2508, 2586, 2518, 2560, 2621, 2529, 2559, 2593, 272, 2561, 2658, 2582, 2666, 2575, 2584, 2625, 2616, 2453, 2500, 2598, 2466]
        #self.sleeveREndVerts = [252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 10, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251]
        self.targetGripVerticesL = [46, 437, 955, 1185, 47, 285, 711, 677, 48, 905, 1041, 49, 741, 889, 45]
        self.targetGripVerticesR = [905, 1041, 49, 435, 50, 570, 992, 1056, 51, 676, 283, 52, 489, 892, 362, 53]
        self.collarVertices = [117, 115, 113, 900, 108, 197, 194, 8, 188, 5, 120]
        self.sleeveLVerts = [211, 2305, 2364, 2247, 2322, 2409, 2319, 2427, 2240, 2320, 2276, 2326, 2334, 2288, 2346, 2314, 2251, 2347, 2304, 2245, 2376, 2315]
        self.sleeveLMidVerts = [2379, 2357, 2293, 2272, 2253, 214, 2351, 2381, 2300, 2352, 2236, 2286, 2430, 2263, 2307, 2387, 2232, 2390, 2277, 2348, 2382, 2227, 2296, 2425]
        self.sleeveLEndVerts = [232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 9, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231]
        self.sleeveLSeamFeature = ClothFeature(verts=self.sleeveLVerts, clothScene=self.clothScene)
        self.sleeveLEndFeature = ClothFeature(verts=self.sleeveLEndVerts, clothScene=self.clothScene)
        self.sleeveLMidFeature = ClothFeature(verts=self.sleeveLMidVerts, clothScene=self.clothScene)
        self.collarFeature = ClothFeature(verts=self.collarVertices, clothScene=self.clothScene)
        self.gripFeatureL = ClothFeature(verts=self.targetGripVerticesL, clothScene=self.clothScene, b1spanverts=[889,1041], b2spanverts=[47,677])
        self.gripFeatureR = ClothFeature(verts=self.targetGripVerticesR, clothScene=self.clothScene, b1spanverts=[362,889], b2spanverts=[51,992])


        self.simulateCloth = clothSimulation
        if self.simulateCloth:
            self.handleNode = HandleNode(self.clothScene, org=np.array([0.05, 0.034, -0.975]))

        if not renderCloth:
            self.clothScene.renderClothFill = False
            self.clothScene.renderClothBoundary = False
            self.clothScene.renderClothWires = False

        # load rewards into the RewardsData structure
        if self.uprightReward:
            self.rewardsData.addReward(label="upright", rmin=-2.5, rmax=0, rval=0, rweight=self.uprightRewardWeight)

        if self.stableHeadReward:
            self.rewardsData.addReward(label="stable head",rmin=-1.2,rmax=0,rval=0, rweight=self.stableHeadRewardWeight)

        if self.elbowFlairReward:
            self.rewardsData.addReward(label="elbow flair", rmin=-1.0, rmax=0, rval=0,
                                       rweight=self.elbowFlairRewardWeight)

        if self.limbProgressReward:
            self.rewardsData.addReward(label="limb progress", rmin=0.0, rmax=1.0, rval=0,
                                       rweight=self.limbProgressRewardWeight)

        if self.contactGeoReward:
            self.rewardsData.addReward(label="contact geo", rmin=0, rmax=1.0, rval=0,
                                       rweight=self.contactGeoRewardWeight)

        if self.deformationPenalty:
            self.rewardsData.addReward(label="deformation", rmin=-1.0, rmax=0, rval=0,
                                       rweight=self.deformationPenaltyWeight)

        if self.oracleDisplacementReward:
            self.rewardsData.addReward(label="oracle", rmin=-0.1, rmax=0.1, rval=0,
                                       rweight=self.oracleDisplacementRewardWeight)

        if self.restPoseReward:
            self.rewardsData.addReward(label="rest pose", rmin=-1.0, rmax=0, rval=0,
                                       rweight=self.restPoseRewardWeight)

        self.state_save_directory = "saved_control_states/"
        self.saveStateOnReset = False

        #compute/load topical haptic sensors
        if self.hapticSensorLocationFile is None:
            self.computeHapticSensorLocations()
            self.saveHapticSensorLocations("hapticSensorLocations")
        else:
            self.loadHapticSensorLocations(filename=self.hapticSensorLocationFile)

    def _getFile(self):
        return __file__

    def updateBeforeSimulation(self):
        #any pre-sim updates should happen here

        #update features
        if self.sleeveLSeamFeature is not None:
            self.sleeveLSeamFeature.fitPlane()
        if self.sleeveLEndFeature is not None:
            self.sleeveLEndFeature.fitPlane()
        if self.sleeveLMidFeature is not None:
            self.sleeveLMidFeature.fitPlane()
        if self.collarFeature is not None:
            self.collarFeature.fitPlane()
        self.gripFeatureL.fitPlane()
        self.gripFeatureR.fitPlane()

        #update handle nodes
        if self.handleNode is not None and self.reset_number > 0:
            if self.updateHandleNodeFrom >= 0:
                self.handleNode.setTransform(self.robot_skeleton.bodynodes[self.updateHandleNodeFrom].T)
            self.handleNode.step()

        wLFingertip1 = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
        self.localLeftEfShoulder1 = self.robot_skeleton.bodynodes[8].to_local(wLFingertip1)  # left fingertip in left shoulder local frame

        a=0

    def checkTermination(self, tau, s, obs):
        #check the termination conditions and return: done,reward
        topHead = self.robot_skeleton.bodynodes[14].to_world(np.array([0, 0.25, 0]))
        bottomHead = self.robot_skeleton.bodynodes[14].to_world(np.zeros(3))
        bottomNeck = self.robot_skeleton.bodynodes[13].to_world(np.zeros(3))

        if np.amax(np.absolute(s[:len(self.robot_skeleton.q)])) > 10:
            print("Detecting potential instability")
            print(s)
            return True, -500
        elif not np.isfinite(s).all():
            print("Infinite value detected..." + str(s))
            return True, -500

        if self.collarTermination:
            if not (self.collarFeature.contains(l0=bottomNeck, l1=bottomHead)[0] or
                                                 self.collarFeature.contains(l0=bottomHead, l1=topHead)[0]):
                return True, -500

        if self.sleeveEndTerm and self.limbProgress <= 0:
            limbInsertionError = pyutils.limbFeatureProgress(
                limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesL,
                                                  offset=np.array([0, -0.065, 0])), feature=self.sleeveLEndFeature)
            if limbInsertionError > 0:
                return True, -500

        if self.numSteps == 140:
            if self.saveStateOnReset and self.reset_number > 0:
                fname = self.state_save_directory + "sleeveL"
                print(fname)
                count = 0
                objfname_ix = fname + "%05d" % count
                charfname_ix = fname + "_char%05d" % count
                while os.path.isfile(objfname_ix + ".obj"):
                    count += 1
                    objfname_ix = fname + "%05d" % count
                    charfname_ix = fname + "_char%05d" % count
                print(objfname_ix)
                self.saveObjState(filename=objfname_ix)
                self.saveCharacterState(filename=charfname_ix)

        return False, 0

    def computeReward(self, tau):
        #compute and return reward at the current state
        wRFingertip2 = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)
        wLFingertip2 = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
        localLeftEfShoulder2 = self.robot_skeleton.bodynodes[8].to_local(wLFingertip2)  # left fingertip in left shoulder local frame

        reward_record = []

        self.prevTau = tau

        # reward for maintaining posture
        reward_upright = 0
        if self.uprightReward:
            reward_upright = max(-2.5, -abs(self.robot_skeleton.q[0]) - abs(self.robot_skeleton.q[1]))
            reward_record.append(reward_upright)

        reward_stableHead = 0
        if self.stableHeadReward:
            reward_stableHead = max(-1.2, -abs(self.robot_skeleton.q[19]) - abs(self.robot_skeleton.q[20]))
            reward_record.append(reward_stableHead)

        reward_elbow_flair = 0
        if self.elbowFlairReward:
            root = self.robot_skeleton.bodynodes[1].to_world(np.zeros(3))
            spine = self.robot_skeleton.bodynodes[2].to_world(np.zeros(3))
            elbow = self.robot_skeleton.bodynodes[self.elbowFlairNode].to_world(np.zeros(3))
            dist = pyutils.distToLine(p=elbow, l0=root, l1=spine)
            z = 0.5
            s = 16
            l = 0.2
            reward_elbow_flair = -(1 - (z * math.tanh(s * (dist - l)) + z))
            reward_record.append(reward_elbow_flair)
            # print("reward_elbow_flair: " + str(reward_elbow_flair))

        reward_limbprogress = 0
        if self.limbProgressReward:
            if self.simulateCloth:
                self.limbProgress = pyutils.limbFeatureProgress(
                    limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesL,
                                                      offset=np.array([0, -0.065, 0])), feature=self.sleeveLSeamFeature)
                reward_limbprogress = self.limbProgress
                if reward_limbprogress < 0:  # remove euclidean distance penalty before containment
                    reward_limbprogress = 0
            reward_record.append(reward_limbprogress)

        avgContactGeodesic = None
        if self.numSteps > 0 and self.simulateCloth:
            #contactInfo = pyutils.getContactIXGeoSide(sensorix=21, clothscene=self.clothScene, meshgraph=self.separatedMesh)

            contactInfo = pyutils.getContactIXGeoSideHighRes(nodeix=12, nodecom=self.robot_skeleton.bodynodes[12].com(), hapticSensorLocations=self.hapticSensorLocations, clothscene=self.clothScene, meshgraph=self.separatedMesh)
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

        clothDeformation = 0
        if self.simulateCloth is True:
            clothDeformation = self.clothScene.getMaxDeformationRatio(0)
            self.deformation = clothDeformation

        reward_clothdeformation = 0
        if self.deformationPenalty is True:
            #reward_clothdeformation = (math.tanh(9.24 - 0.5 * clothDeformation) - 1) / 2.0  # near 0 at 15, ramps up to -1.0 at ~22 and remains constant
            reward_clothdeformation = -(math.tanh(0.14*(clothDeformation-25)) + 1)/2.0 # near 0 at 15, ramps up to -1.0 at ~35 and remains constant
            reward_record.append(reward_clothdeformation)
        self.previousDeformationReward = reward_clothdeformation
        # force magnitude penalty
        reward_ctrl = -np.square(tau).sum()

        reward_oracleDisplacement = 0
        if self.oracleDisplacementReward:
            if np.linalg.norm(self.prevOracle) > 0 and self.localLeftEfShoulder1 is not None:
                # world_ef_displacement = wRFingertip2 - wRFingertip1
                relative_displacement = localLeftEfShoulder2 - self.localLeftEfShoulder1
                oracle0 = self.robot_skeleton.bodynodes[3].to_local(wLFingertip2 + self.prevOracle) - localLeftEfShoulder2
                # oracle0 = oracle0/np.linalg.norm(oracle0)
                reward_oracleDisplacement += relative_displacement.dot(oracle0)
            reward_record.append(reward_oracleDisplacement)

        reward_restPose = 0
        if self.restPoseReward:
            if self.restPose is not None:
                z = 0.5  # half the max magnitude (e.g. 0.5 -> [0,1])
                s = 1.0  # steepness (higher is steeper)
                l = 4.2  # translation
                dist = np.linalg.norm(self.robot_skeleton.q - self.restPose)
                reward_restPose = -(z * math.tanh(s * (dist - l)) + z)
                # print("distance: " + str(dist) + " -> " + str(reward_restPose))
            reward_record.append(reward_restPose)

        # update the reward data storage
        self.rewardsData.update(rewards=reward_record)

        self.reward = reward_ctrl * 0 \
                      + reward_upright * self.uprightRewardWeight \
                      + reward_stableHead * self.stableHeadRewardWeight \
                      + reward_limbprogress * self.limbProgressRewardWeight \
                      + reward_contactGeo * self.contactGeoRewardWeight \
                      + reward_clothdeformation * self.deformationPenaltyWeight \
                      + reward_oracleDisplacement * self.oracleDisplacementRewardWeight \
                      + reward_elbow_flair * self.elbowFlairRewardWeight \
                      + reward_restPose * self.restPoseRewardWeight

        return self.reward

    def _get_obs(self):
        f_size = 3*self.totalSensors
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

        if self.featureInObs:
            centroid = self.sleeveLMidFeature.plane.org
            efL = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
            disp = centroid-efL
            obs = np.concatenate([obs, centroid, disp]).ravel()

        if self.oracleInObs:
            oracle = np.zeros(3)
            if self.reset_number == 0:
                a=0 #nothing
            elif self.limbProgress > 0:
                oracle = self.sleeveLSeamFeature.plane.normal
            else:
                #minContactGeodesic, minGeoVix, _side = pyutils.getMinContactGeodesic(sensorix=21,clothscene=self.clothScene,meshgraph=self.separatedMesh,returnOnlyGeo=False)
                minContactGeodesic, minGeoVix, _side = pyutils.getMinContactGeodesicHighRes(nodeix=12, nodecom=self.robot_skeleton.bodynodes[12].com(), hapticSensorLocations=self.hapticSensorLocations, clothscene=self.clothScene, meshgraph=self.separatedMesh, returnOnlyGeo=False)
                if minGeoVix is None:
                    '''
                    #oracle points to the garment when ef not in contact
                    efL = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
                    #closeVert = self.clothScene.getCloseVertex(p=efR)
                    #target = self.clothScene.getVertexPos(vid=closeVert)
                    centroid = self.CP2Feature.plane.org
                    target = centroid
                    vec = target - efL
                    oracle = vec/np.linalg.norm(vec)
                    '''

                    # new: oracle points to the tuck triangle centroid when not in contact with cloth
                    target = np.zeros(3)
                    for c in self.previousContainmentTriangle:
                        target += c
                    target /= 3.0
                    efL = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
                    vec = target - efL
                    oracle = vec / np.linalg.norm(vec)
                else:
                    vixSide = 0
                    if _side:
                        vixSide = 1
                    if minGeoVix >= 0:
                        oracle = self.separatedMesh.geoVectorAt(minGeoVix, side=vixSide)
            self.prevOracle = oracle
            obs = np.concatenate([obs, oracle]).ravel()

        if self.contactIDInObs:
            HSIDs = self.clothScene.getHapticSensorContactIDs()
            obs = np.concatenate([obs, HSIDs]).ravel()

        return obs

    def additionalResets(self):
        #do any additional resetting here

        if self.simulateCloth:
            self.sleeveLSeamFeature.fitPlane()
            self.sleeveLEndFeature.fitPlane()
            self.sleeveLMidFeature.fitPlane()
            self.gripFeatureL.fitPlane()
            self.gripFeatureR.fitPlane()
            #ensure relative correctness of normals
            CP2_CP1 = self.sleeveLEndFeature.plane.org - self.sleeveLMidFeature.plane.org
            CP2_CP0 = self.sleeveLSeamFeature.plane.org - self.sleeveLMidFeature.plane.org

            # if CP2 normal is not facing the sleeve end invert it
            if CP2_CP1.dot(self.sleeveLMidFeature.plane.normal) < 0:
                self.sleeveLMidFeature.plane.normal *= -1.0

            # if CP1 normal is facing the sleeve middle invert it
            if CP2_CP1.dot(self.sleeveLEndFeature.plane.normal) < 0:
                self.sleeveLEndFeature.plane.normal *= -1.0

            # if CP0 normal is not facing sleeve middle invert it
            if CP2_CP0.dot(self.sleeveLSeamFeature.plane.normal) > 0:
                self.sleeveLSeamFeature.plane.normal *= -1.0

        if self.reset_number == 0 and self.simulateCloth:
            self.separatedMesh.initSeparatedMeshGraph()
            self.separatedMesh.updateWeights()
            # TODO: compute geodesic depends on cloth state! Deterministic initial condition required!
            # option: maybe compute this before setting the cloth state at all in initialization?
            self.separatedMesh.computeGeodesic(feature=self.sleeveLMidFeature, oneSided=True, side=0, normalSide=0)

        #self.clothScene.translateCloth(0, np.array([0.05, 0.025, 0]))
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-0.01, high=0.01, size=self.robot_skeleton.ndofs)
        #qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        '''qpos = np.array(
            [-0.0483053659505, 0.0321213273351, 0.0173036909392, 0.00486290205677, -0.00284350018845, -0.634602301004,
             -0.359172622713, 0.0792754054027, 2.66867203095, 0.00489456931428, 0.000476966442889, 0.0234663491334,
             -0.0254520098678, 0.172782859361, -1.31351102137, 0.702315566312, 1.73993331669, -0.0422811572637,
             0.586669332152, -0.0122329947565, 0.00179736869435, -8.0625896949e-05])
        self.set_state(qpos, qvel)'''

        self.loadCharacterState(filename="characterState_startarm2")
        qpos = np.array(self.robot_skeleton.q)

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
            #resetStateNumber = 7 #best in the rtuck set?
            #resetStateNumber = 0 #best in the triangle_rtuck set?
            #resetStateNumber = 2
            #resetStateNumber = self.reset_number%self.resetDistributionSize
            #print("resetStateNumber: " + str(resetStateNumber))
            charfname_ix = self.resetDistributionPrefix + "_char%05d" % resetStateNumber
            #print(charfname_ix)
            self.clothScene.setResetState(cid=0, index=resetStateNumber)
            self.loadCharacterState(filename=charfname_ix)
            #self.restPose = np.array(self.robot_skeleton.q)
            #self.set_state(qpos, qvel)
        else:
            self.set_state(qpos, qvel)

        #self.restPose = qpos


        if self.handleNode is not None:
            self.handleNode.clearHandles()
            self.handleNode.addVertices(verts=self.targetGripVerticesR)
            self.handleNode.setOrgToCentroid()
            if self.updateHandleNodeFrom >= 0:
                self.handleNode.setTransform(self.robot_skeleton.bodynodes[self.updateHandleNodeFrom].T)
            self.handleNode.recomputeOffsets()

        if self.simulateCloth:
            self.sleeveLSeamFeature.fitPlane()
            self.sleeveLEndFeature.fitPlane()
            self.sleeveLMidFeature.fitPlane()
            self.collarFeature.fitPlane()
            self.gripFeatureL.fitPlane()
            self.gripFeatureR.fitPlane()

        if self.limbProgressReward:
            self.limbProgress = pyutils.limbFeatureProgress(limb=pyutils.limbFromNodeSequence(self.robot_skeleton, nodes=self.limbNodesL,offset=np.array([0,-0.065,0])), feature=self.sleeveLSeamFeature)

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

        if self.sleeveLSeamFeature is not None:
            self.sleeveLSeamFeature.drawProjectionPoly(renderNormal=True, renderBasis=False)
        if self.sleeveLEndFeature is not None:
            self.sleeveLEndFeature.drawProjectionPoly(renderNormal=True, renderBasis=False)
        if self.sleeveLMidFeature is not None:
            self.sleeveLMidFeature.drawProjectionPoly(renderNormal=True, renderBasis=False)
        if self.collarFeature is not None:
            self.collarFeature.drawProjectionPoly(renderNormal=False, renderBasis=False)
        self.gripFeatureL.drawProjectionPoly(renderNormal=False, renderBasis=False)
        self.gripFeatureR.drawProjectionPoly(renderNormal=False, renderBasis=False, fillColor=[0.0,1.0,0.0])

        # render geodesic
        if False:
            for v in range(self.clothScene.getNumVertices()):
                side1geo = self.separatedMesh.nodes[v + self.separatedMesh.numv].geodesic
                side0geo = self.separatedMesh.nodes[v].geodesic

                pos = self.clothScene.getVertexPos(vid=v)
                norm = self.clothScene.getVertNormal(vid=v)
                renderUtils.setColor(color=renderUtils.heatmapColor(minimum=0, maximum=self.separatedMesh.maxGeo, value=self.separatedMesh.maxGeo-side0geo))
                renderUtils.drawSphere(pos=pos-norm*0.01, rad=0.01)
                renderUtils.setColor(color=renderUtils.heatmapColor(minimum=0, maximum=self.separatedMesh.maxGeo, value=self.separatedMesh.maxGeo-side1geo))
                renderUtils.drawSphere(pos=pos + norm * 0.01, rad=0.01)

        efL = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
        renderUtils.drawArrow(p0=efL, p1=efL + self.prevOracle)

        #render high res haptic sensors
        renderUtils.setColor([0.0, 0, 0.8])
        for node,locations in enumerate(self.hapticSensorLocations):
            for p in locations:
                renderUtils.drawSphere(pos=self.robot_skeleton.bodynodes[node].to_world(p), rad=0.005)


        m_viewport = self.viewer.viewport
        # print(m_viewport)
        self.rewardsData.render(topLeft=[m_viewport[2] - 410, m_viewport[3] - 15],
                                dimensions=[400, -m_viewport[3] + 30])

        textHeight = 15
        textLines = 2

        if self.renderUI:
            renderUtils.setColor(color=[0.,0,0])
            self.clothScene.drawText(x=15., y=textLines * textHeight, text="Steps = " + str(self.numSteps), color=(0., 0, 0))
            textLines += 1
            self.clothScene.drawText(x=15., y=textLines*textHeight, text="Reward = " + str(self.reward), color=(0., 0, 0))
            textLines += 1
            self.clothScene.drawText(x=15., y=textLines * textHeight, text="Cumulative Reward = " + str(self.cumulativeReward), color=(0., 0, 0))
            textLines += 1
            if self.numSteps > 0:
                renderUtils.renderDofs(robot=self.robot_skeleton, restPose=None, renderRestPose=False)

            self.clothScene.drawText(x=450., y=self.viewer.viewport[3] - 24, text="Limb Progress: ", color=(0., 0, 0))
            renderUtils.drawProgressBar(topLeft=[600, self.viewer.viewport[3] - 12], h=16, w=60, progress=self.limbProgress, color=[0.0, 3.0, 0])
            self.clothScene.drawText(x=400., y=self.viewer.viewport[3] - 47, text="Deformation Penalty:           [~15, ~34]", color=(0., 0, 0))
            renderUtils.drawProgressBar(topLeft=[600, self.viewer.viewport[3] - 35], h=16, w=60, progress=-self.previousDeformationReward, color=[1.0, 0.0, 0])

    def computeHapticSensorLocations(self):
        ''' sensorConcentrations = np.ones(len(self.robot_skeleton.bodynodes))*100
        sensorConcentrations[4] = 150
        sensorConcentrations[5] = 150
        sensorConcentrations[8] = 150
        sensorConcentrations[9] = 150
        sensorConcentrations[6] = 250
        sensorConcentrations[7] = 350
        sensorConcentrations[11] = 250
        sensorConcentrations[12] = 350'''

        self.updateClothCollisionStructures()
        # test capsule projection
        activeCapsules = self.clothScene.getCollisionCapsuleInfo()
        collisionSpheresInfo = self.clothScene.getCollisionSpheresInfo()
        capsuleNodes = self.clothScene.collisionCapsuleBodynodes

        #initialize and empty list for all nodes
        self.hapticSensorLocations = []
        for i in range(len(self.robot_skeleton.bodynodes)):
            self.hapticSensorLocations.append([])

        for r in range(activeCapsules.shape[0]):
            for c in range(activeCapsules.shape[1]):
                if activeCapsules[r, c] > 0:
                    node = int(capsuleNodes[r,c])
                    print("computing node " + str(node) + " | " + str(self.robot_skeleton.bodynodes[node].name)+" haptic sensors")
                    c0 = collisionSpheresInfo[r * 9:r * 9 + 3]
                    r0 = collisionSpheresInfo[r * 9 + 3]
                    c1 = collisionSpheresInfo[c * 9:c * 9 + 3]
                    r1 = collisionSpheresInfo[c * 9 + 3]
                    #samplePoints = pyutils.getSurfaceSamples(sensorConcentrations[node], _p0=c0, _p1=c1, _r0=r0, _r1=r1, endCap0=True, endCap1=True)
                    #print("     concentration " + str(sensorConcentrations[node])+ " -> " +str(len(samplePoints)) + " sensors...")
                    samplePoints = pyutils.getSurfaceSamples(self.hapticSensorsPerNode[node], _p0=c0, _p1=c1, _r0=r0, _r1=r1, endCap0=True, endCap1=True, fixedSampleNumber=True)
                    print("     num samples " + str(self.hapticSensorsPerNode[node])+ " -> " +str(len(samplePoints)) + " sensors...")

                    for i in range(100):
                        #print("         relaxation " + str(i))
                        samplePoints = pyutils.relaxCapsulePoints(samplePoints, c0, c1, r0, r1)
                    localSensorLocations = []
                    for p in samplePoints:
                        localSensorLocations.append(self.robot_skeleton.bodynodes[node].to_local(p))
                    self.hapticSensorLocations[node] = localSensorLocations


    def saveHapticSensorLocations(self, filename):
        f = open(filename, 'w')

        for node,locations in enumerate(self.hapticSensorLocations):
            for p in locations:
                f.write("n " + str(node) + " p " + str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + "\n")
        f.close()

    def loadHapticSensorLocations(self, filename):
        # initialize and empty list for all nodes
        self.hapticSensorLocations = []
        self.hapticSensorsPerNode = []
        for i in range(len(self.robot_skeleton.bodynodes)):
            self.hapticSensorLocations.append([])
            self.hapticSensorsPerNode.append(0)

        f = open(filename, 'r')
        for ix, line in enumerate(f):
            words = line.split()

            node = int(words[1])
            offset = np.array([float(words[3]), float(words[4]), float(words[5])])

            self.hapticSensorLocations[node].append(offset)
            self.hapticSensorsPerNode[node] += 1

        print("Loaded haptic sensors per node: " + str(self.hapticSensorsPerNode))

        f.close()

    def updateClothCollisionStructures(self, capsules=False, hapticSensors=False):
        a = 0
        # collision spheres creation
        fingertip = np.array([0.0, -0.06, 0.0])
        z = np.array([0., 0, 0])
        cs0 = self.robot_skeleton.bodynodes[1].to_world(z)
        cs1 = self.robot_skeleton.bodynodes[2].to_world(z)
        cs2 = self.robot_skeleton.bodynodes[14].to_world(z)
        cs3 = self.robot_skeleton.bodynodes[14].to_world(np.array([0, 0.175, 0]))
        cs4 = self.robot_skeleton.bodynodes[4].to_world(z)
        cs5 = self.robot_skeleton.bodynodes[5].to_world(z)
        cs6 = self.robot_skeleton.bodynodes[6].to_world(z)
        cs7 = self.robot_skeleton.bodynodes[7].to_world(z)
        cs8 = self.robot_skeleton.bodynodes[7].to_world(fingertip)
        cs9 = self.robot_skeleton.bodynodes[9].to_world(z)
        cs10 = self.robot_skeleton.bodynodes[10].to_world(z)
        cs11 = self.robot_skeleton.bodynodes[11].to_world(z)
        cs12 = self.robot_skeleton.bodynodes[12].to_world(z)
        cs13 = self.robot_skeleton.bodynodes[12].to_world(fingertip)
        csVars0 = np.array([0.15, -1, -1, 0, 0, 0])
        csVars1 = np.array([0.07, -1, -1, 0, 0, 0])
        csVars2 = np.array([0.1, -1, -1, 0, 0, 0])
        csVars3 = np.array([0.1, -1, -1, 0, 0, 0])
        csVars4 = np.array([0.065, -1, -1, 0, 0, 0])
        csVars5 = np.array([0.05, -1, -1, 0, 0, 0])
        csVars6 = np.array([0.0365, -1, -1, 0, 0, 0])
        csVars7 = np.array([0.04, -1, -1, 0, 0, 0])
        csVars8 = np.array([0.046, -1, -1, 0, 0, 0])
        csVars9 = np.array([0.065, -1, -1, 0, 0, 0])
        csVars10 = np.array([0.05, -1, -1, 0, 0, 0])
        csVars11 = np.array([0.0365, -1, -1, 0, 0, 0])
        csVars12 = np.array([0.04, -1, -1, 0, 0, 0])
        csVars13 = np.array([0.046, -1, -1, 0, 0, 0])
        collisionSpheresInfo = np.concatenate(
            [cs0, csVars0, cs1, csVars1, cs2, csVars2, cs3, csVars3, cs4, csVars4, cs5, csVars5, cs6, csVars6, cs7,
             csVars7, cs8, csVars8, cs9, csVars9, cs10, csVars10, cs11, csVars11, cs12, csVars12, cs13,
             csVars13]).ravel()

        # inflate collision objects
        # for i in range(int(len(collisionSpheresInfo)/9)):
        #    collisionSpheresInfo[i*9 + 3] *= 1.15

        self.collisionSphereInfo = np.array(collisionSpheresInfo)
        # collisionSpheresInfo = np.concatenate([cs0, csVars0, cs1, csVars1]).ravel()
        if np.isnan(np.sum(
                collisionSpheresInfo)):  # this will keep nans from propagating into PhysX resulting in segfault on reset()
            return
        self.clothScene.setCollisionSpheresInfo(collisionSpheresInfo)

        if capsules is True:
            # collision capsules creation
            collisionCapsuleInfo = np.zeros((14, 14))
            collisionCapsuleInfo[0, 1] = 1
            collisionCapsuleInfo[1, 2] = 1
            collisionCapsuleInfo[1, 4] = 1
            collisionCapsuleInfo[1, 9] = 1
            collisionCapsuleInfo[2, 3] = 1
            collisionCapsuleInfo[4, 5] = 1
            collisionCapsuleInfo[5, 6] = 1
            collisionCapsuleInfo[6, 7] = 1
            collisionCapsuleInfo[7, 8] = 1
            collisionCapsuleInfo[9, 10] = 1
            collisionCapsuleInfo[10, 11] = 1
            collisionCapsuleInfo[11, 12] = 1
            collisionCapsuleInfo[12, 13] = 1
            collisionCapsuleBodynodes = -1 * np.ones((14, 14))
            collisionCapsuleBodynodes[0, 1] = 1
            collisionCapsuleBodynodes[1, 2] = 13
            collisionCapsuleBodynodes[1, 4] = 3
            collisionCapsuleBodynodes[1, 9] = 8
            collisionCapsuleBodynodes[2, 3] = 14
            collisionCapsuleBodynodes[4, 5] = 4
            collisionCapsuleBodynodes[5, 6] = 5
            collisionCapsuleBodynodes[6, 7] = 6
            collisionCapsuleBodynodes[7, 8] = 7
            collisionCapsuleBodynodes[9, 10] = 9
            collisionCapsuleBodynodes[10, 11] = 10
            collisionCapsuleBodynodes[11, 12] = 11
            collisionCapsuleBodynodes[12, 13] = 12
            self.clothScene.setCollisionCapsuleInfo(collisionCapsuleInfo, collisionCapsuleBodynodes)
            self.collisionCapsuleInfo = np.array(collisionCapsuleInfo)

        if hapticSensors is True:
            #TODO: update this to pull from computed locations
            # hapticSensorLocations = np.concatenate([cs0, LERP(cs0, cs1, 0.33), LERP(cs0, cs1, 0.66), cs1, LERP(cs1, cs2, 0.33), LERP(cs1, cs2, 0.66), cs2, LERP(cs2, cs3, 0.33), LERP(cs2, cs3, 0.66), cs3])
            # hapticSensorLocations = np.concatenate([cs0, LERP(cs0, cs1, 0.25), LERP(cs0, cs1, 0.5), LERP(cs0, cs1, 0.75), cs1, LERP(cs1, cs2, 0.25), LERP(cs1, cs2, 0.5), LERP(cs1, cs2, 0.75), cs2, LERP(cs2, cs3, 0.25), LERP(cs2, cs3, 0.5), LERP(cs2, cs3, 0.75), cs3])
            hapticSensorLocations = np.zeros(0)

            if len(self.hapticSensorLocations) > 0:
                #load haptic sensors from datastructure
                for i in range(len(self.hapticSensorLocations)):
                    for offset in self.hapticSensorLocations[i]:
                        #if(hapticSensorLocations is None):
                        #    hapticSensorLocations = np.array(self.robot_skeleton.bodynodes[i].to_world(offset))
                        #else:
                        hapticSensorLocations = np.concatenate([hapticSensorLocations, self.robot_skeleton.bodynodes[i].to_world(offset)])

                hapticSensorRadii = np.zeros(int(len(hapticSensorLocations)/3))

                self.clothScene.setHapticSensorLocations(hapticSensorLocations)
                self.clothScene.setHapticSensorRadii(hapticSensorRadii)