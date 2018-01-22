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

from scipy.optimize import minimize

import pickle

class Controller(object):
    def __init__(self, env, policyfilename=None, name=None, obs_subset=[]):
        self.env = env #needed to set env state variables on setup for use
        self.name = name
        prefix = os.path.dirname(os.path.abspath(__file__))
        prefix = os.path.join(prefix, '../../../../rllab/data/local/experiment/')
        if name is None:
            self.name = policyfilename
        self.policy = None
        if policyfilename is not None:
            self.policy = pickle.load(open(prefix+policyfilename + "/policy.pkl", "rb"))
        self.obs_subset = obs_subset #list of index,length tuples to slice obs for input

    def query(self, obs):
        obs_subset = np.array([])
        for s in self.obs_subset:
            obs_subset = np.concatenate([obs_subset, obs[s[0]:s[0]+s[1]]]).ravel()
        a, a_info = self.policy.get_action(obs_subset)
        return a

    def setup(self):
        print("base setup ... overwrite this for specific control requirements")
        #TODO: subclasses for setup requirements

    def update(self):
        print("default update")
        #TODO: subclasses update targets, etc...

    def transition(self):
        #return true when a controller detects task completion to transition to the next controller
        return False

class DropGripController(Controller):
    def __init__(self, env, policyfilename=None, name=None, obs_subset=[]):
        obs_subset = [(0,154), (163,9)]
        name = "DropGrip"
        policyfilename = "experiment_2018_01_02_dropgrip2"
        Controller.__init__(self, env, policyfilename, name, obs_subset)

    def setup(self):
        self.env.fingertip = np.array([0, -0.065, 0])
        a=0

    def update(self):
        self.env.leftTarget = pyutils.getVertCentroid(verts=self.env.targetGripVertices, clothscene=self.env.clothScene) + pyutils.getVertAvgNorm(verts=self.env.targetGripVertices, clothscene=self.env.clothScene)*0.03

    def transition(self):
        efL = self.env.robot_skeleton.bodynodes[12].to_world(np.array([0,-0.065,0]))
        dist = np.linalg.norm(self.env.leftTarget - efL)
        print("dist: " + str(dist))
        if dist < 0.035:
            return True
        return False

class RightTuckController(Controller):
    def __init__(self, env, policyfilename=None, name=None, obs_subset=[]):
        obs_subset = [(0,172)]
        #policyfilename = "experiment_2018_01_04_phaseinterpolate_toR3_cont"
        policyfilename = "experiment_2018_01_08_distribution_rightTuck_warm"
        name="Right Tuck"
        Controller.__init__(self, env, policyfilename, name, obs_subset)

    def setup(self):
        self.env.fingertip = np.array([0, -0.065, 0])
        #setup cloth handle
        self.env.updateHandleNodeFrom = 12
        if self.env.handleNode is not None:
            self.env.handleNode.clearHandles()
        self.env.handleNode = HandleNode(self.env.clothScene, org=np.array([0.05, 0.034, -0.975]))
        self.env.handleNode.addVertices(verts=self.env.targetGripVertices)
        self.env.handleNode.setOrgToCentroid()
        self.env.handleNode.setTransform(self.env.robot_skeleton.bodynodes[self.env.updateHandleNodeFrom].T)
        self.env.handleNode.recomputeOffsets()
        #setup ef targets
        self.rightTarget = np.array([-0.09569568,  0.14657749, -0.17376665])
        self.leftTarget = np.array([-0.07484753,  0.21876009, -0.26294776])
        self.env.contactSensorIX = None
        a=0

    def update(self):
        if self.env.handleNode is not None:
            if self.env.updateHandleNodeFrom >= 0:
                self.env.handleNode.setTransform(self.env.robot_skeleton.bodynodes[self.env.updateHandleNodeFrom].T)
            self.env.handleNode.step()
        a=0

    def transition(self):
        efR = self.env.robot_skeleton.bodynodes[7].to_world(self.env.fingertip)
        efL = self.env.robot_skeleton.bodynodes[12].to_world(self.env.fingertip)
        shoulderR0 = self.env.robot_skeleton.bodynodes[4].to_world(np.zeros(3))
        shoulderR1 = self.env.robot_skeleton.bodynodes[3].to_world(np.zeros(3))
        shoulderR = (shoulderR0+shoulderR1)/2.0
        v0 = efR-shoulderR
        v1 = efL-shoulderR
        v0Len = np.linalg.norm(v0)
        v1Len = np.linalg.norm(v1)
        v0 = v0/v0Len
        v1 = v1/v1Len
        CIDS = self.env.clothScene.getHapticSensorContactIDs()
        print("v0.dot(v1): " + str(v0.dot(v1)) + " | v1Len > v2Len: " + str(v1Len > v0Len) + " | CID: " + str(CIDS[12]))
        if v0.dot(v1) > 0.9:
            if v1Len > v0Len+0.05:
                if CIDS[12] >= 0.8:
                    return True
        return False

class LeftTuckController(Controller):
    def __init__(self, env, policyfilename=None, name=None, obs_subset=[]):
        obs_subset = [(0,172)]
        policyfilename = "experiment_2018_01_04_phaseinterpolate_toL_cont"
        name="Left Tuck"
        Controller.__init__(self, env, policyfilename, name, obs_subset)

    def setup(self):
        self.env.contactSensorIX = None
        self.env.fingertip = np.array([0, -0.065, 0])
        #setup cloth handle
        self.env.updateHandleNodeFrom = 7
        if self.env.handleNode is not None:
            self.env.handleNode.clearHandles()
        self.env.handleNode = HandleNode(self.env.clothScene, org=np.array([0.05, 0.034, -0.975]))
        self.env.handleNode.addVertices(verts=self.env.targetGripVertices)
        self.env.handleNode.setOrgToCentroid()
        self.env.handleNode.setTransform(self.env.robot_skeleton.bodynodes[self.env.updateHandleNodeFrom].T)
        self.env.handleNode.recomputeOffsets()
        #setup ef targets
        self.rightTarget = np.array([ 0.07219914,  0.02462782, -0.37559271])
        self.leftTarget = np.array([ 0.06795917, -0.02272099, -0.12309984])
        #self.env.contactSensorIX = 21
        a=0

    def update(self):
        if self.env.handleNode is not None:
            if self.env.updateHandleNodeFrom >= 0:
                self.env.handleNode.setTransform(self.env.robot_skeleton.bodynodes[self.env.updateHandleNodeFrom].T)
            self.env.handleNode.step()
        a=0

class MatchGripController(Controller):
    def __init__(self, env, policyfilename=None, name=None, obs_subset=[]):
        obs_subset = [(0,172)]
        #policyfilename = "experiment_2018_01_04_phaseinterpolate_matchgrip3_cont"
        policyfilename = "experiment_2018_01_14_matchgrip_dist_lowpose"
        name="Match Grip"
        Controller.__init__(self, env, policyfilename, name, obs_subset)

    def setup(self):
        self.env.fingertip = np.array([0, -0.095, 0])
        # setup cloth handle
        self.env.updateHandleNodeFrom = 12
        if self.env.handleNode is not None:
            self.env.handleNode.clearHandles()
        self.env.handleNode = HandleNode(self.env.clothScene, org=np.array([0.05, 0.034, -0.975]))
        self.env.handleNode.addVertices(verts=self.env.targetGripVertices)
        self.env.handleNode.setOrgToCentroid()
        self.env.handleNode.setTransform(self.env.robot_skeleton.bodynodes[self.env.updateHandleNodeFrom].T)
        self.env.handleNode.recomputeOffsets()
        #setup ef targets
        self.env.leftTarget = np.array([ 0.04543329, -0.1115875,  -0.34433692])
        a=0

    def update(self):
        self.env.rightTarget = pyutils.getVertCentroid(verts=self.env.targetGripVertices, clothscene=self.env.clothScene) + pyutils.getVertAvgNorm(verts=self.env.targetGripVertices, clothscene=self.env.clothScene)*0.03
        if self.env.handleNode is not None:
            if self.env.updateHandleNodeFrom >= 0:
                self.env.handleNode.setTransform(self.env.robot_skeleton.bodynodes[self.env.updateHandleNodeFrom].T)
            self.env.handleNode.step()
        a=0

class RightSleeveController(Controller):
    def __init__(self, env, policyfilename=None, name=None, obs_subset=[]):
        obs_subset = [(0,132), (172, 9), (132, 22)]
        #policyfilename = "experiment_2017_12_12_1sdSleeve_progressfocus_cont2"
        policyfilename = "experiment_2018_01_09_tshirtR_dist_warm"
        name="Right Sleeve"
        Controller.__init__(self, env, policyfilename, name, obs_subset)

    def setup(self):
        self.env.fingertip = np.array([0, -0.065, 0])
        #setup cloth handle
        self.env.updateHandleNodeFrom = 12
        if self.env.handleNode is not None:
            self.env.handleNode.clearHandles()
        self.env.handleNode = HandleNode(self.env.clothScene, org=np.array([0.05, 0.034, -0.975]))
        self.env.handleNode.addVertices(verts=self.env.targetGripVertices)
        self.env.handleNode.setOrgToCentroid()
        self.env.handleNode.setTransform(self.env.robot_skeleton.bodynodes[self.env.updateHandleNodeFrom].T)
        self.env.handleNode.recomputeOffsets()

        #geodesic
        self.env.separatedMesh.initSeparatedMeshGraph()
        self.env.separatedMesh.updateWeights()
        self.env.separatedMesh.computeGeodesic(feature=self.env.sleeveRMidFeature, oneSided=True, side=0, normalSide=0)

        #feature/oracle setup
        self.env.focusFeature = self.env.sleeveRMidFeature  # if set, this feature centroid is used to get the "feature" obs
        self.env.focusFeatureNode = 7  # if set, this body node is used to fill feature displacement obs
        self.env.progressFeature = self.env.sleeveRSeamFeature  # if set, this feature is used to fill oracle normal and check arm progress
        self.env.contactSensorIX = 12

        a=0

    def update(self):
        if self.env.handleNode is not None:
            if self.env.updateHandleNodeFrom >= 0:
                self.env.handleNode.setTransform(self.env.robot_skeleton.bodynodes[self.env.updateHandleNodeFrom].T)
            self.env.handleNode.step()
        #limb progress
        self.env.limbProgress = pyutils.limbFeatureProgress(limb=pyutils.limbFromNodeSequence(self.env.robot_skeleton, nodes=self.env.limbNodesR,offset=np.array([0,-0.065,0])), feature=self.env.sleeveRSeamFeature)
        a=0

    def transition(self):
        if self.env.limbProgress > 0.6:
            return True
        return False

class LeftSleeveController(Controller):
    def __init__(self, env, policyfilename=None, name=None, obs_subset=[]):
        obs_subset = [(0,132), (172, 9), (132, 22)]
        #policyfilename = "experiment_2017_12_12_1sdSleeve_progressfocus_cont2"
        policyfilename = "experiment_2017_12_08_2ndSleeve_cont"
        name="Left Sleeve"
        Controller.__init__(self, env, policyfilename, name, obs_subset)

    def setup(self):
        self.env.fingertip = np.array([0, -0.065, 0])
        #setup cloth handle
        self.env.updateHandleNodeFrom = 7
        if self.env.handleNode is not None:
            self.env.handleNode.clearHandles()
        self.env.handleNode = HandleNode(self.env.clothScene, org=np.array([0.05, 0.034, -0.975]))
        self.env.handleNode.addVertices(verts=self.env.targetGripVertices)
        self.env.handleNode.setOrgToCentroid()
        self.env.handleNode.setTransform(self.env.robot_skeleton.bodynodes[self.env.updateHandleNodeFrom].T)
        self.env.handleNode.recomputeOffsets()

        #geodesic
        self.env.separatedMesh.initSeparatedMeshGraph()
        self.env.separatedMesh.updateWeights()
        self.env.separatedMesh.computeGeodesic(feature=self.env.sleeveLMidFeature, oneSided=True, side=0, normalSide=0)

        #feature/oracle setup
        self.env.focusFeature = self.env.sleeveLMidFeature  # if set, this feature centroid is used to get the "feature" obs
        self.env.focusFeatureNode = 12  # if set, this body node is used to fill feature displacement obs
        self.env.progressFeature = self.env.sleeveLSeamFeature  # if set, this feature is used to fill oracle normal and check arm progress
        self.env.contactSensorIX = 21

        a=0

    def update(self):
        if self.env.handleNode is not None:
            if self.env.updateHandleNodeFrom >= 0:
                self.env.handleNode.setTransform(self.env.robot_skeleton.bodynodes[self.env.updateHandleNodeFrom].T)
            self.env.handleNode.step()
        #limb progress
        self.env.limbProgress = pyutils.limbFeatureProgress(limb=pyutils.limbFromNodeSequence(self.env.robot_skeleton, nodes=self.env.limbNodesL,offset=np.array([0,-0.065,0])), feature=self.env.sleeveLSeamFeature)
        a=0

    def transition(self):
        if self.env.limbProgress > 0.7:
            return True
        return False

class SPDController(Controller):
    def __init__(self, env, target=None):
        obs_subset = []
        policyfilename = None
        name = "SPD"
        self.target = target
        Controller.__init__(self, env, policyfilename, name, obs_subset)

        self.h = 0.02
        self.skel = env.robot_skeleton
        ndofs = self.skel.ndofs
        self.qhat = self.skel.q
        self.Kp = np.diagflat([400.0] * (ndofs))
        self.Kd = np.diagflat([40.0] * (ndofs))
        self.preoffset = 0.0

    def setup(self):
        #reset the target
        cur_q = np.array(self.skel.q)
        self.env.loadCharacterState(filename="characterState_regrip")
        self.env.restPose = np.array(self.skel.q)
        self.target = np.array(self.skel.q)
        self.env.robot_skeleton.set_positions(cur_q)
        a=0

    def update(self):
        if self.env.handleNode is not None:
            if self.env.updateHandleNodeFrom >= 0:
                self.env.handleNode.setTransform(self.env.robot_skeleton.bodynodes[self.env.updateHandleNodeFrom].T)
            self.env.handleNode.step()
        a=0

    def transition(self):
        pDist = np.linalg.norm(self.skel.q - self.env.restPose)
        print(pDist)
        if pDist < 0.1:
            return True
        return False

    def query(self, obs):
        #SPD
        self.qhat = self.target
        skel = self.skel
        p = -self.Kp.dot(skel.q + skel.dq * self.h - self.qhat)
        d = -self.Kd.dot(skel.dq)
        b = -skel.c + p + d + skel.constraint_forces()
        A = skel.M + self.Kd * self.h

        x = np.linalg.solve(A, b)

        #invM = np.linalg.inv(A)
        #x = invM.dot(b)
        tau = p + d - self.Kd.dot(x) * self.h
        return tau

class SPDIKController(Controller):
    def __init__(self, env, target=None):
        obs_subset = []
        policyfilename = None
        name = "SPDIK"
        self.target = target
        self.efL_target = None
        self.efR_target = None
        self.efL_start = None
        self.efR_start = None
        self.referencePose = None
        Controller.__init__(self, env, policyfilename, name, obs_subset)

        self.h = 0.02
        self.skel = env.robot_skeleton
        ndofs = self.skel.ndofs
        self.qhat = self.skel.q
        self.Kp = np.diagflat([400.0] * (ndofs))
        self.Kd = np.diagflat([40.0] * (ndofs))
        self.preoffset = 0.0

    def setup(self):
        #reset the target
        cur_q = np.array(self.skel.q)
        self.env.loadCharacterState(filename="characterState_regrip")
        self.env.restPose = np.array(self.skel.q)
        self.target = np.array(self.skel.q)
        self.referencePose = np.array(self.skel.q)
        self.efL_target = self.skel.bodynodes[12].to_world(self.env.fingertip)
        self.efR_target = self.skel.bodynodes[7].to_world(self.env.fingertip)
        self.env.robot_skeleton.set_positions(cur_q)
        self.efL_start = self.skel.bodynodes[12].to_world(self.env.fingertip)
        self.efR_start = self.skel.bodynodes[7].to_world(self.env.fingertip)
        a=0

    def f(self, x):
        self.skel.set_positions(x)

        lhsL = self.skel.bodynodes[12].to_world(self.env.fingertip)
        rhsL = self.efL_start + (self.efL_target - self.efL_start) * min(self.env.stepsSinceControlSwitch / 100.0, 1.0)
        self.env.leftTarget = rhsL

        lhsR = self.skel.bodynodes[7].to_world(self.env.fingertip)
        rhsR = self.efR_start + (self.efR_target - self.efR_start) * min(self.env.stepsSinceControlSwitch / 100.0, 1.0)
        self.env.rightTarget = rhsR

        return 0.5 * np.linalg.norm(lhsL - rhsL) ** 2 + 0.5 * np.linalg.norm(lhsR - rhsR) ** 2

    def g(self, x):
        self.skel.set_positions(x)

        lhsL = self.skel.bodynodes[12].to_world(self.env.fingertip)
        rhsL = self.efL_target
        JL = self.skel.bodynodes[12].linear_jacobian()
        gL = (lhsL - rhsL).dot(JL)

        lhsR = self.skel.bodynodes[7].to_world(self.env.fingertip)
        rhsR = self.efR_target
        JR = self.skel.bodynodes[7].linear_jacobian()
        gR = (lhsR - rhsR).dot(JR)

        return gR+gL

    def update(self):
        if self.env.handleNode is not None:
            if self.env.updateHandleNodeFrom >= 0:
                self.env.handleNode.setTransform(self.env.robot_skeleton.bodynodes[self.env.updateHandleNodeFrom].T)
            self.env.handleNode.step()

        c_q = np.array(self.skel.q)
        #TODO: IK to set self.target
        res = minimize(self.f,
                       x0=self.skel.positions(),
                       jac=self.g,
                       method="SLSQP")

        self.target = np.array(self.skel.q)
        self.target[:2] = self.referencePose[:2]
        self.env.restPose = np.array(self.skel.q)
        self.skel.set_positions(c_q)
        a=0

    def transition(self):
        pDist = np.linalg.norm(self.skel.q - self.env.restPose)
        #print(pDist)
        #if pDist < 0.1:
        #    return True
        return False

    def query(self, obs):
        #SPD
        self.qhat = self.target
        skel = self.skel
        p = -self.Kp.dot(skel.q + skel.dq * self.h - self.qhat)
        d = -self.Kd.dot(skel.dq)
        b = -skel.c + p + d + skel.constraint_forces()
        A = skel.M + self.Kd * self.h

        x = np.linalg.solve(A, b)

        #invM = np.linalg.inv(A)
        #x = invM.dot(b)
        tau = p + d - self.Kd.dot(x) * self.h
        return tau

class DartClothUpperBodyDataDrivenClothTshirtMasterEnv(DartClothUpperBodyDataDrivenClothBaseEnv, utils.EzPickle):
    def __init__(self):
        #feature flags
        rendering = True
        clothSimulation = True
        renderCloth = True

        #other flags
        self.collarTermination = True  # if true, rollout terminates when collar is off the head/neck
        self.collarTerminationCD = 10 #number of frames to ignore collar at the start of simulation (gives time for the cloth to drop)
        self.hapticsAware       = True  # if false, 0's for haptic input
        self.resetTime = 0
        self.save_state_on_control_switch = False #if true, the cloth and character state is saved when controllers are switched
        self.state_save_directory = "saved_control_states/"

        #other variables
        self.restPose = None
        self.localRightEfShoulder1 = None
        self.localLeftEfShoulder1 = None
        self.rightTarget = np.zeros(3)
        self.leftTarget = np.zeros(3)
        self.prevErrors = None #stores the errors taken from DART each iteration
        self.limbProgress = -1
        self.fingertip = np.array([0, -0.065, 0])

        self.actuatedDofs = np.arange(22)
        observation_size = len(self.actuatedDofs)*3 #q(sin,cos), dq #(0,66)
        observation_size += 66 #haptics                             #(66,66)
        observation_size += 22 #contact IDs                         #(132,22)
        observation_size += 9 #right target                         #(154,9)
        observation_size += 9 #left target                          #(163,9)
        observation_size += 6 #feature                              #(172,6)
        observation_size += 3 #oracle                               #(178,3)

        DartClothUpperBodyDataDrivenClothBaseEnv.__init__(self,
                                                          rendering=rendering,
                                                          screensize=(1080,720),
                                                          clothMeshFile="tshirt_m.obj",
                                                          clothScale=1.4,
                                                          obs_size=observation_size,
                                                          simulateCloth=clothSimulation)

        # clothing features
        self.collarVertices = [117, 115, 113, 900, 108, 197, 194, 8, 188, 5, 120]
        self.targetGripVertices = [570, 1041, 285, 1056, 435, 992, 50, 489, 787, 327, 362, 676, 887, 54, 55]
        self.sleeveRVerts = [2580, 2495, 2508, 2586, 2518, 2560, 2621, 2529, 2559, 2593, 272, 2561, 2658, 2582, 2666, 2575, 2584, 2625, 2616, 2453, 2500, 2598, 2466]
        self.sleeveRMidVerts = [2556, 2646, 2641, 2574, 2478, 2647, 2650, 269, 2630, 2528, 2607, 2662, 2581, 2458, 2516, 2499, 2555, 2644, 2482, 2653, 2507, 2648, 2573, 2601, 2645]
        self.sleeveREndVerts = [252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 10, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251]
        self.sleeveLVerts = [211, 2305, 2364, 2247, 2322, 2409, 2319, 2427, 2240, 2320, 2276, 2326, 2334, 2288, 2346, 2314, 2251, 2347, 2304, 2245, 2376, 2315]
        self.sleeveLMidVerts = [2379, 2357, 2293, 2272, 2253, 214, 2351, 2381, 2300, 2352, 2236, 2286, 2430, 2263, 2307, 2387, 2232, 2390, 2277, 2348, 2382, 2227, 2296, 2425]
        self.sleeveLEndVerts = [232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 9, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231]

        self.collarFeature = ClothFeature(verts=self.collarVertices, clothScene=self.clothScene)
        self.sleeveRSeamFeature = ClothFeature(verts=self.sleeveRVerts, clothScene=self.clothScene)
        self.sleeveRMidFeature = ClothFeature(verts=self.sleeveRMidVerts, clothScene=self.clothScene)
        self.sleeveREndFeature = ClothFeature(verts=self.sleeveREndVerts, clothScene=self.clothScene)
        self.sleeveLSeamFeature = ClothFeature(verts=self.sleeveLVerts, clothScene=self.clothScene)
        self.sleeveLMidFeature = ClothFeature(verts=self.sleeveLMidVerts, clothScene=self.clothScene)
        self.sleeveLEndFeature = ClothFeature(verts=self.sleeveLEndVerts, clothScene=self.clothScene)

        #variables for returning feature obs
        self.focusFeature = None        #if set, this feature centroid is used to get the "feature" obs
        self.focusFeatureNode = None    #if set, this body node is used to fill feature displacement obs
        self.progressFeature = None     #if set, this feature is used to fill oracle normal and check arm progress
        self.contactSensorIX = None     #if set, used to compute oracle

        self.simulateCloth = clothSimulation
        self.handleNode = None
        if not renderCloth:
            self.clothScene.renderClothFill = False
            self.clothScene.renderClothBoundary = False
            self.clothScene.renderClothWires = False

        #controller initialzation
        self.controllers = [
            DropGripController(self),
            RightTuckController(self),
            RightSleeveController(self),
            SPDController(self),
            #SPDIKController(self),
            #MatchGripController(self),
            LeftTuckController(self),
            LeftSleeveController(self)
        ]
        self.currentController = None
        self.stepsSinceControlSwitch = 0

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
        self.limbProgress = -1 #reset this in controller

        #update feature planes
        self.collarFeature.fitPlane()
        self.sleeveRSeamFeature.fitPlane()
        self.sleeveRMidFeature.fitPlane()
        self.sleeveREndFeature.fitPlane()
        self.sleeveLSeamFeature.fitPlane()
        self.sleeveLMidFeature.fitPlane()
        self.sleeveLEndFeature.fitPlane()

        self.additionalAction = np.zeros(22) #reset control
        #update controller specific variables and produce
        if self.currentController is not None:
            self.controllers[self.currentController].update()
            if self.controllers[self.currentController].transition():
                changed = self.currentController
                self.currentController = min(len(self.controllers)-1, self.currentController+1)
                changed = (changed != self.currentController)
                if changed:
                    self.controllers[self.currentController].setup()
                    self.controllers[self.currentController].update()
            obs = self._get_obs()
            self.additionalAction = self.controllers[self.currentController].query(obs)

        self.stepsSinceControlSwitch += 1
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
        elif self.collarTermination and self.simulateCloth and self.collarTerminationCD < self.numSteps:
            if not (self.collarFeature.contains(l0=bottomNeck, l1=bottomHead)[0] or
                        self.collarFeature.contains(l0=bottomHead, l1=topHead)[0]):
                print("collar term")
                return True, -500

        return False, 0

    def computeReward(self, tau):
        #compute and return reward at the current state
        #unnecessary for master control env... not meant for training
        return 0

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

        #haptics
        f = np.zeros(f_size)
        if self.simulateCloth and self.hapticsAware:
            f = self.clothScene.getHapticSensorObs()#get force from simulation
        obs = np.concatenate([obs, f]).ravel()

        #contactIDs
        HSIDs = self.clothScene.getHapticSensorContactIDs()
        obs = np.concatenate([obs, HSIDs]).ravel()

        #right ef target
        efR = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)
        obs = np.concatenate([obs, self.rightTarget, efR, self.rightTarget-efR]).ravel()

        #left ef target
        efL = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
        obs = np.concatenate([obs, self.leftTarget, efL, self.leftTarget-efL]).ravel()

        #feature
        if self.focusFeature is None or self.focusFeatureNode is None:
            obs = np.concatenate([obs, np.zeros(6)]).ravel()
        else:
            centroid = self.focusFeature.plane.org
            ef = self.robot_skeleton.bodynodes[self.focusFeatureNode].to_world(self.fingertip)
            disp = centroid - ef
            obs = np.concatenate([obs, centroid, disp]).ravel()

        #oracle
        oracle = np.zeros(3)
        if self.reset_number == 0:
            a = 0  # nothing
        elif self.limbProgress > 0:
            oracle = self.progressFeature.plane.normal
        else:
            minGeoVix = None
            minContactGeodesic = None
            _side = None
            if self.contactSensorIX is not None:
                minContactGeodesic, minGeoVix, _side = pyutils.getMinContactGeodesic(sensorix=self.contactSensorIX,
                                                                                     clothscene=self.clothScene,
                                                                                     meshgraph=self.separatedMesh,
                                                                                     returnOnlyGeo=False)
            if minGeoVix is None:
                if self.focusFeatureNode is not None:
                    # oracle points to the garment when ef not in contact
                    ef = self.robot_skeleton.bodynodes[self.focusFeatureNode].to_world(self.fingertip)
                    # closeVert = self.clothScene.getCloseVertex(p=efR)
                    # target = self.clothScene.getVertexPos(vid=closeVert)

                    centroid = self.focusFeature.plane.org

                    target = centroid
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

        return obs

    def additionalResets(self):
        self.resetTime = time.time()
        #do any additional resetting here
        #fingertip = np.array([0, -0.065, 0])
        self.currentController = 0
        if self.simulateCloth:
            up = np.array([0,1.0,0])
            varianceR = pyutils.rotateY(((random.random()-0.5)*2.0)*0.3)
            adjustR = pyutils.rotateY(0.2)
            R = self.clothScene.rotateTo(v1=np.array([0,0,1.0]), v2=up)
            self.clothScene.translateCloth(0, np.array([-0.01, 0.0255, 0]))
            self.clothScene.rotateCloth(cid=0, R=R)
            self.clothScene.rotateCloth(cid=0, R=adjustR)
            self.clothScene.rotateCloth(cid=0, R=varianceR)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-0.1, high=0.1, size=self.robot_skeleton.ndofs)
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)

        self.set_state(qpos, qvel)
        self.restPose = qpos

        self.controllers[self.currentController].setup()

        if self.simulateCloth:
            self.collarFeature.fitPlane()
            self.sleeveRSeamFeature.fitPlane()
            self.sleeveRMidFeature.fitPlane()
            self.sleeveREndFeature.fitPlane()
            self.sleeveLSeamFeature.fitPlane()
            self.sleeveLMidFeature.fitPlane()
            self.sleeveLEndFeature.fitPlane()
        if self.handleNode is not None:
            self.handleNode.clearHandles()
            self.handleNode = None

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


        #restPose rendering
        links = pyutils.getRobotLinks(self.robot_skeleton, pose=self.restPose)
        renderUtils.drawLines(lines=links)

        #render targets
        #fingertip = np.array([0,-0.065,0])

        efR = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)
        renderUtils.setColor(color=[1.0,0,0])
        renderUtils.drawSphere(pos=self.rightTarget,rad=0.02)
        renderUtils.drawLineStrip(points=[self.rightTarget, efR])

        efL = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
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

            if self.stepsSinceControlSwitch < 50 and len(self.controllers) > 0:
                label = self.controllers[self.currentController].name
                self.clothScene.drawText(x=self.viewer.viewport[2] / 2, y=self.viewer.viewport[3] - 60,
                                         text="Active Controller = " + str(label), color=(0., 0, 0))
