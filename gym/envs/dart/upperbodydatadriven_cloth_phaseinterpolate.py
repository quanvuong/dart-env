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

class OpennessMetricObject(object):
    '''a polygonal region from shoulders to grip feature'''
    def __init__(self, clothScene, nL, nM, nR, vR, vL):
        self.clothScene = clothScene
        self.nL = nL
        self.nM = nM
        self.nR = nR
        self.vL = vL
        self.vR = vR
        self.area = 0
        self.sortedArea = 0 #area of sorted polygon
        self.triangleArea = 0 #area method 2: sum of 3 triangle areas
        self.plane = Plane()
        self.projectedPoints = []
        self.points = []

    def update(self):
        npL = self.nL.to_world(np.zeros(3))
        npM = self.nM.to_world(np.zeros(3))
        npR = self.nR.to_world(np.zeros(3))
        vpL = self.clothScene.getVertexPos(vid=self.vL)
        vpR = self.clothScene.getVertexPos(vid=self.vR)
        self.points = [npL, npM, npR, vpR, vpL]
        self.plane = fitPlane(points=[npL, npR, vpR, vpL], normhint=self.plane.normal, basis1hint=self.plane.basis1, basis2hint=self.plane.basis2)
        #project points to plane
        points = [npL, npR, vpR, vpL]
        self.projectedPoints = []
        points2D = []
        sortedPoints2D = []
        for p in points:
            self.projectedPoints.append(self.plane.projectPoint(p))
            points2D.append(self.plane.get2D(self.projectedPoints[-1],project=False))
        #now get the area
        self.area = 0
        prev = points2D[-1]
        sorted = True
        for ix,p in enumerate(points2D):
            det = p[0]*prev[1] + p[1]*(-prev[0])
            if det > 0:
                sorted = False
                #print("     "+str(ix)+": clockwise")
            self.area -= ((prev[0] + p[0]) * (prev[1] - p[1]))
            prev = p

        #print("sorted: " + str(sorted))

        sortedPoints2D.append(points2D[0])
        sortedPoints2D.append(points2D[1])
        if sorted:
            sortedPoints2D.append(points2D[2])
            sortedPoints2D.append(points2D[3])
        else:
            sortedPoints2D.append(points2D[3])
            sortedPoints2D.append(points2D[2])

        self.sortedArea = 0
        prev = sortedPoints2D[-1]
        for p in sortedPoints2D:
            det = p[0] * prev[1] + p[1] * (-prev[0])
            if det > 0:
                print("still wrong")
            self.sortedArea -= ((prev[0]+p[0]) * (prev[1]-p[1]))
            prev = p

        #calculate triangle based area
        v0 = npL-vpL
        v1 = npM-vpL
        v2 = npR-vpR
        v3 = npM-vpR
        self.triangleArea = 0.5 * (np.linalg.norm(np.cross(v0,v1)) + np.linalg.norm(np.cross(v2,v3)) + np.linalg.norm(np.cross(-v1,-v3)))

    def drawRegion(self, fillColor=[0.8,0.2,0.2]):
        renderUtils.setColor([0.0, 0., 0.])
        for p in self.projectedPoints:
            renderUtils.drawSphere(p, rad=0.0025)

        renderUtils.drawPolygon(points=self.projectedPoints, filled=True, fillColor=fillColor)

        #renderUtils.drawArrow(p0=self.plane.org, p1=self.plane.org+self.plane.normal)
        #self.plane.draw()

        renderUtils.setColor([0.0, 0., 1.])
        if len(self.points) > 0:
            renderUtils.drawTriangle(p0=self.points[0], p1=self.points[1], p2=self.points[4])
            renderUtils.drawTriangle(p0=self.points[1], p1=self.points[3], p2=self.points[4])
            renderUtils.drawTriangle(p0=self.points[1], p1=self.points[2], p2=self.points[3])

class DartClothUpperBodyDataDrivenClothPhaseInterpolateEnv(DartClothUpperBodyDataDrivenClothBaseEnv, utils.EzPickle):
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
        self.stableHeadReward           = True  # if True, rewarded for - head/torso angle
        self.elbowFlairReward           = False
        self.deformationPenalty         = True
        self.restPoseReward             = True
        self.rightTargetReward          = False
        self.leftTargetReward           = False
        self.contactSurfaceReward       = True #reward for end effector touching inside cloth
        self.opennessReward             = False #use openness metric for reward
        self.containmentReward          = False #use containment percentage as reward
        self.efContainmentReward        = False #use end effector containment as reward (binary)
        self.elbowHandElevationReward   = False #penalize elbow above the hand (tuck elbow down)
        self.triangleContainmentReward  = True #active ef rewarded for intersection with triangle from shoulders to passive ef. Also penalized for distance to triangle
        self.triangleAlignmentReward    = True #dot product reward between triangle normal and character torso vector
        self.sleeveForwardReward        = True # penalize sleeve for being behind the character
        self.efForwardReward            = True # penalize ef for being behind the character

        #other flags
        self.collarTermination = True  # if true, rollout terminates when collar is off the head/neck
        self.collarTerminationCD = 0 #number of frames to ignore collar at the start of simulation (gives time for the cloth to drop)
        self.hapticsAware       = True  # if false, 0's for haptic input
        self.loadTargetsFromROMPositions = False
        self.resetPoseFromROMPoints = False
        self.resetTime = 0
        self.resetStateFromDistribution = True
        #self.resetDistributionPrefix = "saved_control_states_old/DropGrip"
        self.resetDistributionPrefix = "saved_control_states/enter_seq_rtuck"
        self.resetDistributionSize = 20
        self.testingCapsuleRelaxation = False

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
        self.previousEfContainment = 0
        self.fingertip = np.array([0, -0.085, 0])
        self.previousContainmentTriangle = [np.zeros(3),np.zeros(3),np.zeros(3)]
        self.characterFrontBackPlane = Plane()
        self.samplePoints = [] #store sample points for debugging purposes

        #grid sampling / local points
        self.localPointGrid = None #set in reset
        #front body only
        self.clothContainmentHeuristicVerts = [960, 690, 1012, 776, 821, 930, 1029, 971, 520, 951, 946, 1000, 978, 1203, 1159, 863, 443, 1127, 992, 609]
        #front and side body
        #self.clothContainmentHeuristicVerts = [960, 690, 1012, 776, 821, 930, 1029, 971, 520, 951, 946, 1000, 978, 1203, 1159, 863, 443, 1127, 992, 609, 1706, 2138, 1715, 1629, 87, 81, 1431, 71, 68]
        #front, back and side body
        #self.clothContainmentHeuristicVerts = [960, 690, 1012, 776, 821, 930, 1029, 971, 520, 951, 946, 1000, 978, 1203, 1159, 863, 443, 1127, 992, 609, 1706, 2138, 1715, 1629, 87, 81, 1431, 71, 68, 1641, 1967, 2097, 1601, 2056, 1895, 2193, 1941, 1946, 1553, 2042, 2176]
        #front,side,back body and shoulders/collar
        #self.clothContainmentHeuristicVerts = [960, 690, 1012, 776, 821, 930, 1029, 971, 520, 951, 946, 1000, 978, 1203, 1159, 863, 443, 1127, 992, 609, 1706, 2138, 1715, 1629, 87, 81, 1431, 71, 68, 1641, 1967, 2097, 1601, 2056, 1895, 2193, 1941, 1946, 1553, 2042, 2176, 2667, 480, 1133, 342, 1167, 832, 2587, 124, 1447, 1289, 2457, 1793, 1259]
        self.clothContainmentHeuristic = None #set in reset
        self.clothContainmentRecord = []
        self.clothContainmentMethod = 2
        self.clothOpennessMetric = None #also set in reset (off)
        self.graphClothOpenness = False
        self.clothOpennessGraph = None
        self.graphContainmentHeuristic = False
        self.containmentHeuristicGraph = None
        self.saveGraphsOnReset = False

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
                                                          screensize=(1080,720),
                                                          clothMeshFile="tshirt_m.obj",
                                                          clothMeshStateFile = "endDropGrip1.obj",
                                                          clothScale=1.4,
                                                          obs_size=observation_size,
                                                          simulateCloth=clothSimulation)

        #define pose and obj files for reset
        '''self.resetStateFileNames = ["endDropGrip1"] #each state name 'n' should refer to both a character state file: " characterState_'n' " and the cloth state file: " 'n'.obj ".
        #load reset poses from file
        for name in self.resetStateFileNames:
            self.clothScene.addResetStateFrom(filename=name+".obj")'''

        # clothing features
        self.collarVertices = [117, 115, 113, 900, 108, 197, 194, 8, 188, 5, 120]
        self.targetGripVerticesL = [46, 437, 955, 1185, 47, 285, 711, 677, 48, 905, 1041, 49, 741, 889, 45]
        self.targetGripVerticesR = [905, 1041, 49, 435, 50, 570, 992, 1056, 51, 676, 283, 52, 489, 892, 362, 53]
        #self.targetGripVertices = [570, 1041, 285, 1056, 435, 992, 50, 489, 787, 327, 362, 676, 887, 54, 55]
        self.waistVertices = [0, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 1, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170]
        self.sleeveRMidVerts = [2556, 2646, 2641, 2574, 2478, 2647, 2650, 269, 2630, 2528, 2607, 2662, 2581, 2458, 2516, 2499, 2555, 2644, 2482, 2653, 2507, 2648, 2573, 2601, 2645]
        self.collarFeature = ClothFeature(verts=self.collarVertices, clothScene=self.clothScene)
        self.gripFeatureL = ClothFeature(verts=self.targetGripVerticesL, clothScene=self.clothScene, b1spanverts=[889,1041], b2spanverts=[47,677])
        self.gripFeatureR = ClothFeature(verts=self.targetGripVerticesR, clothScene=self.clothScene, b1spanverts=[362,889], b2spanverts=[51,992])
        self.waistFeature = ClothFeature(verts=self.waistVertices, clothScene=self.clothScene)
        self.CP2Feature = ClothFeature(verts=self.sleeveRMidVerts, clothScene=self.clothScene)

        self.simulateCloth = clothSimulation
        if self.simulateCloth:
            self.handleNode = HandleNode(self.clothScene, org=np.array([0.05, 0.034, -0.975]))

        if not renderCloth:
            self.clothScene.renderClothFill = False
            self.clothScene.renderClothBoundary = False
            self.clothScene.renderClothWires = False

        self.state_save_directory = "saved_control_states/"
        self.saveStateOnReset = False

    def _getFile(self):
        return __file__

    def updateBeforeSimulation(self):
        #any pre-sim updates should happen here
        wRFingertip1 = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)
        wLFingertip1 = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
        self.localRightEfShoulder1 = self.robot_skeleton.bodynodes[3].to_local(wRFingertip1)  # right fingertip in right shoulder local frame
        self.localLeftEfShoulder1 = self.robot_skeleton.bodynodes[8].to_local(wLFingertip1)  # left fingertip in left shoulder local frame
        #self.leftTarget = pyutils.getVertCentroid(verts=self.targetGripVertices, clothscene=self.clothScene) + pyutils.getVertAvgNorm(verts=self.targetGripVertices, clothscene=self.clothScene)*0.03

        if self.collarFeature is not None:
            self.collarFeature.fitPlane()
        self.gripFeatureL.fitPlane()
        self.gripFeatureR.fitPlane()
        self.waistFeature.fitPlane()
        self.CP2Feature.fitPlane()
        if self.opennessReward:
            self.waistFeature.computeArea()

        if self.clothOpennessMetric is not None:
            self.clothOpennessMetric.update()

        if self.graphClothOpenness and self.clothOpennessGraph is not None:
            self.clothOpennessGraph.addToLinePlot([self.waistFeature.area, self.clothOpennessMetric.sortedArea, self.clothOpennessMetric.triangleArea ])

        efcontainment = 0
        if self.clothContainmentHeuristic is not None:
            efcontainment = self.clothContainmentHeuristic.contained(point=self.robot_skeleton.bodynodes[7].to_world(np.array([0,-0.075, 0])), method=self.clothContainmentMethod)
            if self.localPointGrid is not None:
                points = self.localPointGrid.getWorldPoints()
                self.clothContainmentRecord = self.clothContainmentHeuristic.setContained(points=points, method=self.clothContainmentMethod)

        if self.graphContainmentHeuristic and len(self.clothContainmentRecord) > 0 and self.containmentHeuristicGraph is not None:
            containedPercent = 0
            for r in self.clothContainmentRecord:
                if r:
                    containedPercent += 1
            containedPercent /= len(self.clothContainmentRecord)
            #end effector containment
            self.containmentHeuristicGraph.addToLinePlot([containedPercent, efcontainment])

        if self.triangleContainmentReward:
            self.previousContainmentTriangle = [
                self.robot_skeleton.bodynodes[9].to_world(np.zeros(3)),
                self.robot_skeleton.bodynodes[4].to_world(np.zeros(3)),
                self.gripFeatureL.plane.org
                #self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
            ]

        # update handle nodes
        if self.handleNode is not None:
            if self.updateHandleNodeFrom >= 0:
                self.handleNode.setTransform(self.robot_skeleton.bodynodes[self.updateHandleNodeFrom].T)
            self.handleNode.step()

        spineCenter = self.robot_skeleton.bodynodes[2].to_world(np.zeros(3))
        characterForward = self.robot_skeleton.bodynodes[2].to_world(np.array([0.0, 0.0, -1.0])) - spineCenter
        self.characterFrontBackPlane = Plane(org=spineCenter, normal=characterForward)

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
            if not (self.collarFeature.contains(l0=bottomNeck, l1=bottomHead)[0] or self.collarFeature.contains(l0=bottomHead, l1=topHead)[0]):
                return True, -20000

        if self.numSteps == 99:
            if self.saveStateOnReset and self.reset_number > 0:
                fname = self.state_save_directory + "triangle_rtuck"
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

        self.previousDeformationReward = reward_clothdeformation

        reward_contact_surface = 0
        if self.contactSurfaceReward:
            CID = self.clothScene.getHapticSensorContactIDs()[12]
            #if CID > 0:
            #    reward_contact_surface = 1.0
            #print(CID)
            reward_contact_surface = CID #-1.0 for full outside, 1.0 for full inside
            #print(reward_contact_surface)

        # force magnitude penalty
        reward_ctrl = -np.square(tau).sum()

        # reward for maintaining posture
        reward_upright = 0
        if self.uprightReward:
            reward_upright = max(-2.5, -abs(self.robot_skeleton.q[0]) - abs(self.robot_skeleton.q[1]))

        reward_stableHead = 0
        if self.stableHeadReward:
            reward_stableHead = max(-1.2, -abs(self.robot_skeleton.q[19]) - abs(self.robot_skeleton.q[20]))

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
            rDist = np.linalg.norm(self.rightTarget - wRFingertip2)
            #reward_rightTarget = -rDist - rDist**2
            reward_rightTarget = -rDist
            '''if rDist < 0.02:
                reward_rightTarget += 0.25'''

        reward_leftTarget = 0
        if self.leftTargetReward:
            lDist = np.linalg.norm(self.leftTarget - wLFingertip2)
            #reward_leftTarget = -lDist - lDist**2
            reward_leftTarget = -lDist
            '''if lDist < 0.02:
                reward_leftTarget += 0.25'''

        #penalty for elbow above the hand (may reduce the elbow flair)
        reward_elbowHandElevation = 0
        if self.elbowHandElevationReward:
            elbow = self.robot_skeleton.bodynodes[5].to_world(np.zeros(3))
            if wRFingertip2[1] < elbow[1]:
                reward_elbowHandElevation = wRFingertip2[1] - elbow[1]

        #reward percentage of point grid contained
        reward_containment = 0
        if self.containmentReward:
            containedPercent = 0
            for r in self.clothContainmentRecord:
                if r:
                    containedPercent += 1
            containedPercent /= len(self.clothContainmentRecord)
            reward_containment = containedPercent

        #binary reward signal for end effector containment
        reward_efContainment = 0
        if self.efContainmentReward:
            efcontainment = self.clothContainmentHeuristic.contained(point=self.robot_skeleton.bodynodes[7].to_world(self.fingertip), method=self.clothContainmentMethod)
            if efcontainment:
                reward_efContainment = 1
        self.previousEfContainment = reward_efContainment

        reward_openness = 0
        if self.opennessReward:
            #reward_openness = self.clothOpennessMetric.sortedArea
            reward_openness = self.waistFeature.area

        reward_triangleContainment = 0
        if self.triangleContainmentReward:
            #check intersection
            lines = [
                [self.robot_skeleton.bodynodes[7].to_world(self.fingertip), self.robot_skeleton.bodynodes[6].to_world(np.zeros(3))],
                [self.robot_skeleton.bodynodes[6].to_world(np.zeros(3)), self.robot_skeleton.bodynodes[5].to_world(np.zeros(3))]
            ]
            intersect = False
            aligned = False
            for l in lines:
                hit,dist,pos,aligned = pyutils.triangleLineSegmentIntersection(self.previousContainmentTriangle[0],self.previousContainmentTriangle[1],self.previousContainmentTriangle[2],l0=l[0], l1=l[1])
                if hit:
                    #print("hit: " + str(dist) + " | " + str(pos) + " | " + str(aligned))
                    intersect = True
                    aligned = aligned
                    break
            if intersect:
                if aligned:
                    reward_triangleContainment = 1
                else:
                    reward_triangleContainment = -1
            else:
                #if no intersection, get distance
                triangleCentroid = (self.previousContainmentTriangle[0] + self.previousContainmentTriangle[1] + self.previousContainmentTriangle[2])/3.0
                dist = np.linalg.norm(lines[0][0] - triangleCentroid)
                #dist, pos = pyutils.distToTriangle(self.previousContainmentTriangle[0],self.previousContainmentTriangle[1],self.previousContainmentTriangle[2],p=lines[0][0])
                #print(dist)
                reward_triangleContainment = -dist

        reward_triangleAlignment = 0
        if self.triangleAlignmentReward:
            U = self.previousContainmentTriangle[1] - self.previousContainmentTriangle[0]
            V = self.previousContainmentTriangle[2] - self.previousContainmentTriangle[0]

            tnorm = np.cross(U, V)
            tnorm /= np.linalg.norm(tnorm)

            torsoV = self.robot_skeleton.bodynodes[2].to_world(np.zeros(3))-self.robot_skeleton.bodynodes[1].to_world(np.zeros(3))
            torsoV /= np.linalg.norm(torsoV)

            reward_triangleAlignment = -tnorm.dot(torsoV)

        reward_sleeveForward = 0
        if self.sleeveForwardReward:
            vec = self.characterFrontBackPlane.org - self.CP2Feature.plane.org
            dist = vec.dot(self.characterFrontBackPlane.normal)
            if dist > 0:
                reward_sleeveForward = -dist

        reward_efForward = 0
        if self.efForwardReward:
            vec = self.characterFrontBackPlane.org - wRFingertip2
            dist = vec.dot(self.characterFrontBackPlane.normal)
            if dist > 0:
                reward_efForward = -dist

        #print("reward_sleeveForward " + str(reward_sleeveForward))
        #print("reward_efForward " + str(reward_efForward))
        #print("reward_triangleContainment " + str(reward_triangleContainment))

        #print("reward_restPose: " + str(reward_restPose))
        #print("reward_leftTarget: " + str(reward_leftTarget))
        self.reward = reward_ctrl * 0 \
                      + reward_upright * 2 \
                      + reward_stableHead * 3 \
                      + reward_clothdeformation * 20 \
                      + reward_restPose \
                      + reward_rightTarget*100 \
                      + reward_leftTarget*100 \
                      + reward_contact_surface * 3 \
                      + reward_openness*10 \
                      + reward_containment*10 \
                      + reward_efContainment*10 \
                      + reward_elbowHandElevation*5 \
                      + reward_triangleContainment*10 \
                      + reward_triangleAlignment * 2 \
                      + reward_sleeveForward * 5 \
                      + reward_efForward * 15
        # TODO: revisit cloth deformation penalty
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

        return obs

    def additionalResets(self):
        '''if self.resetTime > 0:
            print("reset " + str(self.reset_number) + " after " + str(time.time()-self.resetTime))
        '''

        self.resetTime = time.time()
        self.samplePoints = []
        #do any additional resetting here

        if self.localPointGrid is None and self.containmentReward:
            gridDimensions = np.array([0.1,0.05,0.08])
            gridSamples = np.array([3,2,3])
            points = pyutils.gridSample(dimSamples=gridSamples, org=np.array([0, 0, -0.2]), dimensions=gridDimensions)
            self.localPointGrid = pyutils.localPointSet(node=self.robot_skeleton.bodynodes[2], points=points)
            self.localPointGrid.worldPoints = points
            self.localPointGrid.updateLocal()
        if self.clothContainmentHeuristic is None and self.containmentReward:
            self.clothContainmentHeuristic = pyutils.clothContainmentHeuristic(clothScene=self.clothScene,verts=self.clothContainmentHeuristicVerts, constraintFeatures=[self.waistFeature])

        #if self.clothOpennessMetric is None:
        '''nL = self.robot_skeleton.bodynodes[9]
        nR = self.robot_skeleton.bodynodes[4]
        nM = self.robot_skeleton.bodynodes[2]
        vL = 889
        vR = 1041
        self.clothOpennessMetric = OpennessMetricObject(self.clothScene, nL=nL, nM=nM, nR=nR, vR=vR, vL=vL)
        '''

        if self.graphClothOpenness and self.opennessReward:
            if self.saveGraphsOnReset and self.clothOpennessGraph is not None:
                self.clothOpennessGraph.save(filename="clothOpennessGraph"+str(self.reset_number))
            print(self.clothOpennessGraph)
            if self.clothOpennessGraph is not None:
                self.clothOpennessGraph.close()
            self.clothOpennessGraph = pyutils.LineGrapher(title="Cloth Openness", numPlots=3)

        if self.graphContainmentHeuristic and self.containmentReward:
            if self.saveGraphsOnReset and self.containmentHeuristicGraph is not None:
                self.containmentHeuristicGraph.save(filename="containmentHeuristicGraph2m"+str(self.reset_number))
            if self.containmentHeuristicGraph is not None:
                self.containmentHeuristicGraph.close()
            self.containmentHeuristicGraph = pyutils.LineGrapher(title="Cloth Containment (method 2)", numPlots=2)

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
        self.set_state(qpos, qvel)

        #find end effector targets and set restPose from solution
        self.rightTarget = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)
        self.leftTarget = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
        #print("right target: " + str(self.rightTarget))
        #print("left target: " + str(self.leftTarget))
        self.restPose = np.array(self.robot_skeleton.q)

        if self.resetStateFromDistribution:
            if self.reset_number == 0: #load the distribution
                count = 0
                objfname_ix = self.resetDistributionPrefix + "%05d" % count
                while os.path.isfile(objfname_ix + ".obj"):
                    count += 1
                    #print(objfname_ix)
                    self.clothScene.addResetStateFrom(filename=objfname_ix+".obj")
                    objfname_ix = self.resetDistributionPrefix + "%05d" % count
                if count != self.resetDistributionSize:
                    print("Distribution size is not synced! " + str(count)+"/"+str(self.resetDistributionSize))
            resetStateNumber = random.randint(0,self.resetDistributionSize-1)
            #resetStateNumber = 0
            charfname_ix = self.resetDistributionPrefix + "_char%05d" % resetStateNumber
            self.loadCharacterState(filename=charfname_ix)
            # update physx capsules
            self.updateClothCollisionStructures(hapticSensors=True)
            self.clothScene.clearInterpolation()
            self.clothScene.setResetState(cid=0, index=resetStateNumber)
            qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-0.1, high=0.1, size=self.robot_skeleton.ndofs)
            self.robot_skeleton.set_velocities(qvel)

        else:
            self.loadCharacterState(filename="characterState_endDropGrip1")

        if self.handleNode is not None:
            self.handleNode.clearHandles()
            self.handleNode.addVertices(verts=self.targetGripVerticesL)
            self.handleNode.setOrgToCentroid()
            if self.updateHandleNodeFrom >= 0:
                self.handleNode.setTransform(self.robot_skeleton.bodynodes[self.updateHandleNodeFrom].T)
            self.handleNode.recomputeOffsets()


        if self.simulateCloth:
            self.collarFeature.fitPlane()
            self.gripFeatureL.fitPlane()
            self.gripFeatureR.fitPlane()
            self.CP2Feature.fitPlane()
            self.waistFeature.fitPlane(normhint=np.array([0,1.0,0]))

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
            self.collarFeature.drawProjectionPoly(renderBasis=False, renderNormal=False)
        self.gripFeatureL.drawProjectionPoly(renderBasis=False, renderNormal=False)
        #if self.waistFeature is not None:
        #    self.waistFeature.drawProjectionPoly(renderBasis=False)

        renderUtils.setColor([0,0,0])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[4].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[4].to_world(np.array([0.0,-0.3,-0.075]))])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[9].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[9].to_world(np.array([0.0,-0.3,-0.075]))])

        renderUtils.drawLines(lines=pyutils.getRobotLinks(robot=self.robot_skeleton, pose=self.restPose))

        reward_sleeveForward = 0
        if self.sleeveForwardReward:
            vec = self.characterFrontBackPlane.org - self.CP2Feature.plane.org
            dist = vec.dot(self.characterFrontBackPlane.normal)
            renderUtils.setColor([0,1.0,0])
            if dist > 0:
                reward_sleeveForward = -dist
                renderUtils.setColor([1.0, 0, 0])

            cp = self.characterFrontBackPlane.projectPoint(self.CP2Feature.plane.org)
            renderUtils.drawSphere(pos=cp,rad=0.01)
            renderUtils.drawSphere(pos=self.characterFrontBackPlane.org,rad=0.01)
            renderUtils.drawLines(lines=[[cp,self.CP2Feature.plane.org]])

        if self.testingCapsuleRelaxation:
            # test capsule projection
            c0 = np.array([0, 0.1, -0.5])
            r0 = 0.05
            c1 = np.array([0, -0.1, -0.5])
            r1 = 0.025

            renderUtils.setColor([0.8, 0, 0.8])
            renderUtils.drawSphere(pos=c0, rad=r0)
            renderUtils.drawSphere(pos=c1, rad=r1)
            renderUtils.drawLines(lines=[[c0, c1]])

            if len(self.samplePoints) == 0:
                for i in range(50):
                    p = np.random.uniform(low=-0.2, high=0.2, size=3)
                    p += np.array([0, 0, -0.5])
                    self.samplePoints.append(p)

            self.samplePoints = pyutils.relaxCapsulePoints(self.samplePoints,c0,c1,r0,r1)

            for p in self.samplePoints:
                renderUtils.setColor([0.0, 0, 0.8])
                renderUtils.drawSphere(pos=p, rad=0.005)
                #pos, dist = pyutils.projectToCapsule(p, c0,c1,r0,r1)
                #renderUtils.drawLines(lines=[[p, pos]])
                #renderUtils.setColor([0.0, 0.8, 0])
                #renderUtils.drawSphere(pos=pos, rad=0.01)

        #self.characterFrontBackPlane.draw()

        if self.triangleContainmentReward:
            renderUtils.setColor([1.0, 1.0, 0])
            renderUtils.drawTriangle(self.previousContainmentTriangle[0],self.previousContainmentTriangle[1],self.previousContainmentTriangle[2])
            U = self.previousContainmentTriangle[1] - self.previousContainmentTriangle[0]
            V = self.previousContainmentTriangle[2] - self.previousContainmentTriangle[0]

            tnorm = np.cross(U, V)
            tnorm /= np.linalg.norm(tnorm)
            centroid = (self.previousContainmentTriangle[0] + self.previousContainmentTriangle[1] + self.previousContainmentTriangle[2])/3.0
            renderUtils.drawLines(lines=[[centroid, centroid+tnorm]])

        #triangle intersect/distance testing
        '''lines = [
            [self.robot_skeleton.bodynodes[7].to_world(self.fingertip),
             self.robot_skeleton.bodynodes[6].to_world(np.zeros(3))],
            [self.robot_skeleton.bodynodes[6].to_world(np.zeros(3)),
             self.robot_skeleton.bodynodes[5].to_world(np.zeros(3))]
        ]
        for l in lines:
            hit, dist, pos, aligned = pyutils.triangleLineSegmentIntersection(self.previousContainmentTriangle[0],
                                                                          self.previousContainmentTriangle[1],
                                                                          self.previousContainmentTriangle[2], l0=l[0],
                                                                          l1=l[1])
        dist, pos = pyutils.distToTriangle(self.previousContainmentTriangle[0], self.previousContainmentTriangle[1],
                                           self.previousContainmentTriangle[2], p=lines[0][0])
        renderUtils.drawLines(lines=[[lines[0][0], pos]])'''

        #test grid sampling
        #print("render")
        #self.timeToTest = time.time()
        '''
        if self.localPointGrid is not None:
            points = self.localPointGrid.getWorldPoints()
            #renderUtils.drawLineStrip(points=points)
            if len(self.clothContainmentRecord) == len(points):
                containment = self.clothContainmentRecord
                #self.timeToTest = time.time()-self.timeToTest
                self.clothContainmentHeuristic.drawVerts()
                for pix,p in enumerate(points):
                    if containment[pix]:
                        renderUtils.setColor(color=[0,0.8,0])
                        renderUtils.drawSphere(pos=p, rad=0.02)
                    else:
                        renderUtils.setColor(color=[0.8, 0.0, 0])
                        renderUtils.drawTetrahedron(pos=p, rad=0.02, solid=False)
                        #renderUtils.drawSphere(pos=p, rad=0.02, solid=False)
            else:
                for p in points:
                    renderUtils.drawSphere(pos=p, rad=0.02)
        '''
        #trapezoidal openness metric
        #if self.clothOpennessMetric is not None:
        #    self.clothOpennessMetric.drawRegion()

        if self.efContainmentReward:
            renderUtils.setColor([1.0,0,0])
            if self.previousEfContainment:
                renderUtils.setColor([0,1.0,0])
            renderUtils.drawSphere(pos=np.array([0,0.5,0]),rad=0.05)

        #render targets
        if self.rightTargetReward:
            efR = self.robot_skeleton.bodynodes[7].to_world(self.fingertip)
            renderUtils.setColor(color=[1.0,0,0])
            #renderUtils.drawSphere(pos=self.rightTarget,rad=0.02)
            renderUtils.drawLineStrip(points=[self.rightTarget, efR])
        if self.leftTargetReward:
            efL = self.robot_skeleton.bodynodes[12].to_world(self.fingertip)
            renderUtils.setColor(color=[0, 1.0, 0])
            #renderUtils.drawSphere(pos=self.leftTarget,rad=0.02)
            renderUtils.drawLineStrip(points=[self.leftTarget, efL])

        textHeight = 15
        textLines = 2

        if self.renderUI:
            renderUtils.setColor(color=[0.,0,0])
            self.clothScene.drawText(x=15., y=textLines*textHeight, text="Steps = " + str(self.numSteps), color=(0., 0, 0))
            textLines += 1
            self.clothScene.drawText(x=15., y=textLines*textHeight, text="Rollout ID = " + str(self.reset_number), color=(0., 0, 0))
            textLines += 1
            self.clothScene.drawText(x=15., y=textLines*textHeight, text="Reward = " + str(self.reward), color=(0., 0, 0))
            textLines += 1
            self.clothScene.drawText(x=15., y=textLines * textHeight, text="Cumulative Reward = " + str(self.cumulativeReward), color=(0., 0, 0))
            textLines += 1
            self.clothScene.drawText(x=15., y=textLines * textHeight, text="Sleeve Forward Reward = " + str(reward_sleeveForward), color=(0., 0, 0))
            textLines += 1
            #self.clothScene.drawText(x=15., y=textLines * textHeight, text="triangle dist = " + str(dist), color=(0., 0, 0))
            #textLines += 1

            #if self.clothOpennessMetric is not None:
            #    self.clothScene.drawText(x=15., y=textLines * textHeight, text="Trapezoid Area = " + str(self.clothOpennessMetric.area), color=(0., 0, 0))
            #    textLines += 1
            #self.clothScene.drawText(x=15., y=textLines * textHeight, text="Containment Test Time = " + str(self.timeToTest), color=(0., 0, 0))
            #textLines += 1
            if self.numSteps > 0:
                renderUtils.renderDofs(robot=self.robot_skeleton, restPose=self.restPose, renderRestPose=True)

            renderUtils.drawProgressBar(topLeft=[600, self.viewer.viewport[3] - 30], h=16, w=60, progress=-self.previousDeformationReward, color=[1.0, 0.0, 0])
