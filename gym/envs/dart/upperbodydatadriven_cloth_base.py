# This environment is created by Alexander Clegg (alexanderwclegg@gmail.com)

import numpy as np
from gym import utils
from gym.envs.dart.dart_cloth_env import *
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

class SPDController():
    def __init__(self, env, target=None):
        self.name = "SPD"
        self.target = target
        self.env = env
        self.h = 0.08
        print(env)
        self.skel = env.robot_skeleton
        ndofs = self.skel.ndofs
        self.qhat = self.skel.q
        #self.Kp = np.diagflat([400.0] * (ndofs))
        self.Kp = np.diagflat([200.0] * (ndofs))
        #self.Kd = np.diagflat([40.0] * (ndofs))
        self.Kd = np.diagflat([10.0] * (ndofs))
        self.preoffset = 0.0

    def setup(self):
        #reset the target
        cur_q = np.array(self.skel.q)
        self.env.loadCharacterState(filename="characterState_regrip")
        self.env.restPose = np.array(self.skel.q)
        self.target = np.array(self.skel.q)
        self.env.robot_skeleton.set_positions(cur_q)
        a=0

    def query(self):
        #SPD
        self.qhat = np.array(self.target)
        if self.qhat is None:
            return np.zeros(self.skel.ndofs)
        #print("pose: " + str(self.skel.q) + " | target: " + str(self.qhat))
        skel = self.skel
        p = -self.Kp.dot(skel.q + skel.dq * self.h - self.qhat)
        d = -self.Kd.dot(skel.dq)
        b = -skel.c + p + d + skel.constraint_forces()
        A = skel.M + self.Kd * self.h

        x = np.linalg.solve(A, b)

        #invM = np.linalg.inv(A)
        #x = invM.dot(b)
        tau = p + d - self.Kd.dot(x) * self.h
        #print("tau_out: " + str(tau))
        return tau

class DartClothUpperBodyDataDrivenClothBaseEnv(DartClothEnv, utils.EzPickle):
    def __init__(self, rendering=True, screensize=(1080,720), clothMeshFile="", clothMeshStateFile=None, clothScale=1.4, obs_size=0, simulateCloth=True, recurrency=0, SPDActionSpace=False, lockTorso=False, gravity=False, frameskip=4, dt=0.002):
        self.prefix = os.path.dirname(__file__)

        #rendering variables
        self.useOpenGL = rendering
        self.screenSize = screensize
        self.renderDARTWorld = True
        self.renderUI = True
        self.renderRewardsData = True
        self.avgtimings = {}

        #sim variables
        self.gravity = gravity
        self.dataDrivenJointLimts = True
        self.lockTorso = lockTorso
        self.lockSpine = lockTorso
        self.additionalAction = np.zeros(22) #added to input action for step
        self.SPD = None #no target yet
        self.SPDTarget = None #if set, reset calls setup on the SPDController and udates/queries it
        self.recurrency = recurrency
        self.actionTrajectory = []
        self.SPDTorqueLimits = False

        #rewards data tracking
        self.rewardsData = renderUtils.RewardsData([],[],[],[])

        #output for rendering controls
        self.recordForRendering = False
        #self.recordForRenderingOutputPrefix = "saved_render_states/jacket/jacket"
        self.recordForRenderingOutputPrefix = "saved_render_states/tshirtseq_spd/tshirtseq"

        #other tracking variables
        self.rewardTrajectory = [] #store all rewards since the last reset

        #violation graphing
        self.graphViolation = False
        self.violationGraphFrequency = 1

        #cloth "badness" graphing
        self.graphClothViolation = False
        self.saveClothViolationGraphOnReset = True
        self.clothViolationGraph = None
        if self.graphClothViolation:
            self.clothViolationGraph = pyutils.LineGrapher(title="Cloth Violation", numPlots=3)

        #randomness graphing
        self.initialRand = 0
        self.initialnpRand = 0
        self.initialselfnpRand = 0
        self.graphingRandomness = False
        self.randomnessGraph = None
        if self.graphingRandomness:
            self.randomnessGraph = pyutils.LineGrapher(title="Randomness")

        #graphing character pose sum for control determinism tracking
        self.graphPoseSum = False
        if self.graphPoseSum:
            self.poseSumGraph = pyutils.LineGrapher(title="Pose Sum")

        #record character range of motion through random exploration
        self.recordROMPoints = False
        self.loadROMPoints = True
        self.processROMPoints = False
        self.ROMPoints = []
        self.ROMPositions = [] #end effector positions at ROMPoint poses
        self.ROMPointMinDistance = 1.0
        self.ROMFile = self.prefix + "/assets/processedROMPoints_upperbodycapsules_datadriven"#"processedROMPoints"

        self.arm = 0 # 0->both, 1->right, 2->left
        self.actuatedDofs = np.arange(22) # full upper body
        self.lockedDofs = []
        self.limbNodesR = [3, 4, 5, 6, 7]
        self.limbNodesL = [8, 9, 10, 11, 12]
        self.efOffset = np.array([0,-0.06,0])

        if self.arm == 1:
            self.actuatedDofs = np.arange(3, 11) # right arm
            self.lockedDofs = np.concatenate([np.arange(3), np.arange(11, 22)])

        elif self.arm == 2:
            self.actuatedDofs = np.arange(11, 19) # left arm
            self.lockedDofs = np.concatenate([np.arange(11), np.arange(19, 22)])

        if self.lockTorso:
            for i in range(2):
                if i not in self.lockedDofs:
                    self.lockedDofs.append(i)
        if self.lockSpine and 2 not in self.lockedDofs:
            self.lockedDofs.append(2)


        #22 dof upper body
        self.action_scale = np.ones(len(self.actuatedDofs))
        if not SPDActionSpace:
            self.action_scale *= 12
            if 0 in self.actuatedDofs:
                self.action_scale[self.actuatedDofs.tolist().index(0)] = 50
            if 1 in self.actuatedDofs:
                self.action_scale[self.actuatedDofs.tolist().index(1)] = 50

        if self.recurrency > 0:
            self.action_scale = np.concatenate([self.action_scale, np.ones(self.recurrency)])

        self.control_bounds = np.array([np.ones(len(self.actuatedDofs)+self.recurrency), np.ones(len(self.actuatedDofs)+self.recurrency)*-1])

        self.reset_number = 0
        self.numSteps = 0

        #create cloth scene
        clothScene = None

        if clothMeshStateFile is not None:
            clothScene = pyphysx.ClothScene(step=0.01,
                                            mesh_path=self.prefix + "/assets/" + clothMeshFile,
                                            state_path=self.prefix + "/../../../../" + clothMeshStateFile,
                                            scale=clothScale)
        else:
            clothScene = pyphysx.ClothScene(step=0.01,
                                            mesh_path=self.prefix + "/assets/" + clothMeshFile,
                                            scale=clothScale)

        clothScene.togglePinned(0, 0)  # turn off auto-pin

        self.separatedMesh = None
        if simulateCloth:
            self.separatedMesh = meshgraph.MeshGraph(clothscene=clothScene)

        self.reward = 0
        self.cumulativeReward = 0
        self.deformation = 0

        self.obs_size = obs_size
        skelFile = 'UpperBodyCapsules_datadriven.skel'

        #intialize the parent env
        if self.useOpenGL is True:
            DartClothEnv.__init__(self, cloth_scene=clothScene, model_paths=skelFile, frame_skip=frameskip, dt=dt,
                                  observation_size=obs_size, action_bounds=self.control_bounds, screen_width=self.screenSize[0], screen_height=self.screenSize[1])
        else:
            DartClothEnv.__init__(self, cloth_scene=clothScene, model_paths=skelFile, frame_skip=frameskip, dt=dt,
                                  observation_size=obs_size, action_bounds=self.control_bounds , disableViewer = True, visualize = False)

        #rescaling actions for SPD
        if SPDActionSpace:
            for ix, dof in enumerate(self.robot_skeleton.dofs):
                if dof.has_position_limit():
                    self.action_scale[ix] = 1.0
                    self.control_bounds[0][ix] = dof.position_upper_limit()
                    self.control_bounds[1][ix] = dof.position_lower_limit()
                    print("ix: " + str(ix) + " | control_bounds["+str(ix)+"]: " + str(self.control_bounds[0][ix]) + ", " + str(str(self.control_bounds[0][ix])))
                else:
                    self.action_scale[ix] = 3.14
                    self.control_bounds[0][ix] = 1.0
                    self.control_bounds[1][ix] = -1.0
            self.action_space = spaces.Box(self.control_bounds[1], self.control_bounds[0])

        print("action_space: " + str(self.action_space))
        print("action_scale: " + str(self.action_scale))
        print("control_bounds: " + str(self.control_bounds))

        #setup data-driven joint limits
        if self.dataDrivenJointLimts:
            leftarmConstraint = pydart.constraints.HumanArmJointLimitConstraint(self.robot_skeleton.joint('j_bicep_left'), self.robot_skeleton.joint('elbowjL'), True)
            rightarmConstraint = pydart.constraints.HumanArmJointLimitConstraint(self.robot_skeleton.joint('j_bicep_right'), self.robot_skeleton.joint('elbowjR'), False)
            leftarmConstraint.add_to_world(self.dart_world)
            rightarmConstraint.add_to_world(self.dart_world)

        utils.EzPickle.__init__(self)

        if not self.gravity:
            self.dart_world.set_gravity(np.zeros(3))
        else:
            self.dart_world.set_gravity(np.array([0., -9.8, 0]))

        self.clothScene.setFriction(0, 0.5) #reset this anytime as desired

        self.collisionCapsuleInfo = None #set in updateClothCollisionStructures(capsules=True)
        self.collisionSphereInfo = None #set in updateClothCollisionStructures()
        self.updateClothCollisionStructures(capsules=True, hapticSensors=True)
        
        self.simulateCloth = simulateCloth

        #enable DART collision testing
        self.robot_skeleton.set_self_collision_check(True)
        self.robot_skeleton.set_adjacent_body_check(False)

        #setup collision filtering
        collision_filter = self.dart_world.create_collision_filter()
        collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[10],
                                           self.robot_skeleton.bodynodes[12])  # left forearm to fingers
        collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[5],
                                           self.robot_skeleton.bodynodes[7])  # right forearm to fingers
        collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[1],
                                           self.robot_skeleton.bodynodes[13])  # torso to neck
        collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[1],
                                           self.robot_skeleton.bodynodes[14])  # torso to head
        collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[1],
                                           self.robot_skeleton.bodynodes[3])  # torso to right shoulder
        collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[1],
                                           self.robot_skeleton.bodynodes[8])  # torso to left shoulder
        collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[13],
                                           self.robot_skeleton.bodynodes[3])  # neck to right shoulder
        collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[13],
                                           self.robot_skeleton.bodynodes[8])  # neck to left shoulder
        collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[14],
                                           self.robot_skeleton.bodynodes[3])  # head to right shoulder
        collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[14],
                                           self.robot_skeleton.bodynodes[8])  # head to left shoulder
        collision_filter.add_to_black_list(self.robot_skeleton.bodynodes[3],
                                           self.robot_skeleton.bodynodes[8])  # right shoulder to left shoulder

        for i in range(len(self.robot_skeleton.bodynodes)):
            print(self.robot_skeleton.bodynodes[i])
            
        for i in range(len(self.robot_skeleton.dofs)):
            print(self.robot_skeleton.dofs[i])

        #enable joint limits
        for i in range(len(self.robot_skeleton.joints)):
            print(self.robot_skeleton.joints[i])

        #DART does not automatically limit joints with any unlimited dofs
        self.robot_skeleton.joints[4].set_position_limit_enforced(True)
        self.robot_skeleton.joints[9].set_position_limit_enforced(True)

        if self.loadROMPoints:
            self.ROMPoints = pyutils.loadListOfVecs(filename=self.ROMFile)
            self.ROMPositions = pyutils.positionsFromPoses(self.robot_skeleton, poses=self.ROMPoints,nodes=[7,12], offsets=[np.array([0,-0.065,0]),np.array([0,-0.065,0])])
        if self.processROMPoints:
            self._processROMPoints()

        self.clothScene.setSelfCollisionDistance(distance=0.03)
        self.clothScene.step()
        self.clothScene.reset()

    def _getFile(self):
        return __file__

    def _processROMPoints(self):
        positions = pyutils.positionsFromPoses(self.robot_skeleton,poses=self.ROMPoints,nodes=[7,12], offsets=[np.array([0,-0.065,0]),np.array([0,-0.065,0])])
        before = pyutils.averageShortestDistance(positions)#len(self.ROMPoints)
        pyutils.cullPosesFromPositionDistances(numPoses=1000, poses=self.ROMPoints, positions=positions)
        after = pyutils.averageShortestDistance(positions)#len(self.ROMPoints)
        print("before: " + str(before) + " after: " + str(after))
        pyutils.saveList(self.ROMPoints, filename="processedROMPoints", listoflists=True)

    def saveObjState(self, filename=None):
        print("Trying to save the object state")
        print("filename: " + str(filename))
        if filename is None:
            filename = "objState"
        self.clothScene.saveObjState(filename, 0)

    def saveCharacterState(self, filename=None):
        print("saving character state")
        if filename is None:
            filename = "characterState"
        print("filename " + str(filename))
        f = open(filename, 'w')
        for ix,dof in enumerate(self.robot_skeleton.q):
            if ix > 0:
                f.write(" ")
            f.write(str(dof))

        f.write("\n")

        for ix,dof in enumerate(self.robot_skeleton.dq):
            if ix > 0:
                f.write(" ")
            f.write(str(dof))
        f.close()

    def loadCharacterState(self, filename=None):
        openFile = "characterState"
        if filename is not None:
            openFile = filename
        f = open(openFile, 'r')
        qpos = np.zeros(self.robot_skeleton.ndofs)
        qvel = np.zeros(self.robot_skeleton.ndofs)
        for ix, line in enumerate(f):
            if ix > 1: #only want the first 2 file lines
                break
            words = line.split()
            if(len(words) != self.robot_skeleton.ndofs):
                break
            if(ix == 0): #position
                qpos = np.zeros(self.robot_skeleton.ndofs)
                for ixw, w in enumerate(words):
                    qpos[ixw] = float(w)
            else: #velocity (if available)
                qvel = np.zeros(self.robot_skeleton.ndofs)
                for ixw, w in enumerate(words):
                    qvel[ixw] = float(w)

        self.robot_skeleton.set_positions(qpos)
        self.robot_skeleton.set_velocities(qvel)
        f.close()

    def saveCharacterRenderState(self, filename=None):
        if filename is None:
            filename = "characterRenderState"
        #save the position and radius of each endpoint of each capsule for simple loading in Maya
        CSI = self.clothScene.getCollisionSpheresInfo()
        #info format: pos(3), radius, ignore(5)
        CCI = self.clothScene.getCollisionCapsuleInfo()
        #print("CSI: " + str(CSI))
        #print("CCI: " + str(CCI))

        f = open(filename, 'w')
        for capix in range(len(CCI)):
            for capix2 in range(len(CCI[capix])):
                if CCI[capix][capix2] != 0:
                    #this is a capsule
                    f.write(str(CSI[capix*9:capix*9+4]) + " " + str(CSI[capix2*9:capix2*9+4]) + "\n")
        f.close()

    def graphJointConstraintViolation(self, cix=1, pivotDof=8, otherDofs=[5,6,7], numPivotSamples=50, numOtherSamples=50, otherRange=(-1.0,1.0), counting=False):
        startTime = time.time()
        #print("graphJointConstraintViolation: collecting pivot samples")
        #collect samples:
        pivotLimits = [self.robot_skeleton.dofs[pivotDof].position_lower_limit(), self.robot_skeleton.dofs[pivotDof].position_upper_limit()]
        pivotSamples = []
        for i in range(numPivotSamples):
            pivotSamples.append(pivotLimits[0] + (pivotLimits[1]-pivotLimits[0])*(i/(numPivotSamples-1)))
        #print("graphJointConstraintViolation: collecting other samples")
        samples = []
        for dof in otherDofs:
            samples.append([])
            if self.robot_skeleton.dofs[dof].has_position_limit():
                #sample the whole range
                dofLimits = [self.robot_skeleton.dofs[dof].position_lower_limit(), self.robot_skeleton.dofs[dof].position_upper_limit()]
                for i in range(numOtherSamples):
                    samples[-1].append(dofLimits[0] + (dofLimits[1]-dofLimits[0])*(i/(numOtherSamples-1)))
            else:
                #sample otherRange about the current position
                currentPos = self.robot_skeleton.q[dof]
                for i in range(numOtherSamples):
                    samples[-1].append(otherRange[0]+currentPos + (otherRange[1]-otherRange[0])*(i/(numOtherSamples-1)))

        #print("graphJointConstraintViolation: checking violation")
        #check and store constraint violation at each sampled pose combination
        violations = [[] for i in range(len(otherDofs))]
        collisions = [[] for i in range(len(otherDofs))]
        positions = [[] for i in range(len(otherDofs))]
        currentPose = np.array(self.robot_skeleton.q)
        currentVel = np.array(self.robot_skeleton.dq)
        for ps in pivotSamples:
            for ix,list in enumerate(samples):
                violations[ix].append([])
                positions[ix].append([])
                collisions[ix].append([])
                for item in list:
                    qpos = np.array(currentPose)
                    qpos[pivotDof] = ps
                    qpos[otherDofs[ix]] = item
                    self.robot_skeleton.set_positions(qpos)
                    self.dart_world.step()
                    violations[ix][-1].append(self.dart_world.getAllConstraintViolations()[cix])
                    positions[ix][-1].append(item)
                    if self.dart_world.collision_result.num_contacts() > 0:
                        collisions[ix][-1].append(np.array([0.0, 1.0, 0.0, 0.2]))
                    else:
                        collisions[ix][-1].append(np.array([1.0, 0, 0, 0.0]))

        self.set_state(currentPose, currentVel)
        #print("graphJointConstraintViolation: done: " + str(violations))
        filenames = []
        for i in range(len(otherDofs)):
            ylabel = self.robot_skeleton.dofs[pivotDof].name
            xlabel = self.robot_skeleton.dofs[otherDofs[i]].name
            title = "Constraint violation " + xlabel + " vs. " + ylabel
            filename = xlabel + "_vs_" + ylabel
            points = [[currentPose[otherDofs[i]], currentPose[pivotDof]]]
            #print("about to create image " + filename)
            filenames.append(renderUtils.render2DHeatmap(data=violations[i], overlay=collisions[i], extent=[samples[i][0],samples[i][-1], pivotLimits[0],pivotLimits[1]], title=title, xlabel=xlabel, ylabel=ylabel, points=points, filename=filename))
            #print("...done")
        #print("finished creating the images")
        outfilename = ""
        if counting:
            outfilename = "../image_matrix_output/%05d" % self.numSteps
        renderUtils.imageMatrixFrom(filenames=filenames, outfilename=outfilename)
        #print("Done graphing constraints. Took " + str(time.time()-startTime) + " seconds.")
    def updateBeforeSimulation(self):
        #any pre-sim updates should happen here
        a=0

    def checkTermination(self, tau, s, obs):
        #check the termination conditions and return: done,reward
        if np.amax(np.absolute(s[:len(self.robot_skeleton.q)])) > 10:
            print("Detecting potential instability")
            print(s)
            return True, -500
        elif not np.isfinite(s).all():
            print("Infinite value detected..." + str(s))
            return True, -500

        return False, 0

    def computeReward(self, tau):
        #compute and return reward at the current state
        return 0

    def _step(self, a):
        #print("a: " + str(a))
        startTime = time.time()
        if self.reset_number < 1 or not self.simulating:
            return np.zeros(self.obs_size), 0, False, {}

        #save state for rendering
        if self.recordForRendering:
            fname = self.recordForRenderingOutputPrefix
            objfname_ix = fname + "%05d" % self.numSteps
            charfname_ix = fname + "_char%05d" % self.numSteps
            self.saveObjState(filename=objfname_ix)
            self.saveCharacterRenderState(filename=charfname_ix)

        #try:
        if self.graphViolation:
            if self.numSteps%self.violationGraphFrequency == 0:
                self.graphJointConstraintViolation(counting=True)

        if self.graphClothViolation:
            clothViolation = self.clothScene.getNumSelfCollisions()
            maxDef, minDef, avgDef, variance, ratios = self.clothScene.getAllDeformationStats(cid=0)
            defPenalty = (math.tanh(0.14 * (maxDef - 25)) + 1) / 2.0 #taken from reward in envs
            #print(maxDef)
            #print(avgDef)
            self.clothViolationGraph.addToLinePlot(data=[clothViolation/5.0, maxDef, defPenalty*10])

        if self.graphingRandomness:
            if len(self.randomnessGraph.xdata) > 0:
                self.initialRand += random.random()
                self.initialnpRand += np.random.random()
                self.initialselfnpRand += self.np_random.uniform()
                self.randomnessGraph.yData[-3][self.numSteps] = self.initialRand
                self.randomnessGraph.yData[-2][self.numSteps] = self.initialnpRand
                self.randomnessGraph.yData[-1][self.numSteps] = self.initialselfnpRand
                self.randomnessGraph.update()

        if self.graphPoseSum:
            if len(self.poseSumGraph.xdata) > 0:
                poseSum = np.linalg.norm(self.robot_skeleton.q)
                self.poseSumGraph.yData[-1][self.numSteps] = poseSum
                self.poseSumGraph.update()

        startTime2 = time.time()
        self.additionalAction = np.zeros(len(self.robot_skeleton.q))
        #self.additionalAction should be set in updateBeforeSimulation
        self.updateBeforeSimulation()  # any env specific updates before simulation
        # print("updateBeforeSimulation took " + str(time.time() - startTime2))
        try:
            self.avgtimings["updateBeforeSimulation"] += time.time() - startTime2
        except:
            self.avgtimings["updateBeforeSimulation"] = time.time() - startTime2

        full_control = np.array(a)

        if self.SPDTarget is not None and self.SPD is not None:
            self.SPD.target = np.array(self.SPDTarget)
            self.additionalAction += self.SPD.query()
            #print(self.additionalAction)
        else:
            full_control[:len(self.additionalAction)] = full_control[:len(self.additionalAction)] + self.additionalAction
        #print("full_control = " + str(full_control))
        clamped_control = np.array(full_control)
        if not self.SPDTorqueLimits:
            for i in range(len(clamped_control)):
                if clamped_control[i] > self.control_bounds[0][i]:
                    clamped_control[i] = self.control_bounds[0][i]
                if clamped_control[i] < self.control_bounds[1][i]:
                    clamped_control[i] = self.control_bounds[1][i]
        else:
            for i in range(len(clamped_control)):
                if clamped_control[i] > self.action_scale[i]:
                    clamped_control[i] = self.action_scale[i]
                if clamped_control[i] < -self.action_scale[i]:
                    clamped_control[i] = -self.action_scale[i]
        #print("clamped_control = " + str(clamped_control))

        if self.recordROMPoints:
            violation = self.dart_world.getMaxConstraintViolation()
            if violation > 0:
                #print(violation)
                minDist = None
                for p in self.ROMPoints:
                    dist = np.linalg.norm(self.robot_skeleton.q-p)
                    if minDist is None:
                        minDist = dist
                    if dist < minDist:
                        minDist = dist
                        if minDist < self.ROMPointMinDistance:
                            break
                if minDist is not None:
                    if minDist > self.ROMPointMinDistance:
                        self.ROMPoints.append(np.array(self.robot_skeleton.q))
                        print("Saved poses = " + str(len(self.ROMPoints)))
                else: #auto-add when list is empty
                    self.ROMPoints.append(np.array(self.robot_skeleton.q))

        tau = np.array(clamped_control)
        if not self.SPDTorqueLimits:
            tau = np.multiply(clamped_control, self.action_scale)

        #apply action and simulate
        if len(tau) < len(self.robot_skeleton.q):
            newtau = np.array(tau)
            tau = np.zeros(len(self.robot_skeleton.q))
            for ix,dof in enumerate(self.actuatedDofs):
                tau[dof] = newtau[ix]

        startTime2 = time.time()
        if self.SPDTarget is not None and self.SPD is not None:
            #print(self.additionalAction)
            #print("Additional Action: " + str(self.additionalAction))
            self.do_simulation(self.additionalAction, self.frame_skip)
        else:
            #if len(self.actionTrajectory) == 0:
            #    self.actionTrajectory = pyutils.loadListOfVecs(filename="actionTraj.txt")
            #if self.reset_number < 2:
                #if self.reset_number == 1:
                #    self.actionTrajectory.append(np.array(tau))
            self.do_simulation(tau[:len(self.robot_skeleton.q)], self.frame_skip)
            #else:
                #pyutils.saveList(list=self.actionTrajectory, filename="actionTraj.txt", listoflists=True)
                #exit()
            #print(self.actionTrajectory)
            #self.do_simulation(self.actionTrajectory[self.numSteps][:len(self.robot_skeleton.q)], self.frame_skip)

        #print("do_simulation took " + str(time.time() - startTime2))
        try:
            self.avgtimings["do_simulation"] += time.time() - startTime2
        except:
            self.avgtimings["do_simulation"] = time.time() - startTime2


        #set position and 0 velocity of locked dofs
        qpos = self.robot_skeleton.q
        qvel = self.robot_skeleton.dq
        for dof in self.lockedDofs:
            qpos[dof] = 0
            qvel[dof] = 0
        self.set_state(qpos, qvel)

        startTime2 = time.time()
        reward = self.computeReward(tau=tau)
        #print("computeReward took " + str(time.time() - startTime2))
        try:
            self.avgtimings["computeReward"] += time.time() - startTime2
        except:
            self.avgtimings["computeReward"] = time.time() - startTime2


        startTime2 = time.time()
        ob = self._get_obs()
        s = self.state_vector()
        #print("obs and state took " + str(time.time() - startTime2))
        try:
            self.avgtimings["obs"] += time.time() - startTime2
        except:
            self.avgtimings["obs"] = time.time() - startTime2


        #update physx capsules
        self.updateClothCollisionStructures(hapticSensors=True)

        done, terminationReward = self.checkTermination(tau, s, ob)
        reward += terminationReward
        self.reward = reward
        self.cumulativeReward += self.reward
        self.rewardTrajectory.append(self.reward)

        #if done and terminationReward < 0:
        #    print("terminated negatively. reward trajectory: " + str(self.rewardTrajectory))

        self.numSteps += 1
        #print("_step took " + str(time.time() - startTime))
        try:
            self.avgtimings["_step"] += time.time() - startTime2
        except:
            self.avgtimings["_step"] = time.time() - startTime2
        return ob, self.reward, done, {}
        #except:
        #    print("step " + str(self.numSteps) + " failed")
            #self.step(action=np.zeros(len(a)))

    def _get_obs(self):
        print("base observation")
        return np.zeros(self.obs_size)

    def additionalResets(self):
        #do any additional reseting here
        a=0

    def reset_model(self):
        self.rewardsData.reset()

        if self.graphingRandomness:
            self.initialRand = 0
            self.initialnpRand = 0
            self.initialselfnpRand = 0
            self.randomnessGraph.xdata = np.arange(100)
            self.randomnessGraph.plotData(ydata=np.zeros(100))
            self.randomnessGraph.plotData(ydata=np.zeros(100))
            self.randomnessGraph.plotData(ydata=np.zeros(100))

        if self.graphPoseSum:
            self.poseSumGraph.xdata = np.arange(100)
            self.poseSumGraph.plotData(ydata=np.zeros(100))

        seeds=[]
        #seeds = [0, 2, 5, 8, 11, 20, 27, 35, 36, 47, 50, 51] #success seeds for stochastic policy
        #seeds = [0, 1, 2, 3, 5, 8, 11, 12, 13, 14, 18, 19, 20, 23, 27, 35, 38, 50] #success seeds for mean policy
        #difficultySeeds = [37, 39, 42]
        #seeds = seeds+difficultySeeds
        #seed = self.reset_number
        #print(seeds)
        try:
            seed = seeds[self.reset_number]
        except:
            seed = self.reset_number
            #print("all given seeds simulated")
        #seed = 8
        #print("rollout: " + str(self.reset_number+1) +", seed: " + str(seed))
        #random.seed(seed)
        #self.np_random.seed(seed)
        #np.random.seed(seed)
        #self.clothScene.seedRandom(seed) #unecessary

        #print("random.random(): " + str(random.random()))
        #print("np.random.random: " + str(np.random.random()))
        #print("self.np_random.random: " + str(self.np_random.uniform() ))
        if self.graphClothViolation:
            if self.saveClothViolationGraphOnReset:
                self.clothViolationGraph.save(filename="clothViolationGraphRS"+str(self.reset_number))
            self.clothViolationGraph.close()
            self.clothViolationGraph = pyutils.LineGrapher(title="Cloth Violation", numPlots=3)

        self.rewardTrajectory = []
        startTime = time.time()
        #try:
        #print("reset")
        self.cumulativeReward = 0
        self.dart_world.reset()
        self.clothScene.setSelfCollisionDistance(0.03)
        self.clothScene.reset()
        #self.clothScene.setFriction(0, 0.4)

        self.additionalResets()

        #SPD
        if self.SPD is None:
            self.SPD = SPDController(env=self)
        if self.SPDTarget is not None:
            self.SPD.setup()

        #update physx capsules
        self.updateClothCollisionStructures(hapticSensors=True)
        self.clothScene.clearInterpolation()

        if self.recordROMPoints:
            if len(self.ROMPoints) > 1:
                pyutils.saveList(self.ROMPoints, filename="ROMPoints", listoflists=True)

        '''if self.numSteps > 0:
            print("reset_model took " + str(time.time()-startTime))
            for item in self.avgtimings.items():
                print("    " + str(item[0] + " took " + str(item[1]/self.numSteps)))
        '''

        self.avgtimings = {}
        self.reset_number += 1
        self.numSteps = 0

        #if self.reset_number == 1:
        #    self.reset()
        #print("now entering rollout reset_number: " + str(self.reset_number))
        return self._get_obs()
        #except:
        #    print("Failed on reset " + str(self.reset_number))

    def updateClothCollisionStructures(self, capsules=False, hapticSensors=False):
        a=0
        #collision spheres creation
        fingertip = np.array([0.0, -0.06, 0.0])
        z = np.array([0.,0,0])
        cs0 = self.robot_skeleton.bodynodes[1].to_world(z)
        cs1 = self.robot_skeleton.bodynodes[2].to_world(z)
        cs2 = self.robot_skeleton.bodynodes[14].to_world(z)
        cs3 = self.robot_skeleton.bodynodes[14].to_world(np.array([0,0.175,0]))
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
        csVars0 = np.array([0.15, -1, -1, 0,0,0])
        csVars1 = np.array([0.07, -1, -1, 0,0,0])
        csVars2 = np.array([0.1, -1, -1, 0,0,0])
        csVars3 = np.array([0.1, -1, -1, 0,0,0])
        csVars4 = np.array([0.065, -1, -1, 0,0,0])
        csVars5 = np.array([0.05, -1, -1, 0,0,0])
        csVars6 = np.array([0.0365, -1, -1, 0,0,0])
        csVars7 = np.array([0.04, -1, -1, 0,0,0])
        csVars8 = np.array([0.046, -1, -1, 0,0,0])
        csVars9 = np.array([0.065, -1, -1, 0,0,0])
        csVars10 = np.array([0.05, -1, -1, 0,0,0])
        csVars11 = np.array([0.0365, -1, -1, 0,0,0])
        csVars12 = np.array([0.04, -1, -1, 0,0,0])
        csVars13 = np.array([0.046, -1, -1, 0,0,0])
        collisionSpheresInfo = np.concatenate([cs0, csVars0, cs1, csVars1, cs2, csVars2, cs3, csVars3, cs4, csVars4, cs5, csVars5, cs6, csVars6, cs7, csVars7, cs8, csVars8, cs9, csVars9, cs10, csVars10, cs11, csVars11, cs12, csVars12, cs13, csVars13]).ravel()

        #inflate collision objects
        #for i in range(int(len(collisionSpheresInfo)/9)):
        #    collisionSpheresInfo[i*9 + 3] *= 1.15

        self.collisionSphereInfo = np.array(collisionSpheresInfo)
        #collisionSpheresInfo = np.concatenate([cs0, csVars0, cs1, csVars1]).ravel()
        if np.isnan(np.sum(collisionSpheresInfo)): #this will keep nans from propagating into PhysX resulting in segfault on reset()
            return
        self.clothScene.setCollisionSpheresInfo(collisionSpheresInfo)
        
        if capsules is True:
            #collision capsules creation
            collisionCapsuleInfo = np.zeros((14,14))
            collisionCapsuleInfo[0,1] = 1
            collisionCapsuleInfo[1,2] = 1
            collisionCapsuleInfo[1,4] = 1
            collisionCapsuleInfo[1,9] = 1
            collisionCapsuleInfo[2,3] = 1
            collisionCapsuleInfo[4,5] = 1
            collisionCapsuleInfo[5,6] = 1
            collisionCapsuleInfo[6,7] = 1
            collisionCapsuleInfo[7,8] = 1
            collisionCapsuleInfo[9,10] = 1
            collisionCapsuleInfo[10,11] = 1
            collisionCapsuleInfo[11,12] = 1
            collisionCapsuleInfo[12,13] = 1
            collisionCapsuleBodynodes = -1 * np.ones((14,14))
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
            #hapticSensorLocations = np.concatenate([cs0, LERP(cs0, cs1, 0.33), LERP(cs0, cs1, 0.66), cs1, LERP(cs1, cs2, 0.33), LERP(cs1, cs2, 0.66), cs2, LERP(cs2, cs3, 0.33), LERP(cs2, cs3, 0.66), cs3])
            #hapticSensorLocations = np.concatenate([cs0, LERP(cs0, cs1, 0.25), LERP(cs0, cs1, 0.5), LERP(cs0, cs1, 0.75), cs1, LERP(cs1, cs2, 0.25), LERP(cs1, cs2, 0.5), LERP(cs1, cs2, 0.75), cs2, LERP(cs2, cs3, 0.25), LERP(cs2, cs3, 0.5), LERP(cs2, cs3, 0.75), cs3])
            hapticSensorLocations = np.concatenate([cs0, cs1, cs2, cs3, cs4, LERP(cs4, cs5, 0.33), LERP(cs4, cs5, 0.66), cs5, LERP(cs5, cs6, 0.33), LERP(cs5,cs6,0.66), cs6, cs7, cs8, cs9, LERP(cs9, cs10, 0.33), LERP(cs9, cs10, 0.66), cs10, LERP(cs10, cs11, 0.33), LERP(cs10, cs11, 0.66), cs11, cs12, cs13])
            hapticSensorRadii = np.array([csVars0[0], csVars1[0], csVars2[0], csVars3[0], csVars4[0], LERP(csVars4[0], csVars5[0], 0.33), LERP(csVars4[0], csVars5[0], 0.66), csVars5[0], LERP(csVars5[0], csVars6[0], 0.33), LERP(csVars5[0], csVars6[0], 0.66), csVars6[0], csVars7[0], csVars8[0], csVars9[0], LERP(csVars9[0], csVars10[0], 0.33), LERP(csVars9[0], csVars10[0], 0.66), csVars10[0], LERP(csVars10[0], csVars11[0], 0.33), LERP(csVars10[0], csVars11[0], 0.66), csVars11[0], csVars12[0], csVars13[0]])
            self.clothScene.setHapticSensorLocations(hapticSensorLocations)
            self.clothScene.setHapticSensorRadii(hapticSensorRadii)

    def getViewer(self, sim, title=None, extraRenderFunc=None, inputFunc=None, resetFunc=None):
        return DartClothEnv.getViewer(self, sim, title, self.extraRenderFunction, self.inputFunc, self.reset_model)

    def inputFunc(self, repeat=False):
        pyutils.inputGenie(domain=self, repeat=repeat)

    def extraRenderFunction(self):
        '''This function is overwritten by child classes'''
        renderUtils.setColor(color=[0.0, 0.0, 0])
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3d(0,0,0)
        GL.glVertex3d(-1,0,0)
        GL.glEnd()

        renderUtils.setColor([0,0,0])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[4].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[4].to_world(np.array([0.0,-0.3,-0.075]))])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[9].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[9].to_world(np.array([0.0,-0.3,-0.075]))])

        textHeight = 15
        textLines = 2

        # SPD pose rendering
        if self.SPDTarget is not None:
            links = pyutils.getRobotLinks(self.robot_skeleton, pose=self.SPDTarget)
            renderUtils.drawLines(lines=links)

        if self.renderUI:
            renderUtils.setColor(color=[0.,0,0])
            if self.totalTime > 0:
                self.clothScene.drawText(x=15., y=textLines*textHeight, text="Steps = " + str(self.numSteps) + " framerate = " + str(self.numSteps/self.totalTime), color=(0., 0, 0))
                textLines += 1
            self.clothScene.drawText(x=15., y=textLines*textHeight, text="Reward = " + str(self.reward), color=(0., 0, 0))
            textLines += 1
            self.clothScene.drawText(x=15., y=textLines * textHeight, text="Cumulative Reward = " + str(self.cumulativeReward), color=(0., 0, 0))
            textLines += 1
            if self.numSteps > 0:
                renderUtils.renderDofs(robot=self.robot_skeleton, restPose=None, renderRestPose=False)


    def viewer_setup(self):
        if self._get_viewer().scene is not None:
            self._get_viewer().scene.tb.trans[2] = -3.5
            self._get_viewer().scene.tb._set_theta(180)
            self._get_viewer().scene.tb._set_phi(180)
        self.track_skeleton_id = 0
        if not self.renderDARTWorld:
            self.viewer.renderWorld = False
        self.clothScene.renderCollisionCaps = True
        self.clothScene.renderCollisionSpheres = True




