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

class DartClothUpperBodyDataDrivenClothBaseEnv(DartClothEnv, utils.EzPickle):
    def __init__(self, rendering=True, screensize=(1080,720), clothMeshFile="", clothMeshStateFile=None, clothScale=1.4, obs_size=0, simulateCloth=True):
        self.prefix = os.path.dirname(__file__)

        #rendering variables
        self.useOpenGL = rendering
        self.screenSize = screensize
        self.renderDARTWorld = True
        self.renderUI = True

        #sim variables
        self.gravity = False
        self.dataDrivenJointLimts = True
        self.lockTorso = False
        self.lockSpine = False

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
        self.action_scale = np.ones(len(self.actuatedDofs))*12
        if 0 in self.actuatedDofs:
            self.action_scale[self.actuatedDofs.tolist().index(0)] = 50
        if 1 in self.actuatedDofs:
            self.action_scale[self.actuatedDofs.tolist().index(1)] = 50

        self.control_bounds = np.array([np.ones(len(self.actuatedDofs)), np.ones(len(self.actuatedDofs))*-1])

        self.reset_number = 0 #debugging
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
            DartClothEnv.__init__(self, cloth_scene=clothScene, model_paths=skelFile, frame_skip=4,
                                  observation_size=obs_size, action_bounds=self.control_bounds, screen_width=self.screenSize[0], screen_height=self.screenSize[1])
        else:
            DartClothEnv.__init__(self, cloth_scene=clothScene, model_paths=skelFile, frame_skip=4,
                                  observation_size=obs_size, action_bounds=self.control_bounds , disableViewer = True, visualize = False)

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
        #for i in range(len(self.robot_skeleton.joints)):
        #    print(self.robot_skeleton.joints[i])

        #DART does not automatically limit joints with any unlimited dofs
        self.robot_skeleton.joints[4].set_position_limit_enforced(True)
        self.robot_skeleton.joints[9].set_position_limit_enforced(True)

        if self.loadROMPoints:
            self.ROMPoints = pyutils.loadListOfVecs(filename=self.ROMFile)
            self.ROMPositions = pyutils.positionsFromPoses(self.robot_skeleton, poses=self.ROMPoints,nodes=[7,12], offsets=[np.array([0,-0.065,0]),np.array([0,-0.065,0])])
        if self.processROMPoints:
            self._processROMPoints()

    def _getFile(self):
        return __file__

    def _processROMPoints(self):
        positions = pyutils.positionsFromPoses(self.robot_skeleton,poses=self.ROMPoints,nodes=[7,12], offsets=[np.array([0,-0.065,0]),np.array([0,-0.065,0])])
        before = pyutils.averageShortestDistance(positions)#len(self.ROMPoints)
        pyutils.cullPosesFromPositionDistances(numPoses=1000, poses=self.ROMPoints, positions=positions)
        after = pyutils.averageShortestDistance(positions)#len(self.ROMPoints)
        print("before: " + str(before) + " after: " + str(after))
        pyutils.saveList(self.ROMPoints, filename="processedROMPoints", listoflists=True)

    def saveObjState(self):
        print("Trying to save the object state")
        self.clothScene.saveObjState("objState", 0)

    def saveCharacterState(self):
        print("saving character state")
        f = open("characterState", 'w')
        for ix,dof in enumerate(self.robot_skeleton.q):
            if ix > 0:
                f.write(" ")
            f.write(str(dof))
        f.close()

    def loadCharacterState(self, filename=None):
        openFile = "characterState"
        if filename is not None:
            openFile = filename
        f = open(openFile, 'r')
        for ix, line in enumerate(f):
            if ix > 0: #only want the first file line
                break
            words = line.split()
            if(len(words) != self.robot_skeleton.ndofs):
                break
            qpos = np.zeros(self.robot_skeleton.ndofs)
            for ixw, w in enumerate(words):
                qpos[ixw] = float(w)
            self.robot_skeleton.set_positions(qpos)
        f.close()

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
        if self.reset_number == 0 or not self.simulating:
            return np.zeros(self.obs_size), 0, False, {}
        try:
            clamped_control = np.array(a)
            for i in range(len(clamped_control)):
                if clamped_control[i] > self.control_bounds[0][i]:
                    clamped_control[i] = self.control_bounds[0][i]
                if clamped_control[i] < self.control_bounds[1][i]:
                    clamped_control[i] = self.control_bounds[1][i]

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

            tau = np.multiply(clamped_control, self.action_scale)

            self.updateBeforeSimulation() #any env specific updates before simulation

            #apply action and simulate
            if len(tau) < len(self.robot_skeleton.q):
                newtau = np.array(tau)
                tau = np.zeros(len(self.robot_skeleton.q))
                for ix,dof in enumerate(self.actuatedDofs):
                    tau[dof] = newtau[ix]

            self.do_simulation(tau, self.frame_skip)

            #set position and 0 velocity of locked dofs
            qpos = self.robot_skeleton.q
            qvel = self.robot_skeleton.dq
            for dof in self.lockedDofs:
                qpos[dof] = 0
                qvel[dof] = 0
            self.set_state(qpos, qvel)

            reward = self.computeReward(tau=tau)

            ob = self._get_obs()
            s = self.state_vector()

            #update physx capsules
            self.updateClothCollisionStructures(hapticSensors=True)

            done, terminationReward = self.checkTermination(tau, s, ob)
            reward += terminationReward
            self.cumulativeReward += self.reward

            self.numSteps += 1

            return ob, self.reward, done, {}
        except:
            print("step " + str(self.numSteps) + " failed")
            #self.step(action=np.zeros(len(a)))

    def _get_obs(self):
        print("base observation")
        return np.zeros(self.obs_size)

    def additionalResets(self):
        #do any additional reseting here
        a=0

    def reset_model(self):
        try:
            #print("reset")
            self.cumulativeReward = 0
            self.dart_world.reset()
            self.clothScene.reset()

            self.clothScene.setSelfCollisionDistance(0.025)

            self.additionalResets()

            #update physx capsules
            self.updateClothCollisionStructures(hapticSensors=True)
            self.clothScene.clearInterpolation()

            if self.recordROMPoints:
                if len(self.ROMPoints) > 1:
                    pyutils.saveList(self.ROMPoints, filename="ROMPoints", listoflists=True)

            self.reset_number += 1
            self.numSteps = 0
            return self._get_obs()
        except:
            print("Failed on reset " + str(self.reset_number))

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




