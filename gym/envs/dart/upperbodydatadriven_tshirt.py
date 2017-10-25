# This environment is created by Alexander Clegg (alexanderwclegg@gmail.com)

import numpy as np
from gym import utils
from gym.envs.dart.dart_cloth_env import *
import random
import time
import math

from pyPhysX.colors import *
import pyPhysX.pyutils as pyutils
import pyPhysX.renderUtils
import pyPhysX.meshgraph as meshgraph
from pyPhysX.clothfeature import *

import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT

class DartClothUpperBodyDataDrivenTshirtEnv(DartClothEnv, utils.EzPickle):
    def __init__(self):
        self.prefix = os.path.dirname(__file__)

        #rendering variables
        self.useOpenGL = False
        self.screenSize = (1080, 720)
        self.renderDARTWorld = True
        self.renderUI = True
        self.renderDisplacerAccuracy = False
        self.compoundAccuracy = True

        #sim variables
        self.gravity = False
        self.resetRandomPose = False
        self.dataDrivenJointLimts = False
        simulateCloth = True

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

        #task modes
        self.upright_active = False
        self.rightDisplacer_active = False
        self.leftDisplacer_active = False
        self.displacerMod1 = False #temporary switch to modified reward scheme
        self.upReacher_active = False
        self.rightTarget_active = False
        self.leftTarget_active = False
        self.prevTauObs = False #if True, T(t-1) is included in the obs

        self.rightDisplacement = np.zeros(3) #should be unit vector or 0
        self.leftDisplacement = np.zeros(3) #should be unit vector or 0
        self.displacementParameters = [
            0.3,    #probability of 0 vector on reset
            0.01,   #probability of displacement vector reset (each step)
            0.25,   #probability of displacement vector shift (add noise to vector and re-normalize)
            0.05,   #minimum noise magnitude
            0.1     #maximum noise magnitude
        ]
        self.cumulativeAccurateMotionR = 0
        self.cumulativeMotionR = 0
        self.cumulativeFixedMotionR = 0
        self.cumulativeFixedTimeR = 0
        self.cumulativeAccurateMotionL = 0
        self.cumulativeMotionL = 0
        self.cumulativeFixedMotionL = 0
        self.cumulativeFixedTimeL = 0
        self.displacer0TargetR = np.zeros(3) #location of the ef when (0,0,0) task is activated

        self.rightTarget = np.zeros(3)
        self.leftTarget = np.zeros(3)

        #22 dof upper body
        self.action_scale = np.ones(len(self.actuatedDofs))*12
        if 0 in self.actuatedDofs:
            self.action_scale[self.actuatedDofs.tolist().index(0)] = 50
        if 1 in self.actuatedDofs:
            self.action_scale[self.actuatedDofs.tolist().index(1)] = 50

        self.control_bounds = np.array([np.ones(len(self.actuatedDofs)), np.ones(len(self.actuatedDofs))*-1])
        self.prevTau = np.zeros(len(self.actuatedDofs))

        self.reset_number = 0 #debugging
        self.numSteps = 0

        self.hapticObs = False
        observation_size = len(self.actuatedDofs)*3 #q(sin,cos), dq
        if self.prevTauObs:
            observation_size += len(self.actuatedDofs)
        if self.hapticObs:
            observation_size += 66 #TODO: downsize this as necessary
        if self.rightDisplacer_active:
            observation_size += 3
        if self.leftDisplacer_active:
            observation_size += 3
        if self.rightTarget_active:
            observation_size += 6
        if self.leftTarget_active:
            observation_size += 6


        #model_path = 'UpperBodyCapsules_v3.skel'
        model_path = 'UpperBodyCapsules_datadriven.skel'

        #create cloth scene
        clothScene = pyphysx.ClothScene(step=0.01,
                                        mesh_path=self.prefix + "/assets/tshirt_m.obj",
                                        state_path=self.prefix + "/../../../../tshirt_regrip3.obj",
                                        scale=1.4)

        clothScene.togglePinned(0, 0)  # turn off auto-pin

        self.separatedMesh = meshgraph.MeshGraph(clothscene=clothScene)

        #TODO: add other sleeve and check/rename this
        self.CP0Feature = ClothFeature(
            verts=[413, 1932, 1674, 1967, 475, 1517, 828, 881, 1605, 804, 1412, 1970, 682, 469, 155, 612, 1837, 531],
            clothScene=clothScene)

        self.displacerTargets = [[], []]
        self.displacerActual = [[], []]

        self.reward=0

        self.prevTime = time.time()
        self.totalTime = 0

        #intialize the parent env
        if self.useOpenGL is True:
            DartClothEnv.__init__(self, cloth_scene=clothScene, model_paths=model_path, frame_skip=4,
                                  observation_size=observation_size, action_bounds=self.control_bounds, screen_width=self.screenSize[0], screen_height=self.screenSize[1])
        else:
            DartClothEnv.__init__(self, cloth_scene=clothScene, model_paths=model_path, frame_skip=4,
                                  observation_size=observation_size, action_bounds=self.control_bounds , disableViewer = True, visualize = False)

        #setup data-driven joint limits
        if self.dataDrivenJointLimts:
            leftarmConstraint = pydart.constraints.HumanArmJointLimitConstraint(self.robot_skeleton.joint('j_bicep_left'), self.robot_skeleton.joint('elbowjL'), False)
            rightarmConstraint = pydart.constraints.HumanArmJointLimitConstraint(self.robot_skeleton.joint('j_bicep_right'), self.robot_skeleton.joint('elbowjR'), True)
            leftarmConstraint.add_to_world(self.dart_world)
            rightarmConstraint.add_to_world(self.dart_world)

        utils.EzPickle.__init__(self)

        if not self.gravity:
            self.dart_world.set_gravity(np.zeros(3))
        else:
            self.dart_world.set_gravity(np.array([0., -9.8, 0]))
        
        #self.clothScene.seedRandom(random.randint(1,1000))
        self.clothScene.setFriction(0, 0.5)
        
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

        #TODO: make this more generic
        self.torqueGraph = None#pyutils.LineGrapher(title="Torques")

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

    def _getFile(self):
        return __file__

    def _step(self, a):
        #print("a: " + str(a))
        if self.numSteps > 1:
            self.totalTime += (time.time() - self.prevTime)
            #print(self.totalTime)
        self.prevTime = time.time()
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        self.prevTau = np.array(clamped_control)
        tau = np.multiply(clamped_control, self.action_scale)

        if self.reset_number > 0 and self.torqueGraph is not None:
            self.torqueGraph.yData[0][self.numSteps - 1] = tau[0]
            self.torqueGraph.yData[1][self.numSteps - 1] = tau[1]
            self.torqueGraph.update()

        fingertip = np.array([0.0, -0.06, 0.0])
        wRFingertip1 = self.robot_skeleton.bodynodes[7].to_world(fingertip)
        wLFingertip1 = self.robot_skeleton.bodynodes[12].to_world(fingertip)
        #vecR1 = self.target-wRFingertip1
        #vecL1 = self.target2-wLFingertip1
        
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

        wRFingertip2 = self.robot_skeleton.bodynodes[7].to_world(fingertip)
        wLFingertip2 = self.robot_skeleton.bodynodes[12].to_world(fingertip)
        #vecR2 = self.target-wRFingertip2
        #vecL2 = self.target2-wLFingertip2
        
        #force magnitude penalty    
        reward_ctrl = -np.square(tau).sum()

        #reward for maintaining posture
        reward_upright = 0
        if self.upright_active:
            reward_upright = -abs(self.robot_skeleton.q[0])-abs(self.robot_skeleton.q[1])

        #reward for reaching up with both arms.
        reward_upreach = 0
        if self.upReacher_active:
            reward_upreach = wRFingertip2[1] + wLFingertip2[1]

        #reward for following displacement goals
        reward_displacement = 0
        if self.rightDisplacer_active:
            actual_displacement = wRFingertip2 - wRFingertip1
            if np.linalg.norm(self.rightDisplacement) == 0:
                if self.displacerMod1:
                    reward_displacement += -np.linalg.norm(wRFingertip2-self.displacer0TargetR)
                else:
                    reward_displacement += -np.linalg.norm(actual_displacement)
            else:
                reward_displacement += actual_displacement.dot(self.rightDisplacement)
        if self.leftDisplacer_active:
            actual_displacement = wLFingertip2 - wLFingertip1
            if np.linalg.norm(self.leftDisplacement) == 0:
                reward_displacement += -np.linalg.norm(actual_displacement)
            else:
                reward_displacement += actual_displacement.dot(self.leftDisplacement)

        reward_target = 0
        if self.rightTarget_active:
            targetDistR = np.linalg.norm(self.rightTarget-wRFingertip2)
            reward_target -= targetDistR
            if targetDistR < 0.01:
                reward_target += 0.5
        if self.leftTarget_active:
            targetDistL = np.linalg.norm(self.leftTarget-wLFingertip2)
            reward_target -= targetDistL
            if targetDistL < 0.01:
                reward_target += 0.5

        #total reward
        self.reward = reward_ctrl*0 + reward_upright + reward_upreach + reward_displacement + reward_target

        #record accuracy
        if self.renderDisplacerAccuracy and self.useOpenGL and self.numSteps>0:
            vecR = wRFingertip2 - wRFingertip1
            dispMagR = np.linalg.norm(vecR)
            if self.compoundAccuracy and len(self.displacerTargets[0]) > 0:
                self.displacerTargets[0].append(self.displacerTargets[0][-1] + self.rightDisplacement * dispMagR)
            else:
                self.displacerTargets[0].append(wRFingertip1+self.rightDisplacement*dispMagR)
            self.displacerActual[0].append(wRFingertip2)
            if np.linalg.norm(self.rightDisplacement) > 0:
                self.cumulativeMotionR += dispMagR
                self.cumulativeAccurateMotionR += vecR.dot(self.rightDisplacement)
            else:
                self.cumulativeFixedMotionR += dispMagR
                self.cumulativeFixedTimeR += 1
            vecL = wLFingertip2 - wLFingertip1
            dispMagL = np.linalg.norm(vecL)
            if self.compoundAccuracy and len(self.displacerTargets[1]) > 0:
                self.displacerTargets[1].append(self.displacerTargets[1][-1] + self.leftDisplacement * dispMagL)
            else:
                self.displacerTargets[1].append(wLFingertip1 + self.leftDisplacement * dispMagL)
            self.displacerActual[1].append(wLFingertip2)
            if np.linalg.norm(self.leftDisplacement) > 0:
                self.cumulativeMotionL += dispMagL
                self.cumulativeAccurateMotionL += vecL.dot(self.leftDisplacement)
            else:
                self.cumulativeFixedMotionL += dispMagR
                self.cumulativeFixedTimeL += 1

        #compute changes in displacements before the next observation phase
        if self.rightDisplacer_active:
            if random.random() < self.displacementParameters[1]: #reset vector
                if random.random() < self.displacementParameters[0]:
                    self.rightDisplacement = np.zeros(3)
                    self.displacer0TargetR = np.array(wRFingertip2)
                else:
                    self.rightDisplacement = pyutils.sampleDirections(num=1)[0]
            elif random.random() < self.displacementParameters[2] and np.linalg.norm(self.rightDisplacement) > 0: #add noise to vector
                noise = pyutils.sampleDirections(num=1)[0]
                noise *= LERP(self.displacementParameters[3], self.displacementParameters[4], random.random())
                self.rightDisplacement += noise
                self.rightDisplacement /= np.linalg.norm(self.rightDisplacement)
        if self.leftDisplacer_active:
            if random.random() < self.displacementParameters[1]: #reset vector
                if random.random() < self.displacementParameters[0]:
                    self.leftDisplacement = np.zeros(3)
                else:
                    self.leftDisplacement = pyutils.sampleDirections(num=1)[0]
            elif random.random() < self.displacementParameters[2] and np.linalg.norm(self.leftDisplacement) > 0: #add noise to vector
                noise = pyutils.sampleDirections(num=1)[0]
                noise *= LERP(self.displacementParameters[3], self.displacementParameters[4], random.random())
                self.leftDisplacement += noise
                self.leftDisplacement /= np.linalg.norm(self.leftDisplacement)

        ob = self._get_obs()
        s = self.state_vector()
        
        #update physx capsules
        self.updateClothCollisionStructures(hapticSensors=True)
        
        #check cloth deformation for termination
        clothDeformation = 0
        if self.simulateCloth is True:
            clothDeformation = self.clothScene.getMaxDeformationRatio(0)
        
        #check termination conditions
        done = False
        if not np.isfinite(s).all():
            print("Infinite value detected..." + str(s))
            done = True
            self.reward -= 500
        elif (clothDeformation > 20):
            done = True
            self.reward -= 500
        #increment the step counter
        self.numSteps += 1
        
        return ob, self.reward, done, {}

    def _get_obs(self):
        f_size = 66
        '22x3 dofs, 22x3 sensors, 7x2 targets(toggle bit, cartesian, relative)'
        theta = np.zeros(len(self.actuatedDofs))
        dtheta = np.zeros(len(self.actuatedDofs))
        for ix,dof in enumerate(self.actuatedDofs):
            theta[ix] = self.robot_skeleton.q[dof]
            dtheta[ix] = self.robot_skeleton.dq[dof]

        fingertip = np.array([0.0, -0.06, 0.0])
        #vec = self.robot_skeleton.bodynodes[8].to_world(fingertip) - self.target
        #vec2 = self.robot_skeleton.bodynodes[14].to_world(fingertip) - self.target2

        obs = np.concatenate([np.cos(theta), np.sin(theta), dtheta]).ravel()

        if self.prevTauObs:
            obs = np.concatenate([obs, self.prevTau])

        if self.hapticObs:
            f = None
            if self.simulateCloth is True:
                f = self.clothScene.getHapticSensorObs()#get force from simulation
            else:
                f = np.zeros(f_size)
            obs = np.concatenate([obs, f]).ravel()

        if self.rightDisplacer_active:
            obs = np.concatenate([obs, self.rightDisplacement]).ravel()
        if self.leftDisplacer_active:
            obs = np.concatenate([obs, self.leftDisplacement]).ravel()
        if self.rightTarget_active:
            shoulderR = self.robot_skeleton.bodynodes[4].to_world(np.zeros(3))
            efR = self.robot_skeleton.bodynodes[7].to_world(fingertip)
            obs = np.concatenate([obs, shoulderR-self.rightTarget, efR-self.rightTarget]).ravel()
        if self.leftTarget_active:
            shoulderL = self.robot_skeleton.bodynodes[9].to_world(np.zeros(3))
            efL = self.robot_skeleton.bodynodes[12].to_world(fingertip)
            obs = np.concatenate([obs, shoulderL-self.leftTarget, efL-self.leftTarget]).ravel()
        return obs

    def reset_model(self):
        #print("reset")
        self.totalTime = 0
        self.cumulativeReward = 0
        self.dart_world.reset()
        self.clothScene.reset()
        #self.clothScene.translateCloth(0, np.array([-0, 1.0, 0]))
        #self.clothScene.translateCloth(0, np.array([-10.5,0,0]))
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        if self.resetRandomPose:
            qpos = pyutils.getRandomPose(self.robot_skeleton)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        if self.resetRandomPose:
            self.dart_world.check_collision()
            while self.dart_world.collision_result.num_contacts() > 0:
                qpos = pyutils.getRandomPose(self.robot_skeleton)
                self.set_state(qpos, qvel)
                self.dart_world.check_collision()
        #print(self.dart_world.collision_result.num_contacts())

        if random.random() < self.displacementParameters[0]:
            self.rightDisplacement = np.zeros(3)
            wRFingertip2 = self.robot_skeleton.bodynodes[7].to_world(np.array([0,-0.06,0]))
            self.displacer0TargetR = np.array(wRFingertip2)
        else:
            self.rightDisplacement = pyutils.sampleDirections(num=1)[0]
        if random.random() < self.displacementParameters[0]:
            self.leftDisplacement = np.zeros(3)
        else:
            self.leftDisplacement = pyutils.sampleDirections(num=1)[0]

        armLength = 0.75
        if self.rightTarget_active:
            self.rightTarget = self.robot_skeleton.bodynodes[4].to_world(np.zeros(3))+pyutils.sampleDirections(1)[0]*random.random()*armLength
        if self.leftTarget_active:
            self.leftTarget = self.robot_skeleton.bodynodes[9].to_world(np.zeros(3))+pyutils.sampleDirections(1)[0]*random.random()*armLength

        self.clothScene.setSelfCollisionDistance(0.025)

        #update physx capsules
        self.updateClothCollisionStructures(hapticSensors=True)
        self.clothScene.clearInterpolation()

        #debugging
        self.reset_number += 1
        self.numSteps = 0

        if self.torqueGraph is not None:
            xdata = np.arange(400)
            self.torqueGraph.xdata = xdata
            initialYData0 = np.zeros(400)
            initialYData1 = np.zeros(400)
            self.torqueGraph.plotData(ydata=initialYData0)
            self.torqueGraph.plotData(ydata=initialYData1)

        self.displacerTargets = [[], []]
        self.displacerActual = [[], []]
        #self.cumulativeMotionR = 0.00001
        #self.cumulativeAccurateMotionR = 0
        #self.cumulativeMotionL = 0.00001
        #self.cumulativeAccurateMotionL = 0

        return self._get_obs()

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
        csVars8 = np.array([0.036, -1, -1, 0,0,0])
        csVars9 = np.array([0.065, -1, -1, 0,0,0])
        csVars10 = np.array([0.05, -1, -1, 0,0,0])
        csVars11 = np.array([0.0365, -1, -1, 0,0,0])
        csVars12 = np.array([0.04, -1, -1, 0,0,0])
        csVars13 = np.array([0.036, -1, -1, 0,0,0])
        collisionSpheresInfo = np.concatenate([cs0, csVars0, cs1, csVars1, cs2, csVars2, cs3, csVars3, cs4, csVars4, cs5, csVars5, cs6, csVars6, cs7, csVars7, cs8, csVars8, cs9, csVars9, cs10, csVars10, cs11, csVars11, cs12, csVars12, cs13, csVars13]).ravel()
        #collisionSpheresInfo = np.concatenate([cs0, csVars0, cs1, csVars1]).ravel()
        
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
            self.clothScene.setCollisionCapsuleInfo(collisionCapsuleInfo)
            
        if hapticSensors is True:
            #hapticSensorLocations = np.concatenate([cs0, LERP(cs0, cs1, 0.33), LERP(cs0, cs1, 0.66), cs1, LERP(cs1, cs2, 0.33), LERP(cs1, cs2, 0.66), cs2, LERP(cs2, cs3, 0.33), LERP(cs2, cs3, 0.66), cs3])
            #hapticSensorLocations = np.concatenate([cs0, LERP(cs0, cs1, 0.25), LERP(cs0, cs1, 0.5), LERP(cs0, cs1, 0.75), cs1, LERP(cs1, cs2, 0.25), LERP(cs1, cs2, 0.5), LERP(cs1, cs2, 0.75), cs2, LERP(cs2, cs3, 0.25), LERP(cs2, cs3, 0.5), LERP(cs2, cs3, 0.75), cs3])
            hapticSensorLocations = np.concatenate([cs0, cs1, cs2, cs3, cs4, LERP(cs4, cs5, 0.33), LERP(cs4, cs5, 0.66), cs5, LERP(cs5, cs6, 0.33), LERP(cs5,cs6,0.66), cs6, cs7, cs8, cs9, LERP(cs9, cs10, 0.33), LERP(cs9, cs10, 0.66), cs10, LERP(cs10, cs11, 0.33), LERP(cs10, cs11, 0.66), cs11, cs12, cs13])
            self.clothScene.setHapticSensorLocations(hapticSensorLocations)
            
    def getViewer(self, sim, title=None, extraRenderFunc=None, inputFunc=None):
        return DartClothEnv.getViewer(self, sim, title, self.extraRenderFunction, self.inputFunc)

    def inputFunc(self, repeat=False):
        pyutils.inputGenie(domain=self, repeat=repeat)

    def extraRenderFunction(self):
        renderUtils.setColor(color=[0.0, 0.0, 0])
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3d(0,0,0)
        GL.glVertex3d(-1,0,0)
        GL.glEnd()

        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[4].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[4].to_world(np.array([0.0,-0.3,-0.075]))])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[9].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[9].to_world(np.array([0.0,-0.3,-0.075]))])

        #render sample range
        #renderUtils.drawSphere(pos=self.robot_skeleton.bodynodes[4].to_world(np.zeros(3)), rad=0.75, solid=False)

        #render targets
        if self.rightTarget_active:
            renderUtils.setColor(color=[1.0,1.0,0])
            renderUtils.drawSphere(pos=self.rightTarget, rad=0.05)
            renderUtils.drawLineStrip(points=[self.rightTarget, self.robot_skeleton.bodynodes[8].to_world(np.array([0,-0.06,0]))])
        if self.leftTarget_active:
            renderUtils.setColor(color=[0.0, 1.0, 1.0])
            renderUtils.drawSphere(pos=self.leftTarget, rad=0.05)
            renderUtils.drawLineStrip(points=[self.leftTarget, self.robot_skeleton.bodynodes[14].to_world(np.array([0, -0.06, 0]))])

        if self.rightDisplacer_active:
            renderUtils.setColor(color=[0.0,0.0,1.0])
            ef = self.robot_skeleton.bodynodes[7].to_world(np.array([0.0, -0.06, 0.0]))
            if np.linalg.norm(self.rightDisplacement) == 0:
                renderUtils.drawBox(cen=ef,dim=[0.2,0.2,0.2], fill=False)
            else:
                renderUtils.drawArrow(p0=ef, p1=ef+self.rightDisplacement*0.15, hwRatio=0.15)
            if self.renderDisplacerAccuracy:
                renderUtils.setColor(color=[1.0,0,0])
                renderUtils.drawLineStrip(self.displacerTargets[0])
                renderUtils.setColor(color=[0.0, 0, 1.0])
                renderUtils.drawLineStrip(self.displacerActual[0])
                self.clothScene.drawText(x=15., y=90., text="Disp. Acc. R = " + str(self.cumulativeAccurateMotionR/self.cumulativeMotionR), color=(0., 0, 0))
                if self.cumulativeFixedTimeR > 0:
                    self.clothScene.drawText(x=15., y=105., text="Fixed Motion R = " + str(self.cumulativeFixedMotionR/self.cumulativeFixedTimeR), color=(0., 0, 0))
        if self.leftDisplacer_active:
            renderUtils.setColor(color=[0.0, 0.0, 1.0])
            ef = self.robot_skeleton.bodynodes[12].to_world(np.array([0.0, -0.06, 0.0]))
            if np.linalg.norm(self.leftDisplacement) == 0:
                renderUtils.drawBox(cen=ef, dim=[0.2, 0.2, 0.2], fill=False)
            else:
                renderUtils.drawArrow(p0=ef, p1=ef+self.leftDisplacement*0.15, hwRatio=0.15)
            if self.renderDisplacerAccuracy:
                renderUtils.setColor(color=[1.0,1.0,0])
                renderUtils.drawLineStrip(self.displacerTargets[1])
                renderUtils.setColor(color=[0.0, 1.0, 1.0])
                renderUtils.drawLineStrip(self.displacerActual[1])
                self.clothScene.drawText(x=15., y=60., text="Disp. Acc. L = " + str(self.cumulativeAccurateMotionL / self.cumulativeMotionL), color=(0., 0, 0))
                if self.cumulativeFixedTimeL > 0:
                    self.clothScene.drawText(x=15., y=75., text="Fixed Motion L = " + str(self.cumulativeFixedMotionL / self.cumulativeFixedTimeL), color=(0., 0, 0))

        if self.renderUI:
            renderUtils.setColor(color=[0.,0,0])
            if self.totalTime > 0:
                self.clothScene.drawText(x=15., y=30., text="Steps = " + str(self.numSteps) + " framerate = " + str(self.numSteps/self.totalTime), color=(0., 0, 0))
            self.clothScene.drawText(x=15., y=45., text="Reward = " + str(self.reward), color=(0., 0, 0))
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

def LERP(p0, p1, t):
    return p0 + (p1-p0)*t