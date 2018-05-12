# This environment is created by Alexander Clegg (alexanderwclegg@gmail.com)

import numpy as np
from gym import utils
from gym.envs.dart.dart_cloth_env import *
from gym.envs.dart.fullbodydatadriven_lockedFoot_cloth_base import *
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

class DartClothFullBodyDataDrivenLockedFootClothBalanceEnv(DartClothFullBodyDataDrivenLockedFootClothBaseEnv, utils.EzPickle):
    def __init__(self):
        #feature flags
        rendering = False
        clothSimulation = False
        renderCloth = True
        self.renderRestPose = True
        frameskip = 4
        dt = 0.002

        # reward flags
        self.restPoseReward = True
        self.stableCOMReward= True

        # reward weights
        self.restPoseRewardWeight  = 1
        self.stableCOMRewardWeight = 10

        #other variables
        self.prevTau = None
        self.restPose = None
        self.fingertip = np.array([0, -0.08, 0])
        self.toeOffset = np.array([0, 0, -0.2])

        self.actuatedDofs = np.arange(34)
        observation_size = len(self.actuatedDofs) * 3  # q(sin,cos), dq
        if self.stableCOMReward:
            observation_size += 3 #world COM location

        DartClothFullBodyDataDrivenLockedFootClothBaseEnv.__init__(self,
                                                                  rendering=rendering,
                                                                  screensize=(1280,920),
                                                                  clothMeshFile="capri_med.obj",
                                                                  clothScale=np.array([1.0,1.0,1.0]),
                                                                  obs_size=observation_size,
                                                                  simulateCloth=clothSimulation,
                                                                  left_foot_locked = True,
                                                                  frameskip=frameskip,
                                                                  dt=dt)


        self.simulateCloth = clothSimulation

        if not renderCloth:
            self.clothScene.renderClothFill = False
            self.clothScene.renderClothBoundary = False
            self.clothScene.renderClothWires = False

        if self.restPoseReward:
            self.rewardsData.addReward(label="rest pose", rmin=-51.0, rmax=0, rval=0, rweight=self.restPoseRewardWeight)

        if self.stableCOMReward:
            self.rewardsData.addReward(label="stable COM", rmin=-1.0, rmax=0, rval=0, rweight=self.stableCOMRewardWeight)


    def _getFile(self):
        return __file__

    def updateBeforeSimulation(self):
        #any pre-sim updates should happen here
        a=0

    def checkTermination(self, tau, s, obs):
        #check the termination conditions and return: done,reward
        if np.amax(np.absolute(s[:len(self.robot_skeleton.q)])) > 3:
            print("Detecting potential instability")
            print(s)
            print(self.rewardTrajectory)
            print(self.stateTraj[-2:])
            return True, -500
        elif not np.isfinite(s).all():
            print("Infinite value detected..." + str(s))
            return True, -500
        return False, 0

    def computeReward(self, tau):
        #compute and return reward at the current state
        self.prevTau = tau
        reward_record = []

        reward_restPose = 0
        if self.restPoseReward and self.restPose is not None:
            dist = np.linalg.norm(self.robot_skeleton.q - self.restPose)
            reward_restPose = max(-51, -dist)
            reward_record.append(reward_restPose)

        reward_stableCOM = 0
        if self.stableCOMReward:
            footCOM = self.robot_skeleton.bodynodes[0].com()
            bodyCOM = self.robot_skeleton.com()
            projFootCOM = np.array([footCOM[0], footCOM[2]])
            projBodyCOM = np.array([bodyCOM[0], bodyCOM[2]])
            reward_stableCOM = max(-1, -np.linalg.norm(projFootCOM-projBodyCOM))
            reward_record.append(reward_stableCOM)

        # update the reward data storage
        self.rewardsData.update(rewards=reward_record)

        self.reward = reward_restPose * self.restPoseRewardWeight\
                    + reward_stableCOM * self.stableCOMRewardWeight

        return self.reward

    def _get_obs(self):
        theta = np.zeros(len(self.actuatedDofs))
        dtheta = np.zeros(len(self.actuatedDofs))
        for ix, dof in enumerate(self.actuatedDofs):
            theta[ix] = self.robot_skeleton.q[dof]
            dtheta[ix] = self.robot_skeleton.dq[dof]

        obs = np.concatenate([np.cos(theta), np.sin(theta), dtheta]).ravel()

        if self.stableCOMReward:
            #print(self.robot_skeleton.com())
            obs = np.concatenate([obs, np.array(self.robot_skeleton.com())]).ravel()

        return obs

    def additionalResets(self):
        #do any additional resetting here
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-0.01, high=0.01, size=self.robot_skeleton.ndofs)
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        #self.restPose = np.array(qpos)

        #rest pose is one foot
        self.restPose = np.array([0.402149945024, 0.26417742198, 0.694509869782, 0.0703040144608, 0.218999034272, 0.000129805833933, -0.388680623056, -1.57073590178, 0.431718882029, 1.70254936482, 0.00473949413903, -0.00927479809255, 0.136183050592, 0.297959293122, 0.00598932040189, -0.0101696256212, -0.0107320923733, 0.0036338670046, -0.00188298068761, 0.00402389215485, 1.69447708293, -0.00963879710017, -0.00346143295241, -0.00275771262968, -0.00832522365649, 0.00645267428891, -0.00205846347826, 0.00710408240934, 1.52907046286, 0.00590862627021, 9.77641432949e-05, 0.00808337306807, 0.00768925048198, 0.00379325587092])

        if self.simulateCloth:
            self.clothScene.translateCloth(0, np.array([0, 3.0, 0]))
        a=0

    def extraRenderFunction(self):
        renderUtils.setColor(color=[0.0, 0.0, 0])
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3d(0,0,0)
        GL.glVertex3d(-1,0,0)
        GL.glEnd()

        renderUtils.setColor([0,0,0])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[10].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[10].to_world(np.array([0.0,-0.3,-0.075]))])
        renderUtils.drawLineStrip(points=[self.robot_skeleton.bodynodes[15].to_world(np.array([0.0,0,-0.075])), self.robot_skeleton.bodynodes[15].to_world(np.array([0.0,-0.3,-0.075]))])

        #render the body and foot coms
        footCOM = self.robot_skeleton.bodynodes[0].com()
        bodyCOM = self.robot_skeleton.com()

        lines = [
            [np.array([footCOM[0], 1.0, footCOM[2]]), np.array([footCOM[0], -1.0, footCOM[2]])],
            [np.array([bodyCOM[0], 1.0, bodyCOM[2]]), np.array([bodyCOM[0], -1.0, bodyCOM[2]])]
        ]
        #print(lines)
        renderUtils.drawLines(lines=lines)

        if self.renderRestPose:
            links = pyutils.getRobotLinks(self.robot_skeleton, pose=self.restPose)
            renderUtils.drawLines(lines=links)

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
