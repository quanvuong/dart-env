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
        a = a_info['mean']
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

class SPDController(Controller):
    def __init__(self, env, target=None):
        obs_subset = []
        policyfilename = None
        name = "SPD"
        self.target = target
        Controller.__init__(self, env, policyfilename, name, obs_subset)

        self.h = 0.01
        self.skel = env.robot_skeleton
        ndofs = self.skel.ndofs-6
        self.qhat = self.skel.q
        self.Kp = np.diagflat([400.0] * (ndofs))
        self.Kd = np.diagflat([40.0] * (ndofs))

        '''
        for i in range(ndofs):
            if i ==9 or i==10 or i==17 or i==18:
                self.Kd[i][i] *= 0.01
                self.Kp[i][i] *= 0.01
        '''

        #print(self.Kp)
        self.preoffset = 0.0

    def setup(self):
        #reset the target
        #cur_q = np.array(self.skel.q)
        #self.env.loadCharacterState(filename="characterState_regrip")
        self.target = np.array(self.skel.q[6:])
        #self.env.restPose = np.array(self.target)
        #self.target = np.array(self.skel.q)
        #self.env.robot_skeleton.set_positions(cur_q)

        a=0

    def update(self):
        #if self.env.handleNode is not None:
        #    self.env.handleNode.clearHandles();
        #    self.env.handleNode = None
        a=0

    def transition(self):
        return False

    def query(self, obs):
        #SPD
        self.qhat = self.target
        skel = self.skel
        p = -self.Kp.dot(skel.q[6:] + skel.dq[6:] * self.h - self.qhat)
        d = -self.Kd.dot(skel.dq[6:])
        b = -skel.c[6:] + p + d #+ skel.constraint_forces()
        A = skel.M[6:, 6:] + self.Kd * self.h

        x = np.linalg.solve(A, b)

        #invM = np.linalg.inv(A)
        #x = invM.dot(b)
        tau = p - self.Kd.dot(skel.dq[6:] + x * self.h)
        return tau

class DartClothFullBodyDataDrivenClothSPDTestEnv(DartClothFullBodyDataDrivenClothBaseEnv, utils.EzPickle):
    def __init__(self):
        #feature flags
        rendering = True
        clothSimulation = False
        renderCloth = True

        #other variables
        self.prevTau = None
        self.restPose = None

        self.actuatedDofs = np.arange(34)
        observation_size = 0

        DartClothFullBodyDataDrivenClothBaseEnv.__init__(self,
                                                          rendering=rendering,
                                                          screensize=(1080,920),
                                                          clothMeshFile="capri_med.obj",
                                                          clothScale=np.array([1.0,1.0,1.0]),
                                                          obs_size=observation_size,
                                                          simulateCloth=clothSimulation)


        self.simulateCloth = clothSimulation

        if not renderCloth:
            self.clothScene.renderClothFill = False
            self.clothScene.renderClothBoundary = False
            self.clothScene.renderClothWires = False

        self.SPD = SPDController(self)

    def _getFile(self):
        return __file__

    def updateBeforeSimulation(self):
        #any pre-sim updates should happen here
        self.additionalAction = self.SPD.query(None)

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
        self.prevTau = tau
        self.reward = 0
        return self.reward

    def _get_obs(self):
        obs = np.zeros(self.obs_size)
        return obs

    def additionalResets(self):
        #do any additional resetting here
        qpos = np.array(
            [-0.00469234655801, -0.0218378114573, -0.011132330496, 0.00809830385355, 0.00051861417993, 0.0584867818269,
             0.374712375814, 0.0522417260384, -0.00777676124956, 0.00230285789432, -0.00274958108859, -0.008064630425,
             0.00247294825781, -0.0093978116532, 0.195632645271, -0.00276696945071, 0.0075491687512, -0.0116846422966,
             0.00636619242284, 0.00767084047346, -0.00913509000374, 0.00857521738396, 0.199096855493, 0.00787726246678,
             -0.00760402683795, -0.00433642327146, 0.00802311463366, -0.00482248656677, 0.131248337324,
             -0.00662274635457, 0.00333416764933, 0.00546016678096, -0.00150775759695, -0.00861184703697,
             -0.000589790168521, -0.832681560131, 0.00976653127827, 2.24259637323, -0.00374506255585,
             -0.00244949106062])

        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-0.01, high=0.01, size=self.robot_skeleton.ndofs)
        # qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qpos = qpos + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)

        self.set_state(qpos, qvel)
        self.restPose = qpos

        if self.simulateCloth:
            self.clothScene.translateCloth(0, np.array([0, 3.0, 0]))

        self.SPD = SPDController(self)
        self.SPD.setup()
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

        #compute the zero moment point

        #project COM to ground plane
        groundHeight = -1.5
        projCOM = np.array(self.robot_skeleton.com())
        projCOM[1] = groundHeight
        renderUtils.drawSphere(pos=projCOM, rad=0.02)
        renderUtils.setColor(color=[1.0,0,0])
        renderUtils.drawLines(lines=[[projCOM, projCOM+np.array([0,1.0,0])]])

        # draw the convex hull of the ground contact points
        points = []

        lines = []
        if self.dart_world is not None:
            if self.dart_world.collision_result is not None:
                for contact in self.dart_world.collision_result.contacts:
                    #print(contact.point)
                    if contact.skel_id1 == 0 or contact.skel_id2 == 0:
                        lines.append([np.zeros(3), np.array(contact.point)])
                        points.append(np.array([contact.point[0],contact.point[2]]))
        renderUtils.drawLines(lines)

        if len(points) > 0:
            hull = pyutils.convexHull2D(points)
            hull3D = []
            for point in hull:
                hull3D.append(np.array([point[0], groundHeight+0.2, point[1]]))
            renderUtils.drawPolygon(hull3D)

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
