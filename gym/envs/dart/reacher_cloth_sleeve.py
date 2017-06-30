# This environment is created by Alexander Clegg (alexanderwclegg@gmail.com)

import numpy as np
from gym import utils
from gym.envs.dart.dart_cloth_env import *
import random
import time

import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT

class DartClothSleeveReacherEnv(DartClothEnv, utils.EzPickle):
    def __init__(self):
        self.target = np.array([0.8, -0.6, 0.6])
        self.action_scale = np.array([10, 10, 10, 10, 10])
        self.control_bounds = np.array([[1.0, 1.0, 1.0, 1.0, 1.0],[-1.0, -1.0, -1.0, -1.0, -1.0]])
        
        #create cloth scene
        clothScene = pyphysx.ClothScene(step=0.01, mesh_path="/home/alexander/Documents/dev/dart-env/gym/envs/dart/assets/test_sleeve_plane.obj", scale = 0.75)
        clothScene.togglePinned(0,119)
        clothScene.togglePinned(0,109)
        clothScene.togglePinned(0,10)
        
        #clothScene = pyphysx.ClothScene(step=0.01, mesh_path="/home/alexander/Documents/dev/dart-env/gym/envs/dart/assets/square1024.obj")
        #clothScene = pyphysx.ClothScene(step=0.01, sheet=True, sheetW=60, sheetH=15, sheetSpacing=0.025)
        #default:
        #DartClothEnv.__init__(self, 'reacher.skel', 4, 21, self.control_bounds)
        #w/o force obs:
        #DartClothEnv.__init__(self, cloth_scene=clothScene, model_paths='reacher_capsule.skel', frame_skip=4, observation_size=21, action_bounds=self.control_bounds)
        #w/ force obs:
        DartClothEnv.__init__(self, cloth_scene=clothScene, model_paths='ArmCapsules.skel', frame_skip=4, observation_size=(21+30), action_bounds=self.control_bounds)
        
        #TODO: additional observation size for force
        utils.EzPickle.__init__(self)
        
        self.clothScene.seedRandom(random.randint(1,1000))
        self.clothScene.setFriction(0, 0.6)
        
        self.updateClothCollisionStructures(capsules=True, hapticSensors=True)
        
        self.simulateCloth = True
        self.sampleFromHemisphere = True
        self.rotateCloth = False
        self.randomRoll = False
        
        self.trackSuccess = False
        self.renderSuccess = False
        self.targetHistory = []
        self.successHistory = []
        
        self.random_dir = np.array([0,0,1.])
        
        self.reset_number = 0 #debugging
        print("done init")

    def _step(self, a):
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.multiply(clamped_control, self.action_scale)

        #fingertip = np.array([0.0, -0.25, 0.0])
        fingertip = np.array([0.0, -0.0, 0.0])
        wFingertip1 = self.robot_skeleton.bodynodes[4].to_world(fingertip)
        vec1 = self.target-wFingertip1
        
        #apply action and simulate
        self.do_simulation(tau, self.frame_skip)
        
        wFingertip2 = self.robot_skeleton.bodynodes[4].to_world(fingertip)
        vec2 = self.target-wFingertip2
        
        reward_dist = - np.linalg.norm(vec2)
        reward_ctrl = - np.square(tau).sum() * 0.001
        reward_progress = np.dot((wFingertip2 - wFingertip1), vec1/np.linalg.norm(vec1)) * 100
        alive_bonus = -0.001
        reward_prox = 0
        #if -reward_dist < 0.1:
        #    reward_prox += (0.1+reward_dist)*10
        reward = reward_ctrl + alive_bonus + reward_progress + reward_prox
        #reward = reward_dist + reward_ctrl
        
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
            done = True
            reward -= 500
        elif -reward_dist < 0.1:
            done = True
            reward += 100
        elif (clothDeformation > 5):
            done = True
            reward -= 500

        return ob, reward, done, {}

    def _get_obs(self):
        '''get_obs'''
        theta = self.robot_skeleton.q
        #fingertip = np.array([0.0, -0.25, 0.0])
        fingertip = np.array([0.0, -0.0, 0.0])
        vec = self.robot_skeleton.bodynodes[4].to_world(fingertip) - self.target
        
        if self.simulateCloth is True:
            f = self.clothScene.getHapticSensorObs()#get force from simulation 
        else:
            f = np.zeros(30)
        
        #print("ID getobs:" + str(self.clothScene.id))
        #print("f: " + str(f))
        #print("len f = " + str(len(f)))
        return np.concatenate([np.cos(theta), np.sin(theta), self.target, self.robot_skeleton.dq, vec,f]).ravel()
        #return np.concatenate([theta, self.robot_skeleton.dq, vec]).ravel()

    def reset_model(self):
        '''reset_model'''
        #self.clothScene.translateCloth(0, np.array([0,3.1,0]))
        self.dart_world.reset()
        self.clothScene.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qpos[0] -= 1.
        qpos[1] -= 0.
        qpos[2] += 2.9
        qpos[3] += 0.0
        qpos[4] -= 0.

        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        
        #reset cloth tube orientation and rotate sphere position
        v1 = np.array([0,1.,0])
        v2 = np.array([1.,0,0.])
        self.random_dir = v2
        if self.simulateCloth is True:   
            if self.rotateCloth is True:
                while True:
                    v2 = self.clothScene.sampleDirections()[0]
                    if np.dot(v2/np.linalg.norm(v2), np.array([0,-1,0.])) < 1:
                        break
            M = self.clothScene.rotateTo(v1,v2)
            self.clothScene.rotateCloth(0, M)
            self.clothScene.translateCloth(0, np.array([0.15,0,0.]))
        #self.clothScene.translateCloth(0, np.array([-0.75,0,0]))
        #self.clothScene.translateCloth(0, np.array([0,3.1,0]))
        #self.clothScene.rotateCloth(0, self.clothScene.getRotationMatrix(a=random.uniform(0, 6.28), axis=np.array([0,0,1.])))
        #self.clothScene.rotateCloth(0, M)
        
        
        #move cloth out of arm range
        #self.clothScene.translateCloth(0, np.array([-10.5,0,0]))
        
        #old sampling in box
        #'''
        reacher_range = 0.74
        if not self.sampleFromHemisphere:
            while True:
                self.target = self.np_random.uniform(low=-reacher_range, high=reacher_range, size=3)
                #print('target = ' + str(self.target))
                if np.linalg.norm(self.target) < reacher_range: break
        #'''
        
        #sample target from hemisphere
        if self.sampleFromHemisphere is True:
            self.target = self.hemisphereSample(maxradius=reacher_range, minradius=0.6, norm=v2)

        self.dart_world.skeletons[0].q=[0, 0, 0, self.target[0], self.target[1], self.target[2]]

        #update physx capsules
        self.updateClothCollisionStructures(hapticSensors=True)
        self.clothScene.clearInterpolation()

        #debugging
        self.reset_number += 1
        
        obs = self._get_obs()
        
        #self.render()
        #if np.linalg.norm(obs[-39:]) > 0.00001:
        #    print("COLLISION")
        #    self.reset_model()

        return self._get_obs()

    def updateClothCollisionStructures(self, capsules=False, hapticSensors=False):
        #collision spheres creation
        fingertip = np.array([0.0, -0.0, 0.0])
        cs0 = self.robot_skeleton.bodynodes[1].to_world(fingertip)
        cs1 = self.robot_skeleton.bodynodes[2].to_world(fingertip)
        cs2 = self.robot_skeleton.bodynodes[3].to_world(fingertip)
        cs3 = self.robot_skeleton.bodynodes[4].to_world(fingertip)
        csVars0 = np.array([0.065, -1, -1, 0,0,0])
        csVars1 = np.array([0.05, -1, -1, 0,0,0])
        csVars2 = np.array([0.0365, -1, -1, 0,0,0])
        csVars3 = np.array([0.05, -1, -1, 0,0,0])
        collisionSpheresInfo = np.concatenate([cs0, csVars0, cs1, csVars1, cs2, csVars2, cs3, csVars3]).ravel()
        #collisionSpheresInfo = np.concatenate([cs0, csVars0, cs1, csVars1]).ravel()
        #collisionSpheresInfo = np.concatenate([cs3, csVars3]).ravel()
        
        self.clothScene.setCollisionSpheresInfo(collisionSpheresInfo)
        
        if capsules is True:
            #collision capsules creation
            collisionCapsuleInfo = np.zeros((4,4))
            collisionCapsuleInfo[0,1] = 1
            collisionCapsuleInfo[1,2] = 1
            collisionCapsuleInfo[2,3] = 1
            self.clothScene.setCollisionCapsuleInfo(collisionCapsuleInfo)
            
        if hapticSensors is True:
            #hapticSensorLocations = np.concatenate([cs0, LERP(cs0, cs1, 0.33), LERP(cs0, cs1, 0.66), cs1, LERP(cs1, cs2, 0.33), LERP(cs1, cs2, 0.66), cs2, LERP(cs2, cs3, 0.33), LERP(cs2, cs3, 0.66), cs3])
            #hapticSensorLocations = np.concatenate([cs0, LERP(cs0, cs1, 0.25), LERP(cs0, cs1, 0.5), LERP(cs0, cs1, 0.75), cs1, LERP(cs1, cs2, 0.25), LERP(cs1, cs2, 0.5), LERP(cs1, cs2, 0.75), cs2, LERP(cs2, cs3, 0.25), LERP(cs2, cs3, 0.5), LERP(cs2, cs3, 0.75), cs3])
            hapticSensorLocations = np.concatenate([cs0, LERP(cs0, cs1, 0.25), LERP(cs0, cs1, 0.5), LERP(cs0, cs1, 0.75), cs1, LERP(cs1, cs2, 0.25), LERP(cs1, cs2, 0.5), LERP(cs1, cs2, 0.75), cs2, cs3])
            self.clothScene.setHapticSensorLocations(hapticSensorLocations)
            
    def getViewer(self, sim, title=None, extraRenderFunc=None):
        return DartClothEnv.getViewer(self, sim, title, self.extraRenderFunction)
        
    def hemisphereSample(self, maxradius=1, minradius = 0, norm=np.array([0,0,1.]), frustrum = 0.7):
        p = norm
        while True:
            p = self.np_random.uniform(low=-maxradius, high=maxradius, size=3)
            p_n = np.linalg.norm(p)
            if p_n <= maxradius and p_n >= minradius:
                if(np.dot(p/p_n, norm) > frustrum):
                    return p

        
    def extraRenderFunction(self):
        #print("extra render function")
        
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3d(0,0,0)
        GL.glVertex3d(-1,0,0)
        GL.glEnd()
        
        if self.renderSuccess is True:
            for i in range(len(self.targetHistory)):
                p = self.targetHistory[i]
                s = self.successHistory[i]
                GL.glColor3d(1,0.,0)
                if s is True:
                    GL.glColor3d(0,1.,0)
                GL.glPushMatrix()
                GL.glTranslated(p[0], p[1], p[2])
                GLUT.glutSolidSphere(0.01, 10,10)
                GL.glPopMatrix()
        
        #draw hemisphere samples for target sampling
        '''
        GL.glColor3d(0,1,0)
        for i in range(1000):
            normVec = self.random_dir
            p = self.hemisphereSample(maxradius=0.7, minradius=0.4, norm=normVec)

            #p=np.array([0,0,0.])
            #while True:
            #    p = self.np_random.uniform(low=-1.5, high=1.5, size=3)
            #    if np.linalg.norm(p) < 1.5: break
            GL.glPushMatrix()
            GL.glTranslated(p[0], p[1], p[2])
            GLUT.glutSolidSphere(0.01, 10,10)
            GL.glPopMatrix()
        '''
        
        #print("ID:" + str(self.clothScene.id))
        
        
        HSF = self.clothScene.getHapticSensorObs()
        #print("HSF: " + str(HSF))
        for i in range(self.clothScene.getNumHapticSensors()):
            #print("i = " + str(i))
            #print("HSL[i] = " + str(HSL[i*3:i*3+3]))
            #print("HSF[i] = " + str(HSF[i*3:i*3+3]))
            self.clothScene.drawText(x=15., y=60.+15*i, text="||f[" + str(i) + "]|| = " + str(np.linalg.norm(HSF[3*i:3*i+3])), color=(0.,0,0))
        a=0

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -3.5
        self._get_viewer().scene.tb._set_theta(0)
        self.track_skeleton_id = 0
        
def LERP(p0, p1, t):
    return p0 + (p1-p0)*t
