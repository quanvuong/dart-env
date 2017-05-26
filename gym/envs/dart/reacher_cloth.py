# This environment is created by Alexander Clegg (alexanderwclegg@gmail.com)

import numpy as np
from gym import utils
from gym.envs.dart.dart_cloth_env import *
import random
import time
import math

from pyPhysX.colors import *
import pyPhysX.pyutils as pyutils

import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT

def gprint(text):
    'genie print color'
    pyutils.cprint(text, CYAN)

def oprint(text):
    'output genie print color'
    pyutils.cprint(text, MAGENTA)

class DartClothReacherEnv(DartClothEnv, utils.EzPickle):
    def __init__(self):
        self.target = np.array([0.8, -0.6, 0.6])
        #5 dof reacher
        #self.action_scale = np.array([10, 10, 10, 10, 10])
        #self.control_bounds = np.array([[1.0, 1.0, 1.0, 1.0, 1.0],[-1.0, -1.0, -1.0, -1.0, -1.0]])
        
        #5 dof reacher
        #self.action_scale = np.array([ 10, 10, 10, 10 ,10])
        #self.control_bounds = np.array([[ 1.0, 1.0, 1.0, 1.0, 1.0],[ -1.0, -1.0, -1.0, -1.0, -1.0]])
        
        #9 dof reacher
        self.action_scale = np.array([ 10, 10, 10, 10 ,10, 10, 10, 10, 10])
        self.control_bounds = np.array([[ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],[ -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]])
        
        
        #create cloth scene
        clothScene = pyphysx.ClothScene(step=0.01, sheet=True, sheetW=60, sheetH=15, sheetSpacing=0.025)
        
        #default:
        #DartClothEnv.__init__(self, 'reacher.skel', 4, 21, self.control_bounds)
        #w/o force obs:
        #DartClothEnv.__init__(self, cloth_scene=clothScene, model_paths='reacher_capsule.skel', frame_skip=4, observation_size=21, action_bounds=self.control_bounds)
        #w/ force obs:
        #DartClothEnv.__init__(self, cloth_scene=clothScene, model_paths='reacher_capsule.skel', frame_skip=4, observation_size=(21+39), action_bounds=self.control_bounds)
        
        DartClothEnv.__init__(self, cloth_scene=clothScene, model_paths='ArmCapsules2.skel', frame_skip=4, observation_size=(33+30), action_bounds=self.control_bounds, visualize=False)
        
        #TODO: additional observation size for force
        utils.EzPickle.__init__(self)
        
        self.clothScene.seedRandom(random.randint(1,1000))
        self.clothScene.setFriction(0, 1)
        
        self.updateClothCollisionStructures(capsules=True, hapticSensors=True)
        
        self.simulateCloth = False
        self.sampleFromHemisphere = False
        self.rotateCloth = False
        self.randomRoll = False
        
        self.trackSuccess = False
        self.renderSuccess =False
        self.renderFailure = False
        self.targetHistory = []
        self.successHistory = []
        self.successSampleRenderSize = 0.01
        
        self.renderDofs = True #if true, show dofs text 
        self.renderForceText = False
        
        #self.voxels = np.zeros(1000000) #100^3
        
        self.random_dir = np.array([0,0,1.])
        
        self.reset_number = 0 #debugging
        
        self.tag = "its text"
        
        #self.skelVoxelAnalysis(dim=100, radius=0.8, samplerate=0.2, depth=0, efn=5, efo=np.array([0.,-0.06,0]), displayReachable = True, displayUnreachable=True)

    def _step(self, a):
        #print("step")
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.multiply(clamped_control, self.action_scale)

        fingertip = np.array([0.0, -0.06, 0.0])
        wFingertip1 = self.robot_skeleton.bodynodes[5].to_world(fingertip)
        vec1 = self.target-wFingertip1
        
        #apply action and simulate
        self.do_simulation(tau, self.frame_skip)
        
        wFingertip2 = self.robot_skeleton.bodynodes[5].to_world(fingertip)
        vec2 = self.target-wFingertip2
        
        reward_dist = - np.linalg.norm(vec2)
        reward_ctrl = - np.square(tau).sum() * 0.001
        reward_progress = np.dot((wFingertip2 - wFingertip1), vec1/np.linalg.norm(vec1)) * 100
        alive_bonus = -0.001
        reward_prox = 0
        if -reward_dist < 0.1:
            reward_prox += (0.1+reward_dist)*40
        reward = reward_ctrl + alive_bonus + reward_progress + reward_prox
        #reward = reward_dist + reward_ctrl
        
        ob = self._get_obs()
        #print("obs: " + str(ob))

        s = self.state_vector()
        
        #update physx capsules
        self.updateClothCollisionStructures(hapticSensors=True)
        
        #check cloth deformation for termination
        clothDeformation = 0
        if self.simulateCloth is True:
            clothDeformation = self.clothScene.getMaxDeformationRatio(0)
        
        #check termination conditions
        #print("state: " + str(s))
        done = False
        if not np.isfinite(s).all():
            print("Infinite value detected..." + str(s))
            done = True
            reward -= 500
        elif -reward_dist < 0.1:
            done = True
            reward += 50
            if self.trackSuccess is True:
                self.successHistory[-1] = True
        
        elif (clothDeformation > 5):
            done = True
            reward -= 500
        
        return ob, reward, done, {}

    def _get_obs(self):
        theta = self.robot_skeleton.q
        fingertip = np.array([0.0, -0.06, 0.0])
        vec = self.robot_skeleton.bodynodes[5].to_world(fingertip) - self.target
        
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
        self.dart_world.reset()
        self.clothScene.reset()
        self.clothScene.translateCloth(0, np.array([-3.5,0,0]))
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        
        #reset cloth tube orientation and rotate sphere position
        v1 = np.array([0,0,-1])
        v2 = np.array([-1.,0,0])
        self.random_dir = v2
        if self.simulateCloth is True:   
            if self.rotateCloth is True:
                while True:
                    v2 = self.clothScene.sampleDirections()[0]
                    if np.dot(v2/np.linalg.norm(v2), np.array([0,-1,0.])) < 1:
                        break
            M = self.clothScene.rotateTo(v1,v2)
            self.clothScene.translateCloth(0, np.array([0,0,-0.5]))
            self.clothScene.translateCloth(0, np.array([-0.75,0,0]))
            self.clothScene.translateCloth(0, np.array([0,-0.1,0]))
            self.clothScene.rotateCloth(0, self.clothScene.getRotationMatrix(a=random.uniform(0, 6.28), axis=np.array([0,0,1.])))
            self.clothScene.rotateCloth(0, M)
        
        
            #move cloth out of arm range
            self.clothScene.translateCloth(0, np.array([-10.5,0,0]))
        
        #old sampling in box
        #'''
        reacher_range = 0.75
        if not self.sampleFromHemisphere:
            while True:
                self.target = self.np_random.uniform(low=-reacher_range, high=reacher_range, size=3)
                #print('target = ' + str(self.target))
                if np.linalg.norm(self.target) < reacher_range: break
        #'''
        
        #sample target from hemisphere
        if self.sampleFromHemisphere is True:
            self.target = self.hemisphereSample(radius=reacher_range, norm=v2)
        
        dim = 15
        if(self.reset_number < dim*dim*dim):
            self.target = self.voxelCenter(dim=dim, radius=0.8, ix=self.reset_number)
            #print(self.target)
        else:
            self.trackSuccess = False

        self.dart_world.skeletons[0].q=[0, 0, 0, self.target[0], self.target[1], self.target[2]]

        #update physx capsules
        self.updateClothCollisionStructures(hapticSensors=True)
        self.clothScene.clearInterpolation()

        #debugging
        self.reset_number += 1
        
        obs = self._get_obs()
        
        #self.render()
        if self.simulateCloth is True:
            if np.linalg.norm(obs[-30:]) > 0.00001:
                #print("COLLISION")
                self.reset_model()
        
        if self.trackSuccess is True:
            self.targetHistory.append(self.target)
            self.successHistory.append(False)

        return self._get_obs()

    def updateClothCollisionStructures(self, capsules=False, hapticSensors=False):
        #collision spheres creation
        fingertip = np.array([0.0, -0.06, 0.0])
        z = np.array([0.,0,0])
        cs0 = self.robot_skeleton.bodynodes[1].to_world(z)
        cs1 = self.robot_skeleton.bodynodes[3].to_world(z)
        cs2 = self.robot_skeleton.bodynodes[4].to_world(z)
        cs3 = self.robot_skeleton.bodynodes[5].to_world(z)
        cs4 = self.robot_skeleton.bodynodes[5].to_world(fingertip)
        csVars0 = np.array([0.065, -1, -1, 0,0,0])
        csVars1 = np.array([0.05, -1, -1, 0,0,0])
        csVars2 = np.array([0.0365, -1, -1, 0,0,0])
        csVars3 = np.array([0.04, -1, -1, 0,0,0])
        csVars4 = np.array([0.036, -1, -1, 0,0,0])
        collisionSpheresInfo = np.concatenate([cs0, csVars0, cs1, csVars1, cs2, csVars2, cs3, csVars3, cs4, csVars4]).ravel()
        #collisionSpheresInfo = np.concatenate([cs0, csVars0, cs1, csVars1]).ravel()
        
        self.clothScene.setCollisionSpheresInfo(collisionSpheresInfo)
        
        if capsules is True:
            #collision capsules creation
            collisionCapsuleInfo = np.zeros((5,5))
            collisionCapsuleInfo[0,1] = 1
            collisionCapsuleInfo[1,2] = 1
            collisionCapsuleInfo[2,3] = 1
            collisionCapsuleInfo[3,4] = 1
            self.clothScene.setCollisionCapsuleInfo(collisionCapsuleInfo)
            
        if hapticSensors is True:
            #hapticSensorLocations = np.concatenate([cs0, LERP(cs0, cs1, 0.33), LERP(cs0, cs1, 0.66), cs1, LERP(cs1, cs2, 0.33), LERP(cs1, cs2, 0.66), cs2, LERP(cs2, cs3, 0.33), LERP(cs2, cs3, 0.66), cs3])
            #hapticSensorLocations = np.concatenate([cs0, LERP(cs0, cs1, 0.25), LERP(cs0, cs1, 0.5), LERP(cs0, cs1, 0.75), cs1, LERP(cs1, cs2, 0.25), LERP(cs1, cs2, 0.5), LERP(cs1, cs2, 0.75), cs2, LERP(cs2, cs3, 0.25), LERP(cs2, cs3, 0.5), LERP(cs2, cs3, 0.75), cs3])
            hapticSensorLocations = np.concatenate([cs0, LERP(cs0, cs1, 0.25), LERP(cs0, cs1, 0.5), LERP(cs0, cs1, 0.75), cs1, LERP(cs1, cs2, 0.25), LERP(cs1, cs2, 0.5), LERP(cs1, cs2, 0.75), cs2, cs3, cs4])
            self.clothScene.setHapticSensorLocations(hapticSensorLocations)
            
    def getViewer(self, sim, title=None, extraRenderFunc=None, inputFunc=None):
        return DartClothEnv.getViewer(self, sim, title, self.extraRenderFunction, self.inputFunc)
        
    def hemisphereSample(self, radius=1, norm=np.array([0,0,1.]), frustrum = 0.6):
        p = norm
        while True:
            p = self.np_random.uniform(low=-radius, high=radius, size=3)
            p_n = np.linalg.norm(p)
            if p_n < radius:
                if(np.dot(p/p_n, norm) > frustrum):
                    return p

        
    def extraRenderFunction(self):
        #print("extra render function")
        
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3d(0,0,0)
        GL.glVertex3d(-1,0,0)
        GL.glEnd()
        
        if self.renderSuccess is True or self.renderFailure is True:
            for i in range(len(self.targetHistory)):
                p = self.targetHistory[i]
                s = self.successHistory[i]
                if (s and self.renderSuccess) or (not s and self.renderFailure):
                    GL.glColor3d(1,0.,0)
                    if s is True:
                        GL.glColor3d(0,1.,0)
                    GL.glPushMatrix()
                    GL.glTranslated(p[0], p[1], p[2])
                    GLUT.glutSolidSphere(self.successSampleRenderSize, 10,10)
                    GL.glPopMatrix()
        
        #draw hemisphere samples for target sampling
        '''
        GL.glColor3d(0,1,0)
        for i in range(1000):
            p = self.hemisphereSample(radius=1.4, norm=self.random_dir)
            #p=np.array([0,0,0.])
            #while True:
            #    p = self.np_random.uniform(low=-1.5, high=1.5, size=3)
            #    if np.linalg.norm(p) < 1.5: break
            GL.glPushMatrix()
            GL.glTranslated(p[0], p[1], p[2])
            GLUT.glutSolidSphere(0.01, 10,10)
            GL.glPopMatrix()
        '''
        
        #render target vector
        '''
        ef = self.robot_skeleton.bodynodes[5].to_world(np.array([0,-0.06,0]))
        #vec = ef - self.target
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3d(ef[0], ef[1], ef[2])
        GL.glVertex3d(self.target[0], self.target[1], self.target[2])
        GL.glEnd()
        '''
        #print("ID:" + str(self.clothScene.id))
        
        textX = 15.
        if self.renderForceText:
            HSF = self.clothScene.getHapticSensorObs()
            #print("HSF: " + str(HSF))
            for i in range(self.clothScene.getNumHapticSensors()):
                #print("i = " + str(i))
                #print("HSL[i] = " + str(HSL[i*3:i*3+3]))
                #print("HSF[i] = " + str(HSF[i*3:i*3+3]))
                self.clothScene.drawText(x=textX, y=60.+15*i, text="||f[" + str(i) + "]|| = " + str(np.linalg.norm(HSF[3*i:3*i+3])), color=(0.,0,0))
            textX += 160
            
        if self.renderDofs:
            for i in range(len(self.robot_skeleton.q)):
                self.clothScene.drawText(x=textX, y=60.+15*i, text="||q[" + str(i) + "]|| = " + str(self.robot_skeleton.q[i]), color=(0.,0,0))
            textX += 30
        a=0

    def inputFunc(self, repeat=False):
        will = ""
        if not repeat:
            gprint("Your will?")
            will = input('').split()
        else:
            gprint("Anything else?")
            will = input('').split()
        if len(will) == 0:
            gprint(" As you wish.")
            return
        elif will[0] == "done":
            gprint(" As you wish.")
            return
        elif will[0] == "exit":
            gprint(" I await your command.")
            exit()
        elif will[0] == "help":
            gprint("How may I be of service?")
            oprint("    Commands:")
            oprint("        toggle [boolean]: toggles a boolean") 
            oprint("        set [variable] [value]: sets a variable to a value")
            oprint("        exit: exit the program")
            oprint("        done: dismiss genie")
            oprint("        help: *you are here*")
        elif will[0] == "toggle":
            if len(will) == 1:
                oprint(" toggle [boolean]: toggles a boolean")
                oprint("     choices: renderSuccess(rs), renderFailure(rf), trackSuccess(ts)")
            else:
                if hasattr(self, will[1]):
                    if(type(getattr(self, will[1])) == type(False)):
                        setattr(self, will[1], not getattr(self, will[1]))
                        oprint(str(will[1]) + " -> " + str(getattr(self, will[1])))
                    else:
                        gprint(str(will[1]) + " is not a boolean, but rather a " + str(type(type(self.getattr(self, will[1])))))
                else:
                    gprint("I see no variable: " + str(will[1]))
        elif will[0] == "set":
            if len(will) < 2:
                oprint(" set [variable] [value]: sets a variable to a value")
                oprint("     choices: successSampleRenderSize(ssrs)")
            elif len(will) < 3:
                if hasattr(self, will[1]):
                    gprint(str(will[1]) + " is of type " + str(type(getattr(self, will[1]))) + " and has value " + str(getattr(self, will[1])))
                    
                else:
                    gprint("I see no variable: " + str(will[1]))
            else:
                if hasattr(self, will[1]):
                    foundType = False
                    if type(getattr(self, will[1])) == type(0.1):
                        try:
                            setattr(self, will[1], float(will[2]))
                            foundType = True
                        except ValueError:
                            gprint("It seems I can't do that, I need a float.")
                    elif type(getattr(self, will[1])) == type(2):
                        try:
                            setattr(self, will[1], int(will[2]))
                            foundType = True
                        except ValueError:
                            gprint("It seems I can't do that, I need an int.")
                    elif type(getattr(self, will[1])) == type(False):
                        try:
                            setattr(self, will[1], will[2]=='True')
                            foundType = True
                        except ValueError:
                            gprint("It seems I can't do that, I need a boolean.")
                    elif type(getattr(self, will[1])) == type("string"):
                        setattr(self, will[1], will[2]) 
                        foundType = True
                    else:
                        gprint("I don't know how to set that type.")
                        
                    if foundType is True:
                        oprint(will[1] + " = " + str(getattr(self, will[1])))
                else:
                    gprint(" I have no variable: " + str(will[1]))
        else: #unkown command
            gprint("Alas, I know not how to" + will[1] + ".")
            
        #continue in command mode until released
        self.inputFunc(True)
                
            #print("toggling " + will[1])
        #print("input func: " + will)

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -3.5
        self._get_viewer().scene.tb._set_theta(0)
        self.track_skeleton_id = 0
        
    '''def skelVoxelAnalysis(self, dim, radius, samplerate=0.1, depth=0, efn=5, efo=np.array([0.,0,0]), displayReachable = True, displayUnreachable=True):
        #initialize binary voxel structure (0)
        if depth == 0:
            self.voxels = np.zeros(dim*dim*dim)
            
        #step through all joint samples
        qpos = self.robot_skeleton.q
        low  = self.robot_skeleton.dof(depth).position_lower_limit()
        high = self.robot_skeleton.dof(depth).position_upper_limit()
        qpos[depth] = low
        self.robot_skeleton.set_positions(qpos)
        samples = int((high-low)/samplerate)
        if(samples == 0):
            samples = 1
        #print(samples)
        for i in range(samples):
            if depth < 5:
                print("[" + str(depth) + "]" + str(i) + "/" + str(samples))
            qpos = self.robot_skeleton.q
            qpos[depth] = low + i*samplerate
            if depth < len(qpos)-1:
                self.skelVoxelAnalysis(dim, radius, samplerate, depth+1)
            else:
                r = np.array([radius, radius, radius])
                ef = self.robot_skeleton.bodynodes[efn].to_world(efo)
                efp = (ef + r)/(2.*r) #transform into voxel grid space (centered at [-r,-r,-r])
                ix = math.floor(efp[0]*dim) + math.floor(efp[1]*dim*dim) + math.floor(efp[2]*dim*dim*dim)
                self.voxels[ix] += 1
        
        
        #TODO: add voxels to visualization
        if depth == 0:
            for v in range(len(self.voxels)):
                if self.voxels[v] > 0 and displayReachable:
                    self.targetHistory.append(self.voxelCenter(dim, radius, v))
                    self.successHistory.append(True)
                elif displayUnreachable:
                    self.targetHistory.append(self.voxelCenter(dim, radius, v))
                    self.successHistory.append(False)
                
    '''    
    def voxelCenter(self, dim, radius, ix):
        r = np.array([radius, radius, radius])
        z = int(ix/(dim*dim))
        y = int((ix-(z*dim*dim))/dim)
        x = int(ix - (z*dim*dim) - (y*dim))
        s = (2.0*radius)/dim
        p = -r + np.array([x,y,z])*s + 0.5*s
        '''print("ix: " + str(ix))
        print("r: " + str(r))
        print("z: " + str(z))
        print("y: " + str(y))
        print("x: " + str(x))
        print("s: " + str(s))
        print("p: " + str(p))'''
        return p
        
        
def LERP(p0, p1, t):
    return p0 + (p1-p0)*t
