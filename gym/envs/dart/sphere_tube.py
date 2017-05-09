# This environment is created by Alexander Clegg (alexanderwclegg@gmail.com)

import numpy as np
from gym import utils
from gym.envs.dart.dart_cloth_env import *
import random

class DartClothSphereTubeEnv(DartClothEnv, utils.EzPickle):
    def __init__(self):
        self.action_scale = np.array([0.006, 0.006, 0.006])
        self.control_bounds = np.array([[1.0, 1.0, 1.0],[-1.0, -1.0, -1.0]])
        clothScene = pyphysx.ClothScene(step=0.01, tube=True) 
        DartClothEnv.__init__(self, cloth_scene=clothScene, model_paths='sphere.skel', frame_skip=2, observation_size=6, action_bounds=self.control_bounds, dt=0.01)
        utils.EzPickle.__init__(self)
        
        #testing
        self.clothScene.seedRandom(random.randint(1,1000))
        self.clothScene.setFriction(0, 0.5)
        #print("f=" + str(self.clothScene.getFriction(0)))
        #exit()
        #done testing
        
        self.kinematic = True

    def _step(self, a):
        #if self.clothScene.stepsSinceReset < 50:
        #    a *= 0.
    
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.multiply(clamped_control, self.action_scale)

        initialPos = self.robot_skeleton.positions()
        
        #kinematic update
        if self.kinematic:
            self.dart_world.skeletons[1].q += tau
            #print("q' = " + str(self.dart_world.skeletons[1].q))
            
        #dynamic update (including cloth)
        self.do_simulation(tau, self.frame_skip)
        
        ob = self._get_obs()
        #print("obs = " + str(ob))
        endPos = self.robot_skeleton.positions()
        #update sensor and capsule state
        self.clothScene.setHapticSensorLocations(endPos) #set the location of the sphere as the only sensor location
        collisionSphereInfo = np.concatenate([endPos, np.array([0.025, -1, -1, 0,0,0]) ]).ravel()
        self.clothScene.setCollisionSphereInfo(0, collisionSphereInfo)
        
        #debuggin'
        #print("Friction: " + str(self.clothScene.getFriction(0)))
        
        
        #reward = displacement in target direction -(p1-p0)*(p0/||p0||) and life penalty
        reward_ctrl = -np.dot((endPos-initialPos), initialPos/np.linalg.norm(initialPos))
        alive_bonus = -0.01
        reward = reward_ctrl + alive_bonus

        done = (np.linalg.norm(endPos) < 0.01)
        
        #check cloth deformation for termination
        clothDeformation = self.clothScene.getMaxDeformationRatio(0)
        #print("deformation ratio: " + str(clothDeformation))
        
        if done:
            reward += 5
        elif (clothDeformation > 5):
            done = True
            reward -= 10
            
        #print("f=" + str(self.clothScene.getFriction(0)))

        return ob, reward, done, {}

    def _get_obs(self):
        theta = self.robot_skeleton.q
        f = self.clothScene.getHapticSensorObs()#get force from simulation 
        return np.concatenate([theta, f]).ravel()

    def reset_model(self):
        'random position inside 1x1x1 cube centered at (0,0,0))'
        self.dart_world.reset()
        
        #reset cloth tube orientation and rotate sphere position
        v1 = np.array([0,0,1])
        v2 = self.clothScene.sampleDirections()[0]
        M = self.clothScene.rotateTo(v1,v2)
        self.clothScene.rotateCloth(0, M)
        
        self.dart_world.skeletons[1].q = [0,0,1]
        self.dart_world.skeletons[1].q = M.dot(self.dart_world.skeletons[1].q)
        
        #qpos = self.np_random.uniform(low=-.5, high=.5, size=self.robot_skeleton.ndofs)
        #qvel = self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        #self.set_state(qpos, qvel)
        #while True:
        #    self.target = self.np_random.uniform(low=-1, high=1, size=3)
        #    print('target = ' + str(self.target))
        #    if np.linalg.norm(self.target) < 1.5: break


        #self.dart_world.skeletons[0].q=[0, 0, 0, self.target[0], self.target[1], self.target[2]]

        return self._get_obs()


    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -3.5
        self._get_viewer().scene.tb._set_theta(0)
        self.track_skeleton_id = 0
