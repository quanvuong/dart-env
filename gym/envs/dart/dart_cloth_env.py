# Contributors: Alexander Clegg (alexanderwclegg@gmail.com)

import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
import six
from gym import utils

from gym.envs.dart.dart_env import *

from gym.envs.dart.static_cloth_window import *

try:
    import pyPhysX as pyphysx
    pyphysx.init()
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install pyphysx.)".format(e))    

class DartClothEnv(DartEnv, utils.EzPickle):
    """Superclass for all Dart Cloth environments.
    """

    def __init__(self, cloth_scene, model_paths, frame_skip, observation_size, action_bounds, \
                 dt=0.002, obs_type="parameter", action_type="continuous", visualize=True, disableViewer=False,\
                 screen_width=80, screen_height=45):
        
        #pyPhysX initialization (do this in subclasses)
        #print("Initialize clothScene...")
        #self.clothScene = pyphysx.ClothScene(mesh_path="/home/alexander/Documents/dev/robot-assisted-dressing/hapticNavigation/mesh/tshirt_m.obj")
        self.clothScene = cloth_scene
        self.clothScene.printTag()
        #done pyPhysx init
        
        self.simulateCloth = True #toggle this off to remove cloth computation
        
        self.kinematic = False #if false, compute dynamics, otherwise rely on manual position updates
        
        #initialize dart_env
        DartEnv.__init__(self, model_paths, frame_skip, observation_size, action_bounds, dt, obs_type, action_type, visualize, disableViewer, screen_width, screen_height)
        
        utils.EzPickle.__init__(self)

    def _reset(self):
        'Overwrite of DartEnv._reset to add cloth reset'
        #pyPhysX reset
        self.clothScene.reset()
        #done pyPhysX reset
        
        return DartEnv._reset(self)

    def do_simulation(self, tau, n_frames):
        'Overwrite of DartEnv.do_simulation to add cloth simulation step'
        
        if self.add_perturbation:
            if self.perturbation_duration == 0:
                self.perturb_force *= 0
                if np.random.random() < self.perturbation_parameters[0]:
                    axis_rand = np.random.randint(0, 2, 1)[0]
                    direction_rand = np.random.randint(0, 2, 1)[0] * 2 - 1
                    self.perturb_force[axis_rand] = direction_rand * self.perturbation_parameters[1]

            else:
                self.perturbation_duration -= 1

        for _ in range(n_frames):
            if self.add_perturbation:
                self.robot_skeleton.bodynodes[self.perturbation_parameters[2]].add_ext_force(self.perturb_force)

            if not self.kinematic:
                self.robot_skeleton.set_forces(tau)
                self.dart_world.step()
            #pyPhysX step
            if self.simulateCloth:
                self.clothScene.step()
            #done pyPhysX step
        #if(self.clothScene.getMaxDeformationRatio(0) > 5):
        #    self._reset()

    def getViewer(self, sim, title=None, extraRenderFunc=None):
        'Overwrite of DartEnv.getViewer to instantiate StaticClothGLUTWindow instead'
        # glutInit(sys.argv)
        win = StaticClothGLUTWindow(sim, title, self.clothScene, extraRenderFunc)
        win.scene.add_camera(Trackball(theta=-45.0, phi = 0.0, zoom=0.1), 'gym_camera')
        win.scene.set_camera(win.scene.num_cameras()-1)

        # to add speed,
        if self._obs_type == 'image':
            win.run(self.screen_width, self.screen_height, _show_window=self.visualize)
        else:
            win.run(_show_window=self.visualize)
        return win
        
    def updateCollisionSpherePos(self, sid, csid):
        'updates the position given a dartworld to reference'
        #TODO: get collision sphere info, update and push
        '''if(nid >=0 and sid >= 0): #otherwise there is nothing to reference
            skel = self.dart_world.skeletons[sid]
            node = skel.bodynode(nid)
            pos = node.to_world(offset)'''
            
    def updateCollisionSphereOffset(self, sid, csid):
        'updates the local offset from pos given a dartworld to reference'
        #TODO: get collision sphere info, update and push
        '''if(nid >=0 and sid >= 0): #otherwise there is nothing to reference
            skel = self.dart_world.skeletons[sid]
            node = skel.bodynode(nid)
            offset = node.to_local(pos)'''
