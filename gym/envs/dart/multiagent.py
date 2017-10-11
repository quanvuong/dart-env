# This environment is created by Karen Liu (karen.liu@gmail.com)

import numpy as np
import random
from gym import utils
from gym.envs.dart import dart_env

class DartMultiAgentEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.reset_number = 0
        self.numAgents = 3
        self.agent_skel_filename = ""
        self.action_scale = np.array([10,10,10,10,10,10])
        self.control_bounds = np.array([np.ones(6),np.ones(6)*-1])
        self.controlSize = 6
        self.obs_dim = 1
        dart_env.DartEnv.__init__(self, model_paths='groundboxes.skel', frame_skip=1, observation_size=self.obs_dim, action_bounds=self.control_bounds)
        utils.EzPickle.__init__(self)
        for i in range(self.numAgents):
            self.dart_world.add_skeleton('/home/alexander/Documents/dev/dart-env/gym/envs/dart/assets/sphere.urdf')
            #print(self.dart_world.skeletons[ix].q)
            #print(self.dart_world.skeletons[ix].name)
        #for ix, dof in enumerate(self.dart_world.skeletons[-1].dofs):
        #    print("dof " +str(ix)+": "+str(dof.name))

        #self.robot_skeleton.set_self_collision_check(True)
        print("Number of Skeletons: " + str(len(self.dart_world.skeletons)))

    def _step(self, a):
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.multiply(clamped_control, self.action_scale)

        #set floating mass position

        if self.reset_number > 0:
            for i in range(self.numAgents):
                ix = i+1
                #print(ix)
                qpos = self.dart_world.skeletons[ix].q
                qvel = self.dart_world.skeletons[ix].dq
                world = self.dart_world.skeletons[ix].bodynodes[0].to_world(np.zeros(3))
                local = self.dart_world.skeletons[ix].bodynodes[0].to_local(world)
                origin_local = self.dart_world.skeletons[ix].bodynodes[0].to_local(np.zeros(3))
                if np.linalg.norm(origin_local) > 0:
                    qvel[-3:] += (origin_local/np.linalg.norm(origin_local))*0.01
                #qpos[-6:] = np.zeros(6)
                #qpos[10] = 0.05
                #qpos[-3:] = local
                #qvel[3] += 0.05
                #qvel[-3] = 0
                #qvel[]
                #qvel[-3:] += local/np.linalg.norm(local)*0.001
                #qpos[9] = 0.05
                #qpos[3] += 0.01
                #self.dart_world.skeletons[ix].set_positions(qpos)
                self.dart_world.skeletons[ix].set_velocities(qvel)
                #print(self.dart_world.skeletons[ix].q)

        #zero control
        tau = np.zeros(self.controlSize)
        
        #apply action and simulate
        self.do_simulation(tau, self.frame_skip)

        reward = 0
        
        ob = self._get_obs()

        s = self.state_vector()

        done = np.isfinite(s).all()

        return ob, reward, done, {}

    def _get_obs(self):
        theta = self.robot_skeleton.q
        return np.zeros(self.obs_dim)
        #return np.concatenate([theta, self.robot_skeleton.dq, vec]).ravel()

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        for i in range(self.numAgents):
            ix = i + 1
            initq = self.dart_world.skeletons[ix].q
            initq[4] += 0.1
            initq[3] += random.uniform(-0.5, 0.5)
            initq[5] += random.uniform(-0.5, 0.5)
            self.dart_world.skeletons[ix].set_positions(initq)

        self.reset_number += 1

        return self._get_obs()


    def viewer_setup(self):
        #self._get_viewer().scene.tb.trans[2] = -3.5
        #self._get_viewer().scene.tb._set_theta(0)
        #self.track_skeleton_id = 0
        a=0
