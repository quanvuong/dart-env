import numpy as np
from gym import utils
from gym.envs.dart import dart_env
import pickle
import os

from gym.envs.dart.parameter_managers import *


class DartFlockingEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.rendering = True
        self.policyDirectory = "flockingdefault"

        #control is a vector force
        self.control_bounds = np.array([[1.0, 1.0, 1.0],[-1.0, -1.0, -1.0]])
        self.action_scale = 200
        obs_dim = 42 #7 nearest neighbors pos and vel
        self.numSteps = 0

        #simulation variables
        self.numBoids = 8
        self.boidPolicy = None
        self.controlBoid = None

        dart_env.DartEnv.__init__(self, 'boidworld.skel', 4, obs_dim, self.control_bounds, disableViewer=not self.rendering)

        self.controlBoid = self.dart_world.skeletons[1]

        #load additional boids
        for i in range(1, self.numBoids):
            self.dart_world.add_skeleton(filename='dart-env/gym/envs/dart/assets/singleboid.skel')

        #self.dart_world.set_collision_detector(3) # 3 is ode collision detector

        utils.EzPickle.__init__(self)
        
        #self.reset_model()
        #color = np.array([0.,0,0])
        #print("color: " + str(color))
        #print(self.robot_skeleton.q)
        #exit()

    def _step(self, a):
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.zeros(self.robot_skeleton.ndofs)
        #TODO: action logic for all boids

        #tau[3:] = clamped_control * self.action_scale
        if self.controlBoid is not None:
            self.controlBoid.set_forces(np.array([0,0,0, ]))
        self.do_simulation(tau, self.frame_skip)

        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()

        alive_bonus = 1.0
        reward = alive_bonus
        #TODO: flocking rewards
        #reward -= 1e-3 * np.square(a).sum()
        #reward -= 5e-1 * joint_limit_penalty
        #reward -= 1e-7 * total_force_mag

        s = self.state_vector()
        done = not (np.isfinite(s).all())
        ob = self._get_obs()

        return ob, reward, done, {}

    def do_simulation(self, tau, n_frames):
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

            #self.robot_skeleton.set_forces(tau)
            self.dart_world.step()

    def _get_obs(self):
        #TODO: nearest neighbor search
        #TODO: get all relative positions
        obs =  np.zeros(42)
        #obs =  np.concatenate([        ])

        return obs

    def reset_model(self):
        self.numSteps = 0
        self.dart_world.reset()

        #reload policy for non-learning boids
        self.loadPolicy(directory=self.policyDirectory)

        #scramble the boids
        for i in range(len(self.dart_world.skeletons)):
            if i > 0:
                self.dart_world.skeletons[i].set_positions(np.random.uniform(
                    low=np.array([0, 0, 0, -1, -0.9, -1]),
                    high=np.array([0, 0, 0, 1, -0.9, 1]),
                    size=6
                ))
                self.dart_world.skeletons[i].set_velocities(np.random.uniform(
                    low=np.array([0, 0, 0, -1, 0, -1]),
                    high=np.array([0, 0, 0, 1, 0, 1]),
                    size=6
                ))
                for d in self.dart_world.skeletons[i].dofs:
                    d.set_damping_coefficient(0.0001)


        state = self._get_obs()

        return state

    def viewer_setup(self):
        a=0
        #self._get_viewer().scene.tb.trans[2] = -5.5

    def loadPolicy(self, directory):
        prefix = os.path.dirname(os.path.abspath(__file__))
        prefix = os.path.join(prefix, '../../../../rllab/data/local/experiment/')

        self.boidPolicy = pickle.load(open(prefix + directory + "/policy.pkl", "rb"))
