__author__ = 'yuwenhao'

import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from gym.envs.dart.parameter_managers import CartPoleManager

class DartCartPoleSwingUpEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0],[-1.0]])
        self.action_scale = 40
        self.train_UP = True
        self.resample_MP = False  # whether to resample the model paraeters
        self.train_mp_sel = False
        self.perturb_MP = False
        self.avg_div = 0
        self.param_manager = CartPoleManager(self)

        obs_dim = 4
        if self.train_UP:
            obs_dim += self.param_manager.param_dim
        if self.train_mp_sel:
            obs_dim += 5

        self.juggling = True
        if self.juggling:
            obs_dim += 4

        self.dyn_models = [None]
        self.dyn_model_id = 0

        if self.avg_div > 1:
            #obs_dim *= self.avg_div
            obs_dim += self.avg_div

        dart_env.DartEnv.__init__(self, 'cartpole_swingup.skel', 2, obs_dim, self.control_bounds, dt=0.001, disableViewer=True)
        self.current_param = self.param_manager.get_simulator_parameters()
        self.dart_world.skeletons[1].bodynodes[0].set_friction_coeff(0.2)
        self.dart_world.skeletons[1].bodynodes[0].set_restitution_coeff(0.9)
        self.dart_world.skeletons[-1].bodynodes[-1].set_restitution_coeff(0.9)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        #if a[0] > self.control_bounds[0][0] or a[0] < self.control_bounds[1][0]:
        #    a[0] = np.sign(a[0])
        #if np.abs(a[0]) > 1:
        #    a[0] = np.sign(a[0])

        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[0] = a[0] * self.action_scale

        state_act = np.concatenate([self.state_vector(), tau])

        if self.dyn_model_id == 0 or self.dyn_models[self.dyn_model_id-1] is None:
            self.do_simulation(tau, self.frame_skip)
        elif self.dyn_models[self.dyn_model_id-1] is not None:
            new_state = self.dyn_models[self.dyn_model_id-1].do_simulation(self.state_vector(), tau, self.frame_skip)
            self.set_state_vector(new_state)
        ob = self._get_obs()

        ang = self.robot_skeleton.q[1]

        ang_proc = (np.abs(ang) % (2 * np.pi))
        ang_proc = np.min([ang_proc, (2 * np.pi) - ang_proc])

        alive_bonus = 6.0
        ang_cost = 1.0*(ang_proc**2)
        quad_ctrl_cost = 0.01 * np.square(a).sum()
        com_cost = 2.0 * np.abs(self.robot_skeleton.q[0])**2

        reward = alive_bonus - ang_cost - quad_ctrl_cost - com_cost
        if ang_proc < 0.5:
            reward += np.max([5 - (ang_proc) * 4, 0]) + np.max([3-np.abs(self.robot_skeleton.dq[1]), 0])

        done = abs(self.robot_skeleton.dq[1]) > 35 or abs(self.robot_skeleton.q[0]) > 2.0

        if self.juggling:
            if self.dart_world.skeletons[1].com()[1] < 0.2:
                reward -= 50
                done = True

        if self.perturb_MP:
            self.param_manager.set_simulator_parameters(self.current_param + np.random.uniform(-0.01, 0.01, len(self.current_param)))

        if self.dyn_model_id != 0:
            reward *= 1.0

        return ob, reward, done, {'model_parameters':self.param_manager.get_simulator_parameters(), 'state_act': state_act, 'next_state':self.state_vector()}


    def _get_obs(self):
        state = np.concatenate([self.robot_skeleton.q, self.robot_skeleton.dq]).ravel()

        ang = self.robot_skeleton.q[1]
        ang_proc = (ang % (2 * np.pi))
        if ang_proc > np.pi:
            ang_proc -= 2 * np.pi

        state[1] = ang_proc

        if self.train_UP:
            state = np.concatenate([state, self.param_manager.get_simulator_parameters()])

        if self.train_mp_sel:
            state = np.concatenate([state, [self.rand]])

        if self.avg_div > 1:
            #return_state = np.zeros(len(state) * self.avg_div)
            #return_state[self.state_index * len(state):(self.state_index + 1) * len(state)] = state
            return_state = np.zeros(len(state) + self.avg_div)
            return_state[0:len(state)] = state
            return_state[len(state) + self.state_index] = 1.0
            return return_state

        if self.juggling:
            state = np.concatenate([state, self.dart_world.skeletons[1].com()[[0, 1]], self.dart_world.skeletons[1].com_velocity()[[0, 1]]])

        return state

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.1, high=.1, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        '''if self.np_random.uniform(low=0, high=1, size=1)[0] > 0.5:
            qpos[1] += np.pi
        else:
            qpos[1] += -np.pi'''
        #qpos[1]+=self.np_random.uniform(low=-np.pi, high=np.pi, size=1)

        if self.resample_MP:
            self.param_manager.resample_parameters()

        self.current_param = self.param_manager.get_simulator_parameters()
        self.state_index = self.dyn_model_id

        if not self.juggling:
            self.dart_world.skeletons[1].set_positions([0,0,0,100, 0, 0])
        else:
            jug_pos = self.dart_world.skeletons[1].q + self.np_random.uniform(low=-.1, high=.1, size=6)
            jug_vel = self.dart_world.skeletons[1].dq + self.np_random.uniform(low=-.01, high=.01, size=6)
            jug_pos[-1]=0
            jug_vel[-1]=0
            self.dart_world.skeletons[1].set_positions(jug_pos)
            self.dart_world.skeletons[1].set_velocities(jug_vel)

        if self.train_mp_sel:
            self.rand = np.random.random()

        self.set_state(qpos, qvel)

        return self._get_obs()


    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -3.5
        self._get_viewer().scene.tb._set_theta(0)
        self.track_skeleton_id = 0
