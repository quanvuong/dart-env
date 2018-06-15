import numpy as np
from gym import utils
from gym.envs.dart import dart_env

class DartCartPole2PoleEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0, 1.0],[-1.0, -1.0]])
        self.action_scale = 10
        obs_dim = 4
        self.include_action_in_obs = False
        if self.include_action_in_obs:
            obs_dim += len(self.control_bounds[0])
            self.prev_a = np.zeros(len(self.control_bounds[0]))

        dart_env.DartEnv.__init__(self, 'cartpole_multilink/cartpole_2pole.skel', 2, obs_dim, self.control_bounds,
                                  dt=0.005, disableViewer=True)

        # setups for articunet
        self.state_dim = 32
        self.enc_net = []
        self.act_net = []
        self.vf_net = []
        self.merg_net = []
        self.net_modules = []
        self.net_vf_modules = []
        self.generic_modules = []

        # build dynamics model
        self.generic_modules.append([0, [self.state_dim] * 3, self.state_dim, 128, 2, 'world_node1'])
        self.generic_modules.append([3, [self.state_dim] * 3, self.state_dim, 128, 2, 'revolute_node1'])
        self.generic_modules.append([6, [self.state_dim] * 3, self.state_dim, 128, 2, 'pole_node1'])
        self.generic_modules.append([0, [self.state_dim] * 2, 6, 128, 2, 'pole_predictor'])
        self.generic_modules.append([0, [self.state_dim] * 2, 2, 128, 2, 'revolute_predictor'])

        self.generic_modules.append([0, [self.state_dim] * 3, self.state_dim, 128, 2, 'world_node2'])
        self.generic_modules.append([3, [self.state_dim] * 3, self.state_dim, 128, 2, 'revolute_node2'])
        self.generic_modules.append([6, [self.state_dim] * 3, self.state_dim, 128, 2, 'pole_node2'])

        # first pass
        self.net_modules.append([[4,5,6,7,8,9], 2, [None, None, None]]) # [world, jnt, bdnd]
        self.net_modules.append([[10,11,12,13,14,15], 2, [None, None, None]])
        self.net_modules.append([[0, 2, 16], 1, [None, None, [0]]])
        self.net_modules.append([[1, 3, 17], 1, [None, None, [0, 1]]])
        self.net_modules.append([[], 0, [None, [2, 3], [0, 1]]])

        # second pass
        self.net_modules.append([[4, 5, 6, 7, 8, 9], 2+5, [[4], [2,3], [1]]])  # [world, jnt, bdnd]
        self.net_modules.append([[10, 11, 12, 13, 14, 15], 2+5, [[4], [3], [0]]])
        self.net_modules.append([[0, 2, 16], 1+5, [[4], [2, 3], [5]]])
        self.net_modules.append([[1, 3, 17], 1+5, [[4], [2, 3], [5, 6]]])
        self.net_modules.append([[], 0+5, [[4], [7, 8], [5, 6]]])

        # pass to predictor
        self.net_modules.append([[], 4, [[9], [7]]])
        self.net_modules.append([[], 4, [[9], [8]]])
        self.net_modules.append([[], 3, [[9], [5]]])
        self.net_modules.append([[], 3, [[9], [6]]])

        self.net_modules.append([[], None, [[10], [11], [12], [13]], None, False])
        self.reorder_output = np.array([0, 2, 1, 3, 4,5,6,7,8,9, 10,11,12,13,14,15], dtype=np.int32)

        utils.EzPickle.__init__(self)

    def terminated(self):
        ob = self._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(self.state_vector()) < 50).all()
        done = not notdone
        return done

    def _step(self, a):
        a = np.clip(a, -1, 1)
        if self.include_action_in_obs:
            self.prev_a = np.copy(a)

        tau = a * self.action_scale

        self.do_simulation(tau, self.frame_skip)
        #reward = -np.abs(self.robot_skeleton.bodynodes[-1].C[1] - 0.9) + 0.9
        reward = -np.linalg.norm(self.robot_skeleton.bodynodes[-1].C - np.array([0.6, 0.6, 0.0])) + 1.2

        reward -= 0.04 * np.square(a).sum()

        ob = self._get_obs()

        done = self.terminated()

        return ob, reward, done, {}


    def _get_obs(self):
        state = np.concatenate([self.robot_skeleton.q % (2 * np.pi), self.robot_skeleton.dq]).ravel()
        return state

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        if np.random.random() < 0.5:
            qpos[0] += np.pi
        else:
            qpos[0] -= np.pi
        self.set_state(qpos, qvel)

        if self.include_action_in_obs:
            self.prev_a = np.zeros(len(self.control_bounds[0]))

        return self._get_obs()

    def state_vector(self):
        state = np.concatenate([self.robot_skeleton.q, self.robot_skeleton.dq]).ravel()
        for i, bn in enumerate(self.robot_skeleton.bodynodes):
            com = bn.C
            comvel = bn.dC
            state = np.concatenate(
                [state, [com[0], com[1], comvel[0], comvel[1], self.robot_skeleton.q[i], self.robot_skeleton.dq[i]]])
        return state

    def set_state_vector(self, state):
        self.robot_skeleton.q = state[0:len(self.robot_skeleton.q)]
        self.robot_skeleton.dq = state[len(self.robot_skeleton.q):2*len(self.robot_skeleton.q)]
        self.dart_world.skeletons[1].q = [0,0,0, state[len(self.robot_skeleton.q)*2], state[len(self.robot_skeleton.q)*2+1], 0]
        self.dart_world.skeletons[2].q = [0,0,0, state[len(self.robot_skeleton.q)*2+6], state[len(self.robot_skeleton.q)*2+7], 0]
        #self.dart_world.skeletons[1].q = [0, 0, 0, self.robot_skeleton.bodynodes[0].C[0],
        #                                  self.robot_skeleton.bodynodes[0].C[1], 0]
        #self.dart_world.skeletons[2].q = [0, 0, 0, self.robot_skeleton.bodynodes[1].C[0],
        #                                  self.robot_skeleton.bodynodes[1].C[1], 0]

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -3.5
        self._get_viewer().scene.tb._set_theta(0)
        self.track_skeleton_id = 0
