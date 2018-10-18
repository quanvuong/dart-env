__author__ = 'yuwenhao'

import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from gym.envs.dart.inverted_double_pendulum import DartDoubleInvertedPendulumEnv
from gym.envs.dart.cartpole_swingup import DartCartPoleSwingUpEnv
from gym.envs.dart.cart_pole import DartCartPoleEnv
from gym.envs.dart import pylqr

class DDPReward:
    def __init__(self, total_dim, state_sel, act_dim):
        self.total_dim = total_dim
        self.state_sel = state_sel
        self.target_state = np.array([0.0] * len(state_sel))
        self.coefs = [5] * len(state_sel) + [1] * act_dim

    def __call__(self, x, u, t, aux):
        pos_pen = 0.5 * np.dot(self.coefs[0:len(self.state_sel)], np.square(x[self.state_sel] - self.target_state))
        control_pen = np.dot(self.coefs[len(self.state_sel):], (np.square(u)))

        return pos_pen + control_pen

    def cost_dx(self, x, u, t, aux):
        dx_sel = self.coefs[0:len(self.state_sel)] * (x[self.state_sel] - self.target_state)
        dx = np.zeros(len(x))
        dx[self.state_sel] = dx_sel
        return dx

    def cost_du(self, x, u, t, aux):
        du = 2 * np.array(self.coefs[len(self.state_sel):] * u)
        return du

    def cost_dxx(self, x, u, t, aux):
        hess = np.zeros((self.total_dim, self.total_dim))
        for i, sel in enumerate(self.state_sel):
            hess[sel][sel] = self.coefs[i]
        return hess

    def cost_duu(self, x, u, t, aux):
        return 2 * np.diag(self.coefs[len(self.state_sel):])

    def cost_dux(self, x, u, t, aux):
        return np.zeros((len(u), len(x)))

# swing up and balance of double inverted pendulum
class DDPEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.ddp_horizon = 5
        self.ddp_opt_horizon = 20
        self.current_step = 0
        self.ddp_reward = DDPReward(6, [0,1,2], 1)
        self.ilqr = pylqr.PyLQR_iLQRSolver(T=self.ddp_opt_horizon, plant_dyn=self.plant_dyn, cost=self.ddp_reward)
        self.ilqr.cost_du = self.ddp_reward.cost_du
        self.ilqr.cost_dx = self.ddp_reward.cost_dx
        self.ilqr.cost_duu = self.ddp_reward.cost_duu
        self.ilqr.cost_dxx = self.ddp_reward.cost_dxx
        self.ilqr.cost_dux = self.ddp_reward.cost_dux

        self.control_bounds = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
        self.action_scale = np.pi
        self.torque_scale = 40
        dart_env.DartEnv.__init__(self, 'inverted_double_pendulum.skel', 2, 6, self.control_bounds, dt=0.005)

        self.dart_world = self.dart_worlds[0]
        self.robot_skeleton = self.dart_world.skeletons[-1]


        utils.EzPickle.__init__(self)

    def terminated(self):
        done = abs(self.robot_skeleton.dq[1]) > 35 or abs(self.robot_skeleton.q[0]) > 2.0 or abs(self.robot_skeleton.q[2]) > 35
        return done

    def map_angle(self, ang):
        ang_proc = (np.abs(ang) % (2 * np.pi))
        ang_proc = np.min([ang_proc, (2 * np.pi) - ang_proc])
        return ang_proc

    def reward_func(self, a):
        ang1 = self.map_angle(self.robot_skeleton.q[1])
        ang2 = self.map_angle(self.robot_skeleton.q[2])

        alive_bonus = 2.0

        ang_cost = np.cos(ang1) + np.cos(ang2)
        quad_ctrl_cost = 0.01 * np.square(a).sum()
        com_cost = 2.0 * np.abs(self.robot_skeleton.q[0]) ** 2

        reward = alive_bonus + ang_cost - quad_ctrl_cost - com_cost

        if ang1 < 0.5 and ang2 < 0.5:
            reward += np.max([5 - (ang1) * 4, 0]) + np.max([3 - np.abs(self.robot_skeleton.dq[1]), 0])
        return reward

    def step(self, a):
        for i in range(len(a)):
            if a[i] < self.control_bounds[1][i]:
                a[i] = self.control_bounds[1][i]
            if a[i] > self.control_bounds[0][i]:
                a[i] = self.control_bounds[0][i]

        x0 = self.state_vector()
        prev_u = np.array([np.array([0]) for t in range(self.ddp_opt_horizon)])
        self.ddp_reward.target_state = np.array(a * self.action_scale)
        iter = 15

        reward = 0

        for i in range(self.ddp_horizon):
            self.res = self.ilqr.ilqr_iterate(x0, prev_u, n_itrs=iter, tol=1e-6, verbose=False)

            self.set_state_vector(x0)

            prev_u = np.clip(self.res['u_array_opt'], -1, 1)
            tau = np.zeros(3)
            tau[0] = prev_u[0] * self.torque_scale

            self.do_simulation(tau, self.frame_skip)

            ob = self._get_obs()

            rew = self.reward_func(prev_u[0])

            done = self.terminated()

            x0 = self.state_vector()

            reward += rew
            if done:
                break
        self.current_step += 1
        return ob, reward, done, {}

    def _get_obs(self):
        state = np.concatenate([self.robot_skeleton.q, self.robot_skeleton.dq]).ravel()

        state[1] = self.map_angle(state[1])
        state[2] = self.map_angle(state[2])

        return state

    def reset_model(self):
        self.current_step = 0
        self.dart_world.reset()
        qpos = self.robot_skeleton.q  # + self.np_random.uniform(low=-.1, high=.1, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq  # + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        if self.np_random.uniform(0, 1) > 0.5:
            qpos[1] += np.pi
        else:
            qpos[1] += -np.pi
        self.set_state(qpos, qvel)

        #self.dart_world.skeletons[1].set_positions([0, 0, 0, 100, 0, 0])
        #self.dart_world.skeletons[2].set_positions([0, 0, 0, 200, 0, 0])

        return self._get_obs()

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -4.5
        self._get_viewer().scene.tb._set_theta(0)
        self.track_skeleton_id = 0

    def plant_dyn(self, x, u, t, aux):
        self.set_state_vector(x)
        tau = np.zeros(3)
        tau[0] = u * self.torque_scale
        self.do_simulation(tau, self.frame_skip)
        x_new = self.state_vector()
        return x_new

