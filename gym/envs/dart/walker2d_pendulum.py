__author__ = 'yuwenhao'

import numpy as np
from gym import utils
from gym.envs.dart import dart_env


class DartWalker2dPendulumEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0]*6,[-1.0]*6])
        self.action_scale = np.array([100, 100, 20, 100, 100, 20])
        obs_dim = 19

        self.current_task = 0 # 0: balance a pole, 1: walk forward, 2: walk forward with balanced pole
        obs_dim += 1

        dart_env.DartEnv.__init__(self, 'walker2d_pendulum.skel', 4, obs_dim, self.control_bounds, disableViewer=False)

        self.dart_world.set_collision_detector(3)  # 3 is ode collision detector


        utils.EzPickle.__init__(self)

    def _step(self, a):
        pre_state = [self.state_vector()]

        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.zeros(self.robot_skeleton.ndofs)

        tau[4:] = clamped_control * self.action_scale
        posbefore = self.robot_skeleton.q[0]
        self.do_simulation(tau, self.frame_skip)
        posafter,ang = self.robot_skeleton.q[0,2]
        height = self.robot_skeleton.bodynodes[2].com()[1]

        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()

        alive_bonus = 1.0
        vel = (posafter - posbefore) / self.dt
        reward = 0

        reward += alive_bonus
        reward -= 1e-2 * np.square(a).sum()

        action_vio = np.sum(np.exp(np.max([(a-self.control_bounds[0]*1.5), [0]*6], axis=0)) - [1]*6)
        action_vio += np.sum(np.exp(np.max([(self.control_bounds[1]*1.5-a), [0]*6], axis=0)) - [1]*6)
        reward -= 0.1*action_vio


        # uncomment to enable knee joint limit penalty
        joint_limit_penalty = 0
        for j in [-2, -5]:
            if (self.robot_skeleton.q_lower[j] - self.robot_skeleton.q[j]) > -0.05:
                joint_limit_penalty += abs(1.5)
            if (self.robot_skeleton.q_upper[j] - self.robot_skeleton.q[j]) < 0.05:
                joint_limit_penalty += abs(1.5)

        reward -= 5e-1 * joint_limit_penalty

        if self.current_task == 1 or self.current_task == 2:
            reward += vel#-(vel-1.0)**2
        if self.current_task == 0 or self.current_task == 2:
            reward -= np.abs(ang+self.robot_skeleton.q[3])

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .8) and (height < 2.0) and (abs(ang) < 1.0))
        if self.current_task == 0 or self.current_task == 2:
            done = done or not(np.abs(ang+self.robot_skeleton.q[3]) < 0.8)
        '''qpos = self.robot_skeleton.q
        qvel = self.robot_skeleton.dq
        qpos[0:3] = np.array([0, 0, 0.8])
        qvel[0:3] = np.array([0, 0, 0])
        self.set_state(qpos, qvel)'''

        ob = self._get_obs()

        return ob, reward, done, {'pre_state':pre_state, 'vel_rew':(posafter - posbefore) / self.dt, 'action_rew':1e-3 * np.square(a).sum(), 'forcemag':1e-7*total_force_mag, 'done_return':done}

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            np.clip(self.robot_skeleton.dq,-10,10)
        ])
        state[0] = self.robot_skeleton.bodynodes[2].com()[1]

        state = np.concatenate([state, [self.current_task]])

        return state

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        self.current_task = np.random.randint(0, 2)

        return self._get_obs()

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -5.5
