import numpy as np
from gym import utils
from gym.envs.dart import dart_env


class DartReacher6LinkEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.target = np.array([0.8, -0.6, 0.6])
        self.action_scale = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        self.control_bounds = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                        [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]])
        obs_dim = 30
        self.include_task = True
        if not self.include_task:
            obs_dim -= 6
        dart_env.DartEnv.__init__(self, 'reacher_multilink/reacher_6link.skel', 4, obs_dim, self.control_bounds, disableViewer=True)

        self.ignore_joint_list = []
        self.ignore_body_list = [0, 1]
        self.joint_property = ['limit']  # what to include in the joint property part
        self.bodynode_property = []
        self.root_type = 'None'
        self.root_id = 0

        utils.EzPickle.__init__(self)

    def about_to_contact(self):
        return False

    def pad_action(self, a):
        return a

    def terminated(self):
        s = self.state_vector()
        done = not (np.isfinite(s).all())
        fingertip = np.array([0.0, -0.25, 0.0])
        vec = self.robot_skeleton.bodynodes[-1].to_world(fingertip) - self.target
        reward_dist = - np.linalg.norm(vec)
        if (-reward_dist < 0.1):
            done = True
        return done

    def advance(self, a):
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.multiply(clamped_control, self.action_scale)
        self.do_simulation(tau, self.frame_skip)

    def reward_func(self, a):
        fingertip = np.array([0.0, -0.25, 0.0])
        vec = self.robot_skeleton.bodynodes[-1].to_world(fingertip) - self.target
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum() * 0.1
        alive_bonus = 0
        reward = reward_dist + reward_ctrl + alive_bonus
        if (-reward_dist < 0.1):
            reward += 30.0
        return reward

    def _step(self, a):
        self.advance(a)

        reward = self.reward_func(a)

        ob = self._get_obs()

        s = self.state_vector()

        self.num_steps += 1

        done = self.terminated()

        return ob, reward, done, {}

    def _get_obs(self):
        theta = self.robot_skeleton.q
        fingertip = np.array([0.0, -0.25, 0.0])
        vec = self.robot_skeleton.bodynodes[-1].to_world(fingertip) - self.target
        if self.include_task:
            return np.concatenate([self.robot_skeleton.q, self.robot_skeleton.dq, self.target, vec]).ravel()
        else:
            return np.concatenate([self.robot_skeleton.q, self.robot_skeleton.dq]).ravel()

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        while True:
            self.target = self.np_random.uniform(low=-1, high=1, size=3)
            if np.linalg.norm(self.target) < 1.5: break
        target_set = [np.array([0.7, 0.0, 0.0]), np.array([-0.3, -0.0, -0.0]), np.array([0, 0.7, 0.0]), np.array([-0.0, -0.3, -0.0]),
                      np.array([-0.0, -0.0, -0.7]), np.array([-0.0, -0.0, -0.3])]
        self.target = target_set[np.random.randint(len(target_set))]

        self.dart_world.skeletons[0].q = [0, 0, 0, self.target[0], self.target[1], self.target[2]]

        self.num_steps = 0

        return self._get_obs()

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -3.5
        self._get_viewer().scene.tb._set_theta(0)
        self.track_skeleton_id = 0