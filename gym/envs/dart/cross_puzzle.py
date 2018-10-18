import numpy as np
from gym import utils
from gym.envs.dart import dart_env

class DartCrossPuzzle(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.target = np.array([0.0, 0.3, 13.5])
        self.action_scale = 300

        self.hierarchical = True
        self.h_horizon = 50
        self.action_scale = 2.0
        self.num_way_point = 2

        self.control_bounds = np.array([[1.0, 1.0], [-1.0, -1.0]])

        if self.hierarchical:
            self.control_bounds = np.array([[1.0] * (2*self.num_way_point), [-1.0] * (2*self.num_way_point)])

        dart_env.DartEnv.__init__(self, 'cross_puzzle.skel', 2, 4, self.control_bounds, dt=0.01, disableViewer=True)

        utils.EzPickle.__init__(self)

    def step(self, a):
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]

        if not self.hierarchical:
            tau = np.zeros(3)
            tau[[0,2]] = np.multiply(clamped_control, self.action_scale)
            self.do_simulation(tau, self.frame_skip)
        else:
            prev_pos = self.robot_skeleton.bodynodes[-1].com()[[0,2]]
            scaled_a = a * self.action_scale
            targets = []
            for i in range(self.num_way_point):
                new_pos = prev_pos + scaled_a[i*2:i*2+2]
                targets.append(new_pos)
                prev_pos = np.copy(new_pos)

            for i in range(self.h_horizon):
                tau = np.zeros(3)
                tvec = targets[int(i / (self.h_horizon * 1.0 / self.num_way_point))] - self.robot_skeleton.bodynodes[-1].com()[[0,2]]
                tvec /= np.linalg.norm(tvec)
                tau[[0,2]] = tvec * 100
                self.do_simulation(tau, self.frame_skip)
            self.dart_world.skeletons[-2].q = np.array([targets[0][0], 0.0, targets[0][1]])

        ob = self._get_obs()

        vec = self.robot_skeleton.bodynodes[-1].com() - self.target

        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()*0.1
        reward = reward_dist + reward_ctrl

        if self.robot_skeleton.bodynodes[-1].com()[1] < 0.0:
            reward -= 5

        s = self.state_vector()
        #done = not (np.isfinite(s).all() and (-reward_dist > 0.02))
        done = False

        return ob, reward, done, {'done_return':done}

    def _get_obs(self):
        pos = self.robot_skeleton.bodynodes[-1].C
        vel  =self.robot_skeleton.bodynodes[-1].dC
        return np.array([pos[0], pos[2], vel[0], vel[2]])

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qpos[1] = 0.0
        qvel[1] = 0.0
        self.set_state(qpos, qvel)

        return self._get_obs()


    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -30.0
        self._get_viewer().scene.tb._set_theta(-60)
        self.track_skeleton_id = 0
