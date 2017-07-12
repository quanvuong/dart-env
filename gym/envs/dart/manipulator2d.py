import numpy as np
from gym import utils
from gym.envs.dart import dart_env

class DartManipulator2dEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.target = np.array([0.1, 0.01, -0.1])
        self.action_scale = np.array([20, 20, 20])
        self.control_bounds = np.array([[1.0, 1.0, 1.0],[-1.0, -1.0, -1.0]])
        dart_env.DartEnv.__init__(self, 'manipulator2d.skel', 2, 10, self.control_bounds, dt=0.003, disableViewer=False)
        for s in self.dart_world.skeletons:
            s.set_self_collision_check(False)
            '''for n in s.bodynodes:
                n.set_collidable(True)'''

        self.current_task = 0 # 0: reaching, 1: push in close range, 2: push away

        utils.EzPickle.__init__(self)

    def _step(self, a):
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.multiply(clamped_control, self.action_scale)

        self.do_simulation(tau, self.frame_skip)
        ob = self._get_obs()

        vec = self.robot_skeleton.bodynodes[-1].com() - self.target

        if self.current_task == 1 or self.current_task == 2:
            vec = self.dart_world.skeletons[2].com()[[0, 2]] - self.push_target

        reward_dist = self.init_dist - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()*0.01
        reward = reward_dist + reward_ctrl

        s = self.state_vector()
        #done = not (np.isfinite(s).all() and (-reward_dist > 0.02))
        done = False

        return ob, reward, done, {'done_return':done}

    def _get_obs(self):
        reach_target = [self.target[0], self.target[2]]
        if self.current_task == 1 or self.current_task == 2:
            reach_target = self.dart_world.skeletons[2].com()[[0, 2]]
        return np.concatenate([self.robot_skeleton.q, self.robot_skeleton.dq, reach_target, self.push_target]).ravel()

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        while True:
            self.target = self.np_random.uniform(low=-.2, high=.2, size=3)
            self.target[1] = 0.0
            if np.linalg.norm(self.target) < .2: break
        self.target[1] = 0.01
        #options = [np.array([0.1, 0.01, 0.1]), np.array([-0.2, 0.01, 0.05]), np.array([0.2, 0.01, -0.15])]
        #self.target = options[int(np.random.random()*len(options))]

        self.dart_world.skeletons[1].q=[0, 0, 0, self.target[0], self.target[1], self.target[2]]

        self.dart_world.skeletons[2].set_positions([0,0,0,0.8,0.015,0])
        self.dart_world.skeletons[2].set_velocities([0,0,0,0,0,0])
        self.dart_world.skeletons[3].q=[0,0,0,0.9,0.015,0]

        self.current_task = 2#np.random.randint(3)
        self.push_target = np.random.uniform(low=-0.5, high = 0.5, size = 2)

        if self.current_task != 0:
            self.target = self.np_random.uniform(low=-.1, high=.1, size=3)
            self.target = np.array([0.2, 0.015, 0.05])
            self.dart_world.skeletons[2].set_positions([0,0,0,self.target[0],0.015,self.target[2]])
        if self.current_task == 1:
            self.push_target = np.random.uniform(low=-0.1, high = 0.1, size = 2)
            self.dart_world.skeletons[3].q=[0,0,0,self.push_target[0],0.015,self.push_target[1]]
        if self.current_task == 2:
            flip1 = np.random.randint(2) * 2 - 1
            flip2 = np.random.randint(2) * 2 - 1
            self.push_target = np.random.uniform(low= 0.4, high = 0.5, size = 2)
            self.push_target *= np.array([flip1, flip2])
            self.dart_world.skeletons[3].q=[0,0,0,self.push_target[0],0.015,self.push_target[1]]

        vec = self.robot_skeleton.bodynodes[-1].com() - self.target

        if self.current_task == 1 or self.current_task == 2:
            vec = self.dart_world.skeletons[2].com()[[0, 2]] - self.push_target
        self.init_dist = np.linalg.norm(vec)

        return self._get_obs()


    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -1.5
        self._get_viewer().scene.tb._set_theta(-45)
        self.track_skeleton_id = 0
