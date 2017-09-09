# This environment is created by Karen Liu (karen.liu@gmail.com)

import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from gym.envs.dart.sub_tasks import *

class DartReacherEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.target = np.array([0.8, -0.6, 0.6])
        self.action_scale = np.array([20, 20, 20, 20, 20])
        self.control_bounds = np.array([[1.0, 1.0, 1.0, 1.0, 1.0],[-1.0, -1.0, -1.0, -1.0, -1.0]])
        self.avg_div = 0

        obs_dim=21
        self.train_UP = False
        self.perturb_MP = False
        self.state_index = 0
        self.split_task_test = True
        self.tasks = TaskList(3)
        self.tasks.add_world_choice_tasks([0, 1, 2])
        if self.split_task_test:
            obs_dim += self.tasks.task_input_dim()

        dart_env.DartEnv.__init__(self, ['reacher.skel', 'reacher_variation1.skel', 'reacher_variation2.skel']\
                                  , 4, obs_dim, self.control_bounds, disableViewer=True)

        for world in self.dart_worlds:
            for i, joint in enumerate(world.skeletons[-1].joints):
                if i == 0:
                    limit = 3.2
                else:
                    limit = 3.14
                for dof in range(joint.num_dofs()):
                    joint.set_position_upper_limit(dof, limit)
                    joint.set_position_lower_limit(dof, -limit)

        self.dart_world=self.dart_worlds[0]
        self.robot_skeleton=self.dart_world.skeletons[-1]

        utils.EzPickle.__init__(self)

    def _step(self, a):
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.multiply(clamped_control, self.action_scale)

        vec = self.robot_skeleton.bodynodes[-1].C - self.target
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(tau).sum() * 0.001
        alive_bonus = 0
        reward = reward_dist + reward_ctrl + alive_bonus

        self.do_simulation(tau, self.frame_skip)
        ob = self._get_obs()

        s = self.state_vector()
        velocity = np.square(s[5:]).sum()

        done = not (np.isfinite(s).all() and (-reward_dist > 0.1))
#        done = not (np.isfinite(s).all() and (-reward_dist > 0.01) and (velocity < 10000))

        return ob, reward, done, {'state_index':self.state_index, 'dyn_model_id':0}

    def _get_obs(self):
        theta = self.robot_skeleton.q
        vec = self.robot_skeleton.bodynodes[-1].C - self.target
        state = np.concatenate([np.cos(theta), np.sin(theta), self.target, self.robot_skeleton.dq, vec]).ravel()

        if self.split_task_test:
            state = np.concatenate([state, self.tasks.get_task_inputs(self.state_index)])

        if self.avg_div > 1:
            return_state = np.zeros(len(state) + self.avg_div)
            return_state[0:len(state)] = state
            return_state[len(state) + self.state_index] = 1
            return return_state

        return state

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        if self.split_task_test:
            self.state_index = np.random.randint(self.tasks.task_num)
            world_choice, pm_id, pm_val, jt_id, jt_val = self.tasks.resample_task(self.state_index)
            if self.dart_world != self.dart_worlds[world_choice]:
                self.dart_world = self.dart_worlds[world_choice]
                self.robot_skeleton = self.dart_world.skeletons[-1]
                qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
                qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
                self.set_state(qpos, qvel)
                if not self.disableViewer:
                    self._get_viewer().sim = self.dart_world
            for ind, jtid in enumerate(jt_id):
                self.robot_skeleton.joints[jtid].set_position_upper_limit(0, jt_val[ind][1])
                self.robot_skeleton.joints[jtid].set_position_lower_limit(0, jt_val[ind][0])

        while True:
            self.target = self.np_random.uniform(low=-1, high=1, size=3)
            #print('target = ' + str(self.target))
            if np.linalg.norm(self.target) < 1.2: break
        '''if self.np_random.uniform(low=-1, high=1) > 0:
            self.target = np.array([0.1,0.8,0.1])
        else:
            self.target = np.array([-0.1, -0.8, -0.1])'''

        self.dart_world.skeletons[0].q=[0, 0, 0, self.target[0], self.target[1], self.target[2]]

        return self._get_obs()


    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -3.5
        self._get_viewer().scene.tb._set_theta(0)
        self.track_skeleton_id = 0
