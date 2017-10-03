import numpy as np
from gym import utils
from gym.envs.dart import dart_env

class DartBallWalkerEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        control_bounds = np.array([[1.0, 1.0],[-1.0, -1.0]])
        self.action_scale = 40
        dart_env.DartEnv.__init__(self, 'ball_walker.skel', 4, 11, control_bounds, disableViewer=True)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        #if abs(a[0]) > 1:
        #    a[0] = np.sign(a[0])
        clamp_a = np.clip(a, -1, 1)

        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[0] = clamp_a[0] * self.action_scale
        tau[2] = clamp_a[1] * self.action_scale

        pbefore = self.robot_skeleton.C[0]
        self.do_simulation(tau, self.frame_skip)
        pafter = self.robot_skeleton.C[0]
        ob = self._get_obs()

        reward = (pafter - pbefore)/self.dt - 0.1 * np.sum(a**2)

        notdone = np.isfinite(ob).all() and np.linalg.norm(self.robot_skeleton.com_velocity()) < 10
        done = not notdone
        return ob, reward, done, {'dyn_model_id':0}


    def _get_obs(self):
        return np.concatenate([self.robot_skeleton.q[1:], self.robot_skeleton.dq]).ravel()

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        return self._get_obs()


    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -5.5
