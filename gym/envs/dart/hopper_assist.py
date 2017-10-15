import numpy as np
from gym import utils
from gym.envs.dart import dart_env

from gym.envs.dart.parameter_managers import *
from gym.envs.dart.sub_tasks import *
import copy

import joblib, os


class DartHopperAssistEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
        self.action_scale = np.array([200.0, 200.0, 200.0])

        obs_dim = 11

        self.t = 0

        self.spd_tasks = [] # [dim, gain, range, vel]
        # move forward
        self.spd_tasks.append([0, 1500, [1.5], True])

        # not fall
        self.spd_tasks.append([1, 1500, [0.7-1.25, 100], False])

        # not rotate
        self.spd_tasks.append([2, 1500, [-0.4, 0.4], False])

        dart_env.DartEnv.__init__(self, 'hopper_capsule.skel', 4, obs_dim, self.control_bounds, disableViewer=True)

        self.dart_world.set_collision_detector(3)
        self.robot_skeleton = self.dart_world.skeletons[-1]

        self.worst_single_reward = 100

        utils.EzPickle.__init__(self)

    def _spd(self, target_q, id, kp, target_dq = 0.0):
        kp_diag = np.array([kp])
        self.Kp = np.diagflat(kp_diag)
        self.Kd = np.diagflat(kp_diag*self.dt/self.frame_skip)
        if target_dq > 0: # if it's trying to enforce velocity
            self.Kd[0] = self.Kp[0]
            self.Kp *= 0
        invM = np.linalg.inv([self.robot_skeleton.M[id][id]] + self.Kd * self.dt/self.frame_skip)
        p = -self.Kp.dot(self.robot_skeleton.q[id] + self.robot_skeleton.dq[id] * self.dt/self.frame_skip - target_q[id])
        d = -self.Kd.dot(self.robot_skeleton.dq[id])
        qddot = invM.dot(-self.robot_skeleton.c[id] + p + d)
        tau = p + d - self.Kd.dot(qddot) * self.dt/self.frame_skip + self.Kd * target_dq
        return tau

    def do_simulation(self, tau, n_frames):
        total_assistive_tau = 0
        for _ in range(n_frames):
            for spd_task in self.spd_tasks:
                tq = self.robot_skeleton.q
                if not spd_task[3]:
                    if len(spd_task[2]) == 1:
                        tq[spd_task[0]] = spd_task[2][0]
                    else:
                        if tq[spd_task[0]] > spd_task[2][1]:
                            tq[spd_task[0]] = spd_task[2][1] - 0.001
                        elif tq[spd_task[0]] < spd_task[2][0]:
                            tq[spd_task[0]] = spd_task[2][0] + 0.001
                        else:
                            continue
                    spdtau = self._spd(tq, spd_task[0], spd_task[1])
                else:
                    spdtau = self._spd(tq, spd_task[0], spd_task[1], spd_task[2][0])
                tau[spd_task[0]] = spdtau[0][0]
                if np.abs(spdtau[0][0]) > 20:
                    total_assistive_tau += 1
            self.robot_skeleton.set_forces(tau)
            self.dart_world.step()
        return total_assistive_tau

    def advance(self, a):
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[3:] = clamped_control * self.action_scale

        return self.do_simulation(tau, self.frame_skip)

    def _step(self, a):
        self.t += self.dt
        prev_obs = self._get_obs()
        pre_state = [self.state_vector()]
        posbefore = self.robot_skeleton.q[0]
        heightbefore = self.robot_skeleton.q[1]
        state_act = np.concatenate([self.state_vector(), np.clip(a, -1, 1)])
        state_pre = np.copy(self.state_vector())
        total_assistive_tau = self.advance(a)
        posafter, heightafter, ang = self.robot_skeleton.q[0, 1, 2]
        height = self.robot_skeleton.bodynodes[2].com()[1]

        fall_on_ground = False
        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()
            if contact.bodynode1 != self.robot_skeleton.bodynodes[-1] and contact.bodynode2 != \
                    self.robot_skeleton.bodynodes[-1]:
                fall_on_ground = True

        joint_limit_penalty = 0
        for j in [-2]:
            if (self.robot_skeleton.q_lower[j] - self.robot_skeleton.q[j]) > -0.05:
                joint_limit_penalty += abs(1.5)
            if (self.robot_skeleton.q_upper[j] - self.robot_skeleton.q[j]) < 0.05:
                joint_limit_penalty += abs(1.5)

        '''alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt

        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        reward -= 5e-1 * joint_limit_penalty
        # reward -= 1e-7 * total_force_mag'''

        reward = -(total_assistive_tau)

        s = self.state_vector()
        done = not (
        np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (np.abs(self.robot_skeleton.dq) < 100).all())
        #and (height > self.height_threshold_low) and (abs(ang) < .4))

        ob = self._get_obs()

        self.cur_step += 1

        if reward < self.worst_single_reward:
            self.worst_single_reward = reward

        if done:
            reward -= (1000-self.cur_step)*3.0*self.frame_skip

        return ob, reward, done, {'dyn_model_id': 0}

    def _get_obs(self):
        state = np.concatenate([
            self.robot_skeleton.q[1:],
            np.clip(self.robot_skeleton.dq, -10, 10)
        ])
        state[0] = self.robot_skeleton.bodynodes[2].com()[1]

        return state

    def reset_model(self):
        for world in self.dart_worlds:
            world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        state = self._get_obs()

        self.cur_step = 0

        self.height_threshold_low = 0.56 * self.robot_skeleton.bodynodes[2].com()[1]
        self.t = 0
        return state

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -5.5
