import numpy as np
from gym import utils
from gym.envs.dart import dart_env


class DartHopper7LinkEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]])
        self.action_scale = 100
        self.include_action_in_obs = False
        self.randomize_dynamics = False
        obs_dim = 17

        if self.include_action_in_obs:
            obs_dim += len(self.control_bounds[0])
            self.prev_a = np.zeros(len(self.control_bounds[0]))

        dart_env.DartEnv.__init__(self, 'hopper_multilink/hopperid_7link.skel', 4, obs_dim, self.control_bounds, disableViewer=True)

        if self.randomize_dynamics:
            self.bodynode_original_masses = []
            self.bodynode_original_frictions = []
            for bn in self.robot_skeleton.bodynodes:
                self.bodynode_original_masses.append(bn.mass())
                self.bodynode_original_frictions.append(bn.friction_coeff())

        self.dart_world.set_collision_detector(3)

        # setups for articunet
        self.state_dim = 32
        self.enc_net = []
        self.act_net = []
        self.vf_net = []
        self.merg_net = []
        self.net_modules = []
        self.net_vf_modules = []
        self.enc_net.append([self.state_dim, 5, 64, 1, 'planar_enc'])
        if not self.include_action_in_obs:
            self.enc_net.append([self.state_dim, 2, 64, 1, 'revolute_enc'])
        else:
            self.enc_net.append([self.state_dim, 3, 64, 1, 'revolute_enc'])
        self.enc_net.append([self.state_dim, 5, 64, 1, 'vf_planar_enc'])
        if not self.include_action_in_obs:
            self.enc_net.append([self.state_dim, 2, 64, 1, 'vf_revolute_enc'])
        else:
            self.enc_net.append([self.state_dim, 3, 64, 1, 'vf_revolute_enc'])

        # specialize ankle joint
        self.enc_net.append([self.state_dim, 2, 64, 1, 'ankle_enc'])

        self.act_net.append([self.state_dim, 1, 64, 1, 'revolute_act'])

        # specialize ankle joint
        self.act_net.append([self.state_dim, 1, 64, 1, 'ankle_act'])

        self.vf_net.append([self.state_dim, 1, 64, 1, 'vf_out'])
        self.merg_net.append([self.state_dim, 1, 64, 1, 'merger'])

        # 4 - 5, 5 - 7, 6 - 8

        # value function modules
        if not self.include_action_in_obs:
            self.net_vf_modules.append([[7, 16], 3, None])
            self.net_vf_modules.append([[6, 15], 3, [0]])
            self.net_vf_modules.append([[5, 14], 3, [1]])
            self.net_vf_modules.append([[4, 13], 3, [2]])
            self.net_vf_modules.append([[3, 12], 3, [3]])
            self.net_vf_modules.append([[2, 11], 3, [4]])
        else:
            self.net_vf_modules.append([[7, 16, 19], 3, None])
            self.net_vf_modules.append([[6, 15, 19], 3, [0]])
            self.net_vf_modules.append([[5, 14, 18], 3, [1]])
            self.net_vf_modules.append([[4, 13, 17], 3, [2]])
            self.net_vf_modules.append([[3, 12, 16], 3, [3]])
            self.net_vf_modules.append([[2, 11, 15], 3, [4]])
        self.net_vf_modules.append([[0, 1, 8, 9, 10], 2, [5]])
        self.net_vf_modules.append([[], 7, [6]])

        # policy modules
        if not self.include_action_in_obs:
            self.net_modules.append([[7, 16], 4, None])
            self.net_modules.append([[6, 15], 4, [0]])
            self.net_modules.append([[5, 14], 4, [1]])
            self.net_modules.append([[4, 13], 1, [2]])
            self.net_modules.append([[3, 12], 1, [3]])
            self.net_modules.append([[2, 11], 1, [4]])
        else:
            self.net_modules.append([[7, 16, 19], 4, None])
            self.net_modules.append([[6, 15, 19], 4, [0]])
            self.net_modules.append([[5, 14, 18], 4, [1]])
            self.net_modules.append([[4, 13, 17], 1, [2]])
            self.net_modules.append([[3, 12, 16], 1, [3]])
            self.net_modules.append([[2, 11, 15], 1, [4]])
        self.net_modules.append([[0, 1, 8, 9, 10], 0, [5]])

        self.net_modules.append([[], 8, [6, 5], None, False])
        self.net_modules.append([[], 8, [6, 4], None, False])
        self.net_modules.append([[], 8, [6, 3], None, False])
        self.net_modules.append([[], 8, [6, 2], None, False])
        self.net_modules.append([[], 8, [6, 1], None, False])
        self.net_modules.append([[], 8, [6, 0], None, False])

        self.net_modules.append([[], 5, [7]])
        self.net_modules.append([[], 5, [8]])
        self.net_modules.append([[], 5, [9]])
        self.net_modules.append([[], 5, [10]])
        self.net_modules.append([[], 5, [11]])
        self.net_modules.append([[], 6, [12]])

        self.net_modules.append([[], None, [13, 14, 15, 16, 17, 18], None, False])

        # dynamic model
        self.dyn_enc_net = []
        self.dyn_act_net = []  # using actor as decoder
        self.dyn_merg_net = []
        self.dyn_net_modules = []
        self.dyn_enc_net.append([self.state_dim, 6, 256, 1, 'dyn_planar_enc'])
        self.dyn_enc_net.append([self.state_dim, 3, 256, 1, 'dyn_revolute_enc'])
        self.dyn_act_net.append([self.state_dim, 2, 256, 1, 'dyn_planar_dec'])
        self.dyn_act_net.append([self.state_dim, 6, 256, 1, 'dyn_revolute_dec'])
        self.dyn_merg_net.append([self.state_dim, 1, 256, 1, 'dyn_merger'])
        self.dyn_net_modules.append([[8, 17, 23], 1, None])
        self.dyn_net_modules.append([[7, 16, 22], 1, [0]])
        self.dyn_net_modules.append([[6, 15, 21], 1, [1]])
        self.dyn_net_modules.append([[5, 14, 20], 1, [2]])
        self.dyn_net_modules.append([[4, 13, 19], 1, [3]])
        self.dyn_net_modules.append([[3, 12, 18], 1, [4]])
        self.dyn_net_modules.append([[0, 1, 2, 9, 10, 11], 0, [5]])

        self.dyn_net_modules.append([[], 4, [6, 5], None, False])
        self.dyn_net_modules.append([[], 4, [6, 4], None, False])
        self.dyn_net_modules.append([[], 4, [6, 3], None, False])
        self.dyn_net_modules.append([[], 4, [6, 2], None, False])
        self.dyn_net_modules.append([[], 4, [6, 1], None, False])
        self.dyn_net_modules.append([[], 4, [6, 0], None, False])

        self.dyn_net_modules.append([[], 2, [6]])
        self.dyn_net_modules.append([[], 3, [7]])
        self.dyn_net_modules.append([[], 3, [8]])
        self.dyn_net_modules.append([[], 3, [9]])
        self.dyn_net_modules.append([[], 3, [10]])
        self.dyn_net_modules.append([[], 3, [11]])
        self.dyn_net_modules.append([[], 3, [12]])
        self.dyn_net_modules.append([[], None, [13, 14, 15, 16, 17, 18], None, False])
        self.dyn_net_reorder = np.array([0, 1, 2, 6, 8, 10, 12, 14, 16, 3, 4, 5, 7, 9, 11, 13, 15, 17], dtype=np.int32)

        utils.EzPickle.__init__(self)

    def advance(self, a):
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        if self.include_action_in_obs:
            self.prev_a = np.copy(clamped_control)
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[3:] = clamped_control * self.action_scale

        self.do_simulation(tau, self.frame_skip)

    def _step(self, a):
        pre_state = [self.state_vector()]

        posbefore = self.robot_skeleton.q[0]
        self.advance(a)
        posafter,ang = self.robot_skeleton.q[0,2]
        height = self.robot_skeleton.bodynodes[2].com()[1]

        fall_on_ground = False
        contacts = self.dart_world.collision_result.contacts
        for contact in contacts:
            if contact.bodynode1 == self.robot_skeleton.bodynodes[2] or contact.bodynode2 == \
                    self.robot_skeleton.bodynodes[2]:
                fall_on_ground = True

        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        self.accumulated_rew += reward
        self.num_steps += 1.0
        #print(self.num_steps)
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > self.init_height - 0.4) and (height < self.init_height + 0.5) and (abs(ang) < .4))
        if not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all()):
            reward = 0
        #if fall_on_ground:
        #    done = True
        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            self.robot_skeleton.dq,
        ])
        state[0] = self.robot_skeleton.bodynodes[2].com()[1]

        if self.include_action_in_obs:
            state = np.concatenate([state, self.prev_a])

        return state


    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        state = self._get_obs()

        self.init_height = self.robot_skeleton.bodynodes[2].com()[1]

        if self.include_action_in_obs:
            self.prev_a = np.zeros(len(self.control_bounds[0]))

        self.accumulated_rew = 0.0
        self.num_steps = 0.0

        if self.randomize_dynamics:
            for i in range(len(self.robot_skeleton.bodynodes)):
                self.robot_skeleton.bodynodes[i].set_mass(
                    self.bodynode_original_masses[i] + np.random.uniform(-1.5, 1.5))
                self.robot_skeleton.bodynodes[i].set_friction_coeff(
                    self.bodynode_original_frictions[i] + np.random.uniform(-0.5, 0.5))

        return state

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -5.5