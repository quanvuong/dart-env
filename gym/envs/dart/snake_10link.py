import numpy as np
from gym import utils
from gym.envs.dart import dart_env


class DartSnake10LinkEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ,1.0 ,1.0 ,1.0],[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]])
        self.action_scale = 200
        self.include_action_in_obs = False
        self.randomize_dynamics = False
        obs_dim = 23

        if self.include_action_in_obs:
            obs_dim += len(self.control_bounds[0])
            self.prev_a = np.zeros(len(self.control_bounds[0]))

        dart_env.DartEnv.__init__(self, 'snake_multilink/snake_10link.skel', 4, obs_dim, self.control_bounds, disableViewer=True)

        if self.randomize_dynamics:
            self.bodynode_original_masses = []
            self.bodynode_original_frictions = []
            for bn in self.robot_skeleton.bodynodes:
                self.bodynode_original_masses.append(bn.mass())
                self.bodynode_original_frictions.append(bn.friction_coeff())

        self.dart_world.set_collision_detector(3)

        # setups for controller articunet
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

        self.act_net.append([self.state_dim, 1, 64, 1, 'revolute_act'])
        self.vf_net.append([self.state_dim, 1, 64, 1, 'vf_out'])
        self.merg_net.append([self.state_dim, 1, 64, 1, 'merger'])

        # value function modules
        if not self.include_action_in_obs:
            self.net_vf_modules.append([[10, 22], 3, None])
            self.net_vf_modules.append([[9, 21], 3, [0]])
            self.net_vf_modules.append([[8, 20], 3, [1]])
            self.net_vf_modules.append([[7, 19], 3, [2]])
            self.net_vf_modules.append([[6, 18], 3, [3]])
            self.net_vf_modules.append([[5, 17], 3, [4]])
            self.net_vf_modules.append([[4, 16], 3, [5]])
            self.net_vf_modules.append([[3, 15], 3, [6]])
            self.net_vf_modules.append([[2, 14], 3, [7]])
        else:
            self.net_vf_modules.append([[10, 22, 31], 3, None])
            self.net_vf_modules.append([[9, 21, 30], 3, [0]])
            self.net_vf_modules.append([[8, 20, 29], 3, [1]])
            self.net_vf_modules.append([[7, 19, 28], 3, [2]])
            self.net_vf_modules.append([[6, 18, 27], 3, [3]])
            self.net_vf_modules.append([[5, 17, 26], 3, [4]])
            self.net_vf_modules.append([[4, 16, 25], 3, [5]])
            self.net_vf_modules.append([[3, 15, 24], 3, [6]])
            self.net_vf_modules.append([[2, 14, 23], 3, [7]])
        self.net_vf_modules.append([[0, 1, 11, 12, 13], 2, [8]])
        self.net_vf_modules.append([[], 5, [9]])

        # policy modules
        if not self.include_action_in_obs:
            self.net_modules.append([[10, 22], 1, None])
            self.net_modules.append([[9, 21], 1, [0]])
            self.net_modules.append([[8, 20], 1, [1]])
            self.net_modules.append([[7, 19], 1, [2]])
            self.net_modules.append([[6, 18], 1, [3]])
            self.net_modules.append([[5, 17], 1, [4]])
            self.net_modules.append([[4, 16], 1, [5]])
            self.net_modules.append([[3, 15], 1, [6]])
            self.net_modules.append([[2, 14], 1, [7]])
        else:
            self.net_modules.append([[10, 22, 31], 1, None])
            self.net_modules.append([[9, 21, 30], 1, [0]])
            self.net_modules.append([[8, 20, 29], 1, [1]])
            self.net_modules.append([[7, 19, 28], 1, [2]])
            self.net_modules.append([[6, 18, 27], 1, [3]])
            self.net_modules.append([[5, 17, 26], 1, [4]])
            self.net_modules.append([[4, 16, 25], 1, [5]])
            self.net_modules.append([[3, 15, 24], 1, [6]])
            self.net_modules.append([[2, 14, 23], 1, [7]])
        self.net_modules.append([[0, 1, 11, 12, 13], 0, [8]])

        self.net_modules.append([[], 6, [9, 8], None, False])
        self.net_modules.append([[], 6, [9, 7], None, False])
        self.net_modules.append([[], 6, [9, 6], None, False])
        self.net_modules.append([[], 6, [9, 5], None, False])
        self.net_modules.append([[], 6, [9, 4], None, False])
        self.net_modules.append([[], 6, [9, 3], None, False])
        self.net_modules.append([[], 6, [9, 2], None, False])
        self.net_modules.append([[], 6, [9, 1], None, False])
        self.net_modules.append([[], 6, [9, 0], None, False])

        self.net_modules.append([[], 4, [10]])
        self.net_modules.append([[], 4, [11]])
        self.net_modules.append([[], 4, [12]])
        self.net_modules.append([[], 4, [13]])
        self.net_modules.append([[], 4, [14]])
        self.net_modules.append([[], 4, [15]])
        self.net_modules.append([[], 4, [16]])
        self.net_modules.append([[], 4, [17]])
        self.net_modules.append([[], 4, [18]])

        self.net_modules.append([[], None, [19, 20, 21, 22, 23, 24, 25, 26, 27], None, False])

        # dynamics modules # NEED TO BE FIXED LATER
        self.dyn_enc_net = []
        self.dyn_act_net = []  # using actor as decoder
        self.dyn_merg_net = []
        self.dyn_net_modules = []
        self.dyn_enc_net.append([self.state_dim, 12, 256, 1, 'dyn_free_enc'])
        self.dyn_enc_net.append([self.state_dim, 3, 256, 1, 'dyn_revolute_enc'])
        self.dyn_act_net.append([self.state_dim, 2, 256, 1, 'dyn_free_dec'])
        self.dyn_act_net.append([self.state_dim, 12, 256, 1, 'dyn_revolute_dec'])
        self.dyn_merg_net.append([self.state_dim, 1, 256, 1, 'dyn_merger'])
        self.dyn_net_modules.append([[4, 9, 11], 1, None])
        self.dyn_net_modules.append([[3, 8, 10], 1, [0]])
        self.dyn_net_modules.append([[0, 1, 2, 5, 6, 7], 0, [1]])
        self.dyn_net_modules.append([[], 4, [2, 1], None, False])
        self.dyn_net_modules.append([[], 4, [2, 0], None, False])
        self.dyn_net_modules.append([[], 2, [2]])
        self.dyn_net_modules.append([[], 3, [3]])
        self.dyn_net_modules.append([[], 3, [4]])
        self.dyn_net_modules.append([[], None, [5, 6, 7], None, False])
        self.dyn_net_reorder = np.array([0, 1, 2, 6, 8, 3, 4, 5, 7, 9], dtype=np.int32)

        for i in range(0, len(self.robot_skeleton.bodynodes)):
            self.robot_skeleton.bodynodes[i].set_friction_coeff(0)
        self.robot_skeleton.bodynodes[-1].set_friction_coeff(5)

        utils.EzPickle.__init__(self)

    def do_simulation(self, tau, n_frames):
        for _ in range(n_frames):
            for bn in self.robot_skeleton.bodynodes:
                bn_vel = bn.com_spatial_velocity()
                norm_dir = bn.to_world([0, 0, 1]) - bn.to_world([0, 0, 0])
                vel_pos = bn_vel[3:] + np.cross(bn_vel[0:3], norm_dir) * 0.05
                vel_neg = bn_vel[3:] - np.cross(bn_vel[0:3], norm_dir) * 0.05
                fluid_force = [0.0, 0.0, 0.0]
                if np.dot(vel_pos, norm_dir) > 0.0:
                    fluid_force = -50.0 * np.dot(vel_pos, norm_dir) * norm_dir
                if np.dot(vel_neg, norm_dir) < 0.0:
                    fluid_force = -50.0 * np.dot(vel_neg, norm_dir) * norm_dir
                bn.add_ext_force(fluid_force)

            self.robot_skeleton.set_forces(tau)
            self.dart_world.step()

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
        posafter = self.robot_skeleton.q[0]
        deviation = self.robot_skeleton.q[2]

        alive_bonus = 0.1
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        reward -= np.abs(deviation) * 0.1
        s = self.state_vector()
        self.accumulated_rew += reward
        self.num_steps += 1.0
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and abs(deviation) < 1.5)
        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            self.robot_skeleton.dq,
        ])

        if self.include_action_in_obs:
            state = np.concatenate([state, self.prev_a])

        return state


    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        state = self._get_obs()

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