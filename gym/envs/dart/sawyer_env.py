import numpy as np
from gym import utils, error
from gym.envs.dart import dart_env

from gym.envs.dart.static_sawyer_window import *
from gym.envs.dart.norender_window import *

from gym.envs.dart.parameter_managers import *

try:
    import pydart2 as pydart
    import pydart2.joint as Joint
    from pydart2.gui.trackball import Trackball
    pydart.init()
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install pydart2.)".format(e))


class DartSawyerEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([np.ones(13), -1*np.ones(13)])
        #self.control_bounds = np.array([[1.0, 1.0, 1.0],[-1.0, -1.0, -1.0]])
        self.action_scale = 200
        obs_dim = 14
        self.param_manager = hopperContactMassManager(self)

        dart_env.DartEnv.__init__(self, 'sawyer_description/urdf/sawyer_arm.urdf', 5, obs_dim, self.control_bounds, disableViewer=False)

        #self.dart_world.set_collision_detector(3) # 3 is ode collision detector
        self.numSteps = 0

        utils.EzPickle.__init__(self)

        # lock the first 6 dofs
        for i in range(6):
            self.robot_skeleton.dof(i).set_position_lower_limit(0)
            self.robot_skeleton.dof(i).set_position_upper_limit(0)
        #self.robot_skeleton.dof(3).set_position_lower_limit(-0.01)
        #self.robot_skeleton.dof(3).set_position_upper_limit(0.01)
        #self.robot_skeleton.dof(4).set_position_lower_limit(-0.01)
        #self.robot_skeleton.dof(4).set_position_upper_limit(0.01)
        #self.robot_skeleton.dof(5).set_position_lower_limit(-0.01)
        #self.robot_skeleton.dof(5).set_position_upper_limit(0.01)

        print("BodyNodes: ")
        for bodynode in self.robot_skeleton.bodynodes:
            print("     : " + bodynode.name)

        print("Joints: ")
        for joint in self.robot_skeleton.joints:
            print("     : " + joint.name)
            joint.set_position_limit_enforced()

        print("Dofs: ")
        for dof in self.robot_skeleton.dofs:
            print("     : " + dof.name)
            #print("         damping: " + str(dof.damping_coefficient()))
            dof.set_damping_coefficient(2.0)
        self.robot_skeleton.joints[0].set_actuator_type(Joint.Joint.LOCKED)


    def _step(self, a):

        #print("-----------------")
        #print(self.robot_skeleton.q)
        #print("a: " + str(a))

        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau = clamped_control * self.action_scale
        self.do_simulation(tau, self.frame_skip)

        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()


        alive_bonus = 1.0
        reward = 0
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()

        s = self.state_vector()
        done = False
        ob = self._get_obs()

        return ob, reward, done, {'model_parameters':self.param_manager.get_simulator_parameters(), 'action_rew':1e-3 * np.square(a).sum(), 'forcemag':1e-7*total_force_mag, 'done_return':done}

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[6:],
            self.robot_skeleton.dq[6:]
        ])
        state[0] = self.robot_skeleton.bodynodes[2].com()[1]

        return state

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        state = self._get_obs()

        return state

    def viewer_setup(self):
        a=0
        #self._get_viewer().scene.tb.trans[2] = -5.5

    def getViewer(self, sim, title=None):
        # glutInit(sys.argv)
        win = NoRenderWindow(sim, title)
        if not self.disableViewer:
            win = StaticSawyerWindow(sim, title, self)
            win.scene.add_camera(Trackball(theta=-45.0, phi = 0.0, zoom=0.2), 'gym_camera')
            win.scene.set_camera(win.scene.num_cameras()-1)

        # to add speed,
        if self._obs_type == 'image':
            win.run(self.screen_width, self.screen_height, _show_window=self.visualize)
        else:
            win.run(_show_window=self.visualize)
        return win