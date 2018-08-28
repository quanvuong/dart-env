# Contributors: Wenhao Yu (wyu68@gatech.edu) and Dong Xu (donghsu@gatech.edu)

import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT
import sys
import numpy as np
from pydart2.gui.opengl.scene import OpenGLScene
from pydart2.gui.glut.window import *
from gym.envs.dart.static_window import *


class StaticSawyerWindow(StaticGLUTWindow):
    def __init__(self, sim, title, env, extraRenderFunc=None):
        self.env = env
        self.extraRenderFunc = extraRenderFunc
        self.d_down = False #if down, mouse drag changes SPD controller 'd' gain
        self.p_down = False #if down, mouse drag changes SPD controller 'p' gain
        super(StaticSawyerWindow,self).__init__(sim, title)
        
    def extraRender(self):
        'Place any extra rendering functionality here. This can be used to extend the window class'
        #print("Modern Day Warrior ... ")
        #print(self.env.robot_skeleton.q)
        #print(self.env.robot_skeleton.q[3:6])
        # get the window camera transformations
        GL.glLoadIdentity()
        GL.glTranslate(*self.scene.tb.trans)
        GL.glMultMatrixf(self.scene.tb.matrix)

        self.drawSphere(p=self.env.robot_skeleton.bodynodes[0].com(), r=0.01)
        self.drawCube(p=np.zeros(3), r=0.1)

        if self.extraRenderFunc is not None:
            self.extraRenderFunc()

        a=0

    def drawCube(self, p, r=1, solid=True):
        GL.glPushMatrix()
        GL.glTranslated(p[0], p[1], p[2])
        #GL.glScale(r, r, r)
        if solid is True:
            GLUT.glutSolidCube(r)
        else:
            GLUT.glutWireCube(r)
        GL.glPopMatrix()

    def drawSphere(self, p, r, solid=True, slices=10):
        GL.glPushMatrix()
        GL.glTranslated(p[0], p[1], p[2])
        if solid is True:
            GLUT.glutSolidSphere(r, slices, slices)
        else:
            GLUT.glutWireSphere(r, slices, slices)
        GL.glPopMatrix()

    def mykeyboard(self, key, x, y):
        keycode = ord(key)
        #print(keycode)
        #key = key.decode('utf-8')
        # print("key = [%s] = [%d]" % (key, ord(key)))

        # n = sim.num_frames()
        if keycode == 27:
            self.close()
            return
        if keycode == 114: #'r'
            self.env.reset_model()
            return
        if keycode == 112: #'p'
            self.p_down = True
            return
        if keycode == 100: #'d'
            self.d_down = True
            return
        if keycode == 107: #'k'
            self.env.kinematicIK = not self.env.kinematicIK
            return
        self.keyPressed(key, x, y)

    def mykeyboardUp(self, key, x, y):
        keycode = ord(key)

        if keycode == 112: #'p'
            self.p_down = False
            return
        if keycode == 100: #'d'
            self.d_down = False
            return

    def mouseFunc(self, button, state, x, y):

        tb = self.scene.tb
        #print(button)
        if state == 0:  # Mouse pressed
            if button == 0:
                self.mouseLButton = True
            if button == 2:
                self.mouseRButton = True
            self.mouseLastPos = np.array([x, y])
            if button == 3:
                tb.trans[2] += 0.1
            elif button == 4:
                tb.trans[2] -= 0.1
        elif state == 1:
            #self.mouseLastPos = None
            if button == 0:
                self.mouseLButton = False
            if button == 2:
                self.mouseRButton = False

    def motionFunc(self, x, y):
        dx = x - self.mouseLastPos[0]
        dy = y - self.mouseLastPos[1]
        modifiers = GLUT.glutGetModifiers()
        tb = self.scene.tb

        ndofs = self.env.robot_skeleton.ndofs - 6
        if(self.p_down):
            # intercept to allow parameter tuning
            p = self.env.SPDController.Kp[0][0] + dx*0.1
            self.env.SPDController.Kp = np.diagflat([p] * ndofs)
            a=0
        elif self.d_down:
            # intercept to allow parameter tuning
            d = self.env.SPDController.Kd[0][0] + dx * 0.1
            self.env.SPDController.Kd = np.diagflat([d] * ndofs)
        else:
            #typical behavior
            if modifiers == GLUT.GLUT_ACTIVE_SHIFT:
                tb.zoom_to(dx, -dy)
            elif modifiers == GLUT.GLUT_ACTIVE_CTRL:
                tb.trans_to(dx, -dy)
            else:
                tb.drag_to(x, y, dx, -dy)
        self.mouseLastPos = np.array([x, y])