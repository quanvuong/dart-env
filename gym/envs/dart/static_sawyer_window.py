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
    def __init__(self, sim, title, env):
        self.env = env
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