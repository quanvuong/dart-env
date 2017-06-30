# Contributors: Alexander Clegg (alexanderwclegg@gmail.com)

import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT
import sys
import numpy as np
from pydart2.gui.opengl.scene import OpenGLScene
from pydart2.gui.glut.window import *

from gym.envs.dart.static_window import *


class StaticClothGLUTWindow(StaticGLUTWindow):
    def __init__(self, sim, title, clothScene=None, extraRenderFunc=None, inputFunc=None):
        super(StaticClothGLUTWindow,self).__init__(sim, title)
        self.clothScene = clothScene
        self.extraRenderFunc = extraRenderFunc
        self.inputFunc = inputFunc
        self.mouseLButton = False
        self.mouseRButton = False
        
    def extraRender(self):
        'Place any extra rendering functionality here. This can be used to extend the window class'
        #get the window camera transformations
        GL.glLoadIdentity()
        GL.glTranslate(*self.scene.tb.trans)
        GL.glMultMatrixf(self.scene.tb.matrix)
        #GLUT.glutSolidSphere(0.05,10,10) #testing origin location
        if self.clothScene is not None:
            #print("render cloth")
            self.clothScene.render()
        if self.extraRenderFunc is not None:
            self.extraRenderFunc()

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
            self.mouseLastPos = None
            if button == 0:
                self.mouseLButton = False
            if button == 2:
                self.mouseRButton = False


    def motionFunc(self, x, y):
        dx = x - self.mouseLastPos[0]
        dy = y - self.mouseLastPos[1]
        modifiers = GLUT.glutGetModifiers()
        tb = self.scene.tb
        #print(tb.trans)
        # print("mouse motion: " + str(dx))
        #print("SHIFT?" + str(modifiers))
        if self.mouseLButton is True and self.mouseRButton is True:
            tb.zoom_to(dx, -dy)
        elif self.mouseRButton is True:
            tb.trans_to(dx, -dy)
        elif self.mouseLButton is True:
            tb.drag_to(x, y, dx, -dy)
        self.mouseLastPos = np.array([x, y])
            
            
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
            self.clothScene.reset()
            self.sim.reset_model()
            return
        if keycode == 13: #ENTER
            if self.inputFunc is not None:
                self.inputFunc()
        self.keyPressed(key, x, y)
