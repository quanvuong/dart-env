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
    def __init__(self, sim, title, clothScene=None, extraRenderFunc=None):
        super(StaticClothGLUTWindow,self).__init__(sim, title)
        self.clothScene = clothScene
        self.extraRenderFunc = extraRenderFunc
        
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
        self.keyPressed(key, x, y)
