# Contributors: Alexander Clegg (alexanderwclegg@gmail.com)

import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT
import sys
import numpy as np
from pydart2.gui.opengl.scene import OpenGLScene
from pydart2.gui.glut.window import *

from gym.envs.dart.static_window import *

import pyPhysX.pyutils as pyutils

class StaticClothGLUTWindow(StaticGLUTWindow):
    def __init__(self, sim, title, clothScene=None, extraRenderFunc=None, inputFunc=None):
        super(StaticClothGLUTWindow,self).__init__(sim, title)
        self.clothScene = clothScene
        self.extraRenderFunc = extraRenderFunc
        self.inputFunc = inputFunc
        self.mouseLButton = False
        self.mouseRButton = False
        #store the most recent rendering matrices
        self.modelviewM = None
        self.projectionM = None
        self.viewport = None
        self.prevWorldMouse = np.array([0.,0,0])
        self.camForward = np.array([0.,0,1])
        self.camUp = np.array([0.,1.0,0])
        self.camRight = np.array([1.,0.0,0])
        self.mouseLastPos = np.array([0,0])
        self.curInteractorIX = None #set an interactor class to change user interaction with the window
        self.interactors = [BaseInteractor(self), VertexSelectInteractor(self), FrameSelectInteractor(self)] #the list of available interactors
        #self.interactors.append(BaseInteractor(self))
        #self.interactors.append(VertexSelectInteractor(self))
        self.lastContextSwitch = 0 #holds the frame of the last context switch (for label rendering)

    def run(self, _width=None, _height=None, _show_window=True ):
        # Init glut
        self._show_window = _show_window
        GLUT.glutInit(())
        GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA |
                                 GLUT.GLUT_DOUBLE |
                                 GLUT.GLUT_ALPHA |
                                 GLUT.GLUT_DEPTH)
        if _width is not None and _height is not None:
            GLUT.glutInitWindowSize(_width, _height)
            # self.resizeGL(_width, _height) # this line crashes my program ??
        else:
            GLUT.glutInitWindowSize(*self.window_size)
        GLUT.glutInitWindowPosition(0, 0)
        self.window = GLUT.glutCreateWindow(self.title)
        if not _show_window:
            GLUT.glutHideWindow()

        GLUT.glutDisplayFunc(self.drawGL)
        GLUT.glutReshapeFunc(self.resizeGL)
        GLUT.glutKeyboardFunc(self.mykeyboard)
        GLUT.glutKeyboardUpFunc(self.keyboardUp)
        GLUT.glutMouseFunc(self.mouseFunc)
        GLUT.glutMotionFunc(self.motionFunc)
        GLUT.glutPassiveMotionFunc(self.passiveMotionFunc)
        self.initGL(*self.window_size)

        
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

        #grab the current rendering params
        self.modelviewM = GL.glGetDoublev(GL.GL_MODELVIEW_MATRIX)
        self.projectionM = GL.glGetDoublev(GL.GL_PROJECTION_MATRIX)
        self.viewport = GL.glGetInteger(GL.GL_VIEWPORT)

        #unprojections:
        #mouse hover object
        z = GL.glReadPixels(self.mouseLastPos[0],self.viewport[3]-self.mouseLastPos[1], 1, 1, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT)
        self.prevWorldMouse = np.array(self.unproject(np.array([self.mouseLastPos[0],self.viewport[3]-self.mouseLastPos[1], z])))

        #camera forward
        z = GL.glReadPixels(self.viewport[2]/2, self.viewport[3]/2, 1, 1, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT)
        forwardNear = np.array(self.unproject(np.array([self.viewport[2]/2, self.viewport[3]/2, 0])))
        forwardFar = np.array(self.unproject(np.array([self.viewport[2]/2, self.viewport[3]/2, z])))
        self.camForward = forwardFar-forwardNear
        self.camForward = self.camForward / np.linalg.norm(self.camForward)
        #camera right
        rightNear = np.array(self.unproject(np.array([self.viewport[2] - 1, self.viewport[3] / 2, 0])))
        self.camRight = rightNear - forwardNear
        self.camRight = self.camRight / np.linalg.norm(self.camRight)
        #camera up
        upNear = np.array(self.unproject(np.array([self.viewport[2]/2, self.viewport[3]-1, 0])))
        self.camUp = upNear - forwardNear
        self.camUp = self.camUp / np.linalg.norm(self.camUp)

        #self.clothScene.drawText(x=self.viewport[2] / 2, y=self.viewport[3] - 60, text="Forward = " + str(self.camForward), color=(0., 0, 0))
        #self.clothScene.drawText(x=self.viewport[2] / 2, y=self.viewport[3] - 75, text="Up = " + str(self.camUp), color=(0., 0, 0))
        #self.clothScene.drawText(x=self.viewport[2] / 2, y=self.viewport[3] - 90, text="Right = " + str(self.camRight), color=(0., 0, 0))

        #self.drawLine(p0=self.camForward)
        #self.drawLine(p0=self.camUp)
        #self.drawLine(p0=self.camRight)


        if self.lastContextSwitch < 100:
            context = "Default Context"
            if self.curInteractorIX is not None:
                context = self.interactors[self.curInteractorIX].label
            self.clothScene.drawText(x=self.viewport[2]/2, y=self.viewport[3]-30, text="Active Context = " + str(context), color=(0., 0, 0))

        if self.extraRenderFunc is not None:
            self.extraRenderFunc()

        if self.curInteractorIX is not None:
            self.interactors[self.curInteractorIX].contextRender()

        self.lastContextSwitch += 1

    def mouseFunc(self, button, state, x, y):
        if self.curInteractorIX is not None:
            self.interactors[self.curInteractorIX].click(button, state, x, y)
            return

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
        if self.curInteractorIX is not None:
            self.interactors[self.curInteractorIX].drag(x, y)
            return

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

    def passiveMotionFunc(self, x, y):
        if self.curInteractorIX is not None:
            self.interactors[self.curInteractorIX].passiveMotion(x, y)
            return

        self.mouseLastPos = np.array([x, y])
        #print("self.mouseLastPos = " + str(self.mouseLastPos))
            
            
    def mykeyboard(self, key, x, y):
        #regardless of the interactor conext, 'm' always switches contexts
        keycode = ord(key)
        if keycode == 109:
            self.switchInteractorContext()
            return
        if keycode == 27:
            self.close()
            return
        if keycode == 32: #space bar
            self.is_simulating = not self.is_simulating
            return
        #if an interactor context is defined, pass control to it
        if self.curInteractorIX is not None:
            self.interactors[self.curInteractorIX].keyboard(key, x, y)
            return

        #if no interactor context, do the following
        if keycode == 114: #'r'
            self.clothScene.reset()
            self.sim.reset_model()
            return
        if keycode == 13: #ENTER
            if self.inputFunc is not None:
                self.inputFunc()
        self.keyPressed(key, x, y)

    def keyboardUp(self, key, x, y):
        if self.curInteractorIX is not None:
            self.interactors[self.curInteractorIX].keyboardUp(key, x, y)
            return

        keycode = ord(key)

    def unproject(self, winp=np.array([0.,0.,0.])):
        #unproject the given input window space cords and return an object space vector
        obj = GLU.gluUnProject(winp[0], winp[1], winp[2], self.modelviewM, self.projectionM, self.viewport)
        return obj

    def drawSphere(self, p, r, solid=True, slices=10):
        GL.glPushMatrix()
        GL.glTranslated(p[0], p[1], p[2])
        if solid is True:
            GLUT.glutSolidSphere(r, slices, slices)
        else:
            GLUT.glutWireSphere(r, slices, slices)
        GL.glPopMatrix()

    def drawLine(self, p0=np.array([0.,0,0]), p1=np.array([0.,0,0])):
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3d(p0[0],p0[1],p0[2])
        GL.glVertex3d(p1[0],p1[1],p1[2])
        GL.glEnd()

    def switchInteractorContext(self, ix=-1):
        #switch to the interactor in the list with ix or next one if ix=-1
        self.lastContextSwitch = 0
        if len(self.interactors) == 0:
            self.curInteractorIX = None
            return
        if ix == -1:
            if self.curInteractorIX is None:
                self.curInteractorIX = 0
            else:
                self.curInteractorIX = self.curInteractorIX + 1
                if self.curInteractorIX >= len(self.interactors):
                    self.curInteractorIX = None
        elif ix >= len(self.interactors):
            self.curInteractorIX = None
        else:
            self.curInteractorIX = ix

#Define interactor objects
#Each Interactor defines a set of methods which allow the user input (keyboard, mouse, etc) to change the scene
class BaseInteractor(object):
    #base interactor defining class standard methods
    def __init__(self, viewer):
        self.viewer = viewer
        self.label = "Base Interactor"

    def keyboard(self, key, x, y):
        keycode = ord(key)
        print("key down = " + str(keycode))
        if keycode == 27:
            self.viewer.close()
            return
        if keycode == 13: #ENTER
            if self.viewer.inputFunc is not None:
                self.viewer.inputFunc()
        self.viewer.keyPressed(key, x, y)

    def keyboardUp(self, key, x, y):
        keycode = ord(key)
        print("key up = " + str(keycode))

    def click(self, button, state, x, y):
        statetext = "down"
        if state == 1:
            statetext = "up"
        print("Clicked " + str(button) + " " + str(statetext) + " at ["+str(x)+","+str(y)+"]")
        if state == 0:  # Mouse pressed
            if button == 0:
                self.viewer.mouseLButton = True
            if button == 2:
                self.viewer.mouseRButton = True
            self.mouseLastPos = np.array([x, y])
            if button == 3: #mouse wheel up
                a=0
            elif button == 4: #mouse wheel down
                a=0
        elif state == 1: #mouse released
            #self.mouseLastPos = None
            if button == 0:
                self.viewer.mouseLButton = False
            if button == 2:
                self.viewer.mouseRButton = False

    def passiveMotion(self, x, y):
        self.viewer.mouseLastPos = np.array([x, y])

    def drag(self, x, y):
        dx = x - self.mouseLastPos[0]
        dy = y - self.mouseLastPos[1]
        self.viewer.mouseLastPos = np.array([x, y])

    def contextRender(self):
        #place any extra rendering for this context here
        a=0

class VertexSelectInteractor(BaseInteractor):
    #vertrex selector interactor unprojection and vertex selection
    def __init__(self, viewer):
        self.viewer = viewer
        self.label = "Vertex Select Interactor"
        self.v_down = False
        self.selectedVertex = None
        self.selectedVertices = []

    def keyboard(self, key, x, y):
        keycode = ord(key)

        if keycode == 27:
            self.viewer.close()
            return
        if keycode == 13: #ENTER
            if self.viewer.inputFunc is not None:
                self.viewer.inputFunc()
        if keycode == 97: #'a'
            if self.selectedVertex is not None:
                if self.selectedVertex not in self.selectedVertices:
                    self.selectedVertices.append(self.selectedVertex)
            return
        if keycode == 102: #'f'
            self.viewer.clothScene.renderClothFill = not self.viewer.clothScene.renderClothFill
            print("renderClothFill = " + str(self.viewer.clothScene.renderClothFill))
        if keycode == 112:  # 'p'
            #print relevant info
            print("Selected Vertex = " + str(self.selectedVertex))
            print("Selected Vertices = " + str(self.selectedVertices))
        if keycode == 118: #'v'
            self.v_down = True
        if keycode == 119: #'w'
            self.viewer.clothScene.renderClothWires = not self.viewer.clothScene.renderClothWires
            print("renderClothWires = " + str(self.viewer.clothScene.renderClothWires))
        self.viewer.keyPressed(key, x, y)

    def keyboardUp(self, key, x, y):
        keycode = ord(key)
        if keycode == 118: #'v'
            self.v_down = False

    def click(self, button, state, x, y):
        tb = self.viewer.scene.tb
        if state == 0:  # Mouse pressed
            if button == 0:
                self.viewer.mouseLButton = True
                if self.v_down is True:
                    self.selectedVertex = self.viewer.clothScene.getCloseVertex(self.viewer.prevWorldMouse)
                    if self.selectedVertex not in self.selectedVertices:
                        self.selectedVertices.append(self.selectedVertex)
            if button == 2:
                self.viewer.mouseRButton = True
                if self.v_down is True:
                    self.selectedVertex = self.viewer.clothScene.getCloseVertex(self.viewer.prevWorldMouse)
            self.viewer.mouseLastPos = np.array([x, y])
            if button == 3: #wheel up
                if self.v_down is True:
                    if self.selectedVertex is not None:
                        self.selectedVertex += 1
                        if self.selectedVertex >= self.viewer.clothScene.getNumVertices():
                            self.selectedVertex = 0
                        elif self.selectedVertex < 0:
                            self.selectedVertex = self.viewer.clothScene.getNumVertices()-1
                    else:
                        self.selectedVertex = 0
                else:
                    tb.trans[2] += 0.1
            elif button == 4: #wheel down
                if self.v_down is True:
                    if self.selectedVertex is not None:
                        self.selectedVertex -= 1
                        if self.selectedVertex >= self.viewer.clothScene.getNumVertices():
                            self.selectedVertex = 0
                        elif self.selectedVertex < 0:
                            self.selectedVertex = self.viewer.clothScene.getNumVertices()-1
                    else:
                        self.selectedVertex = self.selectedVertex = self.viewer.clothScene.getNumVertices()-1
                else:
                    tb.trans[2] -= 0.1
        elif state == 1: #mouse released
            #self.mouseLastPos = None
            if button == 0:
                self.viewer.mouseLButton = False
            if button == 2:
                self.viewer.mouseRButton = False

    def drag(self, x, y):
        dx = x - self.viewer.mouseLastPos[0]
        dy = y - self.viewer.mouseLastPos[1]

        if self.v_down is False:
            tb = self.viewer.scene.tb
            if self.viewer.mouseLButton is True and self.viewer.mouseRButton is True:
                tb.zoom_to(dx, -dy)
            elif self.viewer.mouseRButton is True:
                tb.trans_to(dx, -dy)
            elif self.viewer.mouseLButton is True:
                tb.drag_to(x, y, dx, -dy)

        self.viewer.mouseLastPos = np.array([x, y])

    def contextRender(self):
        #place any extra rendering for this context here
        if self.v_down is True:
            GL.glColor3d(1,0,0)
        self.viewer.drawSphere(p=self.viewer.prevWorldMouse, r=0.01, solid=True)

        GL.glColor3d(0, 1, 0)
        for v in self.selectedVertices:
            #print("v = " + str(v))
            vx = self.viewer.clothScene.getVertexPos(vid=v)
            self.viewer.drawSphere(p=vx, r=0.009, solid=True)

        selectedVertexText = "None"
        if self.selectedVertex is not None:
            vx = self.viewer.clothScene.getVertexPos(vid=self.selectedVertex)
            GL.glColor3d(1, 1, 0)
            self.viewer.drawSphere(p=vx, r=0.01, solid=True)
            selectedVertexText = str(self.selectedVertex)
        self.viewer.clothScene.drawText(x=self.viewer.viewport[2] / 2, y=self.viewer.viewport[3] - 45, text="Selected Vertex = " + str(selectedVertexText), color=(0., 0, 0))
        a=0

class FrameSelectInteractor(BaseInteractor):
    # base interactor defining class standard methods
    def __init__(self, viewer):
        self.viewer = viewer
        self.label = "Frame Select Interactor"
        self.f_down = False
        self.x_down = False
        self.y_down = False
        self.z_down = False
        self.frame = pyutils.ShapeFrame(org=np.array([0.,0,-1.75]))

    def key_down(self):
        #check for all function key_down booleans
        if self.f_down is True:
            return True
        if self.x_down is True:
            return True
        if self.y_down is True:
            return True
        if self.z_down is True:
            return True
        return False

    def keyboard(self, key, x, y):
        keycode = ord(key)
        if keycode == 27:
            self.viewer.close()
            return
        if keycode == 13:  # ENTER
            if self.viewer.inputFunc is not None:
                self.viewer.inputFunc()
        if keycode == 102:  # 'f'
            self.f_down = True
        if keycode == 112:  # 'p'
            # print relevant info
            print("Frame T = " + str(self.frame.org))
            print("Frame Q = " + str(self.frame.quat))
            print("Frame R = " + str(self.frame.orientation))
        if keycode == 120:  # 'x'
            self.x_down = True
        if keycode == 121:  # 'y'
            self.y_down = True
        if keycode == 122:  # 'z'
            self.z_down = True
        self.viewer.keyPressed(key, x, y)

    def keyboardUp(self, key, x, y):
        keycode = ord(key)
        if keycode == 102:  # 'f'
            self.f_down = False
        if keycode == 120:  # 'x'
            self.x_down = False
        if keycode == 121:  # 'y'
            self.y_down = False
        if keycode == 122:  # 'z'
            self.z_down = False

    def click(self, button, state, x, y):
        tb = self.viewer.scene.tb
        if state == 0:  # Mouse pressed
            if button == 0:
                self.viewer.mouseLButton = True
            if button == 2:
                self.viewer.mouseRButton = True
            self.viewer.mouseLastPos = np.array([x, y])
            if button == 3:  # wheel up
                if self.f_down is True:
                    self.frame.org = self.frame.org + self.viewer.camForward * 0.1
                else:
                    tb.trans[2] += 0.1
            elif button == 4:  # wheel down
                if self.f_down is True:
                    self.frame.org = self.frame.org + self.viewer.camForward * -0.1
                else:
                    tb.trans[2] -= 0.1
        elif state == 1:  # mouse released
            # self.mouseLastPos = None
            if button == 0:
                self.viewer.mouseLButton = False
            if button == 2:
                self.viewer.mouseRButton = False

    def drag(self, x, y):
        dx = x - self.viewer.mouseLastPos[0]
        dy = y - self.viewer.mouseLastPos[1]

        if self.key_down() is False:
            tb = self.viewer.scene.tb
            if self.viewer.mouseLButton is True and self.viewer.mouseRButton is True:
                tb.zoom_to(dx, -dy)
            elif self.viewer.mouseRButton is True:
                tb.trans_to(dx, -dy)
            elif self.viewer.mouseLButton is True:
                tb.drag_to(x, y, dx, -dy)
        elif (self.f_down):
            if self.viewer.mouseLButton is True and self.viewer.mouseRButton is True:
                a=0
            elif self.viewer.mouseRButton is True:
                #compute a rotation about camUp and camRight axiis
                self.frame.applyAxisAngle(dx * (6.28 / 360.), self.viewer.camForward)
                self.frame.applyAxisAngle(dy * (6.28 / 360.), self.viewer.camRight)
                a=0
            elif self.viewer.mouseLButton is True:
                #translate the frame org with camUp and camRight
                self.frame.org = self.frame.org + dx * self.viewer.camRight * 0.001
                self.frame.org = self.frame.org + dy * self.viewer.camUp * -0.001
        elif (self.x_down):
            if self.viewer.mouseLButton is True and self.viewer.mouseRButton is True:
                a=0
            elif self.viewer.mouseLButton is True:
                rightX = np.array([1., 0, 0])
                if self.viewer.camRight.dot(rightX) < 0:
                    rightX *= -1
                self.frame.org = self.frame.org + dx * rightX * 0.001
            elif self.viewer.mouseRButton is True:
                #rotation about x
                self.frame.applyLocalAxisAngle(dy*(6.28/360.), np.array([1.,0,0]))
        elif (self.y_down):
            if self.viewer.mouseLButton is True and self.viewer.mouseRButton is True:
                a=0
            elif self.viewer.mouseLButton is True:
                upY = np.array([0, 1., 0])
                if self.viewer.camUp.dot(upY) > 0:
                    upY *= -1
                self.frame.org = self.frame.org + dy * upY * 0.001
            elif self.viewer.mouseRButton is True:
                #rotation about y
                self.frame.applyLocalAxisAngle(dx * (6.28 / 360.), np.array([0., 1., 0]))
        elif (self.z_down):
            if self.viewer.mouseLButton is True and self.viewer.mouseRButton is True:
                a=0
            elif self.viewer.mouseLButton is True:
                forwardZ = np.array([0, 0., 1.])
                if self.viewer.camForward.dot(forwardZ) > 0:
                    forwardZ *= -1
                self.frame.org = self.frame.org + dy * forwardZ * 0.001
            elif self.viewer.mouseRButton is True:
                # rotation about z
                self.frame.applyLocalAxisAngle(dx * (6.28 / 360.), np.array([0., 0., 1.]))

        self.viewer.mouseLastPos = np.array([x, y])

    def contextRender(self):
        # place any extra rendering for this context here
        GL.glColor3d(0, 1, 0)
        if self.f_down is True:
            GL.glColor3d(1, 0, 0)
        self.frame.draw()

        a = 0