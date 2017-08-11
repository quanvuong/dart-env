# Contributors: Alexander Clegg (alexanderwclegg@gmail.com)

import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT
import sys
import numpy as np
from pydart2.gui.opengl.scene import OpenGLScene
from pydart2.gui.glut.window import *

from pyPhysX.clothHandles import *

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
        self.interactors = []
        [BaseInteractor(self), VertexSelectInteractor(self), FrameSelectInteractor(self), IKInteractor(self), PoseInteractor(self)] #the list of available interactors
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

        self.viewport = GL.glGetInteger(GL.GL_VIEWPORT)
        self.interactors = [BaseInteractor(self), VertexSelectInteractor(self), FrameSelectInteractor(self), IKInteractor(self), PoseInteractor(self)]  # the list of available interactors

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
        self.t_down = False
        self.x_down = False
        self.y_down = False
        self.z_down = False
        self.h_toggle = False #handle continuity toggle. When True, the handleNode will not auto destroy on mouse release
        self.selectedVertex = None
        self.selectedVertices = []
        self.handleNode = None

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
        if keycode == 99:  # 'c'
            self.selectedVertices = []
            return
        if keycode == 102: #'f'
            self.viewer.clothScene.renderClothFill = not self.viewer.clothScene.renderClothFill
            print("renderClothFill = " + str(self.viewer.clothScene.renderClothFill))
        if keycode == 104: #'h'
            self.h_toggle = not self.h_toggle
            if self.h_toggle is False:
                self.destroyHandleNode()
        if keycode == 112:  # 'p'
            #print relevant info
            print("Selected Vertex = " + str(self.selectedVertex))
            print("Selected Vertices = " + str(self.selectedVertices))
        if keycode == 116: #'t'
            self.t_down = True
        if keycode == 118: #'v'
            self.v_down = True
        if keycode == 119: #'w'
            self.viewer.clothScene.renderClothWires = not self.viewer.clothScene.renderClothWires
            print("renderClothWires = " + str(self.viewer.clothScene.renderClothWires))
        if keycode == 120:  # 'x'
            self.x_down = True
        if keycode == 121:  # 'y'
            self.y_down = True
        if keycode == 122:  # 'z'
            self.z_down = True
        self.viewer.keyPressed(key, x, y)

    def keyboardUp(self, key, x, y):
        keycode = ord(key)
        if keycode == 116: #'t'
            self.t_down = False
        if keycode == 118: #'v'
            self.v_down = False
        if keycode == 120:  # 'x'
            self.x_down = False
        if keycode == 121:  # 'y'
            self.y_down = False
        if keycode == 122:  # 'z'
            self.z_down = False

    def createHandleNodeFromSelected(self):
        #weed out already pinned verts to avoid double pinning
        newhandleverts = []
        for v in self.selectedVertices:
            if not self.viewer.clothScene.getPinned(cid=0, vid=v):
                newhandleverts.append(v)

        #also look at the currently selected vert
        if self.selectedVertex is not None:
            if not self.viewer.clothScene.getPinned(cid=0, vid=self.selectedVertex):
                newhandleverts.append(self.selectedVertex)

        if len(newhandleverts) > 0:
            #create a HandleNode from any valid selected verts
            self.handleNode = HandleNode(clothScene=self.viewer.clothScene)


            self.handleNode.addVertices(verts=newhandleverts)
            self.handleNode.setOrgToCentroid()

    def destroyHandleNode(self):
        #destroy the selected HandleNode
        if self.handleNode is not None:
            self.handleNode.clearHandles()
        self.handleNode = None

    def click(self, button, state, x, y):
        tb = self.viewer.scene.tb
        if state == 0:  # Mouse pressed
            if self.t_down or self.x_down or self.y_down or self.z_down:
                if self.handleNode is None:
                    self.createHandleNodeFromSelected()
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
                    if self.selectedVertex is not None:
                        print("selected vertex " + str(self.selectedVertex) + " pos = " + str(self.viewer.clothScene.getVertexPos(cid=0, vid=self.selectedVertex)))
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
                elif self.t_down is True:
                    self.handleNode.org = self.handleNode.org + self.viewer.camForward * 0.05
                    self.handleNode.updateHandles()
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
                    elif self.t_down is True:
                        self.handleNode.org = self.handleNode.org - self.viewer.camForward * 0.05
                        self.handleNode.updateHandles()
                    else:
                        self.selectedVertex = self.selectedVertex = self.viewer.clothScene.getNumVertices()-1
                else:
                    tb.trans[2] -= 0.1
        elif state == 1: #mouse released
            if self.h_toggle is False:
                self.destroyHandleNode()
            #self.mouseLastPos = None
            if button == 0:
                self.viewer.mouseLButton = False
            if button == 2:
                self.viewer.mouseRButton = False

    def drag(self, x, y):
        dx = x - self.viewer.mouseLastPos[0]
        dy = y - self.viewer.mouseLastPos[1]

        if self.v_down is True:
            a=0
        elif self.t_down is True:
            if self.viewer.mouseLButton is True and self.viewer.mouseRButton is True:
                a = 0
            elif self.viewer.mouseRButton is True:
                # compute a rotation about camUp and camRight axiis
                self.handleNode.applyAxisAngle(dx * (6.28 / 360.), self.viewer.camForward)
                self.handleNode.applyAxisAngle(dy * (6.28 / 360.), self.viewer.camRight)
                #self.handleNode.updateHandles()
            elif self.viewer.mouseLButton is True:
                # translate the frame org with camUp and camRight
                self.handleNode.org = self.handleNode.org + dx * self.viewer.camRight * 0.001
                self.handleNode.org = self.handleNode.org + dy * self.viewer.camUp * -0.001
                #self.handleNode.updateHandles()
        elif self.x_down is True:
            if self.viewer.mouseLButton is True and self.viewer.mouseRButton is True:
                a=0
            elif self.viewer.mouseLButton is True:
                rightX = np.array([1., 0, 0])
                if self.viewer.camRight.dot(rightX) < 0:
                    rightX *= -1
                self.handleNode.org = self.handleNode.org + dx * rightX * 0.001
                #self.handleNode.updateHandles()
            elif self.viewer.mouseRButton is True:
                #rotation about x
                self.handleNode.applyLocalAxisAngle(dy*(6.28/360.), np.array([1.,0,0]))
                #self.handleNode.updateHandles()
        elif self.y_down is True:
            if self.viewer.mouseLButton is True and self.viewer.mouseRButton is True:
                a=0
            elif self.viewer.mouseLButton is True:
                upY = np.array([0, 1., 0])
                if self.viewer.camUp.dot(upY) > 0:
                    upY *= -1
                self.handleNode.org = self.handleNode.org + dy * upY * 0.001
                #self.handleNode.updateHandles()
            elif self.viewer.mouseRButton is True:
                #rotation about y
                self.handleNode.applyLocalAxisAngle(dx * (6.28 / 360.), np.array([0., 1., 0]))
                #self.handleNode.updateHandles()
        elif self.z_down is True:
            if self.viewer.mouseLButton is True and self.viewer.mouseRButton is True:
                a=0
            elif self.viewer.mouseLButton is True:
                forwardZ = np.array([0, 0., 1.])
                if self.viewer.camForward.dot(forwardZ) > 0:
                    forwardZ *= -1
                self.handleNode.org = self.handleNode.org + dy * forwardZ * 0.001
                #self.handleNode.updateHandles()
            elif self.viewer.mouseRButton is True:
                # rotation about z
                self.handleNode.applyLocalAxisAngle(dx * (6.28 / 360.), np.array([0., 0., 1.]))
                #self.handleNode.updateHandles()
        else:
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

        if self.handleNode is not None:
            self.handleNode.updateHandles()
            self.handleNode.draw()
        if self.h_toggle is True:
            self.viewer.clothScene.drawText(x=self.viewer.viewport[2] / 2, y=self.viewer.viewport[3] - 60, text="Handle Node is toggled On", color=(0., 0, 0))

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

class IKInteractor(BaseInteractor):
    # base interactor defining class standard methods
    def __init__(self, viewer):
        self.viewer = viewer
        self.label = "Inverse Kinematics Select Interactor"
        self.f_down = False
        self.x_down = False
        self.y_down = False
        self.z_down = False
        self.s_down = False #s_down for handle select
        #inverse kinematics info
        self.ikTargets = []
        self.ikOffsets = []
        self.ikNodes = []
        self.selectedHandle = None
        self.skelix = 1 #change this if the skel file changes

    def key_down(self):
        #check for all function key_down booleans
        if self.f_down is True:
            return True
        if self.s_down is True:
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
            for i in range(len(self.ikNodes)):
                print("IK Handle " + str(i) + " | node " + str(self.ikNodes[i]) + " | offset " + str(self.ikOffsets[i]) + " | target " + str(self.ikTargets[i]))
        if keycode == 115:  # 's'
            self.s_down = True
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
        if keycode == 115:  # 's'
            self.s_down = False
        if keycode == 120:  # 'x'
            self.x_down = False
        if keycode == 121:  # 'y'
            self.y_down = False
        if keycode == 122:  # 'z'
            self.z_down = False

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
            if self.selectedHandle is not None:
                disp = dx * self.viewer.camRight * 0.001 + dy * self.viewer.camUp * -0.001
                if self.viewer.mouseLButton is True:
                    self.ikTargets[self.selectedHandle] += disp
                elif self.viewer.mouseRButton is True:
                    skel = self.viewer.sim.skeletons[self.skelix]
                    node = skel.bodynodes[self.Nodes[self.selectedHandle]]
                    self.ikOffsets[self.selectedHandle] = node.to_local(node.to_world(self.ikOffsets[self.selectedHandle]) + disp)
        elif (self.x_down):
            if self.selectedHandle is not None:
                self.ikOffsets[self.selectedHandle][0] += dx*0.001
        elif (self.y_down):
            if self.selectedHandle is not None:
                self.ikOffsets[self.selectedHandle][1] += dx*0.001
        elif (self.z_down):
            if self.selectedHandle is not None:
                self.ikOffsets[self.selectedHandle][2] += dx*0.001

        self.viewer.mouseLastPos = np.array([x, y])

    def click(self, button, state, x, y):
        tb = self.viewer.scene.tb
        if state == 0:  # Mouse pressed
            if button == 0:
                self.viewer.mouseLButton = True
                if self.s_down:
                    close = None
                    close_dist = 999
                    thresh = 0.5
                    skel = self.viewer.sim.skeletons[self.skelix]
                    for ix, n in enumerate(skel.bodynodes):
                        dist = np.linalg.norm(n.com()-self.viewer.prevWorldMouse)
                        if dist < close_dist and dist < thresh:
                            close_dist = dist
                            close = ix
                    if close is not None:
                        create = True
                        for ix,n in enumerate(self.ikNodes):
                            if n == close:
                                create = False
                                self.selectedHandle = ix
                                break
                        if create:
                            #create new handle
                            self.ikNodes.append(close)
                            self.ikOffsets.append(skel.bodynodes[close].to_local(self.viewer.prevWorldMouse))
                            self.ikTargets.append(self.viewer.prevWorldMouse)
                            self.selectedHandle = len(self.ikNodes)-1
            if button == 2:
                self.viewer.mouseRButton = True
                if self.s_down:
                    close = None
                    close_dist = 999
                    thresh = 0.5
                    skel = self.viewer.sim.skeletons[self.skelix]
                    for ix, n in enumerate(skel.bodynodes):
                        dist = np.linalg.norm(n.com() - self.viewer.prevWorldMouse)
                        if dist < close_dist and dist < thresh:
                            close_dist = dist
                            close = ix
                    if close is not None:
                        for ix,n in enumerate(self.ikNodes):
                            if n == close:
                                del self.ikTargets[ix]
                                del self.ikOffsets[ix]
                                del self.ikNodes[ix]
                            break
                        self.selectedHandle = None
            self.viewer.mouseLastPos = np.array([x, y])
            if button == 3:  # wheel up
                if self.f_down is True:
                    self.ikTargets[self.selectedHandle] += 0.01*self.viewer.camForward
                elif self.s_down is True:
                    if len(self.ikNodes) == 0:
                        self.selectedHandle = None
                    elif self.selectedHandle is None:
                        self.selectedHandle = 0
                    else:
                        self.selectedHandle += 1
                        if self.selectedHandle >= len(self.ikNodes):
                            self.selectedHandle = None
                else:
                    tb.trans[2] += 0.1
            elif button == 4:  # wheel down
                if self.f_down is True:
                    self.ikTargets[self.selectedHandle] -= 0.01 * self.viewer.camForward
                elif self.s_down is True:
                    if len(self.ikNodes) == 0:
                        self.selectedHandle = None
                    elif self.selectedHandle is None:
                        self.selectedHandle = len(self.ikNodes)-1
                    else:
                        self.selectedHandle -= 1
                        if self.selectedHandle < 0:
                            self.selectedHandle = None
                else:
                    tb.trans[2] -= 0.1
        elif state == 1:  # mouse released
            # self.mouseLastPos = None
            if button == 0:
                self.viewer.mouseLButton = False
            if button == 2:
                self.viewer.mouseRButton = False

    def contextRender(self):
        # place any extra rendering for this context here
        for i in range(len(self.ikNodes)):
            skel = self.viewer.sim.skeletons[self.skelix]
            handlepos = skel.bodynodes[self.ikNodes[i]].to_world(self.ikOffsets[i])
            GL.glColor3d(0, 1, 1)
            self.viewer.drawSphere(p=handlepos, r=0.025, solid=True)
            GL.glColor3d(1, 0, 1)
            self.viewer.drawSphere(p=self.ikTargets[i], r=0.025, solid=True)
            GL.glBegin(GL.GL_LINES)
            GL.glVertex3d(self.ikTargets[i][0], self.ikTargets[i][1], self.ikTargets[i][2])
            GL.glVertex3d(handlepos[0], handlepos[1], handlepos[2])
            GL.glEnd()


        #draw selected handle info
        if self.selectedHandle is None:
            self.viewer.clothScene.drawText(x=self.viewer.viewport[2] / 2, y=self.viewer.viewport[3] - 45, text="Selected Handle = " + str(self.selectedHandle), color=(0., 0, 0))
        else:
            skel = self.viewer.sim.skeletons[self.skelix]
            node_name = skel.bodynodes[self.ikNodes[self.selectedHandle]].name
            self.viewer.clothScene.drawText(x=self.viewer.viewport[2] / 2, y=self.viewer.viewport[3] - 45,
                                            text="Selected Handle = " + str(self.selectedHandle) + " | " + str(node_name) + " | offset= " + str(self.ikOffsets[self.selectedHandle]) + " | target = " + str(self.ikTargets[self.selectedHandle]), color=(0., 0, 0))
        a = 0

class PoseInteractor(BaseInteractor):
    # interactor defining GUI influence over skeleton DOFS by clicking displayed DOF boxes
    def __init__(self, viewer):
        self.viewer = viewer
        self.label = "Pose DOF Interactor"
        self.boxRanges = [] #list of np array pairs: [[minx, maxx],[miny, maxy]]
        self.skelix = 1  # change this if the skel file changes
        self.selectedBox = None
        self.boxesDefined = False

    def defineBoxRanges(self):
        #manually defined
        #These values are taken from pyPhysX.renderUtils.renderDofs()
        skel = self.viewer.sim.skeletons[self.skelix]
        topLeft = np.array([2., self.viewer.viewport[3] - 10])
        textWidth = 165.  # pixel width of the text
        numWidth = 50.  # pixel width of the lower/upper bounds text
        barWidth = 90.
        barHeight = -15.
        barSpace = -10.
        #print("topLeft " + str(topLeft))
        for i in range(len(skel.q)):
            self.boxRanges.append([[topLeft[0] + textWidth + numWidth, topLeft[0] + textWidth + numWidth + barWidth], [topLeft[1] + (barHeight + barSpace) * i + barHeight, topLeft[1] + (barHeight + barSpace) * i]])
        #print(self.boxRanges)

    def boxClickTest(self, point):
        if not self.boxesDefined:
            self.defineBoxRanges()
            self.boxesDefined = True
        #return None if nothing hit, or the box/dof index of the clicked box
        point[1] = self.viewer.viewport[3] - point[1]
        for ix,b in enumerate(self.boxRanges):
            if point[0] < b[0][1] and point[0] > b[0][0]:
                if point[1] < b[1][1] and point[1] > b[1][0]:
                    #print("clicked box " + str(ix))
                    return ix
        return None

    def incrementSelectedDOF(self, inc):
        #increment a selected dof by inc respecting joint limits
        if self.selectedBox is not None:
            skel = self.viewer.sim.skeletons[self.skelix]
            qpos = skel.q
            qpos[self.selectedBox] += inc
            qpos[self.selectedBox] = min(max(qpos[self.selectedBox], skel.position_lower_limits()[self.selectedBox]), skel.position_upper_limits()[self.selectedBox])
            skel.set_positions(qpos)

    def click(self, button, state, x, y):
        tb = self.viewer.scene.tb
        if state == 0:  # Mouse pressed
            self.selectedBox = self.boxClickTest(np.array([x,y]))
            if button == 0:
                self.viewer.mouseLButton = True
            if button == 2:
                self.viewer.mouseRButton = True
            self.viewer.mouseLastPos = np.array([x, y])
            if button == 3:  # wheel up
                if self.selectedBox is not None:
                    self.incrementSelectedDOF(0.05)
                else:
                    tb.trans[2] += 0.1
            elif button == 4:  # wheel down
                if self.selectedBox is not None:
                    self.incrementSelectedDOF(-0.05)
                else:
                    tb.trans[2] -= 0.1
        elif state == 1:  # mouse released
            self.selectedBox = None
            # self.mouseLastPos = None
            if button == 0:
                self.viewer.mouseLButton = False
            if button == 2:
                self.viewer.mouseRButton = False

    def drag(self, x, y):
        dx = x - self.viewer.mouseLastPos[0]
        dy = y - self.viewer.mouseLastPos[1]

        if self.selectedBox is None:
            tb = self.viewer.scene.tb
            if self.viewer.mouseLButton is True and self.viewer.mouseRButton is True:
                tb.zoom_to(dx, -dy)
            elif self.viewer.mouseRButton is True:
                tb.trans_to(dx, -dy)
            elif self.viewer.mouseLButton is True:
                tb.drag_to(x, y, dx, -dy)
        else:
            skel = self.viewer.sim.skeletons[self.skelix]
            dofRange = skel.position_upper_limits()[self.selectedBox]-skel.position_lower_limits()[self.selectedBox]
            self.incrementSelectedDOF(dx*((dofRange)/(self.boxRanges[self.selectedBox][0][1]-self.boxRanges[self.selectedBox][0][0])))
        self.viewer.mouseLastPos = np.array([x, y])

    def keyboard(self, key, x, y):
        keycode = ord(key)
        if keycode == 27:
            self.viewer.close()
            return
        if keycode == 13:  # ENTER
            if self.viewer.inputFunc is not None:
                self.viewer.inputFunc()
            return
        if keycode == 112:  # 'p'
            # print relevant info
            skel = self.viewer.sim.skeletons[self.skelix]
            dofstr = "["
            for ix,dof in enumerate(skel.q):
                dofstr = dofstr + str(dof)
                if ix < len(skel.q)-1:
                    dofstr = dofstr + ", "
            dofstr = dofstr + "]"
            print(dofstr)
            return
        self.viewer.keyPressed(key, x, y)

    def contextRender(self):
        if self.selectedBox is not None:
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glPushMatrix()
            GL.glLoadIdentity()
            GL.glOrtho(0, self.viewer.viewport[2], 0, self.viewer.viewport[3], -1, 1)
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glPushMatrix()
            GL.glLoadIdentity()
            GL.glDisable(GL.GL_CULL_FACE)

            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
            GL.glColor3d(3., 3., 0)
            GL.glBegin(GL.GL_QUADS)
            b = self.boxRanges[self.selectedBox]
            #for b in self.boxRanges:
            GL.glVertex2d(b[0][0]-1, b[1][0]-1)
            GL.glVertex2d(b[0][0]-1, b[1][1]+1)
            GL.glVertex2d(b[0][1]+1, b[1][1]+1)
            GL.glVertex2d(b[0][1]+1, b[1][0]-1)
            GL.glEnd()
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glPopMatrix()
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glPopMatrix()

