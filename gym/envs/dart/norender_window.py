import numpy as np

class NoRenderWindow(object):
    #This class replaces static_window.py: StaticGLUTWindow in order to bypass all openGL

    def __init__(self, sim, title):
        self.title = title
        self.sim = sim
        self.is_simulating = True
        self.scene = None

    def runSingleStep(self):
        if self.sim is None:
            return
        if self.is_simulating:
            self.sim.step()

    def run(self, _width=None, _height=None, _show_window=False):
        #initialization should happen here
        print("NoRenderWindow initialized")

    def close(self):
        print("closed?")