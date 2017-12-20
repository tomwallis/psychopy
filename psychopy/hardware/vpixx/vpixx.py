import numpy as np

from __future__ import absolute_import

from psychopy import logging, core, visual

DEBUG = True

from pypixxlib.viewpixx import *
from pypixxlib._libdpx import *

class ViewPixx(object):

    def __init__(self,
                 win,
                 mode='C24',
                 gammaCorrect = None,
                 gamma = None,
                 clut = None):
        """
        :Parameters:

            mode : 'C24' (or 'M16' or 'C48')
            
            gammaCorrect: 'simpleGamma' (or 'clut' or 'None')
            
            gamma: Double
                Gamma value used if gammaCorrect=='simpleGamma'.
                Can be inherited from the window.
            
            clut: 256x3 Double array ranging from -1 to 1
                Used if gammaCorrect=='clut'.

        """

        # Open monitor.
        if DPxIsViewpixx():
            if DPxIsViewpixx3D():
                self.mon = ViewPixx3D()
                self.isViewpixx3D = True
            else:
                self.mon = ViewPixx()
                self.isViewpixx3D = False
            self.initialised = True
            logging.debug("Found and initialised Viewpixx")
        else:
            self.initialised = False
            logging.warning("Couldn't initialise Viewpixx")

        self.win = win
        self.mode = mode
        
        self.mon = ViewPixx()
        self.setVideoMode(mode)
        
        self.setGammaCorrect(gammaCorrect, True)
        
#        self._setupShaders()
#        # replace window methods with our custom ones
#        self.win._prepareFBOrender = self._prepareFBOrender
#        self.win._finishFBOrender = self._finishFBOrender
#        self.win._afterFBOrender = self._afterFBOrender
            
    def setGammaCorrect(self, gammaCorrect, applyGammaCorrect=True):
        if gammaCorrect == 'simpleGamma':
            self.gammaCorrect = 'simpleGamma'
            if clut != None:
                logging.warning("CLUT given, but not used.")
            if gamma is None:
                # inherit from window:
                self.gamma = win.gamma
                if self.gamma == None:
                    logging.warning("Trying to use gamma for gamma correction. However, no gamma given. Therefore no gamma correction will be applied.")
                    self.gammaCorrect == None
            elif len(gamma) > 2:
                # [R,G,B] or [Lum,R,G,B]. In both cases take last three values.
                self.gamma = gamma[-3:]
            else:
                self.gamma = [gamma, gamma, gamma]
        elif gammaCorrect == 'clut':
            self.gammaCorrect = 'clut'
            if gamma != None:
                logging.warning("Gamma given, but not used.")
            if clut == None:
                logging.warning("Trying to use CLUT for gamma correction. However, no CLUT given. Therefore no gamma correction will be applied.")
                self.gammaCorrect == None
            elif clut:
                self.clut = clut;
        else:
            if gammaCorrect != 'None' or gammaCorrect != None:
                logging.warning("No valid gamma correction mode given. Applying no gamma correction at all.")
            if gamma != None:
                logging.warning("Gamma given, but not used.")
            if clut != None:
                logging.warning("CLUT given, but not used.")
            self.gammaCorrect == None
        if applyGammaCorrect:
            self.applyGammaCorrect()
            
    def applyGammaCorrect():
        return
        
    def setGamma(self, newGamma):
        self.gamma = newGamma
        self.gammaCorrect = 'simpleGamma'
        self.applyGammaCorrect()
        
    def setVideoMode(self, videoMode):
        if videoMode == 'C24':
            self.mode = 'C24'
            self.mon.setVideoMode(self.mode)
        elif videoMode == 'M16':
            self.mode = 'M16'
            self.mon.setVideoMode(self.mode)
        elif videoMode == 'C48':
            self.mode = 'C48'
            self.mon.setVideoMode('C48')
        else:
            logging.warning('Video mode was not changed! The given video mode has not been implemented yet.')
            
    def __exit__(self):
        mon.close()
