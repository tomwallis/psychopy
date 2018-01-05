# Author: Jonathan Oesterle

from psychopy import logging
import numpy as np

class VP(object):
    """A class to define a virtual photometer that returns random values.

    usage::

        from psychopy.hardware import virtualPhotometer
        phot = virtualPhotometer.Photometer()
        if phot.OK:
            print(phot.getLum())

    """

    def __init__(self):
        self.lastLum = None
        self.OK = True
        self.type = 'virtual'

    def measure(self):
        """Generate random luminance and set .lastLum to this value.
        """
        lum = np.random.uniform(0, 250)
        lastLum = lum
        return lum
        
    def measure_n(self, n):
        """Generate random luminances and set .lastLum to last value.
        """
        n = int(np.floor(n))
        lums = np.random.uniform(0, 250, n)
        lastLum = lums[-1]
        return lums

    def getLum(self):
        return self.measure()

    def checkOK(self, msg):
        return True

    def sendMessage(self, message):
        return 'Ok'