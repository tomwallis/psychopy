"""Gamma scientific photometer
See http://www.gamma-sci.com/products/light-sensors/

----------
"""
# Author: Jonathan Oesterle

from psychopy import logging
import numpy as np
import struct
import sys

try:
    import serial
except ImportError:
    serial = False


class S470(object):
    """A class to define a UDT flexOptometer S470 connected to a photodiode (e.g. Model 211 Photometric Sensor)

    usage::

        from psychopy.hardware import udt
        phot = udt.S470(port)
        if phot.OK:  # then we successfully made a connection
            print(phot.getLum())

    :parameters:

        port: string

            the serial port that should be checked
    """

    longName = "flexOptmeterS470"
    driverFor = ["s470"]

    def __init__(self, port):
        super(S470, self).__init__()

        if not serial:
            raise ImportError("The module serial is needed to connect to "
                              "photometers. On most systems this can be "
                              "installed with\n\t easy_install pyserial")

        if type(port) in [int, float]:
            # add one so that port 1=COM1
            self.portNumber = port
            self.portString = 'COM%i' % self.portNumber
        else:
            self.portString = port
            self.portNumber = None
        self.isOpen = 0
        self.lastLum = None
        self.type = 'S470'
        self.com = False
        self.OK = True  # until we fail
        self.maxAttempts = 10

        _linux = sys.platform.startswith('linux')
        if sys.platform in ['darwin', 'win32'] or _linux:
            try:
                self.com = serial.Serial(self.portString)
            except Exception:
                msg = ("Couldn't connect to port %s. Is it being used by "
                       "another program?")
                self._error(msg % self.portString)
        else:
            msg = "I don't know how to handle serial ports on %s"
            self._error(msg % sys.platform)

        if self.OK:
            self.com.bytesize = 8
            self.com.baudrate = 38400
            self.com.parity = serial.PARITY_NONE
            try:
                if not self.com.isOpen():
                    self.com.open()
            except Exception:
                msg = "Opened serial port %s, but couldn't connect to S470."
                self._error(msg % self.portString)
            else:
                self.isOpen = 1

        if self.OK:  # we have an open com port. try to send a command
            for repN in range(self.maxAttempts):
                reply = self.sendMessage('')
                if reply == 'Ok':
                    self.OK = True
                    break
                else:
                    self.OK = False  # false so far but keep trying
                    
        if self.OK:  # we have successfully sent and read a command
            logging.info("Successfully opened %s" % self.portString)
            
    def getAvailableCals(self):
        """Get a list of available calibrations.

        See user manual for details.
        """
        reply = self.sendMessage('CAL *')
        return reply
        
    def getCurrentCal(self):
        """Get the currently selected calibration.

        See user manual for details.
        """
        reply = self.sendMessage('CAL')
        return reply
        
    def setCal(self, n):
        """Get the currently selected calibration.

        See user manual for details.
        """
        reply = self.sendMessage('CAL %s' % range)
        return self.checkOK(reply)

    def setRange(self, range):
        """Range is the exponent representative of the desired range, either 3 through 10 for DC radiometer 
            mode, or -6 through -9 for energy measurement (pulse integration) mode. 

        See user manual for details.
        """
        reply = self.sendMessage('RNG %s' % range)
        return self.checkOK(reply)
               
    def setSampleRate(self, sampleRate):
        """While the front-panel display is updated four times per second, it is possible to take a channel's readings 
            much faster over the computer interface, at up to 250 readings per second. This command is to select the 
            internal sample rate at which readings will be made over the computer interface. 

        See user manual for details.
        """
        reply = self.sendMessage('SRT %s' % sampleRate)
        
        print('Desired sample rate: %d' % sampleRate)
        print('Actual sample rate: %s' % reply);

        return reply

    def measure(self):
        """Measure the current luminance and set .lastLum to this value
        """
        reply = self.sendMessage('REA')
        lum = float(reply)
        lastLum = lum
        return lum
        
    def measure_n(self, n):
        """Measure the current luminance and set .lastLum to this value
        """
        n = int(np.floor(n))

        lums = np.zeros((n, 1))
        reply = self.sendMessage('REA')
        lums[1] = float(reply)

        for i in range(2, n):
            retVal = self.com.read(self.com.inWaiting())  # read
            # retVal = self.com.read(20)
            lums[i] = retVal[2:-2] # Remove start and stop signal
            lastLum = lums[i]

        return lum

    def getLum(self):
        """Makes a measurement and returns the luminance value
        """
        return self.measure()

    def checkOK(self, msg):
        """Check that the message from the photometer is OK.
        If there's an error show it (printed).

        Then return True (OK) or False.
        """
        # also check that the reply is what was expected
        if msg != 'Ok':
            if msg == '':
                logging.error('No reply from S470')
                sys.stdout.flush()
            else:
                logging.error('Error message from S470:' + msg)
                sys.stdout.flush()
            return False
        else:
            return True

    def sendMessage(self, message):
        """Send a command to the photometer
        """
        if message[-2:] != '\r\n':
            message += '\r\n'  # append a newline if necessary

        self.com.read(self.com.inWaiting()) # flush buffer.
        self.com.write(message) # request read measurement
        retVal = self.com.read(self.com.inWaiting()) # read
        # retVal = self.com.read(20)
        retVal = retVal[2:-2] # remove start and stop signal

        return retVal

    def _error(self, msg):
        self.OK = False
        logging.error(msg)