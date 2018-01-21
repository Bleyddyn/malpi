#!/usr/bin/python

#standard python libs
import time
import datetime
import socket
import threading
import os
import io
import atexit
#from cStringIO import StringIO
import scipy.misc
import argparse

from picamera import PiCamera
from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor
from PiVideoStream import PiVideoStream
import Driver

class App():
    
    def __init__( self ):
        self.speed = 150
        self.last_command = None
        self.raw = PiVideoStream( resolution=(480,480), imsize=120, framerate=32 )
        self.raw.start()
        self.mh = Adafruit_MotorHAT(addr=0x60)
        self.driver = None

    def do_action(self, action):
        print( "{}".format( action ) )
        if action.startswith( 'forward', 0, len('forward') ):
            self.driveForward(self.speed)
        elif action.startswith( 'backward', 0, len('backward') ):
            self.driveBackward(self.speed)
        elif action.startswith( 'left', 0, len('left') ):
            self.turnLeft(self.speed)
        elif action.startswith( 'right', 0, len('right') ):
            self.turnRight(self.speed)
        elif action.startswith( 'stop', 0, len('stop') ):
            self.stopMotors()
        elif action.startswith( 'speed', 0, len('speed') ):
            self.setSpeed( action[len('speed '):] )
        else:
            print( "invalid command" )

    def stop(self, signum=0, frame=0):
        if self.raw:
            self.raw.stop()
        self.endDrive()
        time.sleep(0.5)
        self.stopMotors()

    def setSpeed( self, newSpeed ):
        try:
            spInt = int(newSpeed)
        except ValueError:
            pass
        else:
            if spInt >= 0 and spInt < 256 and spInt != self.speed:
                self.speed = spInt
                if self.last_command:
                    self.last_command(spInt)

    def startDrive( self, model_name ):
        if not self.driver:
            model_path = os.path.join( "~/models", model_name )
            self.driver = Driver.Driver( model_path, camera=self.raw, controller=self )
            self.driver.startDriving()
            #self.driver._step()

    def endDrive( self ):
        if self.driver:
            self.driver.endDriving()
        # Tell the drive object to stop, saving everything to disk
        self.driver = None

    def setMotor( self, mnum, forward, speed ):
        myMotor = None
        for i in range(10):
            try:
                myMotor = self.mh.getMotor(mnum)
            except Exception as ex:
                myMotor = None
            if myMotor is not None:
                break

        if myMotor is None:
            print( "Error connecting to motor" )
            return

        sent = False
        for i in range(10):
            try:
                # set the speed to start, from 0 (off) to 255 (max speed)
                myMotor.setSpeed(speed)
                if forward:
                    myMotor.run(Adafruit_MotorHAT.FORWARD)
                else:
                    myMotor.run(Adafruit_MotorHAT.BACKWARD)
                sent = True
            except Exception as ex:
                sent = False
            if sent:
                break
        if not sent:
            fwd = "backward"
            if forward:
                fwd = "forward"
            print( "Failed to send motor command: {} {}".format( mnum, fwd ) )

    def driveForward( self, speed=150 ):
        self.last_command = self.driveForward
        self.setMotor(1, True, speed)
        self.setMotor(2, True, speed)
        self.setMotor(3, True, speed)
        self.setMotor(4, True, speed)

    def driveBackward( self, speed=150 ):
        self.last_command = self.driveBackward
        self.setMotor(1, False, speed)
        self.setMotor(2, False, speed)
        self.setMotor(3, False, speed)
        self.setMotor(4, False, speed)

    def turnLeft( self, speed=150 ):
        self.last_command = self.turnLeft
        self.setMotor(1, False, speed)
        self.setMotor(2, False, speed)
        self.setMotor(3, True, speed)
        self.setMotor(4, True, speed)

    def turnRight( self, speed=150 ):
        self.last_command = self.turnRight
        self.setMotor(1, True, speed)
        self.setMotor(2, True, speed)
        self.setMotor(3, False, speed)
        self.setMotor(4, False, speed)

    def stopMotors(self):
        self.last_command = None
        self.mh.getMotor(1).run(Adafruit_MotorHAT.RELEASE)
        self.mh.getMotor(2).run(Adafruit_MotorHAT.RELEASE)
        self.mh.getMotor(3).run(Adafruit_MotorHAT.RELEASE)
        self.mh.getMotor(4).run(Adafruit_MotorHAT.RELEASE)

def getOptions():

    parser = argparse.ArgumentParser(description='Train on robot image/action data.')
    parser.add_argument('model', nargs=1, help='path to a trained Keras model')
    parser.add_argument('--test_only', action="store_true", default=False, help='run tests, then exit')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = getOptions()

    if args.test_only:
        exit()

    try:
        app = App()
        model_name = os.path.expanduser(args.model[0])
        app.startDrive( model_name )
        time.sleep(10)
        app.stop()
    except Exception as ex:
        print( "{}".format( ex ) )
        app.stop()
