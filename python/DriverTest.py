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
import signal

from picamera import PiCamera
from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor
from PiVideoStream import PiVideoStream
import DriveRecorder

class App():
    
    def __init__( self ):
        self.speed = 150
        self.last_command = None
        self.record_path = None
        self.raw = None

    def run(self):
        self.raw = PiVideoStream( resolution=(480,480), imsize=120, framerate=32 )
        self.raw.start()
        self.mh = Adafruit_MotorHAT(addr=0x60)
        self.driver = None

    def handle_client(self, sock):
        for line in sock.makefile('r'):
            if line.startswith( 'video_start', 0, len('video_start') ):
                self.startVideo( line[len('video_start '):] )
            elif line.startswith( 'video_stop', 0, len('video_stop') ):
                self.endVideo( line[len('video_stop '):] )
            elif line.startswith( 'set ', 0, len('set ') ):
                self.adjust( line[len('set '):] )
            elif line.startswith( 'forward', 0, len('forward') ):
                self.driveForward(self.speed)
                self.recordCommand('forward')
            elif line.startswith( 'backward', 0, len('backward') ):
                self.driveBackward(self.speed)
                self.recordCommand('backward')
            elif line.startswith( 'left', 0, len('left') ):
                self.turnLeft(self.speed)
                self.recordCommand('left')
            elif line.startswith( 'right', 0, len('right') ):
                self.turnRight(self.speed)
                self.recordCommand('right')
            elif line.startswith( 'stop', 0, len('stop') ):
                self.stopMotors()
                self.recordCommand('stop')
            elif line.startswith( 'speed', 0, len('speed') ):
                self.setSpeed( line[len('speed '):] )
                self.recordCommand(line)
            elif line.startswith( 'drive_start', 0, len('drive_start') ):
                name = None
                if len(line) > len('drive_start'):
                    name = line[len('drive_start '):]
                self.startDrive(name)
            elif line.startswith( 'drive_end', 0, len('drive_end') ):
                self.endDrive()
            elif line.startswith( 'exit_all', 0, len('exit_all') ):
                self.stop()
            else:
                jpeg_byte_string = self.getImage2()
                sock.sendall( jpeg_byte_string )
                sock.close()

    def stop(self, signum=0, frame=0):
        if self.server:
            self.server.close()
            self.server = None
        if self.raw:
            self.raw.stop()
        self.endDrive()
        time.sleep(0.5)

    def getImage2(self):
        image_string = None
        my_stream = io.BytesIO()
        if self.raw:
            np_image, _ = self.raw.read()
            if np_image is not None:
                scipy.misc.imsave( my_stream, np_image, 'jpeg' )
        my_stream.seek(0)
        image_string = my_stream.read(-1)
        return image_string

    def adjust(self, args):
        if self.raw:
            try:
                attr, value = args.split(' ')
                if attr == 'brightness':
                    value = int(value)
                    if value > 0 and value <= 100:
                        print( "setting brightness: {}".format( str(value) ) )
                        self.raw.brightness = value
                elif attr == 'shutter_speed':
                    value = int(value)
                    self.raw.shutter_speed = value
                    print( "setting shutter speed: {}".format( str(value) ) )
                elif attr == 'iso':
                    value = int(value)
                    self.raw.iso = value
                    print( "setting iso: {}".format( str(value) ) )
                elif attr == 'framerate':
                    value = int(value)
                    self.raw.framerate = value
                    print( "setting framerate: {}".format( str(value) ) )
            except Exception as ex:
                print( "{}".format( ex ) )

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

    def startDrive( self, drive_name ):
        if not self.driver:
            n = datetime.datetime.now()
            fname = n.strftime('drive_%Y%m%d_%H%M%S')
            drive_dir = os.path.join( "/home/andrew/drive", fname )
            self.driver = DriveRecorder.DriveRecorder( drive_dir, video_path=self.videoPath(fname), camera=self.raw, image_delay=0.1, drive_name=drive_name )
            self.driver.startDriving()

    def endDrive( self ):
        if self.driver:
            self.driver.endDriving()
        # Tell the drive object to stop, saving everything to disk
        self.driver = None

    def recordCommand( self, command ):
        if self.driver:
            self.driver.addAction( command )
            #self.driver.addImage( istream, 'jpeg' )

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

if __name__ == "__main__":
    app = App( "/home/andrew/drive/var" )

