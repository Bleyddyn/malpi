#!/usr/bin/python

# To kick off the script, run the following from the python directory:
#   PYTHONPATH=`pwd` python DriveDaemon.py start
#

#standard python libs
import logging
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

from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor
from PiVideoStream import PiVideoStream
import DriveRecorder
import Driver

#third party libs
from daemon import runner

try:
    import config
except:
    print( "Failed to load config file config.py." )
    print( "Try copying config_empty.py to config.py and re-running." )
    exit()

class App():
    
    def __init__( self, var_dir ):
        self.stdin_path = '/dev/null'
        self.stdout_path = '/dev/tty'
        self.stderr_path = '/dev/tty'
        self.var_dir = var_dir
        self.pidfile_path = var_dir + '/drive.pid'
        self.pidfile_timeout = 5
        self.speed = 150
        self.last_command = None
        self.record_path = None
        self.camera = None
        self.raw = None
        self.server = None

    def run(self):
        self.camera = None
        self.raw = PiVideoStream( resolution=(480,480), imsize=120, framerate=32 )
        self.raw.start()
        self.mh = Adafruit_MotorHAT(addr=0x60)
        self.recorder = None
        self.driver = None

        signal.signal(signal.SIGTERM, self.stop)

        server = socket.socket()
        self.server = server
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1",12347))
        server.listen(5)
        logger.info("Starting up")

        while self.server is not None:
            try:
                conn, address = server.accept()
                thread = threading.Thread(target=self.handle_client, args=[conn])
                thread.daemon = True
                thread.start()
            except Exception as ex:
                logger.info( ex )

    def handle_client(self, sock):
        for line in sock.makefile('r'):
            if not self.do_action(line):
                jpeg_byte_string = self.getImage2()
                sock.sendall( jpeg_byte_string )
                sock.close()

    def do_action(self, action):
        if action.startswith( 'video_start', 0, len('video_start') ):
            self.startVideo( action[len('video_start '):] )
        elif action.startswith( 'video_stop', 0, len('video_stop') ):
            self.endVideo( action[len('video_stop '):] )
        elif action.startswith( 'set ', 0, len('set ') ):
            self.adjust( action[len('set '):] )
        elif action.startswith( 'forward', 0, len('forward') ):
            self.driveForward(self.speed)
            self.recordCommand('forward')
        elif action.startswith( 'backward', 0, len('backward') ):
            self.driveBackward(self.speed)
            self.recordCommand('backward')
        elif action.startswith( 'left', 0, len('left') ):
            self.turnLeft(self.speed)
            self.recordCommand('left')
        elif action.startswith( 'right', 0, len('right') ):
            self.turnRight(self.speed)
            self.recordCommand('right')
        elif action.startswith( 'stop', 0, len('stop') ):
            self.stopMotors()
            self.recordCommand('stop')
        elif action.startswith( 'speed', 0, len('speed') ):
            self.setSpeed( action[len('speed '):] )
            self.recordCommand(action)

        elif action.startswith( 'record_start', 0, len('record_start') ):
            name = None
            if len(action) > len('record_start'):
                name = action[len('record_start '):]
            self.startRecording(name)
        elif action.startswith( 'record_end', 0, len('record_end') ):
            self.endRecording()

        elif action.startswith( 'drive_start', 0, len('drive_start') ):
            name = None
            if len(action) > len('drive_start'):
                name = action[len('drive_start '):]
            self.startDrive(name)
        elif action.startswith( 'drive_end', 0, len('drive_end') ):
            self.endDrive()

        elif action.startswith( 'exit_all', 0, len('exit_all') ):
            self.stop()
        else:
            return False
        logger.info(action)
        return True

    def stop(self, signum=0, frame=0):
        logger.info("Shutting down...")
        if self.server:
            self.server.close()
            self.server = None
        if self.camera:
            logger.info("Shutting down camera")
            if self.camera.recording:
                self.camera.stop_recording()
            self.camera.close()
        if self.raw:
            logger.info("Shutting down raw camera")
            self.raw.stop()
        self.endDrive()
        time.sleep(0.5)
        logger.info("Shut down")

    def getImage2(self):
        image_string = None
        my_stream = io.BytesIO()
        if self.camera:
            self.camera.capture(my_stream, 'jpeg')
        if self.raw:
            _, np_image = self.raw.read()
            if np_image is not None:
                scipy.misc.imsave( my_stream, np_image, 'jpeg' )
        my_stream.seek(0)
        image_string = my_stream.read(-1)
        return image_string

    def adjust(self, args):
        if self.camera:
            try:
                attr, value = args.split(' ')
                if attr == 'brightness':
                    value = int(value)
                    if value > 0 and value <= 100:
                        print( "setting brightness: {}".format( str(value) ) )
                        self.camera.brightness = value
                elif attr == 'shutter_speed':
                    value = int(value)
                    self.camera.shutter_speed = value
                    print( "setting shutter speed: {}".format( str(value) ) )
                elif attr == 'iso':
                    value = int(value)
                    self.camera.iso = value
                    print( "setting iso: {}".format( str(value) ) )
                elif attr == 'framerate':
                    value = int(value)
                    self.camera.framerate = value
                    print( "setting framerate: {}".format( str(value) ) )
            except Exception as ex:
                print( "{}".format( ex ) )

    def videoPath( self, filename):
        """ Return the full path for saving a video given a filename.
            Defaults to /var/ramdrive
        """
        filename = os.path.basename(filename)
        if not filename:
            filename = "malpi.h264"
        if not filename.endswith(".h264"):
            filename += ".h264"
        filename = os.path.join("/var/ramdrive", filename)
        return filename

    def startVideo(self, filename):

        filename = self.videoPath( filename )

        #Other possible options
        #camera.annotate_text = "Hello world!"
        #camera.brightness = 50 #0-100
        #camera.contrast = 50 #0-100

        if self.camera:
            if self.camera.recording:
                self.camera.stop_recording()

            self.camera.resolution = (640, 480)
            self.camera.framerate = 15
            self.camera.start_recording(filename)
        if self.raw:
            self.raw.startVideo(filename)
        if self.recorder:
            self.recorder.setVideoFilename(filename)

    def endVideo(self, filename):
        if self.camera.recording:
            self.camera.stop_recording()

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
            if not model_name:
                #model_name = "home_19k_weights.h5"
                #model_name = "home_19k_gru_extra_stride_weights.h5"
                model_name = "home_19k_fc_extra_stride_weights.h5"
            model_path = os.path.join( config.directories['models'], model_name )
            self.driver = Driver.Driver( model_path, camera=self.raw, controller=self )
        self.driver.startDriving()

    def endDrive( self ):
        if self.driver:
            self.driver.endDriving()
        #self.driver = None
        #self.stopMotors()

    def startRecording( self, drive_name ):
        if not self.recorder:
            n = datetime.datetime.now()
            fname = n.strftime('%Y%m%d_%H%M%S.drive')
            drive_dir = os.path.join( config.directories['drives'], fname )
            self.recorder = DriveRecorder.DriveRecorder( drive_dir, video_path=self.videoPath(fname), camera=self.raw, image_delay=0.1, drive_name=drive_name )
            self.recorder.startDriving()
            logger.info("Starting Recording")

    def endRecording( self ):
        if self.recorder:
            self.recorder.endDriving()
        # Tell the drive object to stop, saving everything to disk
        self.recorder = None
        logger.info("Ended Recording")

    def recordCommand( self, command ):
        if self.recorder:
            self.recorder.addAction( command )
            #self.recorder.addImage( istream, 'jpeg' )

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

var_path = os.path.join( config.directories['drives'], 'var' )
app = App( var_path )
logger = logging.getLogger("DriveLog")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = logging.FileHandler( app.var_dir + "/drive.log")
handler.setFormatter(formatter)
logger.addHandler(handler)

def stop_app(signum=0, frame=0):
    app.stop()


daemon_runner = runner.DaemonRunner(app)
#This ensures that the logger file handle does not get closed during daemonization
daemon_runner.daemon_context.files_preserve=[handler.stream]
daemon_runner.do_action()

# Convert to mp4: MP4Box -add video.h264 video.mp4
