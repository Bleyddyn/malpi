#! /usr/bin/python

# ADXL345 Python example 
#
# author:  Jonathan Williamson
# license: BSD, see LICENSE.txt included in this package
# 
# This is an example to show you how to use our ADXL345 Python library
# http://shop.pimoroni.com/products/adafruit-triple-axis-accelerometer

from time import sleep
from time import time
import signal
import socket
import pickle
import datetime
from threading import Thread
import argparse
#import numpy as np

from adxl345 import ADXL345
  
class Accelerometer:
    def __init__(self):
        # initialize the accellerometer
        self.adxl345 = ADXL345()
        self.stopped = False
        self.results = []
        self.t_start = time()
        self.count = 0
        self.elapsed = 0
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # I'm not 100% sure this will result in everything being closed. Best to also call stop().
        self.stopped = True

    def start(self):
        # start the thread to read frames from the video stream
        self.stopped = False
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        self.results = []
        self.t_start = time()
        self.count = 0
        while True:
            axes = self.adxl345.getAxes(True)
            self.results.append( [time(),axes['x'], axes['y'], axes['z']] )
            self.count += 1
            self.elapsed = time() - self.t_start
            sleep(0.0085) # With overhead, this works out to about 100 samples per second
            # if the thread indicator variable is set, stop the thread
            # and resource camera resources
            if self.stopped:
                self.t_end = time()
                return

    def read(self):
        return self.results

    def stop(self,signum=0,frame=0):
        # indicate that the thread should be stopped
        self.stopped = True

def _sendMotorCommand(command):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host ="127.0.0.1"
        port = 12345
        s.connect((host,port))
        s.sendall(command.encode()) 
        s.shutdown(socket.SHUT_WR)
        s.close()
    except Exception as inst:
        error_string = "Error in sendMotorCommand: %s" % str(inst)
        print( error_string )

        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        pass

def _twoByTwo():
    for x in range(3):
        _sendMotorCommand('forward')
        sleep(2)
        _sendMotorCommand('stop')
        sleep(2)

def _crashes():
    for x in range(2):
        _sendMotorCommand('forward')
        sleep(2)
        _sendMotorCommand('backward')
        sleep(1)
        _sendMotorCommand('stop')

def _test():
    accel = Accelerometer()
    print( "ADXL345 on address 0x%x:" % (accel.adxl345.address) )
    t_start = time()
    accel.start()
    t1 = time()
    sleep(2)
#_twoByTwo()
    _crashes()
    t2 = time()
    accel.stop()
    t3 = time()
    sleep(0.1)
    t4 = time()
    results = accel.read()
    t5 = time()

    fmt = '%Y%m%d-%H%M%S'
    filename = "accel_" + datetime.datetime.now().strftime(fmt) + ".pickle"
    with open(filename, 'wb') as f:
        pickle.dump( results, f, pickle.HIGHEST_PROTOCOL)

    print( "Per second: %d/%f = %f" % ( accel.count, accel.elapsed, accel.count / accel.elapsed ) )
    print( "Timing: %f - %f - %f - %f - %f" % (t1-t_start, t2-t1, t3-t2, t4-t3, t5-t4) )
    print( "%f %f" % (t3, accel.t_end) )
#Timing: 2.580505 - 3.181781 - 0.000015 - 0.100149 - 0.000010 )
    print( results[0] )
    print( results[1] )
    print( results[2] )
    print( results[3] )

def parseArguments():
    parser = argparse.ArgumentParser(description='Record accelerometer data until interrupted, then save to file.')
    parser.add_argument('filename', metavar='Filename', help='Filename to save pickled data')
    parser.add_argument('--count', dest='count', type=int, action='store', default=0, help='Maximum number of samples to record (Not yet implemented)')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parseArguments()
    accel = Accelerometer()
    accel.update()
# wait for ctrl-c
    data = accel.read()
    with open(args.filename, 'wb') as f:
        pickle.dump( data, f, pickle.HIGHEST_PROTOCOL)
