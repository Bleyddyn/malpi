# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
from threading import Thread, Lock
from scipy import misc
import numpy as np

class PiVideoStream:
    """ Threaded reading from PiCamera directly into a numpy array
        From: http://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
        Example:
            vs = PiVideoStream( resolution=(width,height))
            vs.start()
            # Short delay for the camera to warm up?
            frame = vs.read()
    """
    def __init__(self, resolution=(320, 240), framerate=32, brightness=None, imsize=79):
        # initialize the camera and stream
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        if brightness:
            self.camera.brightness = brightness
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture, format="rgb", use_video_port=True)

        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.imsize = imsize
        self.pre_frame = None
        self.stopped = False
        self.lock = Lock()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # I'm not 100% sure this will result in everything being closed. Best to also call stop().
        self.stopped = True

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        for f in self.stream:
            # grab the frame from the stream and clear the stream in
            # preparation for the next frame
            with self.lock:
                self.frame = f.array
                self.pre_frame = self.preprocess( self.frame )
            self.rawCapture.truncate(0)

            # if the thread indicator variable is set, stop the thread
            # and resource camera resources
            if self.stopped:
                print "Stopping PiVideoStream"
                self.stream.close()
                self.rawCapture.close()
                self.camera.close()
                return

    def read(self):
        # return the most recent frame and pre-processed frame
        with self.lock:
            return (self.frame, self.pre_frame)
        return None

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def preprocess(self, image):
        image = image.transpose(2,1,0)
        image = misc.imresize(image,(self.imsize,self.imsize))
        image = image.reshape(1,3,self.imsize,self.imsize)
        image = image.astype(np.float32)
        return image
