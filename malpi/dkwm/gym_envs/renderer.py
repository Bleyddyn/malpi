from enum import Enum

import pyglet
from pyglet.gl import GLubyte, glFlush
import numpy as np

from ctypes import POINTER

# Setting this env variable was needed to let pyglet find libc:
# export DYLD_FALLBACK_LIBRARY_PATH=/usr/lib

class DKWMRenderer(object):
    """ A renderer for the DKWM gym.
        Holds and displays the previous image. Draws a given label to one of the display corners.
        Based on: https://github.com/maximecb/gym-miniworld/blob/master/gym_miniworld/miniworld.py
    """

    # See: http://www.blog.pythonlibrary.org/2018/03/20/python-3-an-intro-to-enumerations/
    labelEnum = Enum( "Label", "TopLeft TopRight BottomLeft BottomRight" )

    def __init__( self, window_width, window_height ):
        self.window_width = window_width
        self.window_height = window_height 
        self.window = None
        self.labels = {}

    def close(self):
        if self.window is not None:
            self.window.close()

    def set_obs( self, next_obs ):
        self.last_obs = next_obs

    def clear_label( self, label_id ):
        self.labels.pop( label_id, None )

    def set_label( self, label_text, label_id, location=labelEnum.TopRight ):
# TODO Handle location
        self.labels[label_id] = pyglet.text.Label(
            font_name="Courier",
            font_size=12,
            multiline=True,
            width=400,
            x = 10,
            y = 30
        )
        self.labels[label_id].text = label_text
        
    def render(self, mode='human'):

        img = self.last_obs

        if mode == 'rgb_array':
            return img

        if self.window is None:
            config = pyglet.gl.Config(double_buffer=True)
            self.window = pyglet.window.Window(
                width=self.window_width,
                height=self.window_height,
                resizable=False,
                config=config
            )

        self.window.clear()
        self.window.switch_to()

        img_width = img.shape[1]
        img_height = img.shape[0]

        img = np.ascontiguousarray(np.flip(img, axis=0))
        img_data = pyglet.image.ImageData(
            img_width,
            img_height,
            'RGB',
            img.ctypes.data_as(POINTER(GLubyte)),
            pitch=img_width * 3,
        )

        img_left = (self.window_width - img_width) // 2
        img_top = (self.window_height - img_height) // 2

        img_data.blit(
            img_left,
            img_top,
            0,
            width=img_width,
            height=img_height
        )

        for a_label in self.labels.values():
            # Draw the text label in the window
            a_label.draw()

        # Force execution of queued commands
        glFlush()

        # If we are not running the Pyglet event loop,
        # we have to manually flip the buffers and dispatch events
        if mode == 'human':
            self.window.flip()
            self.window.dispatch_events()

    def reset(self):
        self.labels = {}
        if self.window is not None:
            self.window.clear()
