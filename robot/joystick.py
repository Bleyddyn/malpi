#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 10:44:24 2017

@author: wroscoe
"""

import time
from threading import Thread
import socket

from donkeycar.parts.controller import JoystickController, PS3JoystickController, PS3Joystick

class Vehicle():
    def __init__(self, mem=None):

        if not mem:
            mem = Memory()
        self.mem = mem
        self.parts = []
        self.on = True
        self.threads = []


    def add(self, part, inputs=[], outputs=[], 
            threaded=False, run_condition=None):
        """
        Method to add a part to the vehicle drive loop.

        Parameters
        ----------
            inputs : list
                Channel names to get from memory.
            ouputs : list
                Channel names to save to memory.
            threaded : boolean
                If a part should be run in a separate thread.
        """

        p = part
        print('Adding part {}.'.format(p.__class__.__name__))
        entry={}
        entry['part'] = p
        entry['inputs'] = inputs
        entry['outputs'] = outputs
        entry['run_condition'] = run_condition

        if threaded:
            t = Thread(target=part.update, args=())
            t.daemon = True
            entry['thread'] = t

        self.parts.append(entry)


    def start(self, rate_hz=10, max_loop_count=None):
        """
        Start vehicle's main drive loop.

        This is the main thread of the vehicle. It starts all the new
        threads for the threaded parts then starts an infinit loop
        that runs each part and updates the memory.

        Parameters
        ----------

        rate_hz : int
            The max frequency that the drive loop should run. The actual
            frequency may be less than this if there are many blocking parts.
        max_loop_count : int
            Maxiumum number of loops the drive loop should execute. This is
            used for testing the all the parts of the vehicle work.
        """

        try:

            self.on = True

            for entry in self.parts:
                if entry.get('thread'):
                    #start the update thread
                    entry.get('thread').start()

            #wait until the parts warm up.
            print('Starting vehicle...')
            time.sleep(1)

            loop_count = 0
            while self.on:
                start_time = time.time()
                loop_count += 1

                self.update_parts()

                #stop drive loop if loop_count exceeds max_loopcount
                if max_loop_count and loop_count > max_loop_count:
                    self.on = False

                sleep_time = 1.0 / rate_hz - (time.time() - start_time)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


    def update_parts(self):
        '''
        loop over all parts
        '''
        for entry in self.parts:
            #don't run if there is a run condition that is False
            run = True
            if entry.get('run_condition'):
                run_condition = entry.get('run_condition')
                run = self.mem.get([run_condition])[0]
                #print('run_condition', entry['part'], entry.get('run_condition'), run)
            
            if run:
                p = entry['part']
                #get inputs from memory
                inputs = self.mem.get(entry['inputs'])

                #run the part
                if entry.get('thread'):
                    outputs = p.run_threaded(*inputs)
                else:
                    outputs = p.run(*inputs)

                #save the output to memory
                if outputs is not None:
                    self.mem.put(entry['outputs'], outputs)

                    

    def stop(self):
        print('Shutting down vehicle and its parts...')
        for entry in self.parts:
            try:
                entry['part'].shutdown()
            except Exception as e:
                print(e)
        print(self.mem.d)

def sendCommand( steering, throttle, recording ):
    #valid_commands = ["forward","backward","stop","left","right"]
    min_ctrl = 0.1
    direction = "stop"
    if recording is not None:
        direction = recording
        print( "Recording: {}".format( direction ) )
    else:
        if throttle > min_ctrl:
            direction = "forward"
        elif throttle < -min_ctrl:
            direction = "stop"
        elif steering < -min_ctrl:
            direction = "left"
        elif steering > min_ctrl:
            direction = "right"
        else:
            return
    
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host ="127.0.0.1"
        port = 12347
        s.connect((host,port))
        s.send(direction.encode()) 
        s.close()
    except Exception as inst:
        print( "Failed to send command" )

def sendContinuousCommand( left_throttle, right_throttle, recording, dk=False ):
    #min_ctrl = 0.1
    #if abs(left_throttle) < min_ctrl and abs(right_throttle) < min_ctrl:
    #    left_throttle = 0.0
    #    right_throttle = 0.0

    if recording is not None:
        direction = recording
        print( "Recording: {}".format( direction ) )
    elif dk:
        direction = "dk {} {}".format(left_throttle, right_throttle)
    else:
        direction = "throttles {} {}".format(left_throttle, right_throttle)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host ="127.0.0.1"
        port = 12347
        s.connect((host,port))
        s.send(direction.encode()) 
        s.close()
    except Exception as inst:
        print( "Failed to send continuous command" )

class MalpiJoystickController(JoystickController):
    '''
    A Controller object that maps inputs to actions
    '''
    def __init__(self, *args, **kwargs):
        super(MalpiJoystickController, self).__init__(*args, **kwargs)


    def init_js(self):
        '''
        attempt to init joystick
        '''
        try:
            self.js = PS3Joystick(self.dev_fn)
            if not self.js.init():
                self.js = None
        except FileNotFoundError:
            print(self.dev_fn, "not found.")
            self.js = None
        return self.js is not None

    def init_trigger_maps(self):
        '''
        init set of mapping from buttons to function calls
        '''

        self.button_down_trigger_map = {
            'select' : self.toggle_mode,
            'circle' : self.toggle_manual_recording,
            'triangle' : self.erase_last_N_records,
            'cross' : self.emergency_stop,
            #'dpad_up' : self.increase_max_throttle,
            #'dpad_down' : self.decrease_max_throttle,
            'start' : self.toggle_constant_throttle,
            #"R1" : self.chaos_monkey_on_right,
            #"L1" : self.chaos_monkey_on_left,
        }

        self.button_up_trigger_map = {
            #"R1" : self.chaos_monkey_off,
            #"L1" : self.chaos_monkey_off,
        }

        self.axis_trigger_map = {
            'left_stick_horz' : self.set_steering,
            'right_stick_vert' : self.set_throttle,
        }


if __name__ == "__main__":

    JOYSTICK_MAX_THROTTLE = 1.0
    JOYSTICK_STEERING_SCALE = 1.0
    AUTO_RECORD_ON_THROTTLE = False
    ctr = MalpiJoystickController(throttle_scale=JOYSTICK_MAX_THROTTLE,
                             steering_scale=JOYSTICK_STEERING_SCALE,
                             auto_record_on_throttle=AUTO_RECORD_ON_THROTTLE)
     
    rate_hz=10
    max_loop_count=None
    recording_state = False
    RAW_OUTPUTS = False

    t = Thread(target=ctr.update, args=())
    t.daemon = True
    t.start()

    try:
        #wait until the parts warm up.
        print('Starting vehicle...')
        time.sleep(1)

        loop_count = 0
        done = False
        while not done:
            start_time = time.time()
            loop_count += 1
            if loop_count == 1:
                ctr.print_controls()
                ctr.js.show_map()

            if RAW_OUTPUTS:
                button, button_state, axis, axis_val = ctr.js.poll()
                print( f"Raw: {button}  {button_state}    {axis}  {axis_val}" )
            else:
                outputs = ctr.run_threaded()
                print( "{}".format(outputs) )
                if outputs[2] != "user":
                    break

                if outputs[3] != recording_state:
                    recording_state = outputs[3]
                    rec = 'record_start' if recording_state else 'record_end'
                else:
                    rec = None

                #sendCommand( outputs[0], outputs[1], rec )
                #sendContinuousCommand( outputs[4], outputs[1], rec )
                #sendContinuousCommand( outputs[0], outputs[1], rec, dk=True )
                #print( "L/R: {} {}".format(outputs[0],outputs[1]) )

            #stop drive loop if loop_count exceeds max_loopcount
            if max_loop_count and loop_count > max_loop_count:
                print( "breaking for max count" )
                break

            sleep_time = 1.0 / rate_hz - (time.time() - start_time)
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        #sendContinuousCommand( 0.0, 0.0, 'record_end' )
        ctr.shutdown()
