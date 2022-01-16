#!/usr/bin/env python3
"""
DefaultDriver. A base class with default values for building a DonkeyCar vehicle.
"""
import os
import time
import datetime

import donkeycar as dk

#import parts
from donkeycar.parts.transform import Lambda, TriggeredCallback, DelayedTrigger
from donkeycar.parts.tub_v2 import TubWriter
from donkeycar.parts.controller import LocalWebController, JoystickController
from donkeycar.parts.throttle_filter import ThrottleFilter
from donkeycar.parts.behavior import BehaviorPart
from donkeycar.parts.file_watcher import FileWatcher
from donkeycar.parts.launch import AiLaunch

class TubNamer():
    def __init__(self, path, name_format="{year}{month:02}{day:02}_{num}.{tub}", short_year=False):
        self.path = os.path.expanduser(path)
        self.name_format = name_format
        self.short_year = short_year
        self.next_num = 1

    def create_tub_path(self):
        dt = datetime.datetime.now()
        tub_num = self.next_num
        tub_path = None
        while tub_num < 1000:
            if self.short_year:
                year = dt.strftime('%y')
            else:
                year = dt.year
            name = self.name_format.format( tub="tub", num=tub_num, year=year, month=dt.month, day=dt.day )
            tub_path = os.path.join(self.path, name)
            if not os.path.exists(tub_path):
                break
            tub_num += 1

        self.next_num = tub_num + 1

        return tub_path

class RecordTracker:
    """ TODO: Move this and other parts to a separate file """
    def __init__(self, alert_count, alert_cycle, alert_colors):
        """ alert_count takes a number of records, usually found in cfg.RECORD_COUNT_ALERT
            alert_cycle takes a number of cycles (1/20 of a second), usually found in cfg.RECORD_COUNT_ALERT_CYC
            alert_colors takes a list of count/rgb_tuple, usually found in cfg.RECORD_ALERT_COLOR_ARR
        """
        self.last_num_rec_print = 0
        self.dur_alert = 0
        self.force_alert = 0
        self.alert_count = alert_count
        self.alert_cycle = alert_cycle
        self.alert_colors = alert_colors

    def get_record_alert_color(self, num_records):
        col = (0, 0, 0)
        for count, color in self.alert_colors:
            if num_records >= count:
                col = color
        return col

    def run(self, num_records):
        if num_records is None:
            return 0

        if self.last_num_rec_print != num_records or self.force_alert:
            self.last_num_rec_print = num_records

            if num_records % 10 == 0:
                print("recorded", num_records, "records")

            if num_records % self.alert_count == 0 or self.force_alert:
                self.dur_alert = num_records // self.alert_count * self.alert_cycle
                self.force_alert = 0

        if self.dur_alert > 0:
            self.dur_alert -= 1

        if self.dur_alert != 0:
            return self.get_record_alert_color(num_records)

        return 0

class DefaultDriver():
    '''
    Construct a working robotic vehicle from many parts.
    Each part runs as a job in the Vehicle loop, calling either
    it's run or run_threaded method depending on the constructor flag `threaded`.
    All parts are updated one after another at the framerate given in
    cfg.DRIVE_LOOP_HZ assuming each part finishes processing in a timely manner.
    Parts may have named outputs and inputs. The framework handles passing named outputs
    to parts requesting the same named input.
    '''

    def __init__(self, cfg, model_path=None, use_joystick=False, model_type=None, camera_type='single', meta=[] ):
        self.cfg = cfg
        self.model_path = model_path
        self.use_joystick = use_joystick
        self.model_type = model_type
        self.meta = meta
        self.controller = None
        self.model = None
        self.model_reload_cb = None
        self.vehicle = dk.vehicle.Vehicle()

        self.build(camera_type=camera_type)

    def build(self, camera_type='single'):
        """ In general, sub-classes should not override this method,
            instead override one or more of the specific build methods called from here
        """

        self.build_env() # Anything outside of Python (os.environ, check/create directories, etc.)
        self.build_config()
        self.build_camera(camera_type)
        self.build_controller()
        self.build_inputs()

        model_inputs = self.build_pre_pilot()
        self.build_pilot( model_inputs )
        self.build_post_pilot()

        self.build_displays()
        self.build_drive_train()
        self.build_recording()

        self.user_notifications()

    def build_env(self):
        if self.cfg.DONKEY_GYM:
            #the simulator will use cuda and then we usually run out of resources
            #if we also try to use cuda. so disable for donkey_gym.
            os.environ["CUDA_VISIBLE_DEVICES"]="-1"

        #Sombrero
        if self.cfg.HAVE_SOMBRERO:
            """ This sets GPIO Board pin 37 as a low output """
            from donkeycar.parts.sombrero import Sombrero
            s = Sombrero()

    def build_config(self):
        if self.model_type is None:
            if self.cfg.TRAIN_LOCALIZER:
                self.model_type = "localizer"
            elif self.cfg.TRAIN_BEHAVIORS:
                self.model_type = "behavior"
            else:
                self.model_type = self.cfg.DEFAULT_MODEL_TYPE

    def build_camera(self, camera_type='single'):
        IMAGE_W = self.cfg.IMAGE_W
        IMAGE_H = self.cfg.IMAGE_H
        IMAGE_DEPTH = self.cfg.IMAGE_DEPTH

        print("cfg.CAMERA_TYPE", self.cfg.CAMERA_TYPE)
        if camera_type == "stereo":

            if self.cfg.CAMERA_TYPE == "WEBCAM":
                from donkeycar.parts.camera import Webcam

                camA = Webcam(image_w=IMAGE_W, image_h=IMAGE_H, image_d=IMAGE_DEPTH, iCam = 0)
                camB = Webcam(image_w=IMAGE_W, image_h=IMAGE_H, image_d=IMAGE_DEPTH, iCam = 1)

            elif self.cfg.CAMERA_TYPE == "CVCAM":
                from donkeycar.parts.cv import CvCam

                camA = CvCam(image_w=IMAGE_W, image_h=IMAGE_H, image_d=IMAGE_DEPTH, iCam = 0)
                camB = CvCam(image_w=IMAGE_W, image_h=IMAGE_H, image_d=IMAGE_DEPTH, iCam = 1)
            else:
                raise(Exception("Unsupported camera type: %s" % self.cfg.CAMERA_TYPE))

            self.vehicle.add(camA, outputs=['cam/image_array_a'], threaded=True)
            self.vehicle.add(camB, outputs=['cam/image_array_b'], threaded=True)

            from donkeycar.parts.image import StereoPair

            self.vehicle.add(StereoPair(), inputs=['cam/image_array_a', 'cam/image_array_b'],
                outputs=['cam/image_array'])

        else:
            inputs = []
            outputs = ['cam/image_array']
            threaded = True
            if self.cfg.DONKEY_GYM:
                from donkeycar.parts.dgym import DonkeyGymEnv
                cam_conf = {'img_h': self.cfg.IMAGE_H, 'img_w': self.cfg.IMAGE_W, 'img_d': self.cfg.IMAGE_DEPTH}
                gym_conf = self.cfg.GYM_CONF.copy()
                gym_conf["cam_conf"] = cam_conf
                cam = DonkeyGymEnv(self.cfg.DONKEY_SIM_PATH,
                            env_name=self.cfg.DONKEY_GYM_ENV_NAME,
                            host=self.cfg.SIM_HOST,
                            conf=gym_conf,
                            record_location=self.cfg.SIM_RECORD_LOCATION)
                            #start_sim=self.cfg.DONKEY_GYM_START,
                            #return_info=self.cfg.DONKEY_GYM_INFO,
                            #reset=self.cfg.DONKEY_GYM_RESET)
                threaded = True
                inputs = ['angle', 'throttle']
                if self.cfg.SIM_RECORD_LOCATION:
                    outputs += ['pos/pos_x', 'pos/pos_y', 'pos/pos_z', 'pos/speed', 'pos/cte']
                if self.cfg.SIM_RECORD_GYROACCEL:
                    outputs += ['gyro/gyro_x', 'gyro/gyro_y', 'gyro/gyro_z', 'accel/accel_x', 'accel/accel_y', 'accel/accel_z']
                if self.cfg.SIM_RECORD_VELOCITY:
                    outputs += ['vel/vel_x', 'vel/vel_y', 'vel/vel_z']
                if self.cfg.SIM_RECORD_LIDAR:
                    outputs += ['lidar/dist_array']

            elif self.cfg.CAMERA_TYPE == "PICAM":
                from donkeycar.parts.camera import PiCamera
                cam = PiCamera(image_w=IMAGE_W, image_h=IMAGE_H, image_d=IMAGE_DEPTH)
            elif self.cfg.CAMERA_TYPE == "WEBCAM":
                from donkeycar.parts.camera import Webcam
                cam = Webcam(image_w=IMAGE_W, image_h=IMAGE_H, image_d=IMAGE_DEPTH)
            elif self.cfg.CAMERA_TYPE == "CVCAM":
                from donkeycar.parts.cv import CvCam
                cam = CvCam(image_w=IMAGE_W, image_h=IMAGE_H, image_d=IMAGE_DEPTH)
            elif self.cfg.CAMERA_TYPE == "CSIC":
                from donkeycar.parts.camera import CSICamera
                cam = CSICamera(image_w=IMAGE_W, image_h=IMAGE_H, image_d=IMAGE_DEPTH, framerate=self.cfg.CAMERA_FRAMERATE, gstreamer_flip=self.cfg.CSIC_CAM_GSTREAMER_FLIP_PARM)
            elif self.cfg.CAMERA_TYPE == "V4L":
                from donkeycar.parts.camera import V4LCamera
                cam = V4LCamera(image_w=IMAGE_W, image_h=IMAGE_H, image_d=IMAGE_DEPTH, framerate=self.cfg.CAMERA_FRAMERATE)
            elif self.cfg.CAMERA_TYPE == "MOCK":
                from donkeycar.parts.camera import MockCamera
                cam = MockCamera(image_w=IMAGE_W, image_h=IMAGE_H, image_d=IMAGE_DEPTH)
            else:
                raise(Exception("Unkown camera type: %s" % self.cfg.CAMERA_TYPE))

            self.vehicle.add(cam, inputs=inputs, outputs=outputs, threaded=threaded)

    def build_controller(self):
        if self.use_joystick or self.cfg.USE_JOYSTICK_AS_DEFAULT:
            from donkeycar.parts.controller import get_js_controller

            ctr = get_js_controller(self.cfg)

            if self.cfg.USE_NETWORKED_JS:
                from donkeycar.parts.controller import JoyStickSub
                netwkJs = JoyStickSub(self.cfg.NETWORK_JS_SERVER_IP)
                self.vehicle.add(netwkJs, threaded=True)
                ctr.js = netwkJs

        else:
            #This web controller will create a web server that is capable
            #of managing steering, throttle, and modes, and more.
            ctr = LocalWebController()

        self.vehicle.add(ctr,
              inputs=['cam/image_array'], # TODO: figure out if this is necessary
              outputs=['user/angle', 'user/throttle', 'user/mode', 'recording'],
              threaded=True)

        self.controller = ctr

        #this throttle filter will allow one tap back for esc reverse
        self.vehicle.add(ThrottleFilter(), inputs=['user/throttle'], outputs=['user/throttle'])

    def build_displays(self):
        class LedConditionLogic:
            """ TODO: Move this and other parts to a separate file """
            def __init__(self, cfg, model_type):
                self.cfg = cfg
                self.model_type = model_type

            def run(self, mode, recording, recording_alert, behavior_state, model_file_changed, track_loc):
                #returns a blink rate. 0 for off. -1 for on. positive for rate.

                if track_loc is not None:
                    led.set_rgb(*self.cfg.LOC_COLORS[track_loc])
                    return -1

                if model_file_changed:
                    led.set_rgb(self.cfg.MODEL_RELOADED_LED_R, self.cfg.MODEL_RELOADED_LED_G, self.cfg.MODEL_RELOADED_LED_B)
                    return 0.1
                else:
                    led.set_rgb(self.cfg.LED_R, self.cfg.LED_G, self.cfg.LED_B)

                if recording_alert:
                    led.set_rgb(*recording_alert)
                    return self.cfg.REC_COUNT_ALERT_BLINK_RATE
                else:
                    led.set_rgb(self.cfg.LED_R, self.cfg.LED_G, self.cfg.LED_B)

                if behavior_state is not None and self.model_type == 'behavior':
                    r, g, b = self.cfg.BEHAVIOR_LED_COLORS[behavior_state]
                    led.set_rgb(r, g, b)
                    return -1 #solid on

                if recording:
                    return -1 #solid on
                elif mode == 'user':
                    return 1
                elif mode == 'local_angle':
                    return 0.5
                elif mode == 'local':
                    return 0.1
                return 0

        rec_tracker_part = RecordTracker(self.cfg.REC_COUNT_ALERT, self.cfg.REC_COUNT_ALERT_CYC, self.cfg.RECORD_ALERT_COLOR_ARR)
        self.vehicle.add(rec_tracker_part, inputs=["tub/num_records"], outputs=['records/alert'])

        if self.model_path:
            #this part will signal visual LED, if connected
            self.vehicle.add(FileWatcher(self.model_path, verbose=True), outputs=['modelfile/modified'])

            #these parts will reload the model file, but only when ai is running so we don't interrupt user driving
            self.vehicle.add(FileWatcher(self.model_path), outputs=['modelfile/dirty'], run_condition="ai_running")
            self.vehicle.add(DelayedTrigger(100), inputs=['modelfile/dirty'], outputs=['modelfile/reload'], run_condition="ai_running")
            self.vehicle.add(TriggeredCallback(self.model_path, self.model_reload_cb), inputs=["modelfile/reload"], run_condition="ai_running")

        if self.cfg.HAVE_RGB_LED and not self.cfg.DONKEY_GYM:
            from donkeycar.parts.led_status import RGB_LED
            led = RGB_LED(self.cfg.LED_PIN_R, self.cfg.LED_PIN_G, self.cfg.LED_PIN_B, self.cfg.LED_INVERT)
            led.set_rgb(self.cfg.LED_R, self.cfg.LED_G, self.cfg.LED_B)

            self.vehicle.add(LedConditionLogic(self.cfg, self.model_type), inputs=['user/mode', 'recording', "records/alert", 'behavior/state', 'modelfile/modified', "pilot/loc"],
                  outputs=['led/blink_rate'])

            self.vehicle.add(led, inputs=['led/blink_rate'])

        if self.cfg.AUTO_RECORD_ON_THROTTLE and isinstance(self.controller, JoystickController):
            #then we are not using the circle button. hijack that to force a record count indication
            def show_record_acount_status():
                rec_tracker_part.last_num_rec_print = 0
                rec_tracker_part.force_alert = 1
            self.controller.set_button_down_trigger('circle', show_record_acount_status)

    def build_inputs(self):
        """ Build non-camera inputs.
            These parts should have empty input lists so they don't depend on any other parts
        """

        #IMU
        if self.cfg.HAVE_IMU:
            from donkeycar.parts.imu import Mpu6050
            imu = Mpu6050()
            self.vehicle.add(imu, outputs=['imu/acl_x', 'imu/acl_y', 'imu/acl_z',
                'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z'], threaded=True)

    def build_pre_pilot_conditions(self):

        class AiRunCondition:
            '''
            A bool part to let us know when ai is running.
            '''
            def run(self, mode):
                if mode == "user":
                    return False
                return True

        self.vehicle.add(AiRunCondition(), inputs=['user/mode'], outputs=['ai_running'])

        #See if we should even run the pilot module.
        #This is only needed because the part run_condition only accepts boolean
        class PilotCondition:
            def run(self, mode):
                if mode == 'user':
                    return False
                else:
                    return True

        self.vehicle.add(PilotCondition(), inputs=['user/mode'], outputs=['run_pilot'])

    def build_pre_pilot(self):

        self.build_pre_pilot_conditions()

        class ImgPreProcess():
            '''
            preprocess camera image for inference.
            normalize and crop if needed.
            '''
            def __init__(self, cfg):
                self.cfg = cfg

            def run(self, img_arr):
                img_arr.astype(np.float32) / 255.0

        if "coral" in self.model_type:
            inf_input = 'cam/image_array'
        else:
            inf_input = 'cam/normalized/cropped'
            self.vehicle.add(ImgPreProcess(self.cfg),
                inputs=['cam/image_array'],
                outputs=[inf_input],
                run_condition='run_pilot')

        #Behavioral state
        if self.cfg.TRAIN_BEHAVIORS:
            bh = BehaviorPart(self.cfg.BEHAVIOR_LIST)
            self.vehicle.add(bh, outputs=['behavior/state', 'behavior/label', "behavior/one_hot_state_array"])
            try:
                self.controller.set_button_down_trigger('L1', bh.increment_state)
            except:
                pass

            inputs = [inf_input, "behavior/one_hot_state_array"]
        #IMU
        elif self.model_type == "imu":
            assert(self.cfg.HAVE_IMU)
            inputs=[inf_input,
                'imu/acl_x', 'imu/acl_y', 'imu/acl_z',
                'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z']
        else:
            inputs=[inf_input]

        return inputs

    def get_model(self):
        return dk.utils.get_model_by_type(self.model_type, self.cfg)

    def build_pilot( self, inputs ):

        if self.model_path:
            def load_model(kl, model_path):
                start = time.time()
                print('loading model', model_path)
                kl.load(model_path)
                print('finished loading in %s sec.' % (str(time.time() - start)) )

            def load_weights(kl, weights_path):
                start = time.time()
                try:
                    print('loading model weights', weights_path)
                    kl.model.load_weights(weights_path)
                    print('finished loading in %s sec.' % (str(time.time() - start)) )
                except Exception as e:
                    print(e)
                    print('ERR>> problems loading weights', weights_path)

            def load_model_json(kl, json_fnm):
                start = time.time()
                print('loading model json', json_fnm)
                from tensorflow.python import keras
                try:
                    with open(json_fnm, 'r') as handle:
                        contents = handle.read()
                        kl.model = keras.models.model_from_json(contents)
                    print('finished loading json in %s sec.' % (str(time.time() - start)) )
                except Exception as e:
                    print(e)
                    print("ERR>> problems loading model json", json_fnm)

            #When we have a model, first create an appropriate Keras part
            kl = self.get_model()

            self.model_reload_cb = None

            if '.h5' in self.model_path or '.uff' in self.model_path or 'tflite' in self.model_path or '.pkl' in self.model_path:
                #when we have a .h5 extension
                #load everything from the model file
                load_model(kl, self.model_path)

                def reload_model(filename):
                    load_model(kl, filename)

                self.model_reload_cb = reload_model

            elif '.json' in self.model_path:
                #when we have a .json extension
                #load the model from there and look for a matching
                #.wts file with just weights
                load_model_json(kl, self.model_path)
                weights_path = self.model_path.replace('.json', '.weights')
                load_weights(kl, weights_path)

                def reload_weights(filename):
                    weights_path = filename.replace('.json', '.weights')
                    load_weights(kl, weights_path)

                self.model_reload_cb = reload_weights

            else:
                print("ERR>> Unknown extension type on model file!!")
                return

            outputs=['pilot/angle', 'pilot/throttle']

            if self.cfg.TRAIN_LOCALIZER:
                outputs.append("pilot/loc")

            self.model = kl
            self.vehicle.add(kl, inputs=inputs,
                outputs=outputs,
                run_condition='run_pilot')

    def build_post_pilot(self):
        #Choose what inputs should change the car.
        class DriveMode:
            def __init__(self, throttle_mult):
                self.throttle_mult = throttle_mult

            def run(self, mode,
                        user_angle, user_throttle,
                        pilot_angle, pilot_throttle):
                if mode == 'user':
                    return user_angle, user_throttle

                elif mode == 'local_angle':
                    return pilot_angle, user_throttle

                else:
                    return pilot_angle, pilot_throttle * self.throttle_mult

        self.vehicle.add(DriveMode(self.cfg.AI_THROTTLE_MULT),
              inputs=['user/mode', 'user/angle', 'user/throttle',
                      'pilot/angle', 'pilot/throttle'],
              outputs=['angle', 'throttle'])

        #to give the car a boost when starting ai mode in a race.
        aiLauncher = AiLaunch(self.cfg.AI_LAUNCH_DURATION, self.cfg.AI_LAUNCH_THROTTLE, self.cfg.AI_LAUNCH_KEEP_ENABLED)

        self.vehicle.add(aiLauncher,
            inputs=['user/mode', 'throttle'],
            outputs=['throttle'])

        if isinstance(self.controller, JoystickController):
            self.controller.set_button_down_trigger(self.cfg.AI_LAUNCH_ENABLE_BUTTON, aiLauncher.enable_ai_launch)

    def build_drive_train(self):
        #Drive train setup
        if self.cfg.DONKEY_GYM:
            pass

        elif self.cfg.DRIVE_TRAIN_TYPE == "SERVO_ESC":
            from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle

            steering_controller = PCA9685(self.cfg.STEERING_CHANNEL, self.cfg.PCA9685_I2C_ADDR, busnum=self.cfg.PCA9685_I2C_BUSNUM)
            steering = PWMSteering(controller=steering_controller,
                                            left_pulse=self.cfg.STEERING_LEFT_PWM,
                                            right_pulse=self.cfg.STEERING_RIGHT_PWM)

            throttle_controller = PCA9685(self.cfg.THROTTLE_CHANNEL, self.cfg.PCA9685_I2C_ADDR, busnum=self.cfg.PCA9685_I2C_BUSNUM)
            throttle = PWMThrottle(controller=throttle_controller,
                                            max_pulse=self.cfg.THROTTLE_FORWARD_PWM,
                                            zero_pulse=self.cfg.THROTTLE_STOPPED_PWM,
                                            min_pulse=self.cfg.THROTTLE_REVERSE_PWM)

            self.vehicle.add(steering, inputs=['angle'])
            self.vehicle.add(throttle, inputs=['throttle'])


        elif self.cfg.DRIVE_TRAIN_TYPE == "DC_STEER_THROTTLE":
            from donkeycar.parts.actuator import Mini_HBridge_DC_Motor_PWM

            steering = Mini_HBridge_DC_Motor_PWM(self.cfg.HBRIDGE_PIN_LEFT, self.cfg.HBRIDGE_PIN_RIGHT)
            throttle = Mini_HBridge_DC_Motor_PWM(self.cfg.HBRIDGE_PIN_FWD, self.cfg.HBRIDGE_PIN_BWD)

            self.vehicle.add(steering, inputs=['angle'])
            self.vehicle.add(throttle, inputs=['throttle'])


        elif self.cfg.DRIVE_TRAIN_TYPE == "DC_TWO_WHEEL":
            from donkeycar.parts.actuator import TwoWheelSteeringThrottle, Mini_HBridge_DC_Motor_PWM

            left_motor = Mini_HBridge_DC_Motor_PWM(self.cfg.HBRIDGE_PIN_LEFT_FWD, self.cfg.HBRIDGE_PIN_LEFT_BWD)
            right_motor = Mini_HBridge_DC_Motor_PWM(self.cfg.HBRIDGE_PIN_RIGHT_FWD, self.cfg.HBRIDGE_PIN_RIGHT_BWD)
            two_wheel_control = TwoWheelSteeringThrottle()

            self.vehicle.add(two_wheel_control,
                    inputs=['throttle', 'angle'],
                    outputs=['left_motor_speed', 'right_motor_speed'])

            self.vehicle.add(left_motor, inputs=['left_motor_speed'])
            self.vehicle.add(right_motor, inputs=['right_motor_speed'])

        elif self.cfg.DRIVE_TRAIN_TYPE == "SERVO_HBRIDGE_PWM":
            from donkeycar.parts.actuator import ServoBlaster, PWMSteering
            steering_controller = ServoBlaster(self.cfg.STEERING_CHANNEL) #really pin
            #PWM pulse values should be in the range of 100 to 200
            assert(self.cfg.STEERING_LEFT_PWM <= 200)
            assert(self.cfg.STEERING_RIGHT_PWM <= 200)
            steering = PWMSteering(controller=steering_controller,
                                            left_pulse=self.cfg.STEERING_LEFT_PWM,
                                            right_pulse=self.cfg.STEERING_RIGHT_PWM)


            from donkeycar.parts.actuator import Mini_HBridge_DC_Motor_PWM
            motor = Mini_HBridge_DC_Motor_PWM(self.cfg.HBRIDGE_PIN_FWD, self.cfg.HBRIDGE_PIN_BWD)

            self.vehicle.add(steering, inputs=['angle'])
            self.vehicle.add(motor, inputs=["throttle"])

    def build_recording(self):
        #Ai Recording
        if self.cfg.RECORD_DURING_AI and self.cfg.AUTO_RECORD_ON_THROTTLE:
            class AiRecordingCondition:
                '''
                return True when ai mode, otherwize respect user mode recording flag
                '''
                def run(self, mode, recording):
                    if mode == 'user':
                        return recording
                    return True

            self.vehicle.add(AiRecordingCondition(), inputs=['user/mode', 'recording'], outputs=['recording'])

        #add tub to save data

        inputs=['cam/image_array',
                'user/angle', 'user/throttle',
                'user/mode']

        types=['image_array',
               'float', 'float',
               'str']

        if self.cfg.TRAIN_BEHAVIORS:
            inputs += ['behavior/state', 'behavior/label', "behavior/one_hot_state_array"]
            types += ['int', 'str', 'vector']

        if self.cfg.HAVE_IMU:
            inputs += ['imu/acl_x', 'imu/acl_y', 'imu/acl_z',
                'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z']

            types +=['float', 'float', 'float',
               'float', 'float', 'float']

        if self.cfg.RECORD_DURING_AI:
            inputs += ['pilot/angle', 'pilot/throttle']
            types += ['float', 'float']

        if self.cfg.PUB_CAMERA_IMAGES:
            from donkeycar.parts.network import TCPServeValue
            from donkeycar.parts.image import ImgArrToJpg
            pub = TCPServeValue("camera")
            self.vehicle.add(ImgArrToJpg(), inputs=['cam/image_array'], outputs=['jpg/bin'])
            self.vehicle.add(pub, inputs=['jpg/bin'])

        if self.cfg.DONKEY_GYM:
            if self.cfg.SIM_RECORD_LOCATION:
                inputs += ['pos/pos_x', 'pos/pos_y', 'pos/pos_z', 'pos/speed', 'pos/cte']
                types  += ['float', 'float', 'float', 'float', 'float']
            if self.cfg.SIM_RECORD_GYROACCEL:
                inputs += ['gyro/gyro_x', 'gyro/gyro_y', 'gyro/gyro_z', 'accel/accel_x', 'accel/accel_y', 'accel/accel_z']
                types  += ['float', 'float', 'float', 'float', 'float', 'float']
            if self.cfg.SIM_RECORD_VELOCITY:
                inputs += ['vel/vel_x', 'vel/vel_y', 'vel/vel_z']
                types  += ['float', 'float', 'float']
            if self.cfg.SIM_RECORD_LIDAR:
                inputs += ['lidar/dist_array']
                types  += ['nparray']

        tub_path = TubNamer(path=self.cfg.DATA_PATH).create_tub_path()
        tub_writer = TubWriter(tub_path, inputs=inputs, types=types, metadata=self.meta)
        self.vehicle.add(tub_writer, inputs=inputs, outputs=["tub/num_records"], run_condition='recording')
        self.controller.set_tub(tub_writer) # Only used to call delete_last_n_records

    def user_notifications(self):
        self.controller.print_controls()

        if type(self.controller) is LocalWebController:
            print("You can now go to <your pi ip address>:8887 to drive your car.")
        elif isinstance(self.controller, JoystickController):
            print("You can now move your joystick to drive your car.")

    def start(self):
        self.vehicle.start(rate_hz=self.cfg.DRIVE_LOOP_HZ, max_loop_count=self.cfg.MAX_LOOPS)
