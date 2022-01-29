"""
Drive a donkey 3 car to collect data along with meta information.
Requires the drive function of manage.py to return a vehicle object.

Usage:
    collect_meta.py [--location=<loc>] [--task=<task>] [--driver=<driver>] [<cfg>...] [--model=<model>] [--type=(linear|categorical|rnn|imu|behavior|3d|localizer|latent)]


Options:
    -h --help          Show this screen.
    --location=<loc>   Drive location [default: UCSD Track 2].
    --task=<task>      Task [default: Race].
    --driver=<driver>  Driver [default: Andrew].
    <cfg>              Config entries to include in meta information [default: ['JOYSTICK_MAX_THROTTLE', 'JOYSTICK_STEERING_SCALE']].
"""
from docopt import docopt

import cv2
import numpy as np

import donkeycar as dk
from malpi.dk.drive import DefaultDriver, RecordTracker

class ImageResize():
    """ ImageResize does the same pre-processing as ImgPreProcess in the drive function,
        plus it will resize the final image to the given dimensions.
    """

    def __init__(self, cfg, ai_dim):
        """ ai_dim = (width, height) dimensions of the input layer of the AI. """
        self.cfg = cfg
        self.dim = ai_dim

    def run(self, img_arr):
        if img_arr is None:
            return None
        try:
            img_arr = img_arr.astype(np.float32) / 255.0

            # Should use cubic for upsampling, but INTER_AREA is best for downsampling
            # Linear is default and seems like a good compromise and it's fast
            # See: http://tanbakuchi.com/posts/comparison-of-openv-interpolation-algorithms/
            return cv2.resize(img_arr, self.dim)
        except Exception as ex:
            print( "ImageResize error: {}".format( ex ) )
            raise

    def shutdown(self):
        pass

class MyDriver(DefaultDriver):
    def __init__(self, cfg, model_path=None, use_joystick=False, model_type=None, camera_type='single', meta=[] ):
        super().__init__(cfg, model_path=model_path, use_joystick=use_joystick, model_type=model_type, camera_type=camera_type, meta=meta )

    def build_pre_pilot(self):
        self.build_pre_pilot_conditions()
        inf_input = 'cam/normalized/cropped'
        self.vehicle.add(ImageResize(self.cfg, (self.cfg.IMAGE_MODEL_W, self.cfg.IMAGE_MODEL_H)),
            inputs=['cam/image_array'],
            outputs=[inf_input],
            run_condition='run_pilot')
        return [inf_input]

    def build_displays(self):
        class LedSimpleLogic:
            def __init__(self):
                pass

            def run(self, recording):
                #returns a blink rate. 0 for off. -1 for on. positive for rate.
                if recording:
                    return -1 #solid on

                return 0

        rec_tracker_part = RecordTracker(self.cfg.REC_COUNT_ALERT, self.cfg.REC_COUNT_ALERT_CYC, self.cfg.RECORD_ALERT_COLOR_ARR)
        self.vehicle.add(rec_tracker_part, inputs=["tub/num_records"], outputs=['records/alert'])

        if self.cfg.HAVE_RGB_LED and not self.cfg.DONKEY_GYM:
            from donkeycar.parts.led_status import RGB_LED
            led = RGB_LED(self.cfg.LED_PIN_R, self.cfg.LED_PIN_G, self.cfg.LED_PIN_B, self.cfg.LED_INVERT)
            led.set_rgb(self.cfg.LED_R, self.cfg.LED_G, self.cfg.LED_B)

            self.vehicle.add(LedSimpleLogic(), inputs=['recording'], outputs=['led/blink_rate'])
            self.vehicle.add(led, inputs=['led/blink_rate'])

if __name__ == "__main__":
    args = docopt(__doc__)

    try:
# This will look in the directory containing this script, not necessarily the current dir
        cfg = dk.load_config()
    except FileNotFoundError:
        cfg = dk.load_config("config.py")
    
    args['<cfg>'].extend( ['JOYSTICK_MAX_THROTTLE', 'JOYSTICK_STEERING_SCALE'] )

    meta = {}
    for entry in args['<cfg>']:
        if hasattr(cfg, entry):
            meta[entry] = getattr(cfg, entry)
        else:
            print( "Invalid config key: {}".format( entry ) )

    meta["location"] = args['--location']
    meta["task"] = args['--task']
    meta["driver"] = args['--driver']
    metal = list(meta.items()) # convert to a list of tuples so tub v2 can parse them

    model_type = args['--type']
    
    vehicle = MyDriver(cfg, model_path=args['--model'], model_type=model_type, use_joystick=cfg.USE_JOYSTICK_AS_DEFAULT, meta=metal)
    vehicle.start()
