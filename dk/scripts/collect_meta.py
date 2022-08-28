"""
Drive a donkey car to collect data along with meta information.
"""

import argparse

import cv2
import numpy as np

from fastai.vision.all import *

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

class FastAiPilot(object):

    def __init__(self):
        self.learn = None

    def load(self, model_path):
        if torch.cuda.is_available():
            print("using cuda for torch inference")
            defaults.device = torch.device('cuda')
        else:
            print("cuda not available for torch inference")

        path = os.path.dirname(model_path)
        fname = os.path.basename(model_path)
        self.learn = load_learner(model_path)

    def run(self, img):
        img = (img * 255).astype(np.uint8)
        #print( f"Image: {type(img)}  Shape: {img.shape}  {np.min(img)} {np.max(img)}" )
        #img = pil2tensor(img, dtype=np.float32) # converts to tensor
        #img = Image(img) # Convert to fastAi Image - this class has "apply_tfms"

        pred = self.learn.predict(img)
        #print( f"Pred: {pred}" )
        steering = float(pred[0][0])
        throttle = float(pred[0][1])

        return steering, throttle

class VAEPilot(object):

    def __init__(self):
        self.vae = None
        self.learn = None

    def load(self, vae_path, model_path):
        if torch.cuda.is_available():
            print("using cuda for torch inference")
            defaults.device = torch.device('cuda')
        else:
            print("cuda not available for torch inference")

        #path = os.path.dirname(model_path)
        #fname = os.path.basename(model_path)
        self.vae = load_learner(vae_path)
        self.learn = load_learner(model_path)

    def run(self, img):
        img = (img * 255).astype(np.uint8)
        #print( f"Image: {type(img)}  Shape: {img.shape}  {np.min(img)} {np.max(img)}" )
        #img = pil2tensor(img, dtype=np.float32) # converts to tensor
        #img = Image(img) # Convert to fastAi Image - this class has "apply_tfms"

        _, _, mu, log_var = self.vae.forward(img)

        pred = self.learn.predict(mu, log_var)
        #print( f"Pred: {pred}" )
        steering = float(pred[0][0])
        throttle = float(pred[0][1])

        return steering, throttle

class MyDriver(DefaultDriver):
    def __init__(self, cfg, model_path=None, vae_path=None, use_joystick=False, model_type=None, camera_type='single', meta=[] ):
        super().__init__(cfg, model_path=model_path, use_joystick=use_joystick, model_type=model_type, camera_type=camera_type, meta=meta )
        self.vae_path = vae_path

    def build_pre_pilot(self):
        self.build_pre_pilot_conditions()
        inf_input = 'cam/normalized/cropped'
        self.vehicle.add(ImageResize(self.cfg, (self.cfg.IMAGE_MODEL_W, self.cfg.IMAGE_MODEL_H)),
            inputs=['cam/image_array'],
            outputs=[inf_input],
            run_condition='run_pilot')
        return [inf_input]

    def build_pilot(self, inputs):
        if self.model_path is not None:
            outputs=['pilot/angle', 'pilot/throttle']

            if self.model_type == 'vae':
                self.model = VAEPilot()
                self.model.load(self.vae_path, self.model_path)
            else:
                self.model = FastAiPilot()
                self.model.load(self.model_path)
            self.vehicle.add(self.model, inputs=inputs,
                outputs=outputs,
                run_condition='run_pilot')

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

def getOptions():
    args = argparse.ArgumentParser(description='Collect drive data.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args.add_argument('--type', type=str, default='fastai', help='Type of model to use: fastai or vae')
    args.add_argument('--model', type=str, default=None, help='path to model')
    args.add_argument('--vae', type=str, default=None, help='path to vae model')
    args.add_argument('--cfg', nargs='*', type=str,
            default=['JOYSTICK_MAX_THROTTLE', 'JOYSTICK_STEERING_SCALE'],
            help='Config entries to include in meta information')
    args.add_argument('--location', type=str, default='UCSD Track 2', help='Drive location')
    args.add_argument('--task', type=str, default='Race', help='E.g. Race, Train')
    args.add_argument('--driver', type=str, default='Andrew')

    return args.parse_args()

if __name__ == "__main__":
    args = getOptions()

    try:
# This will look in the directory containing this script, not necessarily the current dir
        cfg = dk.load_config()
    except FileNotFoundError:
        cfg = dk.load_config("config.py")
    
    cfg_set = set(args.cfg)
    cfg_set.add('JOYSTICK_MAX_THROTTLE')
    cfg_set.add('JOYSTICK_STEERING_SCALE')

    meta = {}
    for entry in cfg_set:
        if hasattr(cfg, entry):
            meta[entry] = getattr(cfg, entry)
        else:
            print( "Invalid config key: {}".format( entry ) )

    meta["location"] = args.location
    meta["task"] = args.task
    meta["driver"] = args.driver
    metal = list(meta.items()) # convert to a list of tuples so tub v2 can parse them

    vehicle = MyDriver(cfg, model_path=args.model, vae_path=args.vae, model_type=args.type, use_joystick=cfg.USE_JOYSTICK_AS_DEFAULT, meta=metal)
    vehicle.print()
    vehicle.start()
