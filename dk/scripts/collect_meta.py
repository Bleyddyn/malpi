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

import donkeycar as dk
from donkeycar.parts.keras import KerasPilot
from donkeycar.templates.complete import drive

class ImageResize():

    def __init__(self, dim):
        self.dim = dim

    def run(self, img_arr):
        if img_arr is None:
            return None
        try:
            # Should use cubic for upsampling, but INTER_AREA is best for downsampling
            # Linear is default and seems like a good compromise and it's fast
            # See: http://tanbakuchi.com/posts/comparison-of-openv-interpolation-algorithms/
            return cv2.resize(img_arr, self.dim)
        except Exception as ex:
            print( "ImageResize error: {}".format( ex ) )
            return None

    def shutdown(self):
        pass


if __name__ == "__main__":
    args = docopt(__doc__)

    try:
# This will look in the directory containing this script, not necessarily the current dir
        cfg = dk.load_config()
    except FileNotFoundError:
        cfg = dk.load_config("config.py")
    
    args['<cfg>'].extend( ['JOYSTICK_MAX_THROTTLE', 'JOYSTICK_STEERING_SCALE'] )

    res = {}
    for entry in args['<cfg>']:
        if hasattr(cfg, entry):
            res[entry] = getattr(cfg, entry)
        else:
            print( "Invalid config key: {}".format( entry ) )

    meta = []
    for k,v in res.items():
        meta.append( "{}:{}".format( k, v ) )

    meta.append( "location:{}".format( args['--location'] ) )
    meta.append( "task:{}".format( args['--task'] ) )
    meta.append( "driver:{}".format( args['--driver'] ) )

    model_type = args['--type']
    

    vehicle = drive(cfg, model_path=args['--model'], model_type=model_type, use_joystick=True, meta=meta, start_vehicle=False)

# Add my ImageResize part here
    if hasattr(cfg, 'IMAGE_MODEL_W') and hasattr(cfg, 'IMAGE_MODEL_H'):
        if (cfg.IMAGE_W != cfg.IMAGE_MODEL_W) or (cfg.IMAGE_H != cfg.IMAGE_MODEL_H):
            # Add an image resize part and change the model's input to be the resized image.
            # For use when recording/saving one image size, but the model was trained on a different size.
            vehicle.add( ImageResize( (cfg.IMAGE_MODEL_W, cfg.IMAGE_MODEL_H) ),
                   inputs=['cam/image_array'],
                   outputs=['cam/resized_image'] )
            for part in vehicle.parts:
                if isinstance(part["part"], KerasPilot):
                    inputs = part["inputs"]
                    if 'cam/image_array' in inputs:
                        inputs[inputs.index('cam/image_array')] = 'cam/resized_image'
                    else:
                        inputs[inputs.index('cam/normalized/cropped')] = 'cam/resized_image'
                    part["inputs"] = inputs

    vehicle.start(rate_hz=cfg.DRIVE_LOOP_HZ, max_loop_count=cfg.MAX_LOOPS)
