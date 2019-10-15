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
from donkeycar.drive import DefaultDriver
from donkeycar.utils import normalize_and_crop

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
            img_arr = normalize_and_crop(img_arr, self.cfg)

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
    
    vehicle = MyDriver(cfg, model_path=args['--model'], model_type=model_type, use_joystick=True, meta=meta)
    vehicle.start()
