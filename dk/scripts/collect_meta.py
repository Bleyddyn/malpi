"""
Drive a donkey 2 car to collect data along with meta information.

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

import donkeycar as dk
from manage import drive

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
    

    drive(cfg, model_path=args['--model'], model_type=model_type, use_joystick=True, meta=meta)
