"""
Return one or more config entries in a format for use by 'manage.py drive'

Usage:
    get_config_entry.py <key>...


Options:
    -h --help        Show this screen.
    <key>            One or more config keys.

"""
import os
from docopt import docopt

import donkeycar as dk

if __name__ == "__main__":
    args = docopt(__doc__)

    try:
# This will look in the directory containing this script, not necessarily the current dir
        cfg = dk.load_config()
    except FileNotFoundError:
        cfg = dk.load_config("config.py")
    
#JOYSTICK_MAX_THROTTLE = 0.3
#JOYSTICK_STEERING_SCALE = 1.0
    res = {}
    for entry in args['<key>']:
        if hasattr(cfg, entry):
            res[entry] = getattr(cfg, entry)

    out = ""
    for k,v in res.items():
        out += " --meta=\"{}:{}\"".format( k, v )

    print(out)
