#! /usr/bin/env python3

"""
Test a PyTorch trained DonkeyCar model on one or more simulated tracks.
Optionally record data to a Tub file during the test.

Based on gym_test.py by Tawn Kramer
"""

import argparse
import uuid
from pathlib import Path

import gym

import gym_donkeycar

from fastai.vision.all import *

import donkeycar as dk
from donkeycar.parts.tub_v2 import TubWriter
from malpi.dk.drive import TubNamer

NUM_EPISODES = 3
MAX_TIME_STEPS = 1000


def test_track(env_name, conf, learn, model_path, record):
    env = gym.make(env_name, conf=conf)

    # make sure you have no track loaded
    exit_scene(env)

    simulate(env, learn, model_path, record)

    # exit the scene and close the env
    exit_scene(env)
    env.close()


def select_action(env):
    return env.action_space.sample()  # taking random action from the action_space

def get_meta(model_path):
    try:
# This will look in the directory containing this script, not necessarily the current dir
        cfg = dk.load_config()
    except FileNotFoundError:
        cfg = dk.load_config("config.py")

    cfg_items = ['JOYSTICK_MAX_THROTTLE', 'JOYSTICK_STEERING_SCALE', 'DONKEY_GYM_ENV_NAME']

    meta = {}
    for entry in cfg_items:
        if hasattr(cfg, entry):
            meta[entry] = getattr(cfg, entry)
        else:
            print( "Invalid config key: {}".format( entry ) )

    meta["location"] = 'sim'
    meta["task"] = 'Train'
    meta["driver"] = Path(model_path).name
    metal = list(meta.items()) # convert to a list of tuples so tub v2 can parse them
    return metal

def get_tub( base_path, meta ):
    inputs=['cam/image_array',
            'user/angle', 'user/throttle',
            'user/mode']

    types=['image_array',
           'float', 'float',
           'str']

    inputs += ['pilot/angle', 'pilot/throttle']
    types += ['float', 'float']

    inputs += ['pos/pos_x', 'pos/pos_y', 'pos/pos_z', 'pos/speed', 'pos/cte']
    types  += ['float', 'float', 'float', 'float', 'float']
    inputs += ['vel/vel_x', 'vel/vel_y', 'vel/vel_z']
    types  += ['float', 'float', 'float']

    tub_path = TubNamer(path=base_path).create_tub_path()
    tub_writer = TubWriter(tub_path, inputs=inputs, types=types, metadata=meta)
    return tub_writer

def simulate(env, learn, model_path, record):

    if record:
        meta = get_meta(model_path)
        tub = get_tub( "data", meta )

    for episode in range(NUM_EPISODES):

        # Reset the environment
        obv = env.reset()
        print( f"{type(obv)} {obv.shape}" )
        for t in range(MAX_TIME_STEPS):

            # Select an action
            action = learn.predict(obv)[0]                                            

            # execute the action
            obv, reward, done, info = env.step(action)

            if record:
                tub.run( obv, 0.0, 0.0, "pilot/angle", action[0], action[1],
                    info['pos'][0], info['pos'][1], info['pos'][2], info['speed'], info['cte'],
                    info['vel'][0], info['vel'][1], info['vel'][2]
                    )

            if done:
                print("done w episode.", info)
                break

    if record:
        tub.close()

def exit_scene(env):
    env.viewer.exit_scene()


def get_conf(sim, host, port):
    return {
        "exe_path": sim,
        "host": host,
        "port": port,
        "body_style": "donkey",
        "body_rgb": (256, 128, 128),
        "car_name": "torch",
        "font_size": 100,
        "start_delay": 1,
        "max_cte": 40,
        "cam_config": {'img_h': 256, 'img_w': 256, 'img_d': 3},
        "cam_resolution": (256, 256, 3)
        }


if __name__ == "__main__":

    # Initialize the donkey environment
    # where env_name one of:
    env_list = [
        "donkey-warehouse-v0",
        "donkey-generated-roads-v0",
        "donkey-avc-sparkfun-v0",
        "donkey-generated-track-v0",
        "donkey-roboracingleague-track-v0",
        "donkey-waveshare-v0",
        "donkey-minimonaco-track-v0",
        "donkey-warren-track-v0",
        "donkey-thunderhill-track-v0",
        "donkey-circuit-launch-track-v0",
    ]

    parser = argparse.ArgumentParser(description="Test a PyTorch trained DonkeyCar model on one or more simulated tracks.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--sim",
        type=str,
        default="sim_path",
        help="Path to unity simulator. May be left at default if you would like to start the sim on your own.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to use for tcp.")
    parser.add_argument("--port", type=int, default=9091, help="Port to use for tcp.")
    parser.add_argument("--model", type=str, default="models/export.pkl", help="PyTorch trained model.")
    parser.add_argument("--record", action='store_true', default=False, help="Record data to a tub file?")
    parser.add_argument(
        "--env_name", type=str, default="all", help="Name of donkey sim environment.", choices=env_list + ["all"]
    )

    args = parser.parse_args()

    conf = get_conf(args.sim, args.host, args.port)

    learn = load_learner(args.model)

    if args.env_name == "all":
        for env_name in env_list:
            test_track(env_name, conf, learn, args.model, args.record)

    else:
        test_track(args.env_name, conf, learn, args.model, args.record)

    print("test finished")
