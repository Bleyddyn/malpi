#! /usr/bin/env python3

"""
Test a PyTorch trained DonkeyCar model on one or more simulated tracks.
Optionally record data to a Tub file during the test.

Based on gym_test.py by Tawn Kramer
"""

import argparse
import uuid
from pathlib import Path
import logging

import gym

import gym_donkeycar

from fastai.vision.all import *

import cv2

import donkeycar as dk
from donkeycar.parts.tub_v2 import TubWriter
from malpi.dk.drive import TubNamer
from malpi.dk.vae import VanillaVAE, SplitDriver, CombinedDriver

NUM_EPISODES = 3
MAX_TIME_STEPS = 2000
MAX_LAPS = 3


def test_track(env_name, conf, learn, model_path, record, vae_path=None, verbose=True):
    tub = None

    if record:
        meta = get_meta(model_path)
        tub = get_tub( "data", meta )

    env = gym.make(env_name, conf=conf)

    # make sure you have no track loaded
    exit_scene(env)

    results = simulate(env, learn, model_path, tub, vae_path, verbose=verbose)

    # exit the scene and close the env
    exit_scene(env)
    env.close()

    return results

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

def simulate(env, learn, model_path, tub=None, vae_path=None, verbose=True):

    lap_times = []
    rewards = []
    steps = [0] * NUM_EPISODES

    for episode in range(NUM_EPISODES):

        # Reset the environment
        obv = env.reset()

        laps = []
        lap_count = 0
        ep_reward = 0
        for t in range(MAX_TIME_STEPS):

            # Preprocess the observation, but only for vae based models
            #obv = cv2.resize(obv, (128, 128) )
            #obv = np.transpose(obv,(2,0,1))
            #obv = obv.astype(np.float32) / 255.0
            #if len(obv.shape) > 3:
            #    obv = torch.from_numpy(obv).float().unsqueeze(0)
            #action = learn.predict(obv)[0].detach().numpy()

            # Select an action
            action = learn.predict(obv)[0]

            # execute the action
            obv, reward, done, info = env.step(action)
            ep_reward += reward

            #if t % 100 == 0:
            #    print("t: {}  action: {}  reward: {}  done: {}  info: {}".format(t, action, reward, done, info))
            if tub is not None:
                tub.run( obv, 0.0, 0.0, "pilot/angle", action[0], action[1],
                    info['pos'][0], info['pos'][1], info['pos'][2], info['speed'], info['cte'],
                    info['vel'][0], info['vel'][1], info['vel'][2]
                    )

            if lap_count != info['lap_count']:
                laps.append( info['last_lap_time'] )
                lap_count = info['lap_count']
                #print( f"Lap {lap_count} time: {info['last_lap_time']}" )
                #print( f"Lap index: {env.viewer.handler.starting_line_index}" )

            steps[episode] = t

            if done:
                if verbose and info['hit'] != 'none':
                    print(f"Hit: {info['hit']} after {t} steps")
                break

            if lap_count >= MAX_LAPS:
                break

        if verbose:
            print( f"Lap times: {laps}" )

        if verbose and not done:
            print("Episode finished")

        lap_times.append( laps )
        rewards.append( ep_reward )

    if tub is not None:
        tub.close()

    return {"lap_times": lap_times, "rewards": rewards, "steps": steps}

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
        "max_cte": 100,
        "cam_config": {'img_h': 256, 'img_w': 256, 'img_d': 3},
        "cam_resolution": (256, 256, 3),
        "log_level": logging.WARNING
        }

def get_split_driver(vae_path, driver_path):
    driver = CombinedDriver(vae_path, driver_path, no_var=True)
    return driver

def main( env_name, model, vae_model=None, sim="sim_path", host="127.0.0.1", port=9091, record=False):

    conf = get_conf(sim, host, port)

    if vae_model is not None:
        driver = get_split_driver(vae_model, model)
    else:
        driver = load_learner(model)

    results = None

    if env_name == "all":
        for env_name in env_list:
            test_track(env_name, conf, driver, model, record)

    else:
        results = test_track(env_name, conf, driver, model, record, verbose=False)

    results['driver'] = model

    return results

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
    parser.add_argument("--vae", type=str, default=None, help="A pre-trained vae model.")
    parser.add_argument("--record", action='store_true', default=False, help="Record data to a tub file?")
    parser.add_argument(
        "--env_name", type=str, default="all", help="Name of donkey sim environment.", choices=env_list + ["all"]
    )

    args = parser.parse_args()

    conf = get_conf(args.sim, args.host, args.port)

    if args.vae is not None:
        driver = get_split_driver(args.vae, args.model)
    else:
        driver = load_learner(args.model)

    results = None

    if args.env_name == "all":
        for env_name in env_list:
            test_track(env_name, conf, driver, args.model, args.record)

    else:
        results = test_track(args.env_name, conf, driver, args.model, args.record)

    if results is not None:
        for idx in range(len(results["lap_times"])):
            print( f"Ep {idx+1}: {results['lap_times'][idx]} {results['rewards'][idx]}" )

    print("test finished")
