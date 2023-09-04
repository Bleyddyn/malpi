#! /usr/bin/env python3

"""
Test a PyTorch trained DonkeyCar model on one or more simulated tracks.
Optionally record data to a Tub file during the test.

Based on gym_test.py by Tawn Kramer
"""

from pathlib import Path
import logging

import gym
import gym_donkeycar
import numpy as np
import cv2
import torch

import donkeycar as dk
from donkeycar.parts.tub_v2 import TubWriter
from malpi.dk.drive import TubNamer
from malpi.dk.vae import VanillaVAE, SplitDriver, CombinedDriver, CombinedRNNDriver

#from scripts.train_tracks import DKDriverModule

NUM_EPISODES = 3
MAX_TIME_STEPS = 2000
MAX_LAPS = 3


# DonkeyCar Simulator environment names
env_list = [
    "donkey-warehouse-v0",
    "donkey-generated-roads-v0",
    "donkey-avc-sparkfun-v0",
    "donkey-generated-track-v0",
    "donkey-roboracingleague-track-v0",
    "donkey-waveshare-v0",
    "donkey-minimonaco-track-v0",
    "donkey-warren-track-v0",
    "donkey-circuit-launch-track-v0",
]

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
    vae = False

    if hasattr(learn, 'vae'):
        vae = True

    for episode in range(NUM_EPISODES):

        # Reset the environment
        obv = env.reset()
        learn.reset()

        laps = []
        lap_count = 0
        ep_reward = 0
        for t in range(MAX_TIME_STEPS):

            if vae:
                # Preprocess the observation, but only for vae based models
                obv = cv2.resize(obv, (128, 128) )
                obv = np.transpose(obv,(2,0,1))
                obv = obv.astype(np.float32) / 255.0
                obv = torch.from_numpy(obv).float().unsqueeze(0)
                action = learn.predict(obv)[0].detach().numpy()
            else:
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

"""
  {"track": "name",
    "laps": [[16.27978515625, 14.14013671875, 14.19970703125], [], [16.75, 14.39013671875, 14.36962890625]],
   "steps": [905, 97, 924],
 "rewards": [2153.973355292229, 167.68827394847173, 2480.692061715336]},
"""

def max_laps( results ):
    max_lap = 0
    for track in results.values():
        if not isinstance(track, dict):
            continue
        for test in track.get("laps",[]):
            for lap in test:
                max_lap = max(max_lap, lap)
    return max_lap

def by_laps( track, cutoff=30.0 ):
    track_sum = 0
    for test in track["lap_times"]:
        for i in test:
            track_sum += (cutoff - i)
        if len(test) < 3:
            track_sum -= (cutoff * (len(test) - 3))
    return track_sum

def print_results(results):
    # Outer loop is a list of drivers, inner loop is a dictionary of tracks
    # See malpi/dk/scripts/gym_test.py
    driver_name = None
    max_lap_time = max(max_laps( results ), 30.0)

    for key, value in results.items():
        if driver_name is None and 'driver' == key:
            driver_name = value
            print( f"Driver {driver_name}" )
        
        if key == 'driver':
            continue

        track_name = key
        lap_summary = by_laps( value, max_lap_time )
        rew_summary = sum(value.get('rewards',0))       

        print( f"     track: {track_name}" )
        print( f"      laps: {value['lap_times']}" )
        print( f"     steps: {value['steps']}" )
        print( f"   rewards: {value['rewards']}" )
        print( f"   summary: {lap_summary:.2f} {int(rew_summary)}" )
        print()

def main( env_name, model, model_path, vae_model, sim="sim_path", host="127.0.0.1", port=9091, record=False ):

    conf = get_conf(sim, host, port)

    if model.model.is_rnn:
        driver = CombinedRNNDriver(vae_model, model, no_var=True)
    else:
        driver = CombinedDriver(vae_model, model, no_var=True)

    results = { 'driver': model_path }

    if env_name == "all":
        for env_name in env_list:
            results[env_name] = test_track(env_name, conf, driver, model_path, record)

    else:
        results[env_name] = test_track(env_name, conf, driver, model_path, record, verbose=False)

    return results
