#! /usr/bin/env python3

"""
Test a PyTorch trained DonkeyCar model on one or more simulated tracks.
Optionally record data to a Tub file during the test.

Based on gym_test.py by Tawn Kramer
"""

import os
# suppress warning from numexpr module
os.environ['NUMEXPR_MAX_THREADS'] = '16'

import argparse
import json

# Import DonkeyCar, suppressing it's annoying banner
from contextlib import redirect_stdout
with redirect_stdout(open(os.devnull, "w")):
    import donkeycar as dk

from malpi.dk.test import get_conf, main, env_list, print_results
from malpi.dk.lit import LitVAE, LitVAEWithAux, DKDriverModule

if __name__ == "__main__":

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

    vae_model = LitVAE.load_from_checkpoint(args.vae)
    driver_model = DKDriverModule.load_from_checkpoint(args.model)
    vae_model.to("cpu")
    driver_model.to("cpu")
    #vae_model.eval()

    results = main(args.env_name, driver_model, args.model, vae_model,
            sim=args.sim, host=args.host, port=args.port, record=args.record)

#    with open('results_temp.json', 'w') as f:
#        json.dump(results, f)

    print_results(results)
    print("Finished Testing")
