import json
import pickle
from sklearn.utils import shuffle
import donkeycar as dk
from donkeycar.parts.augment import augment_image
from donkeycar.parts.datastore import Tub
from donkeycar.utils import *

# This is copied from sample.py. Needs to be moved to a library
def model_meta( model ):
    """ Check a model to see if has auxiliary outputs, if so return the number of outputs. """
    aux = None
    z_dim = None
    dropout = None
    fname = model[:-10] + "model.json"
    try:
        with open(fname,'r') as f:
            json_str = f.read()
            data = json.loads(json_str)
            layers = data["config"]["layers"]
            for l in layers:
                if l.get("name","") == "aux_output":
                    aux = l["config"]["units"]
                elif l.get("name","") == "z_mean":
                    z_dim = l["config"]["units"]
                elif l.get("name","").startswith("SpatialDropout_"):
                    # e.g. SpatialDropout_0.4_1
                    dropout = float( l.get("name","").split("_")[1])
    except Exception as ex:
        print( "Failed to read meta data: {}".format( ex ) )
        pass

    return z_dim, dropout, aux


if __name__ == "__main__":
    import argparse
    from vae_model import KerasVAE
    from donkeycar.templates.train import collate_records, preprocessFileList

    parser = argparse.ArgumentParser(description='Generate training data for a WorldModels RNN.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('vae', help='Path to the weights of a trained VAE.')
    parser.add_argument('file', nargs='+', help='Text file with a list of tubs to preprocess.')

    args = parser.parse_args()

    try:
        cfg = dk.load_config()
    except FileNotFoundError:
        cfg = dk.load_config("config.py") # retry in the current directory
    tub_names = preprocessFileList( args.file )

    input_shape = (cfg.IMAGE_W, cfg.IMAGE_H, cfg.IMAGE_DEPTH)
    z_dim, dropout, aux = model_meta(args.vae)
    if aux is None:
        aux = 0

    kl = KerasVAE(input_shape=input_shape, z_dim=z_dim, aux=aux, dropout=dropout)
    kl.set_weights(args.vae)
    kl.compile()

    data = []
    tubs = gather_tubs(cfg, tub_names)
    for idx, tub in enumerate(tubs, start=1):
        print( "Starting {} of {} tubs".format( idx, len(tubs) ) )
        try:
            st_time = tub.meta['start']
        except:
            continue

        rpaths = tub.gather_records()
        for record_path in rpaths:
            try:
                with open(record_path, 'r') as fp:
                    json_data = json.load(fp)
            except:
                continue

            basepath = os.path.dirname(record_path)
            image_filename = json_data["cam/image_array"]
            image_path = os.path.join(basepath, image_filename)
            img_arr = load_scaled_image_arr(image_path, cfg)
            angle = float(json_data['user/angle'])
            throttle = float(json_data["user/throttle"])
            rtime = st_time + float(json_data["milliseconds"]) / 1000.0
            z = kl.encoder.predict( np.array([img_arr]) )
            record = np.insert( z, 0, [rtime, angle, throttle])
            data.append(record)

    np.save( 'rnn_data', data )
    print( "Saved {} records to rnn_data.np".format( len(data) ) )
    print( " shape: {}".format( data[0].shape ) )
