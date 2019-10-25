from sklearn.utils import shuffle
import numpy as np
from donkeycar.parts.augment import augment_image
from donkeycar.utils import load_scaled_image_arr
import keras

def vae_generator(cfg, data, batch_size, isTrainSet=True, min_records_to_train=1000, aug=False, aux=None):
    
    num_records = len(data)

    while True:

        batch_data = []

        keys = list(data.keys())
        keys = shuffle(keys)

        for key in keys:

            if not key in data:
                continue

            _record = data[key]

            if _record['train'] != isTrainSet:
                continue

            batch_data.append(_record)

            if len(batch_data) == batch_size:
                inputs_img = []
                aux_out = []

                for record in batch_data:
                    img_arr = None
                    #get image data if we don't already have it
                    if record['img_data'] is None:
                        img_arr = load_scaled_image_arr(record['image_path'], cfg)

                        if img_arr is None:
                            break
                        
                        if aug:
                            img_arr = augment_image(img_arr)

                        if cfg.CACHE_IMAGES:
                            record['img_data'] = img_arr
                    else:
                        img_arr = record['img_data']
                        
                    if aux is not None:
                        if aux in record['json_data']:
                            aux_out.append(record['json_data'][aux])
                        else:
                            print( "Missing aux data in: {}".format( record ) )
                            continue

                    if img_arr is None:
                        continue

                    inputs_img.append(img_arr)

                X = np.array(inputs_img).reshape(batch_size, cfg.IMAGE_H, cfg.IMAGE_W, cfg.IMAGE_DEPTH)
                y = X

                if aux is not None:
                    aux_out = keras.utils.to_categorical(aux_out, num_classes=7)
                    yield X, {'main_output': y, 'aux_output': aux_out}
                else:
                    yield X, y


                batch_data = []

if __name__ == "__main__":
    import argparse
    import donkeycar as dk
    from donkeycar.templates.train import collate_records, preprocessFileList
    from donkeycar.utils import gather_records

    parser = argparse.ArgumentParser(description='Test VAE data loader.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--aux', default="lanes", help='Name of the auxilliary data to use.')
    parser.add_argument('file', nargs='+', help='Text file with a list of tubs to train on.')

    args = parser.parse_args()

    try:
        cfg = dk.load_config()
    except FileNotFoundError:
        cfg = dk.load_config("config.py") # retry in the current directory
    tub_names = preprocessFileList( args.file )
    input_shape = (cfg.IMAGE_W, cfg.IMAGE_H, cfg.IMAGE_DEPTH)
# Code for multiple inputs: http://digital-thinking.de/deep-learning-combining-numerical-and-text-features-in-deep-neural-networks/
    aux_out = 0
    if args.aux is not None:
        aux_out = 7 # need to get number of aux outputs from data

    opts = { 'cfg' : cfg}
    opts['categorical'] = False
    opts['continuous'] = False

    gen_records = {}
    records = gather_records(cfg, tub_names, verbose=True)
    collate_records(records, gen_records, opts)

    train_gen = vae_generator(cfg, gen_records, cfg.BATCH_SIZE, isTrainSet=True, aug=False, aux=args.aux)
    for X, y in train_gen:
        print( "X  {} {}".format( type(X[0]), X[0].shape ) )
        img = y['main_output'][0]
        print( "main  {} min/max/avg: {}/{}/{}".format( img.shape, np.min(img), np.max(img), np.mean(img) ) )
        print( "aux   {}".format( y['aux_output'].shape ) )
        break
