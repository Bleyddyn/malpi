from sklearn.utils import shuffle
import donkeycar as dk
from donkeycar.parts.augment import augment_image
from donkeycar.utils import *
import keras

def vae_generator(save_best, opts, data, batch_size, isTrainSet=True, min_records_to_train=1000, aug=False, aux=None):
    
    cfg = opts['cfg']
    num_records = len(data)

    while True:

        batch_data = []

        keys = list(data.keys())

        keys = shuffle(keys)

        has_imu = opts.get('has_imu', False)
        has_bvh = opts.get('has_bvh', False)
        img_out = opts.get('img_out', False)
        vae_out = opts.get('vae_out', False)
        model_out_shape = opts['model_out_shape']
        model_in_shape = opts['model_in_shape']

        if img_out:
            import cv2

        for key in keys:

            if not key in data:
                continue

            _record = data[key]

            if _record['train'] != isTrainSet:
                continue

            if opts['continuous']:
                #in continuous mode we need to handle files getting deleted
                filename = _record['image_path']
                if not os.path.exists(filename):
                    data.pop(key, None)
                    continue

            batch_data.append(_record)

            if len(batch_data) == batch_size:
                inputs_img = []
                inputs_imu = []
                inputs_bvh = []
                angles = []
                throttles = []
                out_img = []
                aux_out = []

                for record in batch_data:
                    #get image data if we don't already have it
                    if record['img_data'] is None:
                        filename = record['image_path']
                        
                        img_arr = load_scaled_image_arr(filename, cfg)

                        if img_arr is None:
                            break
                        
                        if aug:
                            img_arr = augment_image(img_arr)

                        if cfg.CACHE_IMAGES:
                            record['img_data'] = img_arr
                    else:
                        img_arr = record['img_data']
                        
                    if img_out:
                        rz_img_arr = cv2.resize(img_arr, (127, 127)) / 255.0
                        out_img.append(rz_img_arr[:,:,0].reshape((127, 127, 1)))
                        
                    if has_imu:
                        inputs_imu.append(record['imu_array'])
                    
                    if has_bvh:
                        inputs_bvh.append(record['behavior_arr'])

                    if aux is not None:
                        if aux in record['json_data']:
                            aux_out.append(record['json_data'][aux])
                        else:
                            print( "Missing aux data in: {}".format( record ) )
                            continue

                    inputs_img.append(img_arr)
                    angles.append(record['angle'])
                    throttles.append(record['throttle'])

                if img_arr is None:
                    continue

                img_arr = np.array(inputs_img).reshape(batch_size,\
                    cfg.IMAGE_H, cfg.IMAGE_W, cfg.IMAGE_DEPTH)

                if has_imu:
                    X = [img_arr, np.array(inputs_imu)]
                elif has_bvh:
                    X = [img_arr, np.array(inputs_bvh)]
                elif vae_out:
                    X = img_arr / 255.0
                else:
                    X = [img_arr]

                if img_out:
                    y = [np.array(out_img), np.array(angles), np.array(throttles)]
                elif vae_out:
                    y = X
                elif model_out_shape[1] == 2:
                    y = [np.array([angles, throttles])]
                else:
                    y = [np.array(angles), np.array(throttles)]

            #batch_size=batch_size, epochs=epochs, validation_data=(x_val, {'main_output': x_val, 'aux_output': y_val}), shuffle=True, callbacks=[early_stop])
                if aux is not None:
                    aux_out = keras.utils.to_categorical(aux_out, num_classes=7)
                    yield X, {'main_output': y, 'aux_output': aux_out}
                else:
                    yield X, y


                batch_data = []

if __name__ == "__main__":
    import argparse
    from vae_model import KerasVAE
    from donkeycar.templates.train import collate_records, preprocessFileList

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
    kl = KerasVAE(input_shape=input_shape, z_dim=128, beta=1.0, aux=aux_out)

    opts = { 'cfg' : cfg}
    opts['categorical'] = False
    opts['continuous'] = False

    gen_records = {}
    records = gather_records(cfg, tub_names, verbose=True)
    collate_records(records, gen_records, opts)

    # These options should be part of the KerasPilot class
    opts['model_out_shape'] = (2, 1)
    opts['model_in_shape'] = input_shape
    opts['vae_out'] = True

    train_gen = vae_generator(None, opts, gen_records, cfg.BATCH_SIZE, isTrainSet=True, aug=False, aux=args.aux)
    for X, y in train_gen:
        print( "X  {} {}".format( type(X[0]), X[0].shape ) )
        print( "main  {}".format( y['main_output'][0].shape ) )
        print( "aux   {}".format( y['aux_output'].shape ) )
        break
