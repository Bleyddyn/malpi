from sklearn.utils import shuffle
from donkeycar.parts.augment import augment_image
from donkeycar.utils import *
from donkeycar.parts.keras import KerasLinear, KerasIMU,\
     KerasCategorical, KerasBehavioral, Keras3D_CNN,\
     KerasRNN_LSTM, KerasLatent
from vae_model import KerasVAE

def generator(save_best, opts, data, batch_size, isTrainSet=True, min_records_to_train=1000, aug=False, aux=None):
    
    cfg = opts['cfg']
    num_records = len(data)

    while True:

        if isTrainSet and opts['continuous']:
            '''
            When continuous training, we look for new records after each epoch.
            This will add new records to the train and validation set.
            '''
            records = gather_records(cfg, tub_names, opts)
            if len(records) > num_records:
                collate_records(records, data, opts)
                new_num_rec = len(data)
                if new_num_rec > num_records:
                    print('picked up', new_num_rec - num_records, 'new records!')
                    num_records = new_num_rec
                    if save_best is not None:
                        save_best.reset_best()
            if num_records < min_records_to_train:
                print("not enough records to train. need %d, have %d. waiting..." % (min_records_to_train, num_records))
                time.sleep(10)
                continue

        batch_data = []

        keys = list(data.keys())

        keys = shuffle(keys)

        kl = opts['keras_pilot']

        if type(kl.model.output) is list:
            model_out_shape = (2, 1)
        else:
            model_out_shape = kl.model.output.shape

        if type(kl.model.input) is list:
            model_in_shape = (2, 1)
        else:    
            model_in_shape = kl.model.input.shape

        has_imu = type(kl) is KerasIMU
        has_bvh = type(kl) is KerasBehavioral
        img_out = type(kl) is KerasLatent
        vae_out = type(kl) is KerasVAE
        
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
                    X = [img_arr / 255.0]
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

                if aux is not None:
                    yield X, y, aux_out
                else:
                    yield X, y


                batch_data = []
