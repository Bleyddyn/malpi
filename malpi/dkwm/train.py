import os
import numpy as np

def load_tub_npz( dirs, base_dir="", max_images=None, verbose=False, aux_name=None, pilot=False, cte=False ):
    """ Load images from pre-processed tub.npz files.
        dirs should be a list of tub names
        load each: base_dir + tub_name + '.npz' (extension is added if needed)
        Add all images from npz['images']
    """
    images = []
    rewards = []
    aux = []
    actions = []
    ct_error = []
    total = 0
    for idx, tub in enumerate(dirs):
        if verbose:
            print( "Loading {}".format( tub ) )
        if not tub.endswith('.npz'):
            tub += '.npz'
        data = np.load( os.path.join( base_dir, tub ) )
        if aux_name is not None and aux_name not in data:
            if verbose:
                print( "  Missing auxiliary data: {}".format( aux_name ) )
            continue

#['obs', 'action', 'reward', 'done']
        img_key = "obs" if "obs" in data else "images"
        reward_key = "reward" if "reward" in data else "rewards"

        if verbose:
            print( "  {}".format( data[img_key].shape ) )
        images.append( data[img_key] )
        total += data[img_key].shape[0]

        if reward_key in data:
            rewards.append( data[reward_key] )
        if aux_name is not None:
            aux.append( data[aux_name] )
        if pilot:
            actions.append( data['actions'] )
        if cte:
            ct_error.append( data['cte'] )

        if max_images is not None and total > max_images:
            break

    images = np.concatenate( images, axis=0 ).astype(np.float32) / 255.0
    if len(rewards) > 0:
        rewards = np.concatenate( rewards, axis=0 )
    if len(aux) > 0:
        aux = np.concatenate( aux, axis=0 )
    if len(actions) > 0:
        actions = np.concatenate( actions, axis=0 )
    if len(ct_error) > 0:
        ct_error = np.concatenate( ct_error, axis=0 )

    if verbose:
        print( "Loaded {} images {}".format( images.shape[0], images.shape[1:] ) )

    output = {'images': images}
    if rewards.shape[0] > 0:
        output['rewards'] = rewards 
    if aux_name is not None:
        output[aux_name] = aux
    if pilot:
        output['actions'] = actions
    if cte:
        output['cte'] = ct_error

    return output

def generate_mdrnn_input_files( vae, dirs, base_dir="", batch_size=128, verbose=False, output_dir=None ):
    """ Generate data files for training the MDRNN.
        Read images from base_dir + dirs, encode all images using vae,
            save means and log_vars, actions, rewards and done flags.
        Output file name is same as inputs but with "_rnnin" appended.
    """

    if output_dir is None:
        output_dir = base_dir

    for idx, tub in enumerate(dirs):
        mus = []
        log_vars = []
        if verbose:
            print( "Loading {}".format( tub ) )
        ext = ""
        if not tub.endswith('.npz'):
            ext = '.npz'
        data = np.load( os.path.join( base_dir, tub + ext ) )

        img_key = "obs" if "obs" in data else "images"
        reward_key = "reward" if "reward" in data else "rewards"
        action_key = "action" if "action" in data else "actions"

        if reward_key not in data:
            if verbose:
                print( "   Skipping, no reward data" )
            continue
        if 'done' not in data:
            print( "   Missing done data" )
        if 'cte' not in data:
            print( "   Missing cte data" )
        images = data[img_key].astype(np.float32) / 255.0
        count = images.shape[0]
        for i in range(0, count, batch_size):
            first = i
            last = i + batch_size
            if last > count:
                last = count
            batch = images[first:last,:,:,:]
            mu, log_var = vae.encoder_mu_log_var.predict(batch)
            mus.append( mu )
            log_vars.append( log_var )
        mus = np.concatenate( mus, axis=0 )
        log_vars = np.concatenate( log_vars, axis=0 )
        rewards = data.get(reward_key, np.zeros( (mus.shape[0],1) ) )
        done = data.get( 'done', np.zeros( (mus.shape[0], 1) ).astype(np.uint8) )
        cte = data.get('cte', np.zeros( (mus.shape[0],1) ) )
        #done[-1] = 1 # Make the end of a tub file be the end of an episode
        if verbose:
            print( "   mu/log_var/actions: {}/{}/{}".format( mus.shape, log_vars.shape, data[action_key].shape ) )
        fname = os.path.join( output_dir, os.path.basename(tub) + "_rnnin.npz" )
        np.savez_compressed(fname, mu=mus, log_var=log_vars, action=data[action_key], reward = rewards, done = done, cte=cte)

def generate_starts( dirs, base_dir="", samples_per_tub=10, sample_from=0.5, verbose=False, output_dir=None ):
    """ Sample mu/log_var from mdrnn input files to use as starting points for the gym wrapper.
        Input files are base_dir + dirs + "_rnnin.npz".
        samples_per_tub: how many mu/log_var to sample from each tub file.
        sample_from: percentage of each tub to sample from, starting at zero.
        Randomly selected samples will be saved to base_dir + "starts.npz"
        TODO: Add an option to only save from the beginning of each tub file rather than random samples.
    """

    mu_list = []
    logvar_list = []

    for idx, tub in enumerate(dirs):
        fname = os.path.join( base_dir, os.path.basename(tub) + "_rnnin.npz" )
        data = np.load( fname )

        mu = data['mu']
        log_var = data['log_var']

        m_idxs = np.random.randint( 0, int(mu.shape[0] * sample_from), samples_per_tub )
        #var_idxs = np.random.randint( 0, int(log_var.shape[0] * sample_from), samples_per_tub )
        mu_list.append(mu[m_idxs])
        logvar_list.append(log_var[m_idxs])

    mu_list = np.concatenate(mu_list, axis=0 )
    logvar_list = np.concatenate(logvar_list, axis=0 )
    np.savez_compressed( os.path.join( output_dir, "starts.npz"), mu=mu_list, log_var=logvar_list)
    if verbose:
        print( "Saved {} starting mu/log_var's at {}".format( mu_list.shape[0], os.path.join( base_dir, "starts.npz")) )
