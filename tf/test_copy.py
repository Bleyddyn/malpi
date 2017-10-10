import keras
import model_keras

def printLayers( model ):
    for layer in model.layers:
        print( layer.name )
        w = layer.get_weights()
        if w is not None:
            if isinstance(w,list):
                for aw in w:
                    if hasattr(aw,'shape'):
                        print( aw.shape )
                    elif isinstance(aw,list):
                        print( len(aw) )
            elif hasattr(w,'shape'):
                print( w.shape )

model = keras.models.load_model('best_model.h5')
#model.save_weights('best_model_weights.h5')

#layer_dict = dict([(layer.name, layer) for layer in model.layers])
#weights = layer_dict['some_name'].get_weights()

num_actions = 5
input_dim = (120,120,3)

model2 = model_keras.make_model_lstm( num_actions, input_dim, batch_size=1, timesteps=1 )
model2.load_weights( 'best_model_weights.h5' )

#names = ( "Conv", "lstm", "Output" )
#for layer in model2.layers:
#    if layer.name.lower().startswith(names):
#        w = layer_dict[layer.name]
#        if w is not None:
#            layer.set_weights( w )

model.summary()
model2.summary()
