import keras
import numpy as np

def r_square_np(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    SS_res = np.sum([np.linalg.norm(y_true[i] - y_pred[i])**2 for i in range(0, y_true.shape[0])])
    SS_tot = np.sum([np.linalg.norm(y_true[i] - np.mean(y_true, axis=0))**2 for i in range(0, y_true.shape[0])])
    return ( 1.0 - SS_res/(SS_tot + 1e-5) )

def cosine_proximity(y_true, y_pred):
    from keras import backend as K
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -K.mean(y_true * y_pred, axis=-1)

def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def load_model(file, model, loss='mse', optimizer='adam', metrics=['accuracy']):
    model.load_weights(file)
    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
    return model

def get_autoencoder(n_layers, enc_activations, dec_activations, layer_sizes, dropout=None):
    input_x = keras.layers.Input(shape=(layer_sizes[0],))
    input_encoded = keras.layers.Input(shape=(layer_sizes[-1],))
    
    x = keras.layers.Dense(layer_sizes[0], activation=enc_activations[0])(input_x)
    for i in range(1, n_layers-1):
        x = keras.layers.Dense(layer_sizes[i], activation=enc_activations[i])(x)
        if dropout:
            x = keras.layers.Dropout(dropout)(x)
        
    encoded = keras.layers.Dense(layer_sizes[-1], activation=enc_activations[-1])(x)
    
    x = keras.layers.Dense(layer_sizes[-1], activation=dec_activations[0])(input_encoded)
    for i in range(0, n_layers-1):
        x = keras.layers.Dense(layer_sizes[i], activation=dec_activations[i])(x)
        if dropout:
            x = keras.layers.Dropout(dropout)(x)
            
    decoded = keras.layers.Dense(layer_sizes[0], activation=dec_activations[-1])(x)
    encoder = keras.models.Model(input_x, encoded, name="encoder")
    decoder = keras.models.Model(input_encoded, decoded, name="decoder")
    autoencoder = keras.models.Model(input_x, decoder(encoder(input_x)))
    
    return autoencoder, encoder, decoder