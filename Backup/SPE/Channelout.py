from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np




class Channelout(tf.keras.layers.Layer):
    '''
    Channel-out activation as described in the original paper by Wang and JaJa.
    "Winning" channel determined by max if arg_abs_max is False. If True,
    it is determined by the argmax of the absolute values.
    '''
    
    def __init__(self, pool_size, arg_abs_max=False, **kwargs):
        super().__init__(**kwargs)
        self._pool_size = pool_size
        self.arg_abs_max = arg_abs_max

    def call(self, x):
        x = tf.convert_to_tensor(x)
        
        shape = K.shape(x)#[0]
        #shape = x.get_shape().as_list()
        
        if shape[0] is None:
            shape[0] = tf.shape(x)[0]
        
        #print (shape)
        
        # Group neurons together by pool size.
        if len(x.get_shape().as_list())>2:
            raw_vals = tf.reshape(tensor=x, 
                                  shape=(shape[0],
                                         shape[1], 
                                         shape[2]//self._pool_size, 
                                         self._pool_size))
        else:
            raw_vals = tf.reshape(tensor=x, 
                                  shape=(shape[0], 
                                         shape[1]//self._pool_size, 
                                         self._pool_size))
        
        # Absolute max function requires argmax of absolute values.
        if self.arg_abs_max: arg_vals = tf.math.abs(raw_vals)
        else: arg_vals = raw_vals
        
        # Take argmax of pooled neurons.
        max_inds = tf.math.argmax(arg_vals, axis=-1)
        
        # Create a binary mask to zero-out non-max values in each pool.
        mask = tf.one_hot(max_inds,self._pool_size)
        activation = tf.math.multiply(raw_vals, mask)
        
        activation = tf.reshape(tensor=activation, shape=shape)
        
        return activation


















