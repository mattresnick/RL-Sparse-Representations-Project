import os
os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout, ReLU
from Channelout import Channelout
from tensorflow_addons.layers import Maxout
from SpecDropout import SpecDropout, SpecAlphaDropout
import numpy as np
import DistributionalRegularizers as DRS

tf.random.set_seed(3953)

class ActorCritic():
    '''
    Class for both actor and critic objects.
    
    Parameters:
        - actor_type (Boolean): Determines whether the object will pertain to
          the actor (True) or critic (False).
        - state_shape (Integer): Number of state dimensions.
        - action_shape (Integer): Number of action dimensions.
        - action_scale (Integer or Float): Scale factor for action dimension(s).
        - f_layer_init (Initialization object): Final layer kernel initialization,
          if applicable.
        - tau (Float): Soft target update parameter.
        - lr (Float): Learning rate for optimizer.
        - layer_sizes (List): List of number of neurons per hidden layer.
        - activation (String): Name of activation function to be used.
        - pool_size (Integer): Neuron pooling size to be used for maxout/channelout
        - dropout_rate (Float): Probability of neuron being dropped for DO layer.
        - use_bn (Boolean): If True, use batchnorm layers.
        - use_do (Boolean): If True, use dropout layers layers.
        - DR (Boolean): If True, use Distributional Regularizer setup.
    
    Non-Trivial Attributes:
        - opt: Network parameter optimizer for training.
        - net: Actor or Critic network.
        - target_net: Target network for Actor or Critic.
        - activation_layers: List storing activation function layers.
        - dropout_layers: List storing dropout layers.
    '''
    
    def __init__(self, actor_type, state_shape, action_shape, action_scale, tau, 
                 lr, layer_sizes=[480,360], activation='relu', pool_size=2, 
                 dropout_rate=0.2, use_bn=True, use_do=True, DR=False):
        
        self.actor_type = actor_type
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.action_scale = action_scale
        self.tau = tau
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.use_do = use_do
        self.num_hid_layers = len(layer_sizes)
        
        self.opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.f_layer_init = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
        
        
        
        
        if self.actor_type:
            self.net = ActorModel(state_shape, action_shape, action_scale, activation, layer_sizes, 
                                  pool_size, dropout_rate, use_bn, use_do, DR)
            self.target_net = ActorModel(state_shape, action_shape, action_scale, activation, 
                                  layer_sizes, pool_size, dropout_rate, use_bn, use_do, DR)
        else:
            self.net = self.createApproximator()
            self.target_net = self.createApproximator()
        
        
        
        
        
    
    def createApproximator(self):
        
        # Layer raw output -> Channelout -> Batchnorm -> Dropout
        
        size_mult = 1.0
        
        if self.actor_type:
            state = Input(shape=(self.state_shape,))
            
            actor_layers = state
            for i,h_nodes in enumerate(self.layer_sizes):
                actor_layers = self.dense_layers[i](actor_layers)
                actor_layers = self.activation_layers[i](actor_layers)
                if self.use_bn: actor_layers = BatchNormalization()(actor_layers)
                if self.use_do: actor_layers = self.dropout_layers[i](actor_layers)
                self.store_activation[i] = actor_layers
            
            actor_layers = self.dense_layers[-1](actor_layers)*self.action_scale
            
            return tf.keras.Model(state, actor_layers)
        
        else:
            state = Input(shape=(self.state_shape))
            action = Input(shape=(self.action_shape))
            
            state_layers = Dense(int(400*size_mult),activation='relu')(state)
            state_layers = BatchNormalization()(state_layers)
            state_layers = Dense(int(300*size_mult),activation='relu')(state_layers)
            state_layers = BatchNormalization()(state_layers)
            
            action_layers = Dense(int(300*size_mult),activation='relu')(action)
            action_layers = BatchNormalization()(action_layers)
            
            layer_merge = Concatenate()([state_layers, action_layers])
            layer_merge = Dense(int(300*size_mult),activation='relu',kernel_regularizer='l2')(layer_merge)
            out_layer = Dense(1, kernel_initializer=self.f_layer_init)(layer_merge)
            
            return tf.keras.Model([state, action], out_layer)
    
    
    def createApproximatorSelu(self):
        
        size_mult = 1.0
        
        if self.actor_type:
            state = Input(shape=(self.state_shape,))
            
            actor_layers = state
            for i,h_nodes in enumerate(self.layer_sizes):
                actor_layers = Dense(h_nodes, activation='selu')(actor_layers)
                if self.use_do: actor_layers = self.dropout_layers[i](actor_layers)
            
            actor_layers = Dense(self.action_shape, activation='tanh',
                kernel_initializer=self.f_layer_init)(actor_layers)*\
                self.action_scale
            
            return tf.keras.Model(state, actor_layers)
        
        else:
            state = Input(shape=(self.state_shape))
            action = Input(shape=(self.action_shape))
            
            state_layers = Dense(int(400*size_mult),activation='relu')(state)
            state_layers = BatchNormalization()(state_layers)
            state_layers = Dense(int(300*size_mult),activation='relu')(state_layers)
            state_layers = BatchNormalization()(state_layers)
            
            action_layers = Dense(int(300*size_mult),activation='relu')(action)
            action_layers = BatchNormalization()(action_layers)
            
            layer_merge = Concatenate()([state_layers, action_layers])
            layer_merge = Dense(int(300*size_mult),activation='relu',kernel_regularizer='l2')(layer_merge)
            out_layer = Dense(1, kernel_initializer=self.f_layer_init)(layer_merge)
            
            return tf.keras.Model([state, action], out_layer)
    
    def softTargetNetUpdate(self):
        '''
        Target network parameter update, where star = Q or mu:
        Theta_star_hat <- tau*Theta_star + (1-tau)*Theta_star_hat
        '''
        
        ac_params = self.net.weights
        t_params = self.target_net.weights
        
        update_params = [param*self.tau + t_params[i]*(1-self.tau) 
                         for i,param in enumerate(ac_params)]
        
        self.target_net.set_weights(update_params)



class ActorModel(tf.keras.Model):
    
    def __init__(self, state_shape, action_shape, action_scale, activation, layer_sizes, 
                 pool_size, dropout_rate, use_bn, use_do, DR):
        super(ActorModel, self).__init__()
        
        self.layer_sizes = layer_sizes
        self.use_bn = use_bn
        self.use_do = use_do
        self.action_scale = action_scale
        
        f_layer_init = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
        self.store_activation = [[] for i in range(len(layer_sizes))]
        
        self.dropout_layers = [SpecDropout(dropout_rate) for i in layer_sizes]
        
        if activation=='maxout':
            self.activation_layers = [Maxout(int(i/pool_size)) for i in layer_sizes]
        elif activation=='channelout':
            self.activation_layers = [Channelout(pool_size) for i in layer_sizes]
        elif activation=='selu':
            self.dropout_layers = [SpecAlphaDropout(dropout_rate) for i in layer_sizes]
        else:
            self.activation_layers = [ReLU() for i in layer_sizes]
        
        if DR:
            self.dense_layers = []
            for h_nodes in layer_sizes:
                he_init=tf.random_normal_initializer(mean=0.0,stddev=2/h_nodes)
                self.dense_layers.append(Dense(h_nodes,kernel_initializer=he_init))
        else:
            self.dense_layers = [Dense(h_nodes) for h_nodes in self.layer_sizes]
                
        self.dense_layers.append(Dense(action_shape, activation='tanh',
                kernel_initializer=f_layer_init))
            
        

    def call(self, inputs, training=False):
        actor_layers = tf.cast(inputs,dtype='float32')
        
        for i,h_nodes in enumerate(self.layer_sizes):
                actor_layers = self.dense_layers[i](actor_layers)
                actor_layers = ReLU()(actor_layers)
                actor_layers = self.activation_layers[i](actor_layers)
                if self.use_bn: actor_layers = BatchNormalization()(actor_layers)
                if self.use_do and training: actor_layers = self.dropout_layers[i](actor_layers)
                self.store_activation[i] = actor_layers.numpy()
            
        actor_layers = self.dense_layers[-1](actor_layers)*self.action_scale
        
        return actor_layers
    
    def packPruning(self, task, T, beta):
        '''
        task (int): Task number
        T (int): total number of tasks
        beta (float): previously obtained max level of sparsity (0 < beta < 1)
        
        PackNet algorithm:
            - For task t, train network and prune to at least 1/T of
              total neurons, and at most beta of total neurons.
            - Once training is complete, freeze neurons associated with this task.
            - For task t+1, train remaining neurons, reprune, freeze.
        '''
        
        pass





