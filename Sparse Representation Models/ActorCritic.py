import os
os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout, ReLU
from CustomActivations import Channelout, ChanneloutWinnerT, SELULayer
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
        - total_tasks (Integer): Total number of tasks to be trained over.
        - cpack (Boolean): If true, use PackNet-compatible Critic class to build critic nets.
    
    Non-Trivial Attributes:
        - opt: Network parameter optimizer for training.
        - net: Actor or Critic network.
        - target_net: Target network for Actor or Critic.
        - activation_layers: List storing activation function layers.
        - dropout_layers: List storing dropout layers.
    '''
    
    def __init__(self, actor_type, state_shape, action_shape, action_scale, tau, lr, 
                 layer_sizes=[480,360], activation='relu', pool_size=2, dropout_rate=0.2, 
                 use_bn=True, use_do=True, DR=False, total_tasks=3, cpack=False):
        
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
        self.t = 0
        
        self.opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.f_layer_init = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
        
        
        
        if self.actor_type:
            
            self.dense_layer_sizes = [(state_shape, layer_sizes[0]), (layer_sizes[0],)]
            for i, l in enumerate(layer_sizes[:-1]):
                self.dense_layer_sizes.append((l,layer_sizes[i+1]))
                self.dense_layer_sizes.append((layer_sizes[i+1],))
            self.dense_layer_sizes.append((layer_sizes[-1],action_shape))
            self.dense_layer_sizes.append((action_shape,))
            
            self.layer_masks = []
            for task in range(total_tasks):
                task_masks = []
                for size in self.dense_layer_sizes:
                    task_masks.append(np.ones(size))
                self.layer_masks.append(task_masks)
            
            self.net = ActorModel(state_shape, action_shape, action_scale, activation, layer_sizes, 
                                  pool_size, dropout_rate, use_bn, use_do, DR, total_tasks, self.layer_masks)
            self.target_net = ActorModel(state_shape, action_shape, action_scale, activation, 
                                  layer_sizes, pool_size, dropout_rate, use_bn, use_do, DR, total_tasks, self.layer_masks)
            
            self.frozen_inds = [[[],[]] for d in self.net.dense_layers]
            self.frozen_vals = [[[],[]] for d in self.net.dense_layers]
            
            self.layerSave()
        
        else:
            if not cpack:
                self.net = self.createApproximator()
                self.target_net = self.createApproximator()
            else:
                
                # This part is not agnostic about the number of layer sizes like the actor is,
                # meaning it will only take two and still work right at the moment.
                self.dense_layer_sizes = [
                (state_shape, layer_sizes[0]), (layer_sizes[0],),
                (layer_sizes[0], layer_sizes[1]), (layer_sizes[1],),
                (action_shape, layer_sizes[1]), (layer_sizes[1],),
                (layer_sizes[1]*2, layer_sizes[1]), (layer_sizes[1],),
                (layer_sizes[1], 1), (1,)]
                
                self.layer_masks = []
                for task in range(total_tasks):
                    task_masks = []
                    for size in self.dense_layer_sizes:
                        task_masks.append(np.ones(size))
                    self.layer_masks.append(task_masks)
                
                self.net = CriticModel(state_shape, action_shape, action_scale, activation, layer_sizes, 
                                       self.dense_layer_sizes, pool_size, dropout_rate, use_bn, use_do, 
                                       DR, total_tasks, self.layer_masks)
                self.target_net = CriticModel(state_shape, action_shape, action_scale, activation, layer_sizes, 
                                      self.dense_layer_sizes, pool_size, dropout_rate, use_bn, use_do, 
                                      DR, total_tasks, self.layer_masks)
                
                self.frozen_inds = [[[],[]] for d in self.net.dense_layers]
                self.frozen_vals = [[[],[]] for d in self.net.dense_layers]
                
                self.layerSave()
            
            
        
        
    
    def createApproximator(self):
        
        # Layer raw output -> Channelout -> Batchnorm -> Dropout
        
        size_mult = 1.0
        
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
    
    
    def layerSave(self,save_mode=True):
        '''
        For saving intialization values (savemode=True) and then reinstating them without overwriting
        the frozen paramerters (savemode=False).
        '''
        if save_mode: 
            self.saved_layers = self.net.dense_layers
            for i in range(len(self.saved_layers)):
                for j in range(len(self.saved_layers[i].trainable_weights)):
                    params = self.saved_layers[i].trainable_weights[j]
                    self.saved_layers[i].trainable_weights[j] = params
                    
        else:
            for i, dlayer in enumerate(self.saved_layers):
                for j, layer in enumerate(dlayer.trainable_weights):
                    
                    # Check for the case of a single-valued layer.
                    if len(self.frozen_inds[i][j])>0:
                        tensor=layer
                        indices=self.frozen_inds[i][j]
                        updates=self.frozen_vals[i][j]
                        self.net.dense_layers[i].trainable_weights[j] = tf.tensor_scatter_nd_add(
                                tensor=tensor,
                                indices=indices, 
                                updates=updates)
                    else:
                        self.net.dense_layers[i].trainable_weights[j] = layer
    
    
    def setTask(self, t):
        self.t = t
    
    # Given a matrix, make binary mask with zeros at indices.
    def makeMask(self, matrix, indices):
        '''
        matrix (Tensor): Tensor to be masked.
        indices (List of Tensors): Indices where values will be masked out.
        '''
        mask = np.ones(matrix.shape)
        index_list = [list(h.numpy()) for h in indices]
        for ind in index_list: mask[tuple(ind)] = 0
        
        return mask
    
    # Returns a tensor with zeros where parameters are already frozen.
    def createPruneTensor(self, layer, indices, i, j):
        
        # Check for the case of a single-valued layer.
        if len(indices)>0:
            mask = self.makeMask(layer, indices)
            mask = tf.convert_to_tensor(mask,dtype='float32')
        
        else:
            mask = tf.ones(shape=layer.shape,dtype='float32')
        
    
        return layer*mask

    
    # Call from packpruning, obtain indices from values below "boundary"
    def saveBinaryMask(self, i, j, indices):
        m = i*2+j
        self.layer_masks[self.t][m] = self.makeMask(self.layer_masks[self.t][m], indices)
    
    
    def packPruning(self, task, T, beta=None):
        '''
        task (int): Task number
        T (int): total number of tasks
        beta (float): previously obtained max level of sparsity (0 < beta < 1)
        '''
        for i, dlayer in enumerate(self.net.dense_layers):
            for j, layer in enumerate(dlayer.trainable_weights):
                
                # Tensor with zeros where parameters are already frozen.
                if task>0: prune_tensor = self.createPruneTensor(layer, self.frozen_inds[i][j], i, j)
                else: prune_tensor = layer
                
                # Get relevant shapes.
                if len(prune_tensor.shape)>1:
                    a,b = prune_tensor.shape[0], prune_tensor.shape[1]
                    M = a*b
                else:
                    M = prune_tensor.shape[0]
                
                
                keep_num = int(np.floor((1/(T-task))*M))
                flat_weightlist = prune_tensor.numpy().reshape((M,1))
                
                # Create norm, index pairs so original indices can be recovered 
                # after sorting if needed.
                l1_norms = np.linalg.norm(flat_weightlist, ord=1, axis=1)
                sorter = list(zip(l1_norms,list(range(len(l1_norms)))))
                
                # Sort by norm value, identify keep_num value boundary.
                sorter.sort(key=lambda x: x[0], reverse=True)
                #sel_inds = np.array(sorter,dtype='int32')[:keep_num,1]
                
                boundary = sorter[keep_num][0]
                frozen_inds = tf.where(tf.math.abs(prune_tensor)>boundary)
                
                self.saveBinaryMask(i,j,tf.where(tf.math.abs(prune_tensor)<=boundary))
                
                self.frozen_inds[i][j].extend(frozen_inds)
                self.frozen_vals[i][j].extend(tf.gather_nd(layer,frozen_inds))
                
                # Prune all weights not frozen for this task (just for testing in between tasks)
                self.net.dense_layers[i].trainable_weights[j] = tf.where(prune_tensor>=boundary, layer, 0)
                
        self.net.setLayerMasks(self.layer_masks)
        self.net.setTask(self.t)
















class ActorModel(tf.keras.Model):
    
    def __init__(self, state_shape, action_shape, action_scale, activation, layer_sizes, 
                 pool_size, dropout_rate, use_bn, use_do, DR, total_tasks, layer_masks):
        super(ActorModel, self).__init__()
        
        self.layer_sizes = layer_sizes
        self.use_bn = use_bn
        self.use_do = use_do
        self.action_scale = action_scale
        self.total_tasks = total_tasks
        self.DR = DR
        self.layer_masks = layer_masks
        
        self.num_layers = len(layer_sizes)
        self.use_cowt = False
        self.t = 0
        
        self.store_activation = [[] for i in range(self.num_layers)]
        self.store_DR_reg_terms = [[] for i in range(self.num_layers)]
        
        self.dropout_layers = self.getDropoutLayers(activation,dropout_rate)
        self.activation_layers = self.getActivationLayers(activation,pool_size)
        
        #DR
        if False:
            self.dense_layers = []
            for h_nodes in layer_sizes:
                he_init=tf.random_normal_initializer(mean=0.0,stddev=2/h_nodes)
                self.dense_layers.append(Dense(h_nodes,kernel_initializer=he_init))
        else:
            self.dense_layers = [Dense(h_nodes) for h_nodes in self.layer_sizes]
            
        f_layer_init = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
        self.dense_layers.append(Dense(action_shape, activation='tanh',
                kernel_initializer=f_layer_init))
        
    def getActivationLayers(self,activation,pool_size):
        if activation=='maxout':
            activation_layers = [Maxout(int(i/pool_size)) for i in range(self.num_layers)]
        elif activation=='channelout':
            activation_layers = [Channelout(pool_size) for i in range(self.num_layers)]
        elif activation=='channelout_winner_t':
            activation_layers = [ChanneloutWinnerT(pool_size) for i in range(self.num_layers)]
            self.use_cowt = True
        elif activation=='selu':
            activation_layers = [SELULayer() for i in range(self.num_layers)]
        else:
            activation_layers = [ReLU() for i in range(self.num_layers)]
        
        return activation_layers
    
    def getDropoutLayers(self,activation,dropout_rate):
        if activation=='selu':
            dropout_layers = [SpecAlphaDropout(dropout_rate) for i in range(self.num_layers)]
        else:
            dropout_layers = [SpecDropout(dropout_rate) for i in range(self.num_layers)]
            
        return dropout_layers
    
    def setLayerMasks(self, layer_masks):
        self.layer_masks = layer_masks
        
    def setTask(self, t):
        self.t = t
    
    def applyMask(self, layer, layer_num):
        if self.t==-1:
            return layer
        
        masked_layer = layer
        for j, param in enumerate(masked_layer.trainable_weights):
            masked_layer.trainable_weights[j] = param*self.layer_masks[self.t][layer_num*2+j]
        return masked_layer
    
    def call(self, inputs, training=False):
        actor_layers = tf.cast(inputs,dtype='float32')
        
        for i,h_nodes in enumerate(self.layer_sizes):
                
                masked_layer = self.applyMask(self.dense_layers[i], i)
                actor_layers = masked_layer(actor_layers)
                
                # Apply activation (cowt requires a special argument)
                if not self.use_cowt: actor_layers = self.activation_layers[i](actor_layers)
                else: actor_layers = self.activation_layers[i](actor_layers, self.t)
                
                # Apply batchnorm/dropout if applicable
                if self.use_bn: actor_layers = BatchNormalization()(actor_layers)
                if self.use_do and training: actor_layers = self.dropout_layers[i](actor_layers)
                
                # Save activation for DR if applicable
                self.store_activation[i] = actor_layers.numpy()
                
                if self.DR:
                    reg_term = DRS.exp_distributional_regularizer(actor_layers,grad=False)
                    #print (actor_layers)
                    self.store_DR_reg_terms[i] = reg_term
            
        actor_layers = self.dense_layers[-1](actor_layers)*self.action_scale
        
        return actor_layers
    





class CriticModel(ActorModel):
    
    def __init__(self, state_shape, action_shape, action_scale, activation, layer_sizes, dense_layer_sizes, 
                 pool_size, dropout_rate, use_bn, use_do, DR, total_tasks, layer_masks):
        super().__init__(state_shape, action_shape, action_scale, activation, layer_sizes, 
                 pool_size, dropout_rate, use_bn, use_do, DR, total_tasks, layer_masks)
        self.dense_layer_sizes = dense_layer_sizes
        self.use_bn = use_bn
        self.use_do = use_do
        self.action_scale = action_scale
        self.total_tasks = total_tasks
        self.DR = DR
        self.layer_masks = layer_masks
        
        self.num_layers = len(dense_layer_sizes)
        self.use_cowt = False
        self.t = 0
        
        self.store_activation = [[] for i in range(self.num_layers)]
        
        self.dropout_layers = self.getDropoutLayers(activation,dropout_rate)
        self.activation_layers = self.getActivationLayers(activation,pool_size)
        
        h_node_list = [size[0] for size in dense_layer_sizes[1::2]]
        
        if DR:
            self.dense_layers = []
            for h_nodes in h_node_list[:-1]:
                he_init=tf.random_normal_initializer(mean=0.0,stddev=2/h_nodes)
                self.dense_layers.append(Dense(h_nodes,kernel_initializer=he_init))
        else:
            self.dense_layers = [Dense(h_nodes) for h_nodes in h_node_list[:-1]]
            
        f_layer_init = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
        self.dense_layers.append(Dense(1, kernel_initializer=f_layer_init))
        
        self.concat_layer = Concatenate()
    
    
    def call(self, inputs, training=False):
        
        state_layers = tf.cast(inputs[0],dtype='float32')
        
        state_layers = self.forwardLayer(state_layers, 0, training)
        state_layers = self.forwardLayer(state_layers, 1, training)
        
        action_layers = tf.cast(inputs[1],dtype='float32')
        action_layers = self.forwardLayer(action_layers, 2, training)
        
        layer_merge = self.concat_layer([state_layers, action_layers])
        layer_merge = self.forwardLayer(layer_merge, 3, training)
        
        out_layer = self.forwardLayer(layer_merge, 4, training)
        
        return out_layer
    
    def forwardLayer(self, input_tensor, entry_point, training):
        masked_layer = self.applyMask(self.dense_layers[entry_point], entry_point)
        f_layer = masked_layer(input_tensor)
        
        # Apply activation (cowt requires a special argument)
        if not self.use_cowt: f_layer = self.activation_layers[entry_point](f_layer)
        else: f_layer = self.activation_layers[entry_point](f_layer, self.t)
        
        # Apply batchnorm/dropout if applicable
        if self.use_bn: f_layer = BatchNormalization()(f_layer)
        if self.use_do and training: f_layer = self.dropout_layers[entry_point](f_layer)
        
        # Save activation for DR if applicable
        self.store_activation[entry_point] = f_layer.numpy()
        
        return f_layer













