import tensorflow as tf
import numpy as np


def exp_distributional_regularizer(activation_matrix, beta_val=0.1, 
                                   lambdaKL=0.01, grad=True):
    shape = tf.shape(activation_matrix).numpy()
    batch_size = shape[0]
    num_neruons = shape[1]
    
    beta = tf.convert_to_tensor(np.repeat(beta_val, num_neruons),dtype='float32')
    ones = tf.convert_to_tensor(np.repeat(1, num_neruons),dtype='float32')
    
    h_ij = activation_matrix
    beta_j = tf.math.reduce_sum(h_ij, axis=0)/batch_size
    beta_j = tf.convert_to_tensor(beta_j,dtype='float32')
    
    if grad:
        dSKL = tf.where(beta_j>beta_val, 
                   (1/beta_j) - (beta/tf.math.square(beta_j)), 
                   0)
        return_val = dSKL
        
    else:
        SKL = tf.where(beta_j>beta_val, 
                       tf.math.log(beta_j) + (beta/beta_j) - tf.math.log(beta) - ones, 
                       0)
        return_val = tf.math.reduce_sum(SKL)
    
    return lambdaKL*return_val




def gaussian_distributional_regularizer(activation_matrix, mu_val=0, 
                                   lambdaKL=0.01, grad=True):
    shape = tf.shape(activation_matrix).numpy()
    batch_size = shape[0]
    num_neruons = shape[1]
    
    sigma_val = 2/num_neruons
    
    mu = tf.convert_to_tensor(np.repeat(mu_val, num_neruons),dtype='float32')
    sigma = tf.convert_to_tensor(np.repeat(sigma_val, num_neruons),dtype='float32')
    halfs = tf.convert_to_tensor(np.repeat(0.5, num_neruons),dtype='float32')
    
    h_ij = activation_matrix
    beta_j = tf.math.reduce_sum(h_ij, axis=0)/batch_size
    beta_j = tf.convert_to_tensor(beta_j,dtype='float32')
    
    SKL = tf.where(beta_j>mu_val, 
                   (tf.math.square(sigma)+tf.math.square(beta_j - mu)/tf.math.square(sigma)*2) - halfs, 
                   0)
    
    if grad:
        dSKL = tf.where(beta_j>mu_val, 
                   ((beta_j - mu)/tf.math.square(sigma)), 
                   0)
        return_val = dSKL
        
    else:
        SKL = tf.where(beta_j>mu_val, 
                   (tf.math.square(sigma)+tf.math.square(beta_j - mu)/tf.math.square(sigma)*2) - halfs, 
                   0)
        return_val = tf.math.reduce_sum(SKL)
        
        
    return lambdaKL*return_val



def bernoulli_distributional_regularizer(activation_matrix, rho_val=0.1, 
                                   lambdaKL=0.01, grad=True):
    shape = tf.shape(activation_matrix).numpy()
    batch_size = shape[0]
    num_neruons = shape[1]
    
    rho = tf.convert_to_tensor(np.repeat(rho_val, num_neruons),dtype='float32')
    ones = tf.convert_to_tensor(np.repeat(1, num_neruons),dtype='float32')
    
    h_ij = activation_matrix
    beta_j = tf.math.reduce_sum(h_ij, axis=0)/batch_size
    beta_j = tf.convert_to_tensor(beta_j,dtype='float32')
    
    if grad:
        dSKL = tf.where(beta_j>rho_val, 
                   ((-1)*rho/beta_j) - ((ones-rho)/(ones-beta_j)), 
                   0)
        return_val = dSKL
        
    else:
        SKL = tf.where(beta_j>rho_val, 
                       rho*tf.math.log(rho/beta_j) + (ones-rho)*tf.math.log((ones-rho)/(ones-beta_j)), 
                       0)
        return_val = tf.math.reduce_sum(SKL)
    
    return lambdaKL*return_val




