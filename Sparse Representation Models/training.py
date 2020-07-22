import os
os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
import tensorflow as tf
import numpy as np
import gym
import pybullet_envs
from ReplayBuffer import IndividualBuffers, makeOUNoise
from ActorCriticDR import ActorCritic
import matplotlib.pyplot as plt
import DistributionalRegularizers as DRS
from math import floor
from time import time





def zero_calculus(outputs, all_nonzero_locs):
    
    # True where activations are zero.
    zero_bool_array = np.array(outputs).flatten()==np.repeat(0,len(outputs))
    
    zero_count = np.sum(zero_bool_array)
    zero_pct = zero_count/len(outputs)
    
    # True where activations are not zero.
    nonzero_bool_array = np.array(outputs).flatten()!=np.repeat(0,len(outputs))
    
    # Indices of nonzero activation values
    nonzero_locs = np.nonzero(nonzero_bool_array)
    unique_nonzero_locs = np.union1d(all_nonzero_locs,nonzero_locs)
    
    #print ('Number of zero activations:',str(zero_count))
    print ('Zero activations as pct of total:', str(round(zero_pct*100,2)))
    print ('Number of unique nonzero activations:',unique_nonzero_locs.shape[0])
    
    return zero_count, zero_pct, unique_nonzero_locs




def gradientMasks(ActorCriticObj):
    grad_shapes = ActorCriticObj.dense_layer_sizes
    masks = []
    for m,mgrad in enumerate(grad_shapes):
        i = int(floor(m/2))
        j = m%2
        
        # Check for the case of a single-valued layer.
        if len(ActorCriticObj.frozen_inds[i][j])>0:
            
            mask = np.ones(mgrad)
            index_list = [list(h.numpy()) for h in ActorCriticObj.frozen_inds[i][j]]
            
            for ind in index_list: mask[tuple(ind)] = 0
            mask = tf.convert_to_tensor(mask,dtype='float32')
        
        else:
            mask = tf.ones(shape=mgrad,dtype='float32')
        
        mask = tf.stop_gradient(mask)
        masks.append(mask)
    
    return masks




def train(ActorObj, CriticObj, buffer, noise, episode_num, step_num, batch_size, 
          gamma, envmt, task, PACK, CPACK, DR):
    '''
    Model training procedure.
    
    Parameters:
        - ActorObj (ActorCritic object): Object which contains the actor network,
          corresponding target network, and all relevant parameters.
        - CriticObj (ActorCritic object): Sames as above, for Critic.
        - buffer (IndividualBuffers object): Experience replay buffer.
        - noise (makeOUNoise object): Noise object for producing action noise.
        - episode_num (Integer): Total number of episodes to be run.
        - step_num (Integer): Maximum number of steps per episode.
        - batch_size (Integer): Number of experiences to sample from replay buffer
          every training step.
        - gamma (Float): Temporal discount scale factor, 0 <= gamma <1.
    
    Returns:
        - reward_store (list): Reward total for each episode.
        - ActorObj (ActorCritic object): Trained actor model.
        - CriticObj (ActorCritic object): Trained critic model.
    '''
    
    
    state_shape = ActorObj.state_shape
    action_scale = ActorObj.action_scale
    zero_store, reward_store = [], []
    loss_store = []
    nonzero_locs = np.array([])
    actor_masks, critic_masks = None, None
    
    # Zero gradients associated with frozen parameters by creating a boolean mask. 
    if task>0:
        actor_masks = gradientMasks(ActorObj)
        if CPACK: critic_masks = gradientMasks(CriticObj)
    
    # Episode control.
    for ep in range(episode_num):
        R_e = 0
        state = np.reshape(envmt.reset(), (1, state_shape))
        ep_losses = []
        
        # Step control.
        for st in range(step_num):
            #envmt.render()
            start=time()
            # Take an action and add noise (OU, Normal, or No).
            action = np.clip(ActorObj.net(state).numpy() + \
                             noise.makeNoise(),(-1)*action_scale,action_scale)
            
            # Obtain results of the action from the environment.
            observation, reward, done, info = envmt.step(action[0])
            
            # Update the buffer, episode cumulative reward, and state.
            buffer.addTransition([state, action, observation, done, reward])
            R_e += reward
            state = np.reshape(observation, (1, state_shape))
                
            # Sample the buffer for experiences.
            states, actions, next_states, done_vals, rewards = buffer.sample(batch_size)
            
            # Train the critic network.
            with tf.GradientTape() as tape1:
                
                # Value target via target networks.
                target_acts = ActorObj.target_net(next_states)
                critic_target = CriticObj.target_net([next_states, target_acts])
                yi = np.reshape(rewards, (batch_size,1))+gamma*critic_target
                
                # Critic output and loss.
                output_c = CriticObj.net([states, actions])
                loss_c = tf.keras.losses.MeanSquaredError()(np.reshape(yi, (batch_size, 1)),output_c)
                
                # Optimize critic parameters with resulting gradient.
                critic_vars = CriticObj.net.trainable_variables
                critic_grads = tape1.gradient(loss_c, critic_vars)
                
                if CPACK:
                    # Apply masks to gradients, effectively freezing parameters.
                    if task>0:
                        for m,mgrad in enumerate(critic_grads):
                            critic_grads[m] = mgrad*critic_masks[m]
                
                CriticObj.opt.apply_gradients(zip(critic_grads, critic_vars))
            
            # Train the actor network.
            with tf.GradientTape() as tape2:
                
                # Using action predictions from actor, get critic output.
                action_preds = ActorObj.net(states, training=True)
                output_ca = CriticObj.net([states, action_preds])
                loss_a = (-1)*tf.math.reduce_mean(output_ca)
                
                # We want to be careful about which parameters can be updated at
                # this step and which can't, due to PackNet
                actor_vars = ActorObj.net.trainable_variables
                
                '''
                if DR:
                    for penalty in ActorObj.net.store_DR_reg_terms:
                        loss_a += penalty
                '''
                actor_grads = tape2.gradient(loss_a, actor_vars)
                
                if PACK:
                    # Apply masks to gradients, effectively freezing parameters.
                    if task>0:
                        for m,mgrad in enumerate(actor_grads):
                            actor_grads[m] = mgrad*actor_masks[m]
                
                ActorObj.opt.apply_gradients(zip(actor_grads, actor_vars))
                
            
            # Update target networks.
            ActorObj.softTargetNetUpdate()
            CriticObj.softTargetNetUpdate()
            
            ep_losses.append([loss_a,loss_c])
            
            
            end=time()
            #print ('Time for step '+ str(st+1) + ':' + str(round(end-start, 2)))
            
            if done: break
        
        reward_store.append(R_e)
        #if len(ep_losses): loss_store.append([ep_losses[-1][0],ep_losses[-1][1]])
        
        print ('Episode #' + str(ep+1) + '. Reward: ' + str(R_e))
        #print ('Actor Loss: ' + str(loss_store[-1][0]) + '. Critic Loss: ' + str(loss_store[-1][1]))
        
        outputs = []
        for i, l in enumerate(ActorObj.net.store_activation):
            outputs.extend(l.flatten())
        
        zero_count, zero_pct, unique_nonzero_locs = zero_calculus(outputs, nonzero_locs)
        
        
        zero_store.append(zero_pct)
        nonzero_locs = list(unique_nonzero_locs)
    
    return reward_store, ActorObj, CriticObj, zero_store





def testing(ActorObj,envmt,state_dim,model_name,name):
    reward_store, roll_avg_rew = [], []
    num_eps = 100
    for ep in range(num_eps):
        R_e = 0
        state = np.reshape(envmt.reset(), (1, state_dim))
        
        while True:
            action = ActorObj.net(state).numpy()
            observation, reward, done, info = envmt.step(action[0])
            
            R_e += reward
            state = np.reshape(observation, (1, state_dim))
            
            if done: break
        
        #print ('Episode #' + str(ep+1) + '. Reward: ' + str(R_e))
        reward_store.append(R_e)   
        roll_avg_rew.append(np.mean(reward_store[-10:]))
        
    fig, ax = plt.subplots(figsize=(12,7))
    #plt.plot(list(range(len(reward_store))), reward_store, label='R_e', color='blue')
    plt.plot(list(range(len(reward_store))), roll_avg_rew, label='Avg. R_e over 10 ep.', color='orange')
    plt.plot(list(range(len(reward_store))), np.repeat(np.mean(reward_store),num_eps), label='Avg. Reward', color='black')
    plt.legend(loc='lower right')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward per Episode')
    plt.show()
    
    
    rew_str = ''
    for r in reward_store:
        rew_str = rew_str+str(r)+'\n'
    rew_str = rew_str[:-1]
    with open('./reward_results/'+model_name+'_'+name+'.txt','a') as write_file:
        write_file.write(rew_str)


