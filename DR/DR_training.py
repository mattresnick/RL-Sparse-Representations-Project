import os
os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
import tensorflow as tf
import numpy as np
import gym
import pybullet_envs
from ReplayBuffer import IndividualBuffers, makeOUNoise
from ActorCritic import ActorCritic
import matplotlib.pyplot as plt
import DistributionalRegularizers as DRS







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
    
    return zero_count, zero_pct, unique_nonzero_locs









def train(ActorObj, CriticObj, buffer, noise, episode_num, step_num, batch_size, gamma):
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
    nonzero_locs = np.array([])
    
    # Episode control.
    for ep in range(episode_num):
        R_e = 0
        state = np.reshape(env.reset(), (1, state_shape))
        ep_losses = []
        
        # Step control.
        for st in range(step_num):
            env.render()
            
            # Take an action and add noise (OU, Normal, or No).
            action = np.clip(ActorObj.net(state).numpy() + \
                             noise.makeNoise(),(-1)*action_scale,action_scale)
            
            # Obtain results of the action from the environment.
            observation, reward, done, info = env.step(action[0])
            
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
                CriticObj.opt.apply_gradients(zip(critic_grads, critic_vars))
            
            # Train the actor network.
            with tf.GradientTape() as tape2:
                
                # Using action predictions from actor, get critic output.
                action_preds = ActorObj.net(states, training=True)
                output_ca = CriticObj.net([states, action_preds])
                
                loss_a = (-1)*tf.math.reduce_mean(output_ca)
                '''
                for act_output in ActorObj.net.store_activation:
                    loss_a += exp_distributional_regularizer(act_output)
                '''
                # Optimize actor parameters by maximizing average critic value.
                actor_vars = ActorObj.net.trainable_variables
                actor_grads = tape2.gradient(loss_a, actor_vars)
                
                for i, act_output in enumerate(ActorObj.net.store_activation):
                    actor_grads[i+1] += DRS.exp_distributional_regularizer(act_output, grad=True)
                
                ActorObj.opt.apply_gradients(zip(actor_grads, actor_vars))
                
            
            # Update target networks.
            ActorObj.softTargetNetUpdate()
            CriticObj.softTargetNetUpdate()
            
            ep_losses.append([loss_a,loss_c])
                
            if done: break
        
        reward_store.append(R_e)
        #if len(ep_losses): loss_store.append([ep_losses[-1][0],ep_losses[-1][1]])
        
        print ('Episode #' + str(ep+1) + '. Reward: ' + str(R_e))
        #print ('Actor Loss: ' + str(loss_store[-1][0]) + '. Critic Loss: ' + str(loss_store[-1][1]))
        
        outputs = []
        for i, l in enumerate(ActorObj.net.store_activation):
            outputs.extend(l.flatten())
        
        zero_count, zero_pct, unique_nonzero_locs = zero_calculus(outputs, nonzero_locs)
        
        print (zero_count)
        print (zero_pct)
        print (unique_nonzero_locs.shape)
        zero_store.append(zero_pct)
        nonzero_locs = list(unique_nonzero_locs)
    
    return reward_store, ActorObj, CriticObj, zero_store





# Prepare simulation environment and store pertinent information.
env_names = ['Pendulum-v0', 'HalfCheetahBulletEnv-v0','MountainCarContinuous-v0']
env = gym.make(env_names[2])
env.seed(4444)
#env.render()
env.reset()


state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_max = env.action_space.high[0]

ac_dict = {'state_shape':state_dim, 
           'action_shape':action_dim, 
           'action_scale':action_max,
           'tau':1e-3} 



actor_dict = {'layer_sizes':[480,360],
              'activation':'relu',
              'pool_size':2,
              'dropout_rate':0.3,
              'use_bn':False,
              'use_do':False,
              'DR':False}

# Create actor and critic objects based on environment information.
ActorObj = ActorCritic(actor_type=True,**ac_dict, lr=1e-3, **actor_dict)
CriticObj = ActorCritic(actor_type=False,**ac_dict, lr=1e-3)


# Make experience buffer and noise.
buffer_size = int(5e4)
BufferObj = IndividualBuffers(buffer_size, state_dim, action_dim)
NoiseObj = makeOUNoise(noise_type='normal',mu=np.zeros(action_dim),sigma=np.full(action_dim,0.2))

# Training arguments.
arg_dict = {'ActorObj':ActorObj, 
            'CriticObj':CriticObj,
            'buffer':BufferObj,
            'noise':NoiseObj,
            'episode_num':200,
            'step_num':1000, 
            'batch_size':64,
            'gamma':0.99}

# Run training procedure and save results.
Total_R_e, TrainedActorObj, TrainedCriticObj, zero_store = train(**arg_dict)



model_name = 'test_expDR_relu_beta10_lambda01_nobatchnorm_pendulum'

#TrainedActorObj.net.save_weights('./saved_models/'+model_name+'_actor')
#TrainedCriticObj.net.save_weights('./saved_models/'+model_name+'_critic')

rew_str = ''
for r in Total_R_e:
    rew_str = rew_str+str(r)+'\n'
rew_str = rew_str[:-1]
with open('./reward_results/'+model_name+'.txt','a') as write_file:
    write_file.write(rew_str)


fig, ax = plt.subplots(figsize=(12,7))
plt.plot(list(range(len(Total_R_e))), Total_R_e)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()

fig, ax = plt.subplots(figsize=(12,7))
plt.plot(list(range(len(zero_store))), zero_store)
plt.xlabel('Episode')
plt.ylabel('Pct Zero Activations')
plt.show()

