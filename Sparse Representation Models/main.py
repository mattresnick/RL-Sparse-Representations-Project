from ReplayBuffer import IndividualBuffers, makeOUNoise
from ActorCritic import ActorCritic
import matplotlib.pyplot as plt
import gym
import pybullet_envs
import numpy as np
from training import train, testing


if __name__=="__main__":
    
    
    activation_name='relu'
    save_name='PackNet_TESTS'
    
    DR = False # Turn Distributional Regularizers on and off.
    PACK = False # Turn PackNet on and off.
    CPACK = False # Turn PackNet on and off for critic net specifically.
    USE_PER = False # Prioritized Experience Replay buffer

    # Obtain pertinent environment information.
    env_names = ['Pendulum-v0']
    # env_names = ['HalfCheetahBulletEnv-v0','HalfCheetahBulletEnv-v0']
    
    # Obtain pertinent environment information. 
    env_names = ['HalfCheetahBulletEnv-v0']
    seeds = [4444, 5021, 1580]
    
    env = gym.make(env_names[0])
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_max = env.action_space.high[0]
    
    ac_dict = {'state_shape':state_dim, 
               'action_shape':action_dim, 
               'action_scale':action_max,
               'tau':1e-3} 
    
    actor_dict = {'layer_sizes':[480,360],
                  'activation':activation_name,
                  'pool_size':2,
                  'dropout_rate':0.0,  # <--- 0.3
                  'use_bn':False,
                  'use_do':True,
                  'DR':DR,
                  'total_tasks':len(env_names),
                  'cpack':False}
    
    # Create actor and critic objects based on environment information.
    ActorObj = ActorCritic(actor_type=True,**ac_dict, lr=1e-3, **actor_dict)
    
    # Critic object has a simpler version if not using PackNet on it. 
    # Which is only needed for multi-task PackNet.
    if not CPACK:
        CriticObj = ActorCritic(actor_type=False,**ac_dict, lr=1e-3, cpack=CPACK)
    else:
        critic_dict = {'layer_sizes':[480,360],
                  'activation':activation_name,
                  'pool_size':2,
                  'dropout_rate':0.3,
                  'use_bn':False,
                  'use_do':False,
                  'DR':DR,
                  'total_tasks':len(env_names),
                  'cpack':CPACK}
        
        CriticObj = ActorCritic(actor_type=False,**ac_dict, lr=1e-3, **critic_dict)
    
    
    Total_R_e_list = []
    for t,name in enumerate(env_names):
        
        envmt = gym.make(name)
        envmt.seed(seeds[t])
        envmt.reset()
        
        state_dim = envmt.observation_space.shape[0]
        action_dim = envmt.action_space.shape[0]
        action_max = envmt.action_space.high[0]
        
        # Make experience buffer and noise.
        buffer_size = int(5e4)
        BufferObj = IndividualBuffers(buffer_size, state_dim, action_dim)
        NoiseObj = makeOUNoise(noise_type='normal',mu=np.zeros(action_dim),sigma=np.full(action_dim,0.2))
        
        # Training arguments.
        arg_dict = {'ActorObj':ActorObj, 
                    'CriticObj':CriticObj,
                    'buffer':BufferObj,
                    'noise':NoiseObj,
                    'episode_num':150,
                    'step_num':1000, 
                    'batch_size':64,
                    'gamma':0.99,
                    'envmt':envmt,
                    'task':t,
                    'PACK':PACK,
                    'CPACK':CPACK,
                    'DR':DR,
                    'USE_PER': USE_PER}
        
        # Run training procedure and save results.
        Total_R_e, TrainedActorObj, TrainedCriticObj, zero_store = train(**arg_dict)
        Total_R_e_list.append(Total_R_e)
        
        if PACK:
            if t<len(env_names)-1:
                ActorObj.packPruning(task=t,T=len(env_names))
                if CPACK: CriticObj.packPruning(task=t,T=len(env_names))
            
            # Reinitialize parameters with original values.
            ActorObj.layerSave(False)
            ActorObj.softTargetNetUpdate()
            
            if CPACK:
                CriticObj.layerSave(False)
                CriticObj.softTargetNetUpdate()
        
        '''
        # Show zero activation percent over training
        fig, ax = plt.subplots(figsize=(12,7))
        plt.plot(list(range(len(zero_store))), zero_store)
        plt.xlabel('Episode')
        plt.ylabel('Pct Zero Activations')
        plt.show()
        '''
            
        
    model_name = save_name+'_'+activation_name
    
    #TrainedActorObj.net.save_weights('./saved_models/'+model_name+'_actor')
    #TrainedCriticObj.net.save_weights('./saved_models/'+model_name+'_critic')
    
    for i, R_e in enumerate(Total_R_e_list):
        fig, ax = plt.subplots(figsize=(12,7))
        plt.plot(list(range(len(R_e))), R_e)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Task #'+str(i+1))
        plt.show()
        
        rew_str = ''
        for r in R_e:
            rew_str = rew_str+str(r)+'\n'
        rew_str = rew_str[:-1]
        with open('./results/'+model_name+'.txt','a') as write_file:
            write_file.write(rew_str)
    
    testing(ActorObj,env,state_dim,model_name,'new_seed')
    
    ActorObj.setTask(0)
    env.seed(seeds[0])
    testing(ActorObj,env,state_dim,model_name,'first_seed')
    
    ActorObj.setTask(1)
    env.seed(seeds[1])
    testing(ActorObj,env,state_dim,model_name,'second_seed')
