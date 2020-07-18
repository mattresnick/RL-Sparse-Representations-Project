import tensorflow as tf
import gym
import pybullet_envs
from ActorCritic import ActorCritic
import numpy as np
import matplotlib.pyplot as plt
import os

'''
env_names = 'HalfCheetahBulletEnv-v0'
env = gym.make(env_names)
env.seed(4444)
env.render()
env.reset()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_max = env.action_space.high[0]

ac_dict = {'state_shape':state_dim, 
           'action_shape':action_dim, 
           'action_scale':action_max,
           'tau':1e-3}

ActorObj = ActorCritic(actor_type=True,**ac_dict, lr=1e-3)
ActorObj.net.load_weights('./saved_models/trained_actor_gamma99full2')


reward_store, roll_avg_rew = [], []
for ep in range(200):
    R_e = 0
    state = np.reshape(env.reset(), (1, state_dim))
    
    while True:
        action = ActorObj.net(state).numpy()
        observation, reward, done, info = env.step(action[0])
        
        R_e += reward
        state = np.reshape(observation, (1, state_dim))
        
        if done: break
    
    print ('Episode #' + str(ep+1) + '. Reward: ' + str(R_e))
    reward_store.append(R_e)   
    roll_avg_rew.append(np.mean(reward_store[-10:]))


rew_str = ''
for r in reward_store:
    rew_str = rew_str+str(r)+'\n'
rew_str = rew_str[:-1]
with open('./saved_models/trained_ep_rewards2.txt','a') as write_file:
    write_file.write(rew_str)



fig, ax = plt.subplots(figsize=(12,7))
plt.plot(list(range(len(reward_store))), reward_store, label='R_e', color='blue')
plt.plot(list(range(len(reward_store))), roll_avg_rew, label='Avg. R_e over 10 ep.', color='orange')
plt.plot(list(range(len(reward_store))), np.repeat(np.mean(reward_store),200), label='Avg. Reward', color='black')
plt.legend(loc='lower right')
plt.xlabel('Episode')
plt.ylabel('Total Reward per Episode')
plt.show()
'''


def shorthand(name):
    longlist = ['channelout_actoronly_pool2_do20.txt',
                'channelout_actoronly_pool2_do50.txt',
                'channelout_actoronly_pool2_nodo.txt',
                'relu_comp_channelout_actoronly_pool2_do50.txt',
                'relu_comp_channelout_actoronly_pool2_nodo.txt']
    
    shortlist = ['C-O, Pool2, p=0.2','C-O, Pool2, p=0.5','C-O, Pool2, p=0',
               'ReLU, p=0.5','ReLU, p=0']
    
    comp_list = np.array([name for i in longlist])==np.array(longlist)
    
    if not any(comp_list): return name
    
    else: return shortlist[np.argmax(comp_list)]

colors = ['red', 'orange', 'yellow', 'green', 'blue', 'magenta']

maindir = './reward_results/'
files = os.listdir(maindir)

all_results, all_averages = [], []

for filename in files:
    with open(maindir+filename,'r') as read_file:
        lines = read_file.read()
    
    data = lines.split('\n')
    
    reward_store, roll_avg_rew = [], []
    for l in data:
        reward_store.append(float(l))
        roll_avg_rew.append(np.mean(reward_store[-10:]))
        
    all_results.append(np.array(reward_store))
    all_averages.append(np.array(roll_avg_rew))

all_results, all_averages = np.array(all_results), np.array(all_averages)

ep_list = list(range(len(reward_store)))

fig, ax = plt.subplots(figsize=(12,7))

for i,result in enumerate(all_results):
    plt.plot(ep_list, result, label=shorthand(files[i]), color=colors[i])
    
plt.legend(loc='lower right')
plt.xlabel('Episode')
plt.ylabel('Total Reward per Episode')
plt.show()



fig, ax = plt.subplots(figsize=(12,7))

for i,result in enumerate(all_averages):
    plt.plot(ep_list, result, label=shorthand(files[i]), color=colors[i])

plt.legend(loc='lower right')
plt.xlabel('Episode')
plt.ylabel('Rolling Avg. Reward (10 episodes)')
plt.show()



