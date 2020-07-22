import numpy as np

np.random.seed(1262)

class IndividualBuffers():
    '''
    Class for replay buffer object. It was too much trouble with shape 
    manipulation if all types of values are handled in one buffer. So this
    doesn't look as spiffy but it works much smoother.
    
    Parameters:
        - maxlength (Integer): Maximum number of experiences that can be stored.
        - state_shape (Integer): Number of state dimensions.
        - action_shape (Integer): Number of action dimensions.
    '''
    
    def __init__(self, maxlength, state_shape, action_shape):
        # Array for buffer structures.
        self.state_buffer = np.zeros((maxlength,state_shape))
        self.action_buffer = np.zeros((maxlength,action_shape))
        self.new_state_buffer = np.zeros((maxlength,state_shape))
        self.scalar_buffer = np.zeros((maxlength,2))
        
        self.maxlength = maxlength
        self.total_count = 0
    
    def addTransition(self, transition):
        # Since the sampling is random anyway, I rewrite from bottom-up
        # instead of popping the bottom and appending to the top.
        count = self.total_count%self.maxlength
        
        self.state_buffer[count] = transition[0]
        self.action_buffer[count]  = transition[1]
        self.new_state_buffer[count]  = transition[2]
        self.scalar_buffer[count]  = [transition[3], transition[4]]
        
        self.total_count+=1
        
    def sample(self, batch_size):
        
        inds = np.random.choice(min(self.total_count,self.maxlength), batch_size)
        
        states = self.state_buffer[inds]
        actions = self.action_buffer[inds]
        new_states = self.new_state_buffer[inds]
        done_vals = self.scalar_buffer[:,0][inds]
        rewards = self.scalar_buffer[:,1][inds]
        
        return states, actions, new_states, done_vals, rewards



class makeOUNoise():
    '''
    Class for noise object.
    
    Parameters:
        - noise_type (String): Type of noise to use. OU, normal, or none.
        - mu (ndarray or Float): Mean of noise.
        - sigma (ndarray or Float): Standard deviation of noise.
        - dt (Float): Timestep constant.
        - theta (Float): OU noise parameter.
    '''
    
    def __init__(self,noise_type,mu=0,sigma=0.2,dt=1e-2,theta=0.15):
        self.noise_type = noise_type
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.theta = theta
        
        self.last_val = 0
        
    def makeNoise(self):
        
        
        if self.noise_type=='OU':
            '''
            Ornstein–Uhlenbeck noise by forward Euler method
            '''
            new_val = self.last_val + self.dt*(-1)*(self.last_val-self.mu)*\
                self.theta + self.sigma*np.sqrt(self.dt)*\
                    np.random.normal(size=self.mu.shape)
            
            self.last_val = new_val
            
            return new_val
    
        elif self.noise_type=='normal': 
            if isinstance(self.mu, np.ndarray): mean = self.mu[0]
            if isinstance(self.sigma, np.ndarray): std_dev = self.sigma[0]
            return np.random.normal(loc=mean, scale=std_dev)
        
        else: return 0


# For full sequence.
'''
def makeOUNoise(mu=0,sigma=0.2,T=1,dt=1e-2,theta=0.15):
    
    step_total = int(T/dt)
    step_arr = np.linspace(0,T,step_total)
    
    process_vals = np.zeros(step_total)
    for i, val in enumerate(process_vals[:-1]):
        process_vals[i+1] = val + dt*(-1)*(val-mu)*theta + sigma*np.sqrt(dt)*\
            np.random.normal(size=mu.shape)#np.random.randn()
    
    #fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    #ax.plot(step_arr, process_vals, lw=2)
    
    return process_vals
'''