import numpy as np
from .SumSegmentTree import SumSegmentTree

# This class was inspired by the following article
# https://pylessons.com/CartPole-PER/


class PER(object):
    '''
    Class for Prioritized Replay Buffer

    Attributes
    ----------
    capacity: int
        The max size of the replay buffer
    alpha: float
        Amount of prioritization to use (0 - none, 1 - full)
    epsilon: float  
        Hyperparameter to avoid some experiences having 0 probability
        of being chosen

    Methods
    -------
    store(experience)
        Adds an experience to the replay buffer
    sample(batch_size)
        Samples a batch of experiences from the replay buffer
    batch_update(batch_idxs, errors)
        Updates a batch of experiences/priorities in the replay buffer
    '''

    def __init__(self, capacity, alpha=0.6, epsilon=0.01):
        self._absolute_error_upper = 1.0
        self._segtree = SumSegmentTree(capacity)
        self._epsilon = epsilon
        self._alpha = alpha

    def store(self, experience):
        '''Add an experience to the replay buffer

        Parameters
        ----------
        experience: tuple
            An experience from the environment
        ''' 
        max_priority = np.max(self._segtree.tree[-self._segtree.capacity:])

        if max_priority == 0:
            max_priority = self._absolute_error_upper

        self._segtree.add(max_priority, experience)

    def sample(self, batch_size):
        '''Samples a batch of experiences from the replay buffer

        Parameters
        ----------
        batch_size: int
            The size of the batch of experiences to fetch

        Returns
        -------
        tuple(list, list)
            A list of tree indexes that hold the priorities and the 
            batch of experiences
        '''
        minibatch = []
        batch_idxs = np.empty((batch_size,), dtype=np.int32)
        priority_segment = self._segtree.total_priority / batch_size

        for i in range(batch_size):
            while True:
                a, b = priority_segment * i, priority_segment * (i + 1)
                value = np.random.uniform(a, b)
                index, _, transition = self._segtree.get_leaf(value)
                if (transition != 0):
                    break

            batch_idxs[i] = index
            minibatch.append(transition)

        return batch_idxs, minibatch

    def batch_update(self, batch_idxs, errors):
        '''updates a batch of experiences in the replay buffer

        Parameters
        ----------
        batch_idxs: list
            A list of indexes to update the experience priorities for
        errors: list
            List of errors used to update priorities for each experience
            in the batch
        '''
        errors += self._epsilon
        clipped_errors = np.minimum(errors, self._absolute_error_upper)
        priorities = np.power(clipped_errors, self._alpha)

        for b, p in zip(batch_idxs, priorities):
            self._segtree.update(b, p)
