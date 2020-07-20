import numpy as np

# This implementation of SumSegmentTree was inspired by this article:
# https://pylessons.com/CartPole-PER/


class SumSegmentTree(object):
    '''
    Class for sum segment tree data structure

    Attributes
    ----------
    capacity: int
        The size of the segment tree array (number of leaf nodes that hold transitions)

    Methods
    -------
    add(priority, transition)
        Add an experience to the replay buffer

    update(tree_index, priority)
        Updates the priority of an experience in the SegmentSumTree

    get_leaf(value)
        For a given value, get the leaf index, priority value and experience
    '''

    def __init__(self, capacity):
        self._capacity = capacity
        self._tree = np.zeros(2 * capacity - 1)
        self._data = np.zeros(capacity, dtype=object)
        self._pointer = 0

    def add(self, priority, transition):
        '''Adds the priority to the SegmentSumTree and experience to the replay buffer

        Parameters
        ----------
        priority: float
            the priority of the experience
        transition: tuple
            the experience to add to the replay buffer
        '''
        tree_index = self._pointer + self._capacity - 1
        self._data[self._pointer] = transition
        self.update(tree_index, priority)
        self._pointer += 1

        if self._pointer >= self._capacity:
            self._pointer = 0

    def update(self, tree_index, priority):
        '''Updates the leaf priority score and propogates the change through the tree

        Parameters
        ----------
        tree_index: int
            the index of the tree to update the priority for
        priority: floatt
            the new priority for the experience
        '''
        priority_delta = priority - self._tree[tree_index]
        self._tree[tree_index] = priority

        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self._tree[tree_index] += priority_delta

    def get_leaf(self, value):
        '''Get the leaf node from a tree as well as the priority value and experience

        Parameters
        ----------
        value: float
            priority sum value to seach tree on

        Returns
        -------
        tuple
            The leaf index, priority value, and experience
        '''
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self._tree):
                leaf_index = parent_index
                break
            else:  # downward search, always search for a higher priority node
                if value <= self._tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    value -= self._tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self._capacity + 1

        return leaf_index, self._tree[leaf_index], self._data[data_index]

    @property
    def total_priority(self):
        '''Get total segment tree sum'''
        return self._tree[0]

    @property
    def tree(self):
        '''Get tree'''
        return self._tree

    @property
    def capacity(self):
        '''Get capacity'''
        return self._capacity
