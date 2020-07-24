# Descriptions/Instructions

All implementation discussed in our report (and some additional implementations that didn't make it) can be found in this folder. 


Some files are self-explanatory: DistributionalRegularizers.py contains the implementation for exponential, Gaussian, and Bernoulli
Distributional Regularizers, CustomActivations contains the implementation for channel-out with max and absolute max pool functions,
as well as a custom layer for SELU (the implementation is not ours, we just needed the existing implementation in Tensorflow Layer
form).


Some files are not as self explanatory, but I tried to keep the documentation as thorough as I could, and I will otherwise do my
best to explain everything in a general sense here. If there's any question as to how to run something or how a particular piece of
the code works in this folder, feel free to contact me (Matt). I tried to keep the additional libraries to a minimum, so the environment 
we made for homework 0 should be sufficient except "tensorflow_addons" is required for Maxout. I believe that's the only additional
dependency.


Any files I feel need some explanation I will describe below:




-- main.py --

This file is like the control panel for all implementations. It controls most aspects of training and testing. Very high level options,
such as whether to implement PackNet or Distributional Regularizers, can be found at the top as flag variables. Most model hyperparameters
are controlled by the first set of dictionaries, and most training parameters can be controlled by the later dictionary (they are all
labeled in the comments). If you wish to test PackNet on multiple environments (as designed) simply add to the list of environment names
and the list of seeds. For instance, training on pendulum over three seeds requires the pendulum name in the list three times over, and 
three different seeds in the seed list. 

-- training.py --

Contains the training function and any other relevent functions to facilitate training (and a function to test a given environment and 
generate/save results). There's a lot to this code, but as I mentioned pretty much everything that isn't trivial has a comment describing
its function. The only thing that may need to be modified here is, in the block of code which calculates the actor gradient, under the DR
flag conditional, the exponential distributional regularizer is called. If you wish to use the Gaussian or Bernoulli DRs, you will just 
need to change this call to the appropriate name.

-- ActorCritic.py --

This file contains the ActorCritic class definition and the Actor and Critic model classes themselves. Again, there's a lot to this code
but it's heavily documented. Most of the PackNet implementation can be found here (some is also within the training file, but the bulk
of it is here).



----
And one last thing of note: you may be wondering what's up with the os import/warning suppression going on at the top of some of the code. 
Tensorflow was having some sort of catastrophic issue with HDF5. I was using a new, clean environment though, and could not find out the
cause. However, whatever the warning was stemming from was not an issue for the code it seems so I surpressed it.
