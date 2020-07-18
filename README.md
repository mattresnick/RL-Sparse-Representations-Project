# RL-Sparse-Representations-Project

Some notes:

I changed the way I created the actor model between SPE and DR implementation. In SPE, I used the functional API and in DR I created a class for it, and the latter works much better, so feel free to port it over. The only major difference in training is that I created special dropouts for the functional way in order to turn the training parameter on and off (it would have been more difficult to send the signal during training otherwise, since I couldn't rewrite the call function for the functional API way). The latter way does not require dropout modifications, and further training is off by default so I only have one call that turns it on in the "training" code.

For DR, you'll find the relevant call by searching for the functions in the "training" code. It just calls the "grad" option per layer during gradient calculation. As I mentioned, there's another way to do it, which is by calling the relevent regularizer function as the "kernel_regularizer" option when creating a particular layer in the actor model. The function would need to be modified to work this way (i.e. by only taking in the layer's weight matrix and still doing the same calculation with the layer's activation output). 

I tried to comment up the code as much as possible, but there will inevitably be parts of the code that are hard to read or confusing (and possibly, issues that I missed). If you encounter issues or just have questions, be in touch.
