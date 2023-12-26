%tensorflow_version 2.x  # this line is not required unless you are in a notebook

!pip install tensorflow_probability==0.8.0rc0 --user --upgrade #restart runtime after this command and continue on to imports

import tensorflow_probability as tfp  # We are using a different module from tensorflow this time that deals with probability
import tensorflow as tf #make sure version above is compatible with tensorFlow

tfd = tfp.distributions  # making a shortcut for later on, tensorFlow probability distribution model
initial_distribution = tfd.Categorical(probs=[0.2, 0.8])  # Refer to point 2 above, probability of 80% and 20%. First day in sequence has 80% chance of being cold and 20% chance after
transition_distribution = tfd.Categorical(probs=[[0.5, 0.5], #2 states so 2 probabilities of landing on each state at beginning of sequence, transition probability. 70% chance of cold day again and 30% chance of hot day
                                                 [0.2, 0.8]])  # refer to points 3 and 4 above
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])  # refer to point 5 above. when doing standard deviation loc stands for average or mean - 0 on hot day and 15 on cold day. Standard deviation on cold day is 5 so range from -5 to 5 degrees and on hot day is 10 so range from 5 to 25 degrees and average temp is 15.

# the loc argument represents the mean and the scale is the standard devitation

model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7) #how many days to predict so number of times to step through probability cycle and run model.