#Q-Learning Example
import gym   # all you have to do to import and use open ai gym!

env = gym.make('FrozenLake-v0')  # we are going to use the FrozenLake enviornment

print(env.observation_space.n)   # get number of states
print(env.action_space.n)   # get number of actions

env.reset()  # reset enviornment to default state

action = env.action_space.sample()  # get a random action 

new_state, reward, done, info = env.step(action)  # take action, notice it returns information about the action

env.render()   # render the GUI for the enviornment 


#Building Q Table
import gym
import numpy as np
import time

env = gym.make('FrozenLake-v0')
STATES = env.observation_space.n
ACTIONS = env.action_space.n

Q = np.zeros((STATES, ACTIONS))  # create a matrix with all 0 values 
Q

#Constants
EPISODES = 2000 # how many times to run the enviornment from the beginning
MAX_STEPS = 100  # max number of steps allowed for each run of enviornment

LEARNING_RATE = 0.81  # learning rate
GAMMA = 0.96


#Picking an Action
epsilon = 0.9  # start with a 90% chance of picking a random action

# code to pick action
if np.random.uniform(0, 1) < epsilon:  # we will check if a randomly selected value is less than epsilon.
    action = env.action_space.sample()  # take random action
else:
    action = np.argmax(Q[state, :])  # use Q table to pick best action based on current values


#Updating Q Values
Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[new_state, :]) - Q[state, action])

