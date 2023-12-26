"""
Reinforcement Learning
The next and final topic in this course covers Reinforcement Learning. This technique is different than many of the other machine learning techniques we have seen earlier and has many applications in training agents (an AI) to interact with enviornments like games. Rather than feeding our machine learning model millions of examples we let our model come up with its own examples by exploring an enviornemt. The concept is simple. Humans learn by exploring and learning from mistakes and past experiences so let's have our computer do the same.


Author: Tim Ruscica
Lecture: Jason Dong


Terminology
Before we dive into explaining reinforcement learning we need to define a few key peices of terminology.

Enviornemt In reinforcement learning tasks we have a notion of the enviornment. This is what our agent will explore. An example of an enviornment in the case of training an AI to play say a game of mario would be the level we are training the agent on.

Agent an agent is an entity that is exploring the enviornment. Our agent will interact and take different actions within the enviornment. In our mario example the mario character within the game would be our agent.

State always our agent will be in what we call a state. The state simply tells us about the status of the agent. The most common example of a state is the location of the agent within the enviornment. Moving locations would change the agents state.

Action any interaction between the agent and enviornment would be considered an action. For example, moving to the left or jumping would be an action. An action may or may not change the current state of the agent. In fact, the act of doing nothing is an action as well! The action of say not pressing a key if we are using our mario example.

Reward every action that our agent takes will result in a reward of some magnitude (positive or negative). The goal of our agent will be to maximize its reward in an enviornment. Sometimes the reward will be clear, for example if an agent performs an action which increases their score in the enviornment we could say they've recieved a positive reward. If the agent were to perform an action which results in them losing score or possibly dying in the enviornment then they would recieve a negative reward.


Example
Environment - What the character is navigating.
Agent - The character itself navigating the environment.
State - Where the character is or its current health.
Action - What the character performs while in the maze.
Reward - How well the character did will tell how much of a reward is recieved.

The most important part of reinforcement learning is determing how to reward the agent. After all, the goal of the agent is to maximize its rewards. This means we should reward the agent appropiatly such that it reaches the desired goal.


Q-Learning
Now that we have a vague idea of how reinforcement learning works it's time to talk about a specific technique in reinforcement learning called Q-Learning.

Q-Learning is a simple yet quite powerful technique in machine learning that involves learning a matrix of action-reward values. This matrix is often reffered to as a Q-Table or Q-Matrix. The matrix is in shape (number of possible states, number of possible actions) where each value at matrix[n, m] represents the agents expected reward given they are in state n and take action m. The Q-learning algorithm defines the way we update the values in the matrix and decide what action to take at each state. The idea is that after a succesful training/learning of this Q-Table/matrix we can determine the action an agent should take in any state by looking at that states row in the matrix and taking the maximium value column as the action.


Example
Creating a Table or Matrix Like Data Struction.
Contains every state as rows and columns as actions in all the different states.

Consider this example.

Let's say A1-A4 are the possible actions and we have 3 states represented by each row (state 1 - state 3).

A1	A2	A3	A4 - All possible actions performed in every given state
0	0	10	5 - 3 states in 3 rows, numbers represent predicted reward for action taken, this would be state row 1
5	10	0	0
10	5	0	0
Table is updated many times while exploring environment
If that was our Q-Table/matrix then the following would be the preffered actions in each state.  

State 1: A3

State 2: A2

State 3: A1

We can see that this is because the values in each of those columns are the highest for those states!

Learning the Q-Table
So that's simple, right? Now how do we create this table and find those values. Well this is where we will dicuss how the Q-Learning algorithm updates the values in our Q-Table.

I'll start by noting that our Q-Table starts of with all 0 values. This is because the agent has yet to learn anything about the enviornment.

Our agent learns by exploring the enviornment and observing the outcome/reward from each action it takes in each state. But how does it know what action to take in each state? There are two ways that our agent can decide on which action to take.

Randomly picking a valid action
Using the current Q-Table to find the best action.
Near the beginning of our agents learning it will mostly take random actions in order to explore the enviornment and enter many different states. As it starts to explore more of the enviornment it will start to gradually rely more on it's learned values (Q-Table) to take actions. This means that as our agent explores more of the enviornment it will develop a better understanding and start to take "correct" or better actions more often. It's important that the agent has a good balance of taking random actions and using learned values to ensure it does get trapped in a local maximum.

After each new action our agent wil record the new state (if any) that it has entered and the reward that it recieved from taking that action. These values will be used to update the Q-Table. The agent will stop taking new actions only once a certain time limit is reached or it has acheived the goal or reached the end of the enviornment.

Updating Q-Values
The formula for updating the Q-Table after each action is as follows:

Q[state,action]=Q[state,action]+α∗(reward+γ∗max(Q[newState,:])−Q[state,action])


State action is row and action as column
+ alpha (a means learning rate ensures q table isn't updated too much)
* (reward + gamma  * max Q of new State) (gamma means discount factor)
- Q state action
α stands for the Learning Rate. It tells when to update moves so it doesn't stay on highest reward
γ stands for the Discount Factor. lower focuses on current reward, higher looks towards the future

Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[new_state, :]) - Q[state, action])

Learning Rate α
The learning rate α is a numeric constant that defines how much change is permitted on each QTable update. A high learning rate means that each update will introduce a large change to the current state-action value. A small learning rate means that each update has a more subtle change. Modifying the learning rate will change how the agent explores the enviornment and how quickly it determines the final values in the QTable.

Discount Factor γ
Discount factor also know as gamma (γ) is used to balance how much focus is put on the current and future reward. A high discount factor means that future rewards will be considered more heavily.


To perform updates on this table we will let the agent explpore the enviornment for a certain period of time and use each of its actions to make an update. Slowly we should start to notice the agent learning and choosing better actions.

Q-Learning Example
For this example we will use the Q-Learning algorithm to train an agent to navigate a popular enviornment from the Open AI Gym. The Open AI Gym was developed so programmers could practice machine learning using unique enviornments. Intersting fact, Elon Musk is one of the founders of OpenAI!

Let's start by looking at what Open AI Gym is.


import gym   # all you have to do to import and use open ai gym!


Once you import gym you can load an enviornment using the line gym.make("enviornment").


env = gym.make('FrozenLake-v1')  # we are going to use the FrozenLake enviornment


There are a few other commands that can be used to interact and get information about the enviornment.


print(env.observation_space.n)   # get number of states
print(env.action_space.n)   # get number of actions
Run
16
4


env.reset()  # reset enviornment to default state
Run
0

action = env.action_space.sample()  # get a random action 
Run
action = env.action_space.sample()  # get a random action 
print(action)
3

new_state, reward, done, info = env.step(action)  # take action, notice it returns information about the action, (state to move into next, reward recieved by taking action [0 or 1], done is lose or win so true or fase to reset environment and no longer be in valid state in environment, shows some information)
#(Hover values to read more)


env.render()   # render the GUI for the enviornment, watching agent train slows it down 10-20x 
Run
(Up) #Action, Starting action is Up. Updates when ran repeatedly
SFFF #highlighted square is agent location, block 1. S = Start, F = Frozen (frozen lake) H=Hole to not fall into
FHFH
FFFH
HFFG


Frozen Lake Enviornment
Now that we have a basic understanding of how the gym enviornment works it's time to discuss the specific problem we will be solving.

The enviornment we loaded above FrozenLake-v0 is one of the simplest enviornments in Open AI Gym. The goal of the agent is to navigate a frozen lake and find the Goal without falling through the ice (render the enviornment above to see an example).

There are:

16 states (one for each square)
4 possible actions (LEFT, RIGHT, DOWN, UP)
4 different types of blocks (F: frozen, H: hole, S: start, G: goal)
Building the Q-Table
The first thing we need to do is build an empty Q-Table that we can use to store and update our values.


import gym
import numpy as np
import time

env = gym.make('FrozenLake-v1')
STATES = env.observation_space.n
ACTIONS = env.action_space.n


Q = np.zeros((STATES, ACTIONS))  # create a matrix with all 0 values 
Q
Run
array([[0., 0., 0., 0.], #initialize with all blank values
       [0., 0., 0., 0.], #at first actions are random, later on are more calulated based on Q table values
       [0., 0., 0., 0.], [16, 4]
       [0., 0., 0., 0.], [Row is state, column is action]
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]])


Constants
As we discussed we need to define some constants that will be used to update our Q-Table and tell our agent when to stop training.


EPISODES = 2000 # how many times to run the enviornment from the beginning, how many times to run around and explore environment
MAX_STEPS = 100  # max number of steps allowed for each run of enviornment, how many steps to take before cutting off because of case of agent performing same moves back and fourth so it can start again with different q values

LEARNING_RATE = 0.81  # learning rate, higher value is faster
GAMMA = 0.96 


Picking an Action
Remember that we can pick an action using one of two methods:

Randomly picking a valid action
Using the current Q-Table to find the best action.
Here we will define a new value  ϵ  that will tell us the probabillity of selecting a random action. This value will start off very high and slowly decrease as the agent learns more about the enviornment.


epsilon = 0.9  # start with a 90% chance of picking a random action, 90% chance random, 10% chance look at Q table action
#when env is explored enough, epsilon is slowly decreased for more optimal route of things to do

# code to pick action
if np.random.uniform(0, 1) < epsilon:  # we will check if a randomly selected value is less than epsilon. pick random value between 0 and 1 but less than epsilon
    action = env.action_space.sample()  # take random action store action here
else:
    action = np.argmax(Q[state, :])  # use Q table to pick best action based on current values and find colunn it's in


Updating Q Values
The code below implements the formula discussed above.


Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[new_state, :]) - Q[state, action])


Putting it Together
Now that we know how to do some basic things we can combine these together to create our Q-Learning algorithm,


import gym
import numpy as np
import time

env = gym.make('FrozenLake-v1')
STATES = env.observation_space.n
ACTIONS = env.action_space.n

Q = np.zeros((STATES, ACTIONS))

EPISODES = 1500 # how many times to run the enviornment from the beginning
MAX_STEPS = 100  # max number of steps allowed for each run of enviornment

LEARNING_RATE = 0.81  # learning rate
GAMMA = 0.96

RENDER = False # if you want to see training set to true, draw environment

epsilon = 0.9


rewards = [] #store all rewards
for episode in range(EPISODES): #for every episode

  state = env.reset() #reset to start state
  for _ in range(MAX_STEPS): #explore environment up to maximum steps
    
    if RENDER: 
      env.render() #render environment

    if np.random.uniform(0, 1) < epsilon: 
      action = env.action_space.sample()  #take action
    else:
      action = np.argmax(Q[state, :])

    next_state, reward, done, _ = env.step(action) #take action

    Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action]) #update Q value based on reward


    state = next_state #current state = next state

    if done: #if agent lost
      rewards.append(reward) #append reward from last step, 1 reward for every valid block step and 0 reward if lost
      epsilon -= 0.001 #reduce epsilon
      break  # reached goal

print(Q) #Q Table
print(f"Average reward: {sum(rewards)/len(rewards)}:")
# and now we can see our Q values!
Run
[[3.26868911e-01 8.87781051e-03 1.09580349e-02 1.14887692e-02] #Q Table values
 [2.03874659e-03 4.11519045e-03 1.46429087e-03 2.79456888e-01]
 [2.88549841e-03 3.99980632e-03 2.76890906e-03 1.55005620e-01]
 [1.27595298e-03 2.44815318e-03 2.90198141e-03 9.61694072e-02]
 [2.77833478e-01 2.38044826e-03 7.40681300e-03 8.91485738e-03]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [1.93395150e-01 1.74651707e-04 1.70637865e-04 9.36554387e-05]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [2.55987505e-03 3.60774633e-03 2.53679302e-03 3.61022869e-01]
 [3.67051824e-03 7.24721315e-01 4.95674737e-03 4.25886025e-03]
 [8.83162161e-01 2.03911322e-03 1.76476018e-03 1.24267637e-03]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [2.04390769e-02 1.61194850e-02 6.08816240e-01 6.01596481e-02]
 [6.82102670e-02 9.81291913e-01 1.69588679e-01 1.30830002e-01]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]]
Average reward: 0.31333333333333335:

# we can plot the training progress and see how the agent improved, graph average reward over 100 steps
import matplotlib.pyplot as plt

def get_average(values):
  return sum(values)/len(values)

avg_rewards = []
for i in range(0, len(rewards), 100): #over 100 episodes
  avg_rewards.append(get_average(rewards[i:i+100])) 

plt.plot(avg_rewards)
plt.ylabel('average reward')
plt.xlabel('episodes (100\'s)')
plt.show()
Run
Shows plot doing poorly in beginning because epsilon value is high due to random actions. Around 600 episodes epsilon increases and reward is highest but eventually platues and bounces up and down

Sources
Violante, Andre. “Simple Reinforcement Learning: Q-Learning.” Medium, Towards Data Science, 1 July 2019, https://towardsdatascience.com/simple-reinforcement-learning-q-learning-fcddc4b6fe56.

Openai. “Openai/Gym.” GitHub, https://github.com/openai/gym/wiki/FrozenLake-v0.

Check out TensorFlow website for learning TensorFlow with very easy to understand examples
From TensorFlow: Generative > DeepDream