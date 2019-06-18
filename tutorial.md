## SAIDA RL Tutorial

In this tutorial, we will show you how to develop an agent and model  for reinforcement learning using SAIDA RL.  
In order to help you understand, we will give you an easy example of [AvoidReavers](/SAIDA_RL/AvoidObserver) scenario using Deep SARSA algorithm.

## Select a Scenario

SAIDA RL provides a few different types of scenario such as combat, avoidance, and survival. 
Choose one of these examples of from them.
In this tutorial, we will describe [AvoidReavers](/SAIDA_RL/AvoidObserver)  scenario as an example.

![avoid_reavers_explained.gif](/SAIDA_RL/assets/image/Avoid_reavers_explained.png)

## Define an Agent

The next thing to do is to define an agent. SAIDA RL has a flexible structure so you can apply various existing off the shelf RL agents(DQN with its variants, A2C, DDPG, PPO etc).  
To define a new agent, you must inherit the parent class [core.common.agent](/SAIDA_RL/api/core.common.html#module-core.common.agent) and implement each function.
Or you can use an algorithm that is already implemented.
From now on, we will demonstrate implementing [Deep SARSA](/SAIDA_RL/api/core.algorithm.html#module-core.algorithm.Deep_sarsa) agent, one of the classical algorithms of RL.

```python
from core.common.agent import Agent
from core.common.util import *
from collections import deque
import random
import numpy as np

# Inherit Agent class as a parent class
class DeepSARSAgent(Agent):
    
        # Detailed description about input parameters see API Doc
        def __init__(self, action_size, model, load_model=False, discount_factor=0.99, learning_rate=0.001,
                 epsilon=1, epsilon_decay=0.999, epsilon_min=0.01,
                 file_path='', training_mode=True, **kwargs):
        
        # Call constructor of parent's class
        super(DeepSARSAgent, self).__init__(**kwargs)
        
        # Set parameters from inputs
        self.load_model = load_model
        self.action_size = action_size
        self.model = model
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.training_mode = training_mode
        self.file_path = file_path
        
        # Set the epsilon as minimum value, if not training mode.
        if not self.training_mode:
            self.epsilon = self.epsilon_min

        # memory for train (S,A,R,S',A')
        self.observations = deque(maxlen=2)
        self.recent_observation = None
        self.recent_action = None


        if self.load_model and os.path.isfile(file_path):
            self.load_weights(file_path)
    
   # Get an action to be taken from observation
   def forward(self, observation):
        
        # Take a random acton with probability = epsilon
        if self.training_mode and np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
        else:
        # Take a best acton with probability = (1 - epsilon)
            state = np.float32(observation)
            q_values = self.model.predict(np.expand_dims(state, 0))
            action = np.argmax(q_values[0])

        # set memory for training
        self.recent_observation = observation
        self.recent_action = action

        return [action]
        
    # Updates the agent's network
    def backward(self, reward, terminal):
        
        self.observations.append([self.recent_observation, self.recent_action, reward, terminal])

        if self.step == 0:
            return

        # Decaying the epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Use a memory to train
        experience = self.observations.popleft()
        state = np.float32(experience[0])
        action = experience[1]
        reward = experience[2]
        done = experience[3]

        # Get next action on next state from current model
        next_state = np.float32(self.recent_observation)
        next_action = self.forward(next_state)

        # Compute Q values for target network update
        # Q(S,A) <- Q(S,A) + alpha(R + gammaQ(S',A') - Q(S,A))
        target = self.model.predict(np.expand_dims(state, 0))[0]
        if done:
            target[action] = reward
        else:
            target[action] = (reward + self.discount_factor *
                              self.model.predict(np.expand_dims(next_state, 0))[0][next_action])

        target = np.reshape(target, [1, self.action_size])

        self.model.fit(np.expand_dims(state, 0), target, epochs=1, verbose=0)
        return

    # Compile the model
   def compile(self, optimizer, metrics=[]):
        self.model.compile(optimizer=optimizer, loss='mse')
        return
    
    # Load trained weight from an HDF5 file.
   def load_weights(self, filepath) :
        self.model.load_weights(filepath)
        return

    # Save trained weight from an HDF5 file.
   def save_weights(self, filepath, overwrite):
        self.model.save_weights(filepath, overwrite)
        return
```

## Create an environment

Create an environment of selected scenario.

```python
from saida_gym.starcraft.avoidReavers import AvoidReavers
env = AvoidReavers(action_type=0, move_angle=30, move_dist=2, frames_per_step=24, vsersion=0, verbose=0)
```

{% include_relative Environment.md %}

### Define a processor for reshaping a data

Processor is a class used to process data exchanged between environment and agent.
This can be very useful if you want to use different form of the observations, actions, and rewards of the environment. 
For example, when you want to modify the raw reward given by environment, implement 'process_step' function and reshape the raw reward into the form you want.  
To define a processor class, [core.common.processor.Processor]() must be inherited.

```python
from core.common.processor import Processor
import numpy as np
import math

class ReaverProcessor(Processor):

    # Process an performed action 
    def process_action(self, action):
        "do what you want with the action"
        return action

    # Process the data given from environment after the step finished.
    def process_step(self, observation, reward, done, info):
        state_array = self.process_observation(observation)
        reward = self.reward_reshape(reward)
        return state_array, reward, done, info

    # Reshape your reward (Optional)
    def reward_reshape(self, reward):
        # Maybe I can give more incentive to the agent, when the agent has reached the goal.
        if reward == 1:  
            reward = reward * 2
        # And prevent to stay at safe start position rather then moving, give a small negative reward in every step.
        elif reward == -1 :
            reward  = -0.1
        
        return reward
    
    # Process the raw observation given from environment
    def process_observation(self, observation, **kwargs):
        # Raw observation data is the form of JSON(precisely, Protobuf).
        # Therefore, we need to transform it into an array or something can be calculated.
        
        # Define the size of state array. 
        # This time, I need 23 numbers to make the state, which consists of 5 factors of agent observation and 6 from 3 enemies one.
        STATE_SIZE = 5 + 3 * 6  
        s = np.zeros(STATE_SIZE) # Make an empty array.
        me = observation.my_unit[0] # Observation for Dropship (Agent)
        # Scale data set in order to learn fast and efficiently.
        s[0] = scale_pos(me.pos_x)  # X of coordinates
        s[1] = scale_pos(me.pos_y)  # Y of coordinates
        s[2] = scale_velocity(me.velocity_x)  # X of velocity
        s[3] = scale_velocity(me.velocity_y)  # y of coordinates
        s[4] = scale_angle(me.angle)  # Angle of head of dropship

        # Observation for Reavers(3 of them)
        for ind, ob in enumerate(observation.en_unit):
            s[ind * 6 + 5] = scale_pos(ob.pos_x - me.pos_x)  # X of relative coordinates
            s[ind * 6 + 6] = scale_pos(ob.pos_y - me.pos_y)  # Y of relative coordinates
            s[ind * 6 + 7] = scale_velocity(ob.velocity_x)  # X of velocity
            s[ind * 6 + 8] = scale_velocity(ob.velocity_y)  # Y of velocity
            s[ind * 6 + 9] = scale_angle(ob.angle)  # Angle of head of Reavers
            s[ind * 6 + 10] = scale_angle(1 if ob.accelerating else 0)  # True if Reaver is accelerating

        return s
    
    @staticmethod
    def scale_velocity(v):
        return v

    @staticmethod
    def scale_angle(angle):
        return (angle - math.pi) / math.pi

    @staticmethod
    def scale_pos(pos):
        return int(pos / 16)
        
```

## build a model 

Create a neural network model to learn. We can define the type, number, and shape of the layer as we want.

```python
from keras.models import Sequential
from keras.layers import Dense, Reshape
model = Sequential()
model.add(Dense(50, input_dim=STATE_SIZE, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(ACTION_SIZE, activation='linear'))
model.summary()
```

- STATE_SIZE : The length of state array 
- ACTION_SIZE : The number of available action. It is defined by the move angle value. 

### Create an Agent

Create an agent using the model, processor, etc. created so far.

 ```python
from core.algorithm.Deep_sarsa import DeepSARSAgent
agent = DeepSARSAgent(ACTION_SIZE, processor=ReaverProcessor(), model=model,
                      epsilon=EPSILON, epsilon_decay=EPSILON_DECAY, epsilon_min=EPSILON_MIN,
                      file_path="", load_model=LOAD_MODEL,
                      discount_factor=0.99, training_mode=TRAINING_MODE)
 ```
 - epsilon : the initial value of eplison for epsilon-greedy policy, which takes an random action with probability = epsilon.  
 - epsilon_decay : epsilon will be decayed in every step by epsilon  = epsilon * epsilon_decay 
 - epsilon_min : the minimum value of epsilon after decayed.
 - file_path : File path to load a trained model(h5f)
 - load_model : True if you want to load a trained model.
 - discount_factor : the value for discounted Q values.
 - training_mode : True if you want to train your model from a scratch.
 
 ### Complie a agent

Create the optimizer to be used for model and compile it.
I am going to use ADAM optimizer, which is  one of the most commonly used gradient descent optimization algorithms 
in machine learning.
```python
agent.compile(Adam(lr=LEARNING_RATE))
```

- LEARNING_RATE :  a hyper parameter that controls how much to change the model's weight from the estimated error.  

### Create a Callback
Define a callback class called on every event that occurs during learning. It is often useful for logging or drawing graphs during training.  
The events that occur during learning are as follows.

 - on_episode_begin : when every episode begins 
 - on_episode_end : when every episode ends
 - on_step_begin : when every step begins
 - on_step_end : when every step ends
 - on_action_begin : when every action begins
 - on_action_end : when every action ends
 
To define the callback class, you must inherit [core.common.callback.Callback](/SAIDA_RL/api/core.common.html#module-core.common.callback) class and implement each function.
Here, I am going to use the pre-defined callback function to draw a graph to observe the reward changing.

```python 
cb_plot = DrawTrainMovingAvgPlotCallback('./save_graph/' + FILE_NAME + '.png', plot_interval=100, time_window=10, l_label=['episode_reward'])
```
- plot_interval : interval of drawing a graph
- time_window : value for calculating moving-average 

### Start training

Start training using the agent and callback created so far.
If you execute this command below, the Starcraft instance will be executed automatically.
 
```python
agent.run(env, MAX_STEP_CNT, train_mode=training_mode, verbose=2, callbacks=callbacks)
```
- env : the envirionment that you created
- MAX_STEP_CNT : number of steps to execute until the game ends
- training_mode : True if you want to update your model. 
- verbose : log level
- callbacks : callback classes that you created. 

Once training is started successfully, this log will be displayed

```
Starting training for 500000 steps ...
206/500000: episode: 1, duration: 7.836s, episode steps: 206, steps per second: 26, episode reward: -32.000, mean reward: -0.155 [-1.000, 1.000], mean action: 5.932 [0.000, 12.000]
```
Log will be displayed in every episode and log contents are changed according to verbose.
- duration : the time during one episode
- episode steps : the number of steps executed by the agent during one episode
- steps per second : the time(second) taken per step
- episode reward : the sum of reward for one episode
- mean reward : average value of episode reward
- mean action : average value of agent's action that executed during an episode  

![avoid_reavers.gif](/SAIDA_RL/assets/image/avoid_reaver_2.png)

### Save the weight

Save the weight after training is done.

```python
import os
agent.save_weights(filepath=os.path.realpath('./save_model/file_name.h5f'), overwrite=True)
```

- filepath : path including name to save the weight
- overwrite : True if you want to overwrite existing file.
