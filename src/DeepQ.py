import numpy as np
from matplotlib import pyplot as plt
import gym
import logging

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Activation
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.metrics import mean_squared_error
except Exception as e:
    logging.error(f"Error evaluating model accuracy: {e}")

class DQNAgent():
    def __init__(self, state_size, action_size, lr = 0.001, gamma = 0.99,
                 explore_proba = 1.0, explore_proba_decay = 0.005,
                 batch_size = 32, memory_buffer = list(),
                 max_memory_buffer = 2000) -> None:
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.explore_proba = explore_proba
        self.explore_proba_decay = explore_proba_decay
        self.batch_size = batch_size
        self.memory_buffer = memory_buffer
        self.max_memory_buffer = max_memory_buffer

        self.model = Sequential([
            Dense(units=24, input_dim=self.state_size, activation='relu'),
            Dense(units=24, activation='relu'),
            Dense(units=self.action_size, activation='linear')
        ])

        self.model.compile(loss='mse', optimizer=Adam(lr=self.lr))

    def compute_action(self, current_state):
        if np.random.uniform(0, 1) < self.explore_proba:
            return np.random.choice(range(self.n_actions))
        else:
            q_values = self.model.predict(current_state)[0]
            return np.argmax(q_values)
        
    def update_explore_proba(self):
        self.explore_proba = self.explore_proba * np.exp(-self.explore_proba_decay)

    def store_episode(self, current_state, action, reward, next_state, done):
        self.memory_buffer.append({
            "current_state":current_state,
            "action":action,
            "reward":reward,
            "next_state":next_state,
            "done" :done
        })

        if len(self.memory_buffer) > self.max_memory_buffer:
            self.memory_buffer.pop(0)

    def train(self):
        np.random.shuffle(self.memory_buffer)
        batch_sample = self.memory_buffer[0:self.batch_size]

        for exp in batch_sample:
            q_current_state = self.model.predict(exp['current_state'])
            q_target = exp['reward']
            if not exp['done']:
                q_target = q_target + self.gamma * np.max(self.model.predict)
            q_current_state[0][exp['action']] = q_target
            self.model.fit(exp['current_state'], q_current_state, verbose=0)

env = gym.make("CartPole-v1")
# We get the shape of a state and the actions space size
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
# Number of episodes to run
n_episodes = 400
# Max iterations per epiode
max_iteration_ep = 500
# We define our agent
agent = DQNAgent(state_size, action_size)
total_steps = 0

# We iterate over episodes
for e in range(n_episodes):
    # We initialize the first state and reshape it to fit 
    #  with the input layer of the DNN
    current_state = env.reset()
    current_state = np.array([current_state])
    for step in range(max_iteration_ep):
        total_steps = total_steps + 1
        # the agent computes the action to perform
        action = agent.compute_action(current_state)
        # the envrionment runs the action and returns
        # the next state, a reward and whether the agent is done
        next_state, reward, done, _ = env.step(action)
        next_state = np.array([next_state])
        
        # We sotre each experience in the memory buffer
        agent.store_episode(current_state, action, reward, next_state, done)
        
        # if the episode is ended, we leave the loop after
        # updating the exploration probability
        if done:
            agent.update_exploration_probability()
            break
        current_state = next_state
    # if the have at least batch_size experiences in the memory buffer
    # than we tain our model
    if total_steps >= batch_size:
        agent.train(batch_size=batch_size)

def make_video():
    env_to_wrap = gym.make('CartPole-v1')
    env = wrappers.Monitor(env_to_wrap, 'videos', force = True)
    rewards = 0
    steps = 0
    done = False
    state = env.reset()
    state = np.array([state])
    while not done:
        action = agent.compute_action(state)
        state, reward, done, _ = env.step(action)
        state = np.array([state])            
        steps += 1
        rewards += reward
    print(rewards)
    env.close()
    env_to_wrap.close()
make_video()