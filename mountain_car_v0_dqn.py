"""
MountainCar-v0 -- Deep Q-learning with Experience Replay (DQN)
"""
import os
import random
from collections import deque

import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class Agent:
    def __init__(self, state_size, action_size, memory_size=50000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state)[0])

    def replay(self, batch_size=32):
        batch_train_x = []
        batch_train_y = []

        for state, action, reward, next_state, done in random.sample(self.memory, batch_size):
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target

            batch_train_x.append(state[0])
            batch_train_y.append(target_f[0])

        self.model.fit(
            np.array(batch_train_x),
            np.array(batch_train_y),
            epochs=1,
            verbose=0
        )

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def load_weights(self, weights_file):
        self.epsilon = self.epsilon_min
        self.model.load_weights(weights_file)

    def save_weights(self, weights_file):
        self.model.save_weights(weights_file)


if __name__ == "__main__":
    # Flag used to enable or disable screen recording
    recording_is_enabled = False

    # Initializes the environment
    env = gym.make('MountainCar-v0')

    # Records the environment
    if recording_is_enabled:
        env = gym.wrappers.Monitor(env, "recording", video_callable=lambda episode_id: True, force=True)

    # Defines training related constants
    batch_size = 32
    num_episodes = 1000
    num_episode_steps = env.spec.max_episode_steps  # constant value
    action_size = env.action_space.n
    state_size = env.observation_space.shape[0]

    # Creates the brain
    agent = Agent(state_size=state_size, action_size=action_size)

    # Loads the model if exists
    if os.path.isfile("mountain-car-v0.h5"):
        agent.load_weights("mountain-car-v0.h5")

    for episode in range(num_episodes):
        # Defines the total reward per episode
        total_reward = 0

        # Resets the environment
        observation = env.reset()

        # Gets the state
        state = np.reshape(observation, [1, state_size])

        for episode_step in range(num_episode_steps):
            # Renders the screen after new environment observation
            env.render(mode="human")

            # Gets a new action
            action = agent.act(state)

            # Takes action and calculates the total reward
            observation, reward, done, _ = env.step(action)

            # Recalculates the reward
            if observation[1] > state[0][1] >= 0 and observation[1] >= 0:
                reward = 20
            if observation[1] < state[0][1] <= 0 and observation[1] <= 0:
                reward = 20
            if done and episode_step < num_episode_steps - 1:
                reward += 10000
            else:
                reward -= 25

            total_reward += reward

            # Gets the next state
            next_state = np.reshape(observation, [1, state_size])

            # Memorizes the experience
            agent.memorize(state, action, reward, next_state, done)

            # Updates the state
            state = next_state

            # Allows agent to learn from previous experiences
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            if done:
                print("Episode %d/%d finished after %d episode steps with total reward = %f."
                      % (episode + 1, num_episodes, episode_step + 1, total_reward))
                break

            elif episode_step >= num_episode_steps - 1:
                print("Episode %d/%d timed out at %d with total reward = %f."
                      % (episode + 1, num_episodes, episode_step + 1, total_reward))

        # Saves the weights
        agent.save_weights("mountain-car-v0.h5")

    # Closes the environment
    env.close()
