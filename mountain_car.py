# -*- coding: utf-8 -*-
import gym

import numpy as np
from brain import Brain


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
    num_episode_steps = 200  # constant value
    action_size = env.action_space.n
    state_size = env.observation_space.shape[0]

    # Creates the brain
    brain = Brain(state_size=state_size, action_size=action_size)

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
            action = brain.act(state)

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
            brain.memorize(state, action, reward, next_state, done)

            # Updates the state
            state = next_state

            if done:
                print("Episode %d/%d finished after %d episode steps with total reward = %f."
                      % (episode + 1, num_episodes, episode_step + 1, total_reward))
                break

            elif episode_step >= num_episode_steps - 1:
                print("Episode %d/%d timed out at %d with total reward = %f."
                      % (episode + 1, num_episodes, episode_step + 1, total_reward))

            if len(brain.memory) > batch_size:
                brain.replay(batch_size)

        # Stores memory on disk
        brain.save("mountain_car.h5")

    # Closes the environment
    env.close()





