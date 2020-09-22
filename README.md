# MountainCar-v0
A car is on a one-dimensional track, positioned between two "mountains". The goal is to drive up the mountain on the right; however, the car's engine is not strong enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build up momentum.

## OpenAI Gym
OpenAI Gym is a toolkit for developing and comparing reinforcement learning algorithms. It supports teaching agents everything from walking to playing games like pong or pinball. Gym is an open source interface to reinforcement learning tasks.

## Reinforcement learning algorithms
- Deep Q-learning (off-policy, model-free)

## Demo video
https://www.youtube.com/watch?v=9-wYKN2I5G0

## Requirements
- [Python 3.6 or 3.7](https://www.python.org/downloads/release/python-360/)
- [CUDA Toolkit 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base)
- [cuDNN v7.6.5](https://developer.nvidia.com/cuda-10.1-download-archive-base)
- [Pipenv](https://pypi.org/project/pipenv/)

## How to install the packages
You can install the required Python packages using the following command:
- `pipenv sync`

## How to run it
You can run the script using the following command: 
- `pipenv run python mountain_car_v0_dqn.py`
