#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 7 21:10:43 2019

Adapted from code created as part of Udemy course: Artificial Intelligence 2018: Build the Most Powerful AI
(https://www.udemy.com/artificial-intelligence-ars/)
"""
# Augmented Random Search

####### NEEDED INSTALLATIONS #########
##
## pip install gym==0.10.5
## pip install pybullet==2.0.8
## conda install -c conda-forge ffmpeg
##  
######################################
# Importing the libraries
import os
import time
import warnings

import numpy as np
import gym
import pybullet_envs
import matplotlib as mpl
import matplotlib.pyplot as plt

from gym import wrappers

# Turn off Deprecation and Future warnings for now
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set the Hyperparameters

class Hp():
    
    def __init__(self, nb_steps, episode_length, learning_rate, nb_directions, noise, env_name):
        self.nb_steps = nb_steps # number of steps to follow
        self.episode_length = episode_length # length of the episode - the max time the AI will move
        self.learning_rate = learning_rate # set the learning rate to control how fast the AI learns
        self.nb_directions = nb_directions # number of perterbations applied to each of the weights (in both directions, positive ,and negative)
        self.nb_best_directions = nb_directions 
        assert self.nb_best_directions <= self.nb_directions
        self.noise = noise # sigma (standard deviation) in the gaussian distribution
        self.seed = 42
        self.env_name = env_name # pyBullet open source implementation of openai gym env
     
# Normalize states

class Normalizer():
    
    def __init__(self, nb_inputs):
        '''
        Initializes the Normalizer class
        
        :param nb_inputs: integer, number of inputs
        '''
        
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)
    
    def observe(self, x):
        
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)
    
    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std

# Build the AI

class Policy():
    
    def __init__(self, input_size, output_size):
        '''
        Initializes the Policy class
        
        :param input_size: int, size of the input array
        :param output_size: int, size of the output array
        '''
        
        self.theta = np.zeros((output_size, input_size))
    
    def evaluate(self, input, delta = None, direction = None):
        '''
        Evaluates the policy results
        
        :param delta: float, change in the reward
        :param direction: boolean, indicates positive or negative progress
        '''
        
        if direction is None:
            return self.theta.dot(input)
        elif direction == "positive":
            return (self.theta + hp.noise*delta).dot(input)
        else:
            return (self.theta - hp.noise*delta).dot(input)
    
    def sample_deltas(self):
        '''
        Returns a sample of the deltas
        
        :return: sample of deltas
        '''
        
        return [np.random.randn(*self.theta.shape) for _ in range(hp.nb_directions)]
    
    def update(self, rollouts, sigma_r):
        '''
        Updates the results
        
        :param rollouts: rewards in both positive and negative directions
        :param sigma_r: float, standard deviation of all the rewards
        '''
        
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d
        self.theta += hp.learning_rate / (hp.nb_best_directions * sigma_r) * step

# Explore the policy in one specific direction, over one episode

def explore(env, normalizer, policy, direction = None, delta = None):
    '''
    Explores the policy on one specific direction over one episode
    
    :param env: string, environment to run on
    :param normalizer: class, normalizes the results
    :param policy: class, policy framework to follow
    :param direction: string, positive or negative perturbations
    :param delta: float, changes to the perturbations
    '''
    
    state = env.reset()
    done = False
    num_plays = 0.
    sum_rewards = 0
    while not done and num_plays < hp.episode_length:
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = policy.evaluate(state, delta, direction)
        state, reward, done, _ = env.step(action)
        reward = max(min(reward, 1), -1)
        sum_rewards += reward
        num_plays += 1
    return sum_rewards

# Train the AI

def train(env, policy, normalizer, hp):
    '''
    Trains the model
    
    :param env: string, environment to run on
    :param policy: class, policy framework to follow
    :param normalizer: xxx
    :param hp: class, hyperparameters to apply to the model
    '''
    
    fig = plt.figure(figsize=(16, 9))

    rewards = []
    st_dev_pos = []
    st_dev_neg = []
    steps = []
    
    for step in range(hp.nb_steps):
        
        # Initialize the perturbations deltas and the positive/negative rewards
        deltas = policy.sample_deltas()
        positive_rewards = [0] * hp.nb_directions
        negative_rewards = [0] * hp.nb_directions
        
        # Get the positive rewards in the positive directions
        for k in range(hp.nb_directions):
            positive_rewards[k] = explore(env, normalizer, policy, direction = "positive", delta = deltas[k])
        
        # Get the negative rewards in the negative/opposite directions
        for k in range(hp.nb_directions):
            negative_rewards[k] = explore(env, normalizer, policy, direction = "negative", delta = deltas[k])
        
        # Gather all the positive/negative rewards to compute the standard deviation of these rewards
        all_rewards = np.array(positive_rewards + negative_rewards)
        sigma_r = all_rewards.std()
        
        # Sort the rollouts by the max(r_pos, r_neg) and selecting the best directions
        scores = {k:max(r_pos, r_neg) for k,(r_pos,r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
        order = sorted(scores.keys(), key = lambda x:scores[x])[:hp.nb_best_directions]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]
        
        # Update the policy
        policy.update(rollouts, sigma_r)
        
        # Print the final reward of the policy after the update
        reward_evaluation = explore(env, normalizer, policy)
#        print('Step:', step, 'Reward:', reward_evaluation)
        print('Step: {} Reward: {} StDev: {}'.format(step, reward_evaluation, sigma_r))

        rewards.append(reward_evaluation)
        st_dev_pos.append(reward_evaluation + sigma_r)
        st_dev_neg.append(reward_evaluation - sigma_r)
        steps.append(step)
        
    ##########
    # TODO: Save rewards for this run to csv so they can all be combined into one graph later
    
    
    ##########
    # Plot rewards
    plt.plot(steps, rewards, 'k', color='#003366')
    plt.fill_between(steps, st_dev_neg, st_dev_pos,
                     alpha=0.5, edgecolor='#003366', facecolor='skyblue',
                     linewidth=1, linestyle='dashdot', antialiased=True)
    plt.xlabel('Step', fontsize=20)
    plt.ylabel('Average Reward', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.suptitle('Reward Function: Env - {}'.format(hp.env_name), y=0.95, fontsize=18)
    plt.title('Eps Length - {}; No. Steps - {}; Learning Rate - {}; Directions - {}; Noise - {}'.format(hp.episode_length, hp.nb_steps, hp.learning_rate, hp.nb_directions, hp.noise), fontsize=14)
    
    plt.show()
    
    fig.savefig('reward_function_{}_eps_length_{}_steps_{}_learning_rate_{}_directions_{}_noise_{}.png'.format(hp.env_name, hp.episode_length, hp.nb_steps, hp.learning_rate, hp.nb_directions, hp.noise))
    
# Run the main code

total_steps = [2000, 1000] # number of steps to follow
episode_lengths = [1000, 2000] # length of the episode - the max time the AI will move
learning_rates = [0.02, 0.05, 0.1] # set the learning rate to control how fast the AI learns
total_directions = [16, 8, 32] # number of perterbations applied to each of the weights (in both directions, positive ,and negative)
noises = [0.01, 0.03, 0.05] # sigma (standard deviation) in the gaussian distribution
env_names = ['HalfCheetahBulletEnv-v0', 
#            'Walker2DBulletEnv-v0', 
#            'HumanoidBulletEnv-v0', 
#            'HopperBulletEnv-v0',
#            'AntBulletEnv-v0',
#            'HumanoidFlagrunBulletEnv-v0'
            ]

def mkdir(base, name):
    '''
    Makes the required directory structure
    
    :param base: string, defined working directory
    :param name: string, folder name to create
    '''
    
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

print("Creating working directory...")
work_dir = mkdir('exp', 'brs')

for nb_steps in total_steps:
    for episode_length in episode_lengths:
        for learning_rate in learning_rates:
            for nb_directions in total_directions:
                for noise in noises:
                    for env_name in env_names:
                        # Add a timer
                        start = time.time()
                        print("Defining the hyperparameters...")
                        hp = Hp(nb_steps, episode_length, learning_rate, nb_directions, noise, env_name)

                        print("Creating monitoring directories...")
                        monitor_dir = mkdir(work_dir, 'monitor_{}_{}_{}_{}_{}_{}'.format(env_name, nb_steps, episode_length, learning_rate, nb_directions, noise))

                        np.random.seed(hp.seed)
                        env = gym.make(hp.env_name)
                        env = wrappers.Monitor(env, monitor_dir, force = True)
                        nb_inputs = env.observation_space.shape[0]
                        nb_outputs = env.action_space.shape[0]

                        print("Defining the policy...")
                        policy = Policy(nb_inputs, nb_outputs)

                        print("Normalizing the results...")
                        normalizer = Normalizer(nb_inputs)

                        print("Training the AI...")
                        train(env, policy, normalizer, hp)

                        # Elapsed time in min
                        elapsed_time = int(time.time() - start)
                        print('\nTotal elapsed time in minutes: {0:.3f}'.format(0.1 * round(elapsed_time) / 6))

                        # Add an end of work message
                        os.system('say "your model has finished training"')
