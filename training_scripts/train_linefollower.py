#!/usr/bin/env python3
import random
import gym_gazebo
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
import os

def main():
    writer = SummaryWriter()
    env = gym.make('LineFollowerEnv-v0')

    observation_space = env.observation_space
    action_space = env.action_space

    epsilon = 0.8
    epsilon_decay = 0.99

    if os.path.exists("q_table.csv"):
        q_table = loadQ("q_table.csv")
        print("Loaded existing Q-table")
    else:
        q_table = np.zeros((observation_space.n, action_space.n))
        print("Starting training from scratch")

    highest_reward = 0

    try:
        state, _ = env.reset()
        for episode in range(1000):
            total_reward = 0
            actions_per_episode = 0
            done = False

            while not done:
                actions_per_episode += 1
                action = choose_action(state, q_table, epsilon, action_space)
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                done = terminated or truncated

                q_table = learn(state, action, next_state, reward, q_table)

                state = next_state
                total_reward += reward

            # Log data
            print(f"Episode {episode}: Epsilon: {epsilon} | Total Reward: {total_reward}")
            print("Q values:")
            print(q_table)
            writer.add_scalar("Total Reward", total_reward, episode)
            writer.add_scalar("Actions per episode", actions_per_episode, episode)

            if total_reward > highest_reward:
                highest_reward = total_reward
                print("New highest reward: {total_reward}.")
                saveQ(q_table, "q_table.csv")

            env.reset()
            epsilon = max(0.01, epsilon * epsilon_decay)  # Decay epsilon

    except KeyboardInterrupt:
        print("Training interrupted by user!")
    
    finally:
        env.close()
        writer.flush()
        writer.close()


def learn(state, action, next_state, reward, q_table):
    alpha = 0.1
    gamma = 0.9

    #find the highest Q value in the next state
    max_next_q = np.max(q_table[next_state])

    #get the current Q-value
    current_q = q_table[state, action]

    #update the Q-value using the Bellman equation
    q_table[state, action] = current_q + alpha * (
        reward + gamma * max_next_q - current_q
    )

    return q_table



def choose_action(state, q_table, epsilon, action_space):
    action = 0

    #choose random number between 0 and 1 and check to see if it is less than epsilon
    #epsilon = 0.1 corresponds to 10% chance of exploring
    if random.random() < epsilon:
        action = action_space.sample()

    # choose the action with the highest Q-value
    else:
        #find highest q value for current state
        q_values = q_table[state]
        max_q = np.max(q_values)

        #handles equal max q values - chooses randomly between them
        best_actions = [i for i, q in enumerate(q_values) if q == max_q]
        action = random.choice(best_actions)
    
    return action



def saveQ(q_table, save_path):

    #save Q-table as a CSV file
    np.savetxt(save_path, q_table, delimiter=",")

    return

def loadQ(load_path):

    q_table = np.loadtxt(load_path, delimiter=",")

    return q_table

if __name__ == "__main__":
    main()