import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

# To plot win rate by current sum
import pandas as pd

# Goal:
# Use Q-Learning to create an agent to play blackjack
# State space: 
# player's current sum, dealer's shown card, player holds "useable ace" 
# Terminal space:
# 1. player sticks
# 2. player hits and the sum of hand exceeds 21
# Action space:
# 0: stick (stay with current cards)
# 1: hit (pick a card)
# Rewards:
# win game: +1.0
#       - if won with natural blackjack: +0.5 additional
# lose game: -1.0
# draw game: 0


def train(episodes: int, render: bool = False):
    # natural = whether there is an additional reward for a natural blackjack
    # sab = false, whether or not we play according to the rules in the textbook
    env = gym.make('Blackjack-v1', natural=False, sab=False, 
                   render_mode="human" if render else None)

    # Initialize Q-learning table
    # 32 x 11 x 2 x 2
    q = np.zeros((env.observation_space[0].n, env.observation_space[1].n, 
                  env.observation_space[2].n, env.action_space.n))
    
    learning_rate_a = 0.3
    discount_factor_g = 0.99

    rewards_per_episode = [0] * episodes

    epsilon = 1
    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng()

    min_epsilon = 0.1

    for i in range(episodes):
        # Restart state and get initial starting conditions
        state = env.reset()[0]

        terminated = False
        truncated = False

        while not terminated and not truncated:
            if rng.random() < epsilon:
                # Take a random action
                action = env.action_space.sample()
            else:
                # Take an action according to the Q-learning table and the state
                action = np.argmax(q[state[0], state[1], state[2], :])

            # Execute action
            new_state, reward, terminated, truncated, info = env.step(action)

            # Update the Q-learning table
            q[state[0], state[1], state[2], action] += learning_rate_a * (
                reward + discount_factor_g 
                * np.max(q[new_state[0], new_state[1], new_state[2], :]) 
                - q[state[0], state[1], state[2], action]
            )

            #print(f"Reward: {reward}")
            #print(f"Choice: {action}")
            #print(f"State: {state}")
            #print(q[state[0], state[1], state[2], :])

            state = new_state
            
            rewards_per_episode[i] = reward

        epsilon = max(min_epsilon, epsilon - epsilon_decay_rate)

    env.close()
    
    # Saving Q Table
    with open("./blackjack/qtable.pkl", 'wb') as f:
        pickle.dump(q, f)
        f.close()

    # Plot results
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        # Plots running sum in 100 episode intervals
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 200):(t+1)])

    plt.plot(sum_rewards)
    plt.savefig("./blackjack/reward.png")


def test(episodes: int, render: bool = False, plot: bool = False, verbose: bool = False):
    env = gym.make('Blackjack-v1', natural=False, sab=False, 
                   render_mode="human" if render else None)

    # Loading pretrained pickle model
    with open('./blackjack/qtable.pkl', 'rb') as f:
        q = pickle.load(f)
        f.close()

    wins = [0] * env.observation_space[0].n
    occurrences = [0] * env.observation_space[0].n

    for i in range(episodes):
        state = env.reset()[0]

        terminated = False
        truncated = False 
        while not terminated and not truncated:
            action = np.argmax(q[state[0], state[1], state[2], :])

            if verbose:
                print(f"State: {state}")
                print(f"Q-values: {q[state]}")
                print(f"Action: {"Stick" if action == 0 else "Hit"}")
            

            new_state, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                occurrences[state[0]] += 1

            if reward == 1:  # Win
                wins[state[0]] += 1
                

            state = new_state

    env.close()

    win_rates = [
        wins[i] / occurrences[i] if occurrences[i] > 0 else 0
        for i in range(env.observation_space[0].n)
    ]

    if plot:
        x_vals = range(env.observation_space[0].n)
        y_vals = win_rates

        plt.figure(figsize=(10, 6))
        plt.bar(x_vals, y_vals, color='blue', alpha=0.7)
        plt.xlabel("Player Sum")
        plt.ylabel("Win Rate")
        plt.title("Win Rate by Player Sum")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(x_vals)
        plt.tight_layout()
        plt.savefig("./blackjack/win_rate_by_sum.png")
        plt.show()
            


if __name__ == '__main__':
    # train(20000, False)
    test(1000, plot=True)