# Tutorial Link
# https://youtu.be/ZhoIgo3qqLU?si=Gb57xf30VkcJxnKm

# RL Environment
import gymnasium as gym

# For Q Learning Table
import numpy as np 

# Plot how the model is doing
import matplotlib.pyplot as plt

# Save the file
import pickle

def run(episodes, is_training=False,render=False,is_slippery=False):
    # Creating an environment for the Frozen Lake problem
    # render_mode = "human" -> allows for us to see the rendered environment 
    # is_slippery = "True" -> the executed isn't always the chosen action, makes the problem not necessarily a "Shortest Path" problem
    env = gym.make('FrozenLake-v1', 
                    map_name="8x8", 
                    is_slippery=is_slippery, 
                    render_mode="human" if render else None)

    # Using a Q-learning Table (64 x 4) and get the argmax at each state to get the value
    if is_training:
        # Intial Q-value for all state-action pairs is 0s
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        # Load the trained Q table
        f = open('./tutorial1/qtable.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    # Model Hyperparamters
    # alpha = multiply this to how much is "learned" at each step of training
    learning_rate_a = 0.9 
    
    # gamma = how much discount the future state values
    discount_factor_g = 0.8

    # Determines the chance that the player follows the greedy algorithm
    # When epsilon = 1, it always picks randomly
    # When epsilon = 0, it always picks according to the greedy algorithm
    # Motivation: To start with a lot of exploration so that the model has a lot of information
    epsilon = 1 
    epsilon_decay_rate = 0.0001

    # Used to determine whether to follow the greedy algorithm, according to the current epsilon value
    rng = np.random.default_rng()

    # Tracking whether or not a reward was collected for the episode (did the player reach the goal?)
    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        # There are 64 states, restarting to initial state
        state = env.reset()[0]

        # When the player falls into a puddle, its state is "terminated" 
        terminated = False 

        # When the player has taken over N steps, its state is "truncated"
        truncated = False

        # While the player is still "alive" and hasn't played for too long
        while(not terminated and not truncated):

            # There are 4 actions
            # 0 = left, 1 = down, 2 = right, 3 = up
            if is_training and rng.random() < epsilon:
                # Take a random action from the action space (EXPLORATION)
                action = env.action_space.sample()
            else:
                # Take an action according to the Q learning table (max expectation)
                action = np.argmax(q[state, :])

            # Execute the new action, and get the variables
            # Reward = 1 when reaching the end state, 0 at any other position
            # _ = has the Probability of whether the move will work? (updated from the video)
            new_state, reward, terminated, truncated, _ = env.step(action)

            # Q-learning equation (Bellman equation)
            # q(s, a) = q(s, a) + alpha * (reward for taking step + gamma * max(q(s', for all a in the Action Space)))
            if is_training:
                q[state, action] = q[state, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                )

            # Assign to new state
            state = new_state

            # If reward was collected in the episode
            if reward == 1:
                rewards_per_episode[i] = 1


        # Decrement epsilon until it reaches 0
        epsilon = max(epsilon - epsilon_decay_rate, 0)

        # Stablize learning rate when epsilon is 0
        if epsilon == 0:
            learning_rate_a = 0.0001

    # Closing gymnasium environment
    env.close()

    # Plotting results
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        # Plots running sum in 100 episode intervals
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100):(t+1)])

    plt.plot(sum_rewards)
    plt.savefig('./tutorial1/reward.png')

    # Saving Q-Table to a file so it can be reused
    if is_training:
        f = open("./tutorial1/qtable.pkl", "wb")
        pickle.dump(q, f)
        f.close()

if __name__ == '__main__':
    run(12000, is_training=True, render=False)