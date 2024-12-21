import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

# States (continuous):
# X-axis position of the cart
# X-velocity of the cart
# Angle of the pole
# Angular velocity of the falling pole

# Actions
# Left (0), Right (1)

# Terminal States:
# Pole falls over (determined by Angle of the Pole)
# Cart is out of bounds (determined by the X-position of the Pole)

def run(is_training=True, render=False):

    env = gym.make('CartPole-v1', render_mode='human' if render else None)

    # Discretizing the range of the states
    # Divide position, velocity, pole angle, and pole angular velocity into 10 segments
    # The number ranges of the states are from the Gymnasium specs (position and angle) and through testing (velocity and angular velocity)
    pos_space = np.linspace(-2.4, 2.4, 10) 
    vel_space = np.linspace(-4, 4, 10)
    ang_space = np.linspace(-.2095, .2095, 10)
    ang_vel_space = np.linspace(-4, 4, 10)

    # Why 11? states
    # 0 < 1 < 2 < 3 < 4
    # By using the numpy digitize function, 
    # we are essentially partitioning the set of states with 10 partitions. 
    # So, there would be 11 values.

    if(is_training):
        q = np.zeros((len(pos_space)+1, len(vel_space)+1, len(ang_space)+1, len(ang_vel_space)+1, env.action_space.n)) # init a 11x11x11x11x2 array
    else:
        f = open('./tutorial4/cartpole.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.1 # alpha or learning rate
    discount_factor_g = 0.99 # gamma or discount factor.

    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 0.00001 # epsilon decay rate
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = []

    i = 0

    # Instead of training for a fix N episodes, we want to train until the cart can balance the pole
    while(True):

        state = env.reset()[0]      # Starting position, starting velocity always 0
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vel_space)

        terminated = False          # True when reached goal

        rewards=0

        # while the rewards isn't too high (reward determined by how long the pole is up) and
        #               we haven't reached the terminal states
        while(not terminated and rewards < 10000):

            if is_training and rng.random() < epsilon:
                # Choose random action  (0=go left, 1=go right)
                action = env.action_space.sample()
            else:
                # Choosing the best action
                action = np.argmax(q[state_p, state_v, state_a, state_av, :])

            new_state,reward,terminated,_,_ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            new_state_a = np.digitize(new_state[2], ang_space)
            new_state_av= np.digitize(new_state[3], ang_vel_space)

            if is_training:
                q[state_p, state_v, state_a, state_av, action] = q[state_p, state_v, state_a, state_av, action] + learning_rate_a * (
                    reward + discount_factor_g*np.max(q[new_state_p, new_state_v, new_state_a, new_state_av,:]) - q[state_p, state_v, state_a, state_av, action]
                )

            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            state_a = new_state_a
            state_av= new_state_av

            rewards+=reward

            if not is_training and rewards%100==0:
                print(f'Episode: {i}  Rewards: {rewards}')

        rewards_per_episode.append(rewards)

        # average reward in the last 100 episodes (running sum)
        mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode)-100:])

        # Print stats for every 100 episodes
        if is_training and i%100==0:
            print(f'Episode: {i} {rewards}  Epsilon: {epsilon:0.2f}  Mean Rewards {mean_rewards:0.1f}')

        # Stop training if we are balancing well enough
        if mean_rewards>1000:
            break

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        i+=1

    env.close()

    # Save Q table to file
    if is_training:
        f = open('./tutorial4/cartpole.pkl','wb')
        pickle.dump(q, f)
        f.close()

    mean_rewards = []
    for t in range(i):
        mean_rewards.append(np.mean(rewards_per_episode[max(0, t-100):(t+1)]))
    plt.plot(mean_rewards)
    plt.savefig(f'./tutorial4/cartpole.png')

if __name__ == '__main__':
    # run(is_training=True, render=False)

    run(is_training=False, render=True)