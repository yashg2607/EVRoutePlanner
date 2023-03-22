import gym
from gym import spaces
import numpy as np
import cv2

class GridEnv(gym.Env):
    def __init__(self):
        self.grid_size = 10
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.grid_size, self.grid_size, 3),
                                            dtype=np.uint8)  # RGB image
        # self.origin = np.random.randint(self.grid_size, size=2)
        # self.destination = np.random.randint(self.grid_size, size=2)
        self.origin = np.array((0, 0))
        self.destination = np.array((5, 7))
        self.grid_shape = self.observation_space.shape[0:2]
        self.path = []
        self.visited_states = []
        self.total_time = 0

    def linearize_coord(self, coord):
        """Linearize a 2D coordinate into a 1D index"""
        return np.ravel_multi_index(coord, self.grid_shape)

    def delinearize_index(self, index):
        """Delinearize a 1D index into a 2D coordinate"""
        return np.unravel_index(index, self.grid_shape)

    def reset(self):
        # self.origin = np.random.randint(self.grid_size, size=2)
        # self.destination = np.random.randint(self.grid_size, size=2)
        self.origin = np.array((0,0))
        self.destination = np.array((5,7))
        self.chargers = np.array([(2,2),(4,6),(6,9)])
        self.current_position = self.origin.copy()
        self.path = []
        self.path.append(self.current_position.copy())
        self.visited_states = []
        self.SOC = 100
        self.total_time = 0
        return self.current_position

    def step(self, action):
        if action == 0:  # up
            self.current_position[0] = max(0, self.current_position[0] - 1)
        elif action == 1:  # down
            self.current_position[0] = min(self.grid_size - 1, self.current_position[0] + 1)
        elif action == 2:  # left
            self.current_position[1] = max(0, self.current_position[1] - 1)
        elif action == 3:  # right
            self.current_position[1] = min(self.grid_size - 1, self.current_position[1] + 1)

        # increment the time
        self.total_time += 1

        # append visited states
        self.visited_states.append(self.linearize_coord(self.current_position.copy()))

        # max out the SOC if charger is encounterd and delete that charger from the list of chargers
        if np.any(np.all(self.chargers == self.current_position, axis=1)):
            self.SOC = 100
            mask = np.any(self.chargers != self.current_position.copy(), axis=1)
            self.chargers = self.chargers[mask]
            # print("Chargers: ",self.chargers)

        # calculate the reward
        reward = self._get_reward()

        # if reached the destination
        reached = self.current_position[0] == self.destination[0] and self.current_position[1] == self.destination[1]
        self.path.append(self.current_position.copy())

        # decrease the SOC
        self.SOC -= 1
        battery_over = self.SOC < 0
        if battery_over:
            reward = reward - 100
        # print("reward: ", reward, " SOC: ", self.SOC, " curr state: ", self.current_position, " battery_over:", battery_over)
        done = reached or battery_over

        return self.current_position, reward, self.total_time, done

    def render(self, mode='rgb_array'):
        img = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)

        for position in self.path:
            img[position[0], position[1], :] = [255, 255, 255]  # visited positions are white

        for position in self.chargers:
            img[position[0], position[1], :] = [255,0,0]
        img[self.origin[0], self.origin[1], :] = [0, 0, 255]  # origin is red
        # img[self.current_position[0], self.current_position[1], :] = [0, 0, 255]  # current position is blue
        img[self.destination[0], self.destination[1], :] = [0, 255, 0]  # destination is green

        return img

    def _get_observation(self):
        return self.render()

    def _get_reward(self):
        if self.current_position[0] == self.destination[0] and self.current_position[1] == self.destination[1]:
            return 1000.0
        elif self.linearize_coord(self.current_position) in self.visited_states:
            return -10
        elif np.any(np.all(self.chargers == self.current_position, axis=1)):
            return 20
        else:
            return -1


## Q-learning

class QLearningAgent:
    def __init__(self, env, learning_rate=0.8, discount_factor=0.99, epsilon=0.3, decay_rate=0.95):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.Q = np.zeros((10, 10, 4))
        self.decay_rate = decay_rate

    def act(self, state):
        if np.random.random() < self.epsilon:
            # random action with probability epsilon
            return self.env.action_space.sample()
        else:
            # greedy action with probability 1 - epsilon
            return np.argmax(self.Q[state[0], state[1], :])

    def learn(self, state, action, reward, next_state):
        td_error = reward + self.discount_factor * np.max(self.Q[next_state[0], next_state[1], :]) - self.Q[
            state[0], state[1], action]
        self.Q[state[0], state[1], action] += self.learning_rate * td_error
        self.learning_rate *= self.decay_rate

    def load(self,filename=None):
        if filename:
            print("file found")
            self.Q = np.load(filename+'.npy')

    def save(self, filename):
        with open(filename+'.npy', 'wb') as f:
            np.save(f, self.Q)
            print("file saved")

def run_q_learning(env, num_episodes=10, continue_learning = False, filename = None):
    agent = QLearningAgent(env)
    if continue_learning:
        agent.load(filename)
    policy = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action = agent.act(state)
            next_state, reward, total_time, done = env.step(action)
            episode_reward += reward
            agent.learn(state, action, reward, next_state)
            state = next_state

        #     # Render the environment
        #     img = env.render()
        #     cv2.namedWindow('Image.tiff', cv2.WINDOW_NORMAL)
        #     cv2.resizeWindow('Image.tiff', 800, 600)
        #     # Show image
        #     cv2.imshow('Image.tiff', img)
        #
        #     # Wait for 1 second
        #     cv2.waitKey(1)
        #
        # # Close all windows
        # cv2.destroyAllWindows()

        # Store the episode reward and policy
        policy.append(np.argmax(agent.Q, axis=2))
        print(f"Episode {episode + 1} completed with reward {episode_reward} and total time {total_time}")

    # save the Q vector
    agent.save(filename)
    return policy


## SARSA Lambda

class SarsaLambda:
    def __init__(self, env, gamma=0.99, alpha=0.5, lambd=0.8, epsilon=0.1, decay_rate=0.99):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.lambd = lambd
        self.epsilon = epsilon
        self.Q = np.zeros((self.env.observation_space.shape[0], self.env.observation_space.shape[1], self.env.action_space.n))
        self.E = np.zeros_like(self.Q)
        self.decay_rate = decay_rate

    def choose_action(self, obs):
        if np.random.uniform(0, 1) < self.epsilon:
            # choose random action with probability epsilon
            return self.env.action_space.sample()
        else:
            # choose greedy action with probability 1 - epsilon
            return np.argmax(self.Q[obs[0], obs[1], :])

    def train(self, num_episodes, continue_learning = False, filename=None):
        if continue_learning:
            self.load(filename)

        self.alpha *= self.decay_rate
        for episode in range(num_episodes):
            episode_reward = 0.0
            obs = self.env.reset()
            action = self.choose_action(obs)
            self.E.fill(0)
            done = False
            while not done:
                next_obs, reward, total_time, done = self.env.step(action)
                next_action = self.choose_action(next_obs)
                td_error = reward + self.gamma * self.Q[next_obs[0], next_obs[1], next_action] - self.Q[obs[0], obs[1], action]
                self.E[obs[0], obs[1], action] += 1
                for i in range(self.Q.shape[0]):
                    for j in range(self.Q.shape[1]):
                        for k in range(self.Q.shape[2]):
                            self.Q[i, j, k] += self.alpha * td_error * self.E[i, j, k]
                            self.E[i, j, k] *= self.gamma * self.lambd

                obs = next_obs
                action = next_action
                episode_reward += reward

            print(f"Episode {episode + 1} completed with reward {episode_reward} and total time {total_time}")

        # save the Q vector
        self.save(filename)

    def load(self,filename=None):
        if filename:
            print("file found")
            self.Q = np.load(filename+'.npy')

    def save(self, filename):
        with open(filename+'.npy', 'wb') as f:
            np.save(f, self.Q)
            print("file saved")

def run_sarsa_lambda(env, num_episodes = 2000):
    agent = SarsaLambda(env, gamma=0.99, alpha=0.4, lambd=0.8, epsilon=0.3, decay_rate=0.95)
    policy = agent.train(num_episodes, continue_learning=True, filename='sarsa2/Q_vector_sarsa')

# TODO: linearize the states and try again with penalising the agent if it goes to the previous state.
# TODO: add the wait times to the charging stations

env = GridEnv()
# policy = run_q_learning(env, num_episodes=20, continue_learning=False, filename='Q_vector')
total_iters = 200000

# for iters in range(total_iters):
#     if iters % 20000 == 0:
#         print(f"currently on {iters} iteration")
#         policy = run_q_learning(env, num_episodes=20000, continue_learning=True, filename='Q_vector')
#         img = env.render()
#         cv2.imwrite(f'image_at_iter_{iters}.tiff', img)

for iters in range(total_iters):
    if iters % 20000 == 0:
        print(f"currently on {iters} iteration")
        run_sarsa_lambda(env, num_episodes=2000)
        img = env.render()
        cv2.imwrite(f'sarsa2/sarsa_image_at_iter_{iters}.tiff', img)


# sarsa = SarsaLambda(env, gamma=0.99, alpha=0.5, lambd=0.8, epsilon=0.1)
# sarsa.train(num_episodes=20000)





# cv2.imshow('image.tiff',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# env = GridEnv()
# obs = env.reset()
# done = False
# total_reward = 0
#
# # while not done:
# #     action = env.action_space.sample() # random action
# #     obs, reward, done, _ = env.step(action)
# #     total_reward += reward
#
# action = env.action_space.sample()
# obs, reward, done, _ = env.step(action)
# total_reward += reward
#
# img = env.render()
# # cv2.imwrite('env.tiff', img)
# # render the image in a window
# cv2.imshow('image.tiff',img)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()

# print("Total reward:", total_reward)