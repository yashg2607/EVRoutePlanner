import gym
from gym import spaces
import numpy as np
import cv2
from tqdm import tqdm
import random

class GridEnv(gym.Env):
    def __init__(self, grid_size, number_of_stations):
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.grid_size, self.grid_size, 3),
                                            dtype=np.uint8)  # RGB image
        # self.chargers = np.array(self._initialize_map_with_charging_stations(number_of_stations))
        # self.origin = np.random.randint(self.grid_size, size=2)
        # self.destination = np.random.randint(self.grid_size, size=2)
        self.origin = np.array((2, 0))
        self.destination = np.array((23, 18))
        self.grid_shape = self.observation_space.shape[0:2]
        self.path = []
        self.visited_states = []
        self.visited_chargers = []
        self.total_time = 0
        self.s = self.linearize_coord(self.origin.copy())


    def linearize_coord(self, coord):
        """Linearize a 2D coordinate into a 1D index"""
        return np.ravel_multi_index(coord, self.grid_shape)

    def delinearize_index(self, index):
        """Delinearize a 1D index into a 2D coordinate"""
        return np.unravel_index(index, self.grid_shape)

    def reset(self):
        # self.origin = np.random.randint(self.grid_size, size=2)
        # self.destination = np.random.randint(self.grid_size, size=2)
        self.origin = np.array((2,0))
        self.destination = np.array((23,18))
        self.chargers = np.array([(6,6),(9,8),(11,9),(16,13),(20,16),(11,5),(13,7),(16,10)])
        self.wait_time = np.array([20, 25, 25, 30, 25, 25, 30, 20])
        self.current_position = self.origin.copy()
        self.path = []
        self.path.append(self.current_position.copy())
        self.visited_states = []
        self.visited_chargers = []
        self.SOC = 100
        self.total_time = 0
        self.s = self.linearize_coord(self.current_position.copy())
        return self.s
    
    def step(self, action):
        if action == 0:  # up
            self.current_position[0] = max(0, self.current_position[0] - 1)
        elif action == 1:  # down
            self.current_position[0] = min(self.grid_size - 1, self.current_position[0] + 1)
        elif action == 2:  # left
            self.current_position[1] = max(0, self.current_position[1] - 1)
        elif action == 3:  # right
            self.current_position[1] = min(self.grid_size - 1, self.current_position[1] + 1)

        self.s = self.linearize_coord(self.current_position.copy())
        # increment the time
        self.total_time += 1

        # append visited states
        self.visited_states.append(self.s)

        # calculate the reward
        reward = self._get_reward()

        # max out the SOC if charger is encounterd and delete that charger from the list of chargers
        if np.any(np.all(self.chargers == self.current_position, axis=1)):
            # print(self.chargers, self.current_position)
            if self.SOC < 20:
                self.SOC = 100
                index = np.where(np.all(self.chargers == self.current_position, axis=1))[0]
                self.total_time = self.total_time - 1 + self.wait_time[int(index)]
                mask = np.any(self.chargers != self.current_position.copy(), axis=1)
                self.chargers = self.chargers[mask]
                self.visited_chargers.append(self.current_position.copy())
            # print("Chargers: ",self.chargers)

        # if reached the destination
        reached = self.current_position[0] == self.destination[0] and self.current_position[1] == self.destination[1]
        self.path.append(self.current_position.copy())

        # decrease the SOC
        self.SOC -= 4
        battery_over = self.SOC < 0
        # if battery_over:
        #     reward = reward - 100

        done = reached or battery_over

        return self.s, reward, self.total_time, self.SOC, done

    def render(self, mode='rgb_array'):
        img = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)

        for position in self.path:
            img[position[0], position[1], :] = [255, 255, 255]  # visited positions are white

        for position in self.chargers:
            img[position[0], position[1], :] = [255,0,0]

        for position in self.visited_chargers:
            img[position[0], position[1], :] = [255,0,255]

        img[self.origin[0], self.origin[1], :] = [0, 0, 255]  # origin is red
        # img[self.current_position[0], self.current_position[1], :] = [0, 0, 255]  # current position is blue
        img[self.destination[0], self.destination[1], :] = [0, 255, 0]  # destination is green

        return img
    
    def _initialize_map_with_charging_stations(self, number_of_stations):
        coordinates = np.zeros((number_of_stations, 2), dtype=np.int32)
        for i in range(number_of_stations):
            x = random.randint(0, self.grid_size-1)
            y = random.randint(0, self.grid_size-1)
            coordinates[i, 0] = x
            coordinates[i, 1] = y
        return coordinates

    def _get_reward(self):
        if self.current_position[0] == self.destination[0] and self.current_position[1] == self.destination[1]:
            return 10000.0
        elif np.any(np.all(self.chargers == self.current_position, axis=1)):
            if self.SOC < 20:
                index = np.where(np.all(self.chargers == self.current_position, axis=1))[0]
                reward = int(500 / (self.wait_time[index] + 1))
                return reward
            else: 
                return -1
        elif self.s in self.visited_states:
            return -200
        else:
            return -1



class SarsaLambda:
    def __init__(self, env, n_states, n_actions, discount, Q, learning_rate, trace_decay, epsilon):
        self.env = env
        self.S = n_states  # state space (assumes 1:nstates)
        self.A = n_actions # action space (assumes 1:nactions)
        self.gamma = discount  # discount
        self.Q = Q  # action value function
        self.N = np.zeros((len(n_states), len(n_actions)))  # trace
        self.alpha = learning_rate  # learning rate
        self.lambda_ = trace_decay  # trace decay rate
        self.epsilon = epsilon 
        self.last_experience = None  # most recent experience tuple (s,a,r)
    
    def choose_greedy_action(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.A)
        Q = lambda a: self.lookahead(s, a)
        return np.argmax([Q(a) for a in self.A])

    def lookahead(self, s, a):
        return self.Q[s][a]

    def update(self, s, a, r, s_prime):
        if self.last_experience is not None:
            gamma, lambda_, Q, alpha, last_experience = self.gamma, self.lambda_, self.Q, self.alpha, self.last_experience
            self.N[last_experience['s'], last_experience['a']] += 1
            delta = last_experience['r'] + gamma * Q[s, a] - Q[last_experience['s'], last_experience['a']]
            self.Q += alpha * delta * self.N
            self.N *= gamma * lambda_
        else:
            self.N[:, :] = 0.0
        self.last_experience = {'s': s, 'a': a, 'r': r}
        return self

    def simulate(self, k, continue_learning = False, filename = None):
        if continue_learning:
            self.load(filename)
        # Iteration over episodes
        for i in range(k):
            episode_reward = 0.0
            s = self.env.reset()
            a = self.choose_greedy_action(s)
            done = False
            while not done:
                s_, r, total_time, soc, done = self.env.step(a)
                a_ = self.choose_greedy_action(s_)
                self.update(s,a,r,s_)
                s = s_
                a = a_
                episode_reward += r
            print(f"Episode {i + 1} completed with reward {episode_reward} with SOC {soc} and total time {total_time}")
            if i % (k/20) == 0:
                img = self.env.render()
                cv2.imwrite(f"image_{i}.tiff", img)
        
        self.save(filename)
    
    def load(self,filename=None):
        if filename:
            print("file found")
            self.Q = np.load(filename+'.npy')

    def save(self, filename):
        with open(filename+'.npy', 'wb') as f:
            np.save(f, self.Q)
            print("file saved")

def main():
    env = GridEnv(25, 6)
    gamma = 0.95
    trace_decay = 0.9
    S = np.arange(env.grid_shape[0]*env.grid_shape[1])
    A = np.arange(env.action_space.n)
    Q = np.zeros((len(S), len(A)))
    alpha = 0.5e-2 # learning rate
    epsilon = 0.1  # probability of random action
    model = SarsaLambda(env, S, A, gamma, Q, alpha, trace_decay, epsilon)
    k = 150000
    continue_learning = True
    filename = "Q_vector"
    model.simulate(k, continue_learning, filename)
    # img = env.render()
    # cv2.imwrite("image.tiff", img)
    # cv2.imshow("image.tiff",img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    

if __name__ == "__main__":
    main()
