import gym
from gym import spaces
import numpy as np
import cv2
from tqdm import tqdm

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
        self.origin = np.array((0,0))
        self.destination = np.array((5,7))
        self.chargers = np.array([(2,2),(4,6),(6,9)])
        self.current_position = self.origin.copy()
        self.path = []
        self.path.append(self.current_position.copy())
        self.visited_states = []
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

        done = reached or battery_over

        return self.s, reward, self.total_time, done

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
    

    def _get_reward(self):
        if self.current_position[0] == self.destination[0] and self.current_position[1] == self.destination[1]:
            return 1000.0
        elif self.linearize_coord(self.current_position) in self.visited_states:
            return -10
        elif np.any(np.all(self.chargers == self.current_position, axis=1)):
            return 20
        else:
            return -1


class EpsilonGreedyExploration:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self, A, s):
        epsilon = self.epsilon
        if np.random.rand() < epsilon:
            return np.random.choice(A)
        Q = lambda a: lookahead(s, a)
        return np.argmax([Q(a) for a in A])



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

    def simulate(self, k):
        # Iteration over episodes
        for i in tqdm(range(k)):
            episode_reward = 0.0
            s = self.env.reset()
            a = self.choose_greedy_action(s)
            done = False
            while not done:
                s_, r, total_time, done = self.env.step(a)
                a_ = self.choose_greedy_action(s_)
                self.update(s,a,r,s_)
                s = s_
                a = a_
                episode_reward += r
            print(f"Episode {k + 1} completed with reward {episode_reward} and total time {total_time}")


def main():
    env = GridEnv()
    gamma = 0.95
    trace_decay = 0.9
    S = np.arange(env.grid_shape[0]*env.grid_shape[1])
    A = np.arange(env.action_space.n)
    Q = np.zeros((len(S), len(A)))
    alpha = 0.1e-2  # learning rate
    epsilon = 0.1  # probability of random action
    model = SarsaLambda(env, S, A, gamma, Q, alpha, trace_decay, epsilon)
    k = 200
    num_games = 1
    # for games in range(num_games):  
    #     s =  
    model.simulate(k)
    img = env.render()
    cv2.imwrite("image.tiff", img)
    # cv2.imshow("image.tiff",img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    

if __name__ == "__main__":
    main()
