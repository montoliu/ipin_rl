import math
import numpy as np
import random


class Envivonment:
    def __init__(self, n_aps, locations, conf):
        self.n_aps = n_aps
        self.min_x = np.min(locations[:, 0])
        self.max_x = np.max(locations[:, 0])
        self.min_y = np.min(locations[:, 1])
        self.max_y = np.max(locations[:, 1])

        self.STEP = conf.get_env_step()
        self.TH_CLOSE = conf.get_env_th_close()
        self.BIG_POSITIVE_REWARD = conf.get_env_big_positive_reward()
        self.BIG_NEGATIVE_REWARD = conf.get_env_big_negative_reward()
        self.SMALL_POSITIVE_REWARD = conf.get_env_small_positive_reward()
        self.SMALL_NEGATIVE_REWARD = conf.get_env_small_negative_reward()
        self.MARGIN = conf.get_env_margin()
        self.GRID_N_CELLS_X = 0
        self.GRID_N_CELLS_Y = 0

        self.GRID_CELL_SIZE = conf.get_grid_cell_size()
        self.grid_x, self.grid_y = self.get_grid()

        self.pos_x = n_aps
        self.pos_y = n_aps + 1
        print("Environment grid size = " + str(self.GRID_N_CELLS_X) + " " + str(self.GRID_N_CELLS_Y))

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # The first is the minimum
    # The last is the maximum
    def get_grid(self):
        n_cells_x = int(np.ceil((self.max_x - self.min_x) / self.GRID_CELL_SIZE)) + 1
        n_cells_y = int(np.ceil((self.max_y - self.min_y) / self.GRID_CELL_SIZE)) + 1

        grid_x = np.zeros(n_cells_x)
        grid_y = np.zeros(n_cells_y)

        x = self.min_x
        for i in range(n_cells_x):
            grid_x[i] = x
            x += self.GRID_CELL_SIZE

        y = self.min_y
        for i in range(n_cells_y):
            grid_y[i] = y
            y += self.GRID_CELL_SIZE

        grid_x[n_cells_x-1] = self.max_x
        grid_y[n_cells_y-1] = self.max_y

        self.GRID_N_CELLS_X = n_cells_x
        self.GRID_N_CELLS_Y = n_cells_y

        return grid_x, grid_y

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Return a random locaition into the environment
    def get_random_location(self):
        x = random.uniform(self.min_x, self.max_x)
        y = random.uniform(self.min_y, self.max_y)
        return np.array([x, y])

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def get_distance(self, loc1, loc2):
       return math.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Each step without reaching the objective -> reward -small number
    # If it reachs the objective -> reward -> big number
    # If it downs out of the environment -> -big number
    def do_step(self, observation, action, real_loc):
        fp = observation[0:len(observation)-2]
        loc = observation[len(observation)-2:len(observation)]
        new_loc = self.move(loc, action)

        if self.is_outside(new_loc):
            done = -1
            reward = self.BIG_NEGATIVE_REWARD
        elif self.is_close(real_loc, new_loc):
            done = 1
            reward = self.BIG_POSITIVE_REWARD
        else:
            d_actual = self.get_distance(loc, real_loc)
            d_new = self.get_distance(new_loc, real_loc)
            if d_new < d_actual:
                reward = self.SMALL_POSITIVE_REWARD
            else:
                reward = self.SMALL_NEGATIVE_REWARD
            done = 0

        new_observation = np.concatenate((fp, new_loc))

        return new_observation, reward, done

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def do_step_test(self, observation, action):
        fp = observation[0:len(observation) - 2]
        loc = observation[len(observation) - 2:len(observation)]
        new_loc = self.move(loc, action)
        new_observation = np.concatenate((fp, new_loc))
        if self.is_outside(new_loc):
            done = -1
        else:
            done = 0
        return new_observation, done

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # 0 up, 1 down, 2 right, 3 left
    def move(self, loc, action):
        new_loc = np.copy(loc)
        if action == 0:
            new_loc[1] += self.STEP  # up
        elif action == 1:
            new_loc[1] -= self.STEP  # down
        elif action == 2:
            new_loc[0] += self.STEP  # right
        elif action == 3:
            new_loc[0] -= self.STEP  # left
        return new_loc

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def is_outside(self, loc):
        if loc[0] < self.min_x - self.MARGIN or \
           loc[0] > self.max_x + self.MARGIN or \
           loc[1] < self.min_y - self.MARGIN or \
           loc[1] > self.max_y + self.MARGIN:
            return True
        return False

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def is_close(self, loc1, loc2):
        d = self.get_distance(loc1, loc2)
        if d < self.TH_CLOSE:
            return True
        return False
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def normalize_loc(self, loc):
        x = (loc[0] - self.min_x) / (self.max_x - self.min_x)
        y = (loc[1] - self.min_y) / (self.max_y - self.min_y)

        return np.array([x,y])

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Discretize continuous localization x,y into position on a grid
    def get_discretize_loc(self, loc):
        if loc[0] < self.min_x:
            x = 0
        elif loc[0] > self.max_x:
            x = self.GRID_N_CELLS_X - 1
        else:
            for i in range(self.GRID_N_CELLS_X - 1):
                if loc[0] <= self.grid_x[i + 1]:
                    x = i
                    break

        if loc[1] < self.min_y:
            y = 0
        elif loc[1] > self.max_y:
            y = self.GRID_N_CELLS_Y - 1
        else:
            for i in range(self.GRID_N_CELLS_Y - 1):
                if loc[1] <= self.grid_y[i + 1]:
                    y = i
                    break

        return x, y

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Real location are more or less in the grid points
    def generate_episodes_grid(self, fps, real_loc):
        n_episodes_per_fp = self.grid_x.shape[0] * self.grid_y.shape[0]

        episodes = np.zeros((fps.shape[0] * n_episodes_per_fp, fps.shape[1] + 2))
        episodes_real_locations = np.zeros((fps.shape[0] * n_episodes_per_fp, 2))

        i = 0
        j = 0
        for fp in fps:
            for x in self.grid_x:
                for y in self.grid_y:
                    new_loc = np.array([x+random.uniform(-0.1, 0.1), y+random.uniform(-0.1, 0.1)])
                    episodes[i,] = np.concatenate((fp, new_loc))
                    episodes_real_locations[i,] = real_loc[j,]
                    i += 1
            j += 1

        return episodes, episodes_real_locations

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def generate_episodes_random(self, fps, real_loc, n_episodes_per_fp):
        episodes = np.zeros((fps.shape[0] * n_episodes_per_fp, fps.shape[1] + 2))
        episodes_real_locations = np.zeros((fps.shape[0] * n_episodes_per_fp, 2))

        i = 0
        j = 0
        for fp in fps:
            for e in range(n_episodes_per_fp):
                random_loc = self.get_random_location()
                episodes[i,] = np.concatenate((fp, random_loc))
                episodes_real_locations[i, ] = real_loc[j, ]
                i += 1
            j += 1

        return episodes, episodes_real_locations

