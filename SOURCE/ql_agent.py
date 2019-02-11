import numpy as np
import random


class ql_agent:
    def __init__(self, env, conf):
        self.ALPHA = conf.AGENT_ALPHA
        self.MAX_STEPS_BY_EPISODE = conf.AGENT_MAX_STEPS_BY_EPISODE
        self.EPS = 0.5
        self.GAMMA = conf.AGENT_GAMMA
        self.N_ACTIONS = conf.AGENT_N_ACTIONS
        self.RSSI_TH = conf.ENV_RSSI_TH

        self.env = env
        self.steps_done = 0
        self.AGENT_N_RANDOM_TEST = conf.AGENT_N_RANDOM_TEST

        self.Q = dict()

     # ---------------------------------------------------------
    # ---------------------------------------------------------
    def save_to_disk(self, filename):
        np.save(filename, self.Q)

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def load_from_disk(self, filename):
        self.Q = np.load(filename + ".npy").item()

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def ith_train(self, i_epoch, episodes, real_locations):
        if i_epoch == 0:
            self.EPS = 0.8
        elif i_epoch == 1:
            self.EPS = 0.6
        elif i_epoch == 2:
            self.EPS = 0.4
        elif i_epoch == 3:
            self.EPS = 0.2
        elif i_epoch == 4:
            self.EPS = 0.1
        self.train(episodes, real_locations)

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def train(self, episodes, real_locations):
        i = 0
        mean_acm_reward = 0
        mean_acm_success = 0
        mean_acm_reward_epoch = 0
        mean_acm_success_epoch = 0

        for ep in episodes:
            n_steps, acm_reward, last_obs, done = self.run_episode(ep, real_locations[i, ])

            if done == 1:
                mean_acm_success += 1
                mean_acm_success_epoch += 1

            i += 1
            mean_acm_reward += acm_reward
            mean_acm_reward_epoch += acm_reward

            if i % 10000 == 0:
                print(str(i) + " --> " + str(mean_acm_reward/10000) + " " + str(mean_acm_success) + " " + str(len(self.Q)))
                mean_acm_reward = 0
                mean_acm_success = 0

        print("{:0.2f}".format(float(mean_acm_reward_epoch) / episodes.shape[0]) + " " + str(mean_acm_success_epoch) + " " + str(len(self.Q)))

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def run_episode(self, episode, real_location):
        observation = episode
        n_step = 1
        acm_reward = 0
        done = 0
        while n_step < self.MAX_STEPS_BY_EPISODE and done == 0:
            action = self.select_action(observation)
            next_observation, reward, done = self.env.do_step(observation, action, real_location)
            acm_reward += reward
            state = self.from_observation_to_state(observation)

            if done == 0:
                next_state = self.from_observation_to_state(next_observation)
                self.actualize_q_values(state, next_state, action, reward)

                #self.print_step(n_step, observation, next_observation, action, state, next_state)
                observation = next_observation
                n_step += 1
            else:
                self.actualize_q_values_done(state, action, reward)
                #self.print_step_done(n_step, observation, action, state)


        return n_step, acm_reward, next_observation, done

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Select an action using E-greedy policy
    def select_action(self, observation):
        sample = random.random()
        eps_threshold = self.EPS
        self.steps_done += 1
        if sample > eps_threshold:
            state = self.from_observation_to_state(observation)
            return self.get_best_action(state)
        else:
            return random.randrange(self.N_ACTIONS)

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # generate AGENT_N_RANDOM_TEST samples. The final is the centroid of all successful estimations
    # The agent reach the objective when the previous where the is a bucle, i.e. now 0(1) and previous 1(0), or now 2(3) and previous 3(2)
    def test(self, fp):
        acm_estimated_locX = 0
        acm_estimated_locY = 0

        previous_action = -1
        for i in range(self.AGENT_N_RANDOM_TEST):
            random_loc = self.env.get_random_location()
            observation = np.concatenate((fp, random_loc))
            n_steps = 0
            done = 0
            while n_steps < self.MAX_STEPS_BY_EPISODE and done == 0:
                state = self.from_observation_to_state(observation)
                best_action = self.get_best_action(state)
                new_observation, done = self.env.do_step_test(observation, best_action)

                if done != -1:
                    is_at_objective = self.check_objective(best_action, previous_action)
                    if is_at_objective:
                        observation[self.env.pos_x] = (observation[self.env.pos_x] + new_observation[self.env.pos_x]) / 2
                        observation[self.env.pos_y] = (observation[self.env.pos_y] + new_observation[self.env.pos_y]) / 2
                        done = 1
                    else:
                        observation = new_observation
                        n_steps += 1
                    previous_action = best_action

            acm_estimated_locX += observation[self.env.pos_x]
            acm_estimated_locY += observation[self.env.pos_y]

        new_loc = np.zeros(2)
        new_loc[0] = acm_estimated_locX / self.AGENT_N_RANDOM_TEST
        new_loc[1] = acm_estimated_locY / self.AGENT_N_RANDOM_TEST
        return new_loc

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def check_objective(self, best_action, previous_action):
        if best_action == 0 and previous_action == 1 or \
           best_action == 1 and previous_action == 0 or \
           best_action == 2 and previous_action == 3 or \
           best_action == 3 and previous_action == 2:
            return True
        return False

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # if a state+action is not in the Q table, get value 0
    def get_best_action(self, state):
        best_value = -1000000
        best_action = 0
        for i_action in range(self.N_ACTIONS):
            sta = state + str(i_action)
            if sta in self.Q:
                v = self.Q[sta]
            else:
                v = 0
            if v > best_value:
                best_value = v
                best_action = i_action
        return best_action

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Return maximum value for a state for all actions
    def get_max_q_value(self, state):
        best = -1000000
        for i_action in range(self.N_ACTIONS):
            sta = state + str(i_action)
            if sta in self.Q:
                v = self.Q[sta]
            else:
                v = 0
            if v > best:
                best = v
        return best

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Return the Q values of an observation and action
    def get_q_value(self, state, action):
        sta = state + str(action)
        if sta not in self.Q:
            return 0.0
        return self.Q[sta]

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Store the Q value of the observation and action
    def set_q_value(self, state, action, q_value):
        sta = state + str(action)
        self.Q[sta] = q_value

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Q-learning equation
    # Q(st,a) = Q(st,a) + alpha*[R + gamma*(maxQ(st+1,a) - Q(st,a)]
    def actualize_q_values(self, state, next_state, action, reward):
        max_st1 = self.get_max_q_value(next_state)
        q_value_st = self.get_q_value(state, action)
        q_value_st = q_value_st + self.ALPHA * (reward + self.GAMMA*max_st1 - q_value_st)
        self.set_q_value(state, action, q_value_st)

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Q-learning equation if done
    # Q(st,a) = Q(st,a) + alpha*[R - Q(st,a)]
    def actualize_q_values_done(self, state, action, reward):
        q_value_st = self.get_q_value(state, action)
        q_value_st = q_value_st + self.ALPHA * (reward - q_value_st)
        self.set_q_value(state, action, q_value_st)

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # TODO debe depender del numero de celdas de cell_x y cell_y
    def from_observation_to_state(self, observation):
        n = observation.shape[0]
        loc = np.array([observation[n-2], observation[n-1]])
        cell_x, cell_y = self.env.get_discretize_loc(loc)
        s = ""
        for i in range(n-2):
            if observation[i] >= self.RSSI_TH:
                s = s + "1"
            else:
                s = s + "0"

        s = s + "{0:0>2}".format(cell_x)   # 2 digits
        s = s + "{0:0>2}".format(cell_y)   # 2 digits
        return s

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def print_step(self, iter, obs, next_obs, action, state, next_state):
        print("---------------")
        print("Iter: " + str(iter) + " " +
              "[" + "{:0.2f}".format(obs[4]) + ", " + "{:0.2f}".format(obs[5]) + "] " +
              "[" + "{:0.2f}".format(next_obs[4]) + ", " + "{:0.2f}".format(next_obs[5]) + "] " +
              "A:" + str(action) + " " + state[4:6] + " " + next_state[4:6])
        print("---------------")
        self.print_Q()

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def print_step_done(self, iter, obs, action, state):
        print("---------------")
        print("Iter: " + str(iter) + " " +
              "[" + "{:0.2f}".format(obs[4]) + ", " + "{:0.2f}".format(obs[5]) + "] " +
              "[--, --] " +
              "A:" + str(action) + " " + state[4:6] + " [--, --]")
        print("---------------")
        self.print_Q()

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def print_Q(self):
        print(len(self.Q))
        for key in self.Q:
            print(key[4:6] + " " + key[6] + " -> " + str(self.Q[key]))