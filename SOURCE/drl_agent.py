from keras.models import load_model
import SOURCE.drl_network as drlnn
import SOURCE.drl_memory as drlmemory
import numpy as np
import random
import math


class drl_agent:
    def __init__(self, env, conf):
        self.MEM_MAX_CAPACITY = conf.AGENT_MEM_MAX_CAPACITY
        self.LEARNING_RATE = conf.AGENT_LEARNING_RATE
        self.NN_INPUT = conf.AGENT_NN_INPUT
        self.NN_HIDDEN = conf.AGENT_NN_HIDDEN
        self.NN_OUTPUT = conf.AGENT_NN_OUTPUT
        self.BATCH_SIZE = conf.AGENT_BATCH_SIZE
        self.MAX_STEPS_BY_EPISODE = conf.AGENT_MAX_STEPS_BY_EPISODE
        self.EPS_START = conf.AGENT_EPS_START
        self.EPS_END = conf.AGENT_EPS_END
        self.EPS_DECAY = conf.AGENT_EPS_DECAY
        self.GAMMA = conf.AGENT_GAMMA

        self.memory = drlmemory.drl_memory(self.MEM_MAX_CAPACITY)
        self.env = env
        self.model = self.__build_model()
        self.steps_done = 0
        self.n_learn_done = 0

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def __build_model(self):
        network = drlnn.drl_network(n_input=self.NN_INPUT, n_hidden=self.NN_HIDDEN, n_output=self.NN_OUTPUT, learning_rate=self.LEARNING_RATE)
        return network.model

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def save_to_disk(self, filename):
        self.model.save(filename)

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def load_from_disk(self, filename):
        self.model = load_model(filename)

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def train(self, episodes, real_locations):
        i = 0
        mean_acm_reward = 0

        for ep in episodes:
            n_steps, acm_reward, last_obs, done = self.run_episode(ep, real_locations[i, ])

            if done == 1:
                str_done = "Success"
            elif done == -1:
                str_done = "Fail"
            else:
                str_done = "Unfinished"

            i += 1
            mean_acm_reward += acm_reward
            if i % 500 == 0:
                print(str(i) + " --> " + str(mean_acm_reward))
                mean_acm_reward = 0


    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def select_action(self, state):
        sample = random.random()
        eps_threshold = 0.5
        if sample > eps_threshold:
            state = np.reshape(state, [1, self.NN_INPUT])
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])
        else:
            return random.randrange(self.NN_OUTPUT)

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def run_episode(self, episode, real_location):
        observation = episode
        n_step = 1
        acm_reward = 0
        while n_step < self.MAX_STEPS_BY_EPISODE:
            observation_norm = self.normalize_observation(observation)
            action = self.select_action(observation_norm)
            next_observation, reward, done = self.env.do_step(observation, action, real_location)
            acm_reward += reward
            if done != 0:
                return n_step, acm_reward, next_observation, done

            next_observation_norm = self.normalize_observation(observation)
            self.memory.push((observation_norm, action, next_observation_norm, reward, done))
            self.learn()
            observation = next_observation
            n_step += 1
        return n_step, acm_reward, next_observation, 0

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def learn(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        # random transition batch is taken from experience replay memory
        transitions = self.memory.sample(self.BATCH_SIZE)

        batch_state, batch_action, batch_next_state, batch_reward, batch_done = (np.array(i) for i in zip(*transitions))

        max_next_q_values = np.amax(self.model.predict(batch_next_state), axis=1)
        batch_targets = batch_reward + self.GAMMA * max_next_q_values
        batch_targets[batch_done, ] = np.array(batch_reward[batch_done])
        targets_f = self.model.predict(batch_state)

        for i in range(len(batch_action)):
            targets_f[i, batch_action[i]] = batch_targets[i]

        history = self.model.fit(batch_state, targets_f, epochs=3, verbose=0)
        self.n_learn_done += 1

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def test(self, fp):
        random_loc = self.env.get_random_location()
        observation = np.concatenate((fp, random_loc))
        n_steps = 0
        while n_steps < self.MAX_STEPS_BY_EPISODE:
            observation_norm = self.normalize_observation(observation)
            drl_observation = np.reshape(observation_norm, [1, self.NN_INPUT])
            act_values = self.model.predict(drl_observation)
            best_action = np.argmax(act_values[0])
            new_observation, done = self.env.do_step_test(observation, best_action)

            if done:
                return new_observation, n_steps

            observation = new_observation
            n_steps += 1

        return new_observation, n_steps

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Normalizamos x, y entre 0 y 1
    def normalize_observation(self, observation):
        fp = observation[0:len(observation) - 2]
        loc = observation[len(observation) - 2:len(observation)]
        loc_norm = self.env.normalize_loc(loc)
        observation_norm = np.concatenate((fp, loc_norm))
        return observation_norm
