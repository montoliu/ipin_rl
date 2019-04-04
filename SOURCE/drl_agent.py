from keras.models import load_model
import SOURCE.drl_network as drlnn
import SOURCE.drl_memory as drlmemory
import numpy as np
import random
import math


class drl_agent:
    def __init__(self, env, conf):
        self.MEM_MAX_CAPACITY = conf.get_drl_mem_max_capacity()
        self.LEARNING_RATE = conf.get_drl_learning_rate()
        self.NN_INPUT = env.get_n_aps() + 2
        self.NN_HIDDEN = conf.get_drl_hidden()
        self.NN_OUTPUT = env.get_n_actions()
        self.BATCH_SIZE = conf.get_drl_batch_size()
        self.MAX_STEPS_BY_EPISODE = conf.get_drl_max_steps_by_episode()
        self.N_EPOCHS = conf.get_drl_n_epochs()
        self.N_RANDOM_TEST = conf.get_drl_n_random_test()

        self.EPS_START = 0.1
        self.EPS_END = 0.9
        self.GAMMA = conf.get_drl_gamma()

        self.memory = drlmemory.drl_memory(self.MEM_MAX_CAPACITY)
        self.env = env
        self.model = self.__build_model()

        self.steps_done = 0

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def __build_model(self):
        network = drlnn.drl_network(n_input=self.NN_INPUT,
                                    n_hidden=self.NN_HIDDEN,
                                    n_output=self.NN_OUTPUT,
                                    learning_rate=self.LEARNING_RATE)
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
            i += 1
            mean_acm_reward += acm_reward

            if i % 10000 == 0:
                print(str(i) + " --> " + str(mean_acm_reward / 10000))
                mean_acm_reward = 0

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

            self.add_to_memory(observation, action, next_observation, reward, done)

            self.learn()
            observation = next_observation
            n_step += 1

        return n_step, acm_reward, next_observation, 0


    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # TODO como cambiar eps_threshold
    def select_action(self, observation):
        observation_norm = self.normalize_observation(observation)
        sample = random.random()
        eps_threshold = 0.5
        if sample > eps_threshold:
            state = np.reshape(observation_norm, [1, self.NN_INPUT])
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])
        else:
            return random.randrange(self.NN_OUTPUT)


    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def add_to_memory(self, observation, action, next_observation, reward, done):
        observation_norm = self.normalize_observation(observation)
        next_observation_norm = self.normalize_observation(next_observation)

        self.memory.push((observation_norm, action, next_observation_norm, reward, done))

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

        history = self.model.fit(batch_state, targets_f, epochs=self.N_EPOCHS, verbose=0)

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Normalizamos x, y entre 0 y 1
    def normalize_observation(self, observation):
        fp = observation[0:len(observation) - 2]
        loc = observation[len(observation) - 2:len(observation)]
        loc_norm = self.env.normalize_loc(loc)
        observation_norm = np.concatenate((fp, loc_norm))
        return observation_norm

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # generate AGENT_N_RANDOM_TEST samples. The final is the centroid of all successful estimations
    # The agent reach the objective when the previous where the is a bucle, i.e. now 0(1) and previous 1(0), or now 2(3) and previous 3(2)
    def test(self, fp):
        acm_estimated_locX_all = 0
        acm_estimated_locY_all = 0
        acm_estimated_locX_goods = 0
        acm_estimated_locY_goods = 0
        n_goods = 0

        # we perform the search starting from several trandom positions
        for i in range(self.N_RANDOM_TEST):
            random_loc = self.env.get_random_location()
            observation = np.concatenate((fp, random_loc))
            n_steps = 0
            done = 0
            previous_action = -1
            while n_steps < self.MAX_STEPS_BY_EPISODE and done == 0:
                best_action = self.get_best_action(observation)
                new_observation, done = self.env.do_step_test(observation, best_action)

                # done == -1 is when the observation is out of the limits
                # In this case the valid observation is the last one before going out
                if done != -1:
                    is_at_objective = self.check_objective(best_action, previous_action)
                    if is_at_objective == 1:
                        observation[self.env.pos_x] = new_observation[self.env.pos_x]
                        observation[self.env.pos_y] = (observation[self.env.pos_y] + new_observation[self.env.pos_y]) / 2
                        done = 1
                    elif is_at_objective == 2:
                        observation[self.env.pos_x] = (observation[self.env.pos_x] + new_observation[self.env.pos_x]) / 2
                        observation[self.env.pos_y] = new_observation[self.env.pos_y]
                        done = 1
                    else:
                        # prepare new iteration
                        observation = new_observation
                        previous_action = best_action
                        n_steps += 1
            #end while

            # if the agent gone out, the observation is not taken into account as goods
            if done != -1:
                acm_estimated_locX_goods += observation[self.env.pos_x]
                acm_estimated_locY_goods += observation[self.env.pos_y]
                n_goods += 1

            acm_estimated_locX_all += observation[self.env.pos_x]
            acm_estimated_locY_all += observation[self.env.pos_y]
        # end for

        # If there are at least one good, we use the goods.
        # If all are not goods, we use the all
        new_loc = np.zeros(2)
        if n_goods > 0:
            new_loc[0] = acm_estimated_locX_goods / n_goods
            new_loc[1] = acm_estimated_locY_goods / n_goods
            success = 1
        else:
            new_loc[0] = acm_estimated_locX_all / self.N_RANDOM_TEST
            new_loc[1] = acm_estimated_locY_all / self.N_RANDOM_TEST
            success = 0
        return new_loc, success


    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def get_best_action(self, observation):
        observation_norm = self.normalize_observation(observation)
        drl_observation = np.reshape(observation_norm, [1, self.NN_INPUT])
        act_values = self.model.predict(drl_observation)
        best_action = np.argmax(act_values[0])
        return best_action

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def check_objective(self, best_action, previous_action):
        if best_action == 0 and previous_action == 1 or \
           best_action == 1 and previous_action == 0:
            # up and down
            return 1
        elif best_action == 2 and previous_action == 3 or \
             best_action == 3 and previous_action == 2:
            #left and right
            return 2
        return 0