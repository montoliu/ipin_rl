# -----------------------------------------------------------------
# DO_SETS: crear conjuntos nuevos de entrenamiento y test
# DO_KNN: ejecuta el algoritmo knn
# DO_DRL: ejecuta el algoritmo DRL (Deep reinforcement learning)
# DO_DRL_TRAINING: entrena el algortimo DRL
# DO_QL: ejecuta el algoritmo QL (Q-learning)
# DO_QL_TRAINING: entrena el algortimo QL
# DO_GRID_MODE: los episodios se crean usando un grid. Si falso, de forma aleatoria
# DO_ONLY_GOOG_APS: quedarse solo con las ap buenas
# -----------------------------------------------------------------
# ENV_N_EPISODES_PER_FP: numero de episodios que se generan por Fingerprint
# ENV_STEP: at each step, how many centimters the agent moves
# ENV_TH_CLOSE: the agent finishes when it is at least this value close to the real value
# ENV_BIG_REWARD: big reward value. It is when the agent reach the solution
# ENV_SMALL_REWARD: small reward value. It is when the agent perform a valid action, e.g. it is closer than in the previous step.
# ENV_MARGIN: maximum value (in cm) from the limits of the environment where the agent can be
# ENV_GRID_CELL_SIZE: size of cells in the environment in meters
# ENV_RSSI_TH: Threshold to discretize RSSI SIGNALS
# -----------------------------------------------------------------
# AGENT_MEM_MAX_CAPACITY: maxima capacidad de la memoria
# AGENT_LEARNING_RATE: tasa aprendizaje de la red neuronal
# AGENT_NN_INPUT: numero de entradas de la NN
# AGENT_NN_HIDDEN: numero de neuronas de la capa oculta
# AGENT_NN_OUTPUT: numero de salidas
# AGENT_BATCH_SIZE: numero de muestras que se usan para entrenar
# AGENT_MAX_STEPS_BY_EPISODE: numero maximo de pasos en cada episodio
# AGENT_EPS_START: e-greedy threshold start value
# AGENT_EPS_END: e-greedy threshold end value
# AGENT_EPS_DECAY: e-greedy threshold decay
# AGENT_GAMMA: Q-learning discount factor
# AGENT_ALPHA: Q-learning learning rate
# AGENT_N_ACTIONS: number of actions
# -----------------------------------------------------------------
# TH_PCT_GOOD_AP: minimum pct of samples with relevant data to be considered a ap as good
# TH_RSSI_GOOD: RSSI value considered as good enough
# KNN_K: k value for knn
# TRAINING_FILENAME: name of the resulting model after training
# database: database file
# -----------------------------------------------------------------
import json


class Configuration:
    def __init__(self, filename):
        self.conf = dict()
        self.set_defaults()
        self.set_from_file(filename)

    # -----------------------------------------------------------------
    # -----------------------------------------------------------------
    def set_defaults(self):
        self.conf["DO_TRAIN_TEST_SETS"] = False

        self.conf["DATABASE"] = ""
        self.conf["EXPERIMENT_NAME"] = ""

        self.conf["USE_GRID_MODE"] = True
        self.conf["GRID_CELL_SIZE"] = 0.5

        self.conf["USE_ONLY_GOOG_APS"] = False
        self.conf["GOOG_APS_TH_PCT"] = 0.3
        self.conf["GOOG_APS_TH_RSSI"] = -60

        self.conf["ENV_N_EPISODES_PER_FP"] = 9
        self.conf["ENV_STEP"] = 0.5
        self.conf["ENV_TH_CLOSE"] = 1
        self.conf["ENV_BIG_REWARD"] = 100
        self.conf["ENV_SMALL_REWARD"] = 1
        self.conf["ENV_MARGIN"] = 0.5
        self.conf["ENV_N_ACTIONS"] = 4

        self.conf["DO_KNN"] = True
        self.conf["KNN_K"] = 3

        self.conf["DO_QL"] = False
        self.conf["QL_DO_TRAINING"] = False
        self.conf["QL_RSSI_TH"] = 0.5
        self.conf["QL_N_RANDOM_TEST"] = 5
        self.conf["QL_N_EPOCHS"] = 5
        self.conf["QL_GAMMA"] = 0.8
        self.conf["QL_ALPHA"] = 0.8
        self.conf["QL_MAX_STEPS_BY_EPISODE"] = 100

    # -----------------------------------------
    # -----------------------------------------------------------------
    def save_configuration(self, filename):
        with open(filename, 'w') as fp:
            json.dump(self.conf, fp)

    # -----------------------------------------------------------------
    # -----------------------------------------------------------------
    def set_from_file(self, filename):
        with open(filename, 'r') as fp:
            d = json.load(fp)

        for key in d:
            self.conf[key] = d[key]

    # -----------------------------------------------------------------
    # -----------------------------------------------------------------
    def is_correct(self):
        all_ok = True
        if self.conf["DATABASE"] == "":
            print("DATABASE cannot be empty")
            all_ok = False
        elif self.conf["EXPERIMENT_NAME"] == "":
            print("EXPERIMENT_NAME cannot be empty")
            all_ok = False

        if all_ok:
            print("Configuration is OK")
        return all_ok

    # -----------------------------------------------------------------
    # -----------------------------------------------------------------
    def get_do_sets(self):
        return self.conf["DO_TRAIN_TEST_SETS"]

    def get_database(self):
        return self.conf["DATABASE"]

    def get_experiment_name(self):
        return self.conf["EXPERIMENT_NAME"]

    def get_use_grid_mode(self):
        return self.conf["USE_GRID_MODE"]

    def get_grid_cell_size(self):
        return self.conf["GRID_CELL_SIZE"]

    def get_use_only_good_aps(self):
        return self.conf["USE_ONLY_GOOG_APS"]

    def get_good_aps_th_pct(self):
        return self.conf["GOOG_APS_TH_PCT"]

    def get_good_aps_th_rssi(self):
        return self.conf["GOOG_APS_TH_RSSI"]

    def get_env_n_episodes_per_fp(self):
        return self.conf["ENV_N_EPISODES_PER_FP"]

    def get_env_step(self):
        return self.conf["ENV_STEP"]

    def get_env_th_close(self):
        return self.conf["ENV_TH_CLOSE"]

    def get_env_big_reward(self):
        return self.conf["ENV_BIG_REWARD"]

    def get_env_small_reward(self):
        return self.conf["ENV_SMALL_REWARD"]

    def get_env_margin(self):
        return self.conf["ENV_MARGIN"]

    def get_env_n_actions(self):
        return self.conf["ENV_N_ACTIONS"]

    def get_do_knn(self):
        return self.conf["DO_KNN"]

    def get_knn_k(self):
        return self.conf["KNN_K"]

    def get_do_ql(self):
        return self.conf["DO_QL"]

    def get_ql_do_training(self):
        return self.conf["QL_DO_TRAINING"]

    def get_ql_rssi_th(self):
        return self.conf["QL_RSSI_TH"]

    def get_ql_n_random_test(self):
        return self.conf["QL_N_RANDOM_TEST"]

    def get_ql_n_epochs(self):
        return self.conf["QL_N_EPOCHS"]

    def get_ql_gamma(self):
        return self.conf["QL_GAMMA"]

    def get_ql_alpha(self):
        return self.conf["QL_ALPHA"]

    def get_ql_max_steps_by_episode(self):
        return self.conf["QL_MAX_STEPS_BY_EPISODE"]
