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
class configuration:
    def __init__(self):
        self.DO_SETS = False
        self.DO_GRID_MODE = True
        self.DO_ONLY_GOOG_APS = False

        self.DO_KNN = False
        self.DO_DRL = False
        self.DO_DRL_TRAINING = False
        self.DO_QL = True
        self.DO_QL_TRAINING = True

        self.ENV_N_EPISODES_PER_FP = 9
        self.ENV_STEP = 0.5
        self.ENV_TH_CLOSE = 1
        self.ENV_BIG_REWARD = 100
        self.ENV_SMALL_REWARD = 1
        self.ENV_MARGIN = 0.5
        self.ENV_GRID_CELL_SIZE = 0.5
        self.ENV_RSSI_TH = 0.3

        self.AGENT_MEM_MAX_CAPACITY = 1000
        self.AGENT_LEARNING_RATE = 0.001
        self.AGENT_NN_INPUT = 170           # N_WAPS + 2
        self.AGENT_NN_HIDDEN = 128
        self.AGENT_NN_OUTPUT = 5
        self.AGENT_BATCH_SIZE = 100
        self.AGENT_MAX_STEPS_BY_EPISODE = 100
        self.AGENT_EPS_START = 0.9  # e-greedy threshold start value
        self.AGENT_EPS_END = 0.01   # e-greedy threshold end value
        self.AGENT_EPS_DECAY = 200  # e-greedy threshold decay
        self.AGENT_GAMMA = 0.8      # Q-learning discount factor
        self.AGENT_ALPHA = 0.8
        self.AGENT_N_ACTIONS = 4
        self.AGENT_N_EPOCHS = 5
        self.AGENT_N_RANDOM_TEST = 5

        self.TH_PCT_GOOD_AP = 0.3
        self.TH_RSSI_GOOD = -60
        self.KNN_K = 3
        self.TRAINING_FILENAME = "../MODELS/ml.model"

        self.DATABASE = "../DATA/mini.csv"
        #self.DATABASE = "../DATA/juguete.csv"
        #self.DATABASE = "../DATA/IPIN16.csv"
