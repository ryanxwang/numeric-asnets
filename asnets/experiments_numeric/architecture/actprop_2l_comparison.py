"""A two-layer configuration for the action/proposition/fluent network with 
ENHSP."""

#### Training settings ####
MAX_OPT_EPOCHS = 1000
# train supervised or RL? (only supervised supported at the moment)
SUPERVISED = True
# learning rate
SUPERVISED_LEARNING_RATE = 0.0003  # EXPERIMENTAL
# can also specify some steps to jump down from initial rate (e.g [(10, 1e-3),
# (20, 1e-4)] jumps down to 1e-3 after 10 epochs, and down to 1e-4 after 20
# epochs)
LEARNING_RATE_STEPS = []  # EXPERIMENTAL
# batch size
SUPERVISED_BATCH_SIZE = 50  # EXPERIMENTAL
# number of batches of optimisation per epoch
OPT_BATCH_PER_EPOCH = 60  # EXPERIMENTAL
# num of epochs after which to do early stopping if success rate is high but
# doesn't increase (0 disables)
SUPERVISED_EARLY_STOP = 20
# save model every N epochs, in addition to normal saving behaviour (on success
# rate increase & at end of training); 0 disables additional saves
SAVE_EVERY_N_EPOCHS = 1
# regularisers; SOME regularisation is needed so that objective is bounded
# below & l2 seems like reasonable default
L2_REG = 0.005
L1_REG = 0.0
# can be used to turn on dropout (at training time only)
DROPOUT = 0.1
# maximum number of observations allowed for a problem
LIMIT_TRAIN_OBS_SIZE = 1200


#### Exploration settings ####
TEACHER_TIMEOUT_S = 1
EXPLORATION_ALGORITHM = 'static'
ROLLOUTS = 2
MIN_EXPLORED = 10
MAX_EXPLORED = 1000
EXPLORATION_LEARNING_RATIO = 1
MAX_REPLAY_SIZE = 15000
# controls strategy used to teacher the planner; try ANY_GOOD_ACTION if you
# want the ASNet to choose evenly between all actions that have minimal teacher
# Q-value, or THERE_CAN_ONLY_BE_ONE to imitate the single action that the
# planner would return if you just ran it on the current state
TRAINING_STRATEGY = 'THERE_CAN_ONLY_BE_ONE'
# use 'ROLLOUT' to only accumulate optimal policy rollouts, or 'ENVELOPE' to
# accumulate entire optimal policy envelopes
TEACHER_EXPERIENCE_MODE = 'ROLLOUT'


#### Evaluation settings ####
# deterministic eval, so only need one round
EVAL_ROUNDS = 1
DET_EVAL = True


#### Model settings ####
NUM_LAYERS = 2
HIDDEN_SIZE = 15
SKIP = True


#### Time limits ####
TIME_LIMIT_SECONDS = int(60 * 60 * 24)
EVAL_TIME_LIMIT_SECONDS = int(60 * 30)
ROUND_TURN_LIMIT = 1000
EVAL_ROUND_TURN_LIMIT = 10000


#### Data settings ####
USE_LMCUT_FEATURES = False
USE_FLUENTS = False
USE_COMPARISONS = True
USE_ACT_HISTORY_FEATURES = True
USE_NUMERIC_LANDMARKS = True
USE_CONTRIBUTIONS = False


#### Domain-specific settings ####
TEACHER_PLANNER = 'enhsp'
DOMAIN_TYPE = 'numeric'
ENHSP_CONFIG = 'hadd-gbfs'
