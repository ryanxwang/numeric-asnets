PDDL_DIR = '../problems/numeric/counters'
COMMON_PDDLS = ['domain.pddl']
TRAIN_PDDLS = [
    'instances/fz_instance_4.pddl',
    'instances/inv_instance_4.pddl',
    'instances/rnd_instance_4_1.pddl',
]  # yapf: disable
TRAIN_NAMES = None
TEST_RUNS = [
    ([f'vanilla/fz_instance_{i}.pddl'], None)
    for i in range(2, 61)
]

# should use hmrmax-astar