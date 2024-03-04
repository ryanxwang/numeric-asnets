"""Sailing problem"""

PDDL_DIR = '../problems/numeric/sailing'
COMMON_PDDLS = ['domain.pddl']
TRAIN_PDDLS = [
    'instances/instance_1_4_1229.pddl',
    'instances/instance_1_5_1229.pddl',
    'instances/instance_2_1_1229.pddl',
]  # yapf: disable
TRAIN_NAMES = None
TEST_RUNS = [
    (['instances/instance_1_4_1229.pddl'], None),
    (['instances/instance_1_5_1229.pddl'], None),
    (['instances/instance_1_6_1229.pddl'], None),
    (['instances/instance_1_7_1229.pddl'], None),
    (['instances/instance_1_10_1229.pddl'], None),
    (['instances/instance_2_1_1229.pddl'], None),
    (['instances/instance_2_2_1229.pddl'], None),
    (['instances/instance_2_4_1229.pddl'], None),
    (['instances/instance_2_7_1229.pddl'], None),
    (['instances/instance_2_8_1229.pddl'], None),
    (['instances/instance_2_9_1229.pddl'], None),
    (['instances/instance_2_10_1229.pddl'], None),
    (['instances/instance_3_4_1229.pddl'], None),
    (['instances/instance_3_5_1229.pddl'], None),
    (['instances/instance_3_6_1229.pddl'], None),
    (['instances/instance_4_2_1229.pddl'], None),
    (['instances/instance_4_5_1229.pddl'], None),
    (['instances/instance_4_6_1229.pddl'], None),
    (['instances/instance_4_7_1229.pddl'], None),
    (['instances/instance_4_10_1229.pddl'], None),
]  # yapf: disable

# use hadd-gbfs