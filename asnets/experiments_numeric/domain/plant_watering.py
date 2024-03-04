"""Plant watering (gardening) problem"""

PDDL_DIR = '../problems/numeric/plant-watering'
COMMON_PDDLS = ['domain.pddl']
TRAIN_PDDLS = [
    'small_instances/mt-plant-watering-constrained/instance_4_1_1.pddl',
    'small_instances/mt-plant-watering-constrained/instance_4_2_1.pddl',
    'small_instances/mt-plant-watering-constrained/instance_4_1_2.pddl',
]  # yapf: disable
TRAIN_NAMES = None
TEST_RUNS = [
    (['small_instances/mt-plant-watering-constrained/instance_4_1_1.pddl'], None),
    (['small_instances/mt-plant-watering-constrained/instance_4_1_2.pddl'], None),
    (['small_instances/mt-plant-watering-constrained/instance_4_1_3.pddl'], None),
    (['small_instances/mt-plant-watering-constrained/instance_4_2_1.pddl'], None),
    (['small_instances/mt-plant-watering-constrained/instance_4_2_2.pddl'], None),
    (['small_instances/mt-plant-watering-constrained/instance_4_2_3.pddl'], None),
    (['small_instances/mt-plant-watering-constrained/instance_4_3_1.pddl'], None),
    (['small_instances/mt-plant-watering-constrained/instance_4_3_2.pddl'], None),
    (['small_instances/mt-plant-watering-constrained/instance_4_3_3.pddl'], None),
]  # yapf: disable
