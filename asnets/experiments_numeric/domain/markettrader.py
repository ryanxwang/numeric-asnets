"""markettrader problem"""

PDDL_DIR = '../problems/numeric/markettrader'
COMMON_PDDLS = ['domain.pddl']
TRAIN_PDDLS = [
    'train/market_train1.pddl',
    'train/market_train2.pddl',
    'train/market_train3.pddl',
    'train/market_train4.pddl',
]  # yapf: disable
TRAIN_NAMES = None
TEST_RUNS = [
    (['instances/pfile01.pddl'], None),
    (['instances/pfile02.pddl'], None),
    (['instances/pfile03.pddl'], None),
    (['instances/pfile04.pddl'], None),
    (['instances/pfile05.pddl'], None),
    (['instances/pfile06.pddl'], None),
    (['instances/pfile07.pddl'], None),
    (['instances/pfile08.pddl'], None),
    (['instances/pfile09.pddl'], None),
    (['instances/pfile10.pddl'], None),
    (['instances/pfile11.pddl'], None),
    (['instances/pfile12.pddl'], None),
    (['instances/pfile13.pddl'], None),
    (['instances/pfile14.pddl'], None),
    (['instances/pfile15.pddl'], None),
    (['instances/pfile16.pddl'], None),
    (['instances/pfile17.pddl'], None),
    (['instances/pfile18.pddl'], None),
    (['instances/pfile19.pddl'], None),
    (['instances/pfile20.pddl'], None),
]  # yapf: disable

# use hadd-gbfs