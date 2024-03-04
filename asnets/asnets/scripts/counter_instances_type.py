from argparse import ArgumentParser
from collections import defaultdict
import json

parser = ArgumentParser(description='Process different types of counter instances')
parser.add_argument(
    'json',
    help='single counter experiment json file')


def main():
    args = parser.parse_args()
    
    with open(args.json, 'r') as f:
        data = json.load(f)
    
    # type suffixes
    vanilla = [str(i) for i in range(2, 21)] + ['24', '28', '32', '36', '40']
    scaled = ['5a', '5b', '6a', '6b', '7a', '7b']
    random_start = ['4a', '5c', '5d', '5e', '6c', '6d', '6e', '7c', '7d', '7e']
    
    result = defaultdict(lambda: defaultdict(int))
    for i, instance in enumerate(data['eval_names']):
        instance = instance.split('_')[1]
        eval_run = data['eval_runs'][i]
        
        type = None
        if instance in vanilla:
            type = 'vanilla'
        elif instance in scaled:
            type = 'scaled'
        elif instance in random_start:
            type = 'random_start'
        
        assert type is not None

        solved = eval_run['goal_reached'][0]
        cost = eval_run['cost'][0]
        if not solved:
            print(f'Failed to solve instance {instance} with type {type}')
        else:
            print(f'Solved instance {instance} with type {type} and cost {cost}')
        
        
        result[type]['total'] += 1
        result[type]['solved'] += 1 if solved else 0
    
    print(result)
        

if __name__ == '__main__':
    main()