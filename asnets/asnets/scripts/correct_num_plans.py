"""Validate ASNets plans for numeric planning and correct plan costs according
to the actual metric using val."""

import argparse
from dataclasses import dataclass
import json
import os
import tempfile
from typing import Optional

# NOTE: this script needs val binaries to be placed at the correct location
# the binaries can be obtained from
#   https://github.com/KCL-Planning/VAL/tree/master


THIS_PATH = os.path.realpath(__file__)
ASNETS_PATH = os.path.realpath(os.path.join(os.path.dirname(THIS_PATH), '../../..'))
VAL_PATH = os.path.join(ASNETS_PATH, 'val-binaries/Validate')

parser = argparse.ArgumentParser(
    description='Validate ASNets plans for numeric planning and correct plan '
                'costs according to the actual metric using val.')
parser.add_argument(
    'experiment_dir', type=str, metavar='experiment-dir',
    help='Path to the experiment directory.')


@dataclass
class ValResult:
    """Class to hold the results of val."""
    plan_cost: Optional[float]
    plan_valid: bool


def parse_val_output(val_output: str) -> ValResult:
    """Parse the output of val and return the plan cost and whether the plan is
    valid.
    
    Args:
        val_output (str): The output of val.
    
    Returns:
        ValResult: The parsed results.
    """
    valid = 'Plan valid' in val_output
    cost = None

    if valid:
        for line in val_output.split('\n'):
            if line.startswith('Final value'):
                cost = float(line.split(': ')[-1])
                break
    
    return ValResult(cost, valid)


def main():
    args = parser.parse_args()

    print(f'Validating plans in {args.experiment_dir}...')

    subdirs = [
        os.path.join(args.experiment_dir, d) 
        for d in os.listdir(args.experiment_dir)
        if d.startswith('P[')
    ]

    for subdir in subdirs:
        results_path = os.path.join(subdir, 'results.json')

        if not os.path.exists(results_path):
            continue
        results = json.load(open(results_path, 'r'))

        # ignoring the training results
        if not results['no_train']:
            continue

        print(f'Validating plan for {results["problem"]}')

        # look into the arguments to see the domain and problem file
        domain_path = results['all_args'][-2].replace('..', ASNETS_PATH)
        problem_path = results['all_args'][-1].replace('..', ASNETS_PATH)

        for i in range(results['trials']):
            # get and format the plan how val expects it
            plan = '\n'.join(map(lambda x: f'({x})', results['trial_paths'][i]))

            with tempfile.NamedTemporaryFile(mode='w') as plan_file:
                plan_file.write(plan)
                plan_file.flush()

                val_output = os.popen(
                    f'{VAL_PATH} {domain_path} {problem_path} '
                    f'{plan_file.name}').read()
                
            val_result = parse_val_output(val_output)
            results['all_goal_reached'][i] = val_result.plan_valid
            results['all_costs'][i] = val_result.plan_cost
        
        # mark this as validated
        results['val_checked'] = True
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()