"""Validate ENHSP plans and plan costs using val."""

import argparse
from dataclasses import dataclass
import json
import os
import re
import tempfile
from typing import Optional

from asnets.scripts.collate_num_baselines import fix_cost, process_subdir

# NOTE: this script needs val binaries to be placed at the correct location
# the binaries can be obtained from
#   https://github.com/KCL-Planning/VAL/tree/master


THIS_PATH = os.path.realpath(__file__)
ASNETS_PATH = os.path.realpath(os.path.join(os.path.dirname(THIS_PATH), '../../..'))
VAL_PATH = os.path.join(ASNETS_PATH, 'val-binaries/Validate')

ENHSP_PLAN_REGEX = r"(\d*)\.\d: (\(.*\))"

parser = argparse.ArgumentParser(
    description='Validate ENHSP plans and plan costs using val.')
parser.add_argument(
    'baseline_dir', type=str, metavar='baseline-dir',
    help='Path to the baseline results directory.')
parser.add_argument(
    '--fix-cost',
    action='store_true',
    help='Fix the plan cost in the baseline results directory.'
)


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




CONVERT_UNDERSCORES = ['delivery', 'farmland_ln', 'farmland', 'supply-chain']


def main():
    args = parser.parse_args()

    print(f'Validating plans in {args.baseline_dir}...')

    messages = []

    for subdir in os.listdir(args.baseline_dir):
        message = []
        result = process_subdir(subdir, args.baseline_dir)

        message.append(f'Validating plan for {result.domain} {result.problem} {result.planner}')

        if not result.success:
            message.append('  plan not found')
            # messages.append('\n'.join(message))
            continue

        # look into the arguments to see the domain and problem file
        domain_path = os.path.join(args.baseline_dir, subdir, 'domain.pddl')
        problem_path = os.path.join(args.baseline_dir, subdir, 'problem.pddl')

        with open(os.path.join(args.baseline_dir, subdir, 'stdout.txt'), 'r') as f:
            content = f.read()
            matches = re.findall(ENHSP_PLAN_REGEX, content)

            # the first capture group is the time step, the second is the action
            # we only care about the action, but order them by time step
            plan = [match[1] for match in sorted(matches, key=lambda x: int(x[0]))]

        if len(plan) > 8000:
            # val will fail if the plan is too long
            continue

        plan = '\n'.join(plan)

        if result.domain in CONVERT_UNDERSCORES:
            plan = plan.replace('_', '-')

        with tempfile.NamedTemporaryFile(mode='w') as plan_file:
            plan_file.write(plan)
            plan_file.flush()

            val_output = os.popen(
                f'{VAL_PATH} {domain_path} {problem_path} '
                f'{plan_file.name}').read()
            
        val_result = parse_val_output(val_output)
        
        if not val_result.plan_valid:
            message.append('  plan invalid!!!')
            messages.append('\n'.join(message))
            continue
    
        if val_result.plan_cost != result.cost:
            message.append('  plan cost mismatch!!!!!')
            message.append(f'    val cost: {val_result.plan_cost}')
            message.append(f'    baseline cost: {result.cost}')
            messages.append('\n'.join(message))

            if args.fix_cost:
                fix_cost(subdir, args.baseline_dir, val_result.plan_cost)
            continue
    
    print('\n\n'.join(messages))


if __name__ == '__main__':
    main()