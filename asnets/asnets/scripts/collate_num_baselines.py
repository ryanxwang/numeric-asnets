#!/usr/bin/env python3
"""Turn stdout.txt files produced by deterministic baselines into .json files
which can be processed by plot_ctime.py (or whatever I called it)."""

import argparse
from collections import namedtuple
import json
import os
import re

from asnets.scripts.collate_results import name_to_size

FIELD_RE = re.compile(r'^(.+):\s*(-?\d+(\.\d+)?)')

RunResult = namedtuple(
    'RunResult',
    ['planner', 'domain', 'problem', 'size', 'success', 'cost', 'time_taken', 'plan_length'])


def extract_kv_from_lines(fp):
    # extract information
    info_dict = {}
    success = False
    for line in fp:
        if 'Found Plan' in line:
            success = True
        match = FIELD_RE.match(line)
        if match is None:
            continue
        fname, num, frac_part = match.groups()
        if frac_part is None:
            value = int(num)
        else:
            value = float(num)
        key = fname.strip()
        info_dict[key] = value
    return info_dict, success


def parse_subdir(subdir):
    planner, domain, problem = subdir.split(':')
    size = name_to_size(problem)
    return planner, domain, problem, size


def process_subdir(subdir, results_dir):
    print('Processing %s' % subdir)
    stdout_path = os.path.join(results_dir, subdir, 'stdout.txt')
    with open(stdout_path, 'r') as fp:
        info_dict, success = extract_kv_from_lines(fp)
    planner, domain, problem, size = parse_subdir(subdir)

    time_taken = None
    cost = info_dict.get('Metric (Search)', None)
    plan_length = info_dict.get('Plan-Length', None)

    if success:
        time_taken = info_dict.get('Grounding Time') \
            + info_dict.get('H1 Setup Time (msec)') \
            + info_dict.get('Planning Time (msec)')
        time_taken /= 1000.0
    
    if os.path.exists(os.path.join(results_dir, subdir, 'fixed-cost.txt')):
        with open(os.path.join(results_dir, subdir, 'fixed-cost.txt'), 'r') as fp:
            cost = float(fp.read())
    
    return RunResult(
        planner=planner,
        domain=domain,
        problem=problem,
        size=size,
        # did we get to the goal?
        success=success,
        # how much did our plan cost?
        cost=cost,
        # how long did it take?
        time_taken=time_taken,
        # how long is the plan, different to cost when there is a metric
        plan_length=plan_length)

def fix_cost(subdir, results_dir, new_cost):
    path = os.path.join(results_dir, subdir, 'fixed-cost.txt')
    with open (path, 'w') as fp:
        fp.write(str(new_cost))

parser = argparse.ArgumentParser(
    description='Collate numeric baseline results into .json file')
parser.add_argument(
    'results_dir', help='directory in which to look for results')
parser.add_argument('out_dir', help='directory to write .json result files to')


def main():
    args = parser.parse_args()
    results_dir = args.results_dir
    subdirs = os.listdir(results_dir)
    results_by_domain_planner = {}
    planners = set()
    for subdir in subdirs:
        try:
            single_result = process_subdir(subdir, results_dir)
        except Exception as ex:
            print("Error processing '%s': %s" % (subdir, ex))
            continue
        key = (single_result.domain, single_result.planner)
        results_by_domain_planner.setdefault(key, []).append(single_result)
        planners.add(single_result.planner)

    # now sort by size, ascending
    for value in results_by_domain_planner.values():
        value.sort(key=lambda r: r.size)

    sorted_results = sorted(
        results_by_domain_planner.items(), key=lambda t: t[0])
    for key, results in sorted_results:
        domain, planner = key
        eval_runs = []
        prob_names = []
        prob_sizes = []
        for run in results:
            run_result = {
                'goal_reached': [run.success],
                'time': [run.time_taken],
                'cost': [run.cost],
            }
            eval_runs.append(run_result)
            prob_names.append(run.problem)
            prob_sizes.append(run.size)
        result = {
            'eval_names': prob_names,
            'eval_sizes': prob_sizes,
            'eval_runs': eval_runs,
        }
        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, 'num:%s:%s.json' %
                                (domain, planner))
        print('Writing to %s' % out_path)
        with open(out_path, 'w') as fp:
            json.dump(result, fp)


if __name__ == '__main__':
    main()
