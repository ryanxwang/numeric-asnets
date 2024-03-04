#!/usr/bin/env python3

import argparse
from importlib import import_module
import os
import os.path as osp
import subprocess


from asnets.interfaces.metricff_interface import METRIC_FF_PATH
from asnets.utils.pddl_utils import extract_domain_problem

TIMEOUT_MIN = 30
MAX_MEMORY_GB = 32  # double IPC; this is actually quite a bit of RAMc
THIS_DIR = osp.dirname(osp.abspath(__file__))
ASNETS_ROOT = osp.abspath(osp.join(THIS_DIR, '../../'))
RESULTS_DIR = osp.join(ASNETS_ROOT, 'experiment-results/baselines-metricff/')



def run_metricff_raw(domain_text,
                     problem_text,
                     result_dir,
                     *,
                     timeout_s = None):

    try:
        os.makedirs(result_dir)
    except OSError:
        return
    domain_path = osp.join(result_dir, 'domain.pddl')
    problem_path = osp.join(result_dir, 'problem.pddl')
    with open(domain_path, 'w') as dom_fp, open(problem_path, 'w') as prob_fp:
        dom_fp.write(domain_text)
        prob_fp.write(problem_text)
    
    command = [
        METRIC_FF_PATH,
        '-o', domain_path,
        '-f', problem_path
    ]
    command_path = os.path.join(result_dir, 'command.sh')
    with open(command_path, 'w') as command_file:
        command_file.write(' '.join(command))
    
    out_path = osp.join(result_dir, 'stdout.txt')
    err_path = osp.join(result_dir, 'stderr.txt')
    with open(out_path, 'w') as out_file, open(err_path, 'w') as err_file:
        proc = subprocess.Popen(
            command,
            stdout=out_file,
            stderr=err_file,
            cwd=result_dir,
            universal_newlines=True)
    
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()


def do_metricff_run(domain_path, problem_path):
    _, domain_name, _, problem_name = extract_domain_problem(
        [domain_path, problem_path])
    dname = '%s:%s' % (domain_name, problem_name)
    result_dir = osp.join(RESULTS_DIR, dname)
    with open(domain_path) as dom_fp, open(problem_path) as prob_fp:
        domain_txt = dom_fp.read()
        prob_txt = prob_fp.read()
        run_metricff_raw(domain_txt, prob_txt, result_dir,
                      timeout_s=TIMEOUT_MIN*60)
    

def get_module_test_problems(prob_mod_name):
    """Get the path to the domain and a list of paths to problems from a given
    experiment module name (e.g "experiments.det_gripper")."""
    prob_mod = import_module(prob_mod_name)
    domain_fname, = prob_mod.COMMON_PDDLS
    domain_path = osp.abspath(osp.join(
        ASNETS_ROOT, prob_mod.PDDL_DIR, domain_fname))
    problems_names = []
    for prob_fnames, name in prob_mod.TEST_RUNS:
        prob_fname, = prob_fnames
        prob_path = osp.abspath(osp.join(
            ASNETS_ROOT, prob_mod.PDDL_DIR, prob_fname))
        # this is hashable! Woo!
        problems_names.append((prob_path, name))
    return domain_path, problems_names


parser = argparse.ArgumentParser(
    description='Run numeric baselines')
parser.add_argument(
    'prob_module',
    metavar='prob-module',
    help='import path for Python file with problem config')

def main():
    args = parser.parse_args()
    # go back to root dir (in case running from script)
    os.chdir(ASNETS_ROOT)

    # load up all problems
    print(f'Importing problem from {args.prob_module}')
    domain_path, problems_names = get_module_test_problems(args.prob_module)
    problems = []
    for problem, name in problems_names:
        assert name is None, "I don't support named problems yet (should " \
            "be easy to do though)"
        problems.append(problem)

    for problem in problems:
        print(f'Problem: {problem}')

        do_metricff_run(domain_path, problem)


if __name__ == '__main__':
    main()
