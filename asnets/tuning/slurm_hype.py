"""Perform automatic hyperparameter tuning on a slurm environment."""

import argparse
from argparse import ArgumentParser
import json
import pickle
import ray
from ray import tune
from ray.air import session
from ray.tune.search.hyperopt import HyperOptSearch
import tempfile
import uuid

from slurm_manager import *
from executor import *

parser = ArgumentParser(
    'Perform automatic hyperparameter tuning on a slurm environment.')
parser.add_argument(
    '--work-dir', help='Working directory', default='~/projects/asnets/asnets')
parser.add_argument(
    '--max-par-trials', default=2, type=int,
    help='Maximum number of trials to run in parallel')
parser.add_argument(
    '--partition', default='planopt', help='Slurm partition to use')
parser.add_argument(
    '--qos', default='planopt', help='Slurm QoS to use')
parser.add_argument(
    '--email', default=None, help='Email address to send notifications to')
parser.add_argument(
    '--time-per-run', default='12:00:00',
    help='Time per trial on a single domain')
parser.add_argument(
    '--mem-per-trial', default=64, type=int, help='Memory per trial (in GB)')
parser.add_argument(
    '--cpu-per-trial', default=16, type=int, help='CPUs per trial')
parser.add_argument(
    '--ssh-remote', default='cluster1.cecs.anu.edu.au', help='SSH remote host')
parser.add_argument(
    '--num-trials', default=200, type=int, help='Number of trials to run')
parser.add_argument(
    'ssh_user', metavar='ssh-user', help='SSH user')
parser.add_argument(
    'arch_module', metavar='arch-module', help='Path to architecture module')
parser.add_argument(
    'prob_modules', metavar='prob-modules', nargs='+', 
    help='Paths to problem modules')


SEARCH_SPACE = {
    'num_layers': 2,
    'hidden_size': tune.randint(12, 30),
    'supervised_learning_rate': tune.loguniform(1e-5, 1e-2),
    'supervised_batch_size': tune.randint(50, 200),
    'opt_batch_per_epoch': tune.randint(50, 200),
    'dropout': tune.choice([0, 0.1, 0.25]),
    'l1_reg': 0.0,
    'l2_reg': tune.loguniform(1e-6, 1e-2),
    'target_rollouts_per_epoch': tune.randint(5, 40),
    'supervised_early_stop': tune.randint(10, 30),
}



def configure_arch_mod(executor: Executor, base_arch: str, config: dict) -> str:
    """Configure the architecture module with the given config.

    Args:
        executor (Executor): The executor to use to run commands remotely.
        base_arch (str): The path to the base architecture module.
        config (dict): The configuration to use.

    Returns:
        str : The path to the configured architecture module on the remote host.
    """
    # create a tune directory at the same level as the arch module remotely
    executor.run(f'mkdir -p {os.path.dirname(base_arch)}/tune')

    # read the contents of the base arch mod, this should exist locally
    with open(base_arch, 'r') as f:
        arch_mod = f.read()
    
    # add the config to the arch mod
    added_config = set()
    lines = arch_mod.split('\n')
    for i, line in enumerate(lines):
        if '=' not in line or line.startswith('#'):
            continue
        var = line.split('=')[0].strip()
        if var.lower() in config.keys():
            lines[i] = f'{var} = {config[var.lower()]}  # Modifed by tune'
            added_config.add(var.lower())
    
    # add any missing config
    lines.extend(['', '#### Added by tune ####'])
    for var in config:
        if var not in added_config and var != 'args':
            lines.append(f'{var.upper()} = {config[var]}')
    
    # write the new arch mod to the tune directory remotely
    arch_path = None
    with tempfile.NamedTemporaryFile('w') as f:
        f.write('\n'.join(lines) + '\n')
        f.flush()
        
        # make an unique arch_path, which should be a valid import path
        arch_path = '{}/tune/arch_{}.py'.format(
            os.path.dirname(base_arch), str(uuid.uuid4()).replace('-', '_'))
        executor.run(f'mkdir -p {os.path.dirname(arch_path)}')
        executor.run(f'touch {arch_path}')
        executor.put(f.name, arch_path)
    
    return arch_path


def collect_coverage(executor: Executor, arch_mod: str, prob_mod: str) -> float:
    """Collect the coverage of a given architecture module on a given problem
    module.

    Args:
        executor (Executor): The executor to use to run commands remotely.
        arch_mod (str): The import path of the architecture module.
        prob_mod (str): The import path of the problem module.

    Returns:
        float: The coverage of the architecture module on the problem module.
    """
    experiments = executor.run('ls experiment-results')[0].split('\n')

    expdir = None
    for subdir in experiments:
        try:
            prob, arch = subdir.split('-')[:2]
        except ValueError:
            continue

            if prob == prob_mod and arch == arch_mod:
                expdir = subdir
                break

    
    if expdir is None:
        raise Exception('Could not find experiment directory')
    
    coverage = 0
    subdirs = executor.run(f'ls experiment-results/{expdir}')[0].split('\n')
    for subdir in subdirs:
        results, err = executor.run(f'cat experiment-results/{expdir}/{subdir}/results.json')
        if err != '':
            # the results.json file doesn't exist, so this is a failed run
            continue
    
        results = json.loads(results)
        if results.get('all_goal_reached', [False])[0] and \
            results.get('no_train', True):
            coverage += 1

    return coverage


def perform_trial(config, reporter) -> None:
    """Perform a single trial.

    Args:
        config: The config to use.
        reporter: The reporter to use.
    """
    # set up basic remote stuff
    args = config['args']
    executor = SSHExecutor(args.ssh_remote, args.ssh_user)
    executor.working_dir = args.work_dir
    slurm = SlurmManager(executor)
    env = Environment(partition=args.partition, qos=args.qos)
    
    # screw Ray
    os.chdir(os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..'
    )))

    # get the architecture module
    arch_path = configure_arch_mod(executor, args.arch_module, config)

    # perform trials on each problem module sequentially
    coverages = []
    for i, prob_path in enumerate(args.prob_modules):
        def module_path_to_import_path(module_path: str) -> str:
            return module_path.replace('/', '.').replace('.py', '')

        prob_mod = module_path_to_import_path(prob_path)
        arch_mod = module_path_to_import_path(arch_path)
        prob_name = prob_path.split('/')[-1].replace('.py', '')

        job_config = JobConfig(
            name=prob_name,
            time=(*map(int, args.time_per_run.split(':')),),
            mem=args.mem_per_trial,
            cpus_per_task=args.cpu_per_trial)
        job = Job(
            cmds=[
                'pwd; hostname; data',
                'echo "Starting Job"',
                'singularity exec --nv /opt/apps/containers/tensorflow-jdk-rwang230523.sif python3 run_experiment {} {}'.format(
                    arch_mod,
                    prob_mod
                ),
                'echo "Job complete"',
                'data'
            ],
            config=job_config,
            env=env)
        job_id = slurm.submit(job)
        slurm.wait(job_id)

        coverages.append(collect_coverage(executor, arch_mod, prob_mod))

    reporter(coverage=sum(coverages))
        


def main():
    args = parser.parse_args()

    # setup ray
    ray.init()

    SEARCH_SPACE['args'] = args  # screw ray
    algo = HyperOptSearch(
        space=SEARCH_SPACE,
        metric='coverage',
        mode='max')
    algo.set_max_concurrency(args.max_par_trials)
    tune.run(
        perform_trial,
        search_alg=algo,
        num_samples=args.num_trials)

if __name__ == '__main__':
    main()