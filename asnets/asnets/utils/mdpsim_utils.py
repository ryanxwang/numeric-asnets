"""Convenience functions for dealing with MDPSim"""
import os
from types import ModuleType
from typing import Any, List


SPECIAL_FUNCTIONS = {'goal-achieved', 'total-time'}

class PDDLLoadError(Exception):
    """PDDL parse exception"""


def parse_problem_args(mdpsim_module: ModuleType,
                       pddls: List[str],
                       problem_name: str = None) -> Any:
    """Parse a problem from a given MDPSim module.

    Args:
        mdpsim_module (ModuleType): The MDPSim module.
        pddls (List[str ]): Path to parse.
        problem_name (str, optional): Name of problem to look for. 
        Defaults to None.

    Raises:
        PDDLLoadError: If the problem could not be loaded.

    Returns:
        Any: The problem.
    """
    for pddl_path in pddls:
        success = mdpsim_module.parse_file(pddl_path)
        if not success:
            raise PDDLLoadError('Could not parse %s' % pddl_path)

    problems = mdpsim_module.get_problems()
    if problem_name is None:
        if len(problems) == 0:
            raise PDDLLoadError('Did not load any problems (?!), aborting')
        sorted_keys = sorted(problems.keys())
        problem = problems[sorted_keys[0]]
    else:
        try:
            problem = problems[problem_name]
        except KeyError:
            raise PDDLLoadError(
                'Could not find problem %s. Available problems: %s' % (
                    problem_name, ', '.join(problems.keys())))
    return problem
