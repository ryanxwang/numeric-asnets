"""An interface to the Metric-FF planner.
"""

import logging
import os
import shutil
import subprocess
from typing import List, Optional
import uuid

from asnets.state_reprs import CanonicalState
from asnets.teacher_cache import TeacherCache, TeacherException
from asnets.utils.pddl_utils import hlist_to_sexprs, replace_init_state

LOGGER = logging.getLogger(__name__)

METRIC_FF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'metricff-x86'
)

class MetricFFQValueCache(TeacherCache):
    def __init__(self,
                 planner_exts: 'PlannerExtensions',
                 timeout_s: float = 1800):
        super().__init__(planner_exts)
        self.timeout_s = timeout_s
    
    def _get_plan(self, tup_state: CanonicalState.TupState) \
        -> Optional[List[str]]:
        """Get a plan from the teacher.

        Raises:
            TeacherException: If the given state is blacklisted or the teacher
            times out.

        Args:
            tup_state (CanonicalState.TupState): The state to get a plan for.

        Returns:
            Optional[List[str]]: The plan, or None if the teacher terminated but
            failed to get a plan.
        """
        if tup_state in self._blacklist:
            raise TeacherException('State is blacklisted.')
        
        # Make a temporary directory to store some files to call metricff with
        guid = uuid.uuid1().hex
        result_dir = os.path.join('/tmp', f'metricff-{self.domain_name}-\
            {self.problem_name}-{guid}')
        
        os.makedirs(result_dir, exist_ok=True)
        
        # Prepare all the files to write to disk
        domain_path = os.path.join(result_dir, 'domain.pddl')
        
        problem_hlist = replace_init_state(self._problem_hlist, tup_state)
        problem_source = hlist_to_sexprs(problem_hlist)
        problem_path = os.path.join(result_dir, 'problem.pddl')

        command = [
            METRIC_FF_PATH,
            '-o', domain_path,
            '-f', problem_path
        ]
        command_path = os.path.join(result_dir, 'command.sh')

        with open(domain_path, 'w') as domain_file:
            domain_file.write(self._domain_source)
        with open(problem_path, 'w') as problem_file:
            problem_file.write(problem_source)
        with open(command_path, 'w') as command_file:
            command_file.write(' '.join(command))
        
        out_path = os.path.join(result_dir, 'stdout.txt')
        err_path = os.path.join(result_dir, 'stderr.txt')
        with open(out_path, 'w') as out_file, open(err_path, 'w') as err_file:
            proc = subprocess.Popen(
                command,
                stdout=out_file,
                stderr=err_file,
                cwd=result_dir,
                universal_newlines=True)

        try:
            proc.wait(timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            self._blacklist.add(tup_state)
            raise TeacherException('Timeout expired.')
        
        with open(out_path, 'r') as out_file, open(err_path, 'r') as err_file:
            out_text = out_file.read()
            err_text = err_file.read()

        if 'problem proven unsolvable' in out_text:
            return None

        plan = self._extract_plan_from_stdout(out_text)
        if plan is None:
            LOGGER.error(f'Metric-FF failed, result_dir: {result_dir}')
        else:
            # clean up the temporary directory
            shutil.rmtree(result_dir)
        
        return plan
        
    def _extract_plan_from_stdout(self, stdout: str) -> Optional[List[str]]:
        """Extract the plan from the stdout of Metric-FF.
        
        Args:
            stdout (str): The stdout of Metric-FF.
        
        Returns:
            Optional[List[str]]: The plan, or None if no plan was found.
        """
        lines = stdout.lower().split('\n')
        plan_start_idx = None
        for i, line in enumerate(lines):
            if line == 'ff: found legal plan as follows':
                plan_start_idx = i + 2
                break
        
        if plan_start_idx is None:
            return None
        
        plan = []
        for line in lines[plan_start_idx:]:
            if ': ' not in line:
                break
            plan.append(line.split(': ')[1])
        
        return plan
        