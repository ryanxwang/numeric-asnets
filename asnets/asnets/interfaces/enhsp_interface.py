"""Tools for interfacing with the unified-planning package by aiplan4eu.
"""

from enhsp_wrapper.enhsp import ENHSP, PlanningStatus
import logging
from typing import List, Optional

from asnets.state_reprs import CanonicalState
from asnets.teacher_cache import TeacherCache, TeacherException
from asnets.utils.pddl_utils import hlist_to_sexprs, replace_init_state

LOGGER = logging.getLogger(__name__)

BLACKLIST_OUTCOMES = frozenset([
    PlanningStatus.TIMEOUT,
    PlanningStatus.MEMEOUT,
    PlanningStatus.ERROR
])


# predefined configurations for ENHSP
ENHSP_CONFIGS = {
    'hadd-gbfs': '-s gbfs -h hadd',
    'hadd-astar': '-s WAStar -wh 1.0 -h hadd',
    'hadd-wastar-1.1': '-s WAStar -wh 1.1 -h hadd',
    'hadd-wastar-2': '-s WAStar -wh 2 -h hadd',
    'hmrp-gbfs': '-s gbfs -h hmrp',
    'hmrp-astar': '-s WAStar -wh 1.0 -h hmrp',
    'hmrp-wastar-1.1': '-s WAStar -wh 1.1 -h hmrp',
    'hmrp-wastar-2': '-s WAStar -wh 2 -h hmrp',
    'hmrp-ha-gbfs': '-s gbfs -h hmrp -ha true',
    'hmrp-ha-astar': '-s WAStar -wh 1.0 -h hmrp -ha true',
    'hmrp-ha-wastar-1.1': '-s WAStar -wh 1.1 -h hmrp -ha true',
    'hmrp-ha-wastar-2': '-s WAStar -wh 2 -h hmrp -ha true',
    'hmrp-ha-ht-gbfs': '-s gbfs -h hmrp -ha true -ht true',
    'hmrp-ha-ht-astar': '-s WAStar -wh 1.0 -h hmrp -ha true -ht true',
    'hmrp-ha-ht-wastar-1.1': '-s WAStar -wh 1.1 -h hmrp -ha true -ht true',
    'hmrp-ha-ht-wastar-2': '-s WAStar -wh 2 -h hmrp -ha true -ht true',
    'hmrmax-astar': '-planner opt-hrmax',
}


class ENHSPCache(TeacherCache):
    def __init__(self,
                 planner_exts: 'PlannerExtensions',
                 enhsp_config: str = 'hadd-gbfs',
                 timeout_s: float = 1800):
        super().__init__(planner_exts)

        self.teacher_params = ENHSP_CONFIGS[enhsp_config] \
            + f' -timeout {timeout_s}'
        self.timeout_s = timeout_s
    
    def _get_plan(self, tup_state: CanonicalState.TupState) \
        -> Optional[List[str]]:
        """Get a plan from the teacher.

        Raises:
            TeacherException: If the given state is blacklisted or the teacher
            fails due to one of the BLACKLIST_OUTCOMES

        Args:
            tup_state (CanonicalState.TupState): The state to get a plan for.

        Returns:
            Optional[List[str]]: The plan, or None if the teacher determined
            a negative outcome, meaning it likely believes there is no plan.
        """
        if tup_state in self._blacklist:
            raise TeacherException('State is blacklisted.')
        
        problem_hlist = replace_init_state(self._problem_hlist, tup_state)
        problem_source = hlist_to_sexprs(problem_hlist)

        planner = ENHSP(self.teacher_params)
        result = planner.plan_from_string(self._domain_source, problem_source)
        
        # Handle the possible outcomes
        if result.status == PlanningStatus.UNKNOWN:
            LOGGER.warn(f'ENHSP returned UNKNOWN status with problem instance {problem_source}, assuming unsolvable.')
            return None
        if result.status == PlanningStatus.UNSOLVABLE:
            return None
        if result.status in BLACKLIST_OUTCOMES:
            self._blacklist.add(tup_state)
            # these shouldn't happen too often with right training problems
            # be verbose about it
            err_msg = 'ENHSP failed: status {}'.format(str(result.status))
            # err_msg += f', problem source {problem_source}'
            raise TeacherException(err_msg)
        
        assert result.status == PlanningStatus.SUCCESS
        
        plan = [
            a.strip('()') 
            for a in result.plan]
        return plan
            
