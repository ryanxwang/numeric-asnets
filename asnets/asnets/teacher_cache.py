# TODO merge this and Teacher

from abc import ABC, abstractmethod
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from asnets.state_reprs import CanonicalState, sample_next_state, \
    simulate_plan, simulate_policy

from asnets.utils.pddl_utils import extract_domain_problem, hlist_to_sexprs

LOGGER = logging.getLogger(__name__)


TEACHER_CACHE_STATS = True
CACHE_HITS = 0
CACHE_MISSES = 0
PRINT_COUNTER = 0
PRINT_FREQUENCY = 1000


class TeacherException(Exception):
    """Generic exception to signal that teacher has timed out or failed while
    trying to execute specified operation."""
    pass

# NOTE there is a distinction between a teacher failing and returning no plan.
# In the former case, we don't know if the teacher would have returned a plan
# given infinite time and should raise. In the latter case, we know that the
# teacher cannot find a plan (likely there is no plan), so just treat it as
# normal.

class TeacherCache(ABC):
    @abstractmethod
    def __init__(self, planner_exts: 'PlannerExtensions'):
        self.planner_exts = planner_exts
        self.state_value_cache: Dict[CanonicalState.TupState, float] = {}
        self.best_action_cache: Dict[CanonicalState.TupState, str] = {}
        # Implementation decides what to blacklist
        self._blacklist: Set[Any] = set()
        
        self.problem_name = planner_exts.problem_name
        domain_h_list, domain_name, problem_hlist, problem_name_pddl = \
            extract_domain_problem(planner_exts.pddl_files, self.problem_name)
        
        assert self.problem_name == problem_name_pddl, \
            f'problem name mismatch: {self.problem_name} vs {problem_name_pddl}'
        
        self._domain_source = hlist_to_sexprs(domain_h_list)
        self._problem_hlist = problem_hlist
        self.domain_name = domain_name
    
    def compute_policy_envelope(self, cstate: CanonicalState) \
        -> List[CanonicalState]:
        """Compute the 'optimal policy envelope' for a given state. Here the
        optimal policy envelope really just means a path to the goal (likely
        the exact path the teacher would take), which is a different notion to
        that of policy envelope for a probabilistic problem.
    
        Raises:
            TeacherException: If the teacher fails to generate plans for any
            state asked for.
        
        Args:
            cstate (CanonicalState): The given state.

        Returns:
            List[CanonicalState]: List of states visited on the way to the goal,
            not including the goal state.
        """
        # warm up the state value cache
        state_value = self.compute_state_value(cstate)
        if state_value is None:
            return []
    
        return simulate_policy(cstate,
                               self.best_action_cache,
                               self.planner_exts)
        
    def compute_q_values(self, cstate: CanonicalState, act_strs: List[str]) \
        -> List[Optional[float]]:
        """Computes Q-values for a list of actions in a given state.

        Raises:
            TeacherException: If the teacher fails to generate plans for any
            state after applying one of the given actions.

        Args:
            cstate (CanonicalState): The given state.
            act_strs (List[str]): List of action strings in mdpsim format (e.g.,
            with parentheses).

        Returns:
            List[Optional[float]]: List of Q-values for each action. If an
            action is not enabled or leads to a dead end, its Q-value is None.
        """
        q_values = []
        for act_str in act_strs:
            # Enforce mdpsim format
            assert act_str[0] == '(' and act_str[-1] == ')', \
                'Action strings must be in mdpsim format'
            action_id = self.planner_exts.problem_meta \
                .act_unique_id_to_index(act_str.strip('()'))
            is_enabled = cstate.acts_enabled[action_id][1]
            
            if not is_enabled:
                q_values.append(None)
                continue
        
            next_state, cost = sample_next_state(cstate, action_id,
                                                 self.planner_exts)
            next_state_value = self.compute_state_value(next_state)
            
            if next_state_value is None:
                # this action leads to a dead end
                q_values.append(None)
            else:
                q_value = cost + next_state_value
                q_values.append(q_value)
            
        return q_values
            
        
    def compute_state_value(self,
                            cstate: CanonicalState,
                            need_best_action: bool = False) \
        -> Union[Optional[float], Tuple[Optional[float], Optional[str]]]:
        """Compute the state value of a given state.
        
        This method is memoized, so it is safe to call repeatedly. Each call to
        an uncached state will call the underlying teacher to generate a plan.
        A states in the plan will have their values and best actions cached or
        updated (if better).

        Raises:
            TeacherException: If the teacher fails to generate a plan for the
            given state.
        
        Args:
            cstate (CanonicalState): The given state.
            need_best_action (bool, optional): Whether to return the best action
            along side the state value. Defaults to False.
        
        Returns:
            Optional[float]: The state value of the given state. If the state is
            terminal, state value is 0 if it is a goal state, and None
            otherwise. If the state is not terminal, state value is the
            cost of the (best) plan by the teacher. If no plan is available,
            state value is None.
            Optional[str]: The best action to take in the given state. Only
            returned when need_best_action is True. If no plan is available,
            this is None, otherwise it is the action recommended by the teacher,
            without parenthesis.
        """
        global CACHE_HITS, CACHE_MISSES, PRINT_COUNTER, PRINT_FREQUENCY, TEACHER_CACHE_STATS
        
        def return_filter(state_value, best_action):
            if need_best_action:
                return state_value, best_action
            else:
                return state_value

        PRINT_COUNTER += 1
        if TEACHER_CACHE_STATS and PRINT_COUNTER % PRINT_FREQUENCY == 0:
            LOGGER.info(f'[{self.planner_exts.problem_name}] Teacher cache hits: {CACHE_HITS}, misses: {CACHE_MISSES}, miss rate: {CACHE_MISSES / (CACHE_HITS + CACHE_MISSES):.4f}')
            PRINT_COUNTER = 0

        tup_state = cstate.to_tup_state()
        if tup_state in self.state_value_cache:
            CACHE_HITS += 1
            return return_filter(self.state_value_cache[tup_state],
                                 self.best_action_cache[tup_state])
        
        CACHE_MISSES += 1
        
        if cstate.is_terminal:
            cost = 0 if cstate.is_goal else None
            self.state_value_cache[tup_state] = cost
            self.best_action_cache[tup_state] = None
            return return_filter(cost, None)
    
        plan = self._get_plan(tup_state)

        if plan is None:
            # could not find a plan
            self.state_value_cache[tup_state] = None
            self.best_action_cache[tup_state] = None
            return return_filter(None, None)
        
        # visit all states except the last
        try:
            visited_states, step_costs \
                = simulate_plan(cstate, plan, self.planner_exts)
        except ValueError as e:
            LOGGER.warn('Teacher found invalid plan')
            raise TeacherException(f'Teacher found invalid plan') from e

        costs_to_goal = np.cumsum(step_costs[::-1])[::-1]
        visited_states = visited_states[:-1]
        assert len(visited_states) == len(plan), \
            f'Plan visited {len(visited_states)} states, \
                but has {len(plan)} actions'
        
        states_acts_costs = zip(visited_states, plan, costs_to_goal)
        for this_cstate, this_act, cost_to_goal in states_acts_costs:
            this_tup_state = this_cstate.to_tup_state()
            if this_tup_state in self.state_value_cache:
                old_val = self.state_value_cache[this_tup_state]
                if old_val is not None and cost_to_goal > old_val:
                    continue
            self.state_value_cache[this_tup_state] = cost_to_goal
            self.best_action_cache[this_tup_state] = this_act
        
        return return_filter(self.state_value_cache[tup_state],
                             self.best_action_cache[tup_state])

    @abstractmethod
    def _get_plan(self, tup_state: CanonicalState.TupState) \
        -> Optional[List[str]]:
        """Get a plan for a given state. This method should not cache, but it
        should blacklist states that it fails to find a plan for.

        Raises:
            TeacherException: If the teacher fails to generate a plan for the
            given state.
        
        Args:
            tup_state (CanonicalState.TupState): The given state.
        
        Returns:
            Optional[List[str]]: The plan for the given state, or None if no
            plan is found.
        """
        pass
