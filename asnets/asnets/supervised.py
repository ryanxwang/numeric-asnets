from collections import Counter, deque
from enum import Enum
from functools import lru_cache
from itertools import repeat
import joblib
import logging
import numpy as np
import os
import rpyc
import setproctitle
import shutil
import tensorflow as tf
from time import time
import tqdm
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from asnets.heur_inputs import ActionCountDataGenerator, \
    HeuristicDataGenerator, LMCutDataGenerator, RelaxedDeadendDetector, \
    NumericLandmarkGenerator
from asnets.utils.mdpsim_utils import parse_problem_args
from asnets.prob_dom_meta import BoundAction, DomainType, get_domain_meta, \
    get_problem_meta
from asnets.interfaces.jpddl_interface import start_jvm
from asnets.interfaces.ssipp_interface import set_up_ssipp
from asnets.multiprob import to_local
from asnets.state_reprs import compute_observation_dim, compute_action_dim, \
    get_action_name, sample_next_state, get_init_cstate, CanonicalState
from asnets.teacher import DomainSpecificTeacher, FDTeacher, MetricFFTeacher, \
    SSiPPTeacher, Teacher, TeacherException, ENHSPTeacher
from asnets.utils.prof_utils import can_profile
from asnets.utils.pddl_utils import get_domain_file, get_problem_file
from asnets.utils.py_utils import RandomPopContainer, TimerContext, \
    strip_parens, weak_ref_to, weighted_batch_iter
from asnets.utils.tf_utils import cross_entropy, empty_feed_value, \
    escape_name_tf

import jpype
import jpype.imports
from jpype.types import *

J_PDDLDomain = None
J_PDDLProblem = None

LOGGER = logging.getLogger(__name__)


@jpype.onJVMStart
def _import_java_classes() -> None:
    """Import Java classes that will be used by this module. This is called
    automatically upon JVM start-up.
    """
    global J_PDDLDomain, J_PDDLProblem

    J_PDDLDomain = jpype.JPackage('com').hstairs.ppmajal.domain.PDDLDomain
    J_PDDLProblem = jpype.JPackage('com').hstairs.ppmajal.problem.PDDLProblem


class WeightedReplayBuffer:
    """Replay buffer for previously-encountered states. The 'weighted' in the
    name comes from the fact that it's really a multiset that lets you sample
    states weighted by multiplicity."""

    def __init__(self):
        """Initialize the replay buffer."""
        self.counter = Counter()
        self.added_items = deque()

    def update(self, new_elems: Iterable[Any]) -> None:
        """Add new elements to the replay buffer.

        Args:
            new_elems (Iterable[Any]): New elements to add to the replay
            buffer.
        """
        item_counter = Counter(new_elems)
        self.counter.update(item_counter)
        self.added_items.append(item_counter)

    def __len__(self) -> int:
        """Get the number of unique elements in the replay buffer.

        Returns:
            int: Number of unique elements in the replay buffer.
        """
        return len(self.counter)

    def get_full_dataset(self) -> Tuple[List[Any], List[int]]:
        """Get the full dataset stored in the replay buffer.

        Returns:
            Tuple[List[Any], List[int]]: List of elements in the replay buffer
            and list of their counts.
        """
        rich_dataset = list(self.counter)
        counts = [self.counter[item] for item in rich_dataset]
        return rich_dataset, counts
    
    def remove_oldest(self):
        """Remove the oldest element from the replay buffer."""
        # make sure we do not empty the replay buffer
        if len(self.added_items) <= 1:
            return
        
        item_counter = self.added_items.popleft()
        self.counter.subtract(item_counter)
        self.counter += Counter()  # remove zero and negative counts


class ProblemServiceConfig(object):
    """Configuration for a ProblemService. This is a separate class so that
    the config can be serialised and sent to the remote server."""

    def __init__(
            self,
            pddl_files: List[str],
            init_problem_name: str,
            domain_type: DomainType,
            *,
            ssipp_dg_heuristic: str = None,
            use_lm_cuts: bool = False,
            use_numeric_landmarks: bool = False,
            use_contributions: bool = False,
            use_act_history: bool = False,
            # ??? what does this do?
            # Oh, it controls the maximum length of training trajectories! That
            # explains why I'm not able to solve some certain big training
            # problems.
            # FIXME: this max_len should be adjusted based on the V(s0)
            # calculated by the teacher planner! Maybe add a separate method
            # for that (like "exposed_find_path_length") that plans on the
            # first state & uses the result to figure out what length should
            # be.
            fd_heuristic="astar-hadd",
            ssipp_teacher_heuristic: str = 'lm-cut',
            enhsp_config: str = 'hadd-gbfs',
            max_len: int = 50,
            teacher_planner: str,
            random_seed: int = None,
            teacher_timeout_s: int = 1800,
            only_one_good_action: bool = False,
            use_teacher_envelope: bool = True):
        """Initialise a ProblemServiceConfig. This Config will allow
        initialisation of a ProblemService, which involves:
        - Initialising mdpsim and ssipp (requires pddl_files, problem_name)
        - Initialising data generators. This might be easiest to achieve with
          just a list of generator class names and arguments (although I
          still need to make sure those are actually deep copied, grumble
          grumble).

        Args:
            pdll_files (List[str]): List of PDDL files to load.
            init_problem_name (str): Name of the problem to load.
            domain_type (DomainType): Type of the domain.
            ssipp_dg_heuristic (str, optional): Name of the heuristic to use.
            Defaults to None.
            use_lm_cuts (bool, optional): Whether to use lm-cut heuristic.
            Defaults to False.
            use_act_history (bool, optional): Whether to use action history
            as input to the heuristic. Defaults to False.
            fd_heuristic (str, optional): Name of the heuristic to use for 
            FastDownward. Defaults to 'astar-hadd'.
            ssipp_teacher_heuristic (str, optional): Name of the heuristic to
            use for SSiPP. Defaults to 'lm-cut'.
            enhsp_config (str, optional): Name of the configuration to use for
            unified-planning when using ENHSP. Defaults to None.
            max_len (int, optional): Maximum length of training trajectories.
            Defaults to 50.
            teacher_planner (str, optional): Name of the planner to use for
            teacher. Defaults to None.
            random_seed (int, optional): Random seed to use. Defaults to None.
            only_one_good_action (bool, optional): Whether to only use the
            teacher action as a positive example. Controls whether planner
            should return accurate Q-values (False) or return Q-values that only
            make its favourite action look good (True). Defaults to False.
            use_teacher_envelope (bool, optional): Whether to use an entire
            policy envelope from teacher (True), or just a rollout (False).
            Defaults to True.
        """
        self.pddl_files = pddl_files
        self.init_problem_name = init_problem_name
        self.domain_type = domain_type
        self.ssipp_dg_heuristic = ssipp_dg_heuristic
        self.use_lm_cuts = use_lm_cuts
        self.use_numeric_landmarks = use_numeric_landmarks
        self.use_contributions = use_contributions
        self.use_act_history = use_act_history
        self.fd_heuristic = fd_heuristic
        self.ssipp_teacher_heuristic = ssipp_teacher_heuristic
        self.enhsp_config = enhsp_config
        self.max_len = max_len
        self.random_seed = random_seed
        self.teacher_planner = teacher_planner
        self.teacher_timeout_s = teacher_timeout_s
        self.only_one_good_action = only_one_good_action
        self.use_teacher_envelope = use_teacher_envelope


class PlannerExtensions(object):
    """Wrapper to hold references to SSiPP and MDPSim modules, and references
    to the relevant loaded problems (like the old ModuleSandbox). Mostly
    keeping this because it makes it convenient to pass stuff around, as I
    often need SSiPP and MDPSim at the same time."""

    def __init__(self,
                 pddl_files: List[str],
                 init_problem_name: str,
                 domain_type: DomainType,
                 *,
                 dg_ssipp_heuristic_name: str = None,
                 dg_use_lm_cuts: bool = False,
                 dg_use_numeric_landmarks: bool = False,
                 dg_use_contributions: bool = False,
                 dg_use_act_history: bool = False):
        """Initialise a PlannerExtensions object.

        Args:
            pddl_files (List[str]): The PDDL files to load.
            init_problem_name (str): The name of the problem to load.
            domain_type (DomainType): The type of the domain.
            dg_ssipp_heuristic_name (str, optional): The heuristic feature
            generator to use. Defaults to None.
            dg_use_lm_cuts (bool, optional): Whether to use the lm-cut heuristic
            feature generator. If the domain is numeric, will perform numeric
            relaxation. Defaults to False.
            dg_use_numeric_landmarks (bool, optional): Whether to use the
            additive numeric landmarks feature generator. Defaults to False.
            dg_use_act_history (bool, optional): Whether to use the action count
            data generator. Defaults to False.
        """
        self.pddl_files = pddl_files
        self.domain_type = domain_type
        LOGGER.info('Parsing %d PDDL files for domain type %s',
                    len(self.pddl_files), domain_type.name)

        import mdpsim  # noqa: F811
        import ssipp  # noqa: F811

        # MDPSim stuff
        self.mdpsim: ModuleType = mdpsim
        self.mdpsim_problem = parse_problem_args(self.mdpsim, self.pddl_files,
                                                 init_problem_name)
        self.problem_name: str = self.mdpsim_problem.name.strip()

        # Maps to PyGroundAction object in MDPSim. Cannot use type hint.
        self.act_ident_to_mdpsim_act: Dict[str, Any] = {
            strip_parens(a.identifier): a
            for a in self.mdpsim_problem.ground_actions
        }

        # Python-side extra data
        self.domain_meta = get_domain_meta(self.mdpsim_problem.domain)
        self.problem_meta = get_problem_meta(self.mdpsim_problem,
                                             self.domain_meta)

        # Either use JPDDL (numeric) or SSiPP (otherwise), ugly!
        if self.domain_type == DomainType.NUMERIC:
            domain_file = get_domain_file(self.pddl_files)
            problem_file = get_problem_file(self.pddl_files, self.problem_name)
            assert domain_file is not None
            assert problem_file is not None

            start_jvm()
            self.j_domain = J_PDDLDomain(domain_file)
            self.j_problem = J_PDDLProblem(
                problem_file,
                self.j_domain
            )
            self.j_problem.prepareForSearch(
                True,  # enable AIBR preprocessing
                False  # stop after grounding
            )

            if dg_use_lm_cuts:
                # set up SSiPP using numeric relaxed problems
                self.ssipp: ModuleType = ssipp
                self.ssipp_problem = set_up_ssipp(
                    self.ssipp, self.pddl_files, self.problem_name,
                    use_numeric_relaxation=True)
                
                self.ssipp_ssp_iface = ssipp.SSPfromPPDDL(self.ssipp_problem)
                
        elif self.domain_type == DomainType.PROBABILISTIC:
            # SSiPP stuff
            self.ssipp: ModuleType = ssipp
            self.ssipp_problem = set_up_ssipp(self.ssipp, self.pddl_files,
                                              self.problem_name)
            # this leaks for some reason; will store it here so I don't have to
            # reconstruct
            #
            # This is an object of the SSPfromPPDDL class, which inherits the
            # interface SSPIface supporting the following methods:
            # - s0: Get initial state for problem
            # - isGoal: Check whether given state is a goal state
            # - applicableActions: List of actions applicable in this state.
            self.ssipp_ssp_iface = ssipp.SSPfromPPDDL(self.ssipp_problem)

        # now set up data generators
        data_gens = [
        ]
        
        # Domain type specific data generators
        if self.domain_type == DomainType.PROBABILISTIC:
            data_gens.append(RelaxedDeadendDetector(weak_ref_to(self)))

            if dg_ssipp_heuristic_name is not None:
                heur_gen = HeuristicDataGenerator(
                    weak_ref_to(self), dg_ssipp_heuristic_name)
                data_gens.append(heur_gen)

        elif self.domain_type == DomainType.NUMERIC and \
                dg_use_numeric_landmarks:
            numeric_landmark_gen = NumericLandmarkGenerator(
                weak_ref_to(self),
                dg_use_contributions)
            data_gens.append(numeric_landmark_gen)

        # Generic data generators
        if dg_use_act_history:
            ad_data_gen = ActionCountDataGenerator(self.problem_meta)
            data_gens.append(ad_data_gen)

        if dg_use_lm_cuts:
            lm_cut_gen = LMCutDataGenerator(weak_ref_to(self))
            data_gens.append(lm_cut_gen)


        self.data_gens = data_gens

    @property
    def ssipp_dead_end_value(self) -> int:
        """Get the value of the dead end state in SSiPP.

        Returns:
            int: The value of the dead end state in SSiPP.
        """
        # HACK We no longer always initialise SSiPP as it is problematic
        # (especially for numeric domains). SSiPP just hardcodes this as 500.
        return 500


def make_problem_service(config, set_proc_title=False):
    """Construct Service class for a particular problem. Note that we must
    construct classes, not instances (unfortunately), as there is no way of
    passing arguments to the service's initialisation code (AFAICT).

    The extra set_proc_title arg can be set to True if you want the
    ProblemService to figure out a descriptive name for the current process in
    top/htop/etc. It's mostly useful when you're starting a single subprocess
    per environment, and you want to know which subprocess corresponds to which
    environment."""
    assert isinstance(config, ProblemServiceConfig)

    class ProblemService(rpyc.Service):
        """Spools up a new Python interpreter and uses it to sandbox SSiPP and
        MDPSim. Can interact with this to train a Q-network."""

        def exposed_collect_trajectory(self, model) -> bool:
            """Collect a single trajectory using the given policy (represented
            as a function from flattened observation vectors to action
            numbers)."""
            return self.internal_collect_trajectory(
                model, stochastic=self.stochastic)
        
        def exposed_explore_from_trajectories(self):
            self.internal_explore_from_trajectories()
        
        def exposed_explore_from_random_state(self):
            self.internal_explore_from_random_state()

        def exposed_dataset_is_empty(self):
            return len(self.replay) == 0

        def exposed_weighted_dataset(self):
            """Return weighted dataset.

            Returns:
                Tuple[np.ndarray, np.ndarray, np.ndarray]: The dataset.
                The first element is tensor of observations (cstates as
                network inputs). The second element is tensor of Q-values at
                each cstate, ordered in the same way as bound_acts_ordered. The
                third element is the weight of each cstate, which is really just
                a count of how many times we saw that cstate.
            """
            rich_obs_qvs, counts = self.replay.get_full_dataset()
            assert len(rich_obs_qvs) > 0, "Empty replay %s" % (self.replay, )
            counts = np.asarray(counts, dtype='float32')
            obs_tensor, qv_tensor = self.flatten_obs_qvs(rich_obs_qvs)
            return obs_tensor, qv_tensor, counts

        def exposed_env_reset(self):
            self.current_state = get_init_cstate(self.p)
            return self.current_state

        def exposed_action_name(self, action_num):
            action_num = to_local(action_num)
            return get_action_name(self.p, action_num)

        def exposed_env_step(self, action_num):
            action_num = to_local(action_num)
            next_cstate, step_cost \
                = sample_next_state(self.current_state, action_num, self.p)
            self.current_state = next_cstate
            return self.current_state, step_cost

        # note to self: RPyC doesn't support @property

        def exposed_get_ssipp_dead_end_value(self):
            return self.p.ssipp_dead_end_value

        def exposed_get_meta(self):
            """Get name, ProblemMeta and DomainMeta for the current problem."""
            return self.problem_meta, self.domain_meta

        def exposed_get_replay_size(self):
            return len(self.replay)
        
        def exposed_trim_replay(self):
            LOGGER.info(f'[{self.p.problem_name}] trimming replay buffer')
            self.replay.remove_oldest()

        def exposed_get_obs_dim(self):
            if not hasattr(self, '_cached_obs_dim'):
                self._cached_obs_dim = compute_observation_dim(self.p)
            return self._cached_obs_dim

        def exposed_get_act_dim(self):
            if not hasattr(self, '_cached_act_dim'):
                self._cached_act_dim = compute_action_dim(self.p)
            return self._cached_act_dim

        def exposed_get_dg_extra_dim(self):
            # TODO: factor this logic out into another function, since it's
            # used in several places (grep for '\.extra_dim for' or something)
            data_gens = self.p.data_gens
            return sum([g.extra_dim for g in data_gens])

        def exposed_get_max_len(self):
            return self.max_len

        def exposed_get_problem_names(self):
            # fetch a list of all problems loaded by MDPSim
            return sorted(self.p.mdpsim.get_problems().keys())

        def exposed_get_current_problem_name(self):
            return self.p.problem_name
        
        def exposed_get_num_traj_states(self):
            return len(self.traj_states)
        
        def exposed_get_num_new_pairs(self):
            return len(self.expl_states)
        
        def exposed_finish_explore(self):
            LOGGER.info("[{}] generated {} pairs".format(
                        self.p.problem_name, len(self.expl_states)))
            self.replay.update(self.expl_states)
            self.traj_states.clear()
            self.model_cache = {}
            self.expl_states.clear()

        def exposed_initialise(self):
            assert not self.initialised, "Can't double-init"

            self.p = PlannerExtensions(
                config.pddl_files,
                config.init_problem_name,
                config.domain_type,
                dg_ssipp_heuristic_name=config.ssipp_dg_heuristic,
                dg_use_lm_cuts=config.use_lm_cuts,
                dg_use_numeric_landmarks=config.use_numeric_landmarks,
                dg_use_contributions=config.use_contributions,
                dg_use_act_history=config.use_act_history)
            self.domain_meta = self.p.domain_meta
            self.problem_meta = self.p.problem_meta
            self.only_one_good_action = config.only_one_good_action
            self.use_teacher_envelope = config.use_teacher_envelope

            self.traj_states = RandomPopContainer()
            self.model_cache = {}
            self.expl_states = set()

            if config.teacher_planner == 'fd':
                # TODO: consider passing in teacher heuristic here, too; that
                # should give me more control over how the FD teacher works
                # (and let me do inadm. vs. adm. comparisons, among other
                # things)
                self.teacher = FDTeacher(
                    self.p,
                    heuristic=config.fd_heuristic,
                    timeout_s=config.teacher_timeout_s)
            elif config.teacher_planner == 'ssipp':
                self.teacher = SSiPPTeacher(
                    self.p,
                    'lrtdp',
                    config.ssipp_teacher_heuristic,
                    timeout_s=config.teacher_timeout_s)
            elif config.teacher_planner == 'domain-specific':
                self.teacher = DomainSpecificTeacher(self.p)
            elif config.teacher_planner == 'enhsp':
                self.teacher = ENHSPTeacher(
                    self.p,
                    config.teacher_timeout_s,
                    enhsp_config=config.enhsp_config)
            elif config.teacher_planner == 'metricff':
                self.teacher = MetricFFTeacher(
                    self.p,
                    timeout_s=config.teacher_timeout_s)

            # maximum length of a trace to gather
            self.max_len = config.max_len
            # will hold (state, action) pairs to train on
            self.replay = WeightedReplayBuffer()
            # current state for stateful Gym-like methods
            self.current_state = get_init_cstate(self.p)
            # hack to decide whether to get one or many rollouts (XXX)
            self.first_rollout = True

            if set_proc_title:
                # SPT_NOENV truncates the new title to avoid clobbering
                # /proc/PID/environ
                os.environ['SPT_NOENV'] = '1'
                old_title = setproctitle.getproctitle()
                new_title = '[%s] %s' % (self.problem_meta.name, old_title)
                setproctitle.setproctitle(new_title)

            self.stochastic = True

            self.initialised = True

        def on_connect(self, conn):
            # we let the initialiser run later, so that it can execute
            # asynchronously (starting up PlannerExtensions & Planner is
            # expensive because it requires grounding the relevant problem)
            self.initialised = False

        # FIXME: don't cache at this level; it's inefficient when using
        # history-level features, b/c it will lead to lots and lots of
        # near-identical cstates being thrown into the cache
        @lru_cache(None)
        def opt_pol_experience(self, cstate: CanonicalState) \
                -> List[Tuple[CanonicalState, List[Tuple[BoundAction, float]]]]:
            """Get optimal policy from given state.

            Args:
                cstate (CanonicalState): Canonical state to start from.

            Returns:
                List[Tuple[CanonicalState, List[Tuple[BoundAction, float]]]]:
                Experience from the optimal policy, as a list of (state,
                [(action, q-value), ...]) tuples.
            """
            return planner_trace(self.teacher, self.p, cstate,
                                 self.only_one_good_action,
                                 self.use_teacher_envelope)


        def internal_collect_trajectory(self,
                                        model: Callable,
                                        stochastic: bool) -> bool:
            """Collect a single trajectory using the given policy. Add the
            trajectory to the internal trajectory collection.
            
            Args:
                mode (Callable): The policy to use.
                max_len (int): The maximum length of the trajectory.
                stochastic (bool): Whether to use stochastic mode.
            
            Returns:
                bool: Whether the trajectory was successful.
            """
            prob_meta = self.p.problem_meta
            path = []
            hit_goal = False
            cstate = get_init_cstate(self.p)
            
            for _ in range(self.max_len):
                obs = to_local(cstate.to_network_input())
                obs_bytes = obs.tostring()
                if obs_bytes not in self.model_cache:
                    act_dist = model(obs[None], training=False)
                    
                    act_dist = tf.reshape(
                        to_local(act_dist),
                        [
                            -1,
                        ],
                    ).numpy()
                    if not stochastic:
                        chosen = int(np.argmax(act_dist))
                    else:
                        act_dist = act_dist / np.sum(act_dist)
                        chosen = int(
                            np.random.choice(np.arange(act_dist.shape[0]), p=act_dist)
                        )
                    # this cache update is actually thread-safe too thanks to
                    # Python's GIL
                    self.model_cache[obs_bytes] = chosen

                action = self.model_cache[obs_bytes]

                path.append((cstate, prob_meta.bound_acts_ordered[action]))

                cstate, _ = sample_next_state(cstate, action, self.p)
                if cstate.is_terminal:
                    if cstate.is_goal:
                        hit_goal = True
                    break
                
            for cstate, _ in path:
                self.traj_states.add(cstate)

            return hit_goal
        
        def internal_explore_from_trajectories(self) -> None:
            """Explore from the trajectory states."""
            while len(self.traj_states) > 0:
                self.internal_explore_from_random_state()
            
        def internal_explore_from_random_state(self) -> None:
            """Explore from a random state."""
            cstate = self.traj_states.pop_random()

            try:
                teacher_experience = self.opt_pol_experience(cstate)
            except TeacherException as ex:
                LOGGER.warn(f'Teacher error on problem \
                    {self.p.problem_name} ({ex})')
                return

            filtered_envelope = []

            for env_cstate, act in teacher_experience:
                nactions = sum(p[1] for p in env_cstate.acts_enabled)

                if nactions <= 1:
                    # skip states
                    continue
                filtered_envelope.append((env_cstate, act))

            self.expl_states.update(filtered_envelope)
            

        def flatten_obs_qvs(self, rich_obs_qvs):
            cstates, rich_qvs = zip(*rich_obs_qvs)
            obs_tensor = np.stack(
                [s.to_network_input() for s in cstates], axis=0)
            qv_lists = []
            for qv_pairs in rich_qvs:
                qv_dict = dict(qv_pairs)
                qv_list = [
                    qv_dict[ba] for ba in self.problem_meta.bound_acts_ordered
                ]
                qv_lists.append(qv_list)
            qv_tensor = np.array(qv_lists, dtype=float)
            return obs_tensor, qv_tensor

    return ProblemService




@lru_cache(None)
def mock_qvalues(planner: Teacher,
                 planner_exts: PlannerExtensions,
                 action: Optional[str]):
    prob_meta = planner_exts.problem_meta
    if action is None:
        # no good action
        num_acts = len(prob_meta.bound_acts_ordered)
        q_values = [planner.dead_end_value] * num_acts
    else:
        assert action is not None
        planner_action_ident = action.strip('()')
        assert not planner_action_ident.startswith(')') \
            and not planner_action_ident.endswith(')')
        q_values = []
        found = False
        unique_idents = [
            ba.unique_ident for ba in prob_meta.bound_acts_ordered
        ]
        for unique_ident in unique_idents:
            if unique_ident == planner_action_ident:
                q_values.append(0)
                found = True
            else:
                q_values.append(planner.dead_end_value)
        assert found, \
            "no match for '%s' in '%s'" \
            % (planner_action_ident, ", ".join(unique_idents))
    
    return q_values


@can_profile
def planner_trace(planner: Teacher,
                  planner_exts: PlannerExtensions,
                  root_cstate: CanonicalState,
                  only_one_good_action: bool,
                  use_teacher_envelope: bool) \
        -> List[Tuple[CanonicalState, List[Tuple[BoundAction, float]]]]:
    """Extract (s, [q*]) pairs for all s reachable from (state) under some
    (arbitrary) optimal policy.

    Args:
        planner (Teacher): The teacher object to use for planning.
        planner_exts (PlannerExtensions): The planner extensions object.
        root_cstate (CanonicalState): The root state to start planning from.
        only_one_good_action (bool): If True, only the best action will be
        used for each state. This makes planning much faster, but may have an
        effect on learning (either good or bad) in some domains.
        use_teacher_envelope (bool): If True, the expert policy envelope will be
        used for planning. If False, the expert policy rollout will be used.

    Returns:
        List[Tuple[CanonicalState, List[Tuple[BoundAction, float]]]]: A list of
        states with their corresponding Q-values.
    """
    # TODO: do I need to explicitly cache this, or is extract_policy_envelope
    # fast enough?
    prob_meta = planner_exts.problem_meta
    pairs = []
    # not sure how expensive this is, but IIRC not very, so it shouldn't matter
    # if we do it on every epoch
    if use_teacher_envelope:
        pol_list = planner.extract_policy_envelope(root_cstate)
    else:
        pol_list = planner.expert_policy_rollout(root_cstate)
    for i, new_cstate in enumerate(pol_list):
        if only_one_good_action:
            # Shortcut: we get the planner to give us just the single best
            # action, and then construct a vector of pseudo-Q-values which will
            # favour that action. This makes planning much faster, and may have
            # an effect on learning (either good or bad) in some domains.
            planner_action_raw = planner.single_action_label(new_cstate)
            q_values = mock_qvalues(planner, planner_exts, planner_action_raw)
        else:
            # otherwise, get real q-values for all enabled actions; rest get
            # dead_end_value
            en_indices = []
            en_act_names = []
            for idx, (ba, en) in enumerate(new_cstate.acts_enabled):
                if not en:
                    continue
                en_indices.append(idx)
                en_act_names.append('(%s)' % ba.unique_ident)
            en_q_values = planner.q_values(new_cstate, en_act_names)
            assert len(en_q_values) == len(en_indices)
            q_values = [planner.dead_end_value] * len(new_cstate.acts_enabled)
            for idx, value in zip(en_indices, en_q_values):
                q_values[idx] = value

        assert len(prob_meta.bound_acts_ordered) == len(q_values)
        qv_tuple = tuple(zip(prob_meta.bound_acts_ordered, q_values))
        pairs.append((new_cstate, qv_tuple))

    return pairs


class SupervisedObjective(Enum):
    # use xent loss to choose any action with minimal Q-value
    ANY_GOOD_ACTION = 0
    # maximise expected teacher advantage of action taken by policy
    MAX_ADVANTAGE = 1
    # get the teacher to give you an arbitrary good action and use xent loss to
    # match exactly that action (& not the others); makes planning faster!
    THERE_CAN_ONLY_BE_ONE = 2


class SupervisedTrainer:
    @can_profile
    def __init__(self,
                 problems,
                 weight_manager,
                 summary_writer,
                 explorer,
                 strategy,
                 start_time,
                 scratch_dir,
                 snapshot_dir,
                 *,
                 batch_size=64,
                 lr=0.001,
                 lr_steps=[],
                 opt_batches_per_epoch=300,
                 l1_reg_coeff,
                 l2_reg_coeff,
                 l1_l2_reg_coeff,
                 save_training_set=None,
                 use_saved_training_set=None,
                 hide_progress=False,
                 time_out=1000,
                 early_stop=20,
                 save_every=20,
                 dk="dk"
                 ):
        # gets incremented to deal with TF
        self.batches_seen = 0
        self.problems = problems
        self.weight_manager = weight_manager
        # may be None if no summaries tuple()should be written
        self.summary_writer = summary_writer
        self.explorer = explorer
        self.batch_size_per_problem = max(batch_size // len(problems), 1)
        self.opt_batches_per_epoch = opt_batches_per_epoch
        self.hide_progress = hide_progress
        self.strategy = strategy
        self.tf_init_done = False
        self.lr = lr
        self.l1_reg_coeff = l1_reg_coeff
        self.l2_reg_coeff = l2_reg_coeff
        self.l1_l2_reg_coeff = l1_l2_reg_coeff
        self.timer = TimerContext()
        self.save_training_set = save_training_set
        self.use_saved_training_set = use_saved_training_set
        if use_saved_training_set:
            LOGGER.info("Loading saved training set from '%s'",
                        use_saved_training_set)
            self.loaded_training_set = joblib.load(use_saved_training_set)
        lr_steps = [(0, lr)] + sorted(lr_steps)
        for k, lr in lr_steps:
            assert k >= 0, "one of the steps was negative (?)"
            assert isinstance(k, int), \
                "one of the LR step epoch nums (%s) was not an int" % (k, )
            assert lr > 0, \
                "one of the given learning rates was not positive (?)"
        self.lr_steps = lr_steps
        self.lr_steps_remaining = list(lr_steps)
        self.start_time = start_time
        self.timeout = time_out
        self.early_stop = early_stop
        self.save_every = save_every
        self.scratch_dir = scratch_dir
        self.snapshot_dir = snapshot_dir
        self.dk = dk
        self._init_tf()

    @can_profile
    def _init_tf(self):
        """Do setup necessary for network (e.g. initialising weights)."""
        assert not self.tf_init_done, \
            "this class is not designed to be initialised twice"

        LOGGER.info('Initialising network structure')

        if len(self.lr_steps) > 1:
            # using a scheduler to control the learning rate
            boundaries = [i[0] for i in self.lr_steps[1:]]
            values = [i[1] for i in self.lr_steps]
            lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                boundaries, values)
            self.optimiser = tf.keras.optimizers.Adam(
                learning_rate=lr_scheduler)
        else:
            self.optimiser = tf.keras.optimizers.Adam(learning_rate=self.lr)

        self.loss_fn = ManualLoss(
            problems=self.problems,
            weight_manager=self.weight_manager,
            summary_writer=self.summary_writer,
            l1_reg_coeff=self.l1_reg_coeff,
            l2_reg_coeff=self.l2_reg_coeff,
            l1_l2_reg_coeff=self.l1_l2_reg_coeff,
            name="loss_fn",
            strategy=SupervisedObjective.ANY_GOOD_ACTION
        )
        # tensorboard ops
        self._log_ops = {}

        # self.sess.graph.finalize()
        self.tf_init_done = True

    def _optimise(self, n_batches):
        params = self.weight_manager.all_weights

        # Do a check that set(params) is the same as what TF thinks it should
        # be. As tf.Variable is unhashable, we have to use ref() to get the key.
        param_set = set(map(lambda v: v.ref(), params))
        tf_param_set = set(map(
            lambda v: v.ref(),
            self.problems[0].policy.trainable_weights))

        assert param_set == tf_param_set, \
            "network has weird variables---debug this"

        all_batches_iter = self._make_batches(n_batches)
        tr = tqdm.tqdm(all_batches_iter, desc='batch', total=n_batches)

        start_time = time()
        losses = []
        for feed_dict in tr:
            # Each feed_dict is a list of batched data sets for each problem.
            # Each data set is a tuple of obs_tensor and q-value tensor.
            #
            # The obs_tensor has shape [batch_size, obs_dim]
            # The q-value tensor has shape [batch_size, num_actions]
            #
            # Second axis of he q-values are ordered in the same order as action
            # in bound_acts_ordered for the ProblemMeta.

            with tf.name_scope('grads_opt'):
                with tf.GradientTape() as tape:
                    obs_by_prob, qv_by_prob = list(zip(*feed_dict))
                    preds_by_prob = []
                    for i, problem in enumerate(self.problems):
                        preds_by_prob.append(problem.policy(obs_by_prob[i]))
                    loss = self.loss_fn(preds_by_prob, qv_by_prob)
                    grads = tape.gradient(loss, params)
                    grads_and_vars = zip(grads, params)
                    self.optimiser.apply_gradients(
                        grads_and_vars=grads_and_vars)

                    tr.set_postfix(loss=loss)
                    losses.append(loss)

                    if (self.batches_seen % 10) == 0:
                        tf.summary.scalar('train-loss', loss)

                    self.batches_seen += 1

        self.explorer.update_learning_time(time() - start_time)
        return np.mean(losses)

    def train(self, max_epochs):
        best_rate = None
        keep_going = True
        iter_num = 0
        time_since_best = 0
        # fraction of rollouts that have to reach goal in order for problem
        # to be considered "solved"
        solve_thresh = 0.999
        tr = tqdm.trange(max_epochs, desc='epoch', leave=True)
        mean_loss = None
        elapsed_time = time() - self.start_time

        # set up tensorboard logging
        epoch = tf.Variable(0, dtype=tf.int64)
        self.summary_writer.set_as_default(step=epoch)

        for epoch_num in tr:
            # update the epoch variable
            epoch.assign(epoch_num)

            # only extend replay by a bit each time
            succs_probs = self.explorer.extend_replay()
            total_succ_rate = np.mean([s for _, s in succs_probs])
            replay_sizes = self._get_replay_sizes()
            replay_size = sum(replay_sizes)

            tf.summary.scalar('lr', self.optimiser.lr)
            # update output
            tr.set_postfix(
                succ_rate=total_succ_rate,
                net_loss=mean_loss,
                states=replay_size,
                lr=self.optimiser.lr)
            tf.summary.scalar('succ-rate/mean', total_succ_rate)

            for prob, prob_succ_rate in succs_probs:
                pname = escape_name_tf(prob.name)
                tf.summary.scalar('succ-rate/%s' % pname, prob_succ_rate)

            tf.summary.scalar('replay-size', replay_size)
            mean_loss = self._optimise(self.opt_batches_per_epoch)
            iter_num += 1
            # update output again
            tr.set_postfix(
                succ_rate=total_succ_rate,
                net_loss=mean_loss,
                states=replay_size,
                lr=self.optimiser.lr)
            # caller might want us to terminate
            if best_rate is None or total_succ_rate > best_rate + 1e-4:
                time_since_best = 0
            elif total_succ_rate < best_rate and total_succ_rate < solve_thresh:
                # also reset to 0 if our success rate goes back down again
                time_since_best = 0
            else:
                time_since_best += 1
                if self.early_stop \
                        and time_since_best >= self.early_stop \
                        and best_rate >= solve_thresh:
                    LOGGER.info('Terminating (early stopping condition met with'
                                '%d epochs since loss %f)',
                                time_since_best, best_rate)
                    keep_going = False

            should_save = best_rate is None or total_succ_rate >= best_rate \
                or (self.save_every and iter_num % self.save_every == 0) \
                or iter_num == 1  # always save on first iter
            if should_save:
                best_rate = total_succ_rate
                # snapshot!
                # TODO: add snapshot pruning support so that old snapshots
                # can be deleted if desired
                snapshot_path = os.path.join(
                    self.snapshot_dir,
                    'snapshot_%d_%f.pkl' % (iter_num, total_succ_rate))
                self.weight_manager.save(snapshot_path)
                shutil.copy(snapshot_path, self.dk)
            # also, always save timing data
            with open(os.path.join(self.scratch_dir, 'timing.json'), 'w') as fp:
                fp.write(self.timer.to_json())

            tf.summary.flush()

            if self.timeout:
                keep_going = keep_going and elapsed_time <= self.timeout

            if not keep_going:
                LOGGER.info('Terminating early')
                break

        return best_rate, elapsed_time, iter_num

    @can_profile
    def _make_batches(self, n_batches: int):
        """A generator yielding batches of data for training.

        Args:
            n_batches: Number of batches to yield.

        Yields:
            A batch of data as a list, where each element is a batch of data for
            a single problem of the form (obs_tensor, qvs_tensor). The batches
            are order in the same order as the problems in self.problems.
        """
        batch_iters = []

        if self.save_training_set:
            to_save = {}
        cached_shapes = {}
        for problem in self.problems:
            service = problem.problem_service

            if self.use_saved_training_set:
                assert not self.save_training_set, \
                    "saving training set & using a saved set are mutually " \
                    "exclusive options (doesn't make sense to write same " \
                    "dataset back out to disk!)"
                prob_obs_tensor, prob_qv_tensor, prob_counts \
                    = self.loaded_training_set[problem.name]
                it = weighted_batch_iter(
                    (prob_obs_tensor, prob_qv_tensor),
                    prob_counts,
                    self.batch_size_per_problem,
                    n_batches,
                )
                batch_iters.append(it)
                continue

            if service.dataset_is_empty():
                LOGGER.warn("No data for problem '%s' yet (teacher time-out?)",
                            service.get_current_problem_name())
                batch_iters.append(repeat(None))
                if self.save_training_set:
                    to_save[problem.name] = None
            else:
                prob_obs_tensor, prob_qv_tensor, prob_counts \
                    = to_local(service.weighted_dataset())
                it = weighted_batch_iter(
                    (prob_obs_tensor, prob_qv_tensor),
                    prob_counts,
                    self.batch_size_per_problem,
                    n_batches,
                )
                batch_iters.append(it)
                if self.save_training_set:
                    to_save[problem.name] \
                        = (prob_obs_tensor, prob_qv_tensor, prob_counts)
            cached_shapes[problem.name] = (
                service.get_obs_dim(), service.get_act_dim())

        if self.save_training_set:
            LOGGER.info("Saving training set to disk'%s'",
                        self.save_training_set)
            dirname = os.path.dirname(self.save_training_set)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            joblib.dump(to_save, self.save_training_set)

        combined = zip(*batch_iters)

        # yield a complete feed dict
        for combined_batch in combined:
            assert len(combined_batch) == len(self.problems)
            yield_val = []
            have_batch = False
            for problem, batch in zip(self.problems, combined_batch):
                if batch is None:
                    yield_val.append(empty_feed_value(
                        *cached_shapes[problem.name]))
                else:
                    yield_val.append(batch)
                    have_batch = True
            assert have_batch, \
                "don't have any batches at all for training problems"
            yield yield_val

    def _get_replay_sizes(self):
        """Get the sizes of replay buffers for each problem."""
        rv = []
        for problem in self.problems:
            rv.append(to_local(problem.problem_service.get_replay_size()))
        return rv


class ManualLoss(tf.keras.losses.Loss):
    def __init__(self,
                 problems,
                 weight_manager,
                 summary_writer,
                 l1_reg_coeff,
                 l2_reg_coeff,
                 l1_l2_reg_coeff,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name=None,
                 strategy=SupervisedObjective.ANY_GOOD_ACTION):
        super().__init__(reduction, name)
        self.problems = problems
        self.weight_manager = weight_manager
        self.summary_writer = summary_writer
        self.l1_reg_coeff = l1_reg_coeff
        self.l2_reg_coeff = l2_reg_coeff
        self.l1_l2_reg_coeff = l1_l2_reg_coeff
        self.strategy = strategy

    def call(self, act_pred: List[tf.Tensor], q_values: List[tf.Tensor]) \
            -> float:
        assert len(self.problems) == len(act_pred), \
            "inconsistent input data size with num. problems"
        assert len(q_values) == len(act_pred), \
            "inconsistent output data sizes"
        losses = []
        batch_sizes = []
        loss_parts = None
        for i, problem in enumerate(self.problems):
            act_dist, ph_q_values = act_pred[i], q_values[i]
            this_loss, this_loss_parts = self._set_up_losses(
                problem, act_dist, ph_q_values)

            this_batch_size = tf.shape(input=act_dist)[0]
            losses.append(this_loss)
            batch_sizes.append(tf.cast(this_batch_size, tf.float32))
            if loss_parts is None:
                loss_parts = this_loss_parts
            else:
                # we care about these parts because we want to display them to
                # the user (e.g. how much of my loss is L2 regularisation
                # loss?)
                assert len(loss_parts) == len(this_loss_parts), \
                    'diff. loss breakdown for diff. probs. (%s vs %s)' \
                    % (loss_parts, this_loss_parts)
                # sum up all the parts
                new_loss_parts = []
                for old_part, new_part in zip(loss_parts, this_loss_parts):
                    assert old_part[0] == new_part[0], \
                        "names (%s vs. %s) don't match" % (old_part[0],
                                                           new_part[0])
                    to_add = new_part[1] * tf.cast(this_batch_size, tf.float32)
                    new_loss_parts.append((old_part[0], old_part[1] + to_add))
                loss_parts = new_loss_parts
        with tf.name_scope('combine_all_losses'):
            op_loss \
                = sum(l * s for l, s in zip(losses, batch_sizes)) \
                / sum(batch_sizes)

        # this is actually a list of (name, symbolic representation) pairs for
        # components of the loss
        assert loss_parts is not None

        for part_loss_name, part_loss in loss_parts:
            tf.summary.scalar('loss-%s' % part_loss_name, part_loss)

        return op_loss

    @can_profile
    def _set_up_losses(self, problem, act_dist, ph_q_values):
        problem_service = problem.problem_service
        loss_parts = []
        # now the loss ops
        with tf.name_scope('loss'):
            if self.strategy == SupervisedObjective.ANY_GOOD_ACTION \
                    or self.strategy == SupervisedObjective.THERE_CAN_ONLY_BE_ONE:
                best_qv = tf.reduce_min(
                    input_tensor=ph_q_values, axis=-1, keepdims=True)
                # TODO: is 0.01 threshold too big? Hmm.
                act_labels = tf.cast(
                    tf.less(tf.abs(ph_q_values - best_qv), 0.01), 'float32')
                label_sum = tf.reduce_sum(
                    input_tensor=act_labels, axis=-1, keepdims=True)
                act_label_dist = act_labels / tf.math.maximum(label_sum, 1.0)
                # zero out disabled or dead-end actions!
                dead_end_value = to_local(
                    problem_service.get_ssipp_dead_end_value())
                act_label_dist *= tf.cast(act_labels < dead_end_value,
                                          'float32')
                # this tf.cond() call ensures that this still works when batch
                # size is 0 (in which case it returns a loss of 0)
                xent = tf.cond(pred=tf.size(input=act_label_dist) > 0,
                               true_fn=lambda: tf.reduce_mean(
                    input_tensor=cross_entropy(act_dist, act_label_dist),
                    name='xent_reduce'),
                    false_fn=lambda: tf.constant(
                    0.0, dtype=tf.float32, name='xent_ph'),
                    name='xent_cond')
                loss_parts.append(('xent', xent))
            elif self.strategy == SupervisedObjective.MAX_ADVANTAGE:
                state_values = tf.reduce_min(input_tensor=ph_q_values, axis=-1)
                exp_q = act_dist * ph_q_values
                exp_vs = tf.reduce_sum(input_tensor=exp_q, axis=-1)
                # state value is irrelevant to objective, but is included
                # because it ensures that zero loss = optimal policy
                q_loss = tf.reduce_mean(input_tensor=exp_vs - state_values)
                loss_parts.append(('qloss', q_loss))
            else:
                raise ValueError("Unknown strategy %s" % self.strategy)

            # regularisation---we need this because the
            # logisitic-regression-like optimisation problem we're solving
            # generally has no minimum point otherwise
            weights = self.weight_manager.all_weights
            weights_no_bias = [w for w in weights if len(w.shape) > 1]
            weights_all_bias = [w for w in weights if len(w.shape) <= 1]
            # downweight regulariser penalty on biases (for most DL work
            # they're un-penalised, but here I think it pays to have *some*
            # penalty given that there are some problems that we can solve
            # perfectly)
            bias_coeff = 0.05
            if self.l2_reg_coeff:

                def do_l2_reg(lst):
                    return sum(map(tf.nn.l2_loss, lst))

                l2_reg = self.l2_reg_coeff * do_l2_reg(weights_no_bias) \
                    + bias_coeff * self.l2_reg_coeff \
                    * do_l2_reg(weights_all_bias)
                loss_parts.append(('l2reg', l2_reg))

            if self.l1_reg_coeff:

                def do_l1_reg(lst):
                    return sum(tf.linalg.norm(tensor=w, ord=1) for w in lst)

                l1_reg = self.l1_reg_coeff * do_l1_reg(weights_no_bias) \
                    + bias_coeff * self.l1_reg_coeff \
                    * do_l1_reg(weights_all_bias)
                loss_parts.append(('l1reg', l1_reg))

            if self.l1_l2_reg_coeff:
                all_weights_ap = []
                # act_weights[:-1] omits the last layer (which we don't want to
                # apply group sparsity penalty to)
                all_weights_ap.extend(self.weight_manager.act_weights[:-1])
                all_weights_ap.extend(self.weight_manager.prop_weights)
                l1_l2_reg_accum = 0.0
                for weight_dict in all_weights_ap:
                    for trans_mat, bias in weight_dict.values():
                        bias_size, = bias.shape.as_list()
                        tm_shape = trans_mat.shape.as_list()
                        # tm_shape[0] is always 1, tm_shape[1] is size of
                        # input, and tm_shape[2] is network channel count
                        assert len(tm_shape) == 3 and tm_shape[0] == 1 \
                            and tm_shape[2] == bias_size, "tm_shape %s does " \
                            "not match bias size %s" % (tm_shape, bias_size)
                        trans_square = tf.reduce_sum(
                            input_tensor=tf.square(trans_mat), axis=[0, 1])
                        bias_square = tf.square(bias)
                        norms = tf.sqrt(trans_square + bias_square)
                        l1_l2_reg_accum += tf.reduce_sum(input_tensor=norms)
                l1_l2_reg = self.l1_l2_reg_coeff * l1_l2_reg_accum
                loss_parts.append(('l1l2reg', l1_l2_reg))

            with tf.name_scope('combine_parts'):
                loss = sum(p[1] for p in loss_parts)

        return loss, loss_parts
