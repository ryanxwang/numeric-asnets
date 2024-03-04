"""Interface code for rllab. Mainly handles interaction with mdpsim & hard
things like action masking."""

import logging
import numpy as np
from typing import Dict, Iterable, List, Optional, Tuple
from typing_extensions import Self

from asnets.prob_dom_meta import BoundProp, BoundAction, BoundComp, BoundFlnt
from asnets.utils.jpddl_utils import flnt_to_jpddl_id, prop_to_jpddl_id
from asnets.utils.mdpsim_utils import SPECIAL_FUNCTIONS
from asnets.utils.py_utils import strip_parens

import jpype
import jpype.imports
from jpype.types import *

LOGGER = logging.getLogger(__name__)

J_BitSet = None
J_HashMap = None
J_PDDLState = None


@jpype.onJVMStart
def _import_java_classes() -> None:
    """Import Java classes that will be used by this module. This is called
    automatically upon JVM start-up.
    """
    global J_BitSet, J_HashMap, J_PDDLState

    J_BitSet = jpype.JPackage('java').util.BitSet
    J_HashMap = jpype.JPackage('java').util.HashMap
    J_PDDLState = jpype.JPackage('com').hstairs.ppmajal.problem.PDDLState


class CanonicalState(object):
    """The ASNet code uses a lot of state representations. There are
    pure-Python state representations, there are state representations based on
    the SSiPP & MDPsim wrappers, and there are string-based intermediate
    representations used to convert between the other representations. This
    class aims to be a single canonical state class that it is:

    1. Possible to convert to any other representation,
    2. Possible to instantiate from any other representation,
    3. Possible to pickle & send between processes.
    4. Efficient to manipulate.
    5. Relatively light on memory."""

    def __init__(self,
                 bound_prop_truth: Iterable[Tuple[BoundProp, bool]],
                 bound_flnt_value: Iterable[Tuple[BoundFlnt, float]],
                 bound_acts_enabled: Iterable[Tuple[BoundAction, bool]],
                 is_goal: bool,
                 *,
                 bound_comp_truth: Iterable[Tuple[BoundFlnt, bool]] = (),
                 data_gens: Iterable['ActionDataGenerator'] = None,
                 prev_cstate: Optional[Self] = None,
                 prev_act: Optional[BoundAction] = None,
                 is_init_cstate: Optional[bool] = None):
        # note: props and acts should always use the same order! I don't want
        # to be passing around extra data to store "real" order for
        # propositions and actions all the time :(
        # FIXME: replace props_true and acts_enabled with numpy ndarray masks
        # instead of inefficient list-of-tuples structure
        self.props_true = tuple(bound_prop_truth)
        self.flnt_values = tuple(bound_flnt_value)
        self.acts_enabled = tuple(bound_acts_enabled)
        self.comps_true = tuple(bound_comp_truth)
        self.is_goal = is_goal
        self.is_terminal = is_goal or not any(
            enabled for _, enabled in self.acts_enabled)
        self._aux_data = None
        self._aux_data_interp = None
        self._aux_data_interp_to_id = None
        if data_gens is not None:
            self.populate_aux_data(data_gens,
                                   prev_cstate=prev_cstate,
                                   prev_act=prev_act,
                                   is_init_cstate=is_init_cstate)
        # FIXME: make _do_validate conditional on a debug flag or something (I
        # suspect it's actually a bit expensive, so not turning on by default)
        # self._do_validate()

    def _do_validate(self):
        """Run some sanity checks on the newly-constructed state."""
        # first check proposition mask
        for prop_idx, prop_tup in enumerate(self.props_true):
            # should be tuple of (proposition, truth value)
            assert isinstance(prop_tup, tuple) and len(prop_tup) == 2
            assert isinstance(prop_tup[0], BoundProp)
            assert isinstance(prop_tup[1], bool)
            if prop_idx > 0:
                # should come after previous proposition alphabetically
                assert prop_tup[0].unique_ident \
                    > self.props_true[prop_idx - 1][0].unique_ident

        # check fluent values
        for flnt_idx, flnt_tup in enumerate(self.flnt_values):
            # should be tuple of (fluent, value)
            assert isinstance(flnt_tup, tuple) and len(flnt_tup) == 2
            assert isinstance(flnt_tup[0], BoundFlnt)
            assert isinstance(flnt_tup[1], float)
            if flnt_idx > 0:
                # should come after previous fluent alphabetically
                assert flnt_tup[0].unique_ident \
                    > self.flnt_values[flnt_idx - 1][0].unique_ident

        # check comparisons
        for comp_idx, comp_tup in enumerate(self.comps_true):
            assert isinstance(comp_tup, tuple) and len(comp_tup) == 2
            assert isinstance(comp_tup[0], BoundComp)
            assert isinstance(comp_tup[1], bool)
            if comp_idx > 0:
                assert comp_tup[0].unique_ident \
                    > self.comps_true[comp_idx - 1][0].unique_ident

        # next check action mask
        for act_idx, act_tup in enumerate(self.acts_enabled):
            # should be tuple of (action, enabled flag)
            assert isinstance(act_tup, tuple) and len(act_tup) == 2
            assert isinstance(act_tup[0], BoundAction)
            assert isinstance(act_tup[1], bool)
            if act_idx > 0:
                # should come after previous action alphabetically
                assert act_tup[0].unique_ident \
                    > self.acts_enabled[act_idx - 1][0].unique_ident

        # make sure that auxiliary data is 1D ndarray
        if self._aux_data is not None:
            assert isinstance(self._aux_data, np.ndarray), \
                "_aux_data is not ndarray (%r)" % type(self._aux_data)
            assert self._aux_data.ndim == 1

    def __repr__(self) -> str:
        """Return a string representation of the state.

        Returns:
            str: string representation of the state.
        """
        # Python-legible state
        return '%s(%r, %r, %r)' \
            % (self.__class__.__name__,
               self.props_true, self.flnt_values, self.comps_true,
               self.acts_enabled)

    def __str__(self) -> str:
        """Return a human-readable string representation of the state.

        Returns:
            str: human-readable string representation of the state.
        """
        prop_str = ', '.join(p.unique_ident for p, t in self.props_true if t)
        prop_str = prop_str or '-'

        flnt_str = ', '.join('{}: {}'.format(f.unique_ident, v)
                             for f, v in self.flnt_values)
        flnt_str = flnt_str or '-'

        extras = [
            'has aux_data' if self._aux_data is not None else 'no aux_data'
        ]
        if self.is_goal:
            extras.append('is goal')
        if self.is_terminal:
            extras.append('is terminal')
        state_str = 'State %s %s (%s)' % (
            prop_str, flnt_str, ', '.join(extras))
        return state_str

    def _ident_tup(self) \
            -> Tuple[Iterable[Tuple[BoundProp, bool]],
                     Iterable[Tuple[BoundFlnt, float]],
                     Iterable[Tuple[BoundComp, bool]],
                     Iterable[Tuple[BoundAction, bool]],
                     bool]:
        """Return a tuple that uniquely identifies this state.

        Returns:
            a tuple that uniquely identifies this state.
        """
        # This function is used to get a hashable representation for __hash__
        # and __eq__. Note that we don't hash _aux_data because it's an
        # ndarray; instead, hash bool indicating whether we have _aux_data.
        # Later on, we WILL compare on _aux_data in the __eq__ method.
        # (probably it's a bad idea not to include that in the hash, but
        # whatever)
        return (self.props_true, self.flnt_values, self.comps_true,
                self.acts_enabled, self._aux_data is None)

    def __hash__(self) -> int:
        """Return a hash of the state.

        Returns:
            int: hash of the state.
        """
        return hash(self._ident_tup())

    def __eq__(self, other: Self) -> bool:
        """Return whether this state is equal to another state.

        Args:
            other (Self): state to compare against.

        Raises:
            TypeError: if other is not a CanonicalState.

        Returns:
            bool: whether this state is equal to other.
        """
        if not isinstance(other, CanonicalState):
            raise TypeError(
                "Can't compare self (type %s) against other object (type %s)" %
                (type(self), type(other)))
        eq_basic = self._ident_tup() == other._ident_tup()
        if self._aux_data is not None and eq_basic:
            # equality depends on _aux_data being similar/identical
            return np.allclose(self._aux_data, other._aux_data)
        return eq_basic

    ##################################################################
    # Functions for dealing with ActionDataGenerators
    ##################################################################

    @property
    def aux_data(self) -> np.ndarray:
        """Get auxiliary data produced by data generators.

        Returns:
            np.ndarray: auxiliary data produced by data generators.
        """
        if self._aux_data is None:
            raise ValueError("Must run .populate_aux_data() on state before "
                             "using .aux_data attribute.")
        return self._aux_data

    def populate_aux_data(self,
                          data_gens: Iterable['ActionDataGenerator'],
                          *,
                          prev_cstate: Optional[Self] = None,
                          prev_act: Optional[BoundAction] = None,
                          is_init_cstate: Optional[bool] = None) -> None:
        """Populate class with auxiliary data from data generators.

        Args:
            data_gens (Iterable[ActionDataGenerator]): data generators to use.
            prev_cstate (Optional[CanonicalState], optional): previous state.
            prev_act (Optional[BoundAction], optional): previous action.
            is_init_cstate (Optional[bool], optional): whether this is the
            initial state.

        Returns:
            None: None.
        """
        extra_data = []
        interp = []
        requires_memory = False
        for dg in data_gens:
            dg_data = dg.get_extra_data(self,
                                        prev_cstate=prev_cstate,
                                        prev_act=prev_act,
                                        is_init_cstate=is_init_cstate)
            extra_data.append(dg_data)
            interp.extend(dg.dim_names)
            requires_memory |= dg.requires_memory
        if len(extra_data) == 0:
            num_acts = len(self.acts_enabled)
            self._aux_data = np.zeros((num_acts, ), dtype='float32')
        else:
            self._aux_data = np.concatenate(
                extra_data, axis=1).astype('float32').flatten()
        self._aux_dat_interp = interp
        if requires_memory:
            # one of the memory-based DataGenerators (ActionCountDataGenerator)
            # needs to know what slots the dims map onto.
            self._aux_data_interp_to_id = {
                dim_name: idx for idx, dim_name in enumerate(interp)
            }

    ##################################################################
    # MDPSim interop routines
    ##################################################################

    @classmethod
    def from_mdpsim(cls,
                    mdpsim_state,
                    planner_exts,
                    *,
                    prev_cstate=None,
                    prev_act=None,
                    is_init_cstate=None):
        # general strategy: convert both props & actions to string repr, then
        # use those reprs to look up equivalent BoundProposition/BoundAction
        # representation from problem_meta
        data_gens = planner_exts.data_gens
        problem_meta = planner_exts.problem_meta

        mdpsim_props_true \
            = planner_exts.mdpsim_problem.prop_truth_mask(mdpsim_state)
        truth_val_by_name = {
            # <mdpsim_prop>.identifier includes parens around it, which we want
            # to strip
            strip_parens(mp.identifier): truth_value
            for mp, truth_value in mdpsim_props_true
        }
        # now build mask from actual BoundPropositions in right order
        prop_mask = [(bp, truth_val_by_name[bp.unique_ident])
                     for bp in problem_meta.bound_props_ordered]

        mdpsim_flnt_value \
            = planner_exts.mdpsim_problem.fluent_value_mask(mdpsim_state)
        flnt_val_by_name = {
            strip_parens(mf.identifier): v
            for mf, v in mdpsim_flnt_value
        }
        flnt_mask = [(bf, flnt_val_by_name[bf.unique_ident])
                     for bf in problem_meta.bound_flnts_ordered]

        mdpsim_comp_true \
            = planner_exts.mdpsim_problem.comp_truth_mask(mdpsim_state)
        truth_val_by_name = {
            str(mc): truth_value
            for mc, truth_value in mdpsim_comp_true
        }
        # now build mask from actual BoundComp in right order
        comp_mask = [(bc, truth_val_by_name[bc.comparison])
                     for bc in problem_meta.bound_comps_ordered]

        # similar stuff for action selection
        mdpsim_acts_enabled \
            = planner_exts.mdpsim_problem.act_applicable_mask(mdpsim_state)
        act_on_by_name = {
            strip_parens(ma.identifier): enabled
            for ma, enabled in mdpsim_acts_enabled
        }
        act_mask = [(ba, act_on_by_name[ba.unique_ident])
                    for ba in problem_meta.bound_acts_ordered]

        is_goal = mdpsim_state.goal()

        return cls(prop_mask,
                   flnt_mask,
                   act_mask,
                   is_goal,
                   bound_comp_truth=comp_mask,
                   data_gens=data_gens,
                   prev_cstate=prev_cstate,
                   prev_act=prev_act,
                   is_init_cstate=is_init_cstate)

    def _to_prop_string(self):
        # convert this state to a SSiPP-style state string
        prop_string = ', '.join(bp.unique_ident
                                for bp, is_true in self.props_true if is_true)
        # XXX: remove this check once method tested
        assert ')' not in prop_string and '(' not in prop_string
        return prop_string

    def _to_flnt_string(self):
        # convert this state to a fluent string, something like
        # "x agent1: 3, y agent1: 6"
        flnt_string = ', '.join('{}: {}'.format(bf.unique_ident, v)
                                for bf, v in self.flnt_values)
        assert ')' not in flnt_string and '(' not in flnt_string
        return flnt_string

    def to_mdpsim(self, planner_exts):
        # yes, for some reason I originally made MDPSim take SSiPP-style
        # strings in this *one* place
        prop_string = self._to_prop_string()
        flnt_string = self._to_flnt_string()
        problem = planner_exts.mdpsim_problem
        mdpsim_state = problem.intermediate_state(prop_string, flnt_string)
        return mdpsim_state

    ##################################################################
    # SSiPP interop routines
    ##################################################################

    @classmethod
    def from_ssipp(cls,
                   ssipp_state,
                   planner_exts,
                   *,
                   prev_cstate=None,
                   prev_act=None,
                   is_init_cstate=None):
        LOGGER.error("from_ssipp() is deprecated as ssipp does not support "
                     "fluents")
        problem = planner_exts.ssipp_problem
        problem_meta = planner_exts.problem_meta
        data_gens = planner_exts.data_gens
        ssipp_string = problem.string_repr(ssipp_state)

        # I made the (poor) decision of having string_repr return a string of
        # the form "(foo bar baz) (spam ham)" rather than "foo bar baz, spam
        # ham", so I need to strip_parens() here too (still had the foresight
        # to make string_repr return statics though---great!)
        true_prop_names = {
            p
            for p in ssipp_string.strip('()').split(') (') if p
        }

        # sanity check our hacky parse job
        bp_name_set = set(x.unique_ident
                          for x in problem_meta.bound_props_ordered)
        assert set(true_prop_names) <= bp_name_set
        prop_mask = [(bp, bp.unique_ident in true_prop_names)
                     for bp in problem_meta.bound_props_ordered]
        assert len(true_prop_names) == sum(on for _, on in prop_mask)

        # actions are a little harder b/c of ABSTRACTIONS!
        ssp = planner_exts.ssipp_ssp_iface
        ssipp_on_acts = ssp.applicableActions(ssipp_state)
        # yup, need to strip parens from actions too
        ssipp_on_act_names = {strip_parens(a.name()) for a in ssipp_on_acts}
        # FIXME: this is actually a hotspot in problems with a moderate number
        # of actions (say 5k+) if we need to create and delete a lot of states.
        act_mask = [(ba, ba.unique_ident in ssipp_on_act_names)
                    for ba in problem_meta.bound_acts_ordered]
        assert len(ssipp_on_act_names) \
            == sum(enabled for _, enabled in act_mask)

        # finally get goal flag
        is_goal = ssp.isGoal(ssipp_state)

        return cls(prop_mask,
                   [],
                   act_mask,
                   is_goal,
                   data_gens=data_gens,
                   prev_cstate=prev_cstate,
                   prev_act=prev_act,
                   is_init_cstate=is_init_cstate)

    def to_ssipp(self, planner_exts):
        # even though SSiPP doesn't support fluents, we still need to convert
        # to ssipp when using the LM-Cut heuristic with numeric relaxation
        ssipp_string = self._to_prop_string()
        problem = planner_exts.ssipp_problem
        ssipp_state = problem.get_intermediate_state(ssipp_string)
        return ssipp_state

    ##################################################################
    # JPDDL conversion routines
    ##################################################################

    def to_jpddl(self):
        num_fluents = J_HashMap()
        for bf, v in self.flnt_values:
            if bf.func_name in SPECIAL_FUNCTIONS:
                continue
            num_fluents.put(JInt(flnt_to_jpddl_id(bf)), JDouble(v))

        bool_fluents = J_BitSet()
        for bp in (bp for bp, truth in self.props_true if truth):
            bool_fluents.set(prop_to_jpddl_id(bp), True)

        return J_PDDLState(num_fluents, bool_fluents)

    ##################################################################
    # Other conversion routines
    ##################################################################

    # Shorthand for the tuple type returned by to_tup_state()
    TupState = Tuple[Tuple[str], Tuple[Tuple[str, float]]]

    def to_tup_state(self) -> Tuple[Tuple[str], Tuple[Tuple[str, float]]]:
        """Return a tuple that can be easily converted to an initial state in
        a pddl file.

        Returns:
            Tuple[Tuple[str], Tuple[Tuple[str, float]]]: Tuple of tuples of
            strings and floats. The first tuple is true atoms in this state, the
            second is the fluents and their values in this state.
        """
        atom_tup = tuple(
            bp.unique_ident
            for bp, truth in self.props_true if truth)
        flnt_tup = tuple(
            (bf.unique_ident, v)
            for bf, v in self.flnt_values
            if bf.func_name not in SPECIAL_FUNCTIONS)
        return (atom_tup, flnt_tup)

    ##################################################################
    # Network input routines (prepares flat vector to give to ASNet)
    ##################################################################

    use_fluents = False
    use_comparisons = False

    @classmethod
    def network_input_config(cls,
                             use_fluents: bool = False,
                             use_comparisons: bool = False):
        cls.use_fluents = use_fluents
        cls.use_comparisons = use_comparisons

    def to_network_input(self) -> np.ndarray:
        to_concat = []

        # will be 1 for enabled actions, 0 for disabled actions
        act_mask_conv = np.array([enabled for _, enabled in self.acts_enabled],
                                 dtype='float32')
        to_concat.append(act_mask_conv)

        # need this first, should be populated with populate_aux_data before
        # we get here
        to_concat.append(self.aux_data)

        # will be 1 for true props, 0 for false props
        props_conv = np.array([truth for _, truth in self.props_true],
                              dtype='float32')
        to_concat.append(props_conv)

        # value for each fluent
        if CanonicalState.use_fluents:
            flnts_conv = np.array([value for _, value in self.flnt_values],
                                  dtype='float32')
            to_concat.append(flnts_conv)

        if CanonicalState.use_comparisons:
            comps_conv = np.array([truth for _, truth in self.comps_true],
                                  dtype='float32')
            to_concat.append(comps_conv)

        return np.concatenate(to_concat)


def get_init_cstate(planner_exts):
    mdpsim_init = planner_exts.mdpsim_problem.init_state()
    cstate_init = CanonicalState.from_mdpsim(mdpsim_init,
                                             planner_exts,
                                             prev_cstate=None,
                                             prev_act=None,
                                             is_init_cstate=True)
    return cstate_init


def sample_next_state(
        cstate: CanonicalState,
        action_id: int,
        planner_exts: 'PlannerExtensions',
        *,
        ignore_disabled: bool = True):
    """Sample the next state given a canonical state and an action.

    Args:
        cstate (CanonicalState): The current canonical state.
        action_id (int): The action to take, as index in the acts_enabled list.
        planner_exts (PlannerExtensions): The planner extensions.
        ignore_disabled (bool, optional): Whether to ignore disabled actions
        (just return the same state) or raise an exception. Defaults to True.

    Raises:
        ValueError: If ignore_disabled is False and the action is disabled.

    Returns:
        (CanonicalState, int): The next canonical state and the step cost. For
        now step cost is always 1.
    """
    assert isinstance(action_id, int), \
        "Action must be integer, but is %s" % type(action_id)
    assert isinstance(cstate, CanonicalState)

    # TODO: instead of passing in action_id, pass in a BoundAction
    mdpsim_state = cstate.to_mdpsim(planner_exts)
    bound_act, applicable = cstate.acts_enabled[action_id]
    if not applicable:
        tot_enabled = sum(truth for _, truth in cstate.acts_enabled)
        tot_avail = len(cstate.acts_enabled)

        error_str = 'Selected disabled action #{}, {}; {}/{} available'.format(
            action_id, bound_act, tot_enabled, tot_avail)
        LOGGER.warn(error_str)
        if not ignore_disabled:
            raise ValueError(error_str)

        new_cstate = cstate
    else:
        act_ident = bound_act.unique_ident
        mdpsim_action = planner_exts.act_ident_to_mdpsim_act[act_ident]
        new_mdpsim_state = planner_exts.mdpsim_problem.apply(
            mdpsim_state, mdpsim_action)
        new_cstate = CanonicalState.from_mdpsim(new_mdpsim_state,
                                                planner_exts,
                                                prev_cstate=cstate,
                                                prev_act=bound_act,
                                                is_init_cstate=False)

    # XXX: I don't currently calculate the real cost; must sort that
    # out!
    step_cost = 1

    return new_cstate, step_cost


def successors(cstate: CanonicalState, action_id: int,
               planner_exts: 'PlannerExtensions') \
        -> List[Tuple[float, CanonicalState]]:
    """Returns list of (probability, successor state) tuples. Does not work with
    fluents.

    Raises:
        ValueError: if action is not enabled.

    Returns:
        List[Tuple[float, CanonicalState]]: list of (probability, successor
        state) tuples.
    """
    bound_act, applicable = cstate.acts_enabled[action_id]
    if not applicable:
        raise ValueError("Action #%d is not enabled (action: %s)" %
                         (action_id, bound_act))
    ssipp_state = cstate.to_ssipp(planner_exts)
    act_ident = bound_act.unique_ident
    ssipp_action = planner_exts.ssipp_problem.find_action("(%s)" % act_ident)
    cost = ssipp_action.cost(ssipp_state)
    assert cost == 1, \
        "I don't think rest of the code can deal with cost of %s" % (cost, )
    # gives us a list of (probability, ssipp successor state) tuples
    ssipp_successors = planner_exts.ssipp.successors(
        planner_exts.ssipp_ssp_iface, ssipp_state, ssipp_action)
    canon_successors = [(p,
                         CanonicalState.from_ssipp(s,
                                                   planner_exts,
                                                   prev_cstate=cstate,
                                                   prev_act=bound_act,
                                                   is_init_cstate=False))
                        for p, s in ssipp_successors]
    return canon_successors


def get_action_name(planner_exts: 'PlannerExtensions', action_id: int) \
        -> Optional[str]:
    """Returns the name of the action with the given ID.

    Returns:
        Optional[str]: The name of the action, or None if the action ID is
        invalid.
    """
    acts_ordered = planner_exts.problem_meta.bound_acts_ordered
    if 0 <= action_id < len(acts_ordered):
        bound_act = acts_ordered[action_id]
        return bound_act.unique_ident
    return None  # explicit return b/c silent failure is intentional


def compute_observation_dim(planner_exts: 'PlannerExtensions') -> int:
    """Returns the dimension of the observation space. Considers all the data
    generators and the number of propositions/fluents in the problem.

    Args:
        planner_exts (PlannerExtensions): The planner extension object.

    Returns:
        int: The dimension of the observation space.
    """
    extra_dims = sum(dg.extra_dim for dg in planner_exts.data_gens)
    nprops = planner_exts.mdpsim_problem.num_props
    nflnts = planner_exts.mdpsim_problem.num_fluents
    nacts = planner_exts.mdpsim_problem.num_actions
    return nprops + nflnts + (1 + extra_dims) * nacts


def compute_action_dim(planner_exts: 'PlannerExtensions') -> int:
    """Returns the number of actions in the problem.

    Args:
        planner_exts (PlannerExtensions): The planner extension object.

    Returns:
        int: The number of actions in the problem.
    """
    return planner_exts.mdpsim_problem.num_actions


def simulate_plan(init_cstate: CanonicalState,
                  plan_strs: List[str],
                  planner_exts: 'PlannerExtensions') \
        -> Tuple[List[CanonicalState], List[float]]:
    """Simulate a plan to obtain a sequence of states. Will include all states
    visited by the plan in the order the are encountered, including initial
    state and goal state. Only works for deterministic problems, obviously!

    Args:
        init_cstate (CanonicalState): Initial state.
        plan_strs (List[str]): List of action names, in order.
        planner_exts (PlannerExtensions): Extensions to the planner.

    Returns:
        Tuple[List[CanonicalState], List[float]]: List of states visited by
        plan and list of costs incurred by each action. There is exactly one
        more state than action, since the initial state is included.
    """
    cstates = [init_cstate]
    costs = []
    for action_str in plan_strs:
        this_state = cstates[-1]
        assert not this_state.is_terminal
        action_id = planner_exts.problem_meta.act_unique_id_to_index(
            action_str)
        next_state, cost = sample_next_state(
            cstates[-1],
            action_id,
            planner_exts,
            ignore_disabled=False)
        costs.append(cost)
        cstates.append(next_state)

    assert cstates[-1].is_terminal

    return cstates, costs


def simulate_policy(cstate: CanonicalState,
                    policy: Dict[CanonicalState.TupState, str],
                    planner_exts: 'PlannerExtensions') -> List[CanonicalState]:
    """Simulate a policy to obtain a sequence of states. Will include all
    states visited on the way to the goal, except the goal.

    Args:
        init_cstate (CanonicalState): Initial state.
        policy (Dict[CanonicalState.TupState, str]): Policy to simulate. Should
        be a dictionary from CanonicalState.TupState to action names (without
        parenthesis).
        planner_exts (PlannerExtensions): Extensions to the planner.

    Returns:
        List[CanonicalState]: List of states visited by policy, not including
        the goal state.
    """
    visited = []
    while not cstate.is_terminal:
        visited.append(cstate)
        action_id = planner_exts.problem_meta.act_unique_id_to_index(
            policy[cstate.to_tup_state()])
        cstate, _ = sample_next_state(cstate, action_id, planner_exts)

        assert len(visited) < 10000, "Infinite loop?"

    return visited
