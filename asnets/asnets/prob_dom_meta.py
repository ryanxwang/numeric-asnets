"""Stores metadata for problems and domains in a pure-Python format. This
information is theoretically all obtainable from MDPSim extension. However,
I've introduced these extra abstractions so that I can pickle that information
and pass it between processes. C++ extension data structures (including those
from the MDPSim extension) can't be easily pickled, so passing around
information taken *straight* from the extension would not work."""

from enum import auto, Enum
from functools import lru_cache, total_ordering
import re
from typing import Any, Callable, Dict, Iterable, List, Set, Tuple, Union
from typing_extensions import Self

from asnets.utils.mdpsim_utils import SPECIAL_FUNCTIONS


class DomainType(Enum):
    """The type of a domain."""
    DETERMINISTIC = auto()
    PROBABILISTIC = auto()
    NUMERIC = auto()

    # Hacky solution for argparse comptaibility.
    # See https://stackoverflow.com/questions/43968006/support-for-enum-arguments-in-argparse
    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return DomainType[s.upper()]
        except KeyError:
            return s


@total_ordering
class BoundFlnt:
    """Represents a ground fluent."""

    def __init__(self, func_name: str, arguments: Iterable[str]):
        """Create a new fluent.

        Args:
            func_name (str): Name of the function.
            arguments (Iterable[str]): The arguments of the function.
        """
        self.func_name = func_name
        self.arguments = tuple(arguments)
        self.unique_ident = self._compute_unique_ident()

        assert isinstance(self.func_name, str)

    def __repr__(self) -> str:
        """Return a string representation of this fluent.

        Returns:
            str: The string representation.
        """
        return 'BoundFlnt(%r, %r)' % (self.func_name, self.arguments)

    def _compute_unique_ident(self) -> str:
        """Unique identifier for this fluent. Will match SSiPP-style names
        instead of sexpr-style. Think "foo bar baz" rather than "(foo bar baz)".

        Returns:
            str: The unique identifier.
        """
        unique_id = ' '.join((self.func_name, ) + self.arguments)
        return unique_id

    def __eq__(self, other: Self) -> bool:
        """Check if this fluent is equal to another.

        Args:
            other (Self): The other fluent.

        Returns:
            bool: True if this fluent is equal to the other, False otherwise.
        """
        if not isinstance(other, BoundFlnt):
            return NotImplemented
        return self.unique_ident == other.unique_ident

    def __lt__(self, other: Self) -> bool:
        """Check if this fluent is less than another.

        Args:
            other (Self): The other fluent.

        Returns:
            bool: True if this fluent is less than the other, False otherwise.
        """
        if not isinstance(other, BoundFlnt):
            return NotImplemented
        return self.unique_ident < other.unique_ident

    def __hash__(self) -> int:
        """Return the hash of this fluent.

        Returns:
            int: The hash.
        """
        return hash(self.unique_ident)


class UnboundFlnt:
    def __init__(self, func_name: str, params: Iterable[str]):
        """Create a new unbound fluent (function).

        Args:
            func_name (str): Name of the function.
            params (Iterable[str]): The parameters of the function.
        """
        self.func_name = func_name
        self.params = tuple(params)

        assert isinstance(self.func_name, str)
        assert all(isinstance(p, str) for p in self.params)

    def __repr__(self) -> str:
        """Return a string representation of this unbound function.

        Returns:
            str: The string representation.
        """
        return 'UnboundFunc(%r, %r)' % (self.func_name, self.params)

    def bind(self, bindings: Dict[str, str]) -> BoundFlnt:
        """Bind this unbound function to a set of bindings.

        Args:
            bindings (Dict[str, str]): The bindings.

        Raises:
            ValueError: If there is no binding for a parameter.

        Returns:
            BoundFlnt: The bound fluent.
        """
        assert isinstance(bindings, dict), "Bindings must be a dict."
        args = []
        for param_name in self.params:
            if param_name[0] != '?':
                # already bound to a constant
                arg = param_name
            else:
                if param_name not in bindings:
                    raise ValueError(
                        "No binding for parameter %r" % param_name)
                arg = bindings[param_name]
            args.append(arg)
        return BoundFlnt(self.func_name, args)

    def __eq__(self, other: Self) -> bool:
        """Check if this unbound function is equal to another.

        Args:
            other (Self): The other unbound function.

        Returns:
            bool: True if this unbound function is equal to the other, False
            otherwise.
        """
        if not isinstance(other, UnboundFlnt):
            return NotImplemented
        return self.func_name == other.func_name and self.params == other.params

    def __hash__(self) -> int:
        """Return a hash of this unbound function.

        Returns:
            int: The hash.
        """
        return hash(self.func_name) ^ hash(self.params)


@total_ordering
class BoundProp:
    """Represents a ground proposition."""

    def __init__(self, pred_name: str, arguments: Iterable[str]):
        """Create a new bound proposition.

        Args:
            pred_name (str): The name of the predicate.
            arguments (Iterable[str]): The arguments of the predicate.
        """
        self.pred_name = pred_name
        self.arguments = tuple(arguments)
        self.unique_ident = self._compute_unique_ident()

        assert isinstance(self.pred_name, str)
        assert all(isinstance(p, str) for p in self.arguments)

    def __repr__(self) -> str:
        """Return a string representation of this proposition.

        Returns:
            str: The string representation.
        """
        return 'BoundProp(%r, %r)' % (self.pred_name, self.arguments)

    def _compute_unique_ident(self) -> str:
        """Compute a unique identifier for this proposition. Will match SSiPP-
        style names (think "foo bar baz" rather than sexpr-style "(foo bar
        baz)"). This is used for hashing and comparisons.

        Returns:
            str: The unique identifier.
        """
        unique_id = ' '.join((self.pred_name, ) + self.arguments)
        return unique_id

    def __eq__(self, other: Self) -> bool:
        """Check if this proposition is equal to another.

        Args:
            other (Self): The other proposition.

        Returns:
            bool: True if this proposition is equal to the other, False
            otherwise.
        """
        if not isinstance(other, BoundProp):
            return NotImplemented
        return self.unique_ident == other.unique_ident

    def __lt__(self, other: Self) -> bool:
        """Check if this proposition is less than another.

        Args:
            other (Self): The other proposition.

        Returns:
            bool: True if this proposition is less than the other, False
            otherwise.
        """
        if not isinstance(other, BoundProp):
            return NotImplemented
        return self.unique_ident < other.unique_ident

    def __hash__(self) -> int:
        """Return a hash of this proposition.

        Returns:
            int: The hash.
        """
        return hash(self.unique_ident)


class UnboundProp:
    """Represents a proposition which may have free parameters (e.g. as it will
    in an action). .bind() will ground it."""

    def __init__(self, pred_name: str, params: Iterable[str]):
        """Create a new unbound proposition.

        Args:
            pred_name (str): The name of the predicate.
            params (Iterable[str]): The parameters of the predicate.
        """
        # TODO: what if some parameters are already bound? This might happen
        # when you have constants, for instance. Maybe cross that bridge when I
        # get to it.
        self.pred_name = pred_name
        self.params = tuple(params)

        assert isinstance(self.pred_name, str)
        assert all(isinstance(p, str) for p in self.params)

    def __repr__(self):
        return 'UnboundProp(%r, %r)' % (self.pred_name, self.params)

    def bind(self, bindings: Dict[str, str]) -> BoundProp:
        """Bind this proposition to a set of bindings.

        Args:
            bindings (Dict[str, str]): A mapping from parameter names to
            their values.

        Raises:
            ValueError: If a parameter is not bound.

        Returns:
            BoundProp: The bound proposition.
        """
        assert isinstance(bindings, dict), \
            "expected dict of named bindings"
        args = []
        for param_name in self.params:
            if param_name[0] != '?':
                # already bound to constant
                arg = param_name
            else:
                if param_name not in bindings:
                    raise ValueError(
                        "needed bind for parameter %s, didn't get one" %
                        param_name)
                arg = bindings[param_name]
            args.append(arg)
        return BoundProp(self.pred_name, args)

    def __eq__(self, other: Self) -> bool:
        """Check if this proposition is equal to another.

        Args:
            other (Self): The other proposition.

        Returns:
            bool: True if the propositions are equal, False otherwise.
        """
        if not isinstance(other, UnboundProp):
            return NotImplemented
        return self.pred_name == other.pred_name \
            and self.params == other.params

    def __hash__(self) -> int:
        """Return a hash of this proposition.

        Returns:
            int: a hash of this proposition.
        """
        return hash(self.pred_name) ^ hash(self.params)


@total_ordering
class BoundComp:
    """Represents a bound comparison generated from a precondition in an action,
    does not include comparisons from the goal."""

    def __init__(self, comparison: str, schema: 'UnboundComp'):
        """Create a new BoundComp.

        Args:
            comparison (str): The comparison string.
            schema (UnboundComp): The unbound comparison.
        """
        self.comparison = comparison
        self.schema = schema
        self.unique_ident = self._compute_unique_ident()

        assert isinstance(comparison, str)

    def __repr__(self) -> str:
        return f'BoundComp({self.comparison})'

    def _compute_unique_ident(self) -> str:
        """Compute the unique identifier for this comparison.

        Returns:
            str: the unique identifier for this comparison.
        """
        return f'{self.comparison} {self.schema}'

    def __eq__(self, other: Self) -> bool:
        """Return whether two BoundComp instances are equal.

        Args:
            other (Self): the comparison to compare to.

        Returns:
            bool: whether self is equal to other.
        """
        if not isinstance(other, BoundComp):
            return NotImplemented
        return self.unique_ident == other.unique_ident

    def __lt__(self, other: Self) -> bool:
        """Return whether this comparison is less than another comparison.

        Args:
            other (Self): the other comparison to compare to.

        Returns:
            bool: whether this comparison is less than the other comparison.
        """
        if not isinstance(other, BoundComp):
            return NotImplemented
        return self.unique_ident < other.unique_ident

    def __hash__(self) -> int:
        """Return the hash of the comparison.

        Returns:
            int: the hash of the comparison.
        """
        return hash(self.unique_ident)


@total_ordering
class UnboundComp:
    """Represents an unbound comparison generated from a precondition in an 
    action schema.
    """

    def __init__(self, comparison: str, param_names: Iterable[str]):
        self.comparison, param_count = \
            UnboundComp.dedup_mdpsim_comp_schema(comparison)
        self.param_names = tuple(param_names)

        assert isinstance(comparison, str)
        assert '{}' not in comparison  # bad things happen if it started with {}
        assert param_count == len(param_names)
        assert all(isinstance(a, str) for a in self.param_names)

    def __repr__(self) -> str:
        return f'UnboundComp({self.comparison}, {self.param_names})'

    def _ident_tup(self) -> Tuple[str, Tuple[str]]:
        """Return a tuple that uniquely identifies this comparison.

        Returns:
            Tuple[str, Tuple[str]]: a tuple that uniquely identifies this 
            comparison.
        """
        return (self.comparison, self.param_names)

    def __eq__(self, other: Self) -> bool:
        """Return whether two UnboundComp instances are equal.

        Args:
            other (Self): the comparison to compare to.

        Returns:
            bool: whether self is equal to other.
        """
        if not isinstance(other, UnboundComp):
            return NotImplemented
        return self._ident_tup() == other._ident_tup()

    def __hash__(self) -> int:
        """Return the hash of the comparison.

        Returns:
            int: the hash of the comparison.
        """
        return hash(self._ident_tup())

    def __lt__(self, other: Self) -> bool:
        """Return whether this comparison is less than another comparison.

        Args:
            other (Self): the other comparison to compare to.

        Returns:
            bool: whether this comparison is less than the other comparison.
        """
        if not isinstance(other, UnboundComp):
            return NotImplemented
        return (self.comparison, self.param_names) < \
            (other.comparison, other.param_names)

    def bind(self, bindings: Dict[str, str],
             static_flnt_values: Dict[BoundFlnt, float]) -> BoundComp:
        """Bind this comparison to a set of bindings.

        Args:
            bindings (Dict[str, str]): A mapping from parameter names to their
            values.
            static_flnt_values (Dict[BoundFlnt, float]): A mapping from
            static fluents to their values.

        Raises:
            ValueError: If a parameter is not bound.

        Returns:
            BoundComp: The bound comparison.
        """
        assert isinstance(bindings, dict), \
            "expected dict of named bindings"
        args = []
        for param_name in self.param_names:
            if param_name[0] == '?':
                if param_name not in bindings:
                    raise ValueError(
                        "needed bind for parameter %s, didn't get one" %
                        param_name)
                args.append(bindings[param_name])

        grounded = self.comparison.format(*args)
        for static_flnt in static_flnt_values.keys():
            if f'({static_flnt.unique_ident})' in grounded:
                # use the .6g format specifier to try and print numbers the same
                # way c++ prints doubles under default settings, not gonna be
                # exact but should work for most cases
                grounded = grounded.replace(
                    f'({static_flnt.unique_ident})',
                    f'{static_flnt_values[static_flnt]:.6g}')

        return BoundComp(grounded, self)

    @classmethod
    def dedup_mdpsim_comp_schema(cls, schema: str) -> Tuple[str, int]:
        """Deduplicate a comparison schema extracted from mdpsim.

        The comparison shemas extracted directly from mdpsim have  variables
        represented as strings of form `?v16`. The same unbound comparison
        can appear in multiple action schemas, with the only difference being
        the number in the variable representation. To deal with this, we
        substitute the variable name with a '{}' placeholder, so that it can
        also be used as a format string.

        Args:
            schema (str): the comparison schema to deduplicate.

        Returns:
            Tuple[str, int]: the deduplicated comparison schema, and the number
            of substituted variables.
        """
        return re.subn('\?v\d+', '{}', schema)


@total_ordering
class BoundAction:
    """Represents a ground action."""

    def __init__(self,
                 prototype: 'UnboundAction',
                 arguments: Iterable[str],
                 props: Iterable[BoundProp],
                 flnts: Iterable[BoundFlnt],
                 comps: Iterable[BoundComp]):
        """Create a new ground action.

        Args:
            prototype (UnboundAction): The unbound action that this is a
            ground instance of.
            arguments (Iterable[str]): The arguments to this action.
            props (Iterable[BoundProp]): The ground propositions that are
            relevant to this action.
            flnts (Iterable[BoundFlnt]): The ground fluents that are relevant
            to this action.
            comps (Iterable[BoundComp]): The ground comparisons that are
        """
        self.prototype = prototype
        self.arguments = tuple(arguments)
        self.props = tuple(props)
        self.flnts = tuple(flnts)
        self.comps = tuple(comps)
        self.unique_ident = self._compute_unique_ident()

        assert isinstance(prototype, UnboundAction)
        assert all(isinstance(a, str) for a in self.arguments)
        assert all(isinstance(p, BoundProp) for p in self.props)
        assert all(isinstance(f, BoundFlnt) for f in self.flnts)
        assert all(isinstance(c, BoundComp) for c in self.comps)

    def __repr__(self) -> bool:
        """Return a string representation of this action.

        Returns:
            bool: a string representation of this action.
        """
        return 'BoundAction(%r, %r, %r, %r, %r)' \
            % (self.prototype, self.arguments, self.props, self.flnts,
               self.comps)

    def __str__(self) -> str:
        """Return a string representation of this action.

        Returns:
            str: a string representation of this action.
        """
        return 'Action %s(%s)' % (self.prototype.schema_name, ', '.join(
            self.arguments))

    def _compute_unique_ident(self) -> str:
        """Compute a unique identifier for this action.

        Returns:
            str: a unique identifier for this action.
        """
        unique_id = ' '.join((self.prototype.schema_name, ) + self.arguments)
        return unique_id

    def __eq__(self, other: Self) -> bool:
        """Compare this action to another action.

        Args:
            other (Self): the other action to compare to.

        Returns:
            bool: True if this action is equal to the other action, False
            otherwise.
        """
        if not isinstance(other, BoundAction):
            return NotImplemented
        return self.unique_ident == other.unique_ident

    def __lt__(self, other: Self) -> bool:
        """Compare this action to another action.

        Args:
            other (Self): the other action to compare to.

        Returns:
            bool: True if this action is less than the other action, False
            otherwise.
        """
        if not isinstance(other, BoundAction):
            return NotImplemented
        return self.unique_ident < other.unique_ident

    def __hash__(self) -> int:
        """Return a hash of this action.

        Returns:
            int: a hash of this action.
        """
        return hash(self.unique_ident)

    # FIXME: this method seems to not be used at all. Remove?
    def num_slots(self) -> int:
        """Return the number of slots this action takes up in the state.

        Returns:
            int: the number of slots this action takes up in the state.
        """
        return len(self.props) + len(self.flnts) + len(self.comps)


@total_ordering
class UnboundAction:
    """Represents an action that *may* be lifted. Use .bind() with an argument
    list to ground it."""

    def __init__(self,
                 schema_name: str,
                 param_names: Iterable[str],
                 rel_props: Iterable[UnboundProp],
                 rel_flnts: Iterable[UnboundFlnt],
                 rel_comps: Iterable[UnboundComp]):
        """Create a new UnboundAction.

        Args:
            schema_name (str): The schema name of this action.
            param_names (Iterable[str]): The names of the parameters of this
            action.
            rel_props (Iterable[UnboundProp]): The propositions that are
            relavent to this action.
            rel_flnts (Iterable[UnboundFlnt]): The functions that are relevant
            to this action.
            rel_comps (Iterable[UnboundComp]): The comparisons that are
            relevant to this action.
        """
        self.schema_name = schema_name
        self.param_names = tuple(param_names)
        self.rel_props = tuple(rel_props)
        self.rel_flnts = tuple(rel_flnts)
        self.rel_comps = tuple(rel_comps)

        assert isinstance(schema_name, str)
        assert all(isinstance(a, str) for a in self.param_names)
        assert all(isinstance(p, UnboundProp) for p in self.rel_props)
        assert all(isinstance(f, UnboundFlnt) for f in self.rel_flnts)
        assert all(isinstance(c, UnboundComp) for c in self.rel_comps)

    def __repr__(self) -> str:
        """Return a string representation of this action.

        Returns:
            str: a string representation of this action.
        """
        return 'UnboundAction(%r, %r, %r, %r, %r)' % (
            self.schema_name, self.param_names, self.rel_props, self.rel_flnts,
            self.rel_comps)

    def _ident_tup(self) \
            -> Tuple[str, Tuple[str], Tuple[UnboundProp], Tuple[UnboundFlnt]]:
        """Return a tuple that uniquely identifies this action.

        Returns:
            Tuple[str, Tuple[str], Tuple[UnboundProp], Tuple[UnboundFlnt]]: a
            tuple that uniquely identifies this action.
        """
        return (self.schema_name, self.param_names, self.rel_props,
                self.rel_flnts, self.rel_comps)

    def __eq__(self, other: Self) -> bool:
        """Return whether two UnboundAction instances are equal.

        Args:
            other (Self): the action to compare to.

        Returns:
            bool: whether self is equal to other.
        """
        if not isinstance(other, UnboundAction):
            return NotImplemented
        return self._ident_tup() == other._ident_tup()

    def __lt__(self, other: Self) -> bool:
        """Return whether this action is less than another action.

        Args:
            other (Self): the other action to compare to.

        Returns:
            bool: whether this action is less than the other action.
        """
        if not isinstance(other, UnboundAction):
            return NotImplemented
        # avoid using self.rel_props because that would need ordering on
        # UnboundProp instances
        return (self.schema_name, self.param_names) \
            < (other.schema_name, other.param_names)

    def __hash__(self) -> int:
        """Return the hash of the action.

        Returns:
            int: the hash of the action.
        """
        return hash(self._ident_tup())

    # FIXME: this method seems to not be used at all. Remove?
    def num_slots(self) -> int:
        """Return the number of slots in the action.

        Returns:
            int: the number of slots in the action.
        """
        return len(self.rel_props) + len(self.rel_flnts) + len(self.rel_comps)


class DomainMeta:
    """Represents the meta-information of a domain.
    """

    def __init__(self,
                 name: str,
                 unbound_acts: Iterable[UnboundAction],
                 unbound_comps: Iterable[UnboundComp],
                 pred_names: Iterable[str],
                 func_names: Iterable[str]):
        """Create a new DomainMeta.

        Args:
            name (str): name of the domain.
            unbound_acts (Iterable[UnboundAction]): unbound actions in the
            domain.
            unbound_comps (Iterable[UnboundComp]): unbound comparisons in
            the domain.
            pred_names (Iterable[str]): names of predicates in the domain.
            func_names (Iterable[str]): names of functions in the domain.
        """
        self.name = name
        self.unbound_acts = tuple(unbound_acts)
        self.unbound_comps = tuple(unbound_comps)
        self.pred_names = tuple(pred_names)
        self.func_names = tuple(func_names)

    def __repr__(self) -> str:
        """Return a string representation of this domain.

        Returns:
            str: a string representation of this domain.
        """
        return 'DomainMeta(%s, %s, %s, %s)' \
            % (self.name, self.unbound_acts, self.unbound_comps,
               self.pred_names, self.func_names)

    # * The two methods below can have overlapping slot numbers.

    @lru_cache(None)
    def rel_act_slots_of_pred(self, predicate_name: str) \
            -> List[Tuple[UnboundAction, int]]:
        """Map predicate name to names of relevant action schemas (without 
        duplicates).

        Args:
            predicate_name (str): name of predicate.

        Returns:
            List[Tuple[UnboundAction, int]]: list of tuples of unbound action 
            and slot number.
        """
        assert isinstance(predicate_name, str)
        rv: List[Tuple[UnboundAction, int]] = []
        for ub_act in self.unbound_acts:
            act_rps = self.rel_pred_names(ub_act)
            for slot, other_predicate_name in enumerate(act_rps):
                if predicate_name != other_predicate_name:
                    continue
                rv.append((ub_act, slot))
            # FIXME: maybe the "slots" shouldn't be integers, but rather tuples
            # of names representing parameters of the predicate like in
            # commented code below? Could then make those names consistent with
            # naming of the unbound action's parameters.
            # for ub_prop in ub_act.rel_props:
            #     if ub_prop.pred_name != predicate_name:
            #         continue
            #     slot_ident = ub_prop.params
            #     rv.append((ub_act, slot_ident))
        return rv

    @lru_cache(None)
    def rel_act_slots_of_func(self, function_name: str) \
            -> List[Tuple[UnboundAction, int]]:
        """Map function name to names of relevant action schemas (without 
        duplicates).

        Args:
            function_name (str): name of function.

        Returns:
            List[Tuple[UnboundAction, int]]: list of tuples of unbound action 
            and slot number.
        """
        assert isinstance(function_name, str)
        rv: List[Tuple[UnboundAction, int]] = []
        for ub_act in self.unbound_acts:
            act_rfs = self.rel_func_names(ub_act)
            for slot, other_function_name in enumerate(act_rfs):
                if function_name != other_function_name:
                    continue
                rv.append((ub_act, slot))
            # FIXME: same as rel_act_slots_of_func
        return rv

    @lru_cache(None)
    def rel_act_slots_of_comp(self, comparison: UnboundComp) \
            -> List[Tuple[UnboundAction, int]]:
        """Map comparisons to relevant action schemas (without duplicates).

        Args:
            comparison (UnboundComp): the comparison.

        Returns:
            List[Tuple[UnboundAction, int]]: list of tuples of unbound action 
            and slot number.
        """
        assert isinstance(comparison, UnboundComp)
        rv: List[Tuple[UnboundAction, int]] = []
        for ub_act in self.unbound_acts:
            act_rcs = self.rel_comps(ub_act)
            for slot, other_comparison in enumerate(act_rcs):
                if comparison != other_comparison:
                    continue
                rv.append((ub_act, slot))
        return rv

    @lru_cache(None)
    def rel_pred_names(self, action: UnboundAction) -> List[str]:
        """Return the names of the predicates relevant to the action.

        Args:
            action (UnboundAction): the action.

        Returns:
            List[str]: the names of the predicates relevant to the action,
            with duplicates.
        """
        assert isinstance(action, UnboundAction)
        rv = []
        for unbound_prop in action.rel_props:
            rv.append(unbound_prop.pred_name)
        return rv

    @lru_cache(None)
    def rel_func_names(self, action: UnboundAction) -> List[str]:
        """Return the names of the functions relevant to the action.

        Args:
            action (UnboundAction): the action.

        Returns:
            List[str]: the names of the functions relevant to the action, with
            duplicates.
        """
        assert isinstance(action, UnboundAction)
        rv = []
        for unbound_func in action.rel_flnts:
            rv.append(unbound_func.func_name)
        return rv

    @lru_cache(None)
    def rel_comps(self, action: UnboundAction) -> List[UnboundComp]:
        """Return the names of the comparisons relevant to the action.

        Args:
            action (UnboundAction): the action

        Returns:
            List[UnboundComp]: the unbound comparisons relevant to the action
        """
        assert isinstance(action, UnboundAction)
        rv = []
        for unbound_comp in action.rel_comps:
            rv.append(unbound_comp)
        return rv

    @property
    @lru_cache(None)
    def all_unbound_props(self) \
            -> Tuple[List[UnboundProp], Dict[str, UnboundProp]]:
        """Return all unbound props in the domain.

        Returns:
            Tuple[List[UnboundProp], Dict[str, UnboundProp]]: the list of all
            unbound props and a dictionary mapping predicate names to unbound
            props.
        """
        unbound_props: List[UnboundProp] = []
        ub_prop_set: Set[UnboundProp] = set()
        ub_prop_dict: Dict[str, UnboundProp] = {}
        for unbound_act in self.unbound_acts:
            for ub_prop in unbound_act.rel_props:
                if ub_prop not in ub_prop_set:
                    unbound_props.append(ub_prop)
                    ub_prop_dict[ub_prop.pred_name] = ub_prop
                    # the set is just to stop double-counting
                    ub_prop_set.add(ub_prop)
        return unbound_props, ub_prop_dict

    @property
    @lru_cache(None)
    def all_unbound_flnts(self) \
            -> Tuple[List[UnboundFlnt], Dict[str, UnboundFlnt]]:
        """Return all unbound fluents in the domain.

        Returns:
            Tuple[List[UnboundFlnt], Dict[str, UnboundFlnt]]: the list of all
            unbound fluents and a dictionary mapping function names to
            unbound fluents.
        """
        unbound_funcs = List[UnboundFlnt] = []
        ub_flnt_set: Set[UnboundFlnt] = set()
        ub_flnt_dict: Dict[str, UnboundFlnt] = {}
        for unbound_act in self.unbound_acts:
            for ub_flnt in unbound_act.rel_flnts:
                if ub_flnt not in ub_flnt_set:
                    unbound_funcs.append(ub_flnt)
                    ub_flnt_dict[ub_flnt.func_name] = ub_flnt
                    # the set is just to stop double-counting
                    ub_flnt_set.add(ub_flnt)
        return unbound_funcs, ub_flnt_dict

    def unbound_prop_by_name(self, predicate_name: str) -> UnboundProp:
        """Return the unbound prop with the given name.

        Args:
            predicate_name (str): the name of the predicate.

        Returns:
            UnboundProp: the unbound prop with the given name.
        """
        _, ub_prop_dict = self.all_unbound_props
        return ub_prop_dict[predicate_name]

    def unbound_flnt_by_name(self, function_name: str) -> UnboundFlnt:
        """Return the unbound fluent with the given name.

        Args:
            function_name (str): the name of the function.

        Returns:
            UnboundFlnt: the unbound fluent with the given name.
        """
        _, ub_flnt_dict = self.all_unbound_flnts
        return ub_flnt_dict[function_name]

    def _ident_tup(self) \
            -> Tuple[str, Tuple[UnboundAction], Tuple[str], Tuple[str]]:
        """Return a tuple that uniquely identifies this domain.

        Returns:
            Tuple[str, Tuple[UnboundAction], Tuple[str], Tuple[str]]: a tuple
            that uniquely identifies this domain.
        """
        return (self.name, self.unbound_acts, self.pred_names, self.func_names)

    def __eq__(self, other: Self) -> bool:
        """Return whether this domain is equal to another.

        Args:
            other (Self): the other domain.

        Returns:
            bool: whether this domain is equal to another.
        """
        if not isinstance(other, DomainMeta):
            return NotImplemented
        return self._ident_tup() == other._ident_tup()

    def __hash__(self) -> int:
        """Return a hash of this domain.

        Returns:
            int: a hash of this domain.
        """
        return hash(self._ident_tup())


class ProblemMeta:
    # Some notes on members/properties
    # - rel_props is dict mapping ground action name => [relevant prop names]
    # - rel_acts is dict mapping prop name => [relevant ground action name]

    def __init__(self, name: str, domain: DomainMeta,
                 bound_acts_ordered: Iterable[BoundAction],
                 bound_props_ordered: Iterable[BoundProp],
                 bound_flnts_ordered: Iterable[BoundFlnt],
                 bound_comps_ordered: Iterable[BoundComp],
                 goal_props: Iterable[BoundProp],
                 goal_flnts: Iterable[BoundFlnt],
                 static_flnt_values: Dict[BoundFlnt, float]):
        """Initialize a problem meta.

        Args:
            name (str): Name of the problem.
            domain (DomainMeta): Domain of the problem.
            bound_acts_ordered (Iterable[BoundAction]): The bound actions in
            the problem, in lexical order.
            bound_props_ordered (Iterable[BoundProp]): The bound propositions
            in the problem, in lexical order.
            bound_flnts_ordered (Iterable[BoundFlnt]): The bound fluents in the
            problem, in lexical order.
            bound_comps_ordered (Iterable[BoundComp]): The bound comparisons in
            the problem, in lexical order.
            goal_props (Iterable[BoundProp]): The goal propositions in the
            problem.
            goal_flnts (Iterable[BoundFlnt]): Fluents involved in the goal in
            this problem.
            static_flnt_values (Dict[BoundFlnt, float]): The values of static
            fluents in this problem.
        """
        self.name = name
        self.domain = domain
        self.bound_acts_ordered = tuple(bound_acts_ordered)
        self.bound_props_ordered = tuple(bound_props_ordered)
        self.bound_flnts_ordered = tuple(bound_flnts_ordered)
        self.bound_comps_ordered = tuple(bound_comps_ordered)
        self.goal_props = tuple(goal_props)
        self.goal_flnts = tuple(goal_flnts)
        self.static_flnt_values = static_flnt_values

        self._unique_id_to_index: Dict[str, int] = {
            bound_act.unique_ident: idx
            for idx, bound_act in enumerate(self.bound_acts_ordered)
        }

        # sanity checks
        assert set(self.goal_props) <= set(self.bound_props_ordered)
        assert set(self.goal_flnts) <= set(self.bound_flnts_ordered)

    def __repr__(self) -> str:
        """Return a string representation of this problem meta.

        Returns:
            str: a string representation of this problem meta.
        """
        return 'ProblemMeta(%s, %s, %s, %s, %s, %s, %s)' \
            % (self.name, self.domain, self.bound_acts_ordered,
               self.bound_props_ordered, self.bound_flnts_ordered,
               self.bound_comps_ordered, self.goal_props, self.goal_flnts,
               self.static_flnt_values)

    @property
    def num_props(self) -> int:
        """Return the number of propositions in the problem.

        Returns:
            int: the number of propositions in the problem.
        """
        return len(self.bound_props_ordered)

    @property
    def num_flnts(self) -> int:
        """Return the number of fluents in the problem.

        Returns:
            int: the number of fluents in the problem.
        """
        return len(self.bound_flnts_ordered)

    @property
    def num_acts(self) -> int:
        """Return the number of actions in the problem.

        Returns:
            int: the number of actions in the problem.
        """
        return len(self.bound_acts_ordered)

    @property
    def num_comps(self) -> int:
        """Return the number of comparisons in the problem.

        Returns:
            int: the number of comparisons in the problem.
        """
        return len(self.bound_comps_ordered)

    @lru_cache(None)
    def schema_to_acts(self, unbound_action: UnboundAction) \
            -> List[BoundAction]:
        """Return the bound actions that are instances of the given unbound
        action.

        Returns:
            List[BoundAction]: the bound actions that are instances of the
            unbound action.
        """
        assert isinstance(unbound_action, UnboundAction)
        return [
            a for a in self.bound_acts_ordered if a.prototype == unbound_action
        ]

    @lru_cache(None)
    def pred_to_props(self, pred_name: str) -> List[BoundProp]:
        """Return the bound propositions that have the given predicate name.

        Args:
            pred_name (str): the predicate name.

        Returns:
            List[BoundProp]: the bound propositions that have the given
            predicate name.
        """
        assert isinstance(pred_name, str)
        return [
            p for p in self.bound_props_ordered if p.pred_name == pred_name
        ]

    @lru_cache(None)
    def func_to_flnts(self, func_name: str) -> List[BoundFlnt]:
        """Return the bound fluents that have the given function name.

        Args:
            func_name (str): the predicate name.

        Returns:
            List[BoundFlnt]: the bound fluents that have the given
            predicate name.
        """
        assert isinstance(func_name, str)
        return [
            f for f in self.bound_flnts_ordered if f.func_name == func_name
        ]

    @lru_cache(None)
    def unbound_comp_to_comps(self, unbound_comp: UnboundComp) \
            -> List[BoundComp]:
        """Return the bound comparisons that have the given unbound comparison
        as their schema.

        Args:
            unbound_comp (UnboundComp): the unbound comparison.

        Returns:
            List[BoundComp]: the bound comparisons that have the given unbound
            comparison as their schema.
        """
        assert isinstance(unbound_comp, UnboundComp)
        return [
            f for f in self.bound_comps_ordered if f.schema == unbound_comp]

    def prop_to_pred(self, bound_prop: BoundProp) -> str:
        """Return the predicate name of the given bound proposition.

        Args:
            bound_prop (BoundProp): the bound proposition.

        Returns:
            str: the predicate name of the given bound proposition.
        """
        assert isinstance(bound_prop, BoundProp)
        return bound_prop.pred_name

    def act_to_schema(self, bound_act: BoundAction) -> UnboundAction:
        """Return the unbound action that is the schema of the given bound
        action.

        Args:
            bound_act (BoundAction): the bound action.

        Returns:
            UnboundAction: the unbound action that is the schema of the given
            bound action.
        """
        assert isinstance(bound_act, BoundAction)
        return bound_act.prototype

    def flnt_to_func(self, bound_flnt: BoundFlnt) -> str:
        """Return the function name of the given bound fluent.

        Args:
            bound_flnt (BoundFlnt): the bound fluent.

        Returns:
            str: the function name of the given bound fluent.
        """
        assert isinstance(bound_flnt, BoundFlnt)
        return bound_flnt.func_name

    def comp_to_unbound_comp(self, bound_comp: BoundComp) -> UnboundComp:
        """Return the unbound comparison that is the schema of the given bound
        comparison.

        Args:
            bound_comp (BoundComp): the bound comparison.

        Returns:
            UnboundComp: the unbound comparison that is the schema of the given
        """
        assert isinstance(bound_comp, BoundComp)
        return bound_comp.schema

    def rel_props(self, bound_act: BoundAction) -> List[BoundProp]:
        """Return the relevant bound propositions for the given bound action.

        Args:
            bound_act (BoundAction): the bound action.

        Returns:
            List[BoundProp]: the relevant bound propositions for the given
            bound action.
        """
        assert isinstance(bound_act, BoundAction)
        # no need for special grouping like in rel_acts, since all props can be
        # concatenated before passing them in
        return bound_act.props

    def rel_flnts(self, bound_act: BoundAction) -> List[BoundFlnt]:
        """Return the relevant bound fluents for the given bound action.

        Args:
            bound_act (BoundAction): the bound action.

        Returns:
            List[BoundFlnt]: the relevant bound fluents for the given
            bound action.
        """
        assert isinstance(bound_act, BoundAction)
        # no need for special grouping like in rel_acts, since all fluents can
        # be concatenated before passing them in
        return bound_act.flnts

    def rel_comps(self, bound_act: BoundAction) -> List[BoundComp]:
        """Return the relevant bound comparisons for the given bound action.

        Args:
            bound_act (BoundAction): the bound action.

        Returns:
            List[BoundComp]: the relevant bound comparisons for the given
            bound action.
        """
        assert isinstance(bound_act, BoundAction)
        return bound_act.comps

    @lru_cache(None)
    def rel_act_slots_of_prop(self, bound_prop: BoundProp) \
            -> List[Tuple[UnboundAction, int, List[BoundAction]]]:
        """Return the relevant actions and slots for the given bound 
        proposition.

        Args:
            bound_prop (BoundProp): the bound proposition.

        Returns:
            List[Tuple[UnboundAction, int, List[BoundAction]]]: the relevant
            unbound actions, slots, and bound actions for the given bound
            proposition.
        """
        assert isinstance(bound_prop, BoundProp)
        rv = []
        pred_name = self.prop_to_pred(bound_prop)
        for unbound_act, slot in self.domain.rel_act_slots_of_pred(pred_name):
            bound_acts_for_schema = []
            for bound_act in self.schema_to_acts(unbound_act):
                if bound_prop == self.rel_props(bound_act)[slot]:
                    # TODO: is this the best way to do this? See comment in
                    # DomainMeta.rel_acts.
                    bound_acts_for_schema.append(bound_act)
            rv.append((unbound_act, slot, bound_acts_for_schema))
        return rv

    @lru_cache(None)
    def rel_act_slots_of_flnt(self, bound_flnt: BoundFlnt) \
            -> List[Tuple[UnboundAction, int, List[BoundAction]]]:
        """Return the relevant actions and slots for the given bound fluent.

        Args:
            bound_prop (BoundProp): the bound fluent.

        Returns:
            List[Tuple[UnboundAction, int, List[BoundAction]]]: the relevant
            unbound actions, slots, and bound actions for the given bound
            fluent.
        """
        assert isinstance(bound_flnt, BoundFlnt)
        rv = []
        func_name = self.flnt_to_func(bound_flnt)
        for unbound_act, slot in self.domain.rel_act_slots_of_func(func_name):
            bound_acts_for_schema = []
            for bound_act in self.schema_to_acts(unbound_act):
                if bound_flnt == self.rel_flnts(bound_act)[slot]:
                    # TODO: is this the best way to do this? See comment in
                    # DomainMeta.rel_acts.
                    bound_acts_for_schema.append(bound_act)
            rv.append((unbound_act, slot, bound_acts_for_schema))
        return rv

    @lru_cache(None)
    def rel_act_slots_of_comp(self, bound_comp: BoundComp) \
            -> List[Tuple[UnboundAction, int, List[BoundAction]]]:
        assert isinstance(bound_comp, BoundComp)
        rv = []
        unbound_comp = self.comp_to_unbound_comp(bound_comp)
        for unbound_act, slot in self.domain.rel_act_slots_of_comp(unbound_comp):
            bound_acts_for_schema = []
            for bound_act in self.schema_to_acts(unbound_act):
                if bound_comp == self.rel_comps(bound_act)[slot]:
                    bound_acts_for_schema.append(bound_act)
            rv.append((unbound_act, slot, bound_acts_for_schema))
        return rv

    @lru_cache(None)
    def prop_to_pred_subtensor_ind(self, bound_prop: BoundProp) -> int:
        """Return the index of the given bound proposition in the tensor of
        propositions for the given predicate name.

        Args:
            bound_prop (BoundProp): the bound proposition.

        Returns:
            int: the index of the given bound proposition in the tensor of
            propositions for the given predicate name.
        """
        assert isinstance(bound_prop, BoundProp)
        pred_name = self.prop_to_pred(bound_prop)
        prop_vec = self.pred_to_props(pred_name)
        return prop_vec.index(bound_prop)

    @lru_cache(None)
    def act_to_schema_subtensor_ind(self, bound_act: BoundAction) -> int:
        """Return the index of the given bound action in the tensor of actions
        for the given unbound action.

        Args:
            bound_act (BoundAction): the bound action.

        Returns:
            int: the index of the given bound action in the tensor of actions
            for the given unbound action.
        """
        assert isinstance(bound_act, BoundAction)
        unbound_act = self.act_to_schema(bound_act)
        schema_vec = self.schema_to_acts(unbound_act)
        return schema_vec.index(bound_act)

    @lru_cache(None)
    def flnt_to_func_subtensor_ind(self, bound_flnt: BoundFlnt) -> int:
        """Return the index of the given bound fluent in the tensor of fluents
        for the given predicate name.

        Args:
            bound_flnt (BoundFlnt): the bound fluent.

        Returns:
            int: the index of the given bound fluent in the tensor of
            fluents for the given predicate name.
        """
        assert isinstance(bound_flnt, BoundFlnt)
        func_name = self.flnt_to_func(bound_flnt)
        flnt_vec = self.func_to_flnts(func_name)
        return flnt_vec.index(bound_flnt)

    @lru_cache(None)
    def comp_to_unbound_comp_subtensor_ind(self, bound_comp: BoundComp) -> int:
        assert isinstance(bound_comp, BoundComp)
        unbound_comp = self.comp_to_unbound_comp(bound_comp)
        comp_vec = self.unbound_comp_to_comps(unbound_comp)
        return comp_vec.index(bound_comp)

    @lru_cache(None)
    def _props_by_name(self) -> Dict[str, BoundProp]:
        """Return a dictionary mapping the unique identifier of each bound
        proposition to the bound proposition.

        Returns:
            Dict[str, BoundProp]: a dictionary mapping the unique identifier of
            each bound proposition to the bound proposition.
        """
        all_props = {
            prop.unique_ident: prop
            for prop in self.bound_props_ordered
        }
        return all_props

    @lru_cache(None)
    def _flnts_by_name(self) -> Dict[str, BoundFlnt]:
        """Return a dictionary mapping the unique identifier of each bound
        fluent to the bound fluent.

        Returns:
            Dict[str, BoundFlnt]: a dictionary mapping the unique identifier of
            each bound fluent to the bound fluent.
        """
        all_flnts = {
            flnt.unique_ident: flnt
            for flnt in self.bound_flnts_ordered
        }
        return all_flnts

    @lru_cache(None)
    def _acts_by_name(self) -> Dict[str, BoundAction]:
        """Return a dictionary mapping the unique identifier of each bound
        action to the bound action.

        Returns:
            Dict[str, BoundAction]: a dictionary mapping the unique identifier
            of each bound action to the bound action.
        """
        all_acts = {
            act.unique_ident: act
            for act in self.bound_acts_ordered
        }
        return all_acts

    @lru_cache(None)
    def bound_prop_by_name(self, string: str) -> BoundProp:
        """Return the bound proposition with the given unique identifier.

        Args:
            strin (str): the unique identifier of the bound proposition.

        Returns:
            BoundProp: the bound proposition with the given unique identifier.
        """
        all_props = self._props_by_name()
        return all_props[string]

    @lru_cache(None)
    def bound_flnt_by_name(self, string: str) -> BoundFlnt:
        """Return the bound fluent with the given unique identifier.

        Args:
            strin (str): the unique identifier of the bound fluent.

        Returns:
            BoundProp: the bound fluent with the given unique identifier.
        """
        all_flnts = self._flnts_by_name()
        return all_flnts[string]

    @lru_cache(None)
    def bound_act_by_name(self, string: str) -> BoundAction:
        """Return the bound action with the given unique identifier.

        Args:
            string (str): the unique identifier of the bound action.

        Returns:
            BoundAction: the bound action with the given unique identifier.
        """
        all_acts = self._acts_by_name()
        return all_acts[string]

    def act_unique_id_to_index(self, string: str) -> int:
        """Return the index of the given bound action in the list of all
        bound actions.

        Args:
            string (str): the unique identifier of the bound action.

        Returns:
            int: the index of the given bound action in the list of all bound
            actions.
        """
        return self._unique_id_to_index[string]


def make_unbound_prop(mdpsim_lifted_prop: Any) -> UnboundProp:
    """Make an UnboundProp from an MDPsim PyProposition. Does not check that the
    PyProposition is actually lifted and allows for some terms to be bound, such
    as in the case of constants.

    Args:
        mdpsim_lifted_prop (Any): A PyProposition from MDPsim.

    Returns:
        UnboundProp: The corresponding UnboundProp.
    """
    pred_name: str = mdpsim_lifted_prop.predicate.name
    terms: List[str] = [t.name for t in mdpsim_lifted_prop.terms]
    return UnboundProp(pred_name, terms)


def make_bound_prop(mdpsim_ground_prop: Any) -> BoundProp:
    """Make a BoundProp from an MDPsim PyProposition. Verifies that all terms
    are actually bound.

    Args:
        mdpsim_ground_prop (Any): A PyProposition from MDPsim.

    Returns:
        BoundProp: The corresponding BoundProp.
    """
    pred_name: str = mdpsim_ground_prop.predicate.name
    arguments: List[str] = []
    for term in mdpsim_ground_prop.terms:
        term_name = term.name
        arguments.append(term_name)
        # make sure it's really a binding
        assert not term_name.startswith('?'), \
            f"term '{term_name}' starts with '?'---sure it's not free?"
    bound_prop = BoundProp(pred_name, arguments)
    return bound_prop


def make_unbound_flnt(mdpsim_lifted_flnt: Any) -> UnboundFlnt:
    """Make an UnboundFlnt from an mdpsim PyFluent. Does not check that the
    PyFluent is actually lifted and allows for some terms to be bound, such as
    in the case of constants.

    Args:
        mdpsim_lifted_flnt (Any): A PyFluent from MDPSim.

    Returns:
        UnboundFlnt: The corresponding UnboundFlnt.
    """
    func_name = mdpsim_lifted_flnt.function.name
    terms: List[str] = [t.name for t in mdpsim_lifted_flnt.terms]
    return UnboundFlnt(func_name, terms)


def make_bound_flnt(mdpsim_ground_flnt: Any) -> BoundFlnt:
    """Make a BoundFlnt from an mdpsim PyFluent. Verifies that all terms are
    actually bound.

    Args:
        mdpsim_ground_flnt (Any): A PyFluent from MDPSim.

    Returns:
        BoundFlnt: The corresponding BoundFlnt.
    """
    func_name: str = mdpsim_ground_flnt.function.name
    arguments: List[str] = []
    for term in mdpsim_ground_flnt.terms:
        term_name = term.name
        arguments.append(term_name)
        assert not term_name.startswith('?'), \
            f"term '{term_name}' starts with '?'---sure it's not free?"
    bound_flnt = BoundFlnt(func_name, arguments)
    return bound_flnt


def make_unbound_action(mdpsim_lifted_act: Any) -> UnboundAction:
    """Make an UnboundAction from an mdpsim PyLiftedAction.

    Args:
        mdpsim_lifted_act (Any): A PyLiftedAction object from MDPSim.

    Returns:
        UnboundAction: The corresponding UnboundAction.
    """
    schema_name: str = mdpsim_lifted_act.name
    param_names: List[str] = [
        param.name for param, _ in mdpsim_lifted_act.parameters_and_types
    ]
    rel_props = []
    rel_prop_set = set()
    for mdpsim_prop in mdpsim_lifted_act.involved_propositions:
        unbound_prop = make_unbound_prop(mdpsim_prop)
        if unbound_prop not in rel_prop_set:
            # ignore duplicates
            rel_prop_set.add(unbound_prop)
            rel_props.append(unbound_prop)

    rel_flnts = []
    rel_flnts_set = set()
    for mdpsim_flnt in mdpsim_lifted_act.involved_functions:
        unbound_flnt = make_unbound_flnt(mdpsim_flnt)
        if unbound_flnt not in rel_flnts_set:
            # ignore duplicates
            rel_flnts_set.add(unbound_flnt)
            rel_flnts.append(unbound_flnt)

    rel_comps = []
    rel_comps_set = set()
    for mdpsim_comp in mdpsim_lifted_act.involved_lifted_comparisons:
        unbound_comp = make_unbound_comp(mdpsim_comp)
        if unbound_comp not in rel_comps_set:
            # ignore duplicates
            rel_comps_set.add(unbound_comp)
            rel_comps.append(unbound_comp)

    return UnboundAction(schema_name, param_names, rel_props, rel_flnts,
                         rel_comps)


def make_bound_action(mdpsim_ground_act: Any) \
        -> BoundAction:
    """Make a BoundAction from a PyGroundAction object.

    Args:
        mdpsim_ground_act (Any): A PyGroundAction object from MDPSim.

    Returns:
        BoundAction: The corresponding BoundAction object.
    """
    lifted_act = make_unbound_action(mdpsim_ground_act.lifted_action)
    arguments = [arg.name for arg in mdpsim_ground_act.arguments]
    if not isinstance(arguments, (list, str)):
        raise TypeError('expected args to be list or str')

    # NOTE(Ryan) we need to make sure that the order and number of bounded
    # propositions, fluents, etc. match the order and number of the unbound
    # ones. This is so we can use the index of the unbound one to index into
    # the bounded one.
    #
    # To achieve this, we effectively use string find and replace for the
    # propositions and fluents to ground the lifted ones ourselves. It's
    # much more complicated to ground the comparisons ourselves, as you have
    # to deal with simplification in the case of static fluents. Instead, we
    # use the lifted version of the comparisons to order the grounded
    # comparisons we get from mdpsim. We can't do the same for the props
    # and flnts as UnboundProps and UnboundFluents are made empty mdpsim
    # grounded stuff, not actual lifted stuff.
    #
    # Reorder the comparisons is actually not necessary as they should
    # already be ordered by mdpsim, but it's good to be safe and when
    # that changes

    def make_bounded_list(mdpsim_grounds: List[Any],
                          unbounds: List[Any],
                          ground_to_bound: Callable[[Any], Any],
                          ground_to_unbound: Callable[[Any], Any]):
        unbound_to_idx = {unbound: idx for idx, unbound in enumerate(unbounds)}
        rv = [None] * len(unbounds)
        for mdpsim_ground in mdpsim_grounds:
            bound = ground_to_bound(mdpsim_ground)
            unbound = ground_to_unbound(mdpsim_ground)

            assert unbound in unbound_to_idx, \
                f"unbound {unbound} not in {unbound_to_idx}"

            idx = unbound_to_idx[unbound]
            assert rv[idx] is None or rv[idx] == bound, \
                "have {} at index {} for {} but trying to add {}".format(
                    rv[idx], idx, unbound, bound)

            rv[idx] = bound
        
        assert all(x is not None for x in rv), \
            "not all elements of rv are filled in: {}".format(rv)

        return rv

    bindings = dict(zip(lifted_act.param_names, arguments))
    bound_props = [prop.bind(bindings) for prop in lifted_act.rel_props]
    bound_flnts = [flnt.bind(bindings) for flnt in lifted_act.rel_flnts]
    bound_comps = make_bounded_list(
        mdpsim_ground_act.involved_comparisons,
        lifted_act.rel_comps,
        lambda mdpsim_comp: make_bound_comp(mdpsim_comp),
        lambda mdpsim_comp: make_unbound_comp(mdpsim_comp.lifted_comparison))

    return BoundAction(lifted_act, arguments, bound_props, bound_flnts,
                       bound_comps)


def make_unbound_comp(mdpsim_lifted_comp: Any) -> UnboundComp:
    """Make an UnboundComp from an mdpsim PyComparison.

    Args:
        mdpsim_lifted_comp (Any): A PyLiftedComparison object from MDPSim.

    Returns:
        UnboundComp: The corresponding UnboundComp.
    """
    param_names: List[str] = [
        param.name for param, _ in mdpsim_lifted_comp.parameters_and_types
    ]

    return UnboundComp(str(mdpsim_lifted_comp), param_names)


def make_bound_comp(mdpsim_ground_comp: Any) -> BoundComp:
    """Make a BoundComp from an mdpsim PyGroundComparison.

    Args:
        mdpsim_ground_comp (Any): A PyGroundComparison object from MDPSim.

    Returns:
        BoundComp: The corresponding BoundComp.
    """
    return BoundComp(str(mdpsim_ground_comp),
                     make_unbound_comp(mdpsim_ground_comp.lifted_comparison))


def get_domain_meta(domain: Any) -> DomainMeta:
    """Extracts a nice, Pickle-able subset of the information contained in a
    domain so that we can construct the appropriate network weights.

    Args:
        domain (Any): A PyDomain object from MDPSim.

    Returns:
        DomainMeta: DomainMeta information contained in the PyDomain object.
    """
    pred_names: List[str] = [p.name for p in domain.predicates]
    # ignore mdpsim special functions
    func_names: List[str] = [f.name for f in domain.functions
                             if f.name not in SPECIAL_FUNCTIONS]
    unbound_acts = map(make_unbound_action, domain.lifted_actions)

    # there can be duplicates here
    unbound_comps = set(map(make_unbound_comp, domain.lifted_comparisons))
    return DomainMeta(domain.name, unbound_acts, unbound_comps, pred_names,
                      func_names)


def get_problem_meta(problem: Any, domain_meta: DomainMeta) -> ProblemMeta:
    """Extracts a nice, Pickle-able subset of the information contained in a
    problem so that we can construct the appropriate network weights.

    Args:
        problem (Any): A PyProblem object from MDPSim.
        domain_meta (DomainMeta): The DomainMeta object corresponding to the
        domain of the problem.

    Returns:
        ProblemMeta: ProblemMeta information contained in the PyProblem object.
    """
    # we get given the real domain, but we also do a double-check to make sure
    # that it matches our problem
    other_domain = get_domain_meta(problem.domain)
    assert other_domain == domain_meta, \
        "%r\n!=\n%r" % (other_domain, domain_meta)

    # use network input orders implied by problem.propositions and
    # problem.ground_actions
    bound_props_ordered: List[BoundProp] = []
    goal_props: List[BoundProp] = []
    for mdpsim_prop in problem.propositions:
        bound_prop = make_bound_prop(mdpsim_prop)
        bound_props_ordered.append(bound_prop)
        if mdpsim_prop.in_goal:
            goal_props.append(bound_prop)
    # must sort these!
    bound_props_ordered.sort()

    static_flnt_values: Dict[BoundFlnt, float] = {}
    bound_flnts_ordered: List[BoundFlnt] = []
    goal_flnts: List[BoundFlnt] = []
    for mdpsim_flnt in problem.fluents:
        bound_flnt = make_bound_flnt(mdpsim_flnt)
        bound_flnts_ordered.append(bound_flnt)
        if mdpsim_flnt.in_goal:
            goal_flnts.append(bound_flnt)
        if mdpsim_flnt.is_static:
            static_flnt_values[bound_flnt] = \
                problem.static_fluent_value(mdpsim_flnt)
    # must sort these
    bound_flnts_ordered.sort()

    bound_comps_ordered: List[BoundComp] = []
    for mdpsim_comp in problem.comparisons:
        bound_comp = make_bound_comp(mdpsim_comp)
        bound_comps_ordered.append(bound_comp)
    
    # deduplicate and sort
    bound_comps_ordered = list(set(bound_comps_ordered))
    bound_comps_ordered.sort()

    prop_set: Set[BoundProp] = set(bound_props_ordered)
    flnt_set: Set[BoundFlnt] = set(bound_flnts_ordered)
    comp_set: Set[BoundComp] = set(bound_comps_ordered)
    ub_act_set: Set[UnboundAction] = set(domain_meta.unbound_acts)
    bound_acts_ordered: List[BoundAction] = []
    for mdpsim_act in problem.ground_actions:
        bound_act = make_bound_action(mdpsim_act)
        bound_acts_ordered.append(bound_act)

        # sanity  checks
        assert set(bound_act.props) <= prop_set, \
            "bound_act.props (for act %r) not inside prop_set; odd ones: %r" \
            % (bound_act.unique_ident, set(bound_act.props) - prop_set)
        assert set(bound_act.flnts) <= flnt_set, \
            "bound_act.flnts (for act %r) not inside flnt_set; odd ones: %r" \
            % (bound_act.unique_ident, set(bound_act.flnts) - flnt_set)
        assert set(bound_act.comps) <= comp_set, \
            "bound_act.comps (for act %r) not inside comp_set; odd ones: %r" \
            % (bound_act.unique_ident, set(bound_act.comps) - comp_set)
        assert bound_act.prototype in ub_act_set, \
            "%r (bound_act.prototype) is not in %r (ub_act_set)" \
            % (bound_act.protype, ub_act_set)
    # again, need to sort lexically
    bound_acts_ordered.sort()

    return ProblemMeta(problem.name, domain_meta, bound_acts_ordered,
                       bound_props_ordered, bound_flnts_ordered,
                       bound_comps_ordered, goal_props, goal_flnts,
                       static_flnt_values)
