from functools import lru_cache

from asnets.prob_dom_meta import BoundFlnt, BoundProp

import jpype
import jpype.imports
from jpype.types import *

J_BoolPredicate = None
J_NumFluent = None


@jpype.onJVMStart
def _import_java_classes() -> None:
    """Import Java classes that will be used by this module. This is called
    automatically upon JVM start-up.
    """
    global J_BoolPredicate, J_NumFluent

    J_BoolPredicate = jpype.JPackage(
        'com').hstairs.ppmajal.conditions.BoolPredicate
    J_NumFluent = jpype.JPackage('com').hstairs.ppmajal.expressions.NumFluent


@lru_cache(None)
def flnt_to_jpddl_id(bf: BoundFlnt) -> int:
    """Convert a BoundFluent to an ID number as according to the NumFluent
    class in JPDDL.

    Args:
        bf (BoundFlnt): The bound fluent to convert.

    Returns:
        int: The ID number of the bound fluent.
    """
    matches = [
        id
        for id, num_fluent in enumerate(J_NumFluent.fromIdToNumFluents)
        if str(num_fluent).strip('()').lower() == bf.unique_ident
    ]
    assert len(matches) == 1, f'No unique match for bound fluent {bf}.'
    return matches[0]


@lru_cache(None)
def prop_to_jpddl_id(bp: BoundProp) -> int:
    """Convert a BoundProp to an ID number as according to the BoolPredicate
    class in JPDDL.

    Args:
        bp (BoundProp): The bound prop to convert.

    Returns:
        int: The ID number of the bound prop.
    """
    matches = [
        bool_predicate.getId()
        for bool_predicate in J_BoolPredicate.getPredicatesDB().values()
        if str(bool_predicate).strip('()').lower() == bp.unique_ident
    ]
    assert len(matches) == 1, f'No unique match for bound prop {bp}.'
    return matches[0]
