"""Tools for interfacing with JPDDL using JPype.
"""

# ! Everything in this file that is actually a Java class should have prefix
# ! 'j_' (objects) or `J_` (classes) to avoid confusion with Python.

from dataclasses import dataclass
import os
import logging
from typing import List, Set, Tuple
from asnets.utils.py_utils import get_files_with_extension, run_once

import jpype
import jpype.imports
from jpype.types import *

from asnets.prob_dom_meta import BoundAction
from asnets.state_reprs import CanonicalState

LOGGER = logging.getLogger(__name__)

# Java classes that will be imported automatically upon JVM start-up.
J_LandmarkGenerator = None

JARS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'jars'
)


@run_once
def start_jvm() -> None:
    """Start the JVM. Only the first call to this function will do anything."""
    jpype.startJVM(
        jvmpath=jpype.getDefaultJVMPath(),
        classpath=get_files_with_extension(JARS_DIR, '.jar')
    )


@jpype.onJVMStart
def _import_java_classes() -> None:
    """Import Java classes that will be used by this module. This is called
    automatically upon JVM start-up.
    """
    global J_LandmarkGenerator

    J_LandmarkGenerator = jpype.JPackage('com') \
        .hstairs.ppmajal.pddl.heuristics.advanced.LandmarkGenerator


@jpype.onJVMStart
def _suppress_java_stdout():
    """Suppress JPDDL logging. This is called automatically upon JVM start-up.
    """
    J_System = jpype.JPackage('java').lang.System
    J_PrintStream = jpype.JPackage('java').io.PrintStream
    J_File = jpype.JPackage('java').io.File
    J_System.setOut(J_PrintStream(J_File(os.devnull)))


@dataclass
class NumericLandmark:
    """A numeric landmark.
    
    Attributes:
        actions (List[Tuple[BoundAction, float]]): A list of pairs of actions
            and their contributions to the landmark.
        target (int): The target value of the landmark.
    """
    actions: List[Tuple[BoundAction, float]]
    target: int


class NumericLandmarkGenerator:
    def __init__(self,
                 planner_exts: 'PlannerExtensions',
                 verbose=False):
        """Initialize the numeric cutter.

        Args:
            planner_exts (PlannerExtensions): The planner extensions.
        """
        self.prob_meta = planner_exts.problem_meta
        self.j_landmark_generator = J_LandmarkGenerator(planner_exts.j_problem)
        self._cache = {}
        self.verbose = verbose

        # variables for statistics
        if self.verbose:
            self._prop_landmark_count = 0
            self._num_landmark_count = 0
            self._log_frequency = 1000
            self._log_counter = 0

    def get_landmarks(self, cstate: CanonicalState) -> List[NumericLandmark]:
        """Generate landmarks for the given state.

        Args:
            cstate (CanonicalState): The state to generate landmarks for.

        Returns:
            List[NumericLandmark]: A list of landmarks.
        """
        if cstate in self._cache:
            return self._cache[cstate]

        # See JPDDLPlus for more details on what a landmark here contains.
        j_state = cstate.to_jpddl()
        j_landmarks = self.j_landmark_generator.getActionLandmarks(
            j_state)

        landmarks = []
        for j_landmark in j_landmarks:
            def action_to_bound_action(action) -> BoundAction:
                # the action is a java string of the form
                # (action_name arg1 arg2 ...)
                # the action_name might not be lower case, which mdpsim
                # enforces, this step implicitly verifies all actions in the
                # landmarks are valid.
                return self.prob_meta.bound_act_by_name(
                    str(action).lower().strip('()'))

            landmark = NumericLandmark(
                actions=[
                    (action_to_bound_action(a),
                     j_landmark.contributions()[i])
                    for i, a in enumerate(j_landmark.actions())],
                target=j_landmark.target())
            landmarks.append(landmark)
        
        if self.verbose:
            num_landmark_count = sum([1 for l in landmarks 
                                    if self._is_numeric_landmark(l)])
            self._prop_landmark_count += len(landmarks) - \
                num_landmark_count
            self._num_landmark_count += num_landmark_count
            self._log_counter += 1
            if self._log_counter % self._log_frequency == 0:
                LOGGER.info(f'After {self._log_counter} calls, seen '
                            f'{self._prop_landmark_count} propositional '
                            f'landmarks and {self._num_landmark_count} numeric '
                            'landmarks')
        
        self._cache[cstate] = landmarks
        return landmarks

    def _is_numeric_landmark(self,
                             landmark: NumericLandmark) \
        -> bool:
        """Check if the given landmark is a numeric landmark.

        Args:
            landmark (NumericLandmark): The landmark to check.

        Returns:
            bool: True if the landmark is a numeric landmark, False otherwise.
        """
        return landmark.target != 1 or \
            any([t[1] != 1 for t in landmark.actions])
        