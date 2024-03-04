from asnets.prob_dom_meta import UnboundAction, UnboundComp, DomainMeta, \
    ProblemMeta
from asnets.network_modules import ActionModule, CompModule, FlntModule, \
    PropModule
from asnets.utils.prof_utils import can_profile
from asnets.utils.tf_utils import masked_softmax

import joblib
import numpy as np
import tensorflow as tf
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

NONLINEARITY = 'elu'

WeightDict = Dict[Any, Tuple[tf.Variable, tf.Variable]]


class PropNetworkWeights:
    """Manages weights for a domain-specific problem network. Those weights can
    then be used in problem-specific networks."""

    # WARNING: you need to change __{get,set}state__ if you change __init__ or
    # _make_weights()!

    def __init__(self,
                 dom_meta: DomainMeta,
                 hidden_sizes: Iterable[Tuple[int, int]],
                 extra_dim: int,
                 skip: bool,
                 use_fluents: bool,
                 use_comparisons: bool):
        """Initialises weights for a domain-specific problem network.

        Args:
            dom_meta (DomainMeta): DomainMeta object for the domain.
            hidden_sizes (Iterable[Tuple[int, int]]): List of (act layer size,
            prop layer size) pairs.
            extra_dim (int): Number of extra items included in the input vector
            for each action in the input layer.
            skip (bool): Whether to use skip connections.
            use_fluents (bool): Whether to use fluents. Fluent modules will have
            the same hidden size of prop modules in the same layer.
            use_comparisons (bool): Whether to use comparisons. Comparison
            modules will have the same hidden size of prop modules in the same
            layer.
        """
        self.dom_meta: DomainMeta = dom_meta
        self.hidden_sizes: List[Tuple[int, int]] = list(hidden_sizes)
        self.extra_dim: int = extra_dim
        self.skip: bool = skip
        self.use_fluents: bool = use_fluents
        self.use_comparisons: bool = use_comparisons
        self._make_weights()

    def __getstate__(self):
        """Pickle weights ourselves, since TF stuff is hard to pickle."""
        prop_weights_np = self._serialise_weight_list(self.prop_weights)
        flnt_weights_np = self._serialise_weight_list(self.flnt_weights)
        act_weights_np = self._serialise_weight_list(self.act_weights)
        comp_weights_np = self._serialise_weight_list(self.comp_weights)
        return {
            'dom_meta': self.dom_meta,
            'hidden_sizes': self.hidden_sizes,
            'prop_weights_np': prop_weights_np,
            'flnt_weights_np': flnt_weights_np,
            'act_weights_np': act_weights_np,
            'comp_weights_np': comp_weights_np,
            'extra_dim': self.extra_dim,
            'skip': self.skip,
            'use_fluents': self.use_fluents,
            'use_comparisons': self.use_comparisons
        }

    def __setstate__(self, state):
        """Unpickle weights"""
        self.dom_meta: DomainMeta = state['dom_meta']
        self.hidden_sizes: List[Tuple[int, int]] = state['hidden_sizes']
        self.extra_dim: int = state['extra_dim']
        # old network snapshots always had skip connections turned on
        self.skip: bool = state.get('skip', True)
        self.use_fluents: bool = state.get('use_fluents', False)
        self.use_comparisons: bool = state.get('use_comparisons', False)
        self._make_weights(
            state['prop_weights_np'],
            state.get('flnt_weights_np', None),
            state['act_weights_np'],
            state.get('comp_weights_np', None))

    @staticmethod
    def _serialise_weight_list(weight_list):
        # serialises a list of dicts, each mapping str -> (tensorflow weights,
        # ...)
        rv = []
        for d in weight_list:
            new_d = {}
            for k, v in d.items():
                new_d[k] = v
            rv.append(new_d)
        return rv

    @can_profile
    def _make_weights(self,
                      old_prop_weights=None,
                      old_flnt_weight=None,
                      old_act_weights=None,
                      old_comp_weights=None):
        # *_weights[i] is a dictionary mapping prop/flnt/act names to weights
        # for modules in the i-th proposition layer
        self.prop_weights = []
        self.flnt_weights = []
        self.act_weights = []
        self.comp_weights = []
        self.all_weights = []

        # TODO: constructing weights separately like this (and having
        # to restore with tf.const, etc.) is silly. Should store
        # parameters *purely* by name, and have code responsible for
        # automatically re-instantiating old weights (after network
        # construction) if they exist. TF offers several ways of doing
        # exactly that.

        for hid_idx, hid_sizes in enumerate(self.hidden_sizes):
            act_size, prop_size = hid_sizes

            # make action layer weights
            def act_in_size(unbound_act: UnboundAction, hid_idx: int) -> int:
                preds = self.dom_meta.rel_pred_names(unbound_act)
                funcs = self.dom_meta.rel_func_names(unbound_act)
                comps = self.dom_meta.rel_comps(unbound_act)
                if hid_idx == 0:
                    # first layer, the input is
                    # - whether the predicate is true and whether it is in the
                    #   goal
                    # - the value of the function and whether it is in the goal
                    in_size = len(preds) * 2 \
                        + (len(funcs) * 2 if self.use_fluents else 0) \
                        + (len(comps) * 2 if self.use_comparisons else 0) \
                        + self.extra_dim
                else:
                    # prop/func inputs + skip input from previous action layer
                    in_size = len(preds) * self.hidden_sizes[hid_idx - 1][1] \
                        + (len(funcs) * self.hidden_sizes[hid_idx - 1][1]
                           if self.use_fluents else 0) \
                        + (len(comps) * self.hidden_sizes[hid_idx - 1][1]
                           if self.use_comparisons else 0)
                    if self.skip:
                        in_size = in_size + self.hidden_sizes[hid_idx - 1][0]
                return in_size

            def act_name_pfx(unbound_act: UnboundAction, hid_idx: int) -> str:
                return 'hid_%d_act_%s' % (hid_idx, unbound_act.schema_name)

            self.act_weights.append(self._make_modules_weights(
                hid_idx, act_size, self.dom_meta.unbound_acts,
                old_act_weights, act_in_size, act_name_pfx))

            # make hidden proposition layer weights

            def prop_in_size(pred_name: str, hid_idx: int) -> int:
                rel_act_slots = self.dom_meta.rel_act_slots_of_pred(pred_name)
                # We should never end up with NO relevant actions & slots for a
                # predicate, else there's probably issue with domain.
                assert len(rel_act_slots) > 0, \
                    "no relevant actions for proposition %s" % pred_name

                in_size = len(rel_act_slots) * act_size
                if hid_idx > 0 and self.skip:
                    # skip connection from previous prop layer
                    in_size = in_size + self.hidden_sizes[hid_idx - 1][1]

                return in_size

            def prop_name_pfx(pred_name: str, hid_idx: int) -> str:
                return 'hid_%d_prop_%s' % (hid_idx, pred_name)

            self.prop_weights.append(self._make_modules_weights(
                hid_idx, prop_size, self.dom_meta.pred_names,
                old_prop_weights, prop_in_size, prop_name_pfx))

            # make hidden fluent layer weights
            if self.use_fluents:
                def flnt_in_size(func_name: str, hid_idx: int) -> int:
                    rel_act_slots = self.dom_meta.rel_act_slots_of_func(
                        func_name)
                    # We should never end up with NO relevant actions & slots
                    # for a function, else there's probably issue with domain.
                    assert len(rel_act_slots) > 0, \
                        "no relevant actions for function %s" % func_name

                    in_size = len(rel_act_slots) * act_size
                    if hid_idx > 0 and self.skip:
                        # skip connection from previous flnt layer
                        in_size = in_size + self.hidden_sizes[hid_idx - 1][1]
                    return in_size

                def flnt_name_pfx(func_name: str, hid_idx: int) -> str:
                    return 'hid_%d_flnt_%s' % (hid_idx, func_name)

                self.flnt_weights.append(self._make_modules_weights(
                    hid_idx, prop_size, self.dom_meta.func_names,
                    old_flnt_weight, flnt_in_size, flnt_name_pfx))

            if self.use_comparisons:
                def comp_in_size(unbound_comp: UnboundComp, hid_idx: int) \
                        -> int:
                    rel_act_slots = self.dom_meta.rel_act_slots_of_comp(
                        unbound_comp)
                    assert len(rel_act_slots) > 0, \
                        "no relevant actions for comparison %s" % unbound_comp

                    in_size = len(rel_act_slots) * act_size
                    if hid_idx > 0 and self.skip:
                        in_size = in_size + self.hidden_sizes[hid_idx - 1][1]
                    return in_size

                def comp_name_pfx(unbound_comp: UnboundComp, hid_idx: int) \
                        -> str:
                    return 'hid_%d_comp_%s' % (hid_idx, unbound_comp.comparison)

                self.comp_weights.append(self._make_modules_weights(
                    hid_idx, prop_size, self.dom_meta.unbound_comps,
                    old_comp_weights, comp_in_size, comp_name_pfx))

        # make final layer weights (action)

        def final_act_in_size(unbound_act: UnboundAction, hid_idx: int) -> int:
            preds = self.dom_meta.rel_pred_names(unbound_act)
            funcs = self.dom_meta.rel_func_names(unbound_act)
            comps = self.dom_meta.rel_comps(unbound_act)
            if not self.hidden_sizes:
                in_size = len(preds) * 2 \
                    + (len(funcs) * 2 if self.use_fluents else 0) \
                    + (len(comps) * 2 if self.use_comparisons else 0) \
                    + self.extra_dim
            else:
                in_size = len(preds) * self.hidden_sizes[-1][1] \
                    + (len(funcs) * self.hidden_sizes[-1][1]
                       if self.use_fluents else 0) \
                    + (len(comps) * self.hidden_sizes[-1][1]
                       if self.use_comparisons else 0)
                if self.skip:
                    in_size = in_size + self.hidden_sizes[-1][0]
            return in_size

        def final_act_name_pfx(unbound_act: UnboundAction, hid_idx: int) -> str:
            return 'final_act_%s' % unbound_act.schema_name

        self.act_weights.append(self._make_modules_weights(
            -1, 1, self.dom_meta.unbound_acts, old_act_weights,
            final_act_in_size, final_act_name_pfx))

    def _make_modules_weights(self,
                              hid_idx: int,
                              hid_size: int,
                              module_keys: Iterable[Any],
                              old_weights: Optional[Dict[int, WeightDict]],
                              in_size_func: Callable[[Any, int], int],
                              name_pfx_func: Callable[[Any, int], str]) -> WeightDict:
        """Make weights for a layer of modules of the same type.

        Args:
            hid_idx (int): The index of the hidden layer
            hid_size (int): The size of the hidden layer
            module_keys (Iterable[Any]): The keys of the modules in the layer
            old_weights (Optional[Dict[int, WeightDict]]): Old weights to
            try and reuse if not None
            in_size_func (Callable[[Any, int], int]): A function that takes
            in a module key and the hidden layer index and returns the input
            size of the module
            name_pfx_func (Callable[[Any, int], str]): A function that takes
            in a module key and the hidden layer index and returns the name
            prefix for the module

        Returns:
            WeightDict: The weights for the layer of modules, with keys being
            the module keys and values being the weights for the module
        """
        new_layer = {}
        for key in module_keys:
            in_size = in_size_func(key, hid_idx)
            name_pfx = name_pfx_func(key, hid_idx)
            if old_weights is not None:
                W = tf.Variable(
                    tf.constant_initializer(value=old_weights[hid_idx][key][0].numpy())(
                        shape=(in_size, hid_size)),
                    shape=(in_size, hid_size),
                    name=name_pfx + '/W',
                    trainable=True)
                b = tf.Variable(
                    tf.constant_initializer(value=old_weights[hid_idx][key][1].numpy())(
                        shape=(hid_size, )),
                    shape=(hid_size, ),
                    name=name_pfx + '/b',
                    trainable=True)
            else:
                W = tf.Variable(
                    tf.keras.initializers.VarianceScaling(
                        scale=1.0,
                        mode="fan_avg",
                        distribution="uniform")(shape=(in_size, hid_size)),
                    name=name_pfx + '/W',
                    trainable=True)
                b = tf.Variable(
                    tf.zeros_initializer()(shape=(hid_size, )),
                    name=name_pfx + '/b',
                    trainable=True)

            new_layer[key] = (W, b)
            self.all_weights.extend([W, b])

        return new_layer

    @can_profile
    def save(self, path):
        """Save a snapshot of the current network weights to the given path."""
        joblib.dump(self, path, compress=True)


class PropNetwork(tf.keras.layers.Layer):
    """ASNet.
    """

    def __init__(self,
                 weight_manager: PropNetworkWeights,
                 problem_meta: ProblemMeta,
                 dropout: float = 0.0,
                 debug: bool = False,
                 trainable: bool = True,
                 name: Optional[str] = None,
                 dtype=None,
                 dynamic: bool = False,
                 **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

        self._weight_manager = weight_manager
        self._prob_meta = problem_meta
        self._debug = debug
        # I tried ReLU, tanh, softplus, & leaky ReLU before settling on ELU for
        # best combination of numeric stability + sample efficiency
        self.nonlinearity = getattr(tf.nn, NONLINEARITY)
        # should we include skip connections?
        self.skip = self._weight_manager.skip
        # should we construct fluent modules?
        self.use_fluents = self._weight_manager.use_fluents
        # should we construct comparison modules?
        self.use_comparisons = self._weight_manager.use_comparisons

        self.dropout = dropout

        hidden_sizes = self._weight_manager.hidden_sizes
        dom_meta = self._weight_manager.dom_meta

        # this is useful for getting values from ALL action/proposition layers
        self.act_layers = []
        self.prop_layers = []
        self.flnt_layers = []
        self.comp_layers = []
        self.action_layer_input = {}
        self.weights_collection = []
        self.bias_collection = []

        # hidden layers
        for hid_idx, hid_sizes in enumerate(hidden_sizes):
            act_dict = {}
            for unbound_act in dom_meta.unbound_acts:

                weight, bias = self._weight_manager.act_weights[hid_idx][unbound_act]

                act_dict[unbound_act] = ActionModule(
                    unbound_act=unbound_act,
                    weight=weight,
                    bias=bias,
                    layer_num=hid_idx,
                    dom_meta=dom_meta,
                    prob_meta=self._prob_meta,
                    skip=self.skip,
                    nonlinearity=self.nonlinearity,
                    dropout=self.dropout)

                self.weights_collection.append(weight)
                self.bias_collection.append(bias)
            self.act_layers.append(act_dict)

            pred_dict = {}
            for pred_name in dom_meta.pred_names:

                weight, bias = self._weight_manager.prop_weights[hid_idx][pred_name]

                pred_dict[pred_name] = PropModule(
                    pred_name=pred_name,
                    weight=weight,
                    bias=bias,
                    layer_num=hid_idx,
                    dom_meta=dom_meta,
                    prob_meta=self._prob_meta,
                    skip=self.skip,
                    nonlinearity=self.nonlinearity,
                    dropout=self.dropout)

                self.weights_collection.append(weight)
                self.bias_collection.append(bias)
            self.prop_layers.append(pred_dict)

            if self.use_fluents:
                func_dict = {}
                for func_name in dom_meta.func_names:
                    weight, bias \
                        = self._weight_manager.flnt_weights[hid_idx][func_name]

                    func_dict[func_name] = FlntModule(
                        func_name=func_name,
                        weight=weight,
                        bias=bias,
                        layer_num=hid_idx,
                        dom_meta=dom_meta,
                        prob_meta=self._prob_meta,
                        skip=self.skip,
                        nonlinearity=self.nonlinearity,
                        dropout=self.dropout)

                    self.weights_collection.append(weight)
                    self.bias_collection.append(bias)
                self.flnt_layers.append(func_dict)

            if self.use_comparisons:
                comp_dict = {}
                for unbound_comp in dom_meta.unbound_comps:
                    weight, bias \
                        = self._weight_manager.comp_weights[hid_idx][unbound_comp]

                    comp_dict[unbound_comp] = CompModule(
                        unbound_comp=unbound_comp,
                        weight=weight,
                        bias=bias,
                        layer_num=hid_idx,
                        dom_meta=dom_meta,
                        prob_meta=self._prob_meta,
                        skip=self.skip,
                        nonlinearity=self.nonlinearity,
                        dropout=self.dropout)

                    self.weights_collection.append(weight)
                    self.bias_collection.append(bias)
                self.comp_layers.append(comp_dict)

        # final (action) layer
        finals = {}
        for unbound_act in dom_meta.unbound_acts:

            weight, bias = self._weight_manager.act_weights[len(
                hidden_sizes)][unbound_act]

            finals[unbound_act] = ActionModule(
                unbound_act=unbound_act,
                weight=weight,
                bias=bias,
                layer_num=len(hidden_sizes),
                dom_meta=dom_meta,
                prob_meta=self._prob_meta,
                skip=self.skip,
                nonlinearity=tf.identity,
                dropout=0.0)

            self.weights_collection.append(weight)
            self.bias_collection.append(bias)
        self.act_layers.append(finals)

    def _split_input(self,
                     obs: tf.Tensor,
                     boundeds: Iterable[Any],
                     unboundeds: Iterable[Any],
                     unbounded_to_bounded: Callable[[Any], Any]) \
            -> Tuple[Any, tf.Tensor]:
        """Splits an observation layer up into appropriate tensors grounded by
        variuos unbounded objects (predicates, functions, etc).

        Args:
            obs (tf.Tensor): The observation layer to split up.
            boundeds (Iterable[Any]): The bounded objects, should be in a fixed
            order.
            unboundeds (Iterable[Any]): The unbounded objects.
            unbounded_to_bounded (Callable[[Any], Any]): A function that maps
            an unbounded object to the bounded objects it is a schema for.

        Returns:
            Tuple[Any, tf.Tensor]: A dictionary mapping unbounded objects to
            their inputs.
        """
        rv = {}

        bounded_to_flat_input_idx = {
            bounded: idx
            for idx, bounded in enumerate(boundeds)
        }
        for unbounded in unboundeds:
            sub_boundeds = unbounded_to_bounded(unbounded)
            gather_inds = []
            for sub_bounded in sub_boundeds:
                to_look_up = bounded_to_flat_input_idx[sub_bounded]
                gather_inds.append(to_look_up)

            rv[unbounded] = tf.gather(obs,
                                      gather_inds,
                                      axis=1,
                                      name=f'split_input/{unbounded}')

        return rv

    def _split_extra(self, extra_data):
        """Sometimes we also have input data which goes straight to the
        network. We need to split this up into an unbound action->tensor
        dictionary just like the rest."""
        prob_meta = self._prob_meta
        out_dict = {}
        for unbound_act in prob_meta.domain.unbound_acts:
            ground_acts = prob_meta.schema_to_acts(unbound_act)
            sorted_acts = sorted(ground_acts,
                                 key=prob_meta.act_to_schema_subtensor_ind)
            if len(sorted_acts) == 0:
                # FIXME: make this message scarier
                print("no actions for schema %s?" % unbound_act.schema_name)
            # these are the indices which we must read and concatenate
            tensor_inds = [
                # TODO: make this linear-time (or linearithmic) by using a dict
                prob_meta.bound_acts_ordered.index(act) for act in sorted_acts
            ]

            out_dict[unbound_act] = tf.gather(extra_data,
                                              tensor_inds,
                                              axis=1,
                                              name='split_extra/' +
                                              unbound_act.schema_name)

        return out_dict

    def call(self, inputs, *args, **kwargs):
        # input vector spec:
        #
        # |<--num_acts-->|<--k*num_acts-->|<--num_props-->|<--num_flnts-->|<--num_comps-->|
        # | action mask  |  action data   | propositions  |    fluents    |  comparisons  |
        #
        # 1) `action_mask` tells us whether actions are enabled
        # 2) `action_data` is passed straight to action modules
        # 3) `propositions` tells us what is and isn't true
        # 4) `fluents` tells us what the values of fluents are, only exists if
        #    args.use_fluents is True, and only used if self.use_fluents is
        #    True
        # 5) `comparisons` tells us what comparisons are satisfied, only exists
        #    if args.use_comparisons is True, and only used if
        #    self.use_comparisons is True
        super().call(inputs, *args, **kwargs)

        hidden_sizes = self._weight_manager.hidden_sizes
        dom_meta = self._weight_manager.dom_meta
        prob_meta = self._prob_meta

        mask_size = prob_meta.num_acts
        extra_data_dim = self._weight_manager.extra_dim
        extra_size = extra_data_dim * prob_meta.num_acts

        # split up the input into its various components
        cur_index = 0
        act_mask = inputs[:, cur_index:cur_index + mask_size]
        cur_index += mask_size
        aux_data = inputs[:, cur_index:cur_index + extra_size]
        cur_index += extra_size
        prop_truths = inputs[:, cur_index:cur_index + prob_meta.num_props]
        cur_index += prob_meta.num_props

        flnt_values = None
        if self.use_fluents:
            flnt_values = inputs[:, cur_index:cur_index + prob_meta.num_flnts]
            cur_index += prob_meta.num_flnts

        comp_truths = None
        if self.use_comparisons:
            comp_truths = inputs[:, cur_index:cur_index + prob_meta.num_comps]
            cur_index += prob_meta.num_comps

        def merge_with_goal_vec(in_vec, goal_vec):
            # FIXME: it doesn't make sense to mess with goal vectors here; that
            # should be ActionDataGenerator's job, or whatever. Should be
            # passed in as part of network input, not fixed as  TF constant!
            reshaped_in_vec = in_vec[:, :, None]
            tf_goals = tf.constant(goal_vec)[None, :, None]
            batch_size = tf.shape(input=reshaped_in_vec)[0]
            tf_goals_broad = tf.tile(tf_goals, (batch_size, 1, 1))
            l_obs = tf.concat([reshaped_in_vec, tf_goals_broad], axis=2)
            return l_obs

        # [None, num_props, 2]
        l_obs_prop = merge_with_goal_vec(
            prop_truths,
            [float(prop in prob_meta.goal_props)
             for prop in prob_meta.bound_props_ordered]
        )
        pred_dict = self._split_input(
            l_obs_prop,
            prob_meta.bound_props_ordered,
            prob_meta.domain.pred_names,
            prob_meta.pred_to_props)

        func_dict = None
        if self.use_fluents:
            # [None, num_flnts, 2]
            l_obs_flnt = merge_with_goal_vec(
                flnt_values,
                [float(flnt in prob_meta.goal_flnts)
                 for flnt in prob_meta.bound_flnts_ordered]
            )
            func_dict = self._split_input(
                l_obs_flnt,
                prob_meta.bound_flnts_ordered,
                prob_meta.domain.func_names,
                prob_meta.func_to_flnts)

        comp_dict = None
        if self.use_comparisons:
            # [None, num_comps, 2]
            # FIXME what does it mean for a comparison to be in the goal,
            # what about the actual goal comparisons which don't exist
            l_obs_comp = merge_with_goal_vec(
                comp_truths,
                [float(False) for _ in prob_meta.bound_comps_ordered]
            )
            comp_dict = self._split_input(
                l_obs_comp,
                prob_meta.bound_comps_ordered,
                prob_meta.domain.unbound_comps,
                prob_meta.unbound_comp_to_comps)

        if extra_data_dim > 0:
            # [None, num_actions, extra_dimension]
            out_shape = (-1, prob_meta.num_acts, extra_data_dim)
            l_act_extra = tf.reshape(aux_data, out_shape)
            extra_dict = self._split_extra(l_act_extra)
        else:
            extra_dict = None

        # this is useful for getting values from ALL action/proposition layers
        self.act_layers_outcome = []
        self.prop_layers_outcome = []
        self.flnt_layers_outcome = []
        self.copm_layers_outcome = []

        # Input + hidden layers
        prev_act_dict = {}
        prev_pred_dict = {}
        prev_func_dict = {}
        prev_comp_dict = {}
        for hid_idx, hid_sizes in enumerate(hidden_sizes):

            act_outcome_dict = {}
            act_module_dict = self.act_layers[hid_idx]
            for unbound_act in dom_meta.unbound_acts:
                act_outcome_dict[unbound_act] = act_module_dict[unbound_act].forward(
                    prev_pred=pred_dict,
                    prev_func=func_dict,
                    prev_comp=comp_dict,
                    extra_input=extra_dict,
                    prev_act=prev_act_dict.get(unbound_act, None))

            self.act_layers_outcome.append(act_outcome_dict)
            prev_act_dict = act_outcome_dict

            pred_dict = {}
            pred_module_dict = self.prop_layers[hid_idx]
            for pred_name in dom_meta.pred_names:
                pred_dict[pred_name] = pred_module_dict[pred_name].forward(
                    prev_act=act_outcome_dict,
                    prev_self=prev_pred_dict.get(pred_name, None))

            self.prop_layers_outcome.append(pred_dict)
            prev_pred_dict = pred_dict

            if self.use_fluents:
                func_dict = {}
                func_module_dict = self.flnt_layers[hid_idx]
                for func_name in dom_meta.func_names:
                    func_dict[func_name] = func_module_dict[func_name].forward(
                        prev_act=act_outcome_dict,
                        prev_self=prev_func_dict.get(func_name, None))

                self.flnt_layers_outcome.append(func_dict)
                prev_func_dict = func_dict

            if self.use_comparisons:
                comp_dict = {}
                comp_module_dict = self.comp_layers[hid_idx]
                for unbound_comp in dom_meta.unbound_comps:
                    comp_dict[unbound_comp] = comp_module_dict[unbound_comp].forward(
                        prev_act=act_outcome_dict,
                        prev_self=prev_comp_dict.get(unbound_comp, None))

                self.copm_layers_outcome.append(comp_dict)
                prev_comp_dict = comp_dict

        # final (action) layer
        finals = {}
        act_module_dict = self.act_layers[len(hidden_sizes)]
        for unbound_act in dom_meta.unbound_acts:
            finals[unbound_act] = act_module_dict[unbound_act].forward(
                prev_pred=pred_dict,
                prev_func=func_dict,
                prev_comp=comp_dict,
                extra_input=extra_dict,
                prev_act=prev_act_dict.get(unbound_act, None))

        l_pre_softmax = _merge_finals(prob_meta, finals)
        # voila!
        return masked_softmax(l_pre_softmax, act_mask)


def _merge_finals(prob_meta, final_acts):
    # we make a huge tensor of actions that we'll have to reorder
    sorted_final_acts = sorted(final_acts.items(), key=lambda t: t[0])
    # also get some metadata about which positions in tensor correspond to
    # which schemas
    unbound_to_super_ind = {
        t[0]: idx
        for idx, t in enumerate(sorted_final_acts)
    }
    # indiv_sizes[i] is the number of bound acts associated with the i-th
    # schema
    indiv_sizes = [
        len(prob_meta.schema_to_acts(ub)) for ub, _ in sorted_final_acts
    ]
    # cumul_sizes[i] is the sum of the number of ground actions associated
    # with each action schema *before* the i-th schema
    cumul_sizes = np.cumsum([0] + indiv_sizes)
    # this stores indices that we have to look up
    gather_list = []
    for ground_act in prob_meta.bound_acts_ordered:
        subact_ind = prob_meta.act_to_schema_subtensor_ind(ground_act)
        superact_ind = unbound_to_super_ind[ground_act.prototype]
        actual_ind = cumul_sizes[superact_ind] + subact_ind
        assert 0 <= actual_ind < prob_meta.num_acts, \
            "action index %d for %r out of range [0, %d)" \
            % (actual_ind, ground_act, prob_meta.num_acts)
        gather_list.append(actual_ind)

    # now let's actually build and reorder our huge tensor of action
    # selection probs
    cat_super_acts = tf.concat([t[1] for t in sorted_final_acts],
                               axis=1,
                               name='merge_finals/cat')
    rv = tf.gather(cat_super_acts[:, :, 0],
                   np.array(gather_list),
                   axis=1,
                   name='merge_finals/reorder')

    return rv
