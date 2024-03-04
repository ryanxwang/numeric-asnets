import abc
import tensorflow as tf
from typing import Any, Callable, Dict, List, Optional, Tuple

from asnets.ops.asnet_ops import multi_gather_concat, multi_pool_concat
from asnets.prob_dom_meta import BoundAction, BoundComp, BoundFlnt, BoundProp, \
    DomainMeta, ProblemMeta, UnboundAction, UnboundComp, UnboundFlnt, \
    UnboundProp


class NetworkModule(abc.ABC):
    """ABC for a network module."""

    def __init__(self,
                 weight: tf.Variable,
                 bias: tf.Variable,
                 layer_num: int,
                 dom_meta: DomainMeta,
                 prob_meta: ProblemMeta,
                 skip: bool,
                 *,
                 nonlinearity: Optional[Callable] = None,
                 dropout: float = 0.0):
        self.W = weight
        self.b = bias
        self.layer_num = layer_num
        self.dom_meta = dom_meta
        self.prob_meta = prob_meta
        self.skip = skip
        self.nonlinearity = nonlinearity
        self.dropout = dropout
        self.name_pfx = None
        self.conv_input = None

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def compute_output(self, conv_input):
        with tf.name_scope(self.name_pfx + '/conv'):
            conv_result = _apply_conv_matmul(conv_input, self.W)
            rv = self.nonlinearity(conv_result + self.b[None, :])

        if self.dropout > 0:
            rv = tf.nn.dropout(rv, self.dropout, name=self.name_pfx + '/drop')

        return rv


class ActionModule(NetworkModule):
    """A network module for a single action schema."""

    def __init__(self,
                 unbound_act: UnboundAction,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.unbound_act = unbound_act
        self.name_pfx = f'act_mod_{unbound_act.schema_name}_{self.layer_num}'

    def forward(self,
                prev_pred: Dict[UnboundProp, tf.Tensor],
                *,
                prev_func: Dict[UnboundFlnt, tf.Tensor] = None,
                prev_comp: Dict[UnboundComp, tf.Tensor] = None,
                extra_input: Optional[Dict[UnboundAction, tf.Tensor]] = None,
                prev_act: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Forward pass for this action module.

        Args:
            prev_pred (Dict[UnboundProp, tf.Tensor]): Predicate input to this
            module. In the first layer, this is the predicate input to the
            network. In later layers, this is the activation of the previous
            proposition layer.

            prev_func (Dict[UnboundFlnt, tf.Tensor]): Fluent input to this
            module. In the first layer, this is the fluent input to the network.
            In later layers, this is the activation of the previous fluent
            layer.

            extra_input (Optional[Dict[UnboundAction, tf.Tensor]], optional): 
            Extra input to the network. For now this is only used in the first
            layer, where extra_input is the heuristic features for each action
            schema.

            prev_act (Optional[tf.Tensor], optional): Activation of the same
            module in the previuos layer, used for skip connections. Defaults to
            None.

        Returns:
            tf.Tensor: Activation of all action modules corresponding to this
            action schema.
        """
        use_flnt = prev_func is not None
        use_comp = prev_comp is not None

        # 1. Determine the relevant indices in the input dictionaries to gather
        #    input for each ground action of this action schema.

        # *_index_spec describes how to pick out the needed input from the
        # *_pred dictionary. Each pair (tensor_idx, pools) in the list specifies
        # that the tensor at index tensor_idx (tensor for a predicate or fluent)
        # should be gathered from the pools of indices in pools. For action
        # modules every pool should have only one element. Each pool corresponds
        # to a ground action.

        pred_to_idx, pred_list = _sort_inputs(prev_pred)
        pred_index_spec: List[Tuple[int, List[List[int]]]] = []
        for rel_pred_idx, rel_pred in enumerate(
                self.dom_meta.rel_pred_names(self.unbound_act)):

            # FIXME: this code is quite repetitive, can we factor it out?
            pools = []
            for ground_act in self.prob_meta.schema_to_acts(self.unbound_act):
                # we are looking at the rel_pred_idx-th relevant proposition
                bound_prop = self.prob_meta.rel_props(ground_act)[rel_pred_idx]
                prop_idx = self.prob_meta.prop_to_pred_subtensor_ind(
                    bound_prop)

                pools.append([prop_idx])

            tensor_idx = pred_to_idx[rel_pred]
            pred_index_spec.append((tensor_idx, pools))

        if use_flnt:
            func_to_idx, func_list = _sort_inputs(prev_func)
            func_index_spec: List[Tuple[int, List[List[int]]]] = []
            for rel_func_idx, rel_func in enumerate(
                    self.dom_meta.rel_func_names(self.unbound_act)):

                pools = []
                for ground_act in self.prob_meta.schema_to_acts(self.unbound_act):
                    # we are looking at the rel_func_idx-th relevant fluent
                    bound_flnt = self.prob_meta.rel_flnts(ground_act)[
                        rel_func_idx]
                    flnt_idx = self.prob_meta.flnt_to_func_subtensor_ind(
                        bound_flnt)

                    pools.append([flnt_idx])

                tensor_idx = func_to_idx[rel_func]
                func_index_spec.append((tensor_idx, pools))

        if use_comp:
            comp_to_idx, comp_list = _sort_inputs(prev_comp)
            comp_index_spec: List[Tuple[int, List[List[int]]]] = []
            for rel_comp_idx, rel_comp in enumerate(
                    self.dom_meta.rel_comps(self.unbound_act)):

                pools = []
                for ground_act in self.prob_meta.schema_to_acts(self.unbound_act):
                    # we are looking at the rel_comp_idx-th relevant comparison
                    bound_comp = self.prob_meta.rel_comps(ground_act)[
                        rel_comp_idx]
                    comp_idx = self.prob_meta.comp_to_unbound_comp_subtensor_ind(
                        bound_comp)

                    pools.append([comp_idx])

                tensor_idx = comp_to_idx[rel_comp]
                comp_index_spec.append((tensor_idx, pools))

        # 2. Prepare the extra channels (heuristic features, skip connections)
        extra_chans = []
        if self.layer_num == 0 and extra_input is not None:  # first layer
            extra_chans.append(extra_input[self.unbound_act])

        if self.layer_num > 0 and self.skip:
            assert prev_act is not None, \
                f'{self.name_pfx} requires prev_act for skip connection'
            extra_chans.append(prev_act)
        elif self.layer_num == 0 and self.skip:
            assert prev_act is None, \
                f'{self.name_pfx} is the first layer and somehow has prev_act'

        # 3. Run the custom multi-gather-concat operation
        with tf.name_scope(self.name_pfx + '/mgc'):
            mgc_inputs = []
            mgc_elem_indices = []

            for tensor_idx, pools in pred_index_spec:
                mgc_inputs.append(pred_list[tensor_idx])
                mgc_elem_indices.append(tf.constant(
                    [p[0] for p in pools], dtype=tf.int64))

            if use_flnt:
                for tensor_idx, pools in func_index_spec:
                    mgc_inputs.append(func_list[tensor_idx])
                    mgc_elem_indices.append(tf.constant(
                        [p[0] for p in pools], dtype=tf.int64))

            if use_comp:
                for tensor_idx, pools in comp_index_spec:
                    mgc_inputs.append(comp_list[tensor_idx])
                    mgc_elem_indices.append(tf.constant(
                        [p[0] for p in pools], dtype=tf.int64))

            for extra_chan in extra_chans:
                mgc_inputs.append(extra_chan)
                extra_chan_width = tf.cast(tf.shape(extra_chan)[1], tf.int64)
                mgc_elem_indices.append(
                    tf.range(extra_chan_width, dtype=tf.int64))
                mgc_elem_indices[-1].set_shape(extra_chan.shape[1])

        # 4. Profit!
        return self.compute_output(multi_gather_concat(
            mgc_inputs, mgc_elem_indices))


class PropLayerModule(NetworkModule):
    # Terrible name
    """A network module for something in the proposition layer, base class for
    actual PropModule, FlntModule, and etc."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    @abc.abstractmethod
    def rel_act_slots_of_lifted(self) -> List[Tuple[UnboundAction, int]]:
        """Return the slots of the lifted action schema that are related to the
        given lifted object (predicate, function, etc.). Should just be a
        wrapper for something like DomainMeta.rel_act_slots_of_pred, with the
        input being fixed to the lifted object corresponding to this module.

        Returns:
            List[Tuple[UnboundAction, int]]: A list of pairs (unbound action,
            slot index) that are related to the given lifted object.
        """
        pass

    @property
    @abc.abstractmethod
    def ground_of_lifted(self) -> List[Any]:
        """Return the ground objects corresponding to the lifted object
        corresponding to this module. Should just be a wrapper for something
        like ProblemMeta.pred_to_props, with the input being fixed to the lifted
        object corresponding to this module.

        Returns:
            List[Any]: A list of ground objects corresponding to the lifted
            object corresponding to this module.
        """
        pass

    @abc.abstractmethod
    def rel_act_slots_of_ground(self, ground: Any) \
            -> List[Tuple[UnboundAction, int, List[BoundAction]]]:
        """Return the relevant unbound action, slots, and bound actions for the
        given ground object. Should just be a wrapper for something like
        ProblemMeta.rel_act_slots_of_prop.

        Args:
            ground (Any): The ground object, should be in the list returned by
            ground_of_lifted.

        Returns:
            List[Tuple[UnboundAction, int, List[BoundAction]]]: The relevant
            unbound action, slots, and bound actions for the given ground
            object.
        """
        pass

    def forward(self,
                prev_act: Dict[Any, tf.Tensor],
                *,
                prev_self: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Forward pass for this module.

        Args:
            prev_act (Dict[Any, tf.Tensor]): Action input to this module.
            prev_self (Optional[tf.Tensor], optional): Activation of the same
            module in the previous layer, used for skip connections. Defaults to
            None.

        Returns:
            tf.Tensor: Activation of all modules corresponding to this module,
            e.g. all proposition modules corresponding to a single predicate.
        """
        act_to_idx, act_list = _sort_inputs(prev_act)

        # 1. Determine the relevant indices in the input dictionaries to gather
        #    input for each proposition of this predicate.

        # *_index_spec describes how to pick out the needed input from the
        # *_pred dictionary. Each pair (tensor_idx, pools) in the list specifies
        # that the tensor at index tensor_idx (tensor for an action) should be
        # gathered from the pools of indices in pools.

        act_index_spec: List[Tuple[int, List[List[int]]]] = []
        for unbound_act, slot in self.rel_act_slots_of_lifted:
            pools = []
            for ground in self.ground_of_lifted:
                # every related ground action for this slot.
                ground_acts = [
                    candidate_act
                    for candidate_act_schema, candidate_slot, candidate_acts
                    in self.rel_act_slots_of_ground(ground)
                    for candidate_act in candidate_acts
                    if candidate_act_schema == unbound_act
                    and candidate_slot == slot
                ]  # Such sql, much relational. This doesn't seem efficient.
                pools.append([
                    self.prob_meta.act_to_schema_subtensor_ind(ground_act)
                    for ground_act in ground_acts])

            tensor_idx = act_to_idx[unbound_act]
            act_index_spec.append((tensor_idx, pools))

        # 2. Prepare the extra channels (heuristic features, skip connections)
        extra_chans = []
        if self.layer_num > 0 and self.skip:
            assert prev_self is not None, \
                f'{self.name_pfx} requires prev_self for skip connection'
            extra_chans.append(prev_self)
        elif self.layer_num == 0 and self.skip:
            assert prev_self is None, \
                f'{self.name_pfx} is the first layer and somehow has prev_self'

        # 3. Run the custom multi-gather-concat operation
        with tf.name_scope(self.name_pfx + '/mpc'):
            mpc_inputs = []
            mpc_ragged_pools = []

            for tensor_idx, pools in act_index_spec:
                mpc_inputs.append(act_list[tensor_idx])
                ragged_pool = tf.cast(tf.RaggedTensor.from_row_lengths(
                    sum(pools, []), [len(p) for p in pools]
                ), dtype=tf.int64)
                mpc_ragged_pools.append(ragged_pool)

            assert self.nonlinearity is tf.nn.elu, \
                'Minimum value of -1 is dependent on using elu'
            min_value = -1.0
            conv_input = multi_pool_concat(
                mpc_inputs, mpc_ragged_pools, min_value)

            if extra_chans:
                # TODO: Test if it would be better (faster) to do this in
                # multi_pool_concat
                conv_input = tf.concat([conv_input, *extra_chans], axis=2)

        # 4. Profit!
        return self.compute_output(conv_input)


class PropModule(PropLayerModule):
    """A network module for a single predicate."""

    def __init__(self,
                 pred_name: str,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.pred_name = pred_name
        self.name_pfx = f'prop_mod_{self.pred_name}_{self.layer_num}'

    @property
    def rel_act_slots_of_lifted(self) -> List[Tuple[UnboundAction, int]]:
        return self.dom_meta.rel_act_slots_of_pred(self.pred_name)

    @property
    def ground_of_lifted(self) -> List[BoundProp]:
        return self.prob_meta.pred_to_props(self.pred_name)

    def rel_act_slots_of_ground(self, ground: BoundProp) \
            -> List[Tuple[UnboundAction, int, List[BoundAction]]]:
        return self.prob_meta.rel_act_slots_of_prop(ground)


class FlntModule(PropLayerModule):
    """A network module for a single fluent."""

    def __init__(self,
                 func_name: str,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.func_name = func_name
        self.name_pfx = f'flnt_mod_{self.func_name}_{self.layer_num}'

    @property
    def rel_act_slots_of_lifted(self) -> List[Tuple[UnboundAction, int]]:
        return self.dom_meta.rel_act_slots_of_func(self.func_name)

    @property
    def ground_of_lifted(self) -> List[BoundFlnt]:
        return self.prob_meta.func_to_flnts(self.func_name)

    def rel_act_slots_of_ground(self, ground: BoundFlnt) \
            -> List[Tuple[UnboundAction, int, List[BoundAction]]]:
        return self.prob_meta.rel_act_slots_of_flnt(ground)


class CompModule(PropLayerModule):
    """A network module for a single comparison."""

    def __init__(self,
                 unbound_comp: UnboundComp,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.unbound_comp = unbound_comp
        self.name_pfx = f'comp_mod_{self.unbound_comp}_{self.layer_num}'

    @property
    def rel_act_slots_of_lifted(self) -> List[Tuple[UnboundAction, int]]:
        return self.dom_meta.rel_act_slots_of_comp(self.unbound_comp)

    @property
    def ground_of_lifted(self) -> List[BoundComp]:
        return self.prob_meta.unbound_comp_to_comps(self.unbound_comp)

    def rel_act_slots_of_ground(self, ground: BoundComp) \
            -> List[Tuple[UnboundAction, int, List[BoundAction]]]:
        return self.prob_meta.rel_act_slots_of_comp(ground)


def _sort_inputs(input_dict: Dict[Any, tf.Tensor]) \
        -> Tuple[Dict[Any, int], List[tf.Tensor]]:
    """Decompose a dictionary of inputs into a list of tensors and a mapping
    from the original keys to the indices of the tensors in the list. The
    ordering used for the indexing is arbitrary.

    Args:
        input_dict: A dictionary mapping keys to tensors.

    Returns:
        Dict[Any, int]: A mapping from keys to indices.
        List[tf.Tensor]: A list of tensors.
    """
    input_items_sorted = sorted(input_dict.items(), key=lambda p: p[0])
    key_to_tensor_idx = {
        item[0]: idx
        for idx, item in enumerate(input_items_sorted)
    }
    input_list = [tensor for _, tensor in input_items_sorted]
    return key_to_tensor_idx, input_list


def _apply_conv_matmul(conv_input: tf.Tensor, W: tf.Tensor) -> tf.Tensor:
    reshaped = tf.reshape(conv_input, (-1, conv_input.shape[2]))
    conv_result_reshaped = tf.matmul(reshaped, W)
    conv_shape = tf.shape(input=conv_input)
    batch_size = conv_shape[0]
    # HACK: if conv_input.shape[1] is not Dimension(None) (i.e. if it's
    # known) then I want to keep that b/c it will help shape inference;
    # otherwise I want to use conv_shape[1], which will make the reshape
    # succeed. It turns out the best way I can see to compare
    # conv_input.shape[1] to Dimension(None) is to abuse comparison by
    # checking whether conv_input.shape[1] >= 0 returns None or True (!!).
    # This is stupid, but I can't see a better way of doing it.
    width = conv_shape[1] if (conv_input.shape[1] >= 0) is None \
        else conv_input.shape[1]
    out_shape = (batch_size, width, W.shape[1])
    conv_result = tf.reshape(conv_result_reshaped, out_shape)
    return conv_result
