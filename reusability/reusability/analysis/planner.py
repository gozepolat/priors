from collections import defaultdict
from typing import List
import numpy as np

from reusability import common
from reusability.common import get_uuid


def _repeat_in_array(value: {int, float, List}, length) -> List:
    """Repeat int value in an array"""
    if isinstance(value, List):
        return value

    return [value] * length


def get_conv_output_hw(source_hw, kernel_size, stride, padding=1, dilation=1):
    """Given [D], H, W, get [D_out], H_out, W_out after conv"""
    x = source_hw
    length = len(x)
    kernel_size = _repeat_in_array(kernel_size, length)
    stride = _repeat_in_array(stride, length)
    padding = _repeat_in_array(padding, length)
    dilation = _repeat_in_array(dilation, length)

    return tuple(int((x_in + 2 * padding[i] - dilation[i] *
                      (kernel_size[i] - 1) - 1) / stride[i] + 1)
                 for i, x_in in enumerate(x))


def get_elementwise_op_shape(first_shape, second_shape):
    """Get the output shape for elementwise operation"""
    ndims = max(len(first_shape), len(second_shape))
    new_first_shape = (ndims - len(first_shape)) * [1] + list(first_shape)
    new_second_shape = (ndims - len(second_shape)) * [1] + list(second_shape)

    out_shape = []
    for first, second in zip(new_first_shape, new_second_shape):
        if first != second:
            if min(first, second) != 1:
                raise ValueError(f"Incompatible mul: {first_shape} * {second_shape}")
            out_shape.append(max(first, second))
            continue

        out_shape.append(first)

    return tuple(out_shape)


def make_next_layer_output(prev_layer, kwargs):
    """Create layer output features based on layer type"""
    layer_type = kwargs['layer_type']
    if layer_type == "global_pool":
        # In case of a graph with multiple global pools and same number of out_channels
        name = kwargs.get("name", f"global_pool_{get_uuid()}")
        names = make_feature_names(name, (prev_layer.shape[-1],))
        return np.reshape(names, (1, 1, len(names)))

    prefix = kwargs.get('prefix', 'z')
    name = kwargs['name']
    suffix = kwargs.get('suffix', '')

    # with broadcasting logic
    if layer_type == "elementwise_op":
        second_shape = kwargs['second_element'].shape
        op_shape = get_elementwise_op_shape(second_shape, prev_layer.shape)
        return make_feature_names(name, op_shape,
                                  prefix=prefix, suffix=suffix)

    if layer_type == "batch_norm":
        return make_feature_names(name, prev_layer.shape,
                                  prefix=prefix, suffix=suffix)
    if layer_type == "dense":
        names = make_feature_names(name, (kwargs['output_filters'],),
                                   prefix=prefix, suffix=suffix)
        return np.reshape(names, (1, 1, len(names)))

    if layer_type not in {"conv", "depthwise_conv", "local_pool"}:
        raise NotImplementedError("Only global_pool, local_pool, dense or conv types are supported")

    if layer_type != "local_pool":
        common.TOTAL_CONV_LAYERS += 1
    else:
        out_filters = kwargs.get('output_filters', None)
        if out_filters is not None:
            assert out_filters == prev_layer.shape[-1], "Source and target channels for local_pool do not match"

        kwargs['output_filters'] = prev_layer.shape[-1]

    dilation = kwargs.get('dilation', 1)
    kernel_size = kwargs['kernel_size']
    padding = kwargs.get('padding', 'same')

    hw = prev_layer.shape[:2]

    if padding == 'same':
        padding = kernel_size // 2
    else:
        raise NotImplementedError(f"Padding other than same ({padding}) not supported")

    output_hw = get_conv_output_hw(hw,
                                   kernel_size,
                                   kwargs['strides'],
                                   padding=padding,
                                   dilation=dilation)

    output_shape = (*output_hw, kwargs['output_filters'])

    return make_feature_names(name, output_shape, prefix, suffix)


def push_step(plan, task, prev_layer):
    """Add a new step to the plan and prep the output features"""
    fn, kwargs = task

    kwargs['source'] = prev_layer
    if kwargs.get("target", None) is None:
        kwargs['target'] = make_next_layer_output(prev_layer, kwargs)

    plan.append((fn, kwargs))
    return kwargs['target']


def make_feature_names(name, shape, prefix='z', suffix=''):
    """Create a numpy array of feature names of specified shape, likely [h, w, c] or [c]"""
    if prefix[0] != 'z':
        print("WARNING! Prefix name does not start with z in features")
        prefix = f"z{prefix}"

    if suffix:
        suffix = f"_{suffix}"

    if name:
        name = f"_{name}"

    if len(shape) == 1:
        return np.array([f"{prefix}{name}_c{c}{suffix}" for c in range(shape[0])])

    if len(shape) == 2:
        return np.stack([
            [f"{prefix}{name}_w{w}c{c}{suffix}" for c in range(shape[1])]
            for w in range(shape[0])
        ])

    assert len(shape) == 3, "Shape can only have length 1, 2, or 3"

    return np.stack([
        [
            [f"{prefix}{name}_h{h}w{w}c{c}{suffix}" for c in range(shape[2])]
            for w in range(shape[1])
        ]
        for h in range(shape[0])
    ])


def collect_stats_from_plan(plan, stats=None):
    """Get the total number of contexts for each learnable parameter from the graph plan"""
    if stats is None:
        stats = defaultdict(int)

    while len(plan) > 0:
        fn, kwargs = plan.pop()
        print(fn, kwargs['name'], kwargs.get('suffix', 'unknown'))
        fn(stats, **kwargs)

    return stats
