import logging
import uuid
from enum import IntEnum
import numpy as np

from reusability.graph.dag import Node


class Verbosity(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2


# optimize for models where each channel has the same count
CHANNEL_OPTIMIZE = True

TOTAL_CONV_LAYERS = 0
MAX_NUM_PROCESSES = 12
DEBUG_MODE = False
USE_BIAS = True
DAG_VERBOSITY = Verbosity.LOW
EPSILON = 1e-12
MIN_FLOAT = 1e-323


def get_uuid():
    return uuid.uuid4().hex


def get_nth_from_tuple(tuple_list, n):
    return [_tuple[n] for _tuple in tuple_list]


def flatten(items):
    """Flatten a given nested list"""
    if not isinstance(items, list):
        return [items]

    flattened = []
    for item in items:
        flattened.extend(flatten(item))

    return flattened


def get_all_non_empty_subsets(items: list) -> list:
    """Return a list of all subsets of given items except for the empty set

    size is always pow(2,n) - 1 => len(get_all_non_empty_subsets([2, 3, 4, 5, 6, 7])) == 63"""
    if not items:
        return []

    subsets = []
    while len(items) > 0:
        last = items.pop()
        subsets_without_items = get_all_non_empty_subsets(items)
        subsets.append([last])
        subsets.extend(subsets_without_items)
        subsets.extend([[last] + subset for subset in subsets_without_items if subset])

    return subsets


def get_quadratic_roots(a, b, c):
    discriminant = (b ** 2 - 4 * a * c) ** .5
    return .5 * (-b + discriminant) / a, .5 * (-b - discriminant) / a


def debug(*msg):
    logging.debug(*msg)


def reshape(tensor, new_shape):
    if isinstance(tensor, np.ndarray):
        return np.reshape(tensor, new_shape)

    # TODO discard tf to avoid dependencies?
    import tensorflow as tf
    return tf.reshape(tensor, new_shape)


def get_broadcast_shape(tensor, out_shape):
    broadcast_shape = tuple([1] * (len(out_shape) - len(tensor.shape)) + list(tensor.shape))

    if np.any([min(i, j) != 1 and i != j for i, j in zip(broadcast_shape, out_shape)]):
        raise ValueError(f"tensor with shape: {tensor.shape} can not be broadcasted for output {out_shape} ")

    return broadcast_shape


def expand_dims_for_broadcast(tensor, out_shape):
    new_shape = get_broadcast_shape(tensor, out_shape)
    return reshape(tensor, new_shape)


def get_safe_access_index(tensor, dim_indices):
    assert len(tensor.shape) == len(dim_indices)

    safe_index = tuple([min(i, s - 1) for i, s in zip(dim_indices, tensor.shape)])
    return safe_index


def standardize_name(name, ignore_suffix):
    """Get a canonical name for collecting statistics

    :param name: unique name with a potential suffix
    :param ignore_suffix: if True get rid of the suffix from  name='prefix+suffix'
    :return: prefix or name
    """
    if not isinstance(name, str):
        name = name.name

    if is_feature_node(name):
        return name

    prefix = name.split('+')[0]
    return prefix if ignore_suffix else name


def is_feature_node(node):
    if not isinstance(node, str):
        node = node.name
    return node[0] == 'z'


def is_weight_node(node):
    if not isinstance(node, str):
        node = node.name
    return node[0] == 'y'


class MetricDict:
    def __init__(self, data):
        self.data = data
        self.counts = {key: 1 for key in data}

    def __setitem__(self, key, value):
        if key not in self.data:
            self.data[key] = float(value)
            self.counts[key] = 1
            return

        self.data[key] = (self.data[key] * self.counts[key] + value) / (self.counts[key] + 1)
        self.counts[key] += 1

    def __contains__(self, key):
        return key in self.data

    def __getitem__(self, item):
        return self.data[item]


def cache_node(name, node_dict):
    if name in node_dict:
        return node_dict[name]

    node = Node(name)
    node_dict[name] = node
    return node


def is_sum_node(node):
    if not isinstance(node, str):
        node = node.name
    return node[0:3] == "sum"


def is_multiplication_node(node):
    if not isinstance(node, str):
        node = node.name
    return node[0:3] == "mul"


def is_root_node(node):
    if not isinstance(node, str):
        node = node.name
    return node[:4] == 'root'


def is_backprop_node(node):
    if not isinstance(node, str):
        node = node.name
    return node[0] == 'd'
