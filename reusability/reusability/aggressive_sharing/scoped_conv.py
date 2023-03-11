import tensorflow as tf
from scope import ScopedMeta
import common
from six import add_metaclass
import logging


@add_metaclass(ScopedMeta)
class ScopedConv2D(tf.keras.layers.Conv2D):
    def __init__(self, scope, *args, **kwargs):
        super(ScopedConv2D, self).__init__(*args, **kwargs)
        self.scope = scope


@add_metaclass(ScopedMeta)
class ScopedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, scope, *args, **kwargs):
        super(ScopedDepthwiseConv2D, self).__init__(*args, **kwargs)
        self.scope = scope


def make_config_dict(scope):
    config_dict = {}
    scope_items = scope.split('__')

    for item in scope_items:
        k, v = item.split(':')
        config_dict[k] = v

    return config_dict


def update_scope_dictionary(prefix, scope, config_dict):
    """Bookkeeping for listing which model components share the same parameters"""
    common.SCOPE_DICTIONARY[scope]['meta']['config_dict'] = config_dict

    if not prefix:
        return

    if 'prefixes' not in common.SCOPE_DICTIONARY[scope]['meta']:
        common.SCOPE_DICTIONARY[scope]['meta']['prefixes'] = []

    if prefix not in set(common.SCOPE_DICTIONARY[scope]['meta']['prefixes']):
        common.SCOPE_DICTIONARY[scope]['meta']['prefixes'].append(prefix)


def make_conv2D(prefix='', suffix='', unique=False, in_channels=0, **kwargs):
    """Initialize or load from the global shared dictionary a Conv2D layer"""
    description = []

    if in_channels:
        description.append(f'in_channels:{in_channels}')

    description += [f'{k}:{v}' for k, v in kwargs.items() if k != 'kernel_initializer']

    if suffix:
        description.append(suffix)

    scope = '__'.join(description)
    config_dict = make_config_dict(scope)
    if unique:
        scope = f'{scope}_random'
    logging.info(scope)
    instance = ScopedConv2D(scope, **kwargs)
    scope = instance.scope
    update_scope_dictionary(prefix, scope, config_dict)
    return instance


def make_depthwise_conv2D(prefix='', suffix='', unique=False, in_channels=0, **kwargs):
    """Initialize or load from the global shared dictionary a depthwise Conv2D layer"""
    description = []

    if in_channels:
        description.append(f'in_channels:{in_channels}')
        description.append(f'filters:{in_channels}')

    description += [f'{k}:{v}' for k, v in kwargs.items() if k != 'depthwise_initializer']

    if suffix:
        description.append(suffix)

    scope = '__'.join(description)
    config_dict = make_config_dict(scope)
    if unique:
        scope = f'{scope}_random'
    logging.info(scope)
    instance = ScopedDepthwiseConv2D(scope, **kwargs)
    scope = instance.scope
    update_scope_dictionary(prefix, scope, config_dict)
    return instance
