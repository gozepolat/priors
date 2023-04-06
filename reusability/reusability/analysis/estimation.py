from collections import defaultdict
from collections.abc import Iterable
import numpy as np

from reusability.analysis.planner import (
    push_step, get_elementwise_op_shape, make_feature_names,
    get_conv_output_hw, _repeat_in_array, collect_stats_from_plan
)
from reusability.common import (
    USE_BIAS, expand_dims_for_broadcast, get_safe_access_index, get_uuid
)
from reusability import common


def add_elementwise_op_stats(stats, source, target, second_element, **_):
    """Deal with source/target of different shape with broadcast semantics"""
    second = second_element
    first = source

    out_shape = get_elementwise_op_shape(first.shape, second.shape)
    assert out_shape == target.shape

    first = expand_dims_for_broadcast(first, out_shape)
    second = expand_dims_for_broadcast(second, out_shape)

    warned = False
    with np.nditer(target, flags=['multi_index']) as it:
        for target_name in it:
            # index (h, w, c) from target, i.e. target_name = target[hwc]
            hwc = it.multi_index
            count = stats[target_name.item()]
            if not warned and count == 0:
                print(f"WARNING! {target_name} has zero count")
                warned = True

            safe_index = get_safe_access_index(first, hwc)
            stats[first[safe_index]] += count
            safe_index = get_safe_access_index(second, hwc)
            stats[second[safe_index]] += count


def add_skip_stats(stats, source, target, name=""):
    """Add the aggregated number of contexts for skip layers"""
    source_h, source_w, source_c = source.shape
    target_h, target_w, target_c = target.shape

    assert source_h == target_h and source_w == target_w

    if source_c != target_c:
        if not name:
            name = f"skip_{source[0, 0, 0]}*_{target[0][0][0]}*"

        add_layer_stats(stats, name, source, target,
                        kernel_size=1, stride=1,
                        has_bias=False,
                        has_bn=False,
                        layer_type="conv")
        return

    warned = False
    with np.nditer(target, flags=['multi_index']) as it:
        for target_name in it:
            # Here hwc = (h, w, c), i.e. target_name = target[hwc]
            hwc = it.multi_index
            source_name = source[hwc]
            count = stats[target_name.item()]
            if not warned and count == 0:
                print(f"WARNING! {target_name} has zero count")
                warned = True

            stats[source_name] += count


def add_local_pooling_stats(stats, name, source, target, kernel_size, strides, is_output_layer=False, **_):
    """Add the aggregated number of contexts for local pooling"""
    height, width, channel = source.shape
    assert target.shape[-1] == channel, f"Local pooling has {target.shape[-1]} channels but source has {channel}"

    with np.nditer(target, flags=['multi_index']) as it:
        for target_name in it:
            t_name = target_name.item()

            if t_name not in stats and is_output_layer:
                stats[t_name] = 1

            h, w, c = it.multi_index
            source_coord = h * strides, w * strides, c

            # source and target channels are the same for local pooling
            add_stats_for_kernel_op_with_source_channel(stats, name, source, target, source_coord,
                                                        (h, w, c), kernel_size, create_weights=False)


def add_global_pooling_stats(stats, source, target, is_output_layer=False, **_):
    """Update the aggregated number of contexts of source based on target counts"""
    height, width, channel = source.shape
    assert target.shape[-1] == channel, f"Global pooling has {target.shape[-1]} channels but source has {channel}"
    assert target.shape[0] == 1 == target.shape[1]

    warned = False
    for c in range(channel):
        name = target[0, 0, c]

        if name not in stats and is_output_layer:
            stats[name] = 1

        count = stats[name]

        if not warned and count == 0:
            print(f"WARNING! {name} has zero count!")
            warned = True

        for h in range(height):
            for w in range(width):
                source_name = source[h, w, c]
                stats[source_name] += count


def _add_squeeze_and_excite_block_steps(plan, x, filters, input_filters, se_ratio,
                                        prefix, name, suffix):
    filters_se = max(1, int(input_filters * se_ratio))
    se = push_step(plan, (add_global_pooling_stats,
                          {"layer_type": "global_pool",
                           "name": f"{name}_se_squeeze_{get_uuid()}"}),
                   x)

    # skipped reshape as our global pooling already gives (1, 1, c)

    # se_reduce
    se = push_step(plan,
                   (add_layer_stats, dict(
                       output_filters=filters_se,
                       kernel_size=1,
                       strides=1,
                       has_bias=True,
                       has_bn=False,
                       padding="same",
                       prefix=prefix,
                       name=f"k1s1#sc{se.shape[-1]}#tc{filters_se}",
                       suffix=f"se_reduce_{name}{suffix}",
                       layer_type="conv",
                   )),
                   se)

    # se_expand
    se = push_step(plan,
                   (add_layer_stats, dict(
                       output_filters=filters,
                       kernel_size=1,
                       strides=1,
                       has_bias=True,
                       has_bn=False,
                       padding="same",
                       prefix=prefix,
                       name=f"k1s1#sc{se.shape[-1]}#tc{filters}",
                       suffix=f"se_expand_{name}{suffix}",
                       layer_type="conv",
                   )),
                   se)
    assert se.shape[-1] == x.shape[-1]

    x = push_step(plan, (add_elementwise_op_stats,
                         {
                             'second_element': se,
                             'name': f"elementwise_multiply_{get_uuid()}",
                             'layer_type': "elementwise_op"
                         }),
                  x)
    return x


def add_identity_block_stats(
        stats,
        *args,
        **kwargs,
):
    """ResNet identity_block"""
    plan = []
    add_identity_block_steps(plan, *args, **kwargs)
    stats = collect_stats_from_plan(plan, stats=stats)
    return stats


def add_conv_block_steps(
        plan,
        source,
        target,
        input_filters,
        output_filters,
        filters,
        kernel_size,
        strides,
        stage,
        block,
        name='',
        suffix='',
        has_bias=False,
        **_):
    source_height, source_width, source_channels = source.shape
    assert source_channels == input_filters

    if plan is None:
        plan = []

    x = source
    filters1, filters2, filters3 = filters
    conv_name_base = name + '_res_conv' + str(stage) + block + '_branch'
    x = push_step(plan,
                  (add_layer_stats, dict(
                      output_filters=filters1,
                      kernel_size=1,
                      strides=strides,
                      padding="same",
                      has_bias=has_bias,
                      has_bn=not has_bias,
                      name=f"k1s{strides}#sc{x.shape[-1]}#tc{filters1}",
                      suffix=f"{conv_name_base}_2a" + suffix,
                      layer_type="conv"
                  )),
                  x)
    # skip act
    x = push_step(plan,
                  (add_layer_stats, dict(
                      output_filters=filters2,
                      kernel_size=kernel_size,
                      strides=1,
                      padding="same",
                      has_bias=has_bias,
                      has_bn=not has_bias,
                      name=f"k{kernel_size}s1#sc{x.shape[-1]}#tc{filters2}",
                      suffix=f"{conv_name_base}_2b" + suffix,
                      layer_type="conv"
                  )),
                  x)
    # skip act
    x = push_step(plan,
                  (add_layer_stats, dict(
                      output_filters=filters3,
                      kernel_size=1,
                      strides=1,
                      padding="same",
                      has_bias=has_bias,
                      has_bn=not has_bias,
                      name=f"k1s1#sc{x.shape[-1]}#tc{filters3}",
                      suffix=f"{conv_name_base}_2c" + suffix,
                      layer_type="conv"
                  )),
                  x)
    shortcut = push_step(plan,
                         (add_layer_stats, dict(
                             output_filters=filters3,
                             kernel_size=1,
                             strides=strides,
                             padding="same",
                             has_bias=has_bias,
                             has_bn=not has_bias,
                             name=f"k1s{strides}#sc{source.shape[-1]}#tc{filters3}",
                             suffix=f"{conv_name_base}_shortcut" + suffix,
                             layer_type="conv"
                         )),
                         source)

    # Note that target can be None
    x = push_step(plan,
                  (add_elementwise_op_stats,
                   {
                       'name': f"residual_sum_{get_uuid()}",
                       'second_element': shortcut,
                       'target': target,
                       'layer_type': "elementwise_op"
                   }
                   ),
                  x)
    # skip act
    assert x.shape[-1] == output_filters
    return x


def add_conv_block_stats(
        stats,
        *args,
        **kwargs
):
    """ResNet conv_block"""
    plan = []
    add_conv_block_steps(plan, *args, **kwargs)
    stats = collect_stats_from_plan(plan, stats=stats)
    return stats


def add_mb_conv_block_steps(
        plan,
        source,
        target,
        input_filters,
        output_filters,
        expand_ratio,
        kernel_size,
        strides,
        se_ratio,
        name,
        prefix='z',
        suffix='',
        has_bias=False,
        **_):
    """EfficientNetv2 MBConvBlock"""
    source_height, source_width, source_channels = source.shape
    assert source_channels == input_filters

    if plan is None:
        print("WARNING! If not testing mb_conv_block by itself, plan is incorrectly None.")
        plan = []
    x = source
    filters = input_filters * expand_ratio

    if expand_ratio != 1:
        x = push_step(plan,
                      (add_layer_stats, dict(
                          output_filters=filters,
                          kernel_size=1,
                          strides=1,
                          padding="same",
                          has_bias=has_bias,
                          has_bn=not has_bias,
                          prefix=prefix,
                          name=f"k1s1#sc{x.shape[-1]}#tc{filters}",
                          suffix=f"mbconv_expand_{name}{suffix}",
                          layer_type="conv"
                      )),
                      x)
        # skipped act

    # Depthwise
    x = push_step(plan,
                  (add_layer_stats, dict(
                      output_filters=filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding="same",
                      has_bias=has_bias,
                      has_bn=not has_bias,
                      target=None if 0 < se_ratio <= 1 else target,
                      prefix=prefix,
                      name=f"dwconv_k{kernel_size}s{strides}#sc{x.shape[-1]}#tc{filters}",
                      suffix=f"mbconv_dwconv_{name}{suffix}",
                      layer_type="depthwise_conv",
                  )),
                  x)

    # skipped act

    # squeeze and excite
    if 0 < se_ratio <= 1:
        x = _add_squeeze_and_excite_block_steps(plan, x, filters, input_filters, se_ratio,
                                                prefix, f"mbconv_{name}", suffix)
        # project_conv
        x = push_step(plan,
                      (add_layer_stats, dict(
                          output_filters=output_filters,
                          kernel_size=1,
                          strides=1,
                          padding="same",
                          has_bias=has_bias,
                          has_bn=not has_bias,
                          target=None if strides == 1 and input_filters == output_filters else target,
                          prefix=prefix,
                          name=f"k1s1#sc{x.shape[-1]}#tc{output_filters}",
                          suffix=f"mbconv_project_{name}{suffix}",
                          layer_type="conv",
                      )),
                      x)

        # skipped act

        if strides == 1 and input_filters == output_filters:
            # skipped drop based on survival prob for now
            # add residual sum

            # Note that target can be None
            x = push_step(plan,
                          (add_elementwise_op_stats,
                           {
                               'name': f"residual_sum_{get_uuid()}",
                               'second_element': source,
                               'target': target,
                               'layer_type': "elementwise_op"
                           }),
                          x)

    assert x.shape[-1] == output_filters
    return x


def add_mb_conv_block_stats(
        stats,
        *args,
        **kwargs
):
    """MBConvBlock, mobile inverted residual bottleneck"""
    plan = []
    add_mb_conv_block_steps(plan, *args, **kwargs)
    stats = collect_stats_from_plan(plan, stats=stats)
    return stats


def add_fused_mb_conv_block_steps(
        plan,
        source,
        target,
        input_filters,
        output_filters,
        expand_ratio,
        kernel_size,
        strides,
        se_ratio,
        name,
        prefix='z',
        suffix='',
        has_bias=False,
        **_):
    """EfficientNetv2 FusedMBConvBlock"""
    source_height, source_width, source_channels = source.shape
    assert source_channels == input_filters

    if plan is None:
        plan = []

    x = source
    filters = input_filters * expand_ratio

    if expand_ratio != 1:
        x = push_step(plan,
                      (add_layer_stats, dict(
                          output_filters=filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding="same",
                          has_bias=has_bias,
                          has_bn=not has_bias,
                          prefix=prefix,
                          name=f"k1s1#sc{x.shape[-1]}#tc{filters}",
                          suffix=f"fusedmb_expand_{name}{suffix}",
                          layer_type="conv"
                      )),
                      x)

        # skipped act

    if 0 < se_ratio <= 1:
        x = _add_squeeze_and_excite_block_steps(plan, x, filters, input_filters, se_ratio,
                                                prefix, f"fusedmb_{name}", suffix)
    # project_conv
    x = push_step(plan,
                  (add_layer_stats, dict(
                      output_filters=output_filters,
                      kernel_size=1 if expand_ratio != 1 else kernel_size,
                      strides=1 if expand_ratio != 1 else strides,
                      padding="same",
                      has_bias=has_bias,
                      has_bn=not has_bias,
                      target=None if strides == 1 and input_filters == output_filters else target,
                      prefix=prefix,
                      name=f"k1s1#sc{x.shape[-1]}#tc{output_filters}",
                      suffix=f"fusedmb_project_{name}{suffix}",
                      layer_type="conv",
                  )),
                  x)

    # skipped act

    if strides == 1 and input_filters == output_filters:
        # skipped drop based on survival prob for now
        # add residual from source
        x = push_step(plan,
                      (add_elementwise_op_stats,
                       {
                           'name': f"residual_sum_{get_uuid()}",
                           'second_element': source,
                           'target': target,
                           'layer_type': "elementwise_op"
                       }),
                      x)

    assert x.shape[-1] == output_filters
    return x


def add_fused_mb_conv_block_stats(stats, *args, **kwargs):
    """Fused version of MBConvBlock, mobile inverted residual bottleneck"""
    plan = []
    add_fused_mb_conv_block_steps(plan, *args, **kwargs)
    stats = collect_stats_from_plan(plan, stats=stats)
    return stats


def add_stats_for_kernel_op_with_source_channel(stats, name, source, target, coord, target_coord,
                                                kernel_size, suffix='', create_weights=True):
    """Add stats for any operation that requires a windowed operation on source

     For convolution, create_weights must be true, for max_pooling it should be false
    """
    source_height, source_width, source_channels = source.shape
    center_h, center_w, source_c = coord
    target_h, target_w, target_c = target_coord

    assert source_c < source_channels and target_c < target.shape[-1]

    offset = kernel_size // 2
    target_name = target[target_h, target_w, target_c]
    target_count = stats[target_name]

    if target_count == 0:
        return

    name_suffix = f"n{name}ic{source_c}tc{target_c}{suffix}"

    for h in range(-offset, offset + 1):
        conv_h = center_h + h
        if not (0 <= conv_h < source_height):
            continue

        for w in range(-offset, offset + 1):
            conv_w = center_w + w

            if not (0 <= conv_w < source_width):
                continue

            source_name = source[conv_h, conv_w, source_c]
            weight_y = h + offset
            weight_x = w + offset

            if create_weights:
                weight_name = f'y{weight_y}x{weight_x}{name_suffix}'
                stats[weight_name] += target_count

            stats[source_name] += target_count


def add_batch_norm_stats(stats, name, source, target, **_):
    """Add aggregated number of counts for batch normalization layer"""
    assert source.shape == target.shape

    uid_suffix = f"{name}_{get_uuid()}"
    warned = False
    with np.nditer(target, flags=['multi_index']) as it:
        for target_name in it:
            count = stats[target_name.item()]

            # (h, w, c) = hwc i.e. target_name = target[hwc]
            hwc = it.multi_index
            c = hwc[-1]

            if not warned and count == 0:
                print(f"WARNING! {target_name.item()} has zero count")
                warned = True

            bn_weight = f"y_bnw_{c}{uid_suffix}"
            bn_bias = f"y_bnb_{c}{uid_suffix}"
            bn_running_mean = f"y_bnm_{c}{uid_suffix}"
            bn_running_var = f"y_bnv_{c}{uid_suffix}"
            stats[bn_weight] += count
            stats[bn_bias] += count
            stats[bn_running_mean] += count
            stats[bn_running_var] += count
            source[hwc] += count


def add_identity_stats(stats, **_):
    return stats


w_set = set()


def add_stats_for_kernel_op_with_source_channel_optimized(stats, name, source, target, coord, target_coord,
                                                          kernel_size, suffix='',
                                                          create_weights=True,
                                                          optimize=True, multiplier=None):
    """Add stats for any operation that requires a windowed operation on source

     For convolution, create_weights must be true, for max_pooling it should be false
    """
    source_height, source_width, source_channels = source.shape
    target_height, target_width, target_channels = target.shape
    center_h, center_w, source_c = coord
    target_h, target_w, target_c = target_coord
    if multiplier is None:
        multiplier = target_channels

    assert (source_c < source_channels and target_c < target_channels
            and target_h < target_height and target_w < target_width
            and center_h < source_height and center_w < source_width)

    offset = kernel_size // 2

    target_name = target[target_h, target_w, target_c]
    target_count = stats[target_name]

    if target_count == 0:
        return

    def get_name_suffix(channel):
        return f"n{name}ic{source_c}tc{channel}{suffix}"

    name_suffix = get_name_suffix(target_c)

    for h in range(-offset, offset + 1):
        conv_h = center_h + h
        if not (0 <= conv_h < source_height):
            continue

        for w in range(-offset, offset + 1):
            conv_w = center_w + w

            if not (0 <= conv_w < source_width):
                continue

            source_name = source[conv_h, conv_w, source_c]
            weight_y = h + offset
            weight_x = w + offset
            if common.CHANNEL_OPTIMIZE and optimize:
                for target_channel in range(target_channels):
                    if create_weights:
                        name_suffix = get_name_suffix(target_channel)
                        weight_name = f'y{weight_y}x{weight_x}{name_suffix}'
                        if common.DEBUG_MODE and weight_name not in w_set:
                            print(weight_name)
                            w_set.add(weight_name)
                        stats[weight_name] += target_count

                stats[source_name] += target_count * multiplier
            else:
                if create_weights:
                    weight_name = f'y{weight_y}x{weight_x}{name_suffix}'
                    if common.DEBUG_MODE and weight_name not in w_set:
                        print(weight_name)
                        w_set.add(weight_name)
                    stats[weight_name] += target_count

                stats[source_name] += target_count


def update_bias_or_bn_weights(stats, count, has_bias, has_bn, bias_name,
                              bn_weight, bn_running_mean, bn_running_var, bn_bias):
    if has_bias:
        stats[bias_name] += count
    elif has_bn:
        stats[bn_weight] += count
        stats[bn_running_mean] += count
        stats[bn_running_var] += count
        stats[bn_bias] += count


def add_convolve_stats(stats, name, source, target,
                       has_bias, has_bn, kernel_size,
                       strides, layer_type, suffix, is_output_layer,
                       target_c, uid_suffix):
    """Calculate the number of aggregated contexts when convolving for one single target channel"""

    assert not (has_bias and has_bn), "Pick either bias or batch_norm, not both"
    source_channels = source.shape[-1]
    target_height, target_width, target_channels = target.shape

    kernel_fn = add_stats_for_kernel_op_with_source_channel

    if common.CHANNEL_OPTIMIZE:
        kernel_fn = add_stats_for_kernel_op_with_source_channel_optimized

    def get_bias_name(channel):
        return f"yb_tc{channel}_{name}_{suffix}"

    def get_bn_weight(channel):
        return f"y_bnw_tc{channel}_{uid_suffix}"

    def get_bn_bias(channel):
        return f"y_bnb_tc{channel}_{uid_suffix}"

    def get_bn_running_mean(channel):
        return f"y_bnm_tc{channel}_{uid_suffix}"

    def get_bn_running_var(channel):
        return f"y_bnv_tc{channel}_{uid_suffix}"

    bias_name = get_bias_name(target_c)
    bn_weight = get_bn_weight(target_c)
    bn_running_mean = get_bn_running_mean(target_c)
    bn_running_var = get_bn_running_var(target_c)
    bn_bias = get_bn_bias(target_c)

    warned = False
    for h in range(target_height):
        for w in range(target_width):
            target_coord = h, w, target_c
            output_name = target[h, w, target_c]

            if output_name not in stats and is_output_layer:
                stats[output_name] = 1

            count = stats[output_name]

            if not warned and count == 0:
                print(f"WARNING! {output_name} has zero count")
                warned = True

            if common.CHANNEL_OPTIMIZE:
                for target_channel in range(target_channels):
                    update_bias_or_bn_weights(stats, count, has_bias, has_bn,
                                              bias_name=get_bias_name(target_channel),
                                              bn_weight=get_bn_weight(target_channel),
                                              bn_running_mean=get_bn_running_mean(target_channel),
                                              bn_running_var=get_bn_running_var(target_channel),
                                              bn_bias=get_bn_bias(target_channel)
                                              )
            else:
                update_bias_or_bn_weights(stats, count, has_bias, has_bn, bias_name,
                                          bn_weight, bn_running_mean, bn_running_var, bn_bias)

            if layer_type == "depthwise_conv":
                if common.CHANNEL_OPTIMIZE:
                    # 1-to-1 mapping necessary here as we discard the outer loop with target_channels
                    # optimize false since we do not want to multiply counts with #target channels
                    for target_channel in range(target_channels):
                        source_coord = h * strides, w * strides, target_channel
                        kernel_fn(stats, name, source, target, source_coord, target_coord,
                                  kernel_size, suffix, optimize=False)
                else:
                    source_coord = h * strides, w * strides, target_c
                    kernel_fn(stats, name, source, target, source_coord, target_coord,
                              kernel_size, suffix)
                continue

            for source_c in range(source_channels):
                source_coord = h * strides, w * strides, source_c
                kernel_fn(stats, name, source, target, source_coord, target_coord,
                          kernel_size, suffix)


def add_layer_stats(stats, name, source, target, has_bias, has_bn=False,
                    kernel_size=1, strides=1,
                    layer_type="dense", suffix='', is_output_layer=False,
                    padding="same", **_):
    """Create fake nodes and estimate stats for the aggregated number of contexts for a given layer"""
    assert layer_type in {"dense", "conv", "depthwise_conv"}

    hw = get_conv_output_hw(source.shape[:2], kernel_size, strides,
                            padding=kernel_size // 2, dilation=1)
    assert hw == target.shape[:2]

    if suffix:
        suffix = f"+{suffix}"

    target_height, target_width, target_channels = target.shape
    source_channels = source.shape[-1]

    if layer_type == "dense":
        assert kernel_size == 1, "Dense layer can not have kernel size != 1"
    elif layer_type == "depthwise_conv":
        if target_channels != source_channels:
            raise NotImplementedError("depth multiplier > 1 not supported")

    if padding != "same":
        raise NotImplementedError("padding != 'same' not supported yet")

    uid_suffix = f"{name}_{get_uuid()}"

    if not common.CHANNEL_OPTIMIZE:
        # dense is just conv with height, width 1, and kernel_size 1
        for target_c in range(target_channels):
            add_convolve_stats(stats, name, source, target,
                               has_bias, has_bn, kernel_size,
                               strides, layer_type, suffix, is_output_layer,
                               target_c, uid_suffix)
        return

    # Optimize channelwise calculation for common DL architectures such as CNN
    # NOTE that this assumes all output channels contribute to source equally
    # This can speed up calculation ~#channels times for vanilla conv
    # TODO optimize depthwise conv ~9x if we repeat the first source channel for all source
    add_convolve_stats(stats, name, source, target,
                       has_bias, has_bn, kernel_size,
                       strides, layer_type, suffix, is_output_layer,
                       0, uid_suffix)


LAYER_FUNCTION_DICT = {"default": add_layer_stats,
                       "conv": add_layer_stats,
                       "depthwise_conv": add_layer_stats,
                       "MBConvBlock": add_mb_conv_block_stats,
                       "FusedMBConvBlock": add_fused_mb_conv_block_stats,
                       "ConvBlock": add_conv_block_stats,
                       "identity_block": add_identity_block_stats,
                       "global_pooling": add_global_pooling_stats,
                       "local_pooling": add_local_pooling_stats,
                       "no_op": add_identity_stats,
                       "elementwise_op": add_elementwise_op_stats,
                       "skip_connection": add_skip_stats,
                       "batch_norm": add_batch_norm_stats}


def add_prefix(a, prefix="d"):
    return f"{prefix}_{a}"


def make_d_nodes(x):
    """Add a d_ prefix to the name or names of all elements of x"""
    if isinstance(x, str):
        return f"d_{x}"

    assert isinstance(x, np.ndarray), f"Expected str or np.ndarray, got {type(x)}"
    vectorized_add_prefix = np.vectorize(add_prefix)
    return vectorized_add_prefix(x, "d")


def add_dag_stats(
        stats,
        source,
        block_kwargs,
        with_backprop=False,
        **_
):
    """Estimate the aggregated number of contexts for a DAG described with block_kwargs"""
    plan = []
    skip_connection_list = []
    x = source
    feature_layers = [source]
    for kwargs in block_kwargs:
        layer_fn = kwargs.pop("layer_fn", "")
        skip_targets = kwargs.pop("skip_targets", [])
        skip_connection_list.append(skip_targets)

        # try inferring layer_fn
        if not layer_fn:
            layer_type = kwargs["layer_type"]
            if layer_type in {"conv", "dense", "depthwise_conv"}:
                print(f"Assuming layer_fn is 'default' for {layer_type}")
                layer_fn = "default"
            else:
                raise NotImplementedError("Not sure what the layer_fn is")

        x_source = x
        x = push_step(plan, (LAYER_FUNCTION_DICT[layer_fn], kwargs), x_source)
        feature_layers.append(x)

    # add skip connections to the plan
    new_plan = []
    for i, p in enumerate(plan):
        skip_targets = skip_connection_list[i]

        if len(skip_targets) == 0:
            new_plan.append(p)
            continue

        for j in skip_targets:
            push_step(new_plan,
                      (add_skip_stats, {"target": feature_layers[j]}),
                      feature_layers[i])

    if stats is None:
        stats = defaultdict(int)

    if with_backprop:
        raise NotImplementedError("Backpropagation analysis is not supported yet")
    else:
        # add the last feature layer to the stats
        with np.nditer(feature_layers[-1]) as it:
            for target_name in it:
                stats[target_name.item()] = 1

    return collect_stats_from_plan(new_plan, stats=stats)


def _add_residual_to_block_args(block_args, skip_block_size, residual_type):
    depth = len(block_args)

    for i in range(skip_block_size):
        for d in range(i + skip_block_size, depth, skip_block_size):
            for previous_layer_i in range(d - skip_block_size, i - 1, -1):
                print(f"appending {d} to {previous_layer_i} cause skip size: {skip_block_size}")
                block_args[previous_layer_i]["skip_targets"].append(d)
                if residual_type != 'dense':
                    break

        if residual_type == 'vanilla':
            break


def make_dag_block_kwargs(num_input_channels, channels, depth, *_args,
                          image_height=1, image_width=1, kernel_sizes=1,
                          strides=1, skip_block_size=-1, is_skip_identity=True,
                          residual_type='vanilla', has_bias=USE_BIAS, has_bn=False,
                          dag_type='mlp', block_kwargs=None,
                          source=None, ):
    """Generalization for any MLP or CNN"""
    args_positive = (np.all(np.array(channels) > 0) and
                     np.all(np.array([num_input_channels, depth, image_height, image_width]) > 0) and
                     np.all(np.array(kernel_sizes) > 0))
    assert is_skip_identity

    if not args_positive:
        raise ValueError(f"Args not positive: {num_input_channels}, {channels}, {depth}, "
                         f"{image_height} {image_width} {kernel_sizes}")

    if skip_block_size == -1:
        skip_block_size = depth + 1

    channels = _repeat_in_array(channels, depth)
    strides = _repeat_in_array(strides, depth)
    kernel_sizes = _repeat_in_array(kernel_sizes, depth)

    print("DAG type", dag_type)
    if dag_type == 'mlp':
        kernel_sizes = _repeat_in_array(1, depth)
        assert image_height == 1 and image_width == 1, "Please use channels for width instead"
        # cnn_skip connects input to output for each index for h,w,c, (in mlp h,w would be 1)
    elif dag_type != 'cnn':
        raise NotImplementedError(f"dag_type {dag_type} is not implemented")

    if block_kwargs is None:
        # first function just returns source without changing it
        block_kwargs = [{"layer_fn": "no_op", "target": source,
                         "skip_targets": [], 'name': 'no_op'}]
        all_filters = [num_input_channels] + channels

        for d in range(depth):
            ks = kernel_sizes[d]
            sc = all_filters[d]
            tc = all_filters[d + 1]

            if dag_type == "cnn":
                layer_type = "conv"
            elif dag_type == "mlp":
                layer_type = "dense"
            else:
                raise NotImplementedError("No other type can be easily inferred")

            block_kwargs.append({"layer_fn": "default",
                                 "kernel_size": ks,
                                 "input_filters": sc,
                                 "output_filters": tc,
                                 "num_repeat": 1,
                                 "has_bias": has_bias,
                                 "has_bn": has_bn,
                                 "padding": "same",
                                 "name": f"k{ks}s{strides[d]}#sc{sc}#tc{tc}",
                                 "suffix": str(d),
                                 "strides": strides[d],
                                 "layer_type": layer_type,
                                 "skip_targets": []})

        _add_residual_to_block_args(block_kwargs, skip_block_size, residual_type)

    return block_kwargs


def make_constant_width_mlp_stats(num_input_channels: int,
                                  channels: int, depth: int,
                                  cross_shared: bool = False) -> dict:
    """Skip graph creation to directly estimate constant width mlp stats"""
    narrow = {}
    delim = '_'

    if cross_shared:
        delim = '+'

    val = 1
    if not num_input_channels or num_input_channels < 1:
        num_input_channels = channels

    for j in range(depth):
        w_range = channels if j < depth - 1 else num_input_channels
        w_range *= channels
        for i in range(w_range):
            key = f'yw{i}{delim}d{depth-j}'
            if key in narrow:
                assert cross_shared
                narrow[key] += val
            else:
                narrow[key] = val
            assert narrow[key] != 0, f"{val} should not be 0 with {channels} and {j}"

        val = int(channels) * int(val)
    return narrow


def estimate_dag_stats(num_input_channels, channels, depth, *_args,
                       image_height=1, image_width=1, kernel_sizes=1,
                       strides=1, skip_block_size=-1, is_skip_identity=True,
                       residual_type='vanilla', dag_type='mlp', block_kwargs=None,
                       has_bias=USE_BIAS, has_bn=False, stats=None,
                       source=None, name='dag', suffix='', optimize=False, **kwargs):
    """Estimate the number of contexts for each learnable parameter of a CNN or MLP"""
    if optimize and (dag_type == 'mlp') and not (isinstance(channels, Iterable) or has_bn or has_bias):
        print("Skipping dag construction and directly estimating stats from constant width mlp")
        return make_constant_width_mlp_stats(num_input_channels, channels, depth, **kwargs)

    if source is None:
        input_shape = (image_height, image_width, num_input_channels)
        img_input = make_feature_names(name='input',
                                       shape=input_shape,
                                       prefix='z0',
                                       suffix='bid0')
        source = img_input

    # in case a single kernel_size is given
    kernel_sizes = kwargs.get('kernel_size', kernel_sizes)

    block_kwargs = make_dag_block_kwargs(num_input_channels, channels, depth, *_args,
                                         image_height=image_height, image_width=image_width,
                                         kernel_sizes=kernel_sizes,
                                         strides=strides, skip_block_size=skip_block_size,
                                         is_skip_identity=is_skip_identity,
                                         residual_type=residual_type, dag_type=dag_type,
                                         block_kwargs=block_kwargs, has_bias=has_bias,
                                         has_bn=has_bn, source=source, )

    return add_dag_stats(
        stats=stats,
        name=f"{name}_d{depth}",
        source=source,
        block_kwargs=block_kwargs,
        dag_type=dag_type,
        suffix=suffix,
        **kwargs
    )


def add_identity_block_steps(
        plan,
        source,
        target,
        input_filters,
        output_filters,
        filters,
        kernel_size,
        stage,
        block,
        name='',
        suffix='',
        has_bias=False,
        **_):
    source_height, source_width, source_channels = source.shape

    assert source_channels == input_filters

    if plan is None:
        plan = []

    x = source
    filters1, filters2, filters3 = filters
    conv_name_base = name + '_res_id' + str(stage) + block + '_branch'

    x = push_step(plan,
                  (add_layer_stats, dict(
                      output_filters=filters1,
                      kernel_size=1,
                      strides=1,
                      padding="same",
                      has_bias=has_bias,
                      has_bn=not has_bias,
                      name=f"k1s1#sc{x.shape[-1]}#tc{filters1}",
                      suffix=f"{conv_name_base}_2a" + suffix,
                      layer_type="conv"
                  )),
                  x)

    # skip act
    x = push_step(plan,
                  (add_layer_stats, dict(
                      output_filters=filters2,
                      kernel_size=kernel_size,
                      strides=1,
                      padding="same",
                      has_bias=has_bias,
                      has_bn=not has_bias,
                      name=f"k{kernel_size}s1#sc{x.shape[-1]}#tc{filters2}",
                      suffix=f"{conv_name_base}_2b" + suffix,
                      layer_type="conv"
                  )),
                  x)
    # skip act
    x = push_step(plan,
                  (add_layer_stats, dict(
                      output_filters=filters3,
                      kernel_size=1,
                      strides=1,
                      padding="same",
                      has_bias=has_bias,
                      has_bn=not has_bias,
                      name=f"k1s1#sc{x.shape[-1]}#tc{filters3}",
                      suffix=f"{conv_name_base}_2c" + suffix,
                      layer_type="conv"
                  )),
                  x)

    # Note that target can be None
    x = push_step(plan,
                  (add_elementwise_op_stats,
                   {
                       'name': f"residual_sum_{get_uuid()}",
                       'second_element': source,
                       'target': target,
                       'layer_type': "elementwise_op"
                   }
                   ),
                  x)
    # skip act
    assert x.shape[-1] == output_filters
    return x
